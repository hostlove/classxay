"""Scheme B: source-aligned local features, global ViT, and one ordinal head."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

import timm
from timm.data import resolve_model_data_config


class GlobalFeatureViT(nn.Module):
    """Extract the 384-dimensional CLS token from the complete cervical ROI."""

    def __init__(self, image_size=320):
        super().__init__()
        self.image_size = image_size
        self.encoder = timm.create_model(
            "vit_small_patch16_224.augreg_in21k_ft_in1k",
            img_size=image_size, pretrained=False, num_classes=0,
        )
        config = resolve_model_data_config(self.encoder)
        self.register_buffer("mean", torch.tensor(config["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(config["std"]).view(1, 3, 1, 1))

    def forward(self, image):
        if image.ndim != 4 or image.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f"global_image must be [B, 3, {self.image_size}, {self.image_size}]")
        if not image.is_floating_point():
            raise ValueError("global_image must contain floats in [0, 1]")
        image = (image.to(dtype=self.mean.dtype) - self.mean) / self.std
        return self.encoder.forward_features(image)[:, 0]


class LocalFeatureCNN(nn.Module):
    """Use one shared truncated ResNet-18 for all windows; keep 128 channels."""

    stride = 8
    offset = 0.5

    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=None)
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2,
        )
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, images, content_boxes):
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("local_images must be [B, N, 3, H, W]")
        if not images.is_floating_point():
            raise ValueError("local_images must contain floats in [0, 1]")
        batch, windows, _, height, width = images.shape
        if windows < 1 or min(height, width) < 8 or height % 8 or width % 8:
            raise ValueError("use at least one window and input sides divisible by 8")
        if content_boxes.shape != (batch, windows, 4):
            raise ValueError("content_boxes must be [B, N, 4]")
        x = images.reshape(batch * windows, 3, height, width).to(dtype=self.mean.dtype)
        features = self.encoder((x - self.mean) / self.std)
        rows, columns = features.shape[-2:]
        boxes = content_boxes.reshape(-1, 4).to(device=features.device, dtype=torch.float32)
        left, top, right, bottom = boxes.unbind(-1)
        column_centers = self.offset + self.stride * torch.arange(columns, device=features.device)
        row_centers = self.offset + self.stride * torch.arange(rows, device=features.device)
        valid_columns = (column_centers >= left[:, None]) & (column_centers < right[:, None])
        valid_rows = (
            (row_centers >= top[:, None]) & (row_centers < bottom[:, None])
            & valid_columns.any(-1, keepdim=True)
        )
        pooled = features.masked_fill(~valid_columns[:, None, None, :], 0).sum(-1)
        pooled = pooled / valid_columns.sum(-1).clamp_min(1)[:, None, None]
        row_features = pooled.transpose(1, 2).masked_fill(~valid_rows[..., None], 0)
        return row_features.reshape(batch, windows, rows, 128), valid_rows.reshape(batch, windows, rows)


def align_row_features(row_features, row_valid, source_intervals, content_boxes,
                       *, grid_size=128, stride=8.0, offset=0.5):
    """Map row features to ROI height coordinates and average overlapping views.

    Pixel-edge boxes and normalized source intervals must match the ACTUAL
    rounded letterbox resize. Invalid rows never contribute to interpolation.
    Returns features [B, J, C], validity [B, J], and coverage counts [B, J].
    """
    if row_features.ndim != 4 or row_features.shape[2] < 1:
        raise ValueError("row_features must be [B, N, R, C] with R >= 1")
    batch, windows, rows, channels = row_features.shape
    if windows < 1 or channels < 1 or row_valid.shape != (batch, windows, rows):
        raise ValueError("invalid number of windows/channels or row_valid shape")
    if row_valid.dtype != torch.bool:
        raise ValueError("row_valid must be boolean")
    if source_intervals.shape != (batch, windows, 2) or content_boxes.shape != (batch, windows, 4):
        raise ValueError("source_intervals or content_boxes has an incompatible shape")
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 1:
        raise ValueError("grid_size must be a positive integer")
    if not math.isfinite(stride) or stride <= 0 or not math.isfinite(offset):
        raise ValueError("invalid feature coordinates")

    device = row_features.device
    intervals = source_intervals.to(device=device, dtype=torch.float32)
    boxes = content_boxes.to(device=device, dtype=torch.float32)
    source_top, source_bottom = intervals.unbind(-1)
    left, content_top, right, content_bottom = boxes.unbind(-1)
    usable = (
        torch.isfinite(intervals).all(-1) & torch.isfinite(boxes).all(-1)
        & (source_top >= 0) & (source_bottom <= 1) & (source_bottom > source_top)
        & (right > left) & (content_bottom > content_top)
    )
    centers = offset + stride * torch.arange(rows, device=device, dtype=torch.float32)
    valid = (
        row_valid.to(device=device) & usable[..., None]
        & (centers >= content_top[..., None]) & (centers < content_bottom[..., None])
    )
    window_valid = valid.any(-1)
    grid = (torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5) / grid_size
    covered = window_valid[..., None] & (grid >= source_top[..., None]) & (grid < source_bottom[..., None])

    # Avoid NaN arithmetic in unused windows before applying coverage masks.
    source_top = torch.where(window_valid, source_top, 0.0)
    source_length = torch.where(window_valid, source_bottom - source_top, 1.0)
    content_top = torch.where(window_valid, content_top, 0.0)
    content_length = torch.where(window_valid, content_bottom - content_top, 1.0)
    relative = ((grid - source_top[..., None]) / source_length[..., None]).clamp(0, 1)
    y_input = content_top[..., None] + relative * content_length[..., None]
    y_feature = ((y_input - offset) / stride).clamp(0, rows - 1)

    # Nearest valid neighbours handle padding, missing rows, and endpoints.
    row_index = torch.arange(rows, device=device)
    previous = torch.where(valid, row_index, -1).cummax(-1).values
    following = torch.where(valid, row_index, rows).flip((-1,)).cummin(-1).values.flip((-1,))
    lower = previous.gather(-1, y_feature.floor().long())
    upper = following.gather(-1, y_feature.ceil().long())
    lower = torch.where(lower >= 0, lower, upper).clamp(0, rows - 1)
    upper = torch.where(upper < rows, upper, lower)
    fraction = ((y_feature - lower) / (upper - lower).clamp_min(1)).clamp(0, 1)
    clean = row_features.float().masked_fill(~valid[..., None], 0)
    h_lower = clean.gather(2, lower[..., None].expand(-1, -1, -1, channels))
    h_upper = clean.gather(2, upper[..., None].expand(-1, -1, -1, channels))
    samples = (1 - fraction[..., None]) * h_lower + fraction[..., None] * h_upper
    coverage_count = covered.sum(1)
    curve = samples.masked_fill(~covered[..., None], 0).sum(1)
    curve = curve / coverage_count.clamp_min(1)[..., None]
    return curve, coverage_count > 0, coverage_count


class FeatureOrdinalHead(nn.Module):
    """Pool aligned local features, fuse features, and enforce ordered thresholds."""

    def __init__(self, global_dim=384, local_dim=128, hidden_dim=256,
                 dropout=0.1, mean_mode="spatial"):
        super().__init__()
        if mean_mode not in ("spatial", "coverage"):
            raise ValueError("mean_mode must be 'spatial' or 'coverage'")
        self.mean_mode = mean_mode
        self.global_dim, self.local_dim = global_dim, local_dim
        input_dim = global_dim + 2 * local_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1, bias=False),
        )
        self.first_threshold = nn.Parameter(torch.tensor(-1.0))
        self.threshold_gaps = nn.Parameter(torch.full((2,), math.log(math.expm1(1.0))))

    def forward(self, global_features, aligned_features, grid_valid, coverage_count):
        if aligned_features.ndim != 3 or aligned_features.shape[-1] != self.local_dim:
            raise ValueError("aligned_features has an incompatible shape")
        batch, grid_size, _ = aligned_features.shape
        if global_features.shape != (batch, self.global_dim):
            raise ValueError("global_features has an incompatible shape")
        if grid_valid.shape != (batch, grid_size) or coverage_count.shape != grid_valid.shape:
            raise ValueError("grid validity or coverage has an incompatible shape")
        if grid_valid.dtype != torch.bool:
            raise ValueError("grid_valid must be boolean")
        if not grid_valid.any(-1).all():
            raise ValueError("no valid local coverage: check masks/crops; do not silently classify")

        h = aligned_features.float().masked_fill(~grid_valid[..., None], 0)
        weights = grid_valid.float() if self.mean_mode == "spatial" else coverage_count.float()
        weights = weights.masked_fill(~grid_valid, 0)
        local_mean = (h * weights[..., None]).sum(1) / weights.sum(1).clamp_min(1)[:, None]
        local_max = h.masked_fill(~grid_valid[..., None], -torch.inf).amax(1)
        fused = torch.cat((global_features.float(), local_mean, local_max), -1)
        severity = self.mlp(fused).squeeze(-1).float()
        base = self.first_threshold.float().reshape(1)
        gaps = F.softplus(self.threshold_gaps.float()) + 1e-4
        thresholds = torch.cat((base, base + gaps.cumsum(0)))
        logits = severity[:, None] - thresholds
        cumulative = logits.sigmoid()
        probabilities = torch.cat((
            1 - cumulative[:, :1],
            cumulative[:, :-1] - cumulative[:, 1:],
            cumulative[:, -1:],
        ), -1)
        return {
            "logits": logits, "cumulative": cumulative,
            "probabilities": probabilities, "prediction": probabilities.argmax(-1),
            "severity": severity, "thresholds": thresholds,
            "local_mean": local_mean, "local_max": local_max,
        }


class SchemeBClassifier(nn.Module):
    """Jointly trainable classification branches; cropping itself is separate."""

    def __init__(self, *, global_size=320, grid_size=128, mean_mode="spatial", dropout=0.1):
        super().__init__()
        self.global_vit = GlobalFeatureViT(global_size)
        self.local_cnn = LocalFeatureCNN()
        self.grid_size = grid_size
        self.head = FeatureOrdinalHead(mean_mode=mean_mode, dropout=dropout)

    def forward(self, global_image, local_images, content_boxes, source_intervals):
        global_features = self.global_vit(global_image)
        row_features, row_valid = self.local_cnn(local_images, content_boxes)
        aligned, grid_valid, coverage_count = align_row_features(
            row_features, row_valid, source_intervals, content_boxes,
            grid_size=self.grid_size, stride=self.local_cnn.stride, offset=self.local_cnn.offset,
        )
        result = self.head(global_features, aligned, grid_valid, coverage_count)
        result.update(
            global_features=global_features, row_features=row_features, row_valid=row_valid,
            aligned_features=aligned, grid_valid=grid_valid, coverage_count=coverage_count,
        )
        return result


def ordinal_logits_loss(logits, labels):
    """Unweighted image-level cumulative BCE; use raw logits, not probabilities."""
    if logits.ndim != 2 or logits.shape[1] != 3 or labels.shape != logits.shape[:1]:
        raise ValueError("expected logits [B, 3] and labels [B]")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("labels must be integer grades")
    labels = labels.to(device=logits.device)
    if ((labels < 0) | (labels > 3)).any():
        raise ValueError("grades must be in [0, 3]")
    targets = labels[:, None] >= torch.arange(1, 4, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits.float(), targets.float())
