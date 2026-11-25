"""Prepare unbatched inputs from an original radiograph and its aligned mask."""

from math import ceil, isfinite
from numbers import Integral, Real

import numpy as np
from PIL import Image
import torch
from scipy import ndimage


def _expanded_roi(foreground, margin):
    """Keep 8-connected components of at least max(5 px, 0.5% of foreground)."""
    components, count = ndimage.label(foreground, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        raise ValueError("segmentation mask has no foreground")
    areas = np.bincount(components.ravel())
    keep = areas >= max(5, ceil(foreground.sum() * 0.005))
    keep[0] = False
    cleaned = keep[components]
    rows = np.flatnonzero(cleaned.any(axis=1))
    columns = np.flatnonzero(cleaned.any(axis=0))
    if rows.size == 0:
        raise ValueError("segmentation mask has no foreground after small-fragment removal")
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1
    dx, dy = ceil((right - left) * margin / 100), ceil((bottom - top) * margin / 100)
    height, width = foreground.shape
    return max(0, left - dx), max(0, top - dy), min(width, right + dx), min(height, bottom + dy)


def _letterbox(pixels, size):
    """Fit a grayscale crop inside (height, width) and return its content edges."""
    height, width = pixels.shape
    target_height, target_width = size
    scale = min(target_height / height, target_width / width)
    fitted_height = min(target_height, max(1, round(height * scale)))
    fitted_width = min(target_width, max(1, round(width * scale)))
    top = (target_height - fitted_height) // 2
    left = (target_width - fitted_width) // 2
    bottom = top + fitted_height
    right = left + fitted_width
    resized = Image.fromarray(pixels).resize(
        (fitted_width, fitted_height), Image.Resampling.BILINEAR
    )
    canvas = np.zeros(size, dtype=np.uint8)
    canvas[top:bottom, left:right] = np.asarray(resized)
    tensor = torch.from_numpy(canvas).float().div_(255).unsqueeze(0).repeat(3, 1, 1)
    return tensor, (left, top, right, bottom)


def prepare_case(
    image,
    mask,
    *,
    margin=15,
    n_windows=6,
    overlap=.25,
    global_size=(320, 320),
    local_size=(192, 384),
):
    """Build model inputs for one grayscale uint8 radiograph and a binary mask.

    Image can be a 2D uint8 array or a PIL image in mode L or 1. Mask must
    have the same shape and use zero for background and a single positive
    value for foreground. Margin is a percentage of the cleaned mask box
    added on each side. Both target sizes use (height, width).

    The global image contains the entire expanded mask box. Local images are
    overlapping horizontal windows inside it. Both use the original pixels.
    Returns unbatched float32 tensors: global_image [3,Hg,Wg], local_images
    [N,3,Hl,Wl], content_boxes [N,4] in letterbox pixel-edge coordinates
    (left, top, right, bottom), and source_intervals [N,2] giving each window's
    top and bottom edges relative to the expanded box's height.
    """
    if isinstance(image, Image.Image):
        if image.mode not in ("L", "1"):
            raise ValueError("PIL image must be 8-bit grayscale (L) or binary (1)")
        image = np.asarray(image.convert("L"))
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 2
        or image.dtype != np.uint8
        or image.size == 0
    ):
        raise ValueError("image must be a nonempty grayscale uint8 array [H,W]")
    if not isinstance(mask, np.ndarray) or mask.shape != image.shape:
        raise ValueError("mask must be a binary array aligned to the image [H,W]")
    if mask.dtype.kind not in "buif":
        raise ValueError("mask must use zero background and one positive foreground value")
    if mask.dtype != np.bool_:
        values = np.unique(mask)
        if not np.isfinite(values).all() or values[0] < 0 or len(values) > 2 or (
            len(values) == 2 and values[0] != 0
        ):
            raise ValueError("mask must use zero background and one positive foreground value")
    if not isinstance(margin, Real) or not isfinite(margin) or margin < 0:
        raise ValueError("margin must be a finite nonnegative percentage")
    if isinstance(n_windows, bool) or not isinstance(n_windows, Integral) or n_windows < 1:
        raise ValueError("n_windows must be a positive integer")
    if not isinstance(overlap, Real) or not 0 <= float(overlap) < 1:
        raise ValueError("overlap must be in [0,1)")
    for size in (global_size, local_size):
        if (
            not isinstance(size, (tuple, list))
            or len(size) != 2
            or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in size)
        ):
            raise ValueError("target sizes must be (height, width) positive integer pairs")

    left, top, right, bottom = _expanded_roi(mask if mask.dtype == np.bool_ else mask > 0, margin)
    roi_height = bottom - top
    window_height = ceil(roi_height / (1 + (n_windows - 1) * (1 - overlap)))
    starts = np.rint(np.linspace(0, roi_height - window_height, n_windows)).astype(int)

    global_image, _ = _letterbox(image[top:bottom, left:right], global_size)
    local_images = []
    content_boxes = []
    source_intervals = []
    for start in starts:
        end = start + window_height
        local_image, content_box = _letterbox(image[top + start:top + end, left:right], local_size)
        local_images.append(local_image)
        content_boxes.append(content_box)
        source_intervals.append((start / roi_height, end / roi_height))
    return {
        "global_image": global_image,
        "local_images": torch.stack(local_images),
        "content_boxes": torch.tensor(content_boxes, dtype=torch.float32),
        "source_intervals": torch.tensor(source_intervals, dtype=torch.float32),
    }
