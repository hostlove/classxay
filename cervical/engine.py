"""Image-level training, metrics, checkpoints, and CSV predictions."""

import csv
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock, Timeout
from torch.utils.data import DataLoader

from .data import CervicalDataset, GRADE_NAMES, load_splits, prediction_cases
from .model import SchemeBClassifier, ordinal_logits_loss


FORMAT = "cervical-scheme-b-v1"
TRAIN_DEFAULTS = {"batch_size": 2, "lr": 1e-4, "weight_decay": 1e-4, "seed": 42, "weight_by_grade": False}


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable in this PyTorch environment")
    return torch.device(name)


def make_loader(cases, batch_size, workers, device, *, shuffle=False, generator=None):
    return DataLoader(
        CervicalDataset(cases), batch_size=batch_size, shuffle=shuffle, generator=generator,
        num_workers=workers, pin_memory=device.type == "cuda",
    )


def summarize(confusion, loss):
    """Rows = true grades, columns = predicted grades. All four grades count for macro-F1."""
    counts = np.asarray(confusion, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        raise ValueError("cannot compute metrics on an empty split")
    true_count, predicted_count = counts.sum(axis=1), counts.sum(axis=0)
    diagonal = counts.diagonal()
    f1 = np.divide(2.0 * diagonal, true_count + predicted_count,
                   out=np.zeros(4), where=(true_count + predicted_count) > 0)
    grades = np.arange(4)
    distance = np.abs(grades[:, None] - grades[None, :])
    squared_distance = distance ** 2
    expected = np.outer(true_count.astype(float), predicted_count) / total
    expected_disagreement = float((expected * squared_distance).sum())
    observed_disagreement = float((counts * squared_distance).sum())
    return {
        "samples": total, "loss": float(loss), "accuracy": float(diagonal.sum() / total),
        "macro_f1": float(f1.mean()), "mae": float((distance * counts).sum() / total),
        "qwk": 1 - observed_disagreement / expected_disagreement if expected_disagreement else None,
        "recall_by_grade": [float(d / n) if n else None for d, n in zip(diagonal, true_count)],
        "confusion_matrix": counts.tolist(),
    }


def _forward(model, batch, device):
    inputs = {name: tensor.to(device, non_blocking=True) for name, tensor in batch["inputs"].items()}
    return model(**inputs)


def run_epoch(model, loader, device, optimizer=None, *, grade_weights=None):
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros(16, dtype=torch.long, device=device)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    with torch.set_grad_enabled(training):
        for batch in loader:
            labels = batch["grade"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            result = _forward(model, batch, device)
            loss = ordinal_logits_loss(result["logits"], labels, grade_weights=grade_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite ordinal loss")
            if training:
                loss.backward()
                optimizer.step()
            confusion += torch.bincount(4 * labels + result["prediction"], minlength=16)
            loss_sum += loss.detach().double() * labels.numel()
    matrix = confusion.reshape(4, 4).cpu().numpy()
    return summarize(matrix, loss_sum.item() / max(1, int(matrix.sum())))


@contextmanager
def _atomic_output(path, *, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(
        mode="wb" if binary else "w", prefix=f".{path.name}.", suffix=".tmp",
        dir=path.parent, delete=False, encoding=None if binary else "utf-8",
        newline=None if binary else "",
    )
    try:
        with stream:
            yield stream
        os.replace(stream.name, path)
    finally:
        Path(stream.name).unlink(missing_ok=True)


def _write_json(path, value):
    with _atomic_output(path) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _save_checkpoint(path, state):
    with _atomic_output(path, binary=True) as stream:
        torch.save(state, stream)


def _load_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"local checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != FORMAT or state.get("grades") != list(GRADE_NAMES):
        raise ValueError(f"incompatible checkpoint: {path}")
    return state


def _split_digest(partitions):
    records = sorted((split, case.key, case.grade, case.patient_id)
                     for split, cases in partitions.items() for case in cases)
    return hashlib.sha256(json.dumps(records, ensure_ascii=False).encode("utf-8")).hexdigest()


def _training_settings(args, previous):
    settings = TRAIN_DEFAULTS.copy()
    if previous:
        settings.update(previous["settings"])
    for key in TRAIN_DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            if previous and settings[key] != value:
                raise ValueError(f"--resume must keep the saved {key}={settings[key]}")
            settings[key] = value
    return settings


def _log_epoch(row):
    fields = [f"epoch={row['epoch']}"]
    for split in ("train", "val"):
        if split in row:
            metrics = row[split]
            fields.append(f"{split}_loss={metrics['loss']:.5f} {split}_accuracy={metrics['accuracy']:.4f}")
    print("  ".join(fields), flush=True)


def _sync_reports(directory, weights, history):
    latest = history[-1]
    if "val" in latest:
        previous_best = min((item["val"]["loss"] for item in history[:-1]), default=math.inf)
        if latest["val"]["loss"] < previous_best:
            _save_checkpoint(directory / "best.pt", weights)
    _write_json(directory / "history.json", history)


def train(args):
    device = select_device(args.device)
    names = ("train",) if args.train_only else ("train", "val")
    partitions = load_splits(args.data_root, names, split_csv=args.split_csv, train_only=args.train_only)
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(directory / ".training.lock"), timeout=0)
    try:
        with lock:
            _fit(args, directory, partitions, device)
    except Timeout as exc:
        raise ValueError(f"another training process is using {directory}") from exc


def _fit(args, directory, partitions, device):
    last_path = directory / "last.pt"
    if not args.resume and any((directory / name).exists() for name in ("last.pt", "best.pt", "history.json")):
        raise ValueError(f"run outputs already exist in {directory}; use --resume or another output directory")
    previous = _load_checkpoint(last_path) if args.resume else None
    signature = _split_digest(partitions)
    if previous and (previous.get("split_digest") != signature or "optimizer" not in previous):
        raise ValueError("resume requires last.pt and the same image/grade/patient split assignments")
    settings = _training_settings(args, previous)
    torch.manual_seed(settings["seed"])
    model = SchemeBClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    history = []
    if previous:
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        history = previous["history"]
        weights = {key: previous[key] for key in ("format", "grades", "epoch", "model")}
        _sync_reports(directory, weights, history)
        del weights, previous
    start_epoch = history[-1]["epoch"] + 1 if history else 1
    if start_epoch > args.epochs:
        print(f"already completed {start_epoch - 1} epochs in {directory}", flush=True)
        return
    generator = torch.Generator()
    train_loader = make_loader(partitions["train"], settings["batch_size"], args.workers, device,
                               shuffle=True, generator=generator)
    val_loader = None
    if "val" in partitions:
        val_loader = make_loader(partitions["val"], settings["batch_size"], args.workers, device)
    print(f"device={device}  output={directory}  grades={list(GRADE_NAMES)}", flush=True)
    grade_weights = None
    for split, cases in partitions.items():
        counts = np.bincount([case.grade for case in cases], minlength=len(GRADE_NAMES))
        print(f"{split}: {len(cases)} images, counts={counts.tolist()}", flush=True)
        if split == "train" and settings["weight_by_grade"]:
            train_counts = torch.from_numpy(counts).float()
            factors = train_counts.clamp_min(1).rsqrt().masked_fill(train_counts == 0, 0)
            average_weight = train_counts.dot(factors) / train_counts.sum()
            grade_weights = factors / average_weight
            print(f"train_grade_weights={grade_weights.tolist()}", flush=True)
            grade_weights = grade_weights.to(device)
    if val_loader is None:
        print("train-only: training metrics only; no validation split or best checkpoint", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_seed = settings["seed"] + epoch
        torch.manual_seed(epoch_seed)
        generator.manual_seed(epoch_seed)
        row = {
            "epoch": epoch,
            "train": run_epoch(model, train_loader, device, optimizer, grade_weights=grade_weights),
        }
        if val_loader is not None:
            row["val"] = run_epoch(model, val_loader, device)
        history.append(row)
        weights = {"format": FORMAT, "grades": list(GRADE_NAMES), "epoch": epoch, "model": model.state_dict()}
        _save_checkpoint(last_path, {
            **weights, "optimizer": optimizer.state_dict(), "settings": settings,
            "split_digest": signature, "history": history,
        })
        _sync_reports(directory, weights, history)
        _log_epoch(row)


def _inference_model(path, device):
    state = _load_checkpoint(path)
    model = SchemeBClassifier()
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def evaluate(args):
    device = select_device(args.device)
    cases = load_splits(args.data_root, (args.split,), split_csv=args.split_csv)[args.split]
    model = _inference_model(args.checkpoint, device)
    loader = make_loader(cases, args.batch_size, args.workers, device)
    metrics = {"split": args.split, "grades": list(GRADE_NAMES), **run_epoch(model, loader, device)}
    if args.output:
        _write_json(args.output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))


@torch.inference_mode()
def predict(args):
    device = select_device(args.device)
    cases = prediction_cases(args.input)
    model = _inference_model(args.checkpoint, device)
    loader = make_loader(cases, args.batch_size, args.workers, device)
    with _atomic_output(args.output) as stream:
        writer = csv.writer(stream)
        writer.writerow(("image_path", "predicted_grade", "predicted_label",
                         "p_normal", "p_mild", "p_moderate", "p_severe"))
        for batch in loader:
            result = _forward(model, batch, device)
            probabilities = result["probabilities"]
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("non-finite class probability")
            for key, grade, values in zip(batch["key"], result["prediction"].tolist(), probabilities.tolist()):
                writer.writerow((key, grade, GRADE_NAMES[grade], *(f"{p:.9g}" for p in values)))
    print(f"wrote {len(cases)} image predictions to {args.output}")
