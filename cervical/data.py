"""File pairing, patient-level splits, and the shared training/inference dataset."""

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .inputs import prepare_case


GRADE_NAMES = ("正常", "轻微", "中度", "重度")
GRADE_DIRS = {name: grade for grade, name in enumerate(GRADE_NAMES)}
GRADE_DIRS.update({str(grade): grade for grade in range(len(GRADE_NAMES))})
IMAGE_SUFFIXES = {".jpg", ".jpeg"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Case:
    image: Path
    mask: Path
    key: str
    grade: int | None = None
    patient_id: str | None = None


def _pairs_in(folder: Path, only_image: Path | None = None) -> Iterator[tuple[Path, Path]]:
    """Index a directory once. Match names case-insensitively on Linux and Windows."""
    images: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        stem = path.stem.casefold()
        if path.suffix.casefold() in IMAGE_SUFFIXES:
            target = images
        elif path.suffix.casefold() == ".png" and stem.endswith("_seg"):
            stem = stem[:-4]
            target = masks
        else:
            continue
        if only_image is not None and stem != only_image.stem.casefold():
            continue
        if stem in target:
            raise ValueError(f"ambiguous pair in {folder}: {target[stem].name} and {path.name}")
        target[stem] = path
    for stem in sorted(images.keys() | masks.keys()):
        if stem not in images:
            raise ValueError(f"segmentation PNG has no matching JPG/JPEG: {masks[stem]}")
        if stem not in masks:
            raise ValueError(f"missing <stem>_seg.png for {images[stem]}")
        yield images[stem], masks[stem]


def _grade_cases(root, directory):
    if not directory.is_dir():
        raise ValueError(f"missing split directory {directory}; use a patient split CSV or prepare train/val folders")
    cases = []
    for folder in sorted(directory.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name not in GRADE_DIRS:
            if any(p.is_file() and (p.suffix.lower() in IMAGE_SUFFIXES or p.name.casefold().endswith("_seg.png"))
                   for p in folder.iterdir()):
                raise ValueError(f"unrecognized grade folder containing image data: {folder}")
            continue
        for image, mask in _pairs_in(folder):
            cases.append(Case(image, mask, image.relative_to(root).as_posix(), GRADE_DIRS[folder.name]))
    if not cases:
        raise ValueError(f"no labeled image pairs in {directory}; expected grade directories 0..3 or 正常/轻微/中度/重度")
    return cases


def _manifest_splits(root, csv_path):
    partitions: dict[str, list[Case]] = {name: [] for name in SPLITS}
    seen_images: set[Path] = set()
    patient_split: dict[str, str] = {}
    mask_by_folder: dict[Path, dict[Path, Path]] = {}
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image_path", "patient_id", "split"}.issubset(reader.fieldnames or ()):
            raise ValueError("split CSV requires columns image_path,patient_id,split")
        for row in reader:
            path_text = (row["image_path"] or "").strip().replace("\\", "/")
            patient = (row["patient_id"] or "").strip()
            split = (row["split"] or "").strip().lower()
            location = f"{csv_path}, line {reader.line_num}"
            if not path_text or Path(path_text).is_absolute() or split not in SPLITS or not patient:
                raise ValueError(f"{location}: expected a relative image path, patient ID and train/val/test")
            image = (root / path_text).resolve()
            if not image.is_relative_to(root) or not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"{location}: cannot find this original JPG/JPEG under the data root: {path_text}")
            if image in seen_images:
                raise ValueError(f"{location}: duplicate image in split CSV: {path_text}")
            if patient in patient_split and patient_split[patient] != split:
                raise ValueError(f"{location}: the same patient is assigned to multiple splits")
            folder = image.parent
            if folder.name not in GRADE_DIRS:
                raise ValueError(f"{location}: parent directory of image must give its grade (0..3 or Chinese name)")
            if folder not in mask_by_folder:
                mask_by_folder[folder] = dict(_pairs_in(folder))
            seen_images.add(image)
            patient_split[patient] = split
            partitions[split].append(Case(
                image, mask_by_folder[folder][image], image.relative_to(root).as_posix(),
                GRADE_DIRS[folder.name], patient,
            ))
    return partitions


def load_splits(data_root, names, *, split_csv=None, train_only=False):
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise ValueError(f"data root does not exist: {root}")
    if train_only:
        if split_csv is not None or tuple(names) != ("train",):
            raise ValueError("train-only expects a single unpartitioned grade directory tree")
        return {"train": _grade_cases(root, root)}
    if split_csv is None:
        return {name: _grade_cases(root, root / name) for name in names}
    partitions = _manifest_splits(root, split_csv)
    for name in names:
        if not partitions[name]:
            raise ValueError(f"split CSV has no images for '{name}'")
    return {name: sorted(partitions[name], key=lambda case: case.key) for name in names}


def prediction_cases(source):
    source = Path(source).resolve()
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [Case(image, mask, image.name) for image, mask in _pairs_in(source.parent, only_image=source)]
    if not source.is_dir():
        raise ValueError(f"prediction input must be a JPG/JPEG or a directory of pairs: {source}")
    folders = sorted({p.parent for p in source.rglob("*") if p.is_file() and (
        p.suffix.lower() in IMAGE_SUFFIXES or p.name.casefold().endswith("_seg.png")
    )})
    cases = [Case(image, mask, image.relative_to(source).as_posix())
             for folder in folders for image, mask in _pairs_in(folder)]
    if not cases:
        raise ValueError(f"no JPG/JPEG and segmentation PNG pairs in {source}")
    return cases


def _foreground(segmentation):
    if "A" in segmentation.getbands():
        channel = segmentation.getchannel("A")
    elif "transparency" in segmentation.info:
        channel = segmentation.convert("RGBA").getchannel("A")
    elif segmentation.mode in ("1", "L"):
        channel = segmentation.convert("L")
    else:
        raise ValueError("expected an alpha-channel segmentation PNG or a binary grayscale PNG")
    levels = [index for index, count in enumerate(channel.histogram()) if count]
    if len(levels) > 2 or levels[0] != 0 or levels[-1] == 0:
        raise ValueError("expected a binary mask with zero background and one positive foreground value")
    return np.asarray(channel) > 0


class CervicalDataset(Dataset):
    def __init__(self, cases):
        self.cases = cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        try:
            with Image.open(case.image) as original, Image.open(case.mask) as segmentation:
                if original.size != segmentation.size:
                    raise ValueError(f"original size {original.size} differs from mask size {segmentation.size}")
                image = np.asarray(original.convert("L"))
                mask = _foreground(segmentation)
            inputs = prepare_case(image, mask)
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid image pair {case.key}: {exc}") from exc
        item = {"inputs": inputs, "key": case.key}
        if case.grade is not None:
            item["grade"] = case.grade
        return item
