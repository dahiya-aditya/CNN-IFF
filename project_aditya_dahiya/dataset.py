from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from config import ARTIFACTS_DIR, DATA_DIR, class_names, resize_x, resize_y, seed, train_split, val_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SamplePaths:
    image_path: Path
    label_path: Path | None


def discover_image_files(root: Path | str) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(
        path for path in root_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _label_candidates(image_path: Path, data_root: Path) -> list[Path]:
    candidates = [image_path.with_suffix(".txt")]
    candidates.append(image_path.parent / f"{image_path.stem}.txt")

    try:
        relative_path = image_path.relative_to(data_root)
    except ValueError:
        relative_path = None

    if relative_path is not None:
        candidates.append(data_root / "labels" / relative_path.with_suffix(".txt"))
        if relative_path.parts and relative_path.parts[0] == "images":
            candidates.append(data_root / "labels" / Path(*relative_path.parts[1:]).with_suffix(".txt"))

    unique_candidates: list[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    return unique_candidates


def resolve_label_path(image_path: Path, data_root: Path) -> Path | None:
    for candidate in _label_candidates(image_path, data_root):
        if candidate.exists():
            return candidate
    return None


def parse_yolo_label_file(label_path: Path | None) -> list[tuple[int, float, float, float, float]]:
    if label_path is None or not label_path.exists():
        return []

    annotations: list[tuple[int, float, float, float, float]] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])
        annotations.append((class_id, x_center, y_center, box_width, box_height))
    return annotations


class TargetDataset(Dataset):
    def __init__(self, root: Path | str = DATA_DIR, resize: tuple[int, int] = (resize_x, resize_y)) -> None:
        self.root = Path(root)
        self.resize = resize
        self.image_paths = discover_image_files(self.root)
        if not self.image_paths:
            raise FileNotFoundError(f"No image files were found in {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            original_size = image.size
            if self.resize is not None and image.size != self.resize:
                image = image.resize(self.resize, Image.Resampling.BILINEAR)

        label_path = resolve_label_path(image_path, self.root)
        annotations = parse_yolo_label_file(label_path)

        if annotations:
            target_tensor = torch.tensor(annotations, dtype=torch.float32)
        else:
            target_tensor = torch.zeros((0, 5), dtype=torch.float32)

        return {
            "image": TF.to_tensor(image),
            "target": target_tensor,
            "image_path": str(image_path),
            "label_path": str(label_path) if label_path is not None else None,
            "original_size": original_size,
        }


def detection_collate_fn(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    images = torch.stack([item["image"] for item in batch])
    targets = [item["target"] for item in batch]
    metadata = [
        {
            "image_path": item["image_path"],
            "label_path": item["label_path"],
            "original_size": item["original_size"],
        }
        for item in batch
    ]
    return {"images": images, "targets": targets, "metadata": metadata}


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )


def _flatten_image_name(source_root: Path, image_path: Path) -> str:
    relative_path = image_path.relative_to(source_root)
    flattened = relative_path.with_suffix("").as_posix().replace("/", "__")
    return flattened


def split_image_paths(
    image_paths: Sequence[Path],
    train_ratio: float = train_split,
    val_ratio: float = val_split,
    random_seed: int = seed,
) -> dict[str, list[Path]]:
    if not image_paths:
        return {"train": [], "val": [], "test": []}

    shuffled_paths = list(image_paths)
    random.Random(random_seed).shuffle(shuffled_paths)

    total = len(shuffled_paths)
    train_count = max(1, int(total * train_ratio))
    val_count = max(1 if total > 1 else 0, int(total * val_ratio))

    if train_count + val_count > total:
        val_count = max(0, total - train_count)

    train_paths = shuffled_paths[:train_count]
    val_paths = shuffled_paths[train_count : train_count + val_count]
    test_paths = shuffled_paths[train_count + val_count :]

    return {"train": train_paths, "val": val_paths, "test": test_paths}


def materialize_yolo_dataset(
    source_root: Path | str = DATA_DIR,
    output_root: Path | str = ARTIFACTS_DIR / "yolo_dataset",
    train_ratio: float = train_split,
    val_ratio: float = val_split,
    random_seed: int = seed,
    include_location_substrings: Sequence[str] | None = None,
) -> Path:
    source_path = Path(source_root)
    output_path = Path(output_root)
    image_paths = discover_image_files(source_path)
    if not image_paths:
        raise FileNotFoundError(f"No image files were found in {source_path}")

    # Optionally filter images by substrings (e.g., location markers in filenames)
    if include_location_substrings:
        lowered = [s.lower() for s in include_location_substrings]
        filtered = [p for p in image_paths if any(sub in p.name.lower() for sub in lowered)]
        image_paths = filtered

    splits = split_image_paths(image_paths, train_ratio=train_ratio, val_ratio=val_ratio, random_seed=random_seed)

    for split_name, split_images in splits.items():
        split_images_dir = output_path / split_name / "images"
        split_labels_dir = output_path / split_name / "labels"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            flat_name = _flatten_image_name(source_path, image_path)
            destination_image = split_images_dir / f"{flat_name}{image_path.suffix.lower()}"
            shutil.copy2(image_path, destination_image)

            label_path = resolve_label_path(image_path, source_path)
            destination_label = split_labels_dir / f"{flat_name}.txt"
            if label_path is not None and label_path.exists():
                shutil.copy2(label_path, destination_label)
            else:
                destination_label.write_text("", encoding="utf-8")

    return output_path


def write_data_yaml(dataset_root: Path | str, yaml_path: Path | str, names: Sequence[str] = class_names) -> Path:
    dataset_path = Path(dataset_root)
    yaml_file = Path(yaml_path)
    yaml_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "path": str(dataset_path.resolve()),
        "train": str((dataset_path / "train" / "images").resolve()),
        "val": str((dataset_path / "val" / "images").resolve()),
        "test": str((dataset_path / "test" / "images").resolve()),
        "nc": len(names),
        "names": list(names),
    }

    with yaml_file.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    return yaml_file