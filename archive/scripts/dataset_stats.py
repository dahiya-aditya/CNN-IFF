#!/usr/bin/env python3
"""Quick dataset stats for YOLO-style image + sibling .txt labels.

Default behavior excludes common auxiliary folders inside data/:
- discard/
- duplicates_removed/

Usage examples:
    python scripts/dataset_stats.py
    python scripts/dataset_stats.py --data-dir data
    python scripts/dataset_stats.py --include-discard
    python scripts/dataset_stats.py --show-sources
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SourceStats:
    total_images: int = 0
    annotated_images: int = 0
    images_with_targets: int = 0
    background_only_images: int = 0
    target_instances: int = 0


def is_excluded(path: Path, include_discard: bool) -> bool:
    lower_parts = [p.lower() for p in path.parts]
    if "duplicates_removed" in lower_parts:
        return True
    if not include_discard and "discard" in lower_parts:
        return True
    return False


def count_valid_yolo_lines(label_path: Path) -> int:
    if not label_path.exists():
        return 0

    valid = 0
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            int(float(parts[0]))
            float(parts[1])
            float(parts[2])
            float(parts[3])
            float(parts[4])
        except ValueError:
            continue
        valid += 1
    return valid


def source_from_filename(name: str) -> str:
    stem = Path(name).stem
    m = re.match(r"^(.*)_\d+$", stem)
    return m.group(1) if m else stem


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Show useful annotation stats for data/")
    parser.add_argument("--data-dir", default="data", help="Path to dataset root (default: data)")
    parser.add_argument("--include-discard", action="store_true", help="Include discard/ in totals")
    parser.add_argument("--show-sources", action="store_true", help="Show per-source breakdown")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    all_images = [
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not is_excluded(p, args.include_discard)
    ]

    source_stats: dict[str, SourceStats] = defaultdict(SourceStats)

    total_images = len(all_images)
    annotated_images = 0
    images_with_targets = 0
    background_only_images = 0
    unannotated_images = 0
    total_target_instances = 0

    for img in all_images:
        source = source_from_filename(img.name)
        source_stats[source].total_images += 1

        label_path = img.with_suffix(".txt")
        if not label_path.exists():
            unannotated_images += 1
            continue

        annotated_images += 1
        source_stats[source].annotated_images += 1

        n_targets = count_valid_yolo_lines(label_path)
        if n_targets > 0:
            images_with_targets += 1
            total_target_instances += n_targets
            source_stats[source].images_with_targets += 1
            source_stats[source].target_instances += n_targets
        else:
            background_only_images += 1
            source_stats[source].background_only_images += 1

    # Label-side sanity checks
    all_labels = [
        p
        for p in data_dir.rglob("*.txt")
        if p.is_file() and not is_excluded(p, args.include_discard)
    ]
    orphan_labels = [p for p in all_labels if not p.with_suffix(".jpg").exists() and not p.with_suffix(".jpeg").exists() and not p.with_suffix(".png").exists()]

    discard_images = 0
    discard_dir = data_dir / "discard"
    if discard_dir.exists():
        discard_images = len([p for p in discard_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

    duplicates_images = 0
    duplicates_dir = data_dir / "duplicates_removed"
    if duplicates_dir.exists():
        duplicates_images = len([p for p in duplicates_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

    print(f"Data dir: {data_dir}")
    print()
    print("=== Core Counts ===")
    print(f"Total images:                 {total_images}")
    print(f"Annotated images:             {annotated_images}")
    print(f"Unannotated images:           {unannotated_images}")
    print(f"Images with targets:          {images_with_targets}")
    print(f"Background-only annotations:  {background_only_images}")
    print(f"Total target instances:       {total_target_instances}")
    print()
    print("=== Ratios ===")
    print(f"Annotation coverage:          {pct(annotated_images, total_images):5.1f}%")
    print(f"Target among annotated:       {pct(images_with_targets, annotated_images):5.1f}%")
    print(f"Background among annotated:   {pct(background_only_images, annotated_images):5.1f}%")
    if background_only_images > 0:
        print(f"Target:background ratio:      {images_with_targets/background_only_images:5.2f}:1")
    else:
        print("Target:background ratio:      N/A (no background-only labels)")
    print()
    print("=== Quality / Sanity ===")
    print(f"Avg targets per target image: {total_target_instances / images_with_targets:5.2f}" if images_with_targets else "Avg targets per target image: 0.00")
    print(f"Orphan label files:           {len(orphan_labels)}")
    print(f"Images in discard/:           {discard_images}")
    print(f"Images in duplicates_removed/: {duplicates_images}")

    if args.show_sources:
        print()
        print("=== Per-Source Breakdown ===")
        header = (
            "source",
            "images",
            "annotated",
            "target_imgs",
            "bg_imgs",
            "instances",
            "coverage%",
        )
        print(f"{header[0]:<28} {header[1]:>7} {header[2]:>10} {header[3]:>11} {header[4]:>8} {header[5]:>10} {header[6]:>9}")
        for source, st in sorted(source_stats.items(), key=lambda kv: kv[1].total_images, reverse=True):
            print(
                f"{source[:28]:<28} {st.total_images:>7} {st.annotated_images:>10} {st.images_with_targets:>11} "
                f"{st.background_only_images:>8} {st.target_instances:>10} {pct(st.annotated_images, st.total_images):>8.1f}%"
            )


if __name__ == "__main__":
    main()
