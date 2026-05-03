from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from config import (
    ARTIFACTS_DIR,
    CHECKPOINT_DIR,
    DATA_DIR,
    RUNS_DIR,
    batchsize,
    data_yaml_name,
    epochs,
    final_weights_name,
    patience,
    pretrained_weights,
    resize_x,
    resize_y,
    run_name,
    workers,
)
from dataset import materialize_yolo_dataset, write_data_yaml
from model import TargetDetector


def resolve_device() -> str | int:
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def latest_best_weights(runs_root: Path) -> Path | None:
    weight_files = sorted(runs_root.rglob("best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return weight_files[0] if weight_files else None


def the_trainer(
    data_root: Path | str = DATA_DIR,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    output_root: Path | str = ARTIFACTS_DIR,
    weights: str | Path | None = pretrained_weights,
    run_label: str = run_name,
    include_location_substrings: list[str] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_root)
    dataset_root = output_path / "yolo_dataset"
    data_yaml = output_path / data_yaml_name

    materialize_yolo_dataset(source_root=data_root, output_root=dataset_root, include_location_substrings=include_location_substrings)
    write_data_yaml(dataset_root, data_yaml)

    detector = TargetDetector(weights=weights)
    detector.train(
        data=str(data_yaml),
        epochs=epochs_override or epochs,
        imgsz=max(resize_x, resize_y),
        batch=batch_size_override or batchsize,
        device=resolve_device(),
        project=str(RUNS_DIR),
        name=run_label,
        single_cls=True,
        pretrained=True,
        patience=patience,
        workers=workers,
        verbose=True,
        val=True,
    )

    best_weights = latest_best_weights(RUNS_DIR)
    final_weights = CHECKPOINT_DIR / final_weights_name
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if best_weights is not None:
        shutil.copy2(best_weights, final_weights)

    return {
        "dataset_root": str(dataset_root),
        "data_yaml": str(data_yaml),
        "best_weights": str(best_weights) if best_weights is not None else None,
        "final_weights": str(final_weights),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the target detector with optional location filtering")
    parser.add_argument("--data-dir", default=None, help="Root data dir (overrides config.DATA_DIR)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--weights", default=None, help="Weights path to use (overrides config)")
    parser.add_argument("--run-label", default=None, help="Run label/name")
    parser.add_argument("--locations", default=None, help="Comma-separated location selectors to include (e.g. 1,2,4 or 1,2,3,4,nv)")
    parser.add_argument("--nvg-on", action="store_true", help="Allow location 5 / nv (night vision data) in training")

    args = parser.parse_args()

    include_subs = None
    # map location numbers to filename substrings
    loc_map = {
        "1": ["19-38", "19-41"],
        "2": ["19-48"],
        "3": ["23-29"],
        "4": ["22-13"],
        "5": ["23-50"],
        "nv": ["23-50"],
    }

    def _expand_location_selection(selection: list[str], allow_nvg: bool) -> list[str]:
        expanded: list[str] = []
        for item in selection:
            key = item.lower()
            if key in ("5", "nv") and not allow_nvg:
                continue
            values = loc_map.get(key)
            if values is not None:
                expanded.extend(values)
        return expanded

    if args.locations:
        requested = [s.strip() for s in args.locations.split(",") if s.strip()]
        include_subs = _expand_location_selection(requested, args.nvg_on)
    else:
        default_locations = ["1", "2", "3", "4"]
        if args.nvg_on:
            default_locations.append("5")
        include_subs = _expand_location_selection(default_locations, args.nvg_on)

    result = the_trainer(
        data_root=args.data_dir or DATA_DIR,
        epochs_override=args.epochs,
        batch_size_override=args.batch,
        weights=args.weights or pretrained_weights,
        run_label=args.run_label or run_name,
        include_location_substrings=include_subs,
    )
    print(result)