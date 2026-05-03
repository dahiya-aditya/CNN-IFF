from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from config import final_weights_name, pretrained_weights


class TargetDetector:
    def __init__(self, weights: str | Path | None = None) -> None:
        self.weights = Path(weights) if weights is not None else Path(pretrained_weights)
        self.model = YOLO(str(self.weights), task="detect")

    def train(self, **kwargs: Any) -> Any:
        return self.model.train(**kwargs)

    def predict(self, **kwargs: Any) -> Any:
        return self.model.predict(**kwargs)

    def val(self, **kwargs: Any) -> Any:
        return self.model.val(**kwargs)

    def save(self, destination: str | Path = final_weights_name) -> Path:
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if self.weights.exists():
            import shutil

            shutil.copy2(self.weights, destination_path)
        else:
            self.model.save(str(destination_path))
        return destination_path


def build_model(weights: str | Path | None = None) -> TargetDetector:
    return TargetDetector(weights=weights)