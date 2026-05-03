from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2

from config import confidence_threshold, final_weights_name, iou_threshold, pretrained_weights
from model import TargetDetector


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_detector(weights: str | Path | None = None) -> TargetDetector:
    candidate = Path(weights) if weights is not None else Path("checkpoints") / final_weights_name
    if candidate.suffix.lower() == ".pth":
        pt_candidate = candidate.with_suffix(".pt")
        if not pt_candidate.exists() and candidate.exists():
            import shutil

            shutil.copy2(candidate, pt_candidate)
        candidate = pt_candidate
    if not candidate.exists():
        candidate = Path(pretrained_weights)
    return TargetDetector(weights=candidate)


def _output_path_for_source(source_path: Path, output_path: str | Path | None) -> Path:
    if output_path is not None:
        return Path(output_path)
    return source_path.with_name(f"{source_path.stem}_predicted{source_path.suffix}")


def _draw_result(frame, result) -> tuple[Any, list[dict[str, float]]]:
    detections: list[dict[str, float]] = []
    boxes = result.boxes
    if boxes is None:
        return frame, detections

    for box in boxes:
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
        class_id = int(box.cls[0].item()) if box.cls is not None else 0
        label = f"Target {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        detections.append(
            {
                "class_id": class_id,
                "confidence": confidence,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )

    return frame, detections


def _predict_image(source_path: Path, output_path: Path | None, detector: TargetDetector) -> dict[str, Any]:
    predictions = detector.predict(source=str(source_path), conf=confidence_threshold, iou=iou_threshold, verbose=False)
    result = predictions[0]
    image = cv2.imread(str(source_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {source_path}")
    annotated_image, detections = _draw_result(image, result)
    destination = _output_path_for_source(source_path, output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), annotated_image)
    return {"source": str(source_path), "output": str(destination), "detections": detections}


def _predict_video(source_path: Path, output_path: Path | None, detector: TargetDetector) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    destination = _output_path_for_source(source_path, output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") if destination.suffix.lower() != ".avi" else cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(destination), fourcc, fps, (width, height))

    frame_count = 0
    detection_count = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        result = detector.predict(source=frame, conf=confidence_threshold, iou=iou_threshold, verbose=False)[0]
        annotated_frame, detections = _draw_result(frame, result)
        writer.write(annotated_frame)
        frame_count += 1
        detection_count += len(detections)

    capture.release()
    writer.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames were read from video: {source_path}")

    return {
        "source": str(source_path),
        "output": str(destination),
        "frames": frame_count,
        "detections": detection_count,
    }


def the_predictor(
    source: str | Path | Iterable[str | Path],
    output_path: str | Path | None = None,
    weights: str | Path | None = None,
) -> Any:
    detector = load_detector(weights)

    if isinstance(source, (list, tuple)):
        return [the_predictor(item, output_path=output_path, weights=weights) for item in source]

    source_path = Path(source)
    output_destination = Path(output_path) if output_path is not None else None

    if source_path.suffix.lower() in VIDEO_EXTENSIONS:
        return _predict_video(source_path, output_destination, detector)
    return _predict_image(source_path, output_destination, detector)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python predict.py <image-or-video-path> [output_path]")

    input_path = sys.argv[1]
    explicit_output = sys.argv[2] if len(sys.argv) > 2 else None
    print(the_predictor(input_path, output_path=explicit_output))