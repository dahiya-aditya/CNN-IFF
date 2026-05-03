#!/usr/bin/env python3
"""
Process videos under `demo_video/`: sample at target FPS (default 30), run YOLO inference on each sampled frame,
draw predicted boxes (above confidence threshold) and write a stitched output video per input.

Usage examples:
    python scripts/demo_infer_video.py --input-dir demo_video --output-dir demo_out --fps 30 --conf 0.25

The script prefers `ultralytics.YOLO`. If no `--weights` provided it will try to use
`project_aditya_dahiya/checkpoints/final_weights.pt` then `yolov8n.pt` in workspace.
"""

import argparse
from pathlib import Path
import sys
import cv2
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

import tempfile


def _resolve_weights(provided: str | None, workspace: Path) -> Path:
    if provided:
        return Path(provided)
    candidates = [
        workspace / "project_aditya_dahiya" / "checkpoints" / "final_weights.pt",
        workspace / "yolov8n.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path("yolov8n.pt")


def draw_predictions_on_frame(frame, result, model_names, conf_threshold):
    # result: ultralytics result object
    if getattr(result, "boxes", None) is None:
        return frame
    try:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
    except Exception:
        return frame

    for box, conf, cls in zip(boxes, confs, clss):
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        # Draw red box and label as 'Hostile'
        box_color = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        label = f"Hostile {conf:.2f}"
        t_w, t_h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - t_h - 6), (x1 + t_w + 6, y1), box_color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def process_video(video_path: Path, out_path: Path, model, conf_threshold: float, target_fps: float | None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠ Cannot open video: {video_path}")
        return False

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine output framerate and sampling step.
    # If user passed a target_fps, set out_fps accordingly and sample approximately.
    # If target_fps is None, preserve source video's framerate and process every frame.
    out_fps = float(orig_fps)
    step = 1
    if target_fps is not None and target_fps > 0:
        out_fps = float(target_fps)
        if orig_fps > 0:
            step = max(1, int(round(orig_fps / float(target_fps))))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (width, height))

    model_names = getattr(model, "names", None) if model is not None else None

    frame_idx = 0
    processed = 0
    pbar = tqdm(total=frame_count, desc=video_path.name, unit="fr")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_idx % step) != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            # run inference
            drawn = frame.copy()
            if model is not None:
                try:
                    # ultralytics supports numpy array source for predict
                    res = model.predict(source=frame, conf=conf_threshold, verbose=False)
                    if isinstance(res, (list, tuple)):
                        result = res[0]
                    else:
                        result = res
                except Exception:
                    # fallback: write frame to temp file and run predict on path
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                        cv2.imwrite(tmp.name, frame)
                        res = model.predict(source=tmp.name, conf=conf_threshold, verbose=False)
                        result = res[0] if isinstance(res, (list, tuple)) else res
                try:
                    drawn = draw_predictions_on_frame(drawn, result, model_names, conf_threshold)
                except Exception:
                    pass

            writer.write(drawn)
            processed += 1
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        writer.release()
        cap.release()
    print(f"✓ Wrote {processed} frames -> {out_path} (output fps: {out_fps})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on demo videos and stitch predicted frames back to videos")
    parser.add_argument("--input-dir", default=None, help="Directory containing demo videos or a single video file (default: use demo_video in workspace)")
    parser.add_argument("--output-dir", default=None, help="Directory to write output videos (default: demo_out in workspace for directories, or source folder for single file)")
    parser.add_argument("--weights", default=None, help="Path to YOLO weights (optional)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for drawing boxes")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS for extraction and output (default: match source). Set to override source FPS")
    parser.add_argument("--ext", default=".mp4", help="Output extension (default .mp4)")
    parser.add_argument("--device", default=None, help="Device for model (e.g. 0 or cpu). Pass to ultralytics if needed.")
    args = parser.parse_args()

    workspace_dir = Path(__file__).resolve().parents[1]

    # Default demo folder when no input path provided
    default_demo = workspace_dir / "demo_video"

    video_exts = (".mp4", ".mkv", ".avi", ".mov", ".webm")

    # Determine videos to process and output locations
    videos = []
    explicit_input = args.input_dir is not None

    if not explicit_input:
        # No input specified -> use default demo folder and pick first video
        if not default_demo.exists():
            print(f"❌ Default demo folder not found: {default_demo}")
            sys.exit(1)
        cand = sorted([p for p in default_demo.iterdir() if p.suffix.lower() in video_exts])
        if not cand:
            print(f"❌ No videos found in default demo folder: {default_demo}")
            sys.exit(1)
        videos = [cand[0]]
        # output will be alongside source unless --output-dir provided
        output_dir = Path(args.output_dir) if args.output_dir is not None else videos[0].parent
    else:
        input_path = Path(args.input_dir)
        if not input_path.exists():
            print(f"❌ Input path does not exist: {input_path}")
            sys.exit(1)
        if input_path.is_file():
            videos = [input_path]
            output_dir = Path(args.output_dir) if args.output_dir is not None else input_path.parent
        else:
            videos = sorted([p for p in input_path.iterdir() if p.suffix.lower() in video_exts])
            if not videos:
                print(f"❌ No videos found in {input_path}")
                sys.exit(1)
            output_dir = Path(args.output_dir) if args.output_dir is not None else (workspace_dir / "demo_out")

    # load model if available
    model = None
    if YOLO is None:
        print("⚠ ultralytics not installed — running will skip inference and just copy frames")
    else:
        weights = _resolve_weights(args.weights, workspace_dir)
        try:
            model = YOLO(str(weights)) if args.device is None else YOLO(str(weights), device=args.device)
            print(f"✓ Loaded model {weights}")
        except Exception as e:
            print(f"⚠ Failed to load weights {weights}: {e}\nContinuing without model (frames will be copied)")
            model = None

    # Process each selected video. For single-file cases, prefer Processed_<original> naming.
    for vid in videos:
        if len(videos) == 1:
            out_name = f"Processed_{vid.stem}{args.ext}"
        else:
            out_name = f"{vid.stem}_pred{args.ext}"
        out_path = output_dir / out_name
        process_video(vid, out_path, model, args.conf, args.fps)

    print("All done.")


if __name__ == '__main__':
    main()
