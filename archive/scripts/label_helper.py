#!/usr/bin/env python3
"""
Simple YOLO label helper - Click to draw boxes around targets.

Usage:
    python scripts/label_helper.py
    
Controls:
    - Click and drag to draw rectangle around target
    - Press 's' to save label
    - Press 'n' to skip to next image
    - Press 'q' to quit
"""

import cv2
import shutil
import argparse
from pathlib import Path
import sys

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

class YOLOLabelHelper:
    def __init__(self, data_dir=None, assist=False, weights=None, conf=0.25, iou=0.45, max_det=20, include_labeled=False, start_at=None, labeled_only=False):
        workspace_dir = Path(__file__).resolve().parents[1]
        default_data_dir = workspace_dir / "data"
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir
        self.data_dir.mkdir(exist_ok=True)
        self.include_labeled = include_labeled
        self.start_at = start_at
        self.labeled_only = labeled_only
        self.workspace_dir = workspace_dir
        # folder for discarded/irrelevant images
        self.discard_dir = self.data_dir / "discard"
        self.discard_dir.mkdir(exist_ok=True)
        
        # Find images to label
        if self.labeled_only:
            # only images with existing labels (non-empty)
            self.images = sorted([
                f for f in self.data_dir.glob("*")
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                and (self.data_dir / f"{f.stem}.txt").exists()
                and (self.data_dir / f"{f.stem}.txt").stat().st_size > 0
            ])
        elif self.include_labeled:
            self.images = sorted([
                f for f in self.data_dir.glob("*")
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ])
        else:
            # default: only unlabeled images
            self.images = sorted([
                f for f in self.data_dir.glob("*")
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                and not (self.data_dir / f"{f.stem}.txt").exists()
            ])
        
        if not self.images:
            print("❌ No images found in", self.data_dir)
            sys.exit(1)

        print(f"Found {len(self.images)} image(s) to review")

        # starting index (allow resuming from a specific image)
        self.current_idx = 0
        if self.start_at:
            start_name = Path(self.start_at).name
            start_stem = Path(start_name).stem
            start_key = self._normalized_image_key(start_stem)
            matched = False
            for idx, f in enumerate(self.images):
                candidate_key = self._normalized_image_key(f.stem)
                if f.name == start_name or f.stem == start_stem or candidate_key == start_key:
                    self.current_idx = idx
                    matched = True
                    print(f"Starting from image {idx+1}: {f.name}")
                    break
            if not matched:
                print(f"⚠ Start image not found: {self.start_at} (starting from beginning)")
        self.drawing = False
        self.boxes = []
        self.start_point = None
        self.image = None
        self.image_orig = None
        # interactive edit state
        self.moving = False
        self.moving_idx = None
        self.move_offset = (0, 0)
        self.resizing = False
        self.resizing_idx = None
        self.resizing_corner = None
        self.handle_size = 4
        self.current_box = None
        # Display constraints: fit to this window size when image is larger
        # Use 1920x1080 by default to preserve detail; change if your screen is smaller
        self.max_display = (1920, 1080)
        # scale factors from display coords -> original coords
        self.scale_x = 1.0
        self.scale_y = 1.0
        # zoom state over the fitted display image
        self.zoom = 1.0
        self.zoom_min = 1.0
        self.zoom_max = 8.0
        self.view_origin_x = 0.0
        self.view_origin_y = 0.0
        self.base_display_w = 0
        self.base_display_h = 0
        self.last_mouse_pos = (0, 0)

        self.assist = assist
        self.assist_conf = conf
        self.assist_iou = iou
        self.assist_max_det = max_det
        self.assist_weights = Path(weights) if weights else None
        self.predictor = None
        self.prediction_cache = {}

        if self.assist:
            self._init_predictor()
        
    def mouse_callback(self, event, x, y, flags, param):
        # Mouse callback works in DISPLAY coordinates. Boxes are stored in display coords.
        self.last_mouse_pos = (x, y)
        if self.base_display_w <= 0 or self.base_display_h <= 0:
            return

        if event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1
            if delta > 0:
                self._adjust_zoom(1.25, (x, y))
            else:
                self._adjust_zoom(1 / 1.25, (x, y))
            self.redraw_boxes()
            return

        bx, by = self._window_to_base_point(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            # check if clicking near a corner for resizing
            for idx, box in enumerate(self.boxes):
                corner = self._near_corner(bx, by, box)
                if corner is not None:
                    self.resizing = True
                    self.resizing_idx = idx
                    self.resizing_corner = corner
                    return

            # check if clicking inside an existing box to move
            for idx, box in enumerate(self.boxes):
                if self._point_in_box(bx, by, box):
                    self.moving = True
                    self.moving_idx = idx
                    x1, y1, x2, y2 = box
                    self.move_offset = (bx - x1, by - y1)
                    return

            # otherwise start drawing a new box
            self.drawing = True
            self.start_point = (bx, by)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                x1, y1 = self.start_point
                x2, y2 = bx, by
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                self.current_box = (x1, y1, x2, y2)
                self.redraw_boxes(extra_box=self.current_box, extra_color=(0, 255, 0))
            elif self.moving and self.moving_idx is not None:
                # move the selected box while preserving size
                x1, y1, x2, y2 = self.boxes[self.moving_idx]
                w = x2 - x1
                h = y2 - y1
                nx1 = int(bx - self.move_offset[0])
                ny1 = int(by - self.move_offset[1])
                nx2 = nx1 + w
                ny2 = ny1 + h
                # clamp to display bounds
                nx1 = max(0, min(nx1, self.base_display_w - 1))
                ny1 = max(0, min(ny1, self.base_display_h - 1))
                nx2 = max(nx1 + 1, min(nx2, self.base_display_w))
                ny2 = max(ny1 + 1, min(ny2, self.base_display_h))
                self.boxes[self.moving_idx] = (nx1, ny1, nx2, ny2)
                self.redraw_boxes()
            elif self.resizing and self.resizing_idx is not None:
                # resize by moving the selected corner
                bx1, by1, bx2, by2 = self.boxes[self.resizing_idx]
                if self.resizing_corner == 0:
                    # top-left
                    nx1, ny1 = bx, by
                    nx2, ny2 = bx2, by2
                elif self.resizing_corner == 1:
                    # top-right
                    nx1, ny1 = bx1, by
                    nx2, ny2 = bx, by2
                elif self.resizing_corner == 2:
                    # bottom-right
                    nx1, ny1 = bx1, by1
                    nx2, ny2 = bx, by
                else:
                    # bottom-left
                    nx1, ny1 = bx, by1
                    nx2, ny2 = bx2, by

                # normalize
                nx1, nx2 = min(nx1, nx2), max(nx1, nx2)
                ny1, ny2 = min(ny1, ny2), max(ny1, ny2)
                self.boxes[self.resizing_idx] = (int(nx1), int(ny1), int(nx2), int(ny2))
                self.redraw_boxes()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1, y1 = self.start_point
                x2, y2 = bx, by

                # Normalize to top-left, bottom-right (display coords)
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                self.boxes.append((x1, y1, x2, y2))
                self.current_box = None
                self.redraw_boxes()
            elif self.moving:
                self.moving = False
                self.moving_idx = None
                self.move_offset = (0, 0)
                self.redraw_boxes()
            elif self.resizing:
                self.resizing = False
                self.resizing_idx = None
                self.resizing_corner = None
                self.redraw_boxes()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # right-click to delete a box (quick remove)
            for idx, box in enumerate(self.boxes):
                if self._point_in_box(bx, by, box):
                    self.boxes.pop(idx)
                    self.redraw_boxes()
                    break
    
    def image_to_yolo(self, boxes, height, width):
        """Convert pixel coordinates to YOLO normalized format."""
        yolo_boxes = []
        for x1, y1, x2, y2 in boxes:
            x_center = (x1 + x2) / (2 * width)
            y_center = (y1 + y2) / (2 * height)
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            yolo_boxes.append((0, x_center, y_center, box_width, box_height))
        return yolo_boxes

    def load_existing_boxes(self, img_path):
        """Load previously saved YOLO boxes for the current image, if present."""
        label_path = self.data_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            return []

        orig_h, orig_w = self.image_orig.shape[:2]
        if self.scale_x == 0 or self.scale_y == 0:
            sx = sy = 1.0
        else:
            sx = self.scale_x
            sy = self.scale_y

        loaded_boxes = []
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            x_center = float(parts[1]) * orig_w
            y_center = float(parts[2]) * orig_h
            box_width = float(parts[3]) * orig_w
            box_height = float(parts[4]) * orig_h

            ox1 = x_center - box_width / 2
            oy1 = y_center - box_height / 2
            ox2 = x_center + box_width / 2
            oy2 = y_center + box_height / 2

            dx1 = int(round(ox1 / sx))
            dy1 = int(round(oy1 / sy))
            dx2 = int(round(ox2 / sx))
            dy2 = int(round(oy2 / sy))
            loaded_boxes.append((dx1, dy1, dx2, dy2))

        return loaded_boxes

    def _resolve_assist_weights(self):
        if self.assist_weights is not None:
            return self.assist_weights

        project_dir = self.workspace_dir / "project_aditya_dahiya"
        candidates = [
            project_dir / "checkpoints" / "final_weights.pt",
            project_dir / "checkpoints" / "final_weights.pth",
            project_dir / "yolov8n.pt",
            self.workspace_dir / "yolov8n.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path("yolov8n.pt")

    def _init_predictor(self):
        if YOLO is None:
            print("⚠ Model assist disabled: ultralytics is not available in this environment")
            self.assist = False
            return

        chosen_weights = self._resolve_assist_weights()
        try:
            self.predictor = YOLO(str(chosen_weights), task="detect")
            print(f"✓ Model-assist enabled ({chosen_weights})")
            print(f"  Assist thresholds -> conf: {self.assist_conf}, iou: {self.assist_iou}, max_det: {self.assist_max_det}")
        except Exception as exc:
            print(f"⚠ Model assist disabled: failed to load weights {chosen_weights}: {exc}")
            self.assist = False
            self.predictor = None

    def load_assisted_boxes(self, img_path):
        """Predict boxes with model and map them to display coordinates."""
        if not self.assist or self.predictor is None:
            return []
        cache_key = str(img_path)
        if cache_key in self.prediction_cache:
            return [tuple(b) for b in self.prediction_cache[cache_key]]

        try:
            result = self.predictor.predict(
                source=str(img_path),
                conf=self.assist_conf,
                iou=self.assist_iou,
                max_det=self.assist_max_det,
                verbose=False,
            )[0]
        except Exception as exc:
            print(f"⚠ Assist prediction failed for {img_path.name}: {exc}")
            return []

        if result.boxes is None or result.boxes.xyxy is None:
            self.prediction_cache[cache_key] = []
            return []

        orig_h, orig_w = self.image_orig.shape[:2]
        sx = self.scale_x if self.scale_x != 0 else 1.0
        sy = self.scale_y if self.scale_y != 0 else 1.0

        predicted_boxes = []
        for box in result.boxes.xyxy.cpu().numpy().tolist():
            ox1, oy1, ox2, oy2 = box[:4]
            ox1 = max(0.0, min(float(orig_w - 1), ox1))
            ox2 = max(0.0, min(float(orig_w - 1), ox2))
            oy1 = max(0.0, min(float(orig_h - 1), oy1))
            oy2 = max(0.0, min(float(orig_h - 1), oy2))
            if ox2 <= ox1 or oy2 <= oy1:
                continue

            dx1 = int(ox1 / sx)
            dy1 = int(oy1 / sy)
            dx2 = int(ox2 / sx)
            dy2 = int(oy2 / sy)
            predicted_boxes.append((dx1, dy1, dx2, dy2))

        self.prediction_cache[cache_key] = [tuple(b) for b in predicted_boxes]
        return predicted_boxes

    def load_starting_boxes(self, img_path):
        existing_boxes = self.load_existing_boxes(img_path)
        if existing_boxes:
            return existing_boxes, "existing"
        assisted_boxes = self.load_assisted_boxes(img_path)
        if assisted_boxes:
            return assisted_boxes, "assist"
        return [], "none"

    def redraw_boxes(self, extra_box=None, extra_color=(0, 255, 0)):
        self.image = self._render_current_view()
        if self.image is None:
            return
        for i, box in enumerate(self.boxes):
            self._draw_box_on_image(self.image, box, (255, 0, 0))
        if extra_box is not None:
            self._draw_box_on_image(self.image, extra_box, extra_color)
        # On-screen zoom indicator (top-left)
        try:
            text = f"Zoom: {self.zoom:.2f}x"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            pad = 8
            x0, y0 = 8, 8
            # background box for readability
            cv2.rectangle(self.image, (x0 - 4, y0 - 4), (x0 + tw + pad, y0 + th + pad), (0, 0, 0), -1)
            cv2.putText(self.image, text, (x0 + 2, y0 + th + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        except Exception:
            pass

    def _point_in_box(self, x, y, box):
        x1, y1, x2, y2 = box
        return x >= x1 and x <= x2 and y >= y1 and y <= y2

    def _near_corner(self, x, y, box):
        x1, y1, x2, y2 = box
        hs = self.handle_size
        corners = [ (x1, y1), (x2, y1), (x2, y2), (x1, y2) ]
        for idx, (cx, cy) in enumerate(corners):
            if abs(x - cx) <= hs and abs(y - cy) <= hs:
                return idx
        return None

    def _normalized_image_key(self, stem):
        """Normalize an image stem so `_0190` and `_190` compare the same."""
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return f"{parts[0]}_{int(parts[1])}"
        return stem

    def _set_base_display_image(self, image):
        self.base_display_image = image
        self.base_display_h, self.base_display_w = image.shape[:2]
        self.zoom = 1.0
        self.view_origin_x = 0.0
        self.view_origin_y = 0.0

    def _clamp_view_origin(self):
        if self.base_display_w <= 0 or self.base_display_h <= 0:
            return
        crop_w = min(self.base_display_w, max(1, int(round(self.base_display_w / self.zoom))))
        crop_h = min(self.base_display_h, max(1, int(round(self.base_display_h / self.zoom))))
        max_x = max(0, self.base_display_w - crop_w)
        max_y = max(0, self.base_display_h - crop_h)
        self.view_origin_x = max(0.0, min(float(self.view_origin_x), float(max_x)))
        self.view_origin_y = max(0.0, min(float(self.view_origin_y), float(max_y)))

    def _current_view_rect(self):
        crop_w = min(self.base_display_w, max(1, int(round(self.base_display_w / self.zoom))))
        crop_h = min(self.base_display_h, max(1, int(round(self.base_display_h / self.zoom))))
        self._clamp_view_origin()
        x1 = int(round(self.view_origin_x))
        y1 = int(round(self.view_origin_y))
        x1 = max(0, min(x1, self.base_display_w - crop_w))
        y1 = max(0, min(y1, self.base_display_h - crop_h))
        return x1, y1, crop_w, crop_h

    def _window_to_base_point(self, x, y):
        x1, y1, _, _ = self._current_view_rect()
        bx = x1 + (x / self.zoom)
        by = y1 + (y / self.zoom)
        bx = max(0.0, min(bx, float(self.base_display_w - 1)))
        by = max(0.0, min(by, float(self.base_display_h - 1)))
        return bx, by

    def _base_to_window_point(self, bx, by):
        x1, y1, _, _ = self._current_view_rect()
        wx = (bx - x1) * self.zoom
        wy = (by - y1) * self.zoom
        return wx, wy

    def _adjust_zoom(self, factor, anchor_window_point=None):
        if self.base_display_w <= 0 or self.base_display_h <= 0:
            return
        old_zoom = self.zoom
        new_zoom = max(self.zoom_min, min(self.zoom_max, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            return
        if anchor_window_point is None:
            anchor_window_point = (self.base_display_w / 2.0, self.base_display_h / 2.0)
        ax, ay = anchor_window_point
        base_ax = self.view_origin_x + (ax / old_zoom)
        base_ay = self.view_origin_y + (ay / old_zoom)
        self.zoom = new_zoom
        self.view_origin_x = base_ax - (ax / new_zoom)
        self.view_origin_y = base_ay - (ay / new_zoom)
        self._clamp_view_origin()

    def _render_current_view(self):
        if self.base_display_w <= 0 or self.base_display_h <= 0:
            return None
        x1, y1, crop_w, crop_h = self._current_view_rect()
        crop = self.base_display_image[y1:y1 + crop_h, x1:x1 + crop_w]
        if crop.shape[1] != self.base_display_w or crop.shape[0] != self.base_display_h:
            return cv2.resize(crop, (self.base_display_w, self.base_display_h), interpolation=cv2.INTER_LINEAR)
        return crop.copy()

    def _draw_box_on_image(self, image, box, color):
        x1, y1, x2, y2 = box
        wx1, wy1 = self._base_to_window_point(x1, y1)
        wx2, wy2 = self._base_to_window_point(x2, y2)
        wx1, wx2 = int(round(wx1)), int(round(wx2))
        wy1, wy2 = int(round(wy1)), int(round(wy2))
        if wx2 <= 0 or wy2 <= 0 or wx1 >= self.base_display_w or wy1 >= self.base_display_h:
            return
        cv2.rectangle(image, (wx1, wy1), (wx2, wy2), color, 2)
        hs = max(2, self.handle_size)
        cv2.rectangle(image, (wx1 - hs, wy1 - hs), (wx1 + hs, wy1 + hs), (0, 255, 255), -1)
        cv2.rectangle(image, (wx2 - hs, wy1 - hs), (wx2 + hs, wy1 + hs), (0, 255, 255), -1)
        cv2.rectangle(image, (wx2 - hs, wy2 - hs), (wx2 + hs, wy2 + hs), (0, 255, 255), -1)
        cv2.rectangle(image, (wx1 - hs, wy2 - hs), (wx1 + hs, wy2 + hs), (0, 255, 255), -1)
    
    def save_label(self, img_path):
        """Save YOLO label file."""
        # Convert display-coordinate boxes back to original image pixel coords
        orig_h, orig_w = self.image_orig.shape[:2]
        if self.scale_x == 0 or self.scale_y == 0:
            sx = sy = 1.0
        else:
            sx = self.scale_x
            sy = self.scale_y

        orig_boxes = []
        for x1, y1, x2, y2 in self.boxes:
            ox1 = int(round(x1 * sx))
            oy1 = int(round(y1 * sy))
            ox2 = int(round(x2 * sx))
            oy2 = int(round(y2 * sy))
            orig_boxes.append((ox1, oy1, ox2, oy2))

        yolo_boxes = self.image_to_yolo(orig_boxes, orig_h, orig_w)

        label_path = self.data_dir / f"{img_path.stem}.txt"
        with open(label_path, 'w') as f:
            for class_id, x_center, y_center, box_width, box_height in yolo_boxes:
                f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
        
        print(f"✓ Saved {len(yolo_boxes)} label(s) to {label_path}")
        return True
    
    def run(self):
        cv2.namedWindow("Labeler")
        cv2.setMouseCallback("Labeler", self.mouse_callback)
        
        while self.current_idx < len(self.images):
            img_path = self.images[self.current_idx]
            self.image_orig = cv2.imread(str(img_path))
            
            if self.image_orig is None:
                print(f"⚠ Skipped {img_path.name} (cannot read)")
                self.current_idx += 1
                continue
            # Prepare display-sized image and scaling factors so the full image fits the screen
            orig_h, orig_w = self.image_orig.shape[:2]

            base_image = self.image_orig.copy()

            max_w, max_h = self.max_display
            scale = 1.0
            b_h, b_w = base_image.shape[:2]
            if b_w > max_w or b_h > max_h:
                scale = min(max_w / b_w, max_h / b_h)

            disp_w = max(1, int(b_w * scale))
            disp_h = max(1, int(b_h * scale))
            self._set_base_display_image(cv2.resize(base_image, (disp_w, disp_h)) if scale != 1.0 else base_image.copy())
            # scale factors from display -> original coords
            self.scale_x = (b_w) / disp_w
            self.scale_y = (b_h) / disp_h

            self.boxes, source = self.load_starting_boxes(img_path)
            self.current_box = None
            self.redraw_boxes()
            if source == "existing":
                print(f"↺ Loaded {len(self.boxes)} existing box(es)")
            elif source == "assist":
                print(f"🤖 Loaded {len(self.boxes)} model-predicted box(es)")
            
            print(f"\n[{self.current_idx + 1}/{len(self.images)}] {img_path.name}")
            print("Controls: Click+drag=draw box | wheel/=/- zoom | 0 reset | f=fit | s=save | n=skip | b=back | x=discard | u=undo | d=clear | r=reload | p=predict | q=quit")
            
            while True:
                cv2.imshow("Labeler", self.image)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('='), ord('+')):
                    self._adjust_zoom(1.25, self.last_mouse_pos)
                    self.redraw_boxes()
                    continue

                elif key == ord('-'):
                    self._adjust_zoom(1 / 1.25, self.last_mouse_pos)
                    self.redraw_boxes()
                    continue

                elif key == ord('0'):
                    self.zoom = 1.0
                    self.view_origin_x = 0.0
                    self.view_origin_y = 0.0
                    self.redraw_boxes()
                    continue
                elif key == ord('f'):
                    # Fit-to-image: reset view to show entire image
                    self.zoom = 1.0
                    self.view_origin_x = 0.0
                    self.view_origin_y = 0.0
                    print("↺ Fit to image")
                    self.redraw_boxes()
                    continue
                
                if key == ord('s'):
                    self.save_label(img_path)
                    self.current_idx += 1
                    break
                    
                elif key == ord('n'):
                    print("⊘ Skipped")
                    self.current_idx += 1
                    break

                elif key == ord('b'):
                    if self.current_idx > 0:
                        self.current_idx -= 1
                        print("↶ Went back to previous image")
                    else:
                        print("⚠ Already at the first image")
                    break
                    
                elif key == ord('q'):
                    print("\n✓ Done")
                    cv2.destroyAllWindows()
                    return
                    
                elif key == ord('u') and self.boxes:
                    self.boxes.pop()
                    print("↶ Undid last box")
                    self.redraw_boxes()

                elif key == ord('d'):
                    self.boxes = []
                    print("✖ Cleared all boxes")
                    self.redraw_boxes()

                elif key == ord('r'):
                    self.boxes, source = self.load_starting_boxes(img_path)
                    if source == "existing":
                        print("↺ Reloaded boxes from label file")
                    elif source == "assist":
                        print("↺ Reloaded model-predicted boxes")
                    else:
                        print("↺ Reloaded (no boxes found)")
                    self.redraw_boxes()

                elif key == ord('p'):
                    self.boxes = self.load_assisted_boxes(img_path)
                    print(f"🤖 Applied {len(self.boxes)} model-predicted box(es)")
                    self.redraw_boxes()

                elif key == ord('x'):
                    # Move image (and any existing label) to discard folder
                    try:
                        target_img = self.discard_dir / img_path.name
                        shutil.move(str(img_path), str(target_img))
                        label_path = self.data_dir / f"{img_path.stem}.txt"
                        if label_path.exists():
                            shutil.move(str(label_path), str(self.discard_dir / label_path.name))
                        print(f"⊘ Moved {img_path.name} to discard/")
                    except Exception as e:
                        print(f"⚠ Failed to move file: {e}")
                    # advance to next image (images list still contains original Path objects,
                    # so we'll skip increment here and rely on current_idx to move forward)
                    self.current_idx += 1
                    break
        
        cv2.destroyAllWindows()
        print(f"\n✓ Labeled {self.current_idx} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive YOLO label helper with optional model-assisted prelabels")
    parser.add_argument("--data-dir", default=None, help="Directory containing images and labels (default: ./data)")
    parser.add_argument("--assist", action="store_true", help="Enable model-assisted box preloading for unlabeled images")
    parser.add_argument("--all", dest="all", action="store_true", help="Open all images, including already-annotated ones")
    parser.add_argument("--labeled-only", action="store_true", help="Open only images that already have labels (for consistency review)")
    parser.add_argument("--start", dest="start", default=None, help="Start from this image filename or stem (e.g. '2026-05-01 19-48-35_0030.jpg' or '2026-05-01 19-48-35_0030')")
    parser.add_argument("--weights", default=None, help="Path to weights for assist mode (default: project_aditya_dahiya/checkpoints/final_weights.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for assist predictions")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for assist predictions")
    parser.add_argument("--max-det", type=int, default=20, help="Maximum detections per image in assist mode")
    args = parser.parse_args()

    helper = YOLOLabelHelper(
        data_dir=args.data_dir,
        assist=args.assist,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        include_labeled=args.all,
        start_at=args.start,
        labeled_only=args.labeled_only,
    )
    helper.run()
