from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.parent / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
RUNS_DIR = ROOT_DIR / "runs"

batchsize = 16
epochs = 30
resize_x = 640
resize_y = 640
input_channels = 3

learning_rate = 0.001
weight_decay = 0.0005
patience = 10
workers = 4
seed = 42

train_split = 0.8
val_split = 0.2
test_split = 0.0

class_names = ["Target"]
single_class = True

confidence_threshold = 0.25
iou_threshold = 0.45

pretrained_weights = "yolov8n.pt"
final_weights_name = "final_weights.pth"
data_yaml_name = "dataset.yaml"
run_name = "target_detector"