# project_aditya_dahiya

Single-class target detection for simulated drone footage using a YOLOv8 detector.

This folder is the required submission directory and contains the exact files and structure expected by the grader.

## Directory Structure (Required)

```
project_aditya_dahiya/
	checkpoints/
		final_weights.pth
	data/
		img01.jpg
		img02.jpg
		... (10 sample images, no label files)
	config.py
	dataset.py
	interface.py
	model.py
	predict.py
	train.py
```

Notes:
- `data/` contains 10 raw sample images only (no `.txt` labels) to satisfy the submission requirement.
- Full training data is stored outside this folder in the main workspace `data/` directory.

## Dependencies

This project uses:
- Python 3.10+ recommended
- `ultralytics` (YOLOv8)
- `torch`, `torchvision`
- `opencv-python`
- `pyyaml`

Install (example):

```sh
pip install ultralytics torch torchvision opencv-python pyyaml
```

## Configuration

All hyperparameters live in `config.py`:
- `batchsize`, `epochs`
- `resize_x`, `resize_y`, `input_channels`
- `learning_rate`, `weight_decay`, `patience`, `workers`, `seed`
- `confidence_threshold`, `iou_threshold`
- `pretrained_weights`, `final_weights_name`, `run_name`

The dataset and training pipeline import these values directly from `config.py`.

## Interface Contract

`interface.py` exposes the standardized entry points used by the grading script:
- `TheModel`
- `the_trainer`
- `the_predictor`
- `TheDataset`
- `the_dataloader`
- `the_batch_size`
- `total_epochs`

## Training

Train on the full dataset (outside this folder):

```sh
python train.py
```

Train with location filtering (example):

```sh
python train.py --locations 1,2,3,4
```

Allow night-vision location 5 (nv):

```sh
python train.py --locations 1,2,3,4,5 --nvg-on
```

Outputs:
- Best weights are copied to `checkpoints/final_weights.pt`.
- A `.pth` copy is included as `checkpoints/final_weights.pth` for submission.

## Inference

Run on a single image or video:

```sh
python predict.py path\to\image_or_video.mp4
```

Programmatic call:

```sh
python -c "from predict import the_predictor; the_predictor(source='path/to/image.jpg', weights='checkpoints/final_weights.pth', conf=0.25)"
```

The predictor accepts either a file path or a list of paths. For videos, the output file is written next to the input with a `_predicted` suffix.

## Data Expectations

Training expects YOLO label files with the same base name as each image:

```
image_0001.jpg
image_0001.txt
```

Each label line is in YOLO format:

```
class_id x_center y_center width height
```

## Troubleshooting

- If no GPU is detected, training falls back to CPU.
- If inference outputs no boxes, lower `confidence_threshold` in `config.py`.
- If training fails to find data, check that the full dataset exists in the workspace `data/` directory.