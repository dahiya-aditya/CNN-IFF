# project_aditya_dahiya

Single-class target detection in simulated drone footage using YOLOv8n.

## Overview

This project implements an object detection pipeline for identifying targets in video frames. The model is trained on location-specific drone footage and supports inference on both images and video streams.

## Project Structure

```
CNN-IFF/
├── project_aditya_dahiya/          (submission folder)
│   ├── checkpoints/
│   │   ├── final_weights.pt
│   │   └── final_weights.pth
│   ├── data/                       (10 sample images)
│   ├── config.py
│   ├── dataset.py
│   ├── interface.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
│
├── data/                           (full training dataset, ~1400 labeled images)
│
├── archive/
│   ├── artifacts/
│   ├── runs/
│   ├── scripts/
│   ├── requirements.txt
│   └── yolov8n.pt
│
├── scripts/                        (demo, labeling, validation tools)
├── raw_videos/                     (source video clips, ignored in git)
├── demo_video/                     (demo footage, ignored in git)
├── README.md                       (this file)
├── LICENSE
└── .gitignore
```

The submission folder `project_aditya_dahiya/` is self-contained. The `data/` directory in the root contains the full training dataset. Large outputs and intermediate files are archived in `archive/` or ignored by git.

## Setup

Requirements:
- Python 3.10+
- PyTorch with CUDA support recommended

Install dependencies:

```sh
pip install ultralytics torch torchvision opencv-python pyyaml
```

## Configuration

All training and model parameters are centralized in `config.py`:

- Model: YOLOv8n (pretrained on COCO)
- Image size: 640×640
- Batch size: 16
- Epochs: 30
- Confidence threshold: 0.25
- IOU threshold: 0.45

Modify these values before training to adjust model behavior.

## Training

Train the model on the full dataset:

```sh
python train.py
```

Train on specific locations (e.g., daylight footage only):

```sh
python train.py --locations 1,2,3,4
```

Include night-vision data:

```sh
python train.py --nvg-on
```

Best weights are saved to `checkpoints/final_weights.pt`.

## Inference

Detect targets in an image or video:

```sh
python predict.py path/to/image.jpg
python predict.py path/to/video.mp4
```

Outputs are written to the same directory with a `_predicted` suffix.

Programmatic use:

```python
from predict import the_predictor

results = the_predictor(
    source='image.jpg',
    weights='checkpoints/final_weights.pth',
    conf=0.25
)
```

## Data Format

Training data uses YOLO label format. Each image has a corresponding `.txt` file:

```
image_0001.jpg
image_0001.txt
```

Label file format (one box per line):

```
class_id x_center y_center width height
```

Coordinates are normalized to [0, 1].

## API

The module exports standardized interfaces for training and inference:

- `TargetDetector` (model wrapper)
- `the_trainer()` (training entry point)
- `the_predictor()` (inference entry point)
- `TargetDataset` (dataset loader)

See `interface.py` for the complete contract.

## Resources

External assets and additional data are available on Google Drive:

- **[Full Dataset & Archives](https://drive.google.com/drive/folders/1A41A9omBJulgS6Nqfz0nrhhn8We6-rkr?usp=sharing)**: Contains copies of the archive, raw video, and demo video folders.
- **[Demo Inference Video](https://drive.google.com/file/d/137shdLfOfWhAr1sociHx-XJqZ-Bz9AtQ/view?usp=sharing)**: Processed test video showing model predictions on held-out footage.

## Performance Notes

- GPU strongly recommended for training (NVIDIA GPU with CUDA 12.1+).
- Falls back to CPU if no GPU detected.
- Inference on 640×640 images takes ~50–100ms on GPU, ~500ms on CPU.
- Adjust `confidence_threshold` in `config.py` to tune detection sensitivity.