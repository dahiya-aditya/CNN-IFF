# Automated Target Detection in Simulated Drone Footage

Single-class target detection in simulated drone footage using YOLOv8n. This is the project submission for Aditya Dahiya.

This project implements an automated target detection pipeline using a Convolutional Neural Network (YOLOv8) to identify targets in drone footage. It is specifically built to address the challenges of simulated dynamic environments, detecting human targets across various scales, occlusions, and lighting conditions (including simulated night vision).

## Project Architecture & Note to Graders

This project uses **Ultralytics YOLOv8** for object detection. Because YOLO is an end-to-end detection framework, its training and data loading mechanisms deviate slightly from a standard, generic PyTorch classification pipeline.

As per the general guidelines for non-standard pipelines, please note the following architectural details:

1. **Integrated Training Loop**: YOLO handles the PyTorch `DataLoader`, loss function, and optimizer internally. While `train.py` exposes a `train_model` function that accepts standard arguments (`model`, `num_epochs`, `train_loader`, `loss_fn`, `optimizer`) to remain compatible with automated testing, it intentionally **ignores** custom dataloaders, loss functions, and optimizers. Instead, it relies on the configurations defined in `config.py` and uses YOLO's optimized internal engine.
2. **Dataset Format**: The dataset relies on standard YOLO format (`.txt` label files corresponding to images). The `dataset.py` script automatically materializes the data into a YOLO-friendly directory structure during training.
3. **Weight Files**: Ultralytics YOLO natively expects `.pt` extensions. For strict compliance with the submission format, `train.py` outputs the final weights as `final_weights.pth` inside the `checkpoints/` directory. However, to keep YOLO's internal engine functional, `predict.py` may dynamically create a `.pt` copy of this file at runtime during inference.

## Directory Structure

```text
CNN-IFF/
├── project_aditya_dahiya/          (submission folder - self-contained)
│   ├── checkpoints/
│   │   ├── final_weights.pth       # The final trained model weights
│   │   └── final_weights.pt        # (Optional) Dynamically generated for YOLO compatibility
│   ├── data/                       # Directory containing 10 sample images for testing
│   ├── config.py                   # Hyperparameters and paths
│   ├── dataset.py                  # YOLO dataset materialization and PyTorch Dataset wrapper
│   ├── interface.py                # Standardized API aliases for autograding
│   ├── model.py                    # Wrapper class for the YOLO model
│   ├── predict.py                  # Inference pipeline for images and videos
│   ├── train.py                    # Training pipeline using YOLO's internal loop
│   └── README.md                   # This documentation file
│
├── data/                           (full training dataset, ~1400 labeled images)
├── archive/                        (artifacts, runs, requirements.txt, yolov8n.pt)
├── scripts/                        (demo, labeling, validation tools)
├── raw_videos/                     (source video clips, ignored in git)
├── demo_video/                     (demo footage, ignored in git)
├── README.md                       (root documentation)
└── .gitignore
```
*(Note: Only the contents of the `project_aditya_dahiya/` folder are required for the graded submission)*

## Setup & Dependencies

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
This script will parse the images in the root data directory, split them, generate the required `dataset.yaml` file, and train the YOLOv8 model. The best weights will be saved to `checkpoints/final_weights.pth`.

Train on specific locations (e.g., daylight footage only):
```sh
python train.py --locations 1,2,3,4
```

Include night-vision data:
```sh
python train.py --nvg-on
```

## Inference

Detect targets in an image or video:
```sh
python predict.py path/to/image.jpg
python predict.py path/to/video.mp4
```
Outputs are written to the same directory with a `_predicted` suffix.

## Programmatic Interface (Autograding API)

The `interface.py` file exposes the standardized hooks for automated testing:
```python
from interface import the_trainer, the_predictor, TheModel, the_batch_size, total_epochs

# Initialize the model wrapper
model = TheModel()

# Run training (ignores custom dataloader/loss/optimizer and uses YOLO's engine)
the_trainer(model=model, num_epochs=total_epochs)

# Run inference on a batch of paths
predictions = the_predictor(["data/img1.jpg", "data/img2.jpg"])
```

## Data Format

Training data uses standard YOLO label format. Each image has a corresponding `.txt` file:
```text
image_0001.jpg
image_0001.txt
```
Label file format (one box per line):
```text
class_id x_center y_center width height
```
Coordinates are normalized to [0, 1].

## Performance Notes

- GPU strongly recommended for training (NVIDIA GPU with CUDA 12.1+).
- Falls back to CPU if no GPU detected.
- Inference on 640×640 images takes ~50–100ms on GPU, ~500ms on CPU.
- Adjust `confidence_threshold` in `config.py` to tune detection sensitivity.

## Resources

External assets and additional data are available on Google Drive:
- **[Full Dataset & Archives](https://drive.google.com/drive/folders/1A41A9omBJulgS6Nqfz0nrhhn8We6-rkr?usp=sharing)**: Contains copies of the archive, raw video, and demo video folders.
- **[Demo Inference Video](https://drive.google.com/file/d/137shdLfOfWhAr1sociHx-XJqZ-Bz9AtQ/view?usp=sharing)**: Processed test video showing model predictions on held-out footage.

## Acknowledgment

This project was developed with substantial assistance from **GitHub Copilot**, a generative AI coding assistant. Copilot was used extensively throughout the project for:
- Code generation and refactoring
- Bug identification and fixes
- Documentation writing
- Script development for utilities and validation

All code has been reviewed and tested to ensure correctness and functionality.
