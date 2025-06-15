# PPE Detector

This folder contains a small project experimenting with a YOLOv8 model for detecting personal protective equipment on construction workers.
Here is the Yolo Review.

![YOLO Diagram](PPE_Detection/Resources/yolo_diagram.png)

## Directory overview

- `Modelling.ipynb` – notebook used to download the dataset from Roboflow, explore class distribution and train the model.
- `Model_Test.ipynb` – shows how to run inference on a short video clip.
- `scripts/` – utility functions for data augmentation, visualization and comparing training runs.
- `full_model_2/` – saved model checkpoints and training logs.
- `runs/` – YOLOv8 detection results.
- `Resources/` – background notes including a brief review of the YOLOv8 architecture.

## Methods

- The dataset of workers with and without safety gear is downloaded from Roboflow and split into training and validation sets.
- Albumentations-based augmentations (flip, rotation, brightness/contrast) help increase variety and address class imbalance.
- YOLOv8 is trained using Ultralytics' training pipeline, saving checkpoints and logs under `full_model_2/`.
- Evaluation is performed using mean Average Precision on the validation set, and predictions can be visualized in `runs/`.

  ## Result

[Watch the PPE Video Detector demo ▶️](https://github.com/Milltrader/NeuralNetworkProjects/blob/main/PPE_Detection/runs/detect/predict/Construction%20Site%201%20-%20Stock%20Footage%20Collection%20%5BCtLy8PXiL58%5D.avi)


## Quick start

1. Install the required Python packages:

```bash
pip install ultralytics opencv-python pandas seaborn matplotlib albumentations
```

2. Open `Modelling.ipynb` and execute the cells to download the dataset and train the model.
3. Run `Model_Test.ipynb` or use the YOLO command line interface to test the detector on images or video.

## Disclaimer

This is a study project created for learning purposes. The dataset and model may not generalize well to real-world scenarios.
