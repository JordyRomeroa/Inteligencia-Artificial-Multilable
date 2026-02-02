# YOLO Object Detection – 3-Class Production Pipeline

## 📌 Overview

This project implements a **production-ready object detection system** using **YOLOv8**, focused on detecting **three specific classes**:

- 🧍 Person  
- 🚗 Car  
- 🐶 Dog  

The pipeline covers the **entire machine learning lifecycle**, from dataset validation and exploratory data analysis (EDA) to model training, experiment tracking with **MLflow**, and deployment through a **REST inference API**.

The system is designed with **reproducibility, scalability, and production-readiness** in mind.

---

## 🎯 Objective

Build a robust YOLO-based object detection pipeline that:

- Detects 3 predefined object classes
- Uses transfer learning from YOLOv8 pre-trained weights
- Tracks experiments and models using MLflow
- Exposes the trained model via a REST API
- Ensures reproducibility across runs

---

## 🚀 Key Features

- YOLOv8 object detection (Ultralytics)
- 3-class specialization (person, car, dog)
- Transfer learning from COCO-pretrained YOLOv8n
- Integrated MLflow experiment tracking
- Model Registry with production-ready versions
- REST API for inference (Flask)
- Deterministic training (fixed random seed)

---

## 🧠 Technical Stack

- Ultralytics YOLOv8  
- PyTorch  
- MLflow 2.0+ (SQLite backend)  
- Flask (Inference API)  
- Python 3.9+  

---

## 📊 Dataset Description

- **Format:** YOLO bounding box format (`.txt`)
- **Classes:** 3 (`person`, `car`, `dog`)
- **Image Size:** 416 × 416 (RGB)
- **Total Images:** 500
- **Split:**
  - Train: 400 images (80%)
  - Validation: 50 images (10%)
  - Test: 50 images (10%)

The dataset follows the standard YOLO directory structure.

---

## 🔍 Exploratory Data Analysis (EDA)

Before training, the dataset is validated and analyzed to ensure data quality and consistency.

### EDA includes:
- Image–label alignment verification
- Bounding box format and normalization checks
- Class distribution analysis
- Detection of missing or corrupted labels
- Visualization of sample images with bounding boxes
- Dataset balance validation

This step ensures that training performance is not affected by data-related issues.

---

## 📓 Notebooks Explanation

### `01_dataset_validation.ipynb`
**Purpose:** Dataset sanity checks & EDA

This notebook performs:
- Dataset structure validation
- YOLO label format verification
- Bounding box normalization checks
- Class frequency analysis
- Visualization of labeled samples
- Detection of empty or invalid annotations

**Outcome:** A clean and verified dataset ready for training.

---

### `02_train_yolo.ipynb`
**Purpose:** Model training & experiment tracking

This notebook handles:
- YOLOv8n model initialization
- Transfer learning from COCO pre-trained weights
- Training configuration (epochs, batch size, image size)
- Automatic logging of metrics, parameters, and artifacts with MLflow
- Model registration in the MLflow Model Registry

**Outcome:** A trained and versioned object detection model.

---

### `03_inference.ipynb`
**Purpose:** Model inference & qualitative evaluation

This notebook:
- Loads the trained YOLOv8 model
- Runs inference on validation and test images
- Applies confidence and IoU thresholds
- Visualizes predictions vs ground truth
- Evaluates real-world detection behavior

**Outcome:** Validation of model performance before deployment.

---

##  Installation

```bash
pip install -r requirements.txt
```

Requirements

    Python 3.9+

    CUDA 11.8+ (optional, GPU acceleration)

    8 GB RAM (minimum)

    10 GB free storage

 ## Training Pipeline

    Prepare Dataset

python prepare_dataset.py

    Dataset Validation & EDA

jupyter notebook notebooks/01_dataset_validation.ipynb

    Model Training

jupyter notebook notebooks/02_train_yolo.ipynb

    Inference Testing

jupyter notebook notebooks/03_inference.ipynb

 MLflow Tracking

Launch the MLflow UI:

mlflow ui --backend-store-uri file:///mlruns

## MLflow Setup

    Backend: SQLite

    Experiment: yolo_3class_detection

    Registered Model: yolo_3class_detector

    Stages: Staging → Production

 Inference API

Start the API:

python app/inference_api.py

API Endpoints
Method	Endpoint	Description
GET	/health	API health check
GET	/model-info	Loaded model metadata
POST	/predict	Image upload for object detection
Test API

python test_inference_api.py

## Model Information

    Architecture: YOLOv8n

    Input Size: 416 × 416

    Classes: person, car, dog

    Epochs: 50

    Batch Size: 16

    Loss Function: YOLOv8 default

    Optimizer & LR: YOLOv8 auto scheduler

    Random Seed: 42

## Metrics

    mAP@0.5

    mAP@0.5:0.95

    Precision

    Recall

## Evaluation Setup

    Validation Set: 50 images

    Test Set: 50 images

    Confidence Threshold: 0.5

    IoU Threshold: 0.45

## Project Structure

.
├── data/
│   ├── data.yaml
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
├── notebooks/
│   ├── 01_dataset_validation.ipynb
│   ├── 02_train_yolo.ipynb
│   └── 03_inference.ipynb
├── models/
│   └── yolo_run/
│       └── weights/
│           └── best.pt
├── app/
│   └── inference_api.py
├── prepare_dataset.py
├── test_inference_api.py
└── README.md

##  Reproducibility

    Fixed random seed (SEED = 42)

    Deterministic dataset split (80/10/10)

    YOLOv8 deterministic behavior

    Full experiment tracking with MLflow

##  Production Deployment Notes

    API Framework: Flask (WSGI)

    Default Port: 5000

    Model Format: PyTorch .pt

    Inference Time:

        GPU: ~50–100 ms per image

        CPU: ~200–500 ms per image

##  Project Status

✔ Dataset validated
✔ Model trained and evaluated
✔ Experiments tracked
✔ Model registered
✔ API deployed