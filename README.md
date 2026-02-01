YOLO Object Detection - 3 Class Production Pipeline

Objective
Production-ready YOLO object detection system with 3 classes (person, car, dog). Implements YOLOv8 architecture with MLflow tracking and inference API.

Features
- YOLO object detection on 3 specialized classes
- YOLOv8n backbone architecture
- Single-stage training with MLflow integration
- Model Registry for version control
- REST API for inference
- Reproducible training (seed 42)

Technical Stack
- Ultralytics YOLOv8
- PyTorch
- MLflow 2.0+ (SQLite backend)
- Flask (inference API)
- Python 3.9+

Dataset
- Format: YOLO bounding box format
- Classes: 3 (person, car, dog)
- Images: 416x416 RGB
- Split: train/val/test
- Total: 500 images

Prerequisites
- Python 3.9+
- CUDA 11.8+ (optional, for GPU)
- 8GB RAM
- 10GB storage

Installation
pip install -r requirements.txt

Training Pipeline

1. Data Preparation:
python prepare_dataset.py

2. Dataset Validation:
jupyter notebook notebooks/01_dataset_validation.ipynb

3. Model Training:
jupyter notebook notebooks/02_train_yolo.ipynb

4. Inference Testing:
jupyter notebook notebooks/03_inference.ipynb

MLflow Tracking UI
mlflow ui --backend-store-uri file:///mlruns

Inference API
python app/inference_api.py

Test API:
python test_inference_api.py

API Endpoints
GET /health
GET /model-info
POST /predict (multipart image upload)

Model Information
- Model: YOLOv8n
- Classes: person, car, dog
- Input Size: 416x416
- Loss Function: YOLOv8 default (objectness + classification + localization)
- Training Epochs: 50
- Batch Size: 16
- Learning Rate: auto (YOLOv8 default scheduler)
- Reproducibility: SEED=42

Training Details

Phase 1: Transfer Learning
- Backbone: YOLOv8n pre-trained on COCO
- Classes: 3 (person, car, dog)
- Epochs: 50
- Batch Size: 16
- Device: Auto (GPU if available, CPU fallback)
- Early Stopping: patience=10

Metrics
- mAP50 (mean Average Precision at IoU=0.5)
- mAP50-95 (mean Average Precision at IoU=0.5:0.95)
- Precision
- Recall

Evaluation
- Validation set: 50 images
- Test set: 50 images
- Confidence threshold: 0.5
- IoU threshold: 0.45

File Structure
.
├── data/
│   ├── data.yaml               (YOLO config: 3 classes)
│   ├── images/
│   │   ├── train/              (400 images)
│   │   ├── val/                (50 images)
│   │   └── test/               (50 images)
│   └── labels/                 (YOLO txt format)
├── notebooks/
│   ├── 01_dataset_validation.ipynb
│   ├── 02_train_yolo.ipynb
│   └── 03_inference.ipynb
├── models/
│   └── yolo_run/
│       └── weights/
│           └── best.pt         (trained model)
├── app/
│   └── inference_api.py        (Flask API)
├── prepare_dataset.py
├── test_inference_api.py
└── README.md

Reproducibility
- Random Seed: 42 (NumPy, PyTorch, CUDA)
- Dataset Split: 80/10/10 fixed
- Model: YOLOv8n (deterministic)
- Framework: Ultralytics YOLOv8

MLflow Integration
- Backend: SQLite (file://mlruns)
- Experiment: yolo_3class_detection
- Runs: training, validation, model_registration
- Model Registry: yolo_3class_detector (Production stage)

Production Deployment
- API: Flask WSGI server
- Port: 5000 (default)
- Concurrent Requests: Limited by model inference time
- Model Format: YOLO .pt (PyTorch)
- Inference Time: ~50-100ms per image (GPU), ~200-500ms (CPU)
