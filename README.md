# YOLO 3-Class Object Detection with MLflow

YOLOv8 detection system for **Person, Car, Dog** with complete MLflow experiment tracking and model registry.

## Stack

- **YOLOv8** (Ultralytics) - Transfer learning from COCO
- **MLflow** - Experiment tracking + Model Registry
- **Pascal VOC 2012** - Dataset (person, car, dog classes)
- **Flask** - REST API inference
- **PyTorch** (CUDA 12.1)

## Structure

```
iajordy2/
├── notebooks/              # Complete ML pipeline
│   ├── 01_dataset_validation.ipynb
│   ├── 02_train_yolo.ipynb
│   ├── 03_training.ipynb         # Main training + MLflow
│   ├── 04_prediction.ipynb
│   └── 05_retrain_improved.ipynb # Optimized (dog focus)
├── app/                    # Flask API
│   ├── mlflow_utils.py           # MLflowYOLOTracker
│   ├── api.py / inference_api.py
│   ├── run_server.py             # Start API server
│   └── templates/ static/
├── data/
│   ├── VOCdevkit/                # Original dataset
│   ├── data.yaml                 # YOLO config
│   ├── images/                   # train/val/test
│   └── labels/                   # YOLO format
├── models/                 # Trained models (.pt)
├── runs/mlflow/            # MLflow backend
└── requirements.txt
```

## Setup

**Requirements**: Python 3.9+, CUDA 12.1, 8GB RAM

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python prepare_dataset.py
```

## Training

**Standard Training (Notebook 03)**
```bash
# Run cells: Cell 2 → Cell 4 → Cell 6 → Cell 8
# Output: models/best.pt
# MLflow experiment: yolo_3class_detection
```

**Optimized Training (Notebook 05)**  
```bash
# Run cells: Cell 2 → Cell 5 → Cell 16 → Cell 17 → Cell 19
# Output: models/best_improved.pt + MLflow Model Registry
# Optimizations: copy-paste aug, mixup, NMS tuning for dog groups
# MLflow experiment: yolo_improved_training
```

### Model Path Convention

All notebooks use **single-path convention**:

- **Base model**: `models/best.pt` (notebooks 02, 03, 04)
- **Improved model**: `models/best_improved.pt` (notebook 05)
- **MLflow artifacts**: logged to `artifact_path='model'` (singular)

## MLflow

**Start UI**
```bash
python -m mlflow ui --backend-store-uri "file:///C:/Users/jordy/OneDrive/Desktop/iaaaa/iajordy2/runs/mlflow" --port 5001
```
→ http://localhost:5001

**Tracked Metrics**: mAP50, mAP50-95, precision, recall, per-class AP  
**Artifacts**: model, training curves, confusion matrix, predictions  
**Model Registry**: Registered models via `mlflow.register_model()`

**MLflowYOLOTracker** (app/mlflow_utils.py)  
- `setup_mlflow()` - Initialize backend + experiment  
- `log_training_params()` - Log YOLO config  
- `log_metrics_from_yolo()` - Parse results.csv  
- `log_training_artifacts()` - Log model to artifact_path='model'

## API Endpoints

**Start server** (after training model)
```bash
cd app
python run_server.py
```
→ http://localhost:5000

**POST /predict** - Upload image, get detections  
**POST /api/model/retrain** - Trigger manual retraining  
**GET /api/model/info** - Current model version + metrics

## Training Config

**Standard (Notebook 03)**
- YOLOv8n, 50 epochs, batch=16, img=640
- Optimizer: AdamW (lr=0.001)
- Augmentation: hsv, degrees, translate, scale

**Optimized (Notebook 05)**  
- YOLOv8s, 100 epochs, batch=8, img=640
- Optimizer: AdamW (lr=0.001)
- Advanced aug: copy-paste=0.3, mixup=0.15
- NMS: IoU=0.5, conf=0.15
- Target: Dog AP50 → 0.65+ (from 0.468)

## Expected Performance

- **Overall mAP50**: 0.70+
- **Person AP50**: 0.85+
- **Car AP50**: 0.75+
- **Dog AP50**: 0.65+ (optimized for multi-object groups)

## Workflow

1. **Dataset validation**: Run [01_dataset_validation.ipynb](notebooks/01_dataset_validation.ipynb)
2. **Initial training**: Execute [03_training.ipynb](notebooks/03_training.ipynb)
3. **Optimization**: Use [05_retrain_improved.ipynb](notebooks/05_retrain_improved.ipynb)
4. **Deployment**: Start API with `run_server.py`
5. **Monitoring**: View MLflow UI for experiment comparison

## Troubleshooting

**CUDA OOM**: Reduce batch_size or use YOLOv8n  
**MLflow UI fails**: Check backend path `runs/mlflow/` exists  
**Model not found**: Verify `models/best.pt` after training  
**Low accuracy**: Increase epochs (100+), enable copy-paste/mixup

## Technical Notes

- YOLO outputs: `runs/train/` (training), `runs/detect/` (validation)
- MLflow backend: File-based store at `runs/mlflow/`
- Model Registry: All models registered with version control
- Deterministic mode enabled for reproducibility
- Auto GPU detection if CUDA available

## Requirements

- **OS**: Windows 10/11, Linux, macOS
- **GPU**: NVIDIA 4GB+ VRAM (RTX 3080 recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for dataset + models
- **Python**: 3.9+

---

**License**: MIT
