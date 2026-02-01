import os
import torch
from pathlib import Path
from ultralytics import YOLO
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64

app = Flask(__name__)

MODELS_DIR = Path(__file__).parent.parent / 'models'
BEST_MODEL_PATH = MODELS_DIR / 'yolo_run' / 'weights' / 'best.pt'

model = None

def load_model():
    global model
    if BEST_MODEL_PATH.exists():
        model = YOLO(str(BEST_MODEL_PATH))
        return True
    return False

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    
    results = model.predict(source=image, conf=0.5, verbose=False)
    result = results[0]
    
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().tolist()
            
            class_names = {0: 'person', 1: 'car', 2: 'dog'}
            
            detections.append({
                'class': class_names.get(class_id, 'unknown'),
                'confidence': confidence,
                'bbox': bbox
            })
    
    return jsonify({
        'detections': detections,
        'image_size': image.size,
        'num_detections': len(detections)
    }), 200

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model': 'YOLOv8n',
        'classes': ['person', 'car', 'dog'],
        'num_classes': 3,
        'model_loaded': model is not None,
        'model_path': str(BEST_MODEL_PATH) if BEST_MODEL_PATH.exists() else 'not found'
    }), 200

if __name__ == '__main__':
    print("Loading YOLO model for inference...")
    if load_model():
        print("Model loaded successfully")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Error: Could not load model")
        print(f"Expected path: {BEST_MODEL_PATH}")
