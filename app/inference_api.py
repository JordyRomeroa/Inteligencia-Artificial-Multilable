import os
import torch
from pathlib import Path
from ultralytics import YOLO
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# Configure paths - try multiple locations for best.pt
MODELS_DIR = Path(__file__).parent.parent / 'models'
RUNS_DIR = Path(__file__).parent.parent / 'runs'

best_model_candidates = [
    MODELS_DIR / 'best_improved.pt',  # IMPROVED MODEL (YOLOv8s)
    MODELS_DIR / 'best.pt',  # Copied location (fallback)
    RUNS_DIR / 'detect' / 'yolo_run' / 'weights' / 'best.pt',  # Default YOLO location
    MODELS_DIR / 'yolo_run' / 'weights' / 'best.pt',  # Old location
]

BEST_MODEL_PATH = None
for candidate in best_model_candidates:
    if candidate.exists():
        BEST_MODEL_PATH = candidate
        break

model = None
CLASS_NAMES = ['person', 'car', 'dog']

def load_model():
    global model
    if BEST_MODEL_PATH and BEST_MODEL_PATH.exists():
        model = YOLO(str(BEST_MODEL_PATH))
        print(f"Model loaded from: {BEST_MODEL_PATH}")
        return True
    else:
        print("ERROR: Model not found in any expected location")
        for candidate in best_model_candidates:
            print(f"  - Checked: {candidate}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content - prevents 404 error

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'model_path': str(BEST_MODEL_PATH) if BEST_MODEL_PATH else None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run prediction with optimized parameters for multi-object detection
        # Aggressive settings to detect all 3 classes (person, car, dog)
        # dog is hardest to detect so we need lower thresholds
        results = model.predict(
            source=image, 
            conf=0.05,      # Very low - catches weak detections (especially dogs)
            iou=0.20,       # Very low NMS threshold - keeps overlapping detections
            max_det=200,    # Increased - allows many detections per image
            agnostic_nms=False,  # Keeps detections of different classes even if overlapped
            verbose=False
        )
        result = results[0]
        
        # Extract detections
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                detections.append({
                    'class': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else 'unknown',
                    'class_id': class_id,
                    'confidence': round(confidence, 3),
                    'bbox': {
                        'x1': round(bbox[0], 1),
                        'y1': round(bbox[1], 1),
                        'x2': round(bbox[2], 1),
                        'y2': round(bbox[3], 1)
                    }
                })
        
        # Generate annotated image
        annotated_img = result.plot()
        annotated_pil = Image.fromarray(annotated_img[..., ::-1])  # BGR to RGB
        
        # Convert to base64
        buffered = io.BytesIO()
        annotated_pil.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'detections': detections,
            'num_detections': len(detections),
            'image_size': {'width': image.size[0], 'height': image.size[1]},
            'annotated_image': f'data:image/jpeg;base64,{img_base64}'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model': 'YOLOv8n',
        'classes': CLASS_NAMES,
        'num_classes': len(CLASS_NAMES),
        'model_loaded': model is not None,
        'model_path': str(BEST_MODEL_PATH) if BEST_MODEL_PATH else 'not found'
    }), 200

if __name__ == '__main__':
    print("=" * 60)
    print("YOLO Object Detection API")
    print("=" * 60)
    
    if load_model():
        print(f"\nModel loaded successfully!")
        print(f"Path: {BEST_MODEL_PATH}")
        print(f"Classes: {CLASS_NAMES}")
        print(f"\nAPI Endpoints:")
        print(f"  GET  /           - Web interface")
        print(f"  GET  /health     - Check API health")
        print(f"  GET  /model-info - Model information")
        print(f"  POST /predict    - Predict objects in image")
        print(f"\nStarting server on http://localhost:5000")
        print("=" * 60)
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("\nFailed to load model. Exiting.")
        print("Run notebook 03_training.ipynb to train the model first.")
