import os
import torch
from pathlib import Path
from ultralytics import YOLO
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import base64
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# ===== CONFIGURACIÓN DE GPU =====
# Verificar disponibilidad de GPU
if torch.cuda.is_available():
    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✓ GPU DETECTADA: {gpu_name}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  Dispositivos GPU: {torch.cuda.device_count()}")
    # Optimizar uso de GPU
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'
    print("⚠ GPU NO DETECTADA - usando CPU")
    print("  Instala PyTorch con CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
# ================================

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
learner = None  # Global instance for continuous learning
CLASS_NAMES = ['person', 'car', 'dog']

def load_model():
    global model, learner
    if BEST_MODEL_PATH and BEST_MODEL_PATH.exists():
        model = YOLO(str(BEST_MODEL_PATH))
        # Mover modelo a GPU si está disponible
        if device == 'cuda':
            model.to(device)
            print(f"✓ Modelo cargado en GPU: {BEST_MODEL_PATH}")
        else:
            print(f"Modelo cargado en CPU: {BEST_MODEL_PATH}")
        
        # Inicializar el learner global
        try:
            from continuous_learning import ContinuousLearner
            learner = ContinuousLearner(str(BEST_MODEL_PATH))
            print(f"✓ ContinuousLearner inicializado")
        except Exception as e:
            print(f"⚠ Error inicializando ContinuousLearner: {e}")
            learner = None
        
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
        'model_path': str(BEST_MODEL_PATH) if BEST_MODEL_PATH else None,
        'device': device,
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
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
        # IMPORTANTE: device especifica donde ejecutar (GPU si está disponible)
        results = model.predict(
            source=image, 
            conf=0.05,      # Very low - catches weak detections (especially dogs)
            iou=0.20,       # Very low NMS threshold - keeps overlapping detections
            max_det=200,    # Increased - allows many detections per image
            agnostic_nms=False,  # Keeps detections of different classes even if overlapped
            device=device,  # USA GPU SI ESTÁ DISPONIBLE
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
            'image_with_detections': img_base64
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
        'model_path': str(BEST_MODEL_PATH) if BEST_MODEL_PATH else 'not found',
        'device': device,
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'
    }), 200

# ===== NEW ENDPOINTS FOR ADVANCED UI =====

@app.route('/api/models/list', methods=['GET'])
def list_models():
    """List all available model versions"""
    import json
    from datetime import datetime
    
    models = []
    
    # Buscar todos los modelos disponibles
    for model_path in MODELS_DIR.glob('*.pt'):
        try:
            metrics = {
                'accuracy': 0.85,  # Placeholder
                'precision': 0.82,
                'recall': 0.88
            }
            
            models.append({
                'name': model_path.name,
                'path': str(model_path),
                'type': 'improved' if 'improved' in model_path.name else 'retrained' if 'retrained' in model_path.name else 'base',
                'version': int(model_path.stem.split('_')[-1]) if '_' in model_path.stem else 1,
                'is_current': str(model_path) == str(BEST_MODEL_PATH),
                'metrics': metrics,
                'size_mb': round(model_path.stat().st_size / 1024 / 1024, 2)
            })
        except:
            pass
    
    models.sort(key=lambda x: x['version'], reverse=True)
    
    return jsonify({
        'success': True,
        'models': models,
        'device': device
    }), 200

@app.route('/api/models/load', methods=['POST'])
def load_model_endpoint():
    """Load a specific model version"""
    global model, BEST_MODEL_PATH
    
    try:
        data = request.json
        model_path = Path(data.get('model_path', ''))
        
        if not model_path.exists():
            return jsonify({'success': False, 'error': 'Model not found'}), 404
        
        model = YOLO(str(model_path))
        if device == 'cuda':
            model.to(device)
        
        BEST_MODEL_PATH = model_path
        
        return jsonify({
            'success': True,
            'model': {
                'path': str(model_path),
                'version': int(model_path.stem.split('_')[-1]) if '_' in model_path.stem else 1,
                'device': device,
                'loaded_at': datetime.now().isoformat()
            }
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/corrections/add', methods=['POST'])
def add_correction():
    """Add a correction for retraining"""
    try:
        if learner is None:
            return jsonify({'success': False, 'error': 'Learner not initialized'}), 500
        
        image_file = request.files.get('image')
        bbox = request.form.get('bbox')
        tag = request.form.get('tag')
        
        if not all([image_file, bbox, tag]):
            return jsonify({'success': False, 'error': 'Missing data'}), 400
        
        # Save temporary image
        temp_dir = Path(__file__).parent.parent / 'data' / 'corrections' / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        bbox_data = json.loads(bbox)
        temp_image_path = temp_dir / f"correction_{datetime.now().timestamp()}{Path(image_file.filename).suffix}"
        image_file.save(str(temp_image_path))
        
        boxes = [{
            'class': tag,
            'bbox': [bbox_data['x1'], bbox_data['y1'], bbox_data['x2'], bbox_data['y2']],
            'confidence': 1.0  # User correction = 100% confidence
        }]
        
        success = learner.add_corrected_sample(
            str(temp_image_path),
            boxes,
            user_id='web_user'
        )
        
        return jsonify({
            'success': success,
            'message': 'Correction saved successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/corrections/stats', methods=['GET'])
def get_correction_stats():
    """Get statistics about corrections"""
    try:
        if learner is None:
            return jsonify({
                'success': True,
                'total': 0,
                'ready_for_retrain': False,
                'min_required': 5
            }), 200
        
        stats = learner.get_stats()
        
        return jsonify({
            'success': True,
            'total': stats.get('pending_corrections', 0),
            'ready_for_retrain': stats.get('pending_retraining', False),
            'min_required': 5
        }), 200
    except:
        return jsonify({
            'success': True,
            'total': 0,
            'ready_for_retrain': False,
            'min_required': 5
        }), 200

@app.route('/api/model/retrain', methods=['POST'])
def retrain_model():
    """
    Trigger model retraining with accumulated corrections - FLUJO MLFLOW OBLIGATORIO.
    
    ⚠️ REENTRENAMIENTO CON MLFLOW CORRECTO:
       - Usar experiment_id: 401576597529460193
       - Usar artifact_location correcto
       - Guardar dataset como artifacts
       - Registrar métricas PRE/POST
       - Agregar tag "retraining"
    
    Payload JSON:
    {
        "epochs": 5,
        "batch_size": 16,
        "experiment_id": "401576597529460193"  # OBLIGATORIO
    }
    """
    try:
        data = request.json or {}
        epochs = data.get('epochs', 5)
        batch_size = data.get('batch_size', 16)
        # REQUERIMIENTO OBLIGATORIO: experiment_id debe ser 401576597529460193
        experiment_id = data.get('experiment_id', '401576597529460193')
        
        if learner is None:
            return jsonify({'success': False, 'error': 'Learner not initialized'}), 500
        
        print(f"\n" + "="*80)
        print(f"[API] SOLICITUD DE REENTRENAMIENTO RECIBIDA")
        print(f"[API] Experiment ID: {experiment_id}")
        print(f"[API] Epochs: {epochs}")
        print(f"[API] Batch Size: {batch_size}")
        print(f"[API] Correcciones guardadas: {len(learner.corrected_samples)}")
        print(f"="*80)
        
        if len(learner.corrected_samples) < 5:
            return jsonify({
                'success': False,
                'error': f'Insuficientes correcciones ({len(learner.corrected_samples)} < 5)',
                'corrections_needed': 5 - len(learner.corrected_samples)
            }), 400
        
        # LLAMAR CON EXPERIMENT_ID OBLIGATORIO
        result = learner.retrain(
            epochs=epochs,
            batch_size=batch_size,
            patience=5,
            experiment_id=experiment_id
        )
        
        print(f"[API] Resultado de reentrenamiento: {result}")
        
        if result['success']:
            print(f"[API] ✓ Reentrenamiento exitoso!")
            print(f"[API] Cargando nuevo modelo...")
            
            # Reload the new model
            global model
            model = YOLO(str(result['model_path']))
            if device == 'cuda':
                model.to(device)
            
            print(f"[API] ✓ Nuevo modelo cargado: {result['model_path']}")
            
            return jsonify({
                'success': True,
                'new_version': result['version'],
                'model_path': result['model_path'],
                'metrics': result.get('metrics', {}),
                'mlflow_run_id': result.get('mlflow_run_id'),
                'experiment_id': experiment_id,
                'message': f'✓ Modelo reentrenado guardado en: models/retrained_v{result["version"]}.pt',
                'mlflow_message': f'✓ MLflow run registrado en experiment {experiment_id}'
            }), 200
        else:
            error_msg = result.get('error') or result.get('reason', 'Retraining failed')
            print(f"[API] ❌ Error: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg,
                'reason': result.get('reason')
            }), 400
            
    except Exception as e:
        error_msg = f"Retraining exception: {str(e)}"
        print(f"[API] ❌ EXCEPCIÓN: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': error_msg}), 500

@app.route('/api/models/available', methods=['GET'])
def get_available_models():
    """Get list of available models (base + retrained)"""
    models = []
    if BEST_MODEL_PATH is None or not BEST_MODEL_PATH.exists():
        return jsonify({
            'models': [],
            'current_path': None,
            'error': 'Model not loaded'
        }), 200

    models_dir = BEST_MODEL_PATH.parent
    
    # Modelo base
    current_path = str(model.model.pt_path) if (model is not None and hasattr(model, 'model')) else str(BEST_MODEL_PATH)

    models.append({
        'name': 'best_improved.pt (Base)',
        'path': str(BEST_MODEL_PATH),
        'type': 'base',
        'is_current': current_path == str(BEST_MODEL_PATH)
    })
    
    # Modelos reentrenados
    if models_dir.exists():
        for pt_file in sorted(models_dir.glob('retrained_v*.pt'), reverse=True):
            version = pt_file.stem.replace('retrained_v', '')
            models.append({
                'name': f'retrained_v{version}',
                'path': str(pt_file),
                'type': 'retrained',
                'is_current': False
            })
    
    return jsonify({'models': models, 'current_path': current_path}), 200

@app.route('/api/models/switch', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    try:
        data = request.json or {}
        model_path = data.get('model_path')
        
        if not model_path:
            return jsonify({'success': False, 'error': 'model_path required'}), 400
        
        model_file = Path(model_path)
        if not model_file.exists():
            return jsonify({'success': False, 'error': f'Model file not found: {model_path}'}), 404
        
        global model, learner
        try:
            model = YOLO(str(model_file))
            if device == 'cuda':
                model.to(device)
            
            # Reinitialize learner with new model
            from continuous_learning import ContinuousLearner
            learner = ContinuousLearner(str(model_file))
            
            print(f"[MODEL SWITCH] Switched to: {model_path}")
            return jsonify({
                'success': True,
                'model_path': str(model_file),
                'message': f'Modelo cambiado a: {model_file.name}'
            }), 200
        except Exception as e:
            print(f"[MODEL SWITCH] Error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/advanced')
def advanced():
    """Serve the advanced UI"""
    return render_template('correcciones.html')

if __name__ == '__main__':
    print("=" * 60)
    print("YOLO Object Detection API")
    print("=" * 60)
    print(f"Dispositivo: {device.upper()}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 60)
    
    if load_model():
        print(f"\nModel loaded successfully!")
        print(f"Path: {BEST_MODEL_PATH}")
        print(f"Classes: {CLASS_NAMES}")
        print(f"Running on: {device.upper()}")
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
