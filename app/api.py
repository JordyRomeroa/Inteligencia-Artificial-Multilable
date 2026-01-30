"""
API Flask para predicción multilabel con sistema de correcciones y reentrenamiento.
"""
import os
import json
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import pickle

# Configurar rutas
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / 'static'
TEMPLATE_DIR = BASE_DIR / 'templates'

app = Flask(__name__, static_folder=str(STATIC_DIR), template_folder=str(TEMPLATE_DIR))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Directorios
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data' / 'voc2007'
CORRECTIONS_DIR = PROJECT_ROOT / 'data' / 'corrections'
CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Cargar modelo y clases
def load_model_and_classes():
    """Carga el modelo y clases"""
    # Cargar clases
    with open(DATA_DIR / 'classes.json', 'r') as f:
        classes = json.load(f)
    
    # Definir focal loss
    def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, gamma)
        focal_bce = focal_weight * bce
        return tf.reduce_mean(focal_bce)
    
    # Cargar modelo
    model_path = MODELS_DIR / 'voc_multilabel_final.h5'
    if not model_path.exists():
        model_path = MODELS_DIR / 'food_multilabel_final.h5'
    
    model = keras.models.load_model(
        model_path,
        custom_objects={'focal_loss': focal_loss},
        compile=False
    )
    
    # Cargar pesos guardados del reentrenamiento si existen
    # En TensorFlow 2.20+ busca .weights.h5
    weights_path = MODELS_DIR / 'voc_multilabel_final.weights.h5'
    if weights_path.exists():
        try:
            model.load_weights(str(weights_path))
            print(f'✓ Pesos cargados desde reentrenamiento: {weights_path}')
        except Exception as e:
            print(f'⚠ Error al cargar pesos: {e}')
    
    # IMPORTANTE: Compilar el modelo para que funcione correctamente
    # Usar Adam optimizer con learning rate bajo para estabilidad
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model, classes

model, classes = load_model_and_classes()
NUM_CLASSES = len(classes)

def allowed_file(filename):
    """Verifica si la extensión es permitida"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocesa imagen para predicción"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/favicon.ico')
def favicon():
    """Serve favicon"""
    return send_from_directory(STATIC_DIR, 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html', classes=classes)

@app.route('/test')
def test():
    """Página de test para debugging"""
    return render_template('test.html')

@app.route('/simple')
def simple():
    """Página simplificada funcional"""
    return render_template('simple.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de predicción"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'File type not allowed'}), 400
        
        # Guardar archivo
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocesar y predecir
        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Obtener threshold desde request o usar 0.5
        threshold = float(request.form.get('threshold', 0.5))
        
        # Filtrar solo predicciones positivas (encima del threshold)
        results = []
        for i, prob in enumerate(predictions):
            if prob >= threshold:
                results.append({
                    'label': classes[i],
                    'confidence': float(prob)
                })
        
        # Ordenar por probabilidad (confianza)
        results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'predictions': results,
            'threshold': threshold,
            'all_probabilities': {classes[i]: float(predictions[i]) for i in range(len(classes))}
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/save_correction', methods=['POST'])
def save_correction():
    """Guarda corrección del usuario"""
    try:
        data = request.json
        filename = data.get('filename')
        corrected_labels = data.get('corrected_labels', [])
        
        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided'}), 400
        
        # Crear vector de etiquetas correctas
        label_vector = np.zeros(NUM_CLASSES, dtype=np.float32)
        for label_name in corrected_labels:
            if label_name in classes:
                idx = classes.index(label_name)
                label_vector[idx] = 1.0
        
        # Guardar corrección como JSON
        import datetime
        correction_file = CORRECTIONS_DIR / f'{Path(filename).stem}_correction.json'
        with open(correction_file, 'w') as f:
            json.dump({
                'filename': filename,
                'correct_labels': corrected_labels,
                'label_vector': label_vector.tolist(),
                'timestamp': datetime.datetime.now().isoformat()
            }, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': f'Corrección guardada: {len(corrected_labels)} etiquetas'
        })
    
    except Exception as e:
        print(f'Error en save_correction: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Reentrena el modelo con correcciones acumuladas"""
    try:
        import sys
        sys.path.append(str(PROJECT_ROOT / 'app'))
        from utils import incremental_retrain
        
        # Cargar todas las correcciones
        correction_files = list(CORRECTIONS_DIR.glob('*_correction.json'))
        
        if len(correction_files) == 0:
            return jsonify({'success': False, 'error': 'No hay correcciones para reentrenar'}), 400
        
        print(f'Encontradas {len(correction_files)} correcciones')
        
        # Cargar imágenes y etiquetas corregidas
        images = []
        labels = []
        
        for correction_file in correction_files:
            try:
                with open(correction_file, 'r') as f:
                    correction_data = f.read().strip()
                    
                # Saltar archivos vacíos o corruptos
                if not correction_data:
                    print(f'Archivo vacío, ignorando: {correction_file}')
                    continue
                    
                correction = json.loads(correction_data)
                
                # Cargar imagen
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], correction['filename'])
                if os.path.exists(img_path):
                    img_array = preprocess_image(img_path)[0]  # Remover batch dimension
                    images.append(img_array)
                    labels.append(np.array(correction['label_vector']))
                    print(f'Cargada corrección: {correction["filename"]}')
                else:
                    print(f'Imagen no encontrada: {img_path}')
            except json.JSONDecodeError as e:
                print(f'Error al decodificar JSON {correction_file}: {str(e)}')
                # Intentar eliminar archivo corrupto
                try:
                    correction_file.unlink()
                    print(f'Archivo corrupto eliminado: {correction_file}')
                except:
                    pass
            except Exception as e:
                print(f'Error al cargar corrección {correction_file}: {str(e)}')
        
        if len(images) == 0:
            return jsonify({'success': False, 'error': 'No se encontraron imágenes para reentrenar'}), 400
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f'Reentrenando con {len(images)} imágenes...')
        print(f'Usando learning_rate=1e-6 y epochs=15 para mejor ajuste')
        
        # Reentrenar con learning rate muy bajo y más épocas para pocas imágenes
        history = incremental_retrain(model, images, labels, epochs=15, learning_rate=1e-6, verbose=1)
        
        # Guardar solo los pesos del modelo (no el modelo completo)
        # Esto evita problemas con pickling de funciones personalizadas
        # En TensorFlow 2.20+ el archivo debe terminar en .weights.h5
        try:
            weights_path = str(MODELS_DIR / 'voc_multilabel_final.weights.h5')
            model.save_weights(weights_path)
            print(f'✓ Pesos guardados en: {weights_path}')
            
            # IMPORTANTE: Recargar los pesos en el modelo global para que las siguientes
            # predicciones usen el modelo reentrenado
            model.load_weights(weights_path)
            print(f'✓ Pesos recargados en el modelo en memoria')
            
            # Compilar el modelo después de cargar los pesos con optimizer estable
            from tensorflow.keras.optimizers import Adam
            model.compile(
                optimizer=Adam(learning_rate=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            print(f'✓ Modelo compilado después del reentrenamiento')
        except Exception as e:
            print(f'✗ Error al guardar/recargar pesos: {str(e)}')
        
        return jsonify({
            'success': True,
            'message': f'Modelo reentrenado con {len(images)} imágenes y pesos actualizados',
            'samples': len(images),
            'final_loss': float(history.history['loss'][-1]) if history else None
        })
    
    except Exception as e:
        print(f'Error en retrain: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predicción en batch de múltiples imágenes"""
    files = request.files.getlist('files')
    threshold = float(request.form.get('threshold', 0.5))
    
    if not files:
        return jsonify({'error': 'No files provided'}), 400
    
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predecir
            img_array = preprocess_image(filepath)
            predictions = model.predict(img_array, verbose=0)[0]
            
            # Top 5 predicciones
            top_predictions = []
            for i, prob in enumerate(predictions):
                if prob >= threshold:
                    top_predictions.append({
                        'class': classes[i],
                        'probability': float(prob)
                    })
            
            top_predictions = sorted(top_predictions, key=lambda x: x['probability'], reverse=True)[:5]
            
            results.append({
                'filename': filename,
                'filepath': filepath,
                'predictions': top_predictions,
                'num_labels': len(top_predictions)
            })
    
    return jsonify({
        'success': True,
        'results': results,
        'total': len(results)
    })

@app.route('/get_corrections', methods=['GET'])
def get_corrections():
    """Obtiene estadísticas de correcciones"""
    correction_files = list(CORRECTIONS_DIR.glob('*_correction.json'))
    
    corrections = []
    for correction_file in correction_files:
        with open(correction_file, 'r') as f:
            data = json.load(f)
            corrections.append({
                'filename': data.get('filename', 'unknown'),
                'corrected_labels': data.get('correct_labels', []),
                'timestamp': data.get('timestamp', None)
            })
    
    return jsonify({
        'success': True,
        'total': len(corrections),
        'corrections': corrections
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'num_classes': NUM_CLASSES,
        'classes': classes
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
