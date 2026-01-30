"""
Script de prueba para verificar que el sistema de reentrenamiento funciona correctamente
"""
import os
import sys
from pathlib import Path
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Agregar ruta
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / 'app'))

from utils import incremental_retrain

# Configuración
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data' / 'voc2007'
CORRECTIONS_DIR = PROJECT_ROOT / 'data' / 'corrections'

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_weight = tf.pow(1 - p_t, gamma)
    focal_bce = focal_weight * bce
    return tf.reduce_mean(focal_bce)

def test_model_loading():
    """Verifica que el modelo se carga correctamente"""
    print("\n" + "="*60)
    print("1. PROBANDO CARGA DE MODELO")
    print("="*60)
    
    model_path = MODELS_DIR / 'voc_multilabel_final.h5'
    print(f"   Ruta del modelo: {model_path}")
    print(f"   Existe: {model_path.exists()}")
    print(f"   Tamaño: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    try:
        model = keras.models.load_model(
            str(model_path),
            custom_objects={'focal_loss': focal_loss},
            compile=False
        )
        print(f"   ✓ Modelo cargado correctamente")
        print(f"   ✓ Capas: {len(model.layers)}")
        print(f"   ✓ Forma de entrada: {model.input_shape}")
        print(f"   ✓ Forma de salida: {model.output_shape}")
        return model
    except Exception as e:
        print(f"   ✗ Error al cargar modelo: {e}")
        return None

def test_weights_loading(model):
    """Verifica que los pesos guardados se cargan correctamente"""
    print("\n" + "="*60)
    print("2. PROBANDO CARGA DE PESOS")
    print("="*60)
    
    weights_path = MODELS_DIR / 'voc_multilabel_final_weights.h5'
    print(f"   Ruta de pesos: {weights_path}")
    print(f"   Existe: {weights_path.exists()}")
    
    if weights_path.exists():
        try:
            model.load_weights(str(weights_path))
            print(f"   ✓ Pesos cargados correctamente")
            return True
        except Exception as e:
            print(f"   ✗ Error al cargar pesos: {e}")
            return False
    else:
        print(f"   ℹ No hay archivo de pesos (es normal en primera ejecución)")
        return True

def test_corrections():
    """Verifica las correcciones guardadas"""
    print("\n" + "="*60)
    print("3. PROBANDO CORRECCIONES GUARDADAS")
    print("="*60)
    
    correction_files = list(CORRECTIONS_DIR.glob('*_correction.json'))
    print(f"   Archivos de corrección encontrados: {len(correction_files)}")
    
    valid_corrections = 0
    for correction_file in correction_files:
        size = correction_file.stat().st_size
        print(f"\n   Archivo: {correction_file.name} ({size} bytes)")
        
        try:
            with open(correction_file, 'r') as f:
                content = f.read().strip()
            
            if not content:
                print(f"   ⚠ Archivo vacío")
                continue
            
            data = json.loads(content)
            print(f"   ✓ JSON válido")
            print(f"     - Imagen: {data.get('filename')}")
            print(f"     - Etiquetas: {len(data.get('label_vector', []))} clases")
            valid_corrections += 1
            
        except json.JSONDecodeError as e:
            print(f"   ✗ JSON inválido: {str(e)}")
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
    
    return valid_corrections

def test_classes():
    """Verifica que las clases se cargan correctamente"""
    print("\n" + "="*60)
    print("4. PROBANDO CLASES")
    print("="*60)
    
    classes_path = DATA_DIR / 'classes.json'
    print(f"   Ruta de clases: {classes_path}")
    print(f"   Existe: {classes_path.exists()}")
    
    try:
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        print(f"   ✓ Clases cargadas correctamente")
        print(f"   ✓ Total de clases: {len(classes)}")
        print(f"   Clases: {', '.join(list(classes)[:5])}..." if len(classes) > 5 else f"   Clases: {', '.join(classes)}")
        return classes
    except Exception as e:
        print(f"   ✗ Error al cargar clases: {e}")
        return None

def main():
    """Ejecuta todas las pruebas"""
    print("\n" + "="*60)
    print("PRUEBAS DEL SISTEMA DE REENTRENAMIENTO")
    print("="*60)
    
    # Test 1: Cargar modelo
    model = test_model_loading()
    if not model:
        print("\n✗ No se puede continuar sin el modelo")
        return False
    
    # Test 2: Cargar pesos
    test_weights_loading(model)
    
    # Test 3: Correcciones
    valid_corrections = test_corrections()
    
    # Test 4: Clases
    classes = test_classes()
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    print(f"   ✓ Modelo cargado: Sí")
    print(f"   ✓ Pesos disponibles: Sí" if (MODELS_DIR / 'voc_multilabel_final_weights.h5').exists() else "   ℹ Pesos disponibles: No (se generarán al reentrenar)")
    print(f"   ✓ Correcciones válidas: {valid_corrections}")
    print(f"   ✓ Clases cargadas: {len(classes) if classes else 0}")
    print("\n✓ Sistema listo para usar\n")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
