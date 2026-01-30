"""
Script para probar el ciclo completo de predicción, corrección y reentrenamiento
"""
import requests
import json
from pathlib import Path
import time

BASE_URL = 'http://127.0.0.1:5000'
UPLOADS_DIR = Path('uploads')
CORRECTIONS_DIR = Path('data/corrections')

def copy_test_image():
    """Copia una imagen de prueba a la carpeta de uploads"""
    UPLOADS_DIR.mkdir(exist_ok=True)
    
    # Buscar imágenes en test_images
    test_images = list(Path('data/test_images').glob('*.jpg'))[:3]
    
    if not test_images:
        print("✗ No hay imágenes de prueba en data/test_images")
        return None
    
    # Copiar la primera imagen
    import shutil
    test_img_src = test_images[0]
    test_img_dst = UPLOADS_DIR / test_img_src.name
    shutil.copy2(test_img_src, test_img_dst)
    print(f"✓ Imagen copiada: {test_img_dst.name}")
    return test_img_dst.name

def predict(filename, threshold=0.5):
    """Predecir imagen"""
    print(f"\n{'='*60}")
    print(f"1. PREDICCIÓN: {filename}")
    print(f"{'='*60}")
    
    img_path = UPLOADS_DIR / filename
    if not img_path.exists():
        print(f"✗ Imagen no existe: {img_path}")
        return None
    
    try:
        with open(img_path, 'rb') as f:
            files = {'file': f}
            data = {'threshold': threshold}
            response = requests.post(f'{BASE_URL}/predict', files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Predicción exitosa")
            print(f"  Confianza > {threshold}: {len(result.get('predictions', []))} clases")
            
            predictions = sorted(result.get('predictions', []), key=lambda x: x['confidence'], reverse=True)
            print(f"\n  Top 5 predicciones:")
            for i, pred in enumerate(predictions[:5], 1):
                print(f"    {i}. {pred['class']:15} {pred['confidence']*100:5.1f}%")
            
            return result
        else:
            print(f"✗ Error: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Error de conexión: {e}")
        return None

def save_correction(filename, predictions, selected_indices):
    """Guardar corrección"""
    print(f"\n{'='*60}")
    print(f"2. GUARDAR CORRECCIÓN")
    print(f"{'='*60}")
    
    if not predictions:
        print("✗ No hay predicciones")
        return False
    
    # Construir vector de etiquetas corregidas
    all_classes = predictions.get('classes', [])
    label_vector = [0] * len(all_classes)
    
    for idx in selected_indices:
        if 0 <= idx < len(label_vector):
            label_vector[idx] = 1
    
    corrected_labels = [all_classes[i] for i, v in enumerate(label_vector) if v == 1]
    
    print(f"  Clases seleccionadas: {corrected_labels}")
    print(f"  Vector: {sum(label_vector)} etiquetas marcadas")
    
    try:
        data = {
            'filename': filename,
            'corrected_labels': corrected_labels,
            'label_vector': label_vector
        }
        response = requests.post(f'{BASE_URL}/save_correction', json=data, timeout=10)
        
        if response.status_code == 200:
            print(f"✓ Corrección guardada")
            return True
        else:
            print(f"✗ Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error de conexión: {e}")
        return False

def retrain():
    """Reentrenar modelo"""
    print(f"\n{'='*60}")
    print(f"3. REENTRENAMIENTO")
    print(f"{'='*60}")
    
    # Contar correcciones
    corrections = list(CORRECTIONS_DIR.glob('*_correction.json'))
    print(f"  Correcciones disponibles: {len(corrections)}")
    
    if len(corrections) == 0:
        print("  ⚠ No hay correcciones para reentrenar")
        return False
    
    try:
        response = requests.post(f'{BASE_URL}/retrain', timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Reentrenamiento completado")
            print(f"  Mensajes: {result.get('message')}")
            print(f"  Pérdida final: {result.get('final_loss', 'N/A')}")
            return True
        else:
            print(f"✗ Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error de conexión: {e}")
        return False

def main():
    """Ejecuta el ciclo completo"""
    print("\n" + "="*60)
    print("PRUEBA DE CICLO COMPLETO")
    print("="*60)
    
    # 1. Copiar imagen
    print("\n[PREPARACIÓN]")
    filename = copy_test_image()
    if not filename:
        return False
    
    # 2. Primera predicción
    predictions_1 = predict(filename, threshold=0.5)
    if not predictions_1:
        return False
    
    # 3. Guardar corrección (marcar algunos items)
    # Vamos a marcar como correctos los primeros 3-5 items
    all_preds = predictions_1.get('predictions', [])
    selected = min(3, len(all_preds))
    save_correction(filename, predictions_1, list(range(selected)))
    
    # 4. Esperar un poco
    print("\n⏳ Esperando 2 segundos antes de reentrenar...")
    time.sleep(2)
    
    # 5. Reentrenar
    retrain()
    
    # 6. Segunda predicción (debe cambiar ahora)
    print("\n⏳ Esperando 2 segundos antes de nueva predicción...")
    time.sleep(2)
    
    predictions_2 = predict(filename, threshold=0.5)
    if not predictions_2:
        return False
    
    # 7. Comparar resultados
    print(f"\n{'='*60}")
    print(f"4. COMPARACIÓN ANTES/DESPUÉS")
    print(f"{'='*60}")
    
    preds_1 = set(p['class'] for p in predictions_1.get('predictions', []))
    preds_2 = set(p['class'] for p in predictions_2.get('predictions', []))
    
    print(f"\n  ANTES: {preds_1}")
    print(f"  DESPUÉS: {preds_2}")
    
    if preds_1 != preds_2:
        print(f"\n  ✓ LAS PREDICCIONES CAMBIARON (reentrenamiento funcionó)")
        return True
    else:
        print(f"\n  ⚠ Las predicciones son iguales (modelo no cambió)")
        return False

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
