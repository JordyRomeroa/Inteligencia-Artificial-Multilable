"""
Script para probar los endpoints de la API
"""
import requests
import json
from pathlib import Path

BASE_URL = 'http://127.0.0.1:5000'
DATA_DIR = Path('data/test_images')  # Esperar que haya imágenes de test

def test_predict():
    """Prueba el endpoint /predict"""
    print("\n" + "="*60)
    print("PRUEBA: /predict")
    print("="*60)
    
    # Buscar una imagen en uploads
    uploads_dir = Path('uploads')
    if not uploads_dir.exists():
        uploads_dir.mkdir(exist_ok=True)
    
    # Copiar una imagen de test si está disponible
    test_image = uploads_dir / 'test.jpg'
    test_img_src = Path('data/voc2007/images/') / '000001.jpg'
    
    if not test_image.exists() and test_img_src.exists():
        import shutil
        shutil.copy2(test_img_src, test_image)
        print(f"   ℹ Imagen de test copiada: {test_image}")
    
    if test_image.exists():
        try:
            with open(test_image, 'rb') as f:
                files = {'file': f}
                data = {'threshold': 0.5}
                response = requests.post(f'{BASE_URL}/predict', files=files, data=data, timeout=30)
            
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Predicción exitosa")
                print(f"     - Predicciones: {len(result.get('predictions', []))} clases")
                print(f"     - Con confianza >0.5: {len([p for p in result.get('predictions', []) if p['confidence'] > 0.5])}")
                return True
            else:
                print(f"   ✗ Error: {response.text}")
                return False
        except Exception as e:
            print(f"   ✗ Error de conexión: {e}")
            return False
    else:
        print(f"   ⚠ No hay imagen de test disponible")
        return False

def test_corrections():
    """Prueba el endpoint /get_corrections"""
    print("\n" + "="*60)
    print("PRUEBA: /get_corrections")
    print("="*60)
    
    try:
        response = requests.get(f'{BASE_URL}/get_corrections', timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Correcciones obtenidas")
            print(f"     - Total de correcciones: {len(result.get('corrections', []))}")
            
            for correction in result.get('corrections', [])[:3]:
                print(f"     - {correction.get('filename')}: {correction.get('timestamp')}")
            return True
        else:
            print(f"   ✗ Error: {response.text}")
            return False
    except Exception as e:
        print(f"   ✗ Error de conexión: {e}")
        return False

def test_simple_page():
    """Prueba que la página simple funciona"""
    print("\n" + "="*60)
    print("PRUEBA: /simple (página HTML)")
    print("="*60)
    
    try:
        response = requests.get(f'{BASE_URL}/simple', timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            if '<html' in response.text.lower():
                print(f"   ✓ Página HTML cargada correctamente")
                print(f"     - Tamaño: {len(response.text)} bytes")
                return True
            else:
                print(f"   ✗ Respuesta no es HTML")
                return False
        else:
            print(f"   ✗ Error: {response.text}")
            return False
    except Exception as e:
        print(f"   ✗ Error de conexión: {e}")
        return False

def main():
    """Ejecuta todas las pruebas"""
    print("\n" + "="*60)
    print("PRUEBAS DE LA API")
    print("="*60)
    
    results = {
        'simple_page': test_simple_page(),
        'corrections': test_corrections(),
        'predict': test_predict(),
    }
    
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {status}: {test_name}")
    
    all_passed = all(results.values())
    print(f"\n{'✓ Todas las pruebas pasaron' if all_passed else '✗ Algunas pruebas fallaron'}\n")
    return all_passed

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
