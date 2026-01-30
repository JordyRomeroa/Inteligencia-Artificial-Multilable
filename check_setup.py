"""
Script de prueba rápida para verificar la configuración de la app web
"""
import sys
from pathlib import Path

# Colores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def check(condition, message):
    """Helper para imprimir checks"""
    if condition:
        print(f"{Colors.GREEN}✓{Colors.END} {message}")
        return True
    else:
        print(f"{Colors.RED}✗{Colors.END} {message}")
        return False

def main():
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}Verificación de Configuración - App Web Multilabel{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    PROJECT_ROOT = Path(__file__).parent
    errors = 0
    
    # 1. Verificar directorios
    print(f"{Colors.YELLOW}[1] Verificando estructura de directorios...{Colors.END}")
    
    dirs_to_check = [
        PROJECT_ROOT / 'app',
        PROJECT_ROOT / 'app' / 'templates',
        PROJECT_ROOT / 'app' / 'static',
        PROJECT_ROOT / 'data' / 'corrections',
        PROJECT_ROOT / 'data' / 'uploads',
        PROJECT_ROOT / 'models'
    ]
    
    for dir_path in dirs_to_check:
        if not check(dir_path.exists(), f"Directorio: {dir_path.name}/"):
            errors += 1
    
    # 2. Verificar archivos de la app
    print(f"\n{Colors.YELLOW}[2] Verificando archivos de la aplicación...{Colors.END}")
    
    files_to_check = [
        PROJECT_ROOT / 'app' / 'api.py',
        PROJECT_ROOT / 'app' / 'utils.py',
        PROJECT_ROOT / 'app' / 'templates' / 'index.html',
        PROJECT_ROOT / 'app' / 'static' / 'style.css',
        PROJECT_ROOT / 'app' / 'static' / 'script.js'
    ]
    
    for file_path in files_to_check:
        if not check(file_path.exists(), f"Archivo: {file_path.name}"):
            errors += 1
    
    # 3. Verificar archivos del modelo
    print(f"\n{Colors.YELLOW}[3] Verificando archivos del modelo...{Colors.END}")
    
    model_path = PROJECT_ROOT / 'models' / 'voc_multilabel_final.h5'
    classes_path = PROJECT_ROOT / 'data' / 'voc2007' / 'classes.json'
    
    if not check(model_path.exists(), f"Modelo entrenado: {model_path.name}"):
        print(f"   {Colors.RED}→ Ejecuta el notebook 03_training_real_images.ipynb primero{Colors.END}")
        errors += 1
    
    if not check(classes_path.exists(), f"Archivo de clases: classes.json"):
        print(f"   {Colors.RED}→ Ejecuta el notebook 01_data_analysis.ipynb primero{Colors.END}")
        errors += 1
    
    # 4. Verificar dependencias de Python
    print(f"\n{Colors.YELLOW}[4] Verificando dependencias de Python...{Colors.END}")
    
    dependencies = [
        ('flask', 'Flask'),
        ('tensorflow', 'TensorFlow'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('werkzeug', 'Werkzeug')
    ]
    
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            check(True, f"Dependencia: {display_name}")
        except ImportError:
            check(False, f"Dependencia: {display_name}")
            print(f"   {Colors.YELLOW}→ Instala con: pip install {module_name if module_name != 'PIL' else 'pillow'}{Colors.END}")
            errors += 1
    
    # 5. Verificar configuración del modelo
    if model_path.exists() and classes_path.exists():
        print(f"\n{Colors.YELLOW}[5] Verificando configuración del modelo...{Colors.END}")
        
        try:
            import json
            with open(classes_path, 'r') as f:
                classes = json.load(f)
            
            check(True, f"Clases cargadas: {len(classes)} categorías")
            print(f"   → Clases: {', '.join(classes[:5])}...")
            
        except Exception as e:
            check(False, f"Error al cargar clases: {str(e)}")
            errors += 1
    
    # Resumen final
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    if errors == 0:
        print(f"{Colors.GREEN}✓ Todo listo! Puedes ejecutar la aplicación{Colors.END}")
        print(f"\n{Colors.BLUE}Para iniciar:{Colors.END}")
        print(f"  1. Ejecuta: {Colors.GREEN}python app/api.py{Colors.END}")
        print(f"  2. Abre: {Colors.GREEN}http://localhost:5000{Colors.END}")
        print(f"  3. O usa: {Colors.GREEN}run_app.bat{Colors.END}")
    else:
        print(f"{Colors.RED}✗ Encontrados {errors} error(es) - Revisa los mensajes arriba{Colors.END}")
        print(f"\n{Colors.YELLOW}Pasos recomendados:{Colors.END}")
        print(f"  1. Asegúrate de haber ejecutado los notebooks 01 y 03")
        print(f"  2. Instala las dependencias faltantes")
        print(f"  3. Vuelve a ejecutar este script")
    
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    return 0 if errors == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
