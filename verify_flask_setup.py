"""
Script para verificar que la configuración de Flask está correcta
y que los archivos se sirven correctamente
"""

import sys
from pathlib import Path

# Colores
class C:
    G = '\033[92m'  # Green
    R = '\033[91m'  # Red
    Y = '\033[93m'  # Yellow
    B = '\033[94m'  # Blue
    E = '\033[0m'   # End

def test_static_files():
    """Verifica que los archivos estáticos existen"""
    print(f"\n{C.B}[Verificando archivos estáticos]{C.E}")
    
    BASE = Path(__file__).parent / 'app'
    
    files = {
        'style.css': BASE / 'static' / 'style.css',
        'script.js': BASE / 'static' / 'script.js',
        'favicon.ico': BASE / 'static' / 'favicon.ico',
        'index.html': BASE / 'templates' / 'index.html',
        'test.html': BASE / 'templates' / 'test.html',
    }
    
    all_ok = True
    for name, path in files.items():
        if path.exists():
            size = path.stat().st_size
            print(f"{C.G}✓{C.E} {name:20} ({size:8,} bytes)")
        else:
            print(f"{C.R}✗{C.E} {name:20} NO EXISTE")
            all_ok = False
    
    return all_ok

def test_script_js():
    """Verifica que script.js tiene las funciones necesarias"""
    print(f"\n{C.B}[Verificando funciones en script.js]{C.E}")
    
    script_path = Path(__file__).parent / 'app' / 'static' / 'script.js'
    
    if not script_path.exists():
        print(f"{C.R}✗{C.E} script.js no encontrado")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    functions = [
        'function showTab',
        'function previewImage',
        'function predictImage',
        'function updateThreshold',
        'function saveCorrection',
        'function retrainModel',
        'function showStatus',
        'let currentFilename',
        'let currentPredictions',
        'let batchFiles'
    ]
    
    all_ok = True
    for func in functions:
        if func in content:
            print(f"{C.G}✓{C.E} Definido: {func}")
        else:
            print(f"{C.R}✗{C.E} Faltante: {func}")
            all_ok = False
    
    return all_ok

def test_html():
    """Verifica que index.html está bien configurado"""
    print(f"\n{C.B}[Verificando index.html]{C.E}")
    
    html_path = Path(__file__).parent / 'app' / 'templates' / 'index.html'
    
    if not html_path.exists():
        print(f"{C.R}✗{C.E} index.html no encontrado")
        return False
    
    with open(html_path, 'r') as f:
        content = f.read()
    
    checks = {
        'Link a style.css': 'href="{{ url_for(\'static\', filename=\'style.css\') }}"',
        'Link a script.js': 'href="{{ url_for(\'static\', filename=\'script.js\') }}"',
        'Link a favicon': 'href="{{ url_for(\'static\', filename=\'favicon.ico\') }}"',
        'Jinja2 classes': '{{ classes | tojson }}',
        'Tab buttons': 'onclick="showTab(\'single\')"',
    }
    
    all_ok = True
    for name, check_str in checks.items():
        if check_str in content:
            print(f"{C.G}✓{C.E} {name}")
        else:
            print(f"{C.R}✗{C.E} {name} - No encontrado")
            all_ok = False
    
    # Check for duplicate declarations
    if 'let currentFilename' in content and content.count('let currentFilename') > 1:
        print(f"{C.R}✗{C.E} Declaración duplicada: currentFilename")
        all_ok = False
    else:
        print(f"{C.G}✓{C.E} Sin duplicaciones en index.html")
    
    return all_ok

def test_api_py():
    """Verifica que api.py está configurado correctamente"""
    print(f"\n{C.B}[Verificando api.py]{C.E}")
    
    api_path = Path(__file__).parent / 'app' / 'api.py'
    
    if not api_path.exists():
        print(f"{C.R}✗{C.E} api.py no encontrado")
        return False
    
    with open(api_path, 'r') as f:
        content = f.read()
    
    checks = {
        'STATIC_DIR definido': 'STATIC_DIR = BASE_DIR / \'static\'',
        'TEMPLATE_DIR definido': 'TEMPLATE_DIR = BASE_DIR / \'templates\'',
        'Flask con static_folder': 'static_folder=str(STATIC_DIR)',
        'Flask con template_folder': 'template_folder=str(TEMPLATE_DIR)',
        'Endpoint /favicon.ico': '@app.route(\'/favicon.ico\')',
        'Endpoint /test': '@app.route(\'/test\')',
        'send_from_directory importado': 'send_from_directory',
    }
    
    all_ok = True
    for name, check_str in checks.items():
        if check_str in content:
            print(f"{C.G}✓{C.E} {name}")
        else:
            print(f"{C.R}✗{C.E} {name}")
            all_ok = False
    
    return all_ok

def main():
    print(f"\n{C.B}{'='*60}{C.E}")
    print(f"{C.B}Verificación Completa de Configuración{C.E}")
    print(f"{C.B}{'='*60}{C.E}")
    
    results = {
        'Archivos estáticos': test_static_files(),
        'script.js': test_script_js(),
        'index.html': test_html(),
        'api.py': test_api_py(),
    }
    
    print(f"\n{C.B}{'='*60}{C.E}")
    print(f"{C.B}RESUMEN{C.E}")
    print(f"{C.B}{'='*60}{C.E}")
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = f"{C.G}✓ PASÓ{C.E}" if passed else f"{C.R}✗ FALLÓ{C.E}"
        print(f"{test_name:30} {status}")
    
    print(f"{C.B}{'='*60}{C.E}")
    
    if all_passed:
        print(f"{C.G}✓ TODO ESTÁ CONFIGURADO CORRECTAMENTE{C.E}")
        print(f"\n{C.B}Próximos pasos:{C.E}")
        print(f"  1. Ejecuta: python app/api.py")
        print(f"  2. Abre: http://localhost:5000")
        print(f"  3. O test: http://localhost:5000/test")
        return 0
    else:
        print(f"{C.R}✗ HAY PROBLEMAS EN LA CONFIGURACIÓN{C.E}")
        print(f"\n{C.Y}Revisa los errores arriba y corrige los archivos{C.E}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
