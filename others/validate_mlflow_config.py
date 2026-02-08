#!/usr/bin/env python
"""
Script de Validación de MLflow - Verificar Configuración Obligatoria
=====================================================================

Este script valida que:
1. El experiment_id específico existe
2. El artifact_location está configurado correctamente
3. Los runs se guardarán en la ruta correcta
4. MLflow puede escribir en los directorios requeridos

Ejecutar antes de hacer reentrenamientos:
    python validate_mlflow_config.py
"""

import sys
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

# =====================================================================
# CONFIGURACIÓN OBLIGATORIA
# =====================================================================
REQUIRED_EXPERIMENT_ID = '401576597529460193'
REQUIRED_ARTIFACT_LOCATION = 'file:///c:/Users/jordy/OneDrive/Desktop/iaaaa/iajordy2/runs/mlflow/401576597529460193'
PROJECT_ROOT = Path(__file__).parent
RUNS_DIR = PROJECT_ROOT / 'runs'
MLFLOW_DIR = RUNS_DIR / 'mlflow'
EXPERIMENT_DIR = MLFLOW_DIR / REQUIRED_EXPERIMENT_ID

def print_header(text):
    """Imprimir encabezado de sección"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_check(status, message):
    """Imprimir resultado de validación"""
    symbol = "✓" if status else "✗"
    color = "\033[92m" if status else "\033[91m"  # Green : Red
    reset = "\033[0m"
    print(f"{color}[{symbol}]{reset} {message}")

def validate_directory_structure():
    """Validar que existan los directorios necesarios"""
    print_header("1. VALIDAR ESTRUCTURA DE DIRECTORIOS")
    
    dirs_to_check = [
        (RUNS_DIR, "Directorio runs/"),
        (MLFLOW_DIR, "Directorio runs/mlflow/"),
        (EXPERIMENT_DIR, "Directorio runs/mlflow/401576597529460193/"),
    ]
    
    all_exist = True
    for dir_path, name in dirs_to_check:
        exists = dir_path.exists()
        print_check(exists, f"{name}: {dir_path}")
        if not exists:
            all_exist = False
    
    # Crear directorio si no existe
    if not all_exist:
        print("\n⚠️  Creando directorios faltantes...")
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        print_check(True, "Directorios creados correctamente")
    
    return all_exist or True

def validate_mlflow_configuration():
    """Validar configuración de MLflow"""
    print_header("2. VALIDAR CONFIGURACIÓN DE MLFLOW")
    
    # Convertir ruta a URI para Windows
    mlflow_path = str(MLFLOW_DIR).replace('\\', '/')
    if len(mlflow_path) > 1 and mlflow_path[1] == ':':
        tracking_uri = f"file:///{mlflow_path}"
    else:
        tracking_uri = f"file://{mlflow_path}"
    
    print(f"\nConfigurando MLflow con:")
    print(f"  Tracking URI: {tracking_uri}")
    
    # Configurar MLflow
    mlflow.set_tracking_uri(tracking_uri)
    print_check(True, f"Tracking URI configurado: {tracking_uri}")
    
    # Verificar URI configurado
    configured_uri = mlflow.get_tracking_uri()
    is_correct = configured_uri == tracking_uri
    print_check(is_correct, f"URI confirmado: {configured_uri}")
    
    return is_correct

def validate_experiment_exists():
    """Validar que el experimento específico existe"""
    print_header("3. VALIDAR EXPERIMENTO ESPECÍFICO")
    
    client = MlflowClient()
    
    print(f"\nBuscando experiment con ID: {REQUIRED_EXPERIMENT_ID}")
    
    try:
        # Intentar obtener el experimento por ID
        experiment = client.get_experiment_by_name(f"/Shared/Ultralytics")
        
        if experiment:
            print_check(True, f"Experimento encontrado: {experiment.name}")
            print(f"  ID: {experiment.experiment_id}")
            print(f"  Artifact Location: {experiment.artifact_location}")
            
            # Verificar que el ID coincida
            if str(experiment.experiment_id) == REQUIRED_EXPERIMENT_ID:
                print_check(True, f"ID del experimento coincide con el requerido")
                return True
            else:
                print_check(False, f"ID NO coincide. Encontrado: {experiment.experiment_id}, Requerido: {REQUIRED_EXPERIMENT_ID}")
                return False
        else:
            print_check(False, f"Experimento '/Shared/Ultralytics' no encontrado")
            
            # Listar experimentos disponibles
            print(f"\nExperimentos disponibles:")
            all_experiments = client.search_experiments()
            if all_experiments:
                for exp in all_experiments:
                    print(f"  - {exp.name} (ID: {exp.experiment_id})")
            else:
                print("  (ninguno)")
            
            return False
    
    except Exception as e:
        print_check(False, f"Error buscando experimento: {e}")
        return False

def validate_artifact_location():
    """Validar que el artifact_location es correcto"""
    print_header("4. VALIDAR ARTIFACT LOCATION")
    
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name("/Shared/Ultralytics")
        if experiment:
            actual_location = experiment.artifact_location
            
            print(f"\nArtifact Location Requerido:")
            print(f"  {REQUIRED_ARTIFACT_LOCATION}")
            print(f"\nArtifact Location Actual:")
            print(f"  {actual_location}")
            
            # Normalizar URIs para comparación
            actual_normalized = actual_location.replace('\\', '/').lower()
            required_normalized = REQUIRED_ARTIFACT_LOCATION.replace('\\', '/').lower()
            
            is_correct = actual_normalized == required_normalized
            print_check(is_correct, f"Artifact location coincide")
            
            if not is_correct:
                print(f"\n⚠️  MISMATCH EN ARTIFACT LOCATION")
                print(f"   Se debe usar: {REQUIRED_ARTIFACT_LOCATION}")
            
            return is_correct
    except Exception as e:
        print_check(False, f"Error validando artifact location: {e}")
        return False

def validate_write_permissions():
    """Validar que MLflow puede escribir en los directorios"""
    print_header("5. VALIDAR PERMISOS DE ESCRITURA")
    
    dirs_to_test = [
        (EXPERIMENT_DIR, "Artifact directory"),
    ]
    
    all_writable = True
    for dir_path, name in dirs_to_test:
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Intentar crear un archivo de prueba
        test_file = dir_path / '.mlflow_write_test'
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            test_file.unlink()  # Eliminar archivo de prueba
            print_check(True, f"{name}: {dir_path}")
        except Exception as e:
            print_check(False, f"{name}: {e}")
            all_writable = False
    
    return all_writable

def validate_test_run():
    """Crear un run de prueba para validar que todo funciona"""
    print_header("6. CREAR RUN DE PRUEBA")
    
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name("/Shared/Ultralytics")
        
        if not experiment:
            print_check(False, "No se puede crear run: experimento no encontrado")
            return False
        
        # Usar set_experiment con el nombre conocido (MLflow 3.9.0 no tiene set_experiment_by_id)
        try:
            mlflow.set_experiment("/Shared/Ultralytics")
            print_check(True, f"Experimento establecido: /Shared/Ultralytics (ID: {experiment.experiment_id})")
        except Exception as e:
            print_check(False, f"No se pudo establecer experimento: {e}")
            return False
        
        # Crear run de prueba
        with mlflow.start_run(run_name="validation_test_run"):
            mlflow.log_param("test_param", "validation")
            mlflow.log_metric("test_metric", 0.99)
            
            run_id = mlflow.active_run().info.run_id
            print_check(True, f"Run de prueba creado: {run_id}")
            
            # Verificar que se registró
            run = client.get_run(run_id)
            print_check(True, f"Run verificado en MLflow")
            print(f"  Status: {run.info.status}")
            print(f"  Experiment ID: {run.info.experiment_id}")
        
        return True
    
    except Exception as e:
        print_check(False, f"Error en run de prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_summary(results):
    """Imprimir resumen de validación"""
    print_header("RESUMEN DE VALIDACIÓN")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nResultados: {passed}/{total} validaciones pasadas")
    
    for check_name, result in results.items():
        print_check(result, check_name)
    
    if passed == total:
        print("\n" + "="*80)
        print("✓✓✓ TODAS LAS VALIDACIONES PASARON ✓✓✓")
        print("    MLflow está configurado correctamente para reentrenamiento")
        print("="*80)
        return True
    else:
        print("\n" + "="*80)
        print(f"✗✗✗ {total - passed} VALIDACIONES FALLARON ✗✗✗")
        print("    Revisa los errores arriba y ejecuta:")
        print("    python validate_mlflow_config.py")
        print("="*80)
        return False

def main():
    """Ejecutar todas las validaciones"""
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "VALIDACIÓN DE CONFIGURACIÓN DE MLFLOW - REENTRENAMIENTO".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")
    
    results = {}
    
    # Ejecutar validaciones
    results["Estructura de directorios"] = validate_directory_structure()
    results["Configuración de MLflow"] = validate_mlflow_configuration()
    results["Experimento específico existe"] = validate_experiment_exists()
    results["Artifact location correcto"] = validate_artifact_location()
    results["Permisos de escritura"] = validate_write_permissions()
    results["Run de prueba"] = validate_test_run()
    
    # Imprimir resumen
    success = print_summary(results)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
