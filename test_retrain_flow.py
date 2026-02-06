#!/usr/bin/env python
"""
QUICK START - Prueba del Reentrenamiento con MLflow Corregido
===========================================================

Este script permite probar el flujo de reentrenamiento SIN necesidad
de hacer correcciones manuales desde el frontend.

Uso:
    python test_retrain_flow.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add app to path
sys.path.insert(0, str(Path(__file__).parent / 'app'))

def test_retrain_flow():
    """Test completo del flujo de reentrenamiento"""
    
    print("\n" + "="*80)
    print("TEST FLUJO DE REENTRENAMIENTO - MLflow Corregido")
    print("="*80)
    
    from continuous_learning import ContinuousLearner
    
    PROJECT_ROOT = Path(__file__).parent
    model_path = PROJECT_ROOT / 'models' / 'best_improved.pt'
    
    if not model_path.exists():
        print(f"\n❌ ERROR: Modelo no encontrado en {model_path}")
        print("   Ejecuta primero: python notebooks/03_training.ipynb")
        return False
    
    print(f"\n✓ Modelo encontrado: {model_path}")
    
    # Inicializar ContinuousLearner
    print("\n[1] Inicializando ContinuousLearner...")
    learner = ContinuousLearner(str(model_path), project_root=PROJECT_ROOT)
    print(f"    ✓ Learner inicializado. Versión actual: v{learner.current_version}")
    
    # Simular 10 correcciones manuales para poder reentrenar
    print(f"\n[2] Agregando correcciones simuladas para test...")
    
    import json
    
    # Usar las primeras imágenes disponibles
    data_dir = PROJECT_ROOT / 'data'
    images_dir = None
    
    # Buscar directorio con imágenes
    for img_dir in [data_dir / 'images' / 'test', 
                    data_dir / 'images',
                    data_dir / 'retrain_v1' / 'images']:
        if img_dir.exists() and list(img_dir.glob('*.jpg')):
            images_dir = img_dir
            break
    
    if not images_dir:
        print("   ⚠️  No se encontraron imágenes para simular correcciones")
        print("   Creando correcciones simuladas sin imágenes...")
        
        # Crear correcciones simuladas sin imágenes
        for i in range(10):
            correction = {
                'class': ['person', 'car', 'dog'][i % 3],
                'bbox': [100 + i*10, 100 + i*10, 200 + i*10, 300 + i*10],
                'confidence': 0.4  # Baja confianza
            }
            # Agregar directamente a corrected_samples
            learner.corrected_samples.append({
                'timestamp': datetime.now().isoformat(),
                'image_path': f'simulated_{i}.jpg',
                'boxes': [correction],
                'user_id': 'test_user'
            })
    else:
        print(f"   ✓ Encontrado directorio: {images_dir}")
        image_files = list(images_dir.glob('*.jpg'))[:10]
        
        for idx, img_file in enumerate(image_files):
            correction = {
                'class': ['person', 'car', 'dog'][idx % 3],
                'bbox': [100, 100, 200, 300],
                'confidence': 0.4
            }
            learner.add_corrected_sample(
                str(img_file),
                [correction],
                user_id='test_user'
            )
    
    print(f"    ✓ Agregadas {len(learner.corrected_samples)} correcciones")
    
    # Verificar que hay suficientes correcciones
    if len(learner.corrected_samples) < 5:
        print(f"\n❌ ERROR: {len(learner.corrected_samples)} correcciones < 5 requeridas")
        return False
    
    print(f"    ✓ Correcciones suficientes para reentrenar")
    
    # Realizar reentrenamiento
    print(f"\n[3] Iniciando REENTRENAMIENTO CON MLFLOW CORRECTO...")
    print(f"    Experiment ID: 401576597529460193")
    print(f"    Epochs: 2 (reducido para test)")
    print(f"    Muestras: {len(learner.corrected_samples)}")
    
    try:
        result = learner.retrain(
            epochs=2,  # Reducido para test rápido
            batch_size=8,
            patience=5,
            experiment_id='401576597529460193'
        )
        
        if result['success']:
            print(f"\n✅ REENTRENAMIENTO EXITOSO!")
            print(f"   - Nueva versión: {result['version']}")
            print(f"   - Modelo: {result['model_path']}")
            print(f"   - MLflow Run ID: {result['mlflow_run_id']}")
            print(f"   - Experiment ID: {result['experiment_id']}")
            print(f"   - Métricas: {result.get('metrics', {})}")
            
            print(f"\n[4] Verificando que MLflow guardó correctamente...")
            
            # Verificar que artifacts se guardaron
            mlflow_dir = PROJECT_ROOT / 'runs' / 'mlflow' / '401576597529460193'
            
            if mlflow_dir.exists():
                print(f"    ✓ Directorio MLflow existe: {mlflow_dir}")
                
                # Contar artifacts
                mlruns_dir = mlflow_dir / 'mlruns' / '401576597529460193'
                if mlruns_dir.exists():
                    runs = list(mlruns_dir.glob('*'))
                    print(f"    ✓ Runs encontrados: {len(runs)}")
                    
                    if runs:
                        latest_run = sorted(runs)[-1]
                        artifacts_dir = latest_run / 'artifacts'
                        if artifacts_dir.exists():
                            files = list(artifacts_dir.rglob('*'))
                            print(f"    ✓ Artifacts encontrados: {len([f for f in files if f.is_file()])} archivos")
                            
                            # Listar artifacts clave
                            important_files = [
                                'models/retrained_v*.pt',
                                'retraining_dataset/corrections_applied.json',
                                'retraining_dataset/data.yaml'
                            ]
                            
                            for pattern in important_files:
                                matching = list(artifacts_dir.rglob(pattern.replace('*', '')))
                                for f in matching:
                                    print(f"      ✓ {f.relative_to(artifacts_dir)}")
            else:
                print(f"    ⚠️  Directorio MLflow no encontrado: {mlflow_dir}")
            
            print(f"\n" + "="*80)
            print("✅ TEST COMPLETADO EXITOSAMENTE")
            print("="*80)
            print(f"\nProximos pasos:")
            print(f"1. Ejecutar: python validate_mlflow_config.py")
            print(f"2. Ejecutar: mlflow ui --backend-store-uri file:///{PROJECT_ROOT / 'runs' / 'mlflow'} --port 5001")
            print(f"3. Ver en navegador: http://localhost:5001")
            print(f"4. Verificar experiment 401576597529460193 con run recién creado")
            
            return True
        else:
            print(f"\n❌ REENTRENAMIENTO FALLÓ")
            print(f"   Error: {result.get('error')}")
            print(f"   Razón: {result.get('reason')}")
            return False
            
    except Exception as e:
        print(f"\n❌ EXCEPCIÓN: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_retrain_flow()
    sys.exit(0 if success else 1)
