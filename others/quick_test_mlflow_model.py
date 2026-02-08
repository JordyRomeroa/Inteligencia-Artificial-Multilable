#!/usr/bin/env python
"""
Test Rápido - Verificar Modelo en MLflow
=========================================

Test mínimo para confirmar que el modelo se registra correctamente en MLflow
y aparece visible en la UI.
"""

import sys
from pathlib import Path

# Add app to path  
sys.path.insert(0, str(Path(__file__).parent / 'app'))

def test_model_registration():
    """Test que el modelo se registre correctamente en MLflow"""
    
    print("\n" + "="*80)
    print("TEST RÁPIDO - REGISTRO DE MODELO EN MLFLOW")
    print("="*80)
    
    from continuous_learning import ContinuousLearner
    
    PROJECT_ROOT = Path(__file__).parent
    model_path = PROJECT_ROOT / 'models' / 'best_improved.pt'
    
    if not model_path.exists():
        print(f"\n❌ ERROR: Modelo no encontrado en {model_path}")
        return False
    
    print(f"\n✓ Modelo base encontrado: {model_path}")
    
    # Inicializar ContinuousLearner
    print("\n[1] Inicializando ContinuousLearner...")
    learner = ContinuousLearner(str(model_path), project_root=PROJECT_ROOT)
    print(f"    ✓ Versión actual: v{learner.current_version}")
    
    # Agregar solo 5 correcciones para test rápido
    print(f"\n[2] Agregando 5 correcciones para test rápido...")
    
    data_dir = PROJECT_ROOT / 'data'
    images_dir = data_dir / 'images' / 'test'
    
    if not images_dir.exists() or not list(images_dir.glob('*.jpg')):
        print(f"❌ No se encontraron imágenes en {images_dir}")
        return False
    
    test_images = list(images_dir.glob('*.jpg'))[:5]
    
    for img_path in test_images:
        learner.add_corrected_sample(
            image_path=str(img_path),
            boxes=[
                {'class': 'person', 'bbox': [100, 100, 200, 200], 'confidence': 1.0}
            ],
            user_id='mlflow_test'
        )
    
    print(f"    ✓ Agregadas {len(test_images)} correcciones")
    
    # Reentrenar con 1 epoch solo para test
    print(f"\n[3] Iniciando reentrenamiento (1 epoch - test rápido)...")
    print(f"    Experiment ID: 401576597529460193")
    
    result = learner.retrain(
        epochs=1,  # Solo 1 epoch para test rápido
        batch_size=8,
        experiment_id='401576597529460193'
    )
    
    if result.get('success'):
        print(f"\n✅ REENTRENAMIENTO EXITOSO!")
        print(f"   - Versión: {result.get('version')}")
        print(f"   - Modelo: {result.get('model_path')}")
        print(f"   - MLflow Run ID: {result.get('mlflow_run_id')}")
        
        # Verificar que el modelo está en artifacts
        print(f"\n[4] Verificando registro en MLflow...")
        mlflow_dir = PROJECT_ROOT / 'runs' / 'mlflow' / '401576597529460193'
        
        # Buscar el run reciente
        import os
        from datetime import datetime
        
        run_dirs = []
        for item in mlflow_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Buscar directorios de runs (tienen IDs hexadecimales)
                if len(item.name) == 32:  # MLflow run IDs son 32 caracteres hex
                    run_dirs.append(item)
        
        if run_dirs:
            # Ordenar por fecha de modificación (más reciente primero)
            run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_run = run_dirs[0]
            
            print(f"    ✓ Run más reciente: {latest_run.name}")
            
            # Buscar modelo en artifacts
            artifacts_dir = latest_run / 'artifacts'
            if artifacts_dir.exists():
                models_in_artifacts = list(artifacts_dir.rglob('*.pt'))
                if models_in_artifacts:
                    print(f"    ✓ Modelos encontrados en artifacts: {len(models_in_artifacts)}")
                    for model_artifact in models_in_artifacts:
                        size_mb = model_artifact.stat().st_size / 1024 / 1024
                        rel_path = model_artifact.relative_to(artifacts_dir)
                        print(f"      - {rel_path}: {size_mb:.2f} MB")
                else:
                    print(f"    ⚠️  No se encontraron modelos .pt en artifacts")
            else:
                print(f"    ⚠️  No se encontró directorio artifacts")
        
        print(f"\n" + "="*80)
        print(f"✅ TEST COMPLETADO - Verifica en MLflow UI:")
        print(f"   http://localhost:5001")
        print(f"   Busca experiment: 401576597529460193")
        print(f"   Run ID: {result.get('mlflow_run_id')}")
        print(f"="*80)
        
        return True
    else:
        print(f"\n❌ REENTRENAMIENTO FALLÓ")
        print(f"   Error: {result.get('error')}")
        return False

if __name__ == '__main__':
    test_model_registration()
