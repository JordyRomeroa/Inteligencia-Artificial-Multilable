"""Test de múltiples reentrenamientos para verificar registro en MLflow"""
import os
import sys
from pathlib import Path

# Configurar paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.continuous_learning import ContinuousLearner
from app.mlflow_utils import setup_mlflow

def crear_correcciones_test(num_samples=3):
    """Crear correcciones de prueba"""
    import random
    test_images = list((PROJECT_ROOT / "data" / "images" / "test").glob("*.jpg"))[:num_samples]
    
    corrections = []
    for img_path in test_images:
        correction = {
            "image_path": str(img_path),
            "boxes": [{
                "class": "person",
                "bbox": [
                    random.uniform(100, 400),
                    random.uniform(100, 400),
                    random.uniform(200, 500),
                    random.uniform(200, 500)
                ],
                "confidence": 1.0
            }],
            "user_id": "test_usuario"
        }
        corrections.append(correction)
    
    return corrections

def main():
    print("=" * 80)
    print("TEST DE 11 REENTRENAMIENTOS CONSECUTIVOS")
    print("=" * 80)
    print()
    
    # Encontrar modelo base
    model_path = PROJECT_ROOT / "models" / "best_improved.pt"
    if not model_path.exists():
        print(f"❌ Modelo base no encontrado: {model_path}")
        return
    
    print(f"✓ Modelo base: {model_path}")
    print()
    
    # Número de reentrenamientos
    num_retrains = 11
    successful_retrains = 0
    failed_retrains = 0
    
    for i in range(1, num_retrains + 1):
        print("=" * 80)
        print(f"REENTRENAMIENTO {i}/{num_retrains}")
        print("=" * 80)
        
        try:
            # Inicializar learner
            learner = ContinuousLearner(
                base_model_path=str(model_path),
                project_root=PROJECT_ROOT
            )
            
            # Agregar correcciones de prueba
            print(f"[{i}] Agregando 3 correcciones de prueba...")
            corrections = crear_correcciones_test(num_samples=3)
            
            for correction in corrections:
                learner.add_corrected_sample(
                    image_path=correction["image_path"],
                    boxes=correction["boxes"],
                    user_id=correction["user_id"]
                )
            
            print(f"    ✓ Correcciones agregadas: {len(corrections)}")
            
            # Configurar MLflow
            experiment_id = "401576597529460193"
            
            # Reentrenar (1 epoch para velocidad)
            print(f"[{i}] Iniciando reentrenamiento rápido (1 epoch)...")
            result = learner.retrain(
                epochs=1,
                batch_size=8,
                experiment_id=experiment_id
            )
            
            if result:
                successful_retrains += 1
                print(f"✅ REENTRENAMIENTO {i} EXITOSO")
                print(f"   Modelo: {result}")
                print(f"   Progreso: {successful_retrains}/{i}")
            else:
                failed_retrains += 1
                print(f"❌ REENTRENAMIENTO {i} FALLÓ")
                print(f"   Progreso: {successful_retrains}/{i} (Fallos: {failed_retrains})")
            
        except Exception as e:
            failed_retrains += 1
            print(f"❌ ERROR EN REENTRENAMIENTO {i}: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Resumen final
    print("=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    print(f"Total de reentrenamientos: {num_retrains}")
    print(f"✅ Exitosos: {successful_retrains}")
    print(f"❌ Fallidos: {failed_retrains}")
    print(f"📊 Tasa de éxito: {(successful_retrains/num_retrains)*100:.1f}%")
    print()
    print("=" * 80)
    print("VERIFICAR EN MLFLOW UI:")
    print("http://localhost:5001")
    print(f"Experimento ID: 401576597529460193")
    print("Pestaña: Models (dentro del experimento)")
    print(f"Deberías ver el modelo 'yolo_reentrenado' con versiones 2-{2+successful_retrains-1}")
    print("=" * 80)

if __name__ == "__main__":
    main()
