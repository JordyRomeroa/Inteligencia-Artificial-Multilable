"""Verificar modelos en MLflow Model Registry"""
import os
from pathlib import Path
from mlflow import MlflowClient

# Configurar tracking URI
tracking_uri = Path(__file__).parent / "runs" / "mlflow"
os.environ["MLFLOW_TRACKING_URI"] = f"file:///{tracking_uri}"

print("=" * 80)
print("VERIFICANDO MLFLOW MODEL REGISTRY")
print("=" * 80)
print(f"Tracking URI: {tracking_uri}")
print()

client = MlflowClient()

# Listar todos los modelos registrados
try:
    registered_models = client.search_registered_models()
    
    print(f"📋 Modelos encontrados en Registry: {len(registered_models)}")
    print()
    
    for rm in registered_models:
        print(f"📦 Modelo: {rm.name}")
        print(f"   Descripción: {rm.description}")
        print(f"   Creado: {rm.creation_timestamp}")
        print(f"   Última actualización: {rm.last_updated_timestamp}")
        
        # Obtener versiones del modelo
        versions = client.search_model_versions(f"name='{rm.name}'")
        print(f"   Versiones: {len(versions)}")
        
        for mv in versions:
            print(f"   └─ Versión {mv.version}:")
            print(f"      Source: {mv.source}")
            print(f"      Run ID: {mv.run_id}")
            print(f"      Status: {mv.status}")
            if mv.tags:
                print(f"      Tags: {mv.tags}")
        print()
        
except Exception as e:
    print(f"❌ Error al verificar registry: {e}")
    print(f"   Tipo: {type(e).__name__}")
    import traceback
    traceback.print_exc()

print("=" * 80)
