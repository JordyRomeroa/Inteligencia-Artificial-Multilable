"""Verificar modelos vinculados al experimento de reentrenamiento"""
import os
from pathlib import Path
from mlflow import MlflowClient

# Configurar tracking URI
tracking_uri = Path(__file__).parent / "runs" / "mlflow"
os.environ["MLFLOW_TRACKING_URI"] = f"file:///{tracking_uri}"

print("=" * 80)
print("VERIFICANDO MODELOS EN EXPERIMENTO 401576597529460193")
print("=" * 80)

client = MlflowClient()

# ID del experimento
experiment_id = "401576597529460193"

# Verificar experimento
experiment = client.get_experiment(experiment_id)
print(f"📂 Experimento: {experiment.name}")
print(f"   ID: {experiment.experiment_id}")
print(f"   Artifact Location: {experiment.artifact_location}")
print()

# Obtener todos los runs del experimento
runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=10)
print(f"🔄 Runs recientes en este experimento: {len(runs)}")
print()

for run in runs[:5]:  # Mostrar últimos 5
    print(f"Run ID: {run.info.run_id}")
    print(f"  Nombre: {run.data.tags.get('mlflow.runName', 'N/A')}")
    print(f"  Type: {run.data.tags.get('type', 'N/A')}")
    print(f"  Version: {run.data.tags.get('version', 'N/A')}")
    
    # Verificar si hay modelo logged
    artifacts = client.list_artifacts(run.info.run_id, path="models")
    if artifacts:
        print(f"  📦 Artifacts en models/:")
        for artifact in artifacts:
            print(f"     - {artifact.path}")
    print()

# Buscar modelos registrados que vienen de este experimento
print("=" * 80)
print("MODELOS REGISTRADOS DEL EXPERIMENTO")
print("=" * 80)

all_models = client.search_registered_models()
experiment_models = []

for model in all_models:
    versions = client.search_model_versions(f"name='{model.name}'")
    for version in versions:
        # Verificar si el run pertenece a nuestro experimento
        try:
            run_info = client.get_run(version.run_id)
            if run_info.info.experiment_id == experiment_id:
                experiment_models.append({
                    'model_name': model.name,
                    'version': version.version,
                    'run_id': version.run_id,
                    'source': version.source,
                    'status': version.status,
                    'tags': version.tags
                })
        except:
            pass

if experiment_models:
    print(f"✅ Modelos encontrados: {len(experiment_models)}")
    for em in experiment_models:
        print(f"\n📦 {em['model_name']} v{em['version']}")
        print(f"   Run ID: {em['run_id']}")
        print(f"   Source: {em['source']}")
        print(f"   Status: {em['status']}")
        if em['tags']:
            print(f"   Tags: {em['tags']}")
else:
    print("❌ No se encontraron modelos registrados de este experimento")

print("=" * 80)
