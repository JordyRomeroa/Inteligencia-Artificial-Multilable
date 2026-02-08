#!/usr/bin/env python
"""
Mostrar Modelos Reentrenados en MLflow
=======================================

Este script muestra dónde encontrar los modelos reentrenados en MLflow UI.
"""

from pathlib import Path
import json

def show_mlflow_models_location():
    """Muestra la ubicación de los modelos en MLflow"""
    
    print("\n" + "="*80)
    print("UBICACIÓN DE MODELOS REENTRENADOS EN MLFLOW")
    print("="*80)
    
    project_root = Path(__file__).parent
    mlflow_exp_dir = project_root / 'runs' / 'mlflow' / '401576597529460193'
    
    if not mlflow_exp_dir.exists():
        print(f"\n❌ Directorio MLflow no encontrado: {mlflow_exp_dir}")
        return
    
    print(f"\n✓ Directorio MLflow: {mlflow_exp_dir}")
    
    # Buscar todos los runs
    runs = []
    for item in mlflow_exp_dir.iterdir():
        if item.is_dir() and len(item.name) == 32:  # MLflow run IDs
            # Buscar meta.json para obtener info del run
            meta_file = item / 'meta.json'
            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)
                    
                    # Buscar modelos en artifacts
                    artifacts_dir = item / 'artifacts'
                    models = []
                    if artifacts_dir.exists():
                        models = list(artifacts_dir.rglob('retrained_v*.pt'))
                    
                    if models:  # Solo mostrar runs con modelos reentrenados
                        runs.append({
                            'run_id': item.name,
                            'run_name': meta.get('run_name', 'N/A'),
                            'start_time': meta.get('start_time', 0),
                            'status': meta.get('status', 'UNKNOWN'),
                            'models': models,
                            'artifacts_dir': artifacts_dir
                        })
                except Exception as e:
                    pass
    
    # Ordenar por tiempo (más reciente primero)
    runs.sort(key=lambda x: x['start_time'], reverse=True)
    
    if not runs:
        print("\n⚠️  No se encontraron runs con modelos reentrenados")
        return
    
    print(f"\n✓ Encontrados {len(runs)} runs con modelos reentrenados\n")
    print("="*80)
    
    for i, run in enumerate(runs[:5], 1):  # Mostrar últimos 5
        print(f"\n[{i}] RUN: {run['run_name']}")
        print(f"    Run ID: {run['run_id']}")
        print(f"    Status: {run['status']}")
        print(f"    Modelos en artifacts:")
        
        for model in run['models']:
            rel_path = model.relative_to(run['artifacts_dir'])
            size_mb = model.stat().st_size / 1024 / 1024
            print(f"      ✓ {rel_path}")
            print(f"        Tamaño: {size_mb:.2f} MB")
            print(f"        Ruta completa: {model}")
    
    print("\n" + "="*80)
    print("📍 CÓMO VER EN MLFLOW UI:")
    print("="*80)
    print("""
1. Abre: http://localhost:5001

2. En el menú lateral izquierdo, click en "Runs" 

3. Busca el experiment:
   c:\\Users\\jordy\\OneDrive\\Desktop\\iaaaa\\iajordy2\\runs\\train

4. Haz click en cualquiera de los runs recientes (ejemplo):
   """ + (runs[0]['run_id'] if runs else 'N/A') + """

5. En la página del run, baja hasta "Artifacts" 

6. Verás las carpetas:
   📁 models/          <- AQUÍ ESTÁN TUS MODELOS .pt
   📁 plots/           <- Gráficas de entrenamiento
   📁 config/          <- Configuraciones
   📄 retraining_dataset_metadata.json
   📄 corrections_applied.json
   📄 data.yaml

7. Click en models/ -> retrained_vX.pt para descargar

""")
    
    print("="*80)
    print("💡 NOTA IMPORTANTE:")
    print("="*80)
    print("""
La pestaña "Models" en MLflow muestra el MODEL REGISTRY.

Tus modelos reentrenados están guardados como ARTIFACTS en los RUNS,
no como modelos registrados en el Registry.

Para verlos necesitas ir a:
  Runs -> [Seleccionar run] -> Artifacts -> models/

Si quieres que aparezcan en "Models" también, puedo modificar el código
para registrarlos en el Model Registry.
""")

if __name__ == '__main__':
    show_mlflow_models_location()
