#!/usr/bin/env python
"""
VERIFICATION CHECKLIST - Verifica que EVERY RETRAINING está correcto
=====================================================================

Este script verifica que después de un reentrenamiento:
1. El modelo se guardó en models/retrained_vX.pt
2. MLflow tiene el run en experiment 401576597529460193
3. Los artifacts están en la ruta correcta
4. Los datos de reentrenamiento fueron guardados
5. Las métricas se registraron

Uso:
    python verification_checklist.py <run_id>
    
Ejemplo:
    python verification_checklist.py abc123def456xyz
"""

import sys
import json
from pathlib import Path
from mlflow.tracking import MlflowClient
import mlflow

def check_local_files():
    """Verificar archivos guardados localmente"""
    print("\n" + "="*80)
    print("1. VERIFICAR ARCHIVOS LOCALES")
    print("="*80)
    
    PROJECT_ROOT = Path(__file__).parent
    MODELS_DIR = PROJECT_ROOT / 'models'
    RUNS_DIR = PROJECT_ROOT / 'runs'
    
    checks = []
    
    # Verificar que existen modelos reentrenados
    retrained_models = list(MODELS_DIR.glob('retrained_v*.pt'))
    print(f"\n✓ Modelos reentrenados encontrados: {len(retrained_models)}")
    for model in sorted(retrained_models):
        size_mb = model.stat().st_size / 1024 / 1024
        print(f"  - {model.name} ({size_mb:.2f} MB)")
        checks.append((f"Modelo {model.name} existe", True))
    
    # Verificar metadata files
    metadata_files = list(MODELS_DIR.glob('*_metadata.json'))
    print(f"\n✓ Metadata files: {len(metadata_files)}")
    for meta in sorted(metadata_files):
        with open(meta, 'r') as f:
            data = json.load(f)
        print(f"  - {meta.name}")
        print(f"    Version type: {data.get('version_type')}")
        print(f"    Timestamp: {data.get('timestamp')}")
        checks.append((f"Metadata {meta.name} existe", True))
    
    # Verificar MLflow directory
    mlflow_dir = RUNS_DIR / 'mlflow' / '401576597529460193'
    if mlflow_dir.exists():
        print(f"\n✓ Directorio MLflow existe: {mlflow_dir}")
        checks.append(("MLflow directory 401576597529460193", True))
    else:
        print(f"\n✗ Directorio MLflow NO existe: {mlflow_dir}")
        checks.append(("MLflow directory 401576597529460193", False))
    
    return checks

def check_mlflow_runs():
    """Verificar runs en MLflow"""
    print("\n" + "="*80)
    print("2. VERIFICAR RUNS EN MLFLOW")
    print("="*80)
    
    PROJECT_ROOT = Path(__file__).parent
    mlflow_dir = PROJECT_ROOT / 'runs' / 'mlflow'
    
    # Configurar tracking URI
    mlflow_path = str(mlflow_dir).replace('\\', '/')
    if len(mlflow_path) > 1 and mlflow_path[1] == ':':
        tracking_uri = f"file:///{mlflow_path}"
    else:
        tracking_uri = f"file://{mlflow_path}"
    
    mlflow.set_tracking_uri(tracking_uri)
    
    client = MlflowClient()
    checks = []
    
    try:
        # Buscar experiment
        experiment_id = '401576597529460193'
        experiment = client.get_experiment(experiment_id)
        
        if experiment:
            print(f"\n✓ Experimento encontrado:")
            print(f"  ID: {experiment.experiment_id}")
            print(f"  Name: {experiment.name}")
            print(f"  Artifact Location: {experiment.artifact_location}")
            checks.append(("Experiment 401576597529460193 existe", True))
            
            # Listar runs
            runs = client.search_runs(experiment_id)
            print(f"\n✓ Runs encontrados: {len(runs)}")
            
            for run in runs[-5:]:  # Últimos 5 runs
                print(f"\n  Run ID: {run.info.run_id}")
                print(f"  Status: {run.info.status}")
                print(f"  Start time: {run.info.start_time}")
                
                # Verificar tags
                has_retraining_tag = 'type' in run.data.tags and run.data.tags['type'] == 'retraining'
                print(f"  Tag 'type=retraining': {'✓' if has_retraining_tag else '✗'}")
                checks.append((f"Run {run.info.run_id} tiene tag retraining", has_retraining_tag))
                
                # Contar artifacts
                artifacts = client.list_artifacts(run.info.run_id)
                print(f"  Artifacts: {len(list(artifacts))}")
                
                # Verificar artifacts clave
                artifact_paths = list_artifacts_recursive(client, run.info.run_id)
                
                required_artifacts = [
                    'models/retrained_',
                    'retraining_dataset/corrections_applied.json',
                    'retraining_dataset/retraining_dataset_metadata.json'
                ]
                
                for required in required_artifacts:
                    found = any(required in path for path in artifact_paths)
                    status = "✓" if found else "✗"
                    print(f"    {status} {required}")
                    checks.append((f"Run {run.info.run_id} tiene artifact {required}", found))
        else:
            print(f"✗ Experiment {experiment_id} NO encontrado")
            checks.append(("Experiment 401576597529460193 existe", False))
            
    except Exception as e:
        print(f"✗ Error accediendo MLflow: {e}")
        checks.append(("MLflow accessible", False))
    
    return checks

def list_artifacts_recursive(client, run_id, prefix=''):
    """Listar artifacts recursivamente"""
    artifacts = []
    try:
        items = client.list_artifacts(run_id, prefix)
        for item in items:
            if item.is_dir:
                artifacts.extend(list_artifacts_recursive(client, run_id, item.path))
            else:
                artifacts.append(item.path)
    except:
        pass
    return artifacts

def check_metrics():
    """Verificar que se registraron métricas"""
    print("\n" + "="*80)
    print("3. VERIFICAR MÉTRICAS REGISTRADAS")
    print("="*80)
    
    PROJECT_ROOT = Path(__file__).parent
    mlflow_dir = PROJECT_ROOT / 'runs' / 'mlflow'
    
    mlflow_path = str(mlflow_dir).replace('\\', '/')
    if len(mlflow_path) > 1 and mlflow_path[1] == ':':
        tracking_uri = f"file:///{mlflow_path}"
    else:
        tracking_uri = f"file://{mlflow_path}"
    
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    checks = []
    
    try:
        experiment_id = '401576597529460193'
        runs = client.search_runs(experiment_id)
        
        if runs:
            latest_run = runs[-1]
            metrics = latest_run.data.metrics
            
            print(f"\nÚltimo run: {latest_run.info.run_id}")
            print(f"Métricas registradas: {len(metrics)}")
            
            required_metrics = [
                ('mAP50', 'Métrica de entrenamiento'),
                ('mAP50_95', 'Métrica de validación'),
                ('precision', 'Métrica de precisión'),
                ('recall', 'Métrica de recall'),
                ('val_mAP50', 'Métrica de validación'),
                ('retraining_dataset_samples', 'Métrica de dataset')
            ]
            
            for metric_name, description in required_metrics:
                has_metric = metric_name in metrics
                status = "✓" if has_metric else "✗"
                value = f" = {metrics[metric_name]:.4f}" if has_metric else ""
                print(f"  {status} {metric_name}: {description}{value}")
                checks.append((f"Métrica {metric_name} registrada", has_metric))
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    return checks

def check_reproducibility():
    """Verificar que es reproducible"""
    print("\n" + "="*80)
    print("4. VERIFICAR REPRODUCIBILIDAD")
    print("="*80)
    
    PROJECT_ROOT = Path(__file__).parent
    mlflow_dir = PROJECT_ROOT / 'runs' / 'mlflow'
    
    mlflow_path = str(mlflow_dir).replace('\\', '/')
    if len(mlflow_path) > 1 and mlflow_path[1] == ':':
        tracking_uri = f"file:///{mlflow_path}"
    else:
        tracking_uri = f"file://{mlflow_path}"
    
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    checks = []
    
    try:
        experiment_id = '401576597529460193'
        runs = client.search_runs(experiment_id)
        
        if runs:
            latest_run = runs[-1]
            
            # Verificar parámetros
            params = latest_run.data.params
            print(f"\nParámetros registrados: {len(params)}")
            
            required_params = [
                'train_epochs',
                'train_batch_size',
                'train_num_corrections',
                'model_name'
            ]
            
            for param_name in required_params:
                has_param = param_name in params
                status = "✓" if has_param else "✗"
                value = f" = {params[param_name]}" if has_param else ""
                print(f"  {status} {param_name}{value}")
                checks.append((f"Parámetro {param_name} registrado", has_param))
            
            # Verificar tags
            tags = latest_run.data.tags
            print(f"\nTags registrados:")
            
            required_tags = [
                ('type', 'retraining'),
                ('model_type', 'continuous_learning'),
                ('training_type', 'incremental_retrain')
            ]
            
            for tag_name, expected_value in required_tags:
                has_tag = tag_name in tags
                value_match = tags.get(tag_name) == expected_value if has_tag else False
                status = "✓" if has_tag and value_match else "✗"
                actual = f" = {tags.get(tag_name)}" if has_tag else " (no encontrado)"
                print(f"  {status} {tag_name}{actual}")
                checks.append((f"Tag {tag_name}={expected_value}", has_tag and value_match))
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    return checks

def print_summary(all_checks):
    """Imprimir resumen de verificaciones"""
    print("\n" + "="*80)
    print("RESUMEN DE VERIFICACIONES")
    print("="*80)
    
    passed = sum(1 for _, result in all_checks if result)
    total = len(all_checks)
    
    print(f"\nResultados: {passed}/{total} verificaciones pasadas")
    
    failed_checks = [(name, result) for name, result in all_checks if not result]
    
    if failed_checks:
        print(f"\n⚠️  {len(failed_checks)} verificaciones fallaron:")
        for name, _ in failed_checks:
            print(f"  ✗ {name}")
    else:
        print("\n✅ TODAS LAS VERIFICACIONES PASARON")
        print("   El reentrenamiento se guardó correctamente en MLflow")
    
    return passed == total

def main():
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "VERIFICACIÓN DE REENTRENAMIENTO - MLflow".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")
    
    all_checks = []
    
    # Ejecutar verificaciones
    all_checks.extend(check_local_files())
    all_checks.extend(check_mlflow_runs())
    all_checks.extend(check_metrics())
    all_checks.extend(check_reproducibility())
    
    # Resumen
    success = print_summary(all_checks)
    
    print("\n" + "="*80)
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
