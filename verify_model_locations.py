#!/usr/bin/env python
"""
Verificación de Ubicaciones de Modelos Reentrenados
====================================================

Script para verificar que los modelos se guardan en todas las ubicaciones necesarias:
1. models/ - Modelos locales versionados
2. runs/train/ - Copia directa para acceso rápido
3. runs/train/retrain_vX/ - Directorio completo de entrenamiento (weights, plots, configs)
4. MLflow artifacts - Modelos registrados en experimento MLflow

Uso:
    python verify_model_locations.py [version]
    
Ejemplo:
    python verify_model_locations.py 8
"""

import sys
from pathlib import Path
import json

def verify_model_locations(version=None):
    """Verifica todas las ubicaciones de modelos"""
    
    project_root = Path(__file__).parent
    
    print("\n" + "="*80)
    print("VERIFICACIÓN DE UBICACIONES DE MODELOS REENTRENADOS")
    print("="*80)
    
    if version is None:
        # Buscar la última versión
        models_dir = project_root / 'models'
        versions = []
        for model_file in models_dir.glob('retrained_v*.pt'):
            v = model_file.stem.replace('retrained_v', '')
            if v.isdigit():
                versions.append(int(v))
        
        if not versions:
            print("\n❌ No se encontraron modelos reentrenados")
            return False
        
        version = max(versions)
        print(f"\n✓ Última versión encontrada: v{version}")
    
    version_str = str(version)
    results = {}
    
    # =========================================================================
    # 1. UBICACIÓN: models/retrained_vX.pt
    # =========================================================================
    print(f"\n[1] VERIFICANDO: models/retrained_v{version}.pt")
    models_path = project_root / 'models' / f'retrained_v{version}.pt'
    
    if models_path.exists():
        size_mb = models_path.stat().st_size / 1024 / 1024
        print(f"    ✓ ENCONTRADO: {models_path}")
        print(f"    ✓ Tamaño: {size_mb:.2f} MB")
        print(f"    ✓ Modificado: {models_path.stat().st_mtime}")
        results['models_dir'] = True
        
        # Verificar metadata
        metadata_path = models_path.parent / f'retrained_v{version}_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"    ✓ Metadata encontrado:")
            print(f"      - Versión: {metadata.get('version')}")
            print(f"      - Tipo: {metadata.get('training_type')}")
            print(f"      - Muestras: {metadata.get('num_training_samples')}")
    else:
        print(f"    ✗ NO ENCONTRADO: {models_path}")
        results['models_dir'] = False
    
    # =========================================================================
    # 2. UBICACIÓN: runs/train/retrained_vX.pt
    # =========================================================================
    print(f"\n[2] VERIFICANDO: runs/train/retrained_v{version}.pt")
    runs_train_path = project_root / 'runs' / 'train' / f'retrained_v{version}.pt'
    
    if runs_train_path.exists():
        size_mb = runs_train_path.stat().st_size / 1024 / 1024
        print(f"    ✓ ENCONTRADO: {runs_train_path}")
        print(f"    ✓ Tamaño: {size_mb:.2f} MB")
        results['runs_train_copy'] = True
    else:
        print(f"    ✗ NO ENCONTRADO: {runs_train_path}")
        results['runs_train_copy'] = False
    
    # =========================================================================
    # 3. UBICACIÓN: runs/train/retrain_vX/ (directorio completo)
    # =========================================================================
    print(f"\n[3] VERIFICANDO: runs/train/retrain_v{version}/ (directorio completo)")
    training_dir = project_root / 'runs' / 'train' / f'retrain_v{version}'
    
    if training_dir.exists():
        print(f"    ✓ ENCONTRADO: {training_dir}")
        
        # Verificar weights/
        weights_dir = training_dir / 'weights'
        if weights_dir.exists():
            print(f"    ✓ weights/ encontrado:")
            for weight_file in weights_dir.glob('*.pt'):
                size_mb = weight_file.stat().st_size / 1024 / 1024
                print(f"      - {weight_file.name}: {size_mb:.2f} MB")
            results['training_weights'] = True
        
        # Verificar plots
        plots = list(training_dir.glob('*.png'))
        if plots:
            print(f"    ✓ Plots encontrados: {len(plots)}")
            for plot in plots[:3]:  # Mostrar primeros 3
                print(f"      - {plot.name}")
            if len(plots) > 3:
                print(f"      ... y {len(plots) - 3} más")
            results['training_plots'] = True
        
        # Verificar args.yaml
        args_file = training_dir / 'args.yaml'
        if args_file.exists():
            print(f"    ✓ args.yaml encontrado")
            results['training_config'] = True
    else:
        print(f"    ✗ NO ENCONTRADO: {training_dir}")
        results['training_dir'] = False
    
    # =========================================================================
    # 4. UBICACIÓN: MLflow artifacts
    # =========================================================================
    print(f"\n[4] VERIFICANDO: MLflow artifacts (experiment 401576597529460193)")
    mlflow_exp_dir = project_root / 'runs' / 'mlflow' / '401576597529460193'
    
    if mlflow_exp_dir.exists():
        print(f"    ✓ Directorio MLflow encontrado: {mlflow_exp_dir}")
        
        # Buscar mlruns dentro del directorio del experimento
        mlruns_dirs = list(mlflow_exp_dir.rglob('mlruns'))
        
        if mlruns_dirs:
            print(f"    ✓ MLflow runs encontrados:")
            
            # Buscar runs con modelo registrado
            model_artifacts = []
            for mlrun_dir in mlruns_dirs:
                for artifact_dir in mlrun_dir.rglob('artifacts'):
                    # Buscar modelos en artifacts
                    models_in_artifacts = list(artifact_dir.rglob('*.pt'))
                    model_artifacts.extend(models_in_artifacts)
            
            if model_artifacts:
                print(f"    ✓ Modelos en artifacts: {len(model_artifacts)}")
                for model_artifact in model_artifacts[:3]:
                    size_mb = model_artifact.stat().st_size / 1024 / 1024
                    print(f"      - {model_artifact.relative_to(mlflow_exp_dir)}: {size_mb:.2f} MB")
                results['mlflow_artifacts'] = True
            else:
                print(f"    ⚠️  No se encontraron modelos .pt en artifacts")
                results['mlflow_artifacts'] = False
        else:
            print(f"    ⚠️  No se encontró directorio mlruns")
            results['mlflow_artifacts'] = False
    else:
        print(f"    ✗ NO ENCONTRADO: {mlflow_exp_dir}")
        results['mlflow_artifacts'] = False
    
    # =========================================================================
    # RESUMEN
    # =========================================================================
    print("\n" + "="*80)
    print("RESUMEN DE VERIFICACIÓN")
    print("="*80)
    
    total_checks = len(results)
    passed_checks = sum(1 for v in results.values() if v)
    
    print(f"\nResultados: {passed_checks}/{total_checks} ubicaciones verificadas")
    
    for key, value in results.items():
        status = "✓" if value else "✗"
        print(f"{status} {key}")
    
    if passed_checks == total_checks:
        print("\n" + "="*80)
        print("✅ TODAS LAS UBICACIONES VERIFICADAS EXITOSAMENTE")
        print("="*80)
        print("\nLos modelos están guardados en:")
        print(f"1. {project_root / 'models' / f'retrained_v{version}.pt'}")
        print(f"2. {project_root / 'runs' / 'train' / f'retrained_v{version}.pt'}")
        print(f"3. {project_root / 'runs' / 'train' / f'retrain_v{version}' / 'weights' / 'best.pt'}")
        print(f"4. MLflow artifacts en experiment 401576597529460193")
        print("\n" + "="*80)
        return True
    else:
        print("\n" + "="*80)
        print("⚠️  ALGUNAS UBICACIONES NO ENCONTRADAS")
        print("   Ejecuta un reentrenamiento para generar todos los archivos")
        print("="*80)
        return False

if __name__ == '__main__':
    version = int(sys.argv[1]) if len(sys.argv) > 1 else None
    verify_model_locations(version)
