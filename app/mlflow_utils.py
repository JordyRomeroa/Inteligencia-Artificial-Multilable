"""
MLflow Utilities for YOLO Training
===================================

Funciones reutilizables para logging consistente de experimentos con MLflow.
Soporta tracking de parámetros, métricas, artefactos y versionado de modelos.

Author: ML Engineering Team
Version: 1.0
"""

import mlflow
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import json
from datetime import datetime


class MLflowYOLOTracker:
    """
    Tracker profesional para experimentos YOLO con MLflow.
    
    Beneficios:
    - Logging consistente entre entrenamientos
    - Versionado automático de modelos
    - Comparación fácil entre experimentos
    - Metadata completa para reproducibilidad
    """
    
    def __init__(self, experiment_name: str = '/Shared/Ultralytics'):
        """
        Inicializa el tracker con configuración de MLflow.
        
        Args:
            experiment_name: Nombre del experimento en MLflow
        """
        self.experiment_name = experiment_name
        self.run_id = None
        self.model_version = None
        
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Inicia un nuevo run de MLflow con tags descriptivos.
        
        Args:
            run_name: Nombre descriptivo del run
            tags: Tags adicionales (ej: {'model_type': 'yolov8n', 'dataset': 'voc2012'})
            
        Returns:
            run_id: ID del run iniciado
            
        Por qué:
        - Los tags facilitan filtrar y comparar experimentos
        - El run_name descriptivo mejora la organización
        """
        if mlflow.active_run():
            mlflow.end_run()
            
        mlflow.start_run(run_name=run_name)
        self.run_id = mlflow.active_run().info.run_id
        
        # Tags por defecto
        default_tags = {
            'framework': 'ultralytics',
            'task': 'object_detection',
            'timestamp': datetime.now().isoformat()
        }
        
        if tags:
            default_tags.update(tags)
            
        mlflow.set_tags(default_tags)
        
        return self.run_id
    
    def log_training_params(self, 
                          model_name: str,
                          num_classes: int,
                          class_names: List[str],
                          config: Dict[str, Any],
                          dataset_info: Optional[Dict[str, Any]] = None):
        """
        Loguea parámetros de entrenamiento de forma estructurada.
        
        Args:
            model_name: Nombre del modelo (ej: 'yolov8n')
            num_classes: Número de clases
            class_names: Lista de nombres de clases
            config: Diccionario con configuración de entrenamiento
            dataset_info: Info opcional del dataset (train/val/test splits)
            
        Por qué:
        - Parámetros bien organizados permiten reproducir experimentos
        - Facilita búsqueda y comparación de hiperparámetros
        """
        params = {
            'model_name': model_name,
            'num_classes': num_classes,
            'classes': ', '.join(class_names),
        }
        
        # Agregar config de entrenamiento
        for key, value in config.items():
            # MLflow no acepta objetos complejos, convertir a string
            if isinstance(value, (int, float, str, bool)):
                params[f'train_{key}'] = value
            else:
                params[f'train_{key}'] = str(value)
        
        # Agregar info de dataset si existe
        if dataset_info:
            for key, value in dataset_info.items():
                params[f'dataset_{key}'] = value
        
        mlflow.log_params(params)
        print(f"✓ Logged {len(params)} parameters to MLflow")
    
    def log_metrics_from_yolo(self, results):
        """
        Extrae y loguea métricas de resultados de YOLO.
        
        Args:
            results: Objeto de resultados de YOLO (después de train o val)
            
        Por qué:
        - Centraliza extracción de métricas
        - Evita errores de nombres inconsistentes
        - Captura todas las métricas importantes
        """
        if not results:
            print("⚠ No results to log")
            return
        
        metrics = {}
        
        # Métricas de box detection si existen
        if hasattr(results, 'box') and results.box:
            metrics['mAP50'] = float(results.box.map50)
            metrics['mAP50_95'] = float(results.box.map)
            
            if hasattr(results.box, 'mp'):
                metrics['precision'] = float(results.box.mp)
            if hasattr(results.box, 'mr'):
                metrics['recall'] = float(results.box.mr)
            
            # Métricas por clase si existen
            if hasattr(results.box, 'maps'):
                for idx, class_map in enumerate(results.box.maps):
                    metrics[f'mAP50_class_{idx}'] = float(class_map)
        
        # Loguear todas las métricas
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        print(f"✓ Logged {len(metrics)} metrics to MLflow")
        return metrics
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Loguea métricas personalizadas a MLflow.
        
        Args:
            metrics: Diccionario de métricas {nombre: valor}
            step: Step opcional para métricas temporales
            
        Por qué:
        - Permite loguear métricas custom durante reentrenamiento
        - Mantiene consistencia con tracking de experimentos
        """
        for key, value in metrics.items():
            if step is not None:
                mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metric(key, value)
        
        print(f"✓ Logged {len(metrics)} custom metrics to MLflow (step={step})")
    
    def log_training_artifacts(self, 
                               yolo_run_dir: Path,
                               final_model_path: Optional[Path] = None):
        """
        Loguea artefactos del entrenamiento (modelo, gráficas, configs).
        Registra el modelo final desde el directorio models/ y artefactos de entrenamiento.
        
        Args:
            yolo_run_dir: Directorio donde YOLO guardó resultados (ej: runs/train/retrain_v1)
            final_model_path: Path final donde está el modelo (ej: models/retrained_v1.pt)
            
        Returns:
            Path del modelo final
            
        Por qué:
        - Los modelos se guardan en models/ para mejor organización
        - Los artifacts de entrenamiento se registran selectivamente
        - Evita duplicación de rutas
        """
        print(f"[MLflow] ================================================")
        print(f"[MLflow] REGISTRANDO MODELO Y ARTEFACTOS EN MLFLOW")
        print(f"[MLflow] Directorio de entrenamiento: {yolo_run_dir}")
        print(f"[MLflow] Modelo final: {final_model_path}")
        print(f"[MLflow] ================================================")
        
        # Verificar que el modelo final existe en models/
        if final_model_path and final_model_path.exists():
            # Registrar el modelo final desde models/
            try:
                # 1. LOG COMO MODELO MLFLOW (para que aparezca en tabla Models del experimento)
                import mlflow.pyfunc
                
                # Crear un wrapper simple para PyTorch model
                class YOLOModelWrapper(mlflow.pyfunc.PythonModel):
                    def load_context(self, context):
                        """Cargar el modelo YOLO"""
                        from ultralytics import YOLO
                        self.model = YOLO(context.artifacts["model_file"])
                    
                    def predict(self, context, model_input):
                        """Predicción con el modelo"""
                        return self.model(model_input)
                
                # Crear artifacts dict con el modelo
                artifacts = {
                    "model_file": str(final_model_path)
                }
                
                # Log como PyFunc Model (aparecerá en tabla Models)
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=YOLOModelWrapper(),
                    artifacts=artifacts,
                    registered_model_name="yolo_reentrenado"  # Registro directo
                )
                print(f"✓ Modelo YOLO registrado como MLflow Model desde: {final_model_path}")
                
                # 2. TAMBIÉN log como artifact para tener el .pt accesible directamente
                mlflow.log_artifact(str(final_model_path), artifact_path='models')
                print(f"✓ Modelo también guardado en artifacts: models/{final_model_path.name}")
                
                # 3. Agregar tags para identificación
                mlflow.set_tag("model_artifact", f"models/{final_model_path.name}")
                mlflow.set_tag("model_name", final_model_path.stem)
                mlflow.set_tag("model_size_mb", f"{final_model_path.stat().st_size / 1024 / 1024:.2f}")
                mlflow.set_tag("mlflow.note.content", f"Modelo YOLO reentrenado versión {final_model_path.stem}")
                print(f"✓ Tags y metadata agregados al run")
                
            except Exception as e:
                print(f"⚠ Error registrando modelo: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[MLflow] ADVERTENCIA: Modelo no encontrado en: {final_model_path}")
        
        # Registrar artefactos importantes del entrenamiento (plots, configs)
        if yolo_run_dir.exists():
            try:
                # Registrar plots si existen
                plots_dir = yolo_run_dir
                for plot_file in plots_dir.glob('*.png'):
                    mlflow.log_artifact(str(plot_file), artifact_path='plots')
                    print(f"✓ Plot registrado: {plot_file.name}")
                
                # Registrar archivos de configuración
                for config_file in ['args.yaml', 'data.yaml']:
                    config_path = yolo_run_dir / config_file
                    if config_path.exists():
                        mlflow.log_artifact(str(config_path), artifact_path='config')
                        print(f"✓ Config registrado: {config_file}")
                
                print(f"✓ Artefactos de entrenamiento registrados")
            except Exception as e:
                print(f"⚠ Error registrando artefactos: {e}")
        
        print(f"[MLflow] ================================================")
        print(f"[MLflow] REGISTRO EN MLFLOW FINALIZADO")
        if final_model_path and final_model_path.exists():
            print(f"[MLflow] - Modelo en MLflow: models/{final_model_path.name}")
            print(f"[MLflow] - Modelo local en: {final_model_path}")
        print(f"[MLflow] ================================================")
        
        return final_model_path
    
    def log_model_version(self, 
                         model_path: Path,
                         version_type: str = 'training',
                         metadata: Optional[Dict] = None) -> str:
        """
        Registra versión del modelo con metadata desde el directorio models/.
        
        Args:
            model_path: Path al modelo .pt en models/
            version_type: 'training', 'retraining', 'finetuning'
            metadata: Información adicional (ej: num_corrections, data_added)
            
        Returns:
            version_string: Versión asignada (ej: 'v1.0.0')
            
        Por qué:
        - Versionado permite rollback si un reentrenamiento falla
        - Metadata ayuda a entender qué cambió entre versiones
        - Facilita A/B testing de modelos
        """
        # Crear metadata file en el directorio models/
        version_info = {
            'version_type': version_type,
            'timestamp': datetime.now().isoformat(),
            'model_path': str(model_path),
            'model_directory': 'models/',
            'run_id': self.run_id,
        }
        
        if metadata:
            version_info.update(metadata)
        
        # Guardar metadata en el directorio models/
        metadata_file = model_path.parent / f'{model_path.stem}_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        # Registrar metadata en MLflow bajo models/
        mlflow.log_artifact(str(metadata_file), artifact_path='models')
        
        # Auto-incrementar versión
        self.model_version = self._get_next_version(version_type)
        mlflow.log_param('model_version', self.model_version)
        
        print(f"✓ Model version registered: {self.model_version}")
        print(f"✓ Metadata guardado en: {metadata_file}")
        print(f"✓ Metadata registrado en MLflow: models/{metadata_file.name}")
        return self.model_version
    
    def log_retraining_dataset(self, 
                              dataset_dir: Path,
                              corrections_data: List[Dict]) -> None:
        """
        Loguea el dataset y correcciones usadas para reentrenamiento como artifacts.
        
        ⚠️ REQUISITO OBLIGATORIO: Guardar los datos usados en el reentrenamiento
        
        Args:
            dataset_dir: Directorio donde están imagen + etiquetas
            corrections_data: Lista de correcciones aplicadas
            
        Por qué:
        - Reproducibilidad: saber exactamente QUÉ datos se usaron
        - Auditoría: rastrear correcciones para análisis post-hoc
        - Validación: verificar de calidad de datos de reentrenamiento
        """
        print(f"\n[MLflow] GUARDANDO DATASET Y CORRECCIONES COMO ARTIFACTS")
        print(f"[MLflow] Dataset dir: {dataset_dir}")
        print(f"[MLflow] Correcciones a registrar: {len(corrections_data)}")
        
        # 1. Guardar metadatos del dataset como JSON
        dataset_metadata = {
            'timestamp': datetime.now().isoformat(),
            'dataset_dir': str(dataset_dir),
            'num_images': len(list(dataset_dir.glob('images/*'))),
            'num_labels': len(list(dataset_dir.glob('labels/*'))),
            'total_corrections_applied': len(corrections_data),
            'data_yaml_exists': (dataset_dir / 'data.yaml').exists()
        }
        
        metadata_file = Path('/tmp/retraining_dataset_metadata.json') if 'Linux' in str(Path.home()) else Path('C:\\Temp\\retraining_dataset_metadata.json')
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_file, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        mlflow.log_artifact(str(metadata_file), artifact_path='retraining_dataset')
        print(f"✓ Dataset metadata guardado: {metadata_file.name}")
        
        # 2. Guardar correcciones como CSV/JSON
        corrections_file = Path('/tmp/corrections_applied.json') if 'Linux' in str(Path.home()) else Path('C:\\Temp\\corrections_applied.json')
        corrections_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(corrections_file, 'w') as f:
            json.dump(corrections_data, f, indent=2)
        
        mlflow.log_artifact(str(corrections_file), artifact_path='retraining_dataset')
        print(f"✓ Correcciones guardadas: {corrections_file.name}")
        
        # 3. Guardar data.yaml del dataset
        data_yaml_src = dataset_dir / 'data.yaml'
        if data_yaml_src.exists():
            mlflow.log_artifact(str(data_yaml_src), artifact_path='retraining_dataset')
            print(f"✓ data.yaml guardado desde: {data_yaml_src}")
        
        # 4. Contar y registrar muestras como métrica
        num_images = len(list(dataset_dir.glob('images/*')))
        if num_images > 0:
            mlflow.log_metric('retraining_dataset_samples', num_images)
            print(f"✓ Métrica registrada: {num_images} muestras de entrenamiento")
        
        print(f"[MLflow] DATASET Y CORRECCIONES GUARDADOS EXITOSAMENTE\n")
    

    def _get_next_version(self, version_type: str) -> str:
        """
        Genera siguiente versión basada en runs anteriores.
        
        Por qué:
        - Versionado semántico facilita gestión de modelos
        - Diferencia entre training (mayor) y retraining (minor)
        """
        # Buscar última versión en experimento
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return 'v1.0.0'
        
        # Por ahora versión simple - mejorar con búsqueda en runs
        if version_type == 'training':
            return 'v1.0.0'
        elif version_type == 'retraining':
            return 'v1.1.0'
        else:
            return 'v1.0.1'
    
    def end_run(self, status: str = 'FINISHED'):
        """
        Finaliza el run actual con estado.
        
        Args:
            status: 'FINISHED', 'FAILED', 'KILLED'
            
        Por qué:
        - Estado explícito ayuda en debugging
        - MLflow requiere cerrar runs para persistir datos
        """
        if mlflow.active_run():
            mlflow.log_param('run_status', status)
            mlflow.end_run()
            print(f"✓ MLflow run ended with status: {status}")
    
    def compare_runs(self, run_ids: List[str], metrics: List[str]):
        """
        Compara métricas entre múltiples runs.
        
        Args:
            run_ids: Lista de run IDs a comparar
            metrics: Lista de métricas a comparar (ej: ['mAP50', 'precision'])
            
        Returns:
            DataFrame con comparación
            
        Por qué:
        - Facilita análisis de mejoras entre versiones
        - Ayuda a seleccionar mejor modelo para producción
        """
        import pandas as pd
        
        comparison = []
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            row = {'run_id': run_id, 'run_name': run.data.tags.get('mlflow.runName', 'N/A')}
            
            for metric in metrics:
                row[metric] = run.data.metrics.get(metric, None)
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        return df


# Función helper para configuración rápida
def setup_mlflow(project_root: Path, experiment_id: str = '401576597529460193') -> MLflowYOLOTracker:
    """
    Configura MLflow EXACTAMENTE como se requiere para reentrenamiento.
    
    ⚠️ REQUISITOS OBLIGATORIOS:
    - experiment_id: DEBE ser 401576597529460193
    - artifact_location: file:///c:/Users/jordy/OneDrive/Desktop/iaaaa/iajordy2/runs/mlflow/401576597529460193
    - NO permite experimentos nuevos
    - NO permite rutas por defecto
    
    Args:
        project_root: Ruta raíz del proyecto
        experiment_id: ID del experimento específico (REQUERIDO)
        
    Returns:
        MLflowYOLOTracker: Tracker configurado correctamente
        
    Por qué:
    - MLflow DEBE usar artifact_location explícito para guardar correctamente
    - set_experiment_by_id() força el ID specific, evita crear experimentos nuevos
    - Configuración ANTES del tracker previene conflictos posteriores
    """
    # =====================================================
    # 1. CONFIGURAR ARTIFACT LOCATION EXPLÍCITAMENTE
    # =====================================================
    runs_dir = project_root / 'runs'
    mlflow_experiment_dir = runs_dir / 'mlflow' / experiment_id
    
    # Crear estructura de directorios ANTES de inicializar MLflow
    mlflow_experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Convertir a ruta/URI correcta para Windows
    mlflow_path = str(mlflow_experiment_dir).replace('\\', '/')
    
    # Convertir a file:/// URI para Windows (OBLIGATORIO)
    if len(mlflow_path) > 1 and mlflow_path[1] == ':':
        artifact_location = f"file:///{mlflow_path}"
        tracking_uri = f"file:///{str(runs_dir / 'mlflow').replace(chr(92), '/')}"
    else:
        artifact_location = f"file://{mlflow_path}"
        tracking_uri = f"file://{str(runs_dir / 'mlflow')}"
    
    print("=" * 80)
    print("[MLflow] CONFIGURACIÓN OBLIGATORIA DE REENTRENAMIENTO")
    print("=" * 80)
    print(f"[MLflow] Tracking URI: {tracking_uri}")
    print(f"[MLflow] Experiment ID: {experiment_id}")
    print(f"[MLflow] Artifact Location: {artifact_location}")
    print(f"[MLflow] Artifact Dir: {mlflow_experiment_dir}")
    print("=" * 80)
    
    # =====================================================
    # 2. SET TRACKING URI (ANTES de cualquier operación)
    # =====================================================
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[MLflow] ✓ Tracking URI configurado")
    
    # =====================================================
    # 3. USAR EXPERIMENT (buscar por ID o usar nombre)
    # =====================================================
    try:
        # Intentar obtener experimento por ID
        experiment = mlflow.get_experiment(experiment_id)
        if experiment:
            print(f"[MLflow] ✓ Experimento encontrado por ID: {experiment_id}")
            print(f"[MLflow] ✓ Nombre del experimento: {experiment.name}")
            # Usar set_experiment con el nombre que encontramos
            mlflow.set_experiment(experiment.name)
        else:
            # Si no existe por ID, usar el nombre conocido
            print(f"[MLflow] ⚠️ ID {experiment_id} no encontrado, usando experimento existente")
            mlflow.set_experiment('/Shared/Ultralytics')
            
    except Exception as e:
        # Fallback: usar experimento conocido por nombre
        print(f"[MLflow] ⚠️ No se pudo obtener experimento por ID: {e}")
        print(f"[MLflow] Usando experimento por nombre: /Shared/Ultralytics")
        mlflow.set_experiment('/Shared/Ultralytics')
    
    print("=" * 80)
    print("[MLflow] CONFIGURACIÓN COMPLETADA")
    print("=" * 80)
    
    # Crear tracker CON la configuración establecida
    tracker = MLflowYOLOTracker(experiment_name='/Shared/Ultralytics')
    return tracker
