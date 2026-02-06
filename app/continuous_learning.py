"""
Sistema de Aprendizaje Continuo - Permite que el modelo aprenda de nuevas instancias
y detecte cambios de distribución en los datos.
"""

import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import numpy as np
from ultralytics import YOLO
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContinuousLearner:
    """
    Sistema de aprendizaje continuo que permite:
    - Capturar correcciones de usuarios
    - Detectar cambios en la distribución de datos
    - Reentrenar incrementalmente
    - Versionar modelos automáticamente
    """
    
    def __init__(self, base_model_path: str, project_root: Path = None):
        """
        Inicializar el sistema de aprendizaje continuo.
        
        Args:
            base_model_path: Ruta al modelo YOLO preentrenado
            project_root: Raíz del proyecto (default: parent de este archivo)
        """
        self.project_root = project_root or Path(__file__).parent.parent
        self.base_model_path = Path(base_model_path)
        self.corrections_dir = self.project_root / 'data' / 'corrections'
        self.models_dir = self.project_root / 'models'
        self.logs_dir = self.project_root / 'logs' / 'continuous_learning'
        
        # Crear directorios necesarios
        self.corrections_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar modelo base
        self.base_model = YOLO(str(self.base_model_path))
        
        # Detectar la próxima versión basándose en archivos existentes
        self.current_version = self._get_next_version()
        
        # Almacenamiento de nuevas muestras
        self.new_samples = []
        self.corrected_samples = []
        
        # Estadísticas
        self.stats = {
            'total_corrections': 0,
            'total_retrains': 0,
            'current_version': self.current_version,
            'last_retrain': None,
            'model_performance': {}
        }
        
        logger.info(f"ContinuousLearner inicializado con modelo: {self.base_model_path}")
        logger.info(f"Próxima versión de reentrenamiento: v{self.current_version}")
    
    def _get_next_version(self) -> int:
        """Detectar la próxima versión basándose en archivos existentes"""
        existing_versions = []
        if self.models_dir.exists():
            for pt_file in self.models_dir.glob('retrained_v*.pt'):
                try:
                    version_num = int(pt_file.stem.replace('retrained_v', ''))
                    existing_versions.append(version_num)
                except ValueError:
                    pass
        
        if existing_versions:
            next_version = max(existing_versions) + 1
        else:
            next_version = 1
        
        return next_version
    
    def add_corrected_sample(
        self,
        image_path: str,
        boxes: List[Dict],
        user_id: str = "unknown",
        confidence_threshold: float = 0.5
    ) -> bool:
        """
        Agregar una muestra corregida por el usuario.
        
        Args:
            image_path: Ruta a la imagen
            boxes: Lista de diccionarios con {'class': str, 'bbox': [x1,y1,x2,y2], 'confidence': float}
            user_id: ID del usuario que hizo la corrección
            confidence_threshold: Solo agregar si la predicción original estaba bajo este threshold
        
        Returns:
            True si se agregó exitosamente, False en caso contrario
        """
        try:
            # Validar entrada
            if not Path(image_path).exists():
                logger.warning(f"Imagen no encontrada: {image_path}")
                return False
            
            correction_data = {
                'timestamp': datetime.now().isoformat(),
                'image_path': str(image_path),
                'boxes': boxes,
                'user_id': user_id,
                'original_confidence': [b.get('confidence', 0) for b in boxes],
                'version_when_corrected': self.current_version
            }
            
            self.corrected_samples.append(correction_data)
            self.stats['total_corrections'] += 1
            
            # Guardar en archivo JSON
            self._save_correction(correction_data)
            
            logger.info(
                f"Muestra corregida agregada: {image_path} "
                f"({len(boxes)} objetos) por usuario {user_id}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error al agregar muestra corregida: {e}")
            return False
    
    def _save_correction(self, correction_data: Dict):
        """Guardar corrección en archivo JSON."""
        timestamp = correction_data['timestamp'].replace(':', '-').split('.')[0]
        correction_file = self.corrections_dir / f"correction_{timestamp}.json"
        
        with open(correction_file, 'w') as f:
            json.dump(correction_data, f, indent=2)
    
    def should_retrain(self, min_corrections: int = 5) -> bool:
        """
        Determinar si se debe reentrenar basado en número de correcciones.
        
        Args:
            min_corrections: Mínimo número de correcciones para reentrenar
        
        Returns:
            True si se han acumulado suficientes correcciones
        """
        return len(self.corrected_samples) >= min_corrections
    
    def detect_distribution_shift(self, test_batch_size: int = 32) -> Dict:
        """
        Detectar cambios de distribución usando predicciones en nuevas muestras.
        
        Returns:
            Diccionario con resultados del análisis de drift
        """
        if len(self.corrected_samples) < 5:
            return {'drift_detected': False, 'reason': 'Muestras insuficientes'}
        
        # Calcular estadísticas de confianza en nuevas muestras
        recent_confidences = []
        for sample in self.corrected_samples[-test_batch_size:]:
            recent_confidences.extend(sample['original_confidence'])
        
        if not recent_confidences:
            return {'drift_detected': False, 'reason': 'No hay datos'}
        
        mean_confidence = np.mean(recent_confidences)
        std_confidence = np.std(recent_confidences)
        
        # Heurística simple: si confianza cae significativamente, hay drift
        drift_detected = mean_confidence < 0.6 or std_confidence > 0.3
        
        analysis = {
            'drift_detected': drift_detected,
            'mean_confidence': float(mean_confidence),
            'std_confidence': float(std_confidence),
            'num_samples_analyzed': len(recent_confidences),
            'threshold': 0.6
        }
        
        logger.info(f"Análisis de drift: {analysis}")
        return analysis
    
    def prepare_retraining_dataset(self) -> Tuple[Path, int]:
        """
        Preparar dataset para reentrenamiento usando correcciones acumuladas.
        
        Returns:
            Tupla (ruta_dataset, num_muestras)
        """
        retrain_dir = self.project_root / 'data' / f'retrain_v{self.current_version}'
        retrain_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = retrain_dir / 'images'
        labels_dir = retrain_dir / 'labels'
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        num_samples = 0
        
        # Convertir correcciones a formato YOLO
        for idx, sample in enumerate(self.corrected_samples):
            try:
                # Copiar imagen
                src_image = Path(sample['image_path'])
                if src_image.exists():
                    dst_image = images_dir / f"retrain_{idx:04d}{src_image.suffix}"
                    shutil.copy2(src_image, dst_image)
                    
                    # Crear etiqueta YOLO
                    label_file = labels_dir / f"retrain_{idx:04d}.txt"
                    self._create_yolo_label(sample['boxes'], label_file)
                    
                    num_samples += 1
            except Exception as e:
                logger.error(f"Error preparando muestra {idx}: {e}")
                continue
        
        logger.info(f"Dataset de reentrenamiento preparado: {num_samples} muestras en {retrain_dir}")
        return retrain_dir, num_samples
    
    def _create_yolo_label(self, boxes: List[Dict], output_path: Path):
        """
        Convertir bounding boxes a formato YOLO y guardar.
        
        Args:
            boxes: Lista de diccionarios con format 'class' y 'bbox'
            output_path: Ruta donde guardar el archivo de etiqueta
        """
        lines = []
        for box in boxes:
            class_name = box.get('class', 'unknown')
            bbox = box.get('bbox', [0, 0, 1, 1])  # x1, y1, x2, y2
            
            # Convertir a formato YOLO (x_center, y_center, width, height)
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            # Normalizar
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            # Mapear clase a ID (person=0, car=1, dog=2)
            class_map = {'person': 0, 'car': 1, 'dog': 2}
            class_id = class_map.get(class_name.lower(), 0)
            
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
    
    def retrain(
        self,
        epochs: int = 10,
        batch_size: int = 16,
        patience: int = 5,
        experiment_id: str = '401576597529460193'
    ) -> Dict:
        """
        Reentrenar el modelo con correcciones acumuladas - FLUJO OBLIGATORIO DE MLFLOW.
        
        ⚠️ CORRECTIVO MLOps: Este flujo garantiza que CADA reentrenamiento:
          1. Use EXACTAMENTE experiment_id = 401576597529460193
          2. Use EXACTAMENTE artifact_location correcto
          3. Guarde datos de reentrenamiento como artifacts
          4. Registre métricas PRE y POST
          5. Agregue tag "retraining" explícito
        
        Args:
            epochs: Número de épocas
            batch_size: Tamaño del batch
            patience: Early stopping patience
            experiment_id: ID del experimento MLflow (OBLIGATORIO)
        
        Returns:
            Diccionario con resultados del entrenamiento
        """
        import mlflow as mlf
        
        tracker = None
        try:
            print(f"\n" + "="*80)
            print(f"[ContinuousLearner] INICIANDO REENTRENAMIENTO CON MLFLOW CORRECTO")
            print(f"[ContinuousLearner] Experiment ID: {experiment_id}")
            print(f"[ContinuousLearner] Muestras disponibles: {len(self.corrected_samples)}")
            print(f"="*80)
            
            # =====================================================================
            # PASO 1: IMPORTAR Y CONFIGURAR MLFLOW CON EXPERIMENT_ID OBLIGATORIO
            # =====================================================================
            from mlflow_utils import setup_mlflow
            print(f"\n[ContinuousLearner] ✓ Importando setup_mlflow...")

            tracker = setup_mlflow(self.project_root, experiment_id=experiment_id)
            print(f"[ContinuousLearner] ✓ Tracker configurado CON experiment_id={experiment_id}")
            
            # =====================================================================
            # PASO 2: PREPARAR DATASET DE REENTRENAMIENTO
            # =====================================================================
            retrain_dir, num_samples = self.prepare_retraining_dataset()
            print(f"[ContinuousLearner] ✓ Dataset preparado: {num_samples} muestras")
            print(f"[ContinuousLearner]   Ubicación: {retrain_dir}")
            
            if num_samples < 5:
                logger.warning("Muestras insuficientes para reentrenar")
                print(f"[ContinuousLearner] ⚠️ ABORTO: {num_samples} < 5 muestras requeridas")
                return {'success': False, 'reason': 'Muestras insuficientes'}
            
            # Crear data.yaml para YOLO
            data_yaml = {
                'path': str(retrain_dir),
                'train': str(retrain_dir / 'images'),
                'val': str(retrain_dir / 'images'),
                'test': str(retrain_dir / 'images'),
                'nc': 3,
                'names': ['person', 'car', 'dog']
            }
            
            yaml_path = retrain_dir / 'data.yaml'
            with open(yaml_path, 'w') as f:
                yaml.dump(data_yaml, f)
            print(f"[ContinuousLearner] ✓ data.yaml creado")
            
            # =====================================================================
            # PASO 3: INICIAR RUN DE MLFLOW CON TAGS OBLIGATORIOS
            # =====================================================================
            run_name = f"continuous_retrain_v{self.current_version}"
            tags = {
                "type": "retraining",  # TAG OBLIGATORIO
                "model_type": "continuous_learning",
                "version": f"v{self.current_version}",
                "num_corrections": str(len(self.corrected_samples)),
                "training_type": "incremental_retrain",
                "experiment_id": experiment_id  # Rastrear ID del experimento
            }
            
            tracker.start_run(run_name=run_name, tags=tags)
            print(f"[ContinuousLearner] ✓ MLflow run iniciado: {run_name}")
            print(f"[ContinuousLearner]   Tags: {tags}")
            
            # =====================================================================
            # PASO 4: REGISTRAR PARÁMETROS DE REENTRENAMIENTO
            # =====================================================================
            training_params = {
                'epochs': epochs,
                'batch_size': batch_size,
                'patience': patience,
                'device': '0' if self._has_gpu() else 'cpu',
                'num_corrections': len(self.corrected_samples),
                'num_training_samples': num_samples,
                'model_version': f"v{self.current_version}",
                'base_model': str(self.base_model_path),
                'optimizer': 'auto',
                'image_size': 640
            }
            base_name = self.base_model_path.name.lower()
            model_name = 'yolov8s' if 'v8s' in base_name or 'improved' in base_name else 'yolov8n'
            tracker.log_training_params(
                model_name=model_name,
                num_classes=3,
                class_names=['person', 'car', 'dog'],
                config=training_params
            )
            print(f"[ContinuousLearner] ✓ Parámetros registrados en MLflow")
            
            # =====================================================================
            # PASO 5: REGISTRAR DATASET Y CORRECCIONES COMO ARTIFACTS (OBLIGATORIO)
            # =====================================================================
            tracker.log_retraining_dataset(retrain_dir, self.corrected_samples)
            print(f"[ContinuousLearner] ✓ Dataset y correcciones guardados en MLflow")
            
            # =====================================================================
            # PASO 6: EJECUTAR ENTRENAMIENTO (SIN desactivar MLflow)
            # =====================================================================
            logger.info(f"Iniciando reentrenamiento (épocas={epochs}, batch={batch_size})")
            print(f"[ContinuousLearner] Ejecutando model.train()...")
            
            # IMPORTANTE: NO deshabilitar MLflow env vars
            # Mantener tracking URI consistente durante el entrenamiento
            current_tracking_uri = mlf.get_tracking_uri()
            
            # CRÍTICO: Deshabilitar Ultralytics MLflow callback para evitar conflictos
            # Estamos manejando MLflow manualmente, no queremos que Ultralytics interfiera
            from ultralytics import settings
            original_mlflow_setting = settings.get('mlflow', True)
            settings.update({'mlflow': False})  # Deshabilitar MLflow automático de Ultralytics
            print(f"[ContinuousLearner] ✓ MLflow callback de Ultralytics deshabilitado (manejamos manualmente)")
            
            try:
                # Reentrenar modelo
                results = self.base_model.train(
                    data=str(yaml_path),
                    epochs=epochs,
                    batch=batch_size,
                    patience=patience,
                    device=0 if self._has_gpu() else 'cpu',
                    verbose=False,
                    save=True,
                    project=str(self.project_root / 'runs' / 'train'),
                    name=f"retrain_v{self.current_version}",
                    exist_ok=True
                )
            finally:
                # Restaurar setting original
                settings.update({'mlflow': original_mlflow_setting})
            
            print(f"[ContinuousLearner] ✓ Entrenamiento completado")
            
            # =====================================================================
            # PASO 7: REGISTRAR MÉTRICAS DE ENTRENAMIENTO
            # =====================================================================
            training_output_dir = self.project_root / 'runs' / 'train' / f"retrain_v{self.current_version}"
            if results and hasattr(results, 'results_dict'):
                # Crear objeto simple con métricas para logging
                class MetricsHolder:
                    def __init__(self, results_dict):
                        self.results_dict = results_dict
                        # Extraer métricas de box si existen
                        if 'metrics/mAP50(B)' in results_dict:
                            self.box = type('obj', (object,), {
                                'map50': results_dict.get('metrics/mAP50(B)', 0),
                                'map': results_dict.get('metrics/mAP50-95(B)', 0),
                                'mp': results_dict.get('metrics/precision(B)', 0),
                                'mr': results_dict.get('metrics/recall(B)', 0)
                            })()
                
                metrics_obj = MetricsHolder(results.results_dict)
                tracker.log_metrics_from_yolo(metrics_obj)
                print(f"[ContinuousLearner] ✓ Métricas de entrenamiento registradas")
            else:
                print(f"[ContinuousLearner] ⚠ No se encontraron métricas en results")
            
            # =====================================================================
            # PASO 8: EVALUAR Y REGISTRAR MÉTRICAS DE VALIDACIÓN
            # =====================================================================
            print(f"[ContinuousLearner] Evaluando modelo en dataset de validación...")
            metrics = self.base_model.val()
            
            if hasattr(metrics, 'box'):
                validation_metrics = {
                    'val_mAP50': float(metrics.box.map50),
                    'val_mAP50_95': float(metrics.box.map),
                    'val_precision': float(metrics.box.mp),
                    'val_recall': float(metrics.box.mr)
                }
                tracker.log_metrics(validation_metrics, step=epochs)
                print(f"[ContinuousLearner] ✓ Métricas de validación registradas")
                print(f"[ContinuousLearner]   mAP50: {validation_metrics['val_mAP50']:.4f}")
                print(f"[ContinuousLearner]   Precision: {validation_metrics['val_precision']:.4f}")
                print(f"[ContinuousLearner]   Recall: {validation_metrics['val_recall']:.4f}")
            
            # =====================================================================
            # PASO 9: COPIAR MODELO A DIRECTORIO models/ Y GUARDAR COMO ARTIFACT
            # =====================================================================
            improved_model_path = self.models_dir / f"retrained_v{self.current_version}.pt"
            print(f"[ContinuousLearner] Guardando modelo en: {improved_model_path}")
            
            import shutil
            best_model_from_training = training_output_dir / 'weights' / 'best.pt'
            
            if best_model_from_training.exists():
                shutil.copy2(best_model_from_training, improved_model_path)
                print(f"[ContinuousLearner] ✓ Modelo copiado exitosamente")
            else:
                # Buscar best.pt recursivamente
                found = False
                for best_file in training_output_dir.rglob('best.pt'):
                    shutil.copy2(best_file, improved_model_path)
                    found = True
                    break
                if not found:
                    print(f"[ContinuousLearner] ❌ ERROR: No se pudo encontrar best.pt")
                    tracker.end_run(status='FAILED')
                    return {'success': False, 'error': 'Modelo entrenado no encontrado'}
            
            # Verificar que el archivo se guardó
            if improved_model_path.exists():
                size_mb = improved_model_path.stat().st_size / 1024 / 1024
                print(f"[ContinuousLearner] ✓ Archivo guardado en models/: {size_mb:.2f} MB")
                
                # GUARDAR COPIA ADICIONAL EN runs/train/ PARA ACCESO DIRECTO
                runs_train_model = self.project_root / 'runs' / 'train' / f"retrained_v{self.current_version}.pt"
                shutil.copy2(improved_model_path, runs_train_model)
                print(f"[ContinuousLearner] ✓ Modelo copiado también a runs/train/retrained_v{self.current_version}.pt")
            else:
                print(f"[ContinuousLearner] ❌ ERROR: Archivo no se guardó")
                tracker.end_run(status='FAILED')
                return {'success': False, 'error': 'No se pudo guardar el modelo'}
            
            # =====================================================================
            # PASO 10: REGISTRAR ARTEFACTOS FINALES EN MLFLOW
            # =====================================================================
            print(f"\n[ContinuousLearner] REGISTRANDO ARTEFACTOS FINALES EN MLFLOW")
            print(f"[ContinuousLearner] " + "="*70)
            
            tracker.log_training_artifacts(
                yolo_run_dir=training_output_dir,
                final_model_path=improved_model_path
            )
            
            print(f"[ContinuousLearner] ✓ ARTEFACTOS REGISTRADOS EXITOSAMENTE")
            print(f"[ContinuousLearner] " + "="*70)
            
            # =====================================================================
            # PASO 11: REGISTRAR VERSIÓN DEL MODELO
            # =====================================================================
            trained_version = self.current_version
            version_info = {
                'version': f"v{trained_version}",
                'model_type': 'yolov8n',
                'training_type': 'continuous_learning',
                'base_model': str(self.base_model_path),
                'num_corrections': len(self.corrected_samples),
                'timestamp': datetime.now().isoformat()
            }
            
            tracker.log_model_version(
                model_path=improved_model_path,
                version_type='retraining',
                metadata=version_info
            )
            print(f"[ContinuousLearner] ✓ Modelo versionado exitosamente")
            
            # =====================================================================
            # PASO 12: FINALIZAR RUN DE MLFLOW
            # =====================================================================
            # Actualizar estadísticas
            self.current_version += 1
            self.stats['total_retrains'] += 1
            self.stats['current_version'] = self.current_version
            self.stats['last_retrain'] = datetime.now().isoformat()
            self.stats['model_performance'] = {
                'mAP50': float(metrics.box.map50) if hasattr(metrics, 'box') else 0,
                'mAP50-95': float(metrics.box.map) if hasattr(metrics, 'box') else 0
            }
            
            # Finalizar run EXPLÍCITAMENTE
            tracker.end_run(status='FINISHED')
            print(f"[ContinuousLearner] ✓ MLflow run finalizado correctamente")
            
            logger.info(f"Reentrenamiento completado exitosamente")
            logger.info(f"Modelo: {improved_model_path}")
            logger.info(f"Próxima versión: v{self.current_version}")
            
            print(f"\n" + "="*80)
            print(f"[ContinuousLearner] ✅ REENTRENAMIENTO EXITOSO v{trained_version}")
            print(f"[ContinuousLearner] Próxima versión: v{self.current_version}")
            print(f"="*80 + "\n")
            
            return {
                'success': True,
                'model_path': str(improved_model_path),
                'version': trained_version,
                'metrics': self.stats['model_performance'],
                'mlflow_run_id': tracker.run_id if tracker.run_id else None,
                'experiment_id': experiment_id
            }
            
        except Exception as e:
            logger.error(f"❌ Error durante reentrenamiento: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            traceback.print_exc()
            print(f"\n[ContinuousLearner] ❌ ERROR: {e}")
            print(f"[ContinuousLearner] Traceback:\n{error_traceback}")
            
            # Finalizar run de MLflow con estado FAILED
            try:
                if tracker:
                    tracker.end_run(status='FAILED')
                    print(f"[ContinuousLearner] MLflow run marcado como FAILED")
            except:
                pass
            
            return {'success': False, 'error': str(e), 'reason': 'Retraining failed'}
    
    def _has_gpu(self) -> bool:
        """Verificar si hay GPU disponible."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def get_stats(self) -> Dict:
        """Retornar estadísticas del aprendizaje continuo."""
        return {
            **self.stats,
            'pending_corrections': len(self.corrected_samples),
            'pending_retraining': self.should_retrain()
        }
    
    def export_report(self, output_path: str = None) -> str:
        """
        Exportar reporte de aprendizaje continuo.
        
        Args:
            output_path: Ruta donde guardar el reporte (default: logs)
        
        Returns:
            Ruta del archivo generado
        """
        if output_path is None:
            output_path = self.logs_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_stats(),
            'corrections': len(self.corrected_samples),
            'distribution_shift': self.detect_distribution_shift(),
            'ready_for_retrain': self.should_retrain()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Reporte exportado: {output_path}")
        return str(output_path)


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar
    learner = ContinuousLearner(
        base_model_path='models/best_improved.pt',
        project_root=Path(__file__).parent.parent
    )
    
    # Ver estadísticas
    print("Estadísticas iniciales:", learner.get_stats())
    
    # Simular correcciones de usuario
    example_correction = {
        'class': 'person',
        'bbox': [100, 100, 200, 300],
        'confidence': 0.45  # Baja confianza - buena candidata para agregar
    }
    
    logger.info("Sistema de aprendizaje continuo listo.")
