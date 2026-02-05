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
        patience: int = 5
    ) -> Dict:
        """
        Reentrenar el modelo con correcciones acumuladas.
        
        Args:
            epochs: Número de épocas
            batch_size: Tamaño del batch
            patience: Early stopping patience
        
        Returns:
            Diccionario con resultados del entrenamiento
        """
        try:
            print(f"\n[ContinuousLearner] Iniciando retrain con {len(self.corrected_samples)} muestras...")
            
            # Configurar MLflow correctamente ANTES de YOLO
            import mlflow
            mlflow_dir = self.project_root / 'runs' / 'mlflow'
            mlflow_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(f"file:///{str(mlflow_dir).replace(chr(92), '/')}")
            print(f"[ContinuousLearner] MLflow configurado correctamente")
            
            # Desactivar autologging de MLflow
            mlflow.autolog(disable=True)
            print(f"[ContinuousLearner] MLflow autolog desactivado")
            
            # Preparar dataset
            retrain_dir, num_samples = self.prepare_retraining_dataset()
            print(f"[ContinuousLearner] Dataset preparado: {num_samples} muestras en {retrain_dir}")
            
            if num_samples < 5:
                logger.warning("Muestras insuficientes para reentrenar")
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
            print(f"[ContinuousLearner] data.yaml creado: {yaml_path}")
            
            logger.info(f"Iniciando reentrenamiento (épocas={epochs}, batch={batch_size})")
            print(f"[ContinuousLearner] Ejecutando model.train()...")
            
            # Reentrenar modelo SIN MLflow
            results = self.base_model.train(
                data=str(yaml_path),
                epochs=epochs,
                batch=batch_size,
                patience=patience,
                device=0 if self._has_gpu() else 'cpu',
                verbose=False,
                save=True,
                project=str(self.project_root / 'runs' / 'train'),  # Path local simple
                name=f"retrain_v{self.current_version}",
                exist_ok=True  # Overwrite previous runs
            )
            
            print(f"[ContinuousLearner] Entrenamiento completado")
            
            # Evaluar en dataset de validación
            print(f"[ContinuousLearner] Evaluando modelo...")
            metrics = self.base_model.val()
            
            # Guardar modelo en la carpeta models/
            improved_model_path = self.models_dir / f"retrained_v{self.current_version}.pt"
            print(f"[ContinuousLearner] Guardando modelo en: {improved_model_path}")
            
            # El último best.pt generado por YOLO está en runs/train/retrain_vX/weights/best.pt
            # Copiar ese archivo a models/retrained_vX.pt
            import shutil
            training_output_dir = self.project_root / 'runs' / 'train' / f"retrain_v{self.current_version}"
            best_model_from_training = training_output_dir / 'weights' / 'best.pt'
            
            if best_model_from_training.exists():
                print(f"[ContinuousLearner] Modelo entrenado encontrado en: {best_model_from_training}")
                shutil.copy2(best_model_from_training, improved_model_path)
                print(f"[ContinuousLearner] Modelo copiado a: {improved_model_path}")
            else:
                print(f"[ContinuousLearner] Buscando modelo en directorio de entrenamiento...")
                # Buscar cualquier best.pt en el directorio
                for best_file in training_output_dir.rglob('best.pt'):
                    print(f"[ContinuousLearner] Encontrado: {best_file}")
                    shutil.copy2(best_file, improved_model_path)
                    break
            
            # Verificar que el archivo se guardó
            if improved_model_path.exists():
                size_mb = improved_model_path.stat().st_size / 1024 / 1024
                print(f"[ContinuousLearner] Archivo guardado exitosamente. Tamaño: {size_mb:.2f} MB")
            else:
                print(f"[ContinuousLearner] ERROR: Archivo no se guardó correctamente")
                return {'success': False, 'error': 'No se pudo guardar el modelo', 'reason': 'Model save failed'}
            
            # Actualizar estadísticas
            trained_version = self.current_version  # Guardar la versión que se acaba de entrenar
            self.current_version += 1  # Incrementar para el próximo entrenamiento
            self.stats['total_retrains'] += 1
            self.stats['current_version'] = self.current_version
            self.stats['last_retrain'] = datetime.now().isoformat()
            self.stats['model_performance'] = {
                'mAP50': float(metrics.box.map50) if hasattr(metrics, 'box') else 0,
                'mAP50-95': float(metrics.box.map) if hasattr(metrics, 'box') else 0
            }
            
            logger.info(f"Reentrenamiento completado. Modelo guardado: {improved_model_path}")
            logger.info(f"Próxima versión para reentrenamiento: v{self.current_version}")
            print(f"[ContinuousLearner] Retrain v{trained_version} completado exitosamente!")
            print(f"[ContinuousLearner] Próxima versión será: v{self.current_version}")
            
            return {
                'success': True,
                'model_path': str(improved_model_path),
                'version': trained_version,
                'metrics': self.stats['model_performance']
            }
            
        except Exception as e:
            logger.error(f"Error durante reentrenamiento: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            traceback.print_exc()
            print(f"[ContinuousLearner] ERROR: {e}")
            print(f"[ContinuousLearner] Traceback:\n{error_traceback}")
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
