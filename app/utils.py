"""
🛠️ Utilidades para la aplicación de clasificación multilabel
"""

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image


def preprocess_image(img_input, target_size=(224, 224)):
    """
    Preprocesa una imagen para predicción.
    
    Args:
        img_input: PIL Image, ruta a imagen, o array numpy
        target_size: Tamaño objetivo (height, width)
    
    Returns:
        Array numpy preprocesado [1, height, width, 3]
    """
    # Convertir a PIL Image si es necesario
    if isinstance(img_input, str):
        img = Image.open(img_input)
    elif isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input.astype('uint8'))
    else:
        img = img_input
    
    # Convertir a RGB si es necesario
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize(target_size)
    
    # Convertir a array
    img_array = np.array(img)
    
    # Expandir dimensiones para batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalizar a [0, 255] float32
    img_array = img_array.astype('float32')
    
    return img_array


def predict_multilabel(model, img_input, classes, threshold=0.5, top_k=None):
    """
    Realiza predicción multilabel.
    
    Args:
        model: Modelo de Keras
        img_input: Imagen preprocesada o ruta
        classes: Lista de nombres de clases
        threshold: Umbral de decisión
        top_k: Número máximo de predicciones
    
    Returns:
        Dict con resultados de predicción
    """
    # Preprocesar si es necesario
    if not isinstance(img_input, np.ndarray) or img_input.ndim != 4:
        img_array = preprocess_image(img_input)
    else:
        img_array = img_input
    
    # Predicción
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Aplicar threshold
    binary_predictions = (predictions >= threshold).astype(int)
    predicted_indices = np.where(binary_predictions == 1)[0]
    
    # Crear resultados
    results = []
    for idx in predicted_indices:
        results.append({
            'class': classes[idx],
            'probability': float(predictions[idx]),
            'index': int(idx)
        })
    
    # Ordenar por probabilidad
    results = sorted(results, key=lambda x: x['probability'], reverse=True)
    
    # Limitar a top_k
    if top_k is not None:
        results = results[:top_k]
    
    # Todas las predicciones
    all_predictions = [
        {
            'class': classes[i], 
            'probability': float(predictions[i]),
            'index': i
        }
        for i in range(len(classes))
    ]
    all_predictions = sorted(all_predictions, key=lambda x: x['probability'], reverse=True)
    
    return {
        'predicted_labels': [r['class'] for r in results],
        'predictions': results,
        'all_predictions': all_predictions,
        'num_labels': len(results),
        'threshold': threshold,
        'raw_predictions': predictions.tolist()
    }


def visualize_predictions(predictions_result, max_display=10):
    """
    Formatea las predicciones para visualización.
    
    Args:
        predictions_result: Resultado de predict_multilabel()
        max_display: Número máximo a mostrar
    
    Returns:
        String formateado con las predicciones
    """
    output = []
    output.append(f"Threshold: {predictions_result['threshold']}")
    output.append(f"Etiquetas detectadas: {predictions_result['num_labels']}\n")
    
    if predictions_result['num_labels'] > 0:
        output.append("Predicciones:")
        for i, pred in enumerate(predictions_result['predictions'][:max_display], 1):
            prob_percent = pred['probability'] * 100
            bar = '█' * int(prob_percent / 5)
            output.append(f"{i:2d}. {pred['class']:20s} {prob_percent:6.2f}% {bar}")
    else:
        output.append("No se detectaron etiquetas con el threshold actual.")
    
    return '\n'.join(output)


def get_model_info(model, classes):
    """
    Obtiene información del modelo.
    
    Args:
        model: Modelo de Keras
        classes: Lista de clases
    
    Returns:
        Dict con información del modelo
    """
    return {
        'architecture': model.name,
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'num_classes': len(classes),
        'total_parameters': int(model.count_params()),
        'trainable_parameters': int(sum([
            tf.size(w).numpy() for w in model.trainable_weights
        ])),
        'output_activation': 'sigmoid',
        'loss_function': 'binary_crossentropy',
        'task': 'multilabel_classification'
    }


def apply_threshold(predictions, threshold):
    """
    Aplica threshold a las predicciones.
    
    Args:
        predictions: Array de probabilidades
        threshold: Umbral
    
    Returns:
        Array binario
    """
    return (predictions >= threshold).astype(int)


def get_top_k_predictions(predictions, classes, k=5):
    """
    Obtiene las top-k predicciones.
    
    Args:
        predictions: Array de probabilidades
        classes: Lista de clases
        k: Número de predicciones
    
    Returns:
        Lista de dicts con top-k predicciones
    """
    top_k_indices = np.argsort(predictions)[-k:][::-1]
    
    results = []
    for idx in top_k_indices:
        results.append({
            'class': classes[idx],
            'probability': float(predictions[idx]),
            'index': int(idx)
        })
    
    return results


def format_predictions_json(predictions_result):
    """
    Formatea predicciones como JSON limpio.
    
    Args:
        predictions_result: Resultado de predict_multilabel()
    
    Returns:
        Dict formateado para JSON
    """
    return {
        'threshold': predictions_result['threshold'],
        'num_labels_detected': predictions_result['num_labels'],
        'predicted_labels': predictions_result['predicted_labels'],
        'detailed_predictions': [
            {
                'label': pred['class'],
                'confidence': f"{pred['probability']*100:.2f}%",
                'probability': round(pred['probability'], 4)
            }
            for pred in predictions_result['predictions']
        ]
    }


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calcula métricas multilabel.
    
    Args:
        y_true: Etiquetas verdaderas (binary)
        y_pred: Predicciones (probabilidades)
        threshold: Umbral
    
    Returns:
        Dict con métricas
    """
    from sklearn.metrics import (
        hamming_loss, f1_score, precision_score, 
        recall_score, accuracy_score
    )
    
    y_pred_binary = apply_threshold(y_pred, threshold)
    
    return {
        'hamming_loss': float(hamming_loss(y_true, y_pred_binary)),
        'subset_accuracy': float(accuracy_score(y_true, y_pred_binary)),
        'f1_micro': float(f1_score(y_true, y_pred_binary, average='micro')),
        'f1_macro': float(f1_score(y_true, y_pred_binary, average='macro')),
        'precision_micro': float(precision_score(y_true, y_pred_binary, average='micro')),
        'recall_micro': float(recall_score(y_true, y_pred_binary, average='micro'))
    }


def focal_loss(y_true, y_pred, gamma=0.5, alpha=0.25):
    """
    Focal Loss SUAVIZADO para evitar predicciones todo-positivas.
    Gamma reducido a 0.5 para menos énfasis en ejemplos difíciles.
    
    Args:
        y_true: True labels (batch_size, num_classes)
        y_pred: Predicted probabilities (batch_size, num_classes)
        gamma: Focusing parameter (REDUCIDO a 0.5 para estabilidad)
        alpha: Balancing parameter (default 0.25)
    
    Returns:
        Focal loss value
    """
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    
    # BCE base
    bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    
    # Modulación focal SUAVE (gamma bajo)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_weight = tf.pow(1 - p_t, gamma)
    
    # Aplicar peso focal
    focal_bce = focal_weight * bce
    
    return tf.reduce_mean(focal_bce)


def incremental_retrain(model, images, labels, epochs=15, learning_rate=1e-6, 
                        class_weights=None, verbose=1):
    """
    Incrementally retrain the model with new corrected data.
    
    Args:
        model: Trained Keras model
        images: Numpy array of images (N, 224, 224, 3) normalized [0, 255]
        labels: Numpy array of multilabel targets (N, num_classes)
        epochs: Number of training epochs (default 15 for better convergence with few samples)
        learning_rate: Learning rate for fine-tuning (default 1e-6 for very small adjustments)
        class_weights: Optional dict of class weights
        verbose: Verbosity level (0, 1, 2)
    
    Returns:
        Training history object
    """
    from tensorflow.keras.optimizers import Adam
    
    # Normalize images if needed (expecting [0, 255] range)
    if images.max() > 1.0:
        images = images / 255.0
    
    # Freeze all layers except the last classification layers
    # For EfficientNetB0, unfreeze more layers for better fine-tuning
    for layer in model.layers[:-6]:
        layer.trainable = False
    
    # Unfreeze last 6 layers for fine-tuning
    for layer in model.layers[-6:]:
        layer.trainable = True
    
    # Compile with focal loss suavizado and adaptive learning rate
    # Usar binary_crossentropy en vez de focal_loss para más estabilidad
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',  # Más estable que focal_loss para fine-tuning
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision', thresholds=0.5),
            tf.keras.metrics.Recall(name='recall', thresholds=0.5)
        ]
    )
    
    # Prepare class weights if provided
    sample_weights = None
    if class_weights is not None:
        # Calculate sample weights based on class weights
        sample_weights = np.zeros(len(labels))
        for i, label_vector in enumerate(labels):
            active_classes = np.where(label_vector == 1)[0]
            if len(active_classes) > 0:
                weights = [class_weights.get(cls, 1.0) for cls in active_classes]
                sample_weights[i] = np.mean(weights)
            else:
                sample_weights[i] = 1.0
    
    # Train the model
    history = model.fit(
        images,
        labels,
        epochs=epochs,
        batch_size=min(16, len(images)),
        validation_split=0.2 if len(images) > 10 else 0.0,
        sample_weight=sample_weights,
        verbose=verbose
    )
    
    return history


def calculate_class_weights(labels, max_weight=10.0):
    """
    Calculate class weights based on label frequency.
    
    Args:
        labels: Numpy array of shape (N, num_classes) with binary labels
        max_weight: Maximum weight to assign to rare classes (default 10.0)
    
    Returns:
        Dictionary mapping class index to weight
    """
    num_samples = len(labels)
    num_classes = labels.shape[1]
    
    # Count positive samples per class
    class_counts = labels.sum(axis=0)
    
    # Calculate weights (inverse frequency, clipped)
    class_weights = {}
    for i in range(num_classes):
        if class_counts[i] > 0:
            weight = num_samples / (num_classes * class_counts[i])
            class_weights[i] = min(weight, max_weight)
        else:
            class_weights[i] = 1.0
    
    return class_weights
