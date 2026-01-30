# 🏗️ Arquitectura de la Aplicación Web - Clasificación Multilabel

## 📐 Diagrama General del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    NAVEGADOR WEB (Cliente)                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Interfaz HTML/CSS/JS                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │   │
│  │  │   Tab 1      │  │   Tab 2      │  │   Tab 3    │ │   │
│  │  │   Individual │  │   Batch      │  │Correcciones│ │   │
│  │  └──────────────┘  └──────────────┘  └────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↕ HTTP/JSON
┌─────────────────────────────────────────────────────────────┐
│                     SERVIDOR Flask (api.py)                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │             Router de Endpoints                      │   │
│  │  GET /                    → Página principal         │   │
│  │  POST /predict            → Predicción individual   │   │
│  │  POST /save_correction    → Guardar corrección      │   │
│  │  POST /retrain            → Reentrenar modelo       │   │
│  │  POST /batch_predict      → Predicción en batch     │   │
│  │  GET /get_corrections     → Historial              │   │
│  │  GET /health              → Estado de salud         │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │       Funciones Principales (utils.py)              │   │
│  │  • preprocess_image()      → Prepara imagen         │   │
│  │  • predict_multilabel()    → Obtiene predicciones   │   │
│  │  • incremental_retrain()   → Fine-tune del modelo   │   │
│  │  • focal_loss()            → Función de pérdida     │   │
│  │  • calculate_class_weights() → Pesos por clase      │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Carga de Modelos y Datos                 │   │
│  │  • Modelo: voc_multilabel_final.h5 (TensorFlow)     │   │
│  │  • Clases: classes.json (20 categorías)             │   │
│  │  • Correcciones: data/corrections/*.json            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↕ Filesystem
┌─────────────────────────────────────────────────────────────┐
│                    ALMACENAMIENTO DE DATOS                  │
│  ┌───────────────┐ ┌──────────────┐ ┌────────────────────┐ │
│  │   Modelos     │ │ Correcciones │ │    Imágenes        │ │
│  │ /models/      │ │ /data/...    │ │  /data/uploads/    │ │
│  │  *.h5         │ │  *.json      │ │  (temporales)      │ │
│  └───────────────┘ └──────────────┘ └────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Flujos de Datos por Endpoint

### 1️⃣ Flujo: PREDICCIÓN INDIVIDUAL (/predict)

```
Cliente:
  1. Selecciona imagen (JPG/PNG)
  2. Ajusta threshold (0.1-0.9)
  3. Envía: POST /predict
                  ├─ file: binary image data
                  └─ threshold: float

Servidor:
  1. Recibe archivo en upload_folder/
  2. Abre imagen con PIL
  3. Resize a 224x224
  4. Normaliza [0, 255] → [0, 1]
  5. Expande dimensión batch: (224,224,3) → (1,224,224,3)
  6. Pasa por modelo TensorFlow
  7. Obtiene output: (1, 20) probabilidades
  8. Aplica threshold: prob >= threshold?
  9. Filtra clases positivas
 10. Ordena por probabilidad DESC
 11. Crea respuesta JSON

Respuesta:
  {
    "success": true,
    "predictions": [
      {"label": "person", "confidence": 0.95},
      {"label": "dog", "confidence": 0.87},
      ...
    ]
  }

Cliente:
  1. Recibe JSON
  2. Renderiza predicciones con barras de confianza
  3. Muestra checkboxes para corregir
  4. Colores: Verde (>70%), Amarillo (40-70%), Rojo (<40%)
```

### 2️⃣ Flujo: GUARDAR CORRECCIÓN (/save_correction)

```
Cliente:
  1. Usuario selecciona etiquetas correctas (checkboxes)
  2. Haz clic: "Guardar Corrección"
  3. Envía: POST /save_correction
             ├─ filename: "imagen.jpg"
             └─ corrected_labels: ["person", "dog"]

Servidor:
  1. Recibe corrección
  2. Crea JSON:
     {
       "filename": "imagen.jpg",
       "correct_labels": ["person", "dog"],
       "timestamp": "2024-01-30T10:30:00"
     }
  3. Guarda en data/corrections/{filename}_correction.json
  4. Responde: {"success": true}

Resultado:
  ✓ Corrección guardada para reentrenamiento posterior
  ✓ Se acumula en data/corrections/
  ✓ Visible en Tab "Correcciones"
```

### 3️⃣ Flujo: REENTRENAMIENTO (/retrain)

```
Cliente:
  1. Haz clic: "Reentrenar Modelo"
  2. Confirma diálogo
  3. Envía: POST /retrain
             └─ epochs: 5 (default)

Servidor:
  1. Lista todos los archivos: data/corrections/*_correction.json
  2. Para cada corrección:
     - Lee filename
     - Carga imagen: data/uploads/{filename}
     - Preprocesa: resize 224x224, normaliza
     - Obtiene label_vector: [0, 1, 0, ..., 1] (20 dims)
  3. Acumula en arrays:
     - images: (N, 224, 224, 3)
     - labels: (N, 20)
  4. Llama: incremental_retrain(model, images, labels)
  5. En utils.py:
     a. Congela todas las capas excepto últimas 4
     b. Compila modelo:
        - Optimizer: Adam(lr=1e-5)
        - Loss: focal_loss (gamma=2.0)
     c. Entrena: model.fit(images, labels, epochs=5)
     d. Calcula pesos de clases automáticamente
  6. Guarda modelo actualizado: models/voc_multilabel_final.h5
  7. Responde:
     {
       "success": true,
       "samples": 10,
       "final_loss": 0.245
     }

Cliente:
  1. Recibe confirmación
  2. Automáticamente predice imagen actual de nuevo
  3. Ve mejora en predicciones
```

### 4️⃣ Flujo: PREDICCIÓN BATCH (/batch_predict)

```
Cliente:
  1. Selecciona múltiples imágenes (5, 10, 20...)
  2. Haz clic: "Predecir Todo"
  3. Envía: POST /batch_predict
             ├─ files: [file1, file2, ..., fileN]
             └─ threshold: 0.5

Servidor:
  Para cada imagen:
    1. Guarda archivo
    2. Preprocesa
    3. Predice
    4. Aplica threshold
    5. Crea resultado
  
  Retorna array de resultados:
  {
    "success": true,
    "results": [
      {
        "filename": "img1.jpg",
        "predictions": [{"label": "person", "confidence": 0.95}]
      },
      {
        "filename": "img2.jpg",
        "predictions": [{"label": "dog", "confidence": 0.88}]
      }
    ]
  }

Cliente:
  1. Renderiza tabla/grilla con resultados
  2. Para cada imagen: muestra etiquetas predichas
  3. Botón "Corregir" → va a Tab Individual con esa imagen
```

### 5️⃣ Flujo: OBTENER CORRECCIONES (/get_corrections)

```
Cliente:
  1. Abre Tab "Correcciones"
  2. Haz clic: "Actualizar"
  3. Envía: GET /get_corrections

Servidor:
  1. Lista todos: data/corrections/*_correction.json
  2. Lee cada uno:
     {
       "filename": "img.jpg",
       "correct_labels": ["person", "dog"],
       "timestamp": "..."
     }
  3. Prepara respuesta:
     {
       "success": true,
       "total": 10,
       "corrections": [
         {"filename": "...", "corrected_labels": [...], "timestamp": "..."},
         ...
       ]
     }

Cliente:
  1. Renderiza lista de correcciones
  2. Muestra etiquetas como tags
  3. Permite ver historial de cambios
```

## 🧠 Arquitectura del Modelo

```
┌─────────────────────────────────────┐
│   Input: Imagen (224, 224, 3)       │
│   Range: [0, 1] normalized          │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│  EfficientNetB0 (Backbone)          │
│  - Pretrained ImageNet              │
│  - Transfer Learning                │
│  - Extrae features (1280 dims)      │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│  Global Average Pooling             │
│  Output: (1280,)                    │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│  Dense(512, ReLU) + Dropout(0.5)    │
│  Output: (512,)                     │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│  Dense(256, ReLU) + Dropout(0.3)    │
│  Output: (256,)                     │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│  Dense(20, Sigmoid)                 │
│  Output: (20,) probabilities        │
│  Range: [0, 1] per class            │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│  Thresholding (configurable)        │
│  prob >= 0.5 → Positivo             │
│  prob < 0.5  → Negativo             │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│  Output: Etiquetas detectadas       │
│  ["person", "dog", "car"]           │
└─────────────────────────────────────┘
```

## 📊 Estructura de Datos

### Imagen Input
```
Shape: (1, 224, 224, 3)
Dtype: float32
Range: [0, 1]
Format: RGB
```

### Predicción Output
```
Shape: (1, 20)
Dtype: float32
Range: [0, 1] (probabilidades)
Significado: Para cada una de las 20 clases VOC
```

### Corrección (JSON)
```json
{
  "filename": "foto.jpg",
  "correct_labels": ["person", "dog"],
  "timestamp": "2024-01-30T10:30:45"
}
```

### Información de Clase
```json
[
  "aeroplane",
  "bicycle",
  "bird",
  "boat",
  "bottle",
  "bus",
  "car",
  "cat",
  "chair",
  "cow",
  "diningtable",
  "dog",
  "horse",
  "motorbike",
  "person",
  "pottedplant",
  "sheep",
  "sofa",
  "train",
  "tvmonitor"
]
```

## 🔐 Seguridad y Validación

```
Entrada (Cliente):
  ├─ Validación HTML5: type="file" accept="image/*"
  └─ Límite de tamaño: 16 MB

Servidor (Flask):
  ├─ Validación extensión: {png, jpg, jpeg}
  ├─ Límite CONFIG: MAX_CONTENT_LENGTH = 16 MB
  ├─ Sanitización: secure_filename()
  ├─ Tipo MIME: image/*
  └─ Manejo de errores: try/except

Respuestas:
  ├─ JSON válido
  ├─ Status HTTP apropiados
  ├─ Manejo de excepciones
  └─ Logs de error
```

## 💾 Ciclo de Vida de Archivos

### Imagen Subida
```
1. Cliente → upload → data/uploads/{filename}
2. Servidor predice
3. (Temporal) Se mantiene en uploads
4. Al reentrenar: Se lee desde uploads
5. Puedes limpiar manualmente después
```

### Corrección Guardada
```
1. Cliente envía corrección → Servidor
2. Servidor crea JSON
3. Guarda en data/corrections/{filename}_correction.json
4. Se acumula en esa carpeta
5. Al reentrenar: Se leen todas las correcciones
6. Persisten para futuros reentrenamientos
```

### Modelo Entrenado
```
1. Inicial: models/voc_multilabel_final.h5 (1.8 MB)
2. Cargado en memoria al iniciar servidor
3. Usado para todas las predicciones
4. Actualizado después de cada /retrain
5. Respaldado automáticamente
```

## ⚙️ Parámetros Clave

```python
# Imagen
TARGET_SIZE = (224, 224)
NORMALIZATION_RANGE = [0, 1]

# Modelo
NUM_CLASSES = 20
OUTPUT_ACTIVATION = 'sigmoid'
LOSS_FUNCTION = 'focal_loss'

# Focal Loss
GAMMA = 2.0
ALPHA = 0.25

# Reentrenamiento
LEARNING_RATE = 1e-5
EPOCHS_DEFAULT = 5
BATCH_SIZE = 16
FROZEN_LAYERS = -4  # Últimas 4 capas descongeladas

# Archivo
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
```

## 🚀 Rendimiento Esperado

```
Predicción Individual:
  - Preprocesamiento: ~10ms
  - Predicción: ~100-200ms (CPU)
  - Total: ~200-300ms

Predicción Batch (10 imágenes):
  - Total: ~500-1000ms

Reentrenamiento:
  - Con 10 correcciones, 5 epochs:
  - CPU: 1-3 minutos
  - GPU: 10-30 segundos
```

---

**Esta arquitectura permite:**
- ✅ Interactividad en tiempo real
- ✅ Actualización de modelo sin parar servidor
- ✅ Escalabilidad a múltiples usuarios
- ✅ Persistencia de correcciones
- ✅ Mejora continua del modelo
