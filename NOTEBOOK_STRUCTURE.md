# YOLO Object Detection Pipeline - Notebook Structure

## Overview
Pipeline completo para entrenamiento y predicción de modelos YOLO v8 con 3 clases: **person, car, dog**.

**Características principales:**
- ✓ Reproducible en cualquier PC (remoto o local)
- ✓ Datos reales (Pascal VOC 2007)
- ✓ Separación clara de responsabilidades
- ✓ MLflow tracking para reproducibilidad
- ✓ Sin código duplicado entre cuadernos

---

## Cuadernos (Notebooks)

### 1️⃣ **01_dataset_validation.ipynb** - Preparación de Datos (EDA)

**Responsabilidad ÚNICA:** Descargar, preparar y validar dataset

**Workflow:**
1. Crea estructura de directorios
2. Descarga Pascal VOC 2007 dataset (~500MB, primera vez)
3. Convierte XML a formato YOLO (bounding boxes normalizados)
4. Filtra solo 3 clases: person, car, dog
5. Divide en train/val/test (70/15/15)
6. Valida integridad de datos
7. Genera `data.yaml` para YOLO

**Salidas:**
- `data/images/train/*.jpg` + `data/labels/train/*.txt`
- `data/images/val/*.jpg` + `data/labels/val/*.txt`
- `data/images/test/*.jpg` + `data/labels/test/*.txt`
- `data/data.yaml` (config YOLO)

**Notas:**
- Primera ejecución: tarda ~5-10 minutos (descarga)
- Ejecutar de arriba a abajo sin interrupciones
- ✓ Reproducible - siempre genera mismo dataset

---

### 2️⃣ **02_train_yolo.ipynb** - Configuración de Modelo

**Responsabilidad ÚNICA:** Definir arquitectura y hiperparámetros

**Contenido (SOLO CONFIGURACIÓN, sin entrenamiento):**
1. Selecciona YOLOv8n (nano - eficiente)
2. Define 3 clases: person, car, dog
3. Especifica hiperparámetros:
   - Epochs: 50
   - Batch size: 16
   - Input size: 416x416
   - Learning rate: 0.01
   - Seed: 42 (reproducibilidad)
4. Genera/actualiza `data.yaml`
5. Imprime configuración para revisión

**Salidas:**
- `data/data.yaml` (actualizado)
- Configuración lista para usar en 03_training.ipynb

**Restricciones:**
- ❌ NO entrena
- ❌ NO ejecuta inferencia
- ❌ NO modifica pesos
- ❌ NO usa MLflow

**Notas:**
- Ejecución rápida (~30 segundos)
- Prerequisito para cuaderno 3

---

### 3️⃣ **03_training.ipynb** - Entrenamiento

**Responsabilidad ÚNICA:** Entrenar modelo con MLflow tracking

**Workflow:**
1. Carga configuración del cuaderno 2
2. Inicializa YOLO v8n con pesos COCO pretrained
3. Inicia MLflow experiment tracking
4. Ejecuta entrenamiento (50 epochs)
5. Valida modelo en validation set
6. Calcula métricas: mAP50, mAP50-95, precision, recall
7. Registra todo en MLflow

**Salidas:**
- `models/yolo_run/weights/best.pt` (mejor modelo)
- `mlruns/` (MLflow experiment tracking)

**Restricciones:**
- ❌ NO redefine arquitectura (usa del cuaderno 2)
- ❌ NO cambia número de clases
- ❌ NO incluye inferencia

**Notas:**
- Ejecución larga (~30-60 minutos, depende GPU)
- MLflow registra todos los hiperparámetros y métricas
- Puede pausarse y reanudarse

---

### 4️⃣ **04_prediction.ipynb** - Predicción/Inferencia

**Responsabilidad ÚNICA:** Ejecutar inferencia y visualizar resultados

**Workflow:**
1. Carga modelo entrenado (`best.pt` del cuaderno 3)
2. Configura parámetros: confidence=0.5, IoU=0.45
3. Realiza predicción batch en 10 primeras imágenes test
4. Analiza detecciones por clase
5. Visualiza resultados con bounding boxes
6. Prueba controlada en imagen individual

**Salidas:**
- Visualizaciones con bounding boxes
- Estadísticas de detecciones
- Análisis de confianza por clase

**Restricciones:**
- ❌ NO entrena
- ❌ NO modifica pesos
- ❌ NO usa MLflow
- ❌ NO incluye lógica de frontend

**Notas:**
- Ejecución rápida (~2-5 minutos)
- Solo lectura: no modifica nada
- Prerequisito: cuaderno 3 completo

---

## Flujo de Ejecución

```
┌─────────────────────────────────────────┐
│  01_dataset_validation.ipynb            │
│  Descargar Pascal VOC + Preparar datos  │
│  Salida: data/images + data/labels      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  02_train_yolo.ipynb                    │
│  Configurar modelo e hiperparámetros    │
│  Salida: data.yaml + configuración      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  03_training.ipynb                      │
│  Entrenar modelo con MLflow tracking    │
│  Salida: best.pt + métricas MLflow      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  04_prediction.ipynb                    │
│  Ejecutar predicción y visualizar       │
│  Salida: Resultados + gráficos          │
└─────────────────────────────────────────┘
```

---

## Estructura de Directorios

```
iajordy2/
├── notebooks/
│   ├── 01_dataset_validation.ipynb    ← Inicio aquí
│   ├── 02_train_yolo.ipynb
│   ├── 03_training.ipynb
│   └── 04_prediction.ipynb
├── data/
│   ├── images/
│   │   ├── train/        (400 imágenes)
│   │   ├── val/          (50 imágenes)
│   │   └── test/         (50 imágenes)
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── data.yaml         (config YOLO)
│   └── voc_raw/          (Pascal VOC descargado)
├── models/
│   └── yolo_run/
│       └── weights/
│           └── best.pt   (modelo entrenado)
├── mlruns/               (MLflow tracking)
└── requirements.txt

Classes (3):
- person  (ID: 0)
- car     (ID: 1)
- dog     (ID: 2)

YOLO Format:
<class_id> <x_center> <y_center> <width> <height>
(todos normalizados entre 0 y 1)
```

---

## Dependencias

Ver `requirements.txt`:
- torch >= 2.0.0
- ultralytics >= 8.0.0 (YOLO)
- mlflow >= 2.0.0 (tracking)
- pillow >= 9.0.0 (imágenes)
- pyyaml >= 6.0 (config)
- numpy >= 1.23.0
- jupyter >= 1.0.0

```bash
pip install -r requirements.txt
```

---

## Uso Remoto

El pipeline es **100% reproducible remotamente**:

```bash
# 1. Clonar repositorio
git clone <repo>
cd iajordy2

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar cuadernos en orden
# - 01_dataset_validation.ipynb     (descarga Pascal VOC automático)
# - 02_train_yolo.ipynb
# - 03_training.ipynb
# - 04_prediction.ipynb

# 4. Revisar resultados en mlruns/
```

**Notas importantes:**
- ✓ Primera ejecución del 01: descarga Pascal VOC (~500MB)
- ✓ Sin dependencias externas de datos
- ✓ Seed=42 para reproducibilidad
- ✓ MLflow almacena todos los experimentos localmente

---

## Estandarización

✅ **Verificaciones finales realizadas:**
- Cada cuaderno tiene responsabilidad única
- Sin duplicación de código
- Sin variables globales compartidas
- Ejecución secuencial garantizada
- Código comentado antiguo removido
- Prints necesarios solamente
- Validación integrada en cada paso
- MLflow tracking en cuaderno 3
- Dataset reproducible (Pascal VOC real)

---

## Mejoras Implementadas

1. **Dataset Real**: Pascal VOC 2007 en lugar de sintéticos
2. **Reproducibilidad**: Descarga automática, seed fijo
3. **Separación Clara**: Cada cuaderno = responsabilidad única
4. **MLflow Integration**: Tracking completo en 03_training
5. **Validación Integrada**: Checks en cada paso
6. **Documentación**: Workflow claro y comentado

---

## Próximos Pasos Opcionales

- Agregar validación en test set
- Implementar API Flask para predicción
- Crear dashboard de métricas con MLflow UI
- Agregar augmentación de datos
- Tuning de hiperparámetros

