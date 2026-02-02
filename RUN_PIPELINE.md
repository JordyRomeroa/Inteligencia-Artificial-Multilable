# 🚀 EJECUTAR PIPELINE COMPLETO - YOLO Object Detection

## Estado del Sistema
- ✅ Ambiente virtual configurado (.venv)
- ✅ Dependencias instaladas (requirements.txt)
- ✅ 4 cuadernos preparados y estandarizados
- ✅ Dataset limpio (carpetas vaciadas)

---

## 📋 Secuencia de Ejecución

Ejecuta los cuadernos en este orden exacto:

### 1️⃣ Cuaderno 1: Preparación de Dataset
**Archivo:** `notebooks/01_dataset_validation.ipynb`

**Qué hace:**
- Descarga/prepara dataset (Pascal VOC o sintético)
- Valida estructura YOLO
- Crea `data/` con train/val/test splits
- Genera `data/data.yaml`

**Tiempo:** 2-10 minutos (primera ejecución)
**Prerequisito:** Ninguno

✅ **Estado:** COMPLETADO

---

### 2️⃣ Cuaderno 2: Configuración de Modelo
**Archivo:** `notebooks/02_train_yolo.ipynb`

**Qué hace:**
- Define arquitectura: YOLOv8n
- Especifica hiperparámetros (epochs, batch size, learning rate)
- Genera configuración final

**Tiempo:** <1 minuto
**Prerequisito:** Cuaderno 1 completado

**Cómo ejecutar:**
1. Abre Jupyter: `jupyter notebook notebooks/`
2. Abre `02_train_yolo.ipynb`
3. Click "Run All" o ejecuta cada celda

---

### 3️⃣ Cuaderno 3: Entrenamiento
**Archivo:** `notebooks/03_training.ipynb`

**Qué hace:**
- Descarga pesos COCO preentrenados
- Ejecuta 50 epochs de entrenamiento
- Valida modelo en validation set
- Registra métricas con MLflow
- Guarda `models/yolo_run/weights/best.pt`

**Tiempo:** 30-60 minutos (con GPU: ~10-15 min)
**Prerequisito:** Cuadernos 1 y 2 completados

⚠️ **IMPORTANTE:** Este cuaderno tarda - ejecuta cuando puedas esperar

---

### 4️⃣ Cuaderno 4: Predicción/Inferencia
**Archivo:** `notebooks/04_prediction.ipynb`

**Qué hace:**
- Carga modelo entrenado
- Realiza predicción batch en test images
- Visualiza resultados con bounding boxes
- Analiza estadísticas por clase

**Tiempo:** 2-5 minutos
**Prerequisito:** Cuaderno 3 completado

---

## 🎯 Forma Rápida (Terminal)

```bash
# Navegar al directorio
cd c:\Users\mlata\Documents\iajordy2

# Activar ambiente virtual
.venv\Scripts\Activate.ps1

# Ejecutar cuadernos en orden (notebook/código)
jupyter nbconvert --to notebook --execute notebooks/01_dataset_validation.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_train_yolo.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_training.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_prediction.ipynb
```

---

## 📊 Salidas Esperadas

Después de completar todo:

```
iajordy2/
├── data/
│   ├── images/train/        (400 imágenes)
│   ├── images/val/          (50 imágenes)
│   ├── images/test/         (50 imágenes)
│   ├── labels/...
│   └── data.yaml
├── models/
│   └── yolo_run/
│       └── weights/
│           ├── best.pt      ✅ Modelo entrenado
│           └── last.pt
├── mlruns/                  ✅ Experimentos MLflow
│   └── [tracking data]
└── notebooks/
    ├── 01_dataset_validation.ipynb  ✅ Ejecutado
    ├── 02_train_yolo.ipynb          ✅ Ejecutado
    ├── 03_training.ipynb            ✅ Ejecutado
    └── 04_prediction.ipynb          ✅ Ejecutado
```

---

## 🔍 Verificación

Después de cada cuaderno:

**01_dataset_validation.ipynb:**
- ✓ Carpetas `data/images/` y `data/labels/` con archivos
- ✓ `data/data.yaml` creado

**02_train_yolo.ipynb:**
- ✓ Salida mostrando configuración
- ✓ Hiperparámetros correctos (50 epochs, batch 16)

**03_training.ipynb:**
- ✓ `models/yolo_run/weights/best.pt` creado
- ✓ `mlruns/` con experimentos registrados
- ✓ Métricas mostradas (mAP50, precision, recall)

**04_prediction.ipynb:**
- ✓ Visualizaciones con bounding boxes
- ✓ Análisis de detecciones por clase
- ✓ Estadísticas de confianza

---

## ⚠️ Solución de Problemas

**Error: "Model not found at best.pt"**
→ Asegúrate de completar 03_training.ipynb primero

**Error: "data.yaml not found"**
→ Ejecuta 01_dataset_validation.ipynb nuevamente

**Error: CUDA/GPU**
→ El pipeline usa CPU si GPU no disponible (más lento)

**Dataset descarga falló**
→ Sistema usa fallback sintético automáticamente

---

## 📝 Notas Importantes

- **Orden obligatorio:** 01 → 02 → 03 → 04
- **Sin saltar:** Cada cuaderno depende del anterior
- **Limpieza:** Carpetas `data/`, `models/`, `mlruns/` ya fueron limpiadas
- **Reproducible:** Mismos resultados cada ejecución (seed=42)
- **MLflow tracking:** Ver métricas con `mlflow ui` desde terminal

---

## 🎓 Próximos Pasos

Después de completar el pipeline:

1. **Revisar métricas:** `mlflow ui`
2. **Mejorar modelo:** Ajustar hiperparámetros en 02_train_yolo
3. **Usar en producción:** API Flask ya preparada en `app/`
4. **Reentrenar:** Vuelve a ejecutar desde 03_training

---

¿Listo? **Comienza con el Cuaderno 1** 🚀

