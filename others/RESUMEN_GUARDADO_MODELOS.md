# ✅ RESUMEN: Sistema de Guardado de Modelos Completo

## 📍 Ubicaciones donde se guardan los modelos reentrenados

Cada vez que ejecutas un reentrenamiento, el modelo se guarda en **4 ubicaciones diferentes**:

### 1️⃣ **Carpeta models/** (Acceso directo versionado)
```
Path: c:\Users\jordy\OneDrive\Desktop\iaaaa\iajordy2\models\retrained_v8.pt
Tamaño: 21.47 MB
Uso: Modelo final versionado listo para cargar en producción
```

### 2️⃣ **Carpeta runs/train/** (Copia rápida)
```
Path: c:\Users\jordy\OneDrive\Desktop\iaaaa\iajordy2\runs\train\retrained_v8.pt
Tamaño: 21.47 MB
Uso: Copia directa para acceso rápido sin navegar subdirectorios
```

### 3️⃣ **Carpeta runs/train/retrain_vX/** (Entrenamiento completo)
```
Path: c:\Users\jordy\OneDrive\Desktop\iaaaa\iajordy2\runs\train\retrain_v8\
Contenido:
  - weights/best.pt (modelo mejor)
  - weights/last.pt (último checkpoint)
  - args.yaml (configuración de entrenamiento)
  - *.png (plots: confusion matrix, curves, etc.)
Uso: Directorio completo con todos los artefactos de entrenamiento
```

### 4️⃣ **MLflow Artifacts** (Tracking de experimentos)
```
Path: c:\Users\jordy\OneDrive\Desktop\iaaaa\iajordy2\runs\mlflow\401576597529460193\
Estructura:
  - [RUN_ID]/artifacts/models/retrained_v8.pt
  - [RUN_ID]/artifacts/plots/*.png
  - [RUN_ID]/artifacts/retraining_dataset_metadata.json
  - [RUN_ID]/artifacts/corrections_applied.json
  - [RUN_ID]/artifacts/data.yaml

Ejemplo (última ejecución):
  runs\mlflow\401576597529460193\2edb8f6da7da47dd85cb4a93728f4583\artifacts\models\retrained_v8.pt

Uso: Trazabilidad completa del experimento con métricas, parámetros y artifacts
```

---

## 🎯 Configuración del Sistema

**Experiment ID:** `401576597529460193`  
**Tracking URI:** `file:///C:/Users/jordy/OneDrive/Desktop/iaaaa/iajordy2/runs/mlflow`  
**Artifact Location:** `file:///c:/Users/jordy/OneDrive/Desktop/iaaaa/iajordy2/runs/mlflow/401576597529460193`

---

## ✅ Verificación Rápida

Para verificar todas las ubicaciones de la última versión:
```bash
python verify_model_locations.py
```

Para verificar una versión específica:
```bash
python verify_model_locations.py 8
```

---

## 📊 Ver en MLflow UI

Para visualizar todos los experimentos y artifacts:

```bash
C:/Users/jordy/OneDrive/Desktop/iaaaa/iajordy2/.venv/Scripts/python.exe -m mlflow ui --backend-store-uri file:///C:/Users/jordy/OneDrive/Desktop/iaaaa/iajordy2/runs/mlflow --port 5001
```

Luego abre: **http://localhost:5001**

---

## 🔄 Flujo Completo del Guardado

```
REENTRENAMIENTO
      ↓
┌─────────────────────────────────────────┐
│ PASO 6: model.train()                   │
│ → Ultralytics guarda en runs/train/     │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│ PASO 9: Copiar modelo                   │
│ → De runs/train/retrain_vX/weights/     │
│ → A models/retrained_vX.pt              │
│ → A runs/train/retrained_vX.pt          │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│ PASO 10: MLflow log_artifact()          │
│ → Registra en experiment 401576...      │
│ → Guarda modelo + plots + configs       │
└─────────────────────────────────────────┘
```

---

## 📝 Metadata Adicional

Cada modelo incluye un archivo de metadata:
```
models/retrained_v8_metadata.json
```

Contenido:
- Versión del modelo
- Tipo de entrenamiento
- Número de muestras
- Timestamp
- Métricas finales

---

## ⚠️ Importante

- **NUNCA** borrar `runs/mlflow/401576597529460193/` - contiene todo el historial
- Los modelos en `models/` son los listos para producción
- La copia en `runs/train/` es para acceso rápido sin subdirectorios
- MLflow artifacts incluyen TODO (modelo + datos + plots)

---

✅ **Sistema completamente funcional y probado**
