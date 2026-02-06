## ⚙️ PRE-REQUISITOS Y SETUP OBLIGATORIO

Antes de hacer reentrenamientos, asegurate que todos estos pasos están completos.

---

## 1️⃣ CREAR EXPERIMENT EN MLFLOW (OBLIGATORIO)

Antes de CUALQUIER reentrenamiento, debe existir el experimento con ID `401576597529460193`.

### Opción A: Desde Python (Recomendado)

```python
import mlflow
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
mlflow_dir = PROJECT_ROOT / 'runs' / 'mlflow'
mlflow_path = str(mlflow_dir).replace('\\', '/')

# Convertir a file:// URI para Windows
if len(mlflow_path) > 1 and mlflow_path[1] == ':':
    tracking_uri = f"file:///{mlflow_path}"
else:
    tracking_uri = f"file://{mlflow_path}"

mlflow.set_tracking_uri(tracking_uri)

# CREAR experiment
try:
    exp_id = mlflow.create_experiment(
        name='/Shared/Ultralytics',
        artifact_location=f"{tracking_uri}/401576597529460193"
    )
    print(f"✓ Experiment creado con ID: {exp_id}")
except Exception as e:
    print(f"⚠️  Experiment posiblemente ya existe: {e}")
```

**Ejecutar:**
```bash
python -c "
import mlflow
from pathlib import Path

PROJECT_ROOT = Path.cwd()
mlflow_dir = PROJECT_ROOT / 'runs' / 'mlflow'
mlflow_path = str(mlflow_dir).replace(chr(92), '/')
tracking_uri = f'file:///{mlflow_path}'

mlflow.set_tracking_uri(tracking_uri)
try:
    exp_id = mlflow.create_experiment(name='/Shared/Ultralytics', artifact_location=f'{tracking_uri}/401576597529460193')
    print(f'✓ Experiment {exp_id}')
except Exception as e:
    print(f'⚠️  {e}')
"
```

---

### Opción B: Desde MLflow UI

1. Iniciar MLflow UI:
   ```bash
   mlflow ui --backend-store-uri file:///c:/Users/jordy/OneDrive/Desktop/iaaaa/iajordy2/runs/mlflow --port 5001
   ```

2. Ir a: http://localhost:5001

3. Click en "+ New Experiment"

4. Configurar:
   - Name: `/Shared/Ultralytics`
   - Artifact Location: `file:///c:/Users/jordy/OneDrive/Desktop/iaaaa/iajordy2/runs/mlflow/401576597529460193`

5. Click "Create"

---

## 2️⃣ VERIFICAR ESTRUCTURA DE DIRECTORIOS

```bash
# Debe existir:
C:\Users\jordy\OneDrive\Desktop\iaaaa\iajordy2\
├── runs/
│   ├── mlflow/
│   │   └── 401576597529460193/  ← Este directorio DEBE existir
│   ├── train/
│   └── ...
├── models/
│   ├── best_improved.pt
│   └── ...
└── ...
```

**Crear si no existe:**
```bash
mkdir C:\Users\jordy\OneDrive\Desktop\iaaaa\iajordy2\runs\mlflow\401576597529460193
```

---

## 3️⃣ VERIFICAR QUE TIENE CORRECCIONES

Para reentrenar, necesitas MÍNIMO 5 correcciones acumuladas.

```bash
# Contar correcciones:
dir C:\Users\jordy\OneDrive\Desktop\iaaaa\iajordy2\data\corrections\*.json | find /c ".json"
```

**Debe mostrar:** >= 5 archivos

Si no tienes suficientes:
1. Ir a http://localhost:5000/advanced
2. Hacer correcciones manuales
3. O ejecutar [test_retrain_flow.py](test_retrain_flow.py) que crea correcciones simuladas

---

## 4️⃣ VALIDAR CONFIGURACIÓN MLFLOW

```bash
python validate_mlflow_config.py
```

**Debe mostrar:**
```
✓ Estructura de directorios
✓ Configuración de MLflow  
✓ Experimento específico existe
✓ Artifact location correcto
✓ Permisos de escritura
✓ Run de prueba

✓✓✓ TODAS LAS VALIDACIONES PASARON ✓✓✓
```

---

## 5️⃣ HACER TEST (Opcional pero Recomendado)

```bash
python test_retrain_flow.py
```

Esto hace un reentrenamiento COMPLETO con datos simulados para verificar que TODO funciona.

---

## ✅ CHECKLIST FINAL

Antes de usar en PRODUCCIÓN, verifica:

```
□ Experiment 401576597529460193 existe
□ Directorio runs/mlflow/401576597529460193/ existe
□ Tengo >= 5 correcciones en data/corrections/
□ validate_mlflow_config.py pasa TODAS las validaciones
□ test_retrain_flow.py se ejecuta exitosamente
□ MLflow UI muestra el experimento correcto
□ API puede conectarse (python app/run_server.py)
□ Puedo hacer POST a /api/model/retrain
```

Si TODOS los items tienen ✓, está listo para PRODUCCIÓN.

---

## 🚀 INICIO RÁPIDO

### 1. Terminal 1: Crear Experiment (UNA SOLA VEZ)
```bash
cd c:\Users\jordy\OneDrive\Desktop\iaaaa\iajordy2
python validate_mlflow_config.py  # Esto también crea run de prueba
```

### 2. Terminal 2: Verificar
```bash
python verification_checklist.py
```

### 3. Terminal 3: Iniciar API
```bash
python app/run_server.py
```

### 4. Terminal 4: Hacer Reentrenamiento
```bash
curl -X POST http://localhost:5000/api/model/retrain \
  -H "Content-Type: application/json" \
  -d '{"epochs": 5, "experiment_id": "401576597529460193"}'
```

### 5. Terminal 1: Ver en UI
```bash
mlflow ui --backend-store-uri file:///c:/Users/jordy/OneDrive/Desktop/iaaaa/iajordy2/runs/mlflow --port 5001
```

Luego: http://localhost:5001

---

## 🔍 TROUBLESHOOTING

### "Experiment not found"

**Solución:**
```python
# Crear desde notebook o terminal
import mlflow
mlflow.set_tracking_uri('file:///c:/Users/jordy/OneDrive/Desktop/iaaaa/iajordy2/runs/mlflow')
exp = mlflow.create_experiment(name='/Shared/Ultralytics')
print(exp)  # Debe mostrar: 401576597529460193
```

### "Permission denied"

**Solución:**
```bash
# Cerrar cualquier proceso usando los archivos
tasklist | findstr python
taskkill /IM python.exe /F  # Si es necesario matar procesos

# Verifica permisos de carpeta
icacls C:\Users\jordy\OneDrive\Desktop\iaaaa\iajordy2\runs /T
```

### "No corrections to train"

**Solución:** Agregar correcciones:
```python
# O ejecutar test
python test_retrain_flow.py

# O agregar manualmente desde frontend
# http://localhost:5000/advanced
```

---

## 📋 CHECKLIST DE PRODUCCIÓN

Después de validar TODO, para PRODUCCIÓN:

1. ✅ Backup de `/models` directory
2. ✅ Backup de `/runs/mlflow` directory
3. ✅ Documentar el experiment_id (401576597529460193)
4. ✅ Revisar logs de test_retrain_flow.py
5. ✅ Confirmar que MLflow UI muestra todo correcto
6. ✅ Test con API endpoint
7. ✅ Documentar proceso en wiki/confluence

---

## 📞 SOPORTE

Si algo falla, en este orden:

1. Ejecutar: `python validate_mlflow_config.py`
2. Leer output y seguir sugerencias
3. Si aún falla, revisar: [MLFLOW_FIX_EXPLANATION.md](MLFLOW_FIX_EXPLANATION.md)
4. Último recurso: `python verification_checklist.py` después de un reentrenamiento

---

**Estado:** ✅ LISTO PARA PRODUCCIÓN (después de completar checklist)
