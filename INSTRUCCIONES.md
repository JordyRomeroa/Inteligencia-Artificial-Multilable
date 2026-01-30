# 📋 INSTRUCCIONES PARA PROBAR EL PROYECTO

## ✅ Estado Actual

**Entorno Configurado:**
- ✅ Python 3.10.0 instalado
- ✅ Entorno virtual (`venv`) creado
- ✅ Todas las dependencias instaladas:
  - TensorFlow 2.15.1
  - Keras 3.12.0
  - Streamlit 1.53.1
  - Jupyter Notebook 7.5.3
  - Pandas, NumPy, Matplotlib, Seaborn
  - Scikit-learn

---

## 🚀 OPCIÓN 1: Ejecutar Notebooks (Jupyter)

### Paso 1: Activar el entorno virtual

```powershell
cd c:\Users\mlata\Documents\iajordy2
.\venv\Scripts\Activate.ps1
```

O usa el script batch:
```powershell
start_jupyter.bat
```

### Paso 2: Iniciar Jupyter Notebook

```powershell
python -m jupyter notebook --notebook-dir=notebooks
```

### Paso 3: Abre en tu navegador

Se abrirá automáticamente en: **http://localhost:8888**

### Paso 4: Ejecuta los notebooks en orden:

1. **`01_data_analysis.ipynb`** - Análisis y preparación de datos
2. **`02_modeling.ipynb`** - Diseño de la arquitectura
3. **`03_training_retraining.ipynb`** - Entrenamiento del modelo
4. **`04_prediction.ipynb`** - Predicciones y evaluación

---

## 🖥️ OPCIÓN 2: Ejecutar Aplicación Web (Streamlit)

### Activar entorno y lanzar app:

```powershell
cd c:\Users\mlata\Documents\iajordy2
.\venv\Scripts\Activate.ps1
python -m streamlit run app/app.py
```

O usa el script batch:
```powershell
start_streamlit.bat
```

**Se abrirá en:** http://localhost:8501

### Características de la app:
- 📤 Subir imágenes
- 🔮 Predicción multilabel en tiempo real
- 📊 Visualización de probabilidades
- ⚙️ Ajuste de threshold
- 💾 Descarga de resultados

---

## 🔍 OPCIÓN 3: Verificar Instalación

```powershell
cd c:\Users\mlata\Documents\iajordy2
.\venv\Scripts\Activate.ps1

# Ver todos los paquetes instalados
pip list

# Probar TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"

# Probar Streamlit
python -c "import streamlit; print(f'Streamlit {streamlit.__version__}')"

# Probar Jupyter
python -m jupyter --version
```

---

## 📂 Estructura del Proyecto

```
iajordy2/
├── notebooks/                      ← 4 Notebooks Jupyter
│   ├── 01_data_analysis.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_training_retraining.ipynb
│   └── 04_prediction.ipynb
│
├── app/                            ← Aplicación Web
│   ├── app.py
│   └── utils.py
│
├── models/                         ← Modelos entrenados (se crean)
├── data/                           ← Datos (coloca UECFood256 aquí)
│
├── venv/                           ← Entorno virtual
├── requirements.txt
├── README.md
├── start_jupyter.bat               ← Script para Jupyter
└── start_streamlit.bat             ← Script para Streamlit
```

---

## ⚠️ Notas Importantes

### Para usar el modelo entrenado:

1. Primero ejecuta `03_training_retraining.ipynb` para generar el modelo
2. Esto crea: `models/food_multilabel_final.h5`
3. La aplicación Streamlit lo usará automáticamente

### Para usar datos reales:

1. Descarga UECFood256 de Kaggle
2. Colócalo en `data/UECFood256/`
3. Ejecuta los notebooks para procesar los datos

### Si hay errores de módulos:

```powershell
# Actualizar pip
python -m pip install --upgrade pip

# Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

---

## 💡 Flujo de Trabajo Recomendado

1. **Primero**: Ejecuta los notebooks en orden
   - `01_data_analysis.ipynb` - Carga y explora datos
   - `02_modeling.ipynb` - Define el modelo
   - `03_training_retraining.ipynb` - Entrena el modelo ⏱️ (puede tardar)
   - `04_prediction.ipynb` - Prueba predicciones

2. **Luego**: Usa la app web Streamlit
   - Carga imágenes
   - Obtén predicciones multilabel
   - Ajusta threshold

---

## 🐛 Solución de Problemas

### Error: "comando no encontrado"
```powershell
# Asegúrate de estar en el directorio correcto
cd c:\Users\mlata\Documents\iajordy2

# Activa el entorno
.\venv\Scripts\Activate.ps1
```

### Error: "No module named jupyter"
```powershell
pip install jupyter notebook ipykernel --upgrade
```

### Error: "Modelo no encontrado"
→ Primero ejecuta `03_training_retraining.ipynb`

### Error: "No se abre Jupyter"
```powershell
python -m jupyter notebook --notebook-dir=notebooks --ip=127.0.0.1
```

---

## 📞 Comandos Útiles

```powershell
# Activar entorno
.\venv\Scripts\Activate.ps1

# Desactivar entorno
deactivate

# Ver paquetes instalados
pip list

# Actualizar paquete específico
pip install --upgrade tensorflow

# Limpiar caché
pip cache purge

# Crear nuevo notebook
jupyter notebook

# Listar procesos Python
tasklist | findstr python
```

---

## 🎯 Resumen

✅ **Entorno completamente configurado**  
✅ **Todas las dependencias instaladas**  
✅ **Listo para probar Notebooks y App Web**  

**Próximo paso:** Ejecuta uno de los comandos anteriores para empezar! 🚀
