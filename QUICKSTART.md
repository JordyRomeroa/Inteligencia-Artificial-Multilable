# 🚀 Guía de Inicio Rápido

## Instalación y Configuración

### 1. Crear entorno virtual

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 3. Descargar dataset

Descargar UECFood256 desde Kaggle y extraer en `data/UECFood256/`

O usar Kaggle API:
```powershell
kaggle datasets download -d rkuo2000/uecfood256 -p data/ --unzip
```

### 4. Ejecutar notebooks

```powershell
jupyter notebook
```

Ejecutar en orden:
1. `notebooks/01_data_analysis.ipynb`
2. `notebooks/02_modeling.ipynb`
3. `notebooks/03_training_retraining.ipynb`
4. `notebooks/04_prediction.ipynb`

### 5. Ejecutar aplicación web

```powershell
streamlit run app/app.py
```

## Estructura Esperada

```
iajordy2/
├── data/
│   └── UECFood256/        ← Dataset descargado aquí
├── models/                ← Modelos se guardarán aquí
├── notebooks/             ← 4 notebooks Jupyter
└── app/                   ← Aplicación Streamlit
```

## Problemas Comunes

**Error: Modelo no encontrado**
→ Ejecutar `03_training_retraining.ipynb` primero

**Error: Dataset no encontrado**
→ Descargar UECFood256 y colocar en `data/`

**Error: GPU no detectada**
→ Opcional, el proyecto funciona en CPU
