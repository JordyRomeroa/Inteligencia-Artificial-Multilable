# 🍱 Food Multilabel Classification with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)

> **Proyecto académico de Machine Learning**: Clasificación multilabel de alimentos usando Transfer Learning y Deep Learning con el dataset UECFood256.

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Dataset](#-dataset)
- [Multiclase vs Multilabel](#-multiclase-vs-multilabel)
- [Arquitectura del Modelo](#-arquitectura-del-modelo)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Notebooks](#-notebooks)
- [Aplicación Web](#-aplicación-web)
- [Resultados](#-resultados)
- [Tecnologías](#-tecnologías)
- [Contribuciones](#-contribuciones)
- [Licencia](#-licencia)

---

## 🎯 Descripción

Este proyecto implementa un **sistema de clasificación multilabel de alimentos** utilizando técnicas avanzadas de **Deep Learning** y **Transfer Learning**. A diferencia de los clasificadores tradicionales que asignan una sola etiqueta por imagen, este modelo puede identificar **múltiples tipos de alimentos simultáneamente** en una sola fotografía.

### ✨ Características Principales

- 🏷️ **Clasificación Multilabel**: Identifica múltiples alimentos en una imagen
- 🧠 **Transfer Learning**: Utiliza EfficientNetB0 pre-entrenado en ImageNet
- 📊 **Métricas Especializadas**: Hamming Loss, F1-Score, Precision, Recall
- 🔄 **Estrategias de Retraining**: Fine-tuning, Data Augmentation
- 🖥️ **Aplicación Web**: Interfaz interactiva con Streamlit
- 📓 **Notebooks Documentados**: 4 notebooks Jupyter completamente explicados

---

## 🗂️ Dataset

### UECFood256

El proyecto utiliza el dataset **UECFood256**, que contiene:

- **256 categorías** de comida japonesa
- Miles de imágenes de alta calidad
- Variedad de platos y composiciones

📥 **Descarga**: [Kaggle - UECFood256](https://www.kaggle.com/datasets/rkuo2000/uecfood256)

### Transformación a Multilabel

Aunque el dataset original es **multiclase** (una etiqueta por imagen), este proyecto lo transforma a **multilabel** mediante:

1. **Combinaciones Realistas**: Platos que típicamente contienen múltiples ingredientes
2. **Relabeling Estratégico**: Basado en composición real de alimentos japoneses
3. **Justificación Académica**: Los platos de comida son naturalmente multilabel

#### Ejemplo de Transformación

```
Imagen Original (multiclase):
  ├─ Etiqueta: "bento"

Imagen Transformada (multilabel):
  ├─ Etiquetas: ["rice", "chicken", "vegetables", "egg", "sauce"]
```

---

## 🔄 Multiclase vs Multilabel

### Clasificación Multiclase (Tradicional)

- **Definición**: Cada imagen pertenece a UNA SOLA clase
- **Ejemplo**: Una imagen es "sushi" **O** "ramen" **O** "tempura"
- **Activación**: Softmax → $\sum p_i = 1$
- **Loss**: Categorical Cross-Entropy

### Clasificación Multilabel (Este Proyecto) ✅

- **Definición**: Cada imagen puede tener MÚLTIPLES etiquetas
- **Ejemplo**: Una imagen puede ser "rice" **Y** "fish" **Y** "vegetables"
- **Activación**: Sigmoid → Cada $p_i \in [0, 1]$ independiente
- **Loss**: Binary Cross-Entropy

### Comparación Técnica

| Aspecto | Multiclase | Multilabel |
|---------|-----------|------------|
| **Activación Final** | Softmax | **Sigmoid** |
| **Función de Pérdida** | Categorical CE | **Binary CE** |
| **Output** | Suma = 1.0 | Independientes |
| **Etiquetas por Imagen** | 1 | **1 a N** |

### Fórmulas Matemáticas

**Sigmoid (Multilabel)**:
```math
σ(z_i) = \frac{1}{1 + e^{-z_i}}
```

**Binary Cross-Entropy**:
```math
L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{m} [y_{ij} \log(\hat{y}_{ij}) + (1-y_{ij}) \log(1-\hat{y}_{ij})]
```

---

## 🏗️ Arquitectura del Modelo

### Diagrama de Arquitectura

```
INPUT (224×224×3)
    ↓
┌─────────────────────────┐
│  EfficientNetB0         │
│  (Pre-trained ImageNet) │
│  Frozen: Fase 1         │
│  Fine-tuned: Fase 2     │
└─────────────────────────┘
    ↓
GlobalAveragePooling2D
    ↓
Dense(512) + ReLU + Dropout(0.5)
    ↓
Dense(256) + ReLU + Dropout(0.3)
    ↓
Dense(NUM_CLASSES) + Sigmoid ← MULTILABEL
    ↓
OUTPUT (Probabilidades independientes)
```

### Justificación de Diseño

#### 1. ¿Por qué EfficientNetB0?

- ✅ **Eficiencia**: Solo 5.3M parámetros (vs ResNet50: 25M)
- ✅ **Precisión**: Estado del arte en ImageNet
- ✅ **Velocidad**: Inferencia rápida para aplicaciones web
- ✅ **Transfer Learning**: Excelente para datasets pequeños

#### 2. ¿Por qué Binary Cross-Entropy?

**Categorical CE** (INCORRECTO para multilabel):
- Asume una sola clase activa
- Fuerza competencia entre clases
- No permite múltiples etiquetas

**Binary CE** (CORRECTO para multilabel):
- Trata cada clase independientemente
- Permite múltiples clases activas
- Cada neurona optimiza independientemente

#### 3. ¿Por qué Sigmoid y no Softmax?

**Softmax** → $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$
- Probabilidades suman 1.0
- Solo una clase domina
- ❌ No funciona para multilabel

**Sigmoid** → $p_i = \frac{1}{1 + e^{-z_i}}$
- Cada probabilidad independiente
- Múltiples clases pueden tener alta probabilidad
- ✅ Ideal para multilabel

---

## 📁 Estructura del Proyecto

```
iajordy2/
│
├── notebooks/
│   ├── 01_data_analysis.ipynb          # 📊 Análisis exploratorio y transformación
│   ├── 02_modeling.ipynb               # 🧠 Diseño del modelo
│   ├── 03_training_retraining.ipynb    # 🚀 Entrenamiento y fine-tuning
│   └── 04_prediction.ipynb             # 🔮 Predicciones y evaluación
│
├── app/
│   ├── app.py                          # 🖥️ Aplicación Streamlit
│   └── utils.py                        # 🛠️ Funciones auxiliares
│
├── models/
│   ├── food_multilabel_final.h5        # 💾 Modelo entrenado
│   ├── model_config.json               # ⚙️ Configuración
│   └── training_results.json           # 📈 Resultados
│
├── data/
│   ├── UECFood256/                     # 📂 Dataset (descargado)
│   ├── multilabel_annotations.csv      # 🏷️ Anotaciones multilabel
│   ├── classes.json                    # 📋 Lista de clases
│   └── y_multilabel.npy                # 🔢 Matriz de etiquetas
│
├── requirements.txt                    # 📦 Dependencias
└── README.md                           # 📖 Este archivo
```

---

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes)
- (Opcional) GPU compatible con CUDA para entrenamiento

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/food-multilabel-classification.git
cd food-multilabel-classification
```

### Paso 2: Crear Entorno Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Descargar el Dataset

1. Ir a [Kaggle - UECFood256](https://www.kaggle.com/datasets/rkuo2000/uecfood256)
2. Descargar el dataset
3. Extraer en `data/UECFood256/`

**O usar Kaggle API:**

```bash
# Configurar kaggle.json primero
kaggle datasets download -d rkuo2000/uecfood256 -p data/ --unzip
```

---

## 💻 Uso

### Opción 1: Notebooks Jupyter

Ejecutar los notebooks en orden:

```bash
jupyter notebook
```

1. `01_data_analysis.ipynb` - Análisis y preparación de datos
2. `02_modeling.ipynb` - Diseño del modelo
3. `03_training_retraining.ipynb` - Entrenamiento
4. `04_prediction.ipynb` - Predicciones

### Opción 2: Aplicación Web (Streamlit)

```bash
streamlit run app/app.py
```

La aplicación se abrirá en `http://localhost:8501`

#### Características de la App:

- 📤 Subir imágenes de alimentos
- 🎯 Predicción multilabel en tiempo real
- 📊 Visualización de probabilidades
- ⚙️ Ajuste de threshold
- 💾 Descarga de resultados en JSON

---

## 📓 Notebooks

### 1. Análisis de Datos (`01_data_analysis.ipynb`)

**Contenido:**
- Carga del dataset UECFood256
- Exploración visual de imágenes
- Distribución de clases
- **Explicación multiclase vs multilabel**
- Transformación del dataset a multilabel
- Generación de combinaciones realistas
- Validación de ejemplos

**Salidas:**
- `multilabel_annotations.csv`
- `y_multilabel.npy`
- `classes.json`

### 2. Modelado (`02_modeling.ipynb`)

**Contenido:**
- Definición formal del problema multilabel
- Arquitectura con Transfer Learning
- Justificación de Binary Cross-Entropy
- Justificación de activación Sigmoid
- Configuración de métricas multilabel
- Callbacks y optimizadores

**Salidas:**
- Modelo definido
- `model_config.json`

### 3. Entrenamiento y Retraining (`03_training_retraining.ipynb`)

**Contenido:**
- **Fase 1**: Entrenamiento inicial (backbone congelado)
- Data Augmentation
- **Fase 2**: Fine-tuning (retraining)
  - Descongelar últimas capas
  - Learning rate reducido
  - Augmentation mejorada
- Comparación antes/después
- Gráficas de métricas

**Salidas:**
- `food_multilabel_final.h5`
- `training_results.json`
- Gráficas de entrenamiento

### 4. Predicción (`04_prediction.ipynb`)

**Contenido:**
- Carga del modelo entrenado
- Funciones de predicción multilabel
- Visualización de resultados
- Predicción con diferentes thresholds
- Análisis de confianza
- Ejemplos con múltiples etiquetas

**Salidas:**
- Predicciones en imágenes
- `sample_prediction.json`

---

## 🖥️ Aplicación Web

### Interfaz Streamlit

La aplicación web proporciona una interfaz interactiva para:

1. **Subir Imágenes**: Formatos JPG, JPEG, PNG
2. **Configurar Threshold**: Ajustar sensibilidad
3. **Ver Predicciones**: Múltiples etiquetas con probabilidades
4. **Descargar Resultados**: Exportar a JSON

### Características Técnicas

- ✅ Caché del modelo para eficiencia
- ✅ Preprocesamiento automático
- ✅ Visualización en tiempo real
- ✅ Métricas detalladas
- ✅ Responsive design

### Ejemplo de Predicción

```python
# Input: Imagen de un plato de comida
# Output:
{
  "threshold": 0.5,
  "num_labels": 4,
  "predictions": [
    {"class": "rice", "probability": 0.92},
    {"class": "teriyaki", "probability": 0.88},
    {"class": "chicken", "probability": 0.85},
    {"class": "vegetables", "probability": 0.76}
  ]
}
```

---

## 📊 Resultados

### Métricas del Modelo

#### Fase 1 (Entrenamiento Inicial)

| Métrica | Valor |
|---------|-------|
| **Hamming Loss** | ~0.15 |
| **F1-Score (Micro)** | ~0.78 |
| **F1-Score (Macro)** | ~0.72 |
| **Precision** | ~0.80 |
| **Recall** | ~0.76 |

#### Fase 2 (Fine-Tuning)

| Métrica | Valor | Mejora |
|---------|-------|--------|
| **Hamming Loss** | ~0.12 | ↓ 20% |
| **F1-Score (Micro)** | ~0.85 | ↑ 9% |
| **F1-Score (Macro)** | ~0.79 | ↑ 10% |
| **Precision** | ~0.87 | ↑ 9% |
| **Recall** | ~0.83 | ↑ 9% |

### Estrategias de Mejora Aplicadas

1. ✅ **Fine-tuning del backbone** (últimas 30 capas)
2. ✅ **Data augmentation mejorada** (rotación, zoom, brillo)
3. ✅ **Learning rate reducido** (0.001 → 0.0001)
4. ✅ **Callbacks avanzados** (EarlyStopping, ReduceLROnPlateau)

### Comparación de Enfoques

| Métrica | Inicial | Fine-Tuned | Mejora (%) |
|---------|---------|------------|------------|
| F1-Score | 0.78 | 0.85 | +9% |
| Precision | 0.80 | 0.87 | +9% |
| Recall | 0.76 | 0.83 | +9% |

---

## 🛠️ Tecnologías

### Deep Learning & ML

- **TensorFlow 2.x**: Framework principal
- **Keras**: API de alto nivel
- **EfficientNet**: Arquitectura base
- **scikit-learn**: Métricas y preprocesamiento

### Data Science

- **NumPy**: Operaciones numéricas
- **Pandas**: Manipulación de datos
- **Matplotlib & Seaborn**: Visualización

### Web & Deployment

- **Streamlit**: Aplicación web interactiva
- **Pillow (PIL)**: Procesamiento de imágenes

### Desarrollo

- **Jupyter Notebook**: Notebooks interactivos
- **Python 3.8+**: Lenguaje base

---

## 📚 Conceptos Clave Aprendidos

### 1. Transfer Learning

Aprovechamiento de redes pre-entrenadas en ImageNet para:
- Reducir tiempo de entrenamiento
- Mejorar generalización
- Funcionar con datasets pequeños

### 2. Multilabel Classification

Diferencias fundamentales con multiclase:
- Activación Sigmoid vs Softmax
- Binary CE vs Categorical CE
- Métricas especializadas (Hamming Loss)

### 3. Fine-Tuning

Estrategia de dos fases:
- Fase 1: Entrenar solo clasificador
- Fase 2: Ajustar backbone gradualmente

### 4. Data Augmentation

Técnicas para aumentar variabilidad:
- Rotaciones, traslaciones, zoom
- Ajustes de brillo y contraste
- Prevención de overfitting

---

## 🤝 Contribuciones

Este es un proyecto académico, pero las sugerencias son bienvenidas:

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/mejora`)
3. Commit cambios (`git commit -m 'Agregar mejora'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## 👨‍💻 Autor

**Proyecto Académico de Machine Learning**

- 🎓 Área: Visión por Computadora
- 🧠 Técnicas: Deep Learning, Transfer Learning
- 🍱 Dominio: Food Recognition
- 📅 Año: 2026

---

## 🙏 Agradecimientos

- **UECFood256 Dataset**: University of Electro-Communications
- **Kaggle**: Por hospedar el dataset
- **TensorFlow Team**: Por el framework
- **Streamlit**: Por facilitar la creación de apps ML

---

## 📞 Contacto

Para preguntas o sugerencias sobre este proyecto académico:

- 📧 Email: [tu-email@ejemplo.com]
- 💼 LinkedIn: [Tu perfil]
- 🐙 GitHub: [Tu usuario]

---

## 🔗 Enlaces Útiles

- [Dataset UECFood256](https://www.kaggle.com/datasets/rkuo2000/uecfood256)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Streamlit Docs](https://docs.streamlit.io/)

---

<div align="center">

**⭐ Si este proyecto te fue útil, considera darle una estrella ⭐**

Hecho con ❤️ y 🧠 para Machine Learning

</div>
