# 🚀 Aplicación Web de Clasificación Multilabel Interactiva

## Descripción

Esta aplicación web permite:
- ✅ Subir imágenes y obtener predicciones multilabel
- ✅ Corregir predicciones erróneas seleccionando etiquetas correctas
- ✅ Reentrenar el modelo incrementalmente con las correcciones
- ✅ Evaluar múltiples imágenes en batch
- ✅ Ver historial de correcciones

## Estructura de Archivos

```
iajordy2/
├── app/
│   ├── api.py                  # API Flask con endpoints
│   ├── utils.py                # Funciones de utilidad y reentrenamiento
│   ├── templates/
│   │   └── index.html          # Interfaz web
│   └── static/
│       ├── style.css           # Estilos CSS
│       └── script.js           # JavaScript interactivo
├── data/
│   ├── corrections/            # Correcciones guardadas (JSON)
│   └── uploads/                # Imágenes subidas
├── models/
│   └── voc_multilabel_final.h5 # Modelo entrenado
├── notebooks/                   # Jupyter notebooks
└── classes.json                 # Nombres de las 20 clases
```

## Requisitos Previos

1. Modelo entrenado: `models/voc_multilabel_final.h5`
2. Archivo de clases: `classes.json`
3. Python 3.10 con dependencias instaladas

## Instalación

### 1. Activar el entorno virtual

```bash
.venv\Scripts\activate
```

### 2. Verificar/Instalar dependencias

```bash
pip install flask werkzeug pillow numpy tensorflow scikit-learn
```

## Ejecución

### Paso 1: Ir al directorio de la aplicación

```bash
cd c:\Users\mlata\Documents\iajordy2
```

### Paso 2: Ejecutar la API Flask

```bash
python app/api.py
```

Verás un mensaje como:
```
 * Running on http://127.0.0.1:5000
```

### Paso 3: Abrir en el navegador

Abre tu navegador en: **http://localhost:5000**

## Uso de la Aplicación

### Tab 1: Imagen Individual

1. **Subir Imagen**: Haz clic en el input de archivo y selecciona una imagen
2. **Ajustar Threshold**: Usa el slider para cambiar el umbral de confianza (0.1 - 0.9)
3. **Predecir**: Haz clic en "Predecir" para obtener las etiquetas
4. **Ver Resultados**: Las predicciones se muestran con:
   - 🟢 Verde: alta confianza (>70%)
   - 🟡 Amarillo: confianza media (40-70%)
   - 🔴 Rojo: baja confianza (<40%)
5. **Corregir**: Si el modelo se equivocó:
   - Selecciona las etiquetas correctas en los checkboxes
   - Haz clic en "Guardar Corrección"
6. **Reentrenar**: 
   - Cuando tengas varias correcciones guardadas
   - Haz clic en "Reentrenar Modelo"
   - Espera 1-3 minutos
   - El modelo se actualizará automáticamente

### Tab 2: Evaluación Batch

1. **Seleccionar Múltiples Imágenes**: 
   - Haz clic en el input (acepta múltiples archivos)
   - Selecciona todas las imágenes que quieras evaluar
2. **Predecir Todo**: Haz clic en "Predecir Todo"
3. **Ver Resultados**: Se mostrarán todas las predicciones
4. **Corregir Individual**: Haz clic en "Corregir" en cualquier imagen para ir a la vista individual

### Tab 3: Correcciones

1. **Ver Historial**: Muestra todas las correcciones guardadas
2. **Actualizar**: Haz clic en "Actualizar" para refrescar la lista
3. **Estadísticas**: Muestra cuántas correcciones has hecho

## Endpoints de la API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Página principal |
| POST | `/predict` | Predecir imagen individual |
| POST | `/save_correction` | Guardar corrección de usuario |
| POST | `/retrain` | Reentrenar modelo con correcciones |
| POST | `/batch_predict` | Predecir múltiples imágenes |
| GET | `/get_corrections` | Obtener historial de correcciones |
| GET | `/health` | Estado de la API |

## Formato de Correcciones

Las correcciones se guardan en `data/corrections/` como archivos JSON:

```json
{
    "filename": "imagen.jpg",
    "corrected_labels": ["person", "dog", "car"],
    "timestamp": "2024-01-15T10:30:00"
}
```

## Flujo de Trabajo Recomendado

1. **Primera Evaluación**: Sube 10-20 imágenes en batch
2. **Corregir Errores**: Marca las etiquetas correctas para las imágenes mal clasificadas
3. **Primer Reentrenamiento**: Con ~10 correcciones, ejecuta el primer reentrenamiento
4. **Re-evaluar**: Vuelve a predecir las mismas imágenes para ver mejora
5. **Iteración Continua**: Repite el proceso para seguir mejorando

## Parámetros de Reentrenamiento

En el archivo `app/api.py`, puedes ajustar:

```python
# Línea ~200 en /retrain endpoint
epochs = data.get('epochs', 5)        # Epochs de fine-tuning (default 5)
learning_rate = 1e-5                  # Learning rate muy bajo para no destruir pesos
```

## Troubleshooting

### Error: "Model file not found"
- Asegúrate de que existe `models/voc_multilabel_final.h5`
- Verifica que ejecutaste el notebook 03 de entrenamiento

### Error: "Classes file not found"
- Debe existir `classes.json` en la raíz del proyecto
- Contiene las 20 clases de PASCAL VOC

### Las predicciones son malas
- Ajusta el threshold (slider)
- Guarda más correcciones
- Reentrena el modelo

### El reentrenamiento es muy lento
- Es normal, toma 1-3 minutos con GPU
- Sin GPU puede tomar 10-15 minutos
- Reduce `epochs` en el endpoint si es necesario

### No se guardan las correcciones
- Verifica que existe el directorio `data/corrections/`
- Asegúrate de tener permisos de escritura

## Mejoras Futuras

- [ ] Soporte para custom thresholds por clase
- [ ] Visualización de métricas de reentrenamiento
- [ ] Export de correcciones a CSV
- [ ] Autenticación de usuarios
- [ ] Base de datos para correcciones
- [ ] Integración con datasets externos

## Notas Técnicas

- **Modelo**: EfficientNetB0 fine-tuned en PASCAL VOC 2007
- **Clases**: 20 categorías (person, car, dog, cat, etc.)
- **Loss**: Focal Loss con gamma=2.0
- **Input**: Imágenes 224x224 normalizadas [0, 1]
- **Output**: Vector de 20 probabilidades (sigmoid)
- **Threshold**: Configurable per-predicción (default 0.5)

## Soporte

Si tienes problemas:
1. Revisa la consola de Flask para errores
2. Revisa la consola del navegador (F12) para errores de JavaScript
3. Verifica que todos los archivos existen
4. Asegúrate de que el modelo está correctamente entrenado

---

Desarrollado para proyecto grupal - Clasificación Multilabel con Reentrenamiento Interactivo 🎯
