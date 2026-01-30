# 🚀 INICIO RÁPIDO - Aplicación Web de Clasificación Multilabel

## ¡Qué has obtenido!

Una **aplicación web interactiva** completa para clasificación multilabel con 5 componentes principales:

```
📦 Aplicación Web Completa
├── 🌐 API Flask (Python)
├── 🎨 Interfaz HTML/CSS/JavaScript
├── 🤖 Sistema de Predicción
├── 💾 Sistema de Correcciones
└── 🔄 Sistema de Reentrenamiento
```

## ⚡ Inicio en 3 Pasos

### OPCIÓN 1: Usando el Script de Inicio (Recomendado)

```bash
# En Windows, haz doble clic en:
run_app.bat
```

O desde PowerShell:
```powershell
cd c:\Users\mlata\Documents\iajordy2
.\run_app.bat
```

### OPCIÓN 2: Manualmente

```bash
# 1. Abre PowerShell en la carpeta del proyecto
cd c:\Users\mlata\Documents\iajordy2

# 2. Activa el entorno virtual
.venv\Scripts\Activate.ps1

# 3. Ejecuta el servidor
python app/api.py

# 4. Abre en navegador:
# http://localhost:5000
```

### OPCIÓN 3: Terminal CMD

```cmd
cd c:\Users\mlata\Documents\iajordy2
.venv\Scripts\activate.bat
python app/api.py
```

## 📋 Verificación Previa

Antes de iniciar, asegúrate de que todo esté configurado:

```bash
.venv\Scripts\python.exe check_setup.py
```

Debe mostrar: ✓ Todo listo!

## 🎯 Interfaz Web - 3 Pestañas Principales

### 1️⃣ **Imagen Individual**
```
├─ Subir 1 imagen
├─ Ver predicciones con confianza (%)
├─ Ajustar threshold (0.1 - 0.9)
├─ Corregir etiquetas si hay error
└─ Guardar corrección para reentrenar
```

### 2️⃣ **Evaluación Batch**
```
├─ Subir múltiples imágenes (5, 10, 20...)
├─ Predecir todas a la vez
├─ Ver resultados en grilla
└─ Seleccionar cualquiera para corregir
```

### 3️⃣ **Historial de Correcciones**
```
├─ Ver todas las correcciones guardadas
├─ Saber cuántas has hecho
└─ Decidir cuándo reentrenar
```

## 🔄 Flujo de Trabajo Completo

### Ejemplo: Mejorar Predicciones Paso a Paso

```
PASO 1: EVALUAR
   ↓
   Sube 10 imágenes en "Evaluación Batch"
   ↓
   El modelo predice automáticamente

PASO 2: CORREGIR
   ↓
   Encuentra errores (falsos positivos/negativos)
   ↓
   Haz clic en "Corregir"
   ↓
   Selecciona las etiquetas correctas
   ↓
   Haz clic en "Guardar Corrección"
   ↓
   (Repite con más imágenes)

PASO 3: REENTRENAR
   ↓
   Cuando tengas 5-10 correcciones guardadas
   ↓
   Haz clic en "Reentrenar Modelo"
   ↓
   Espera 1-3 minutos (depende del CPU/GPU)
   ↓
   El modelo se actualiza automáticamente

PASO 4: RE-EVALUAR
   ↓
   Vuelve a predecir las mismas imágenes
   ↓
   Verifica mejora en las predicciones
   ↓
   ¡Repite el proceso!
```

## 🎨 Colores en la Interfaz

- 🟢 **Verde**: Alta confianza (>70%)
- 🟡 **Amarillo**: Confianza media (40-70%)
- 🔴 **Rojo**: Baja confianza (<40%)

## 📊 Controles Principales

| Control | Función |
|---------|---------|
| **Threshold Slider** | Ajusta mínimo de confianza requerida |
| **Predecir** | Realiza predicción con imagen actual |
| **Guardar Corrección** | Guarda etiquetas correctas del usuario |
| **Reentrenar Modelo** | Fine-tune con correcciones guardadas |
| **Predecir Todo** | Batch prediction de múltiples imágenes |
| **Corregir** | Va a vista individual de una imagen |

## 💾 Dónde se Guardan los Datos

```
data/
├─ corrections/          ← Correcciones guardadas (JSON)
├─ uploads/              ← Imágenes subidas temporalmente
└─ voc2007/
   └─ classes.json       ← Las 20 clases del modelo
```

## 🔧 Configuración Personalizable

En `app/api.py` puedes ajustar:

```python
# Línea ~50: Tamaño máximo de archivo
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Cambiar a 32 MB si quieres

# Línea ~200: Epochs de reentrenamiento
epochs = data.get('epochs', 5)  # Cambiar a 10 para más epochs
```

## 🚫 Troubleshooting Rápido

### Error: "Model file not found"
```bash
# Asegúrate de haber ejecutado notebook 03_training_real_images.ipynb
```

### Error: "Port 5000 already in use"
```bash
# Usa otro puerto en api.py:
app.run(host='0.0.0.0', port=5001)  # Cambiar 5000 por 5001
```

### Servidor muy lento al reentrenar
```bash
# Es normal sin GPU. Reduce epochs:
epochs = 3  # En lugar de 5
```

### Las predicciones no mejoran
```bash
# 1. Verifica que hay correcciones guardadas:
#    data/corrections/ debe tener archivos JSON
# 2. Haz más correcciones (mínimo 5-10)
# 3. Reentrena con más epochs
```

## 📈 Métricas a Monitorear

El modelo se mejora cuando:
- ✅ Precisión aumenta (menos falsos positivos)
- ✅ Recall aumenta (menos falsos negativos)
- ✅ Las barras verdes de confianza se hacen más grandes
- ✅ La predicción se vuelve más precisa visualmente

## 🎓 Consejos para Mejores Resultados

1. **Correcciones Variadas**: No corrijas solo un tipo de objeto
2. **Calidad de Imágenes**: Usa imágenes claras y bien iluminadas
3. **Múltiples Reentrenamientos**: No esperes perfección con 1 reentrenamiento
4. **Paciencia**: Puede tomar 10-20 iteraciones para notar mejora significativa
5. **Threshold Ajustado**: A veces bajar el threshold es mejor que reentrenar

## 🔐 Notas de Seguridad

- ✅ Máximo 16 MB por archivo
- ✅ Solo acepta PNG, JPG, JPEG
- ✅ Los archivos se limpian automáticamente
- ✅ No se envían datos a servidores externos

## 📞 Ayuda Rápida

```bash
# Ver logs en tiempo real
# (ver salida de la terminal donde ejecutaste app/api.py)

# Forzar recarga de la página
# Ctrl + Shift + R en navegador

# Limpiar cache del navegador
# F12 → Application → Clear Site Data
```

## 🎉 Próximos Pasos

1. ✅ Ejecuta: `python app/api.py`
2. ✅ Abre: `http://localhost:5000`
3. ✅ Sube una imagen de prueba
4. ✅ Verifica las predicciones
5. ✅ Guarda una corrección si hay error
6. ✅ Reentrena el modelo
7. ✅ ¡Observa la mejora!

---

**¡Disfruta tu aplicación de clasificación multilabel! 🎯**

Para más información, ver: [README_APP.md](README_APP.md)
