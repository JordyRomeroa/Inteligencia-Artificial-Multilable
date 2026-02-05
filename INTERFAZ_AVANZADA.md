# 🎯 INTERFAZ AVANZADA - Guía de Uso

## Acceso

La interfaz avanzada está disponible en:
```
http://localhost:5000/advanced
```

## Características

### 1. **Panel Lateral - Selector de Modelos** 📊
- **Listado de versiones**: Visualiza todos los modelos disponibles
- **Selector de modelo activo**: Cambia entre versiones (actual, mejoradas, reentrenadas)
- **Información en tiempo real**: 
  - Versión actual
  - Estado (Listo ✓ o Entrenando)
  - GPU disponible (GPU ✓ o CPU)
  - Métricas de precisión por versión

**Uso:**
1. Selecciona un modelo del dropdown
2. El sistema cargará automáticamente la nueva versión
3. Todas las predicciones usarán el modelo seleccionado

---

### 2. **Pestaña: PREDICCIÓN** 📸

#### Workflow:
1. **Sube imagen** → Haz click en el área de upload o arrastra
2. **Automáticamente se predice** → El modelo detecta objetos
3. **Visualiza resultados** en dos paneles:

#### Panel Izquierdo: Imagen Anotada
- Muestra la imagen con bounding boxes
- Cada clase con color diferente:
  - 🔴 **Red** = Person
  - 🔵 **Teal** = Car  
  - 🟡 **Yellow** = Dog
- Confianza mostrada en la etiqueta

#### Panel Derecho: Lista de Detecciones
- Cada objeto detectado con:
  - Nombre de clase
  - Porcentaje de confianza
  - Botón "Corregir" si el resultado es incorrecto

---

### 3. **Pestaña: CORRECCIONES** ✏️

Esta es la **PARTE CLAVE para reentrenamiento**.

#### Paso a Paso:

##### Paso 1: Cargar Imagen
1. Haz click en "Cargar Imagen"
2. Selecciona la imagen que quieres corregir (la misma o diferente)

##### Paso 2: Seleccionar Área (BBox)
1. En el canvas aparecerá la imagen
2. **Haz click y arrastra** para dibujar un rectángulo alrededor del objeto
3. El rectángulo se dibuja en **azul punteado**
4. Las coordenadas aparecen en tiempo real abajo del canvas

```
Ejemplo de selección:
┌─────────────────────────┐
│                         │
│  ┌──────────────┐       │  ← Click + Arrastra aquí
│  │    Objeto    │       │
│  └──────────────┘       │
│                         │
└─────────────────────────┘
```

##### Paso 3: Seleccionar Etiqueta Correcta
Después de dibujar el bbox, aparecen **3 botones**:
- 👤 **Person**
- 🚗 **Car**
- 🐕 **Dog**

Haz click en la etiqueta CORRECTA del objeto.

##### Paso 4: Guardar Corrección
1. El botón **"💾 Guardar Corrección"** se activa
2. Haz click para guardar
3. Verás confirmación: ✓ Corrección guardada

---

### 4. **Estado de Correcciones** 📊

Abajo en la sección "CORRECCIONES" aparece:

```
Total guardadas: 5
Listas para reentrenar: No (mín. 10)
```

**Cuando llegues a 10 correcciones:**
- Se activa el botón **"🔄 Iniciar Reentrenamiento"**
- El modelo se entrenará con tus correcciones
- Se genera una nueva versión automáticamente

---

### 5. **Reentrenamiento** 🔄

#### Antes de Reentrenar:
✅ Necesitas **mínimo 10 correcciones guardadas**

#### Cómo Reentrenar:
1. Haz click en **"🔄 Iniciar Reentrenamiento"**
2. Se pedirá confirmación (⏳ puede tardar varios minutos)
3. Mientras se procesa:
   - Verás un spinner en el sidebar
   - Mensaje: "Compilando nuevas muestras..."
4. Cuando termine:
   - ✓ Se crea automáticamente una nueva versión
   - 📊 Aparecerá en el selector de modelos
   - 📈 Se mostrarán nuevas métricas

#### Resultado:
```
✓ Reentrenamiento completado. 
Nuevo modelo: v3 (Retrained)
Precisión: 87.5%
```

---

## 🔄 Workflow Completo (Ejemplo)

### Escenario: El modelo predice mal a los perros

**Día 1:**
1. Abre `/advanced`
2. Sube una imagen con un perro
3. El modelo predice "person" (❌ incorrecto)
4. Click en **"Corregir"** → Pestaña "CORRECCIONES"
5. Carga la imagen → Dibuja bbox alrededor del perro
6. Selecciona etiqueta **"🐕 Dog"** → Guarda corrección
7. Repite esto 9 veces más con diferentes imágenes

**Día 2:**
1. Tienes 10 correcciones guardadas ✓
2. Click en **"🔄 Iniciar Reentrenamiento"**
3. Esperas 5-10 minutos (según tu GPU)
4. ✓ Se crea modelo v2 (Retrained)
5. Automáticamente es el modelo activo
6. Pruebas de nuevo → ¡Ahora detecta perros mejor!

---

## 💡 Tips & Mejores Prácticas

### ✅ Lo Que DEBES Hacer:
- 📐 Selecciona bboxes **precisos** (no demasiado grandes)
- 🎯 Incluye **diferentes clases** en las correcciones
- 📸 Usa **imágenes variadas** (diferentes ángulos, iluminación, etc)
- ⏱️ Espera a **mínimo 10 correcciones** antes de reentrenar
- 📊 Revisa **métricas** después del reentrenamiento

### ❌ Lo Que EVITAR:
- 🚫 Bboxes muy pequeños o muy grandes
- 🚫 Mezclar clases (dibujar persona, etiquetar perro)
- 🚫 Pocas imágenes de la misma clase
- 🚫 Reentrenar con menos de 5 correcciones
- 🚫 Imágenes borrosas o mal iluminadas

---

## 🔧 Endpoints API (Para Desarrolladores)

Si quieres integrar con otras aplicaciones:

### Models
```
GET  /api/models/list              - Listar modelos disponibles
POST /api/models/load              - Cargar un modelo específico
```

### Corrections
```
POST /api/corrections/add           - Agregar corrección
GET  /api/corrections/stats         - Estadísticas
```

### Retraining
```
POST /api/model/retrain             - Iniciar reentrenamiento
```

### Inference
```
POST /predict                       - Predicción básica
GET  /health                        - Health check
GET  /model-info                    - Info del modelo
```

---

## 🐛 Troubleshooting

### Problema: "Modelo no cargado"
**Solución:**
```bash
# Verifica que exista el modelo
ls models/*.pt

# Si no existe, entrena primero:
# Ejecuta notebook 02_train_yolo.ipynb
```

### Problema: "GPU no disponible"
**Solución:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Si retorna False, instala PyTorch con CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Problema: "Reentrenamiento muy lento"
**Causas posibles:**
- Usando CPU en lugar de GPU
- Demasiadas épocas (default: 5)
- Computadora con recursos limitados

**Solución:** Verifica que GPU esté activada (debe decir "GPU ✓" en sidebar)

---

## 📱 Versiones y Historial

El sistema mantiene versiones:
- **v1** - Modelo original (best_improved.pt)
- **v2** - Primera mejora (después de correcciones)
- **v3** - Segunda mejora
- ...

Cada versión:
- ✅ Se puede cargar en cualquier momento
- 📊 Tiene sus propias métricas
- 🔄 Puede compararse con versiones anteriores

---

## ¿Necesitas ayuda?

Si algo no funciona:
1. Revisa la consola del navegador (F12 → Console)
2. Revisa los logs del servidor Flask
3. Verifica que todos los archivos estén en su lugar
4. Reinicia el servidor: `Ctrl+C` y vuelve a ejecutar

