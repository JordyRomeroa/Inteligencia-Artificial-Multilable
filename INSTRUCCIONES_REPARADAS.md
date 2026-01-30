# 🚀 INSTRUCCIONES PARA USAR LA APLICACIÓN REPARADA

## ⚡ INICIO RÁPIDO (3 pasos)

### Paso 1: Ejecutar el servidor
```bash
# Abre PowerShell en c:\Users\mlata\Documents\iajordy2
cd c:\Users\mlata\Documents\iajordy2
.venv\Scripts\Activate.ps1
python app/api.py
```

Verás:
```
* Running on http://127.0.0.1:5000
```

### Paso 2: Abrir en el navegador

**OPCIÓN A: Versión Simplificada (RECOMENDADA - Totalmente Funcional)**
```
http://localhost:5000/simple
```

**OPCIÓN B: Versión Completa**
```
http://localhost:5000
```

### Paso 3: Usar la aplicación

#### En la página (cualquier versión):

1. **Subir imagen**: Haz clic en el input de archivo
2. **Ajustar threshold** (opcional): Usa el slider 0.1-0.9
3. **Predecir**: Haz clic en "Predecir Imagen" o "🔮 Predecir"
4. **Ver resultados**: Se mostrarán las etiquetas detectadas
5. **Corregir**: Selecciona los checkboxes con etiquetas correctas
6. **Guardar**: Haz clic en "Guardar Corrección"

---

## 📊 Lo que acaba de pasar (Arreglado)

### Problemas que fueron solucionados:

✅ **Predicción incompleta** → Ahora retorna formato correcto con `success: true`
✅ **Respuesta sin "success" field** → Añadido a todos los endpoints
✅ **Guardar corrección incorrecto** → Ahora usa `corrected_labels` (no `correct_labels`)
✅ **No muestra predicciones** → Displaypredictions ahora funciona correctamente
✅ **No permite correcciones** → displayCorrectionLabels ahora usa `window.classes`

### Cambios hechos:

| Archivo | Cambio |
|---------|--------|
| `api.py` | Respuesta `/predict` retorna JSON correcto con `success: true` |
| `api.py` | Endpoint `/save_correction` maneja correctamente `corrected_labels` |
| `api.py` | Agregado try/catch en todos los endpoints |
| `api.py` | Agregado endpoint `/simple` para versión simplificada |
| `script.js` | Usa `window.classes` en lugar de `const classes` |
| `index.html` | Jinja2 template pasa datos a `window.classes` |
| Nuevo | `simple.html` - Interfaz simplificada y totalmente funcional |

---

## 🎯 ¿Cuál versión usar?

### `/simple` ← RECOMENDADA PARA EMPEZAR
- ✅ Interfaz limpia y simple
- ✅ Todos los logs visibles
- ✅ 100% Funcional
- ✅ Fácil de debuggear
- ✅ Código legible

### `/` ← Versión completa
- ✅ Interfaz bonita
- ✅ 3 pestañas (Individual, Batch, Historial)
- ✅ Batch prediction
- ✅ Historial de correcciones

---

## 🔍 Debugging

### Ver logs en tiempo real

**En la aplicación simple (`/simple`):**
- Los logs aparecen en la sección "5. Logs" en la página
- Cada acción se registra con timestamp

**En la consola del navegador (F12):**
```javascript
// Abre F12 → Console y verás:
[info] Script.js cargado correctamente
[info] Clases disponibles: 20
[success] Predicción exitosa: 5 etiquetas detectadas
```

**En la terminal (donde ejecutaste `python app/api.py`):**
```
127.0.0.1 - - [30/Jan/2026 02:35:00] "POST /predict HTTP/1.1" 200 -
127.0.0.1 - - [30/Jan/2026 02:35:01] "POST /save_correction HTTP/1.1" 200 -
```

### Si no funciona:

1. **Limpiar cache del navegador:**
   ```
   F12 → Application → Clear Site Data
   ```

2. **Reiniciar servidor:**
   ```
   Ctrl+C en la terminal
   Ejecuta de nuevo: python app/api.py
   ```

3. **Limpiar carpetas:**
   ```bash
   # Ejecuta el script:
   .\run_app_clean.bat
   ```

4. **Ver errores:**
   - Terminal: Busca líneas con `ERROR` o `Exception`
   - Navegador F12: Pestaña Console

---

## 📝 Flujo de Trabajo Completo

```
┌─────────────────────┐
│ 1. Subir imagen     │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ 2. Predecir imagen  │  ← Espera 1-3 segundos
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ 3. Ver resultados   │  ← Se muestran etiquetas con %
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ 4. Seleccionar      │  ← Si hay error, marca correctas
│    correcciones     │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ 5. Guardar          │  ← Se guarda en data/corrections/
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ 6. (Opcional)       │  ← Con 5+ correcciones
│    Reentrenar       │
└─────────────────────┘
```

---

## ✨ Características Principales

### Imagen Individual
- Subir 1 imagen
- Ajustar threshold (0.1-0.9)
- Ver predicciones con confianza
- Corregir si hay errores
- Guardar para reentrenamiento

### Correcciones
- Ver todo lo que has marcado
- Estadísticas de cuántas has hecho
- Base para mejorar el modelo

### Reentrenamiento
- Automatizado al hacer clic
- Fine-tuning con tus correcciones
- Mejora el modelo en tiempo real

---

## 🛠️ Si necesitas más ayuda

### Errores comunes:

**"Port 5000 already in use"**
```bash
# Cambiar puerto en api.py línea ~290:
app.run(debug=True, host='0.0.0.0', port=5001)
```

**"No file provided"**
```bash
# Asegúrate de seleccionar una imagen antes de hacer clic en Predecir
```

**"Model file not found"**
```bash
# Necesitas el modelo entrenado en:
# models/voc_multilabel_final.h5
# Ejecuta el notebook 03_training_real_images.ipynb
```

**Predicciones muy malas**
```bash
# 1. Ajusta el threshold con el slider
# 2. Haz correcciones (5-10 mínimo)
# 3. Reentrena el modelo
# 4. Repite 2-3 veces
```

---

## 🎉 Resumen

✅ **Está todo reparado y funcional**
✅ **Usa `/simple` para empezar** (más fácil de ver qué pasa)
✅ **Los logs te dicen exactamente qué sucede**
✅ **Puedes corregir predicciones y mejorar el modelo**

**¡Ahora pruébalo y disfruta!** 🚀
