# 🎯 GUÍA RÁPIDA: CÓMO PROBAR EL REENTRENAMIENTO

## Estado Actual ✓

✅ **REENTRENAMIENTO FUNCIONA** 
- Los pesos se guardan correctamente con formato `.weights.h5`
- Los pesos se recargan en el modelo en memoria después del reentrenamiento
- Las siguientes predicciones AHORA usarán el modelo actualizado

## Pasos para Probar

### 1. Asegúrate que el Servidor está Corriendo

```bash
cd c:\Users\mlata\Documents\iajordy2
.venv\Scripts\Activate.ps1
python run_server.py
```

Deberías ver:
```
Iniciando servidor Flask...
Accede a: http://127.0.0.1:5000
* Running on http://127.0.0.1:5000
```

### 2. Abre la Interfaz Web

Opción A (RECOMENDADA - Versión Simple):
```
http://127.0.0.1:5000/simple
```

Opción B (Versión Completa):
```
http://127.0.0.1:5000
```

### 3. Ciclo Completo de Prueba

#### Paso 1: Primera Predicción
1. Haz clic en "Elegir Archivo" / "Choose File"
2. Selecciona una imagen (`.jpg` o `.png`)
3. Haz clic en "Predecir" / "Predict"
4. **Anota las predicciones principales** que ve (ejemplo: "person 50.2%")

#### Paso 2: Hacer una Corrección
1. En la sección "2. Predicciones del Modelo", verás las etiquetas detectadas
2. Marca/desmarca los checkboxes con las etiquetas CORRECTAS
   - Marca: etiquetas que SÍ están en la imagen
   - Desmarca: etiquetas que NO están en la imagen
3. Haz clic en "Guardar Corrección" / "Save Correction"
4. Deberías ver ✓ en la interfaz

#### Paso 3: Reentrenar el Modelo
1. Necesitas **mínimo 1 corrección** guardada
2. Haz clic en el botón "Reentrenar Modelo" / "Retrain"
3. Espera 5-10 segundos (debe completar 5 épocas de entrenamiento)
4. Verás el mensaje "✓ Modelo reentrenado"

#### Paso 4: Predicción Después del Reentrenamiento
1. **SIN reiniciar el servidor** (esto es importante!)
2. Carga LA MISMA IMAGEN nuevamente
3. Haz clic en "Predecir"
4. **Las predicciones DEBEN CAMBIAR** comparadas con el paso 1

---

## ¿Cómo Saber que Funcionó?

### Antes del Reentrenamiento (Paso 1)
```
person      50.2% ✓
chair       20.1% 
table       15.3%
```

### Después del Reentrenamiento (Paso 4)
```
chair       85.5% ✓  ← CAMBIÓ
person      30.2%    ← CAMBIÓ
table        8.1%    ← CAMBIÓ
```

Si los porcentajes/etiquetas cambian → **¡EL REENTRENAMIENTO FUNCIONA!** 🎉

---

## Archivos Importantes

| Archivo | Propósito |
|---------|-----------|
| `run_server.py` | Inicia el servidor Flask |
| `app/api.py` | Lógica del API (predicción, corrección, reentrenamiento) |
| `data/corrections/*.json` | Almacena las correcciones que haces |
| `models/voc_multilabel_final.h5` | Modelo base (no cambia) |
| `models/voc_multilabel_final.weights.h5` | Pesos guardados (se actualiza con reentrenamiento) |

---

## Debugging

Si algo no funciona:

### El servidor no responde (error de conexión)
```bash
# En la terminal del servidor, presiona Ctrl+C
# Luego ejecuta:
python run_server.py
```

### No aparecen predicciones
1. Abre F12 (Developer Tools)
2. Pestaña "Console" (Consola)
3. Busca líneas rojas (errores)
4. Cópialo y pregunta

### Reentrenamiento no funciona
1. Asegúrate de haber guardado una corrección primero
2. Espera 2-3 segundos antes de reentrenar
3. Mira la terminal del servidor para ver "Reentrenando con X imágenes..."

### Los pesos no se cargan después de reentrenar
- Esto ya está ARREGLADO en esta versión
- Si ves `Error al guardar/recargar pesos` en la terminal, quiere decir que TensorFlow tiene un problema
- Reinicia el servidor completamente

---

## Comandos Útiles

```bash
# Limpiar todas las correcciones (para empezar de nuevo)
Remove-Item data/corrections/*.json -Force

# Ver qué correcciones tienes guardadas
Get-Content data/corrections/*.json

# Ver logs del servidor en tiempo real
# (Mira la ventana donde ejecutaste python run_server.py)
```

---

## Resumen Rápido ⚡

1. ✅ Servidor corriendo
2. ✅ Abre http://127.0.0.1:5000/simple
3. ✅ Carga imagen → Predice (anota resultados)
4. ✅ Marca correcciones → Guarda
5. ✅ Reentrenamiento → Espera a que termine
6. ✅ Carga misma imagen → Predice (debe cambiar!)

**¡Eso es todo!** 🚀
