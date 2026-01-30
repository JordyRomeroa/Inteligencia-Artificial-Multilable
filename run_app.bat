@echo off
REM Script de inicio para la aplicación web de clasificación multilabel

echo ========================================
echo  Clasificador Multilabel Interactivo
echo ========================================
echo.

REM Verificar que estamos en el directorio correcto
if not exist "app\api.py" (
    echo ERROR: No se encuentra app\api.py
    echo Por favor ejecuta este script desde la raiz del proyecto
    pause
    exit /b 1
)

REM Verificar archivos necesarios
echo [1/4] Verificando archivos necesarios...
if not exist "models\voc_multilabel_final.h5" (
    echo WARNING: No se encuentra models\voc_multilabel_final.h5
    echo Asegurate de haber entrenado el modelo primero (notebook 03)
    pause
)

if not exist "classes.json" (
    echo ERROR: No se encuentra classes.json
    pause
    exit /b 1
)

REM Activar entorno virtual
echo [2/4] Activando entorno virtual...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo WARNING: No se encuentra .venv\Scripts\activate.bat
    echo Intentando ejecutar sin entorno virtual...
)

REM Verificar dependencias
echo [3/4] Verificando dependencias...
python -c "import flask, tensorflow, numpy, PIL" 2>nul
if errorlevel 1 (
    echo.
    echo WARNING: Faltan dependencias. Instalando...
    pip install flask werkzeug pillow numpy tensorflow scikit-learn
)

REM Iniciar servidor
echo [4/4] Iniciando servidor Flask...
echo.
echo ========================================
echo  Servidor corriendo en:
echo  http://localhost:5000
echo ========================================
echo.
echo Presiona Ctrl+C para detener el servidor
echo.

python app\api.py

pause
