@echo off
REM Limpia el cache y reinicia la aplicación

echo ========================================
echo  Limpiando y reiniciando la aplicación
echo ========================================
echo.

REM Limpiar carpeta uploads
echo [1/3] Limpiando imágenes temporales...
if exist "data\uploads" (
    del /Q "data\uploads\*.*" 2>nul
    echo ✓ Uploads limpios
)

REM Limpiar __pycache__
echo [2/3] Limpiando caché de Python...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul

REM Iniciar servidor
echo [3/3] Iniciando servidor Flask...
echo.
echo ========================================
echo  Servidor corriendo en:
echo  http://localhost:5000
echo ========================================
echo.
echo Abre el navegador y presiona Ctrl+Shift+R
echo para limpiar el cache del navegador
echo.
echo Presiona Ctrl+C para detener
echo.

python app\api.py

pause
