@echo off
REM Script para iniciar Jupyter Notebook en el proyecto
REM Asegúrate de estar en el directorio del proyecto

echo Activando entorno virtual...
call venv\Scripts\activate.bat

echo.
echo Verificando instalación de paquetes...
python -m pip list | findstr jupyter tensorflow streamlit

echo.
echo Iniciando Jupyter Notebook...
python -m jupyter notebook --notebook-dir=notebooks

pause
