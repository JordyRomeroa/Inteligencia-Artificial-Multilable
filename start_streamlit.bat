@echo off
REM Script para iniciar la aplicación Streamlit

echo Activando entorno virtual...
call venv\Scripts\activate.bat

echo.
echo Iniciando aplicación Streamlit...
echo La aplicación se abrirá en: http://localhost:8501
echo.

python -m streamlit run app/app.py

pause
