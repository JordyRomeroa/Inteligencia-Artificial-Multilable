"""
Servidor Flask para YOLO Object Detection API
"""
import os
os.environ['FLASK_ENV'] = 'production'

from inference_api import app, load_model

if __name__ == '__main__':
    print("=== YOLO Object Detection API ===")
    print("Servidor iniciado en: http://127.0.0.1:5000")
    print("Presiona Ctrl+C para detener")
    if not load_model():
        print("ADVERTENCIA: No se pudo cargar el modelo. Revisa la ruta en models/.")
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
