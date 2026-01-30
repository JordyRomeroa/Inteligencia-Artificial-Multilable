"""
Servidor Flask para la aplicación de clasificación multilabel
"""
import os
os.environ['FLASK_ENV'] = 'production'

from app.api import app

if __name__ == '__main__':
    print("Iniciando servidor Flask...")
    print("Accede a: http://127.0.0.1:5000")
    print("Presiona Ctrl+C para detener")
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
