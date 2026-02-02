"""
Script para agregar más imágenes de perros al dataset
usando otro dataset de Pascal VOC o buscando más imágenes con dogs
"""

import kagglehub
from pathlib import Path
import xml.etree.ElementTree as ET
import shutil

def augment_dog_data():
    """
    Estrategia:
    1. Descargar Pascal VOC 2012 (más datos)
    2. Filtrar solo imágenes con perros
    3. Agregar al dataset de entrenamiento existente
    """
    
    # Descargar Pascal VOC 2012 (dataset más grande)
    print("Descargando Pascal VOC 2012...")
    dataset_path = kagglehub.dataset_download("huanghanchina/pascal-voc-2012")
    
    # TODO: Implementar extracción de imágenes con perros
    print(f"Dataset descargado en: {dataset_path}")
    print("Implementar filtrado y agregado al dataset existente")

if __name__ == "__main__":
    augment_dog_data()
