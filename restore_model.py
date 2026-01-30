"""
Script para restaurar el modelo corrupto desde backup
"""
import shutil
from pathlib import Path

# Restaurar modelo desde backup
src = Path('models/model_phase1_best.h5')
dst = Path('models/voc_multilabel_final.h5')

if src.exists():
    shutil.copy2(src, dst)
    print(f'✓ Modelo restaurado: {dst} ({dst.stat().st_size} bytes)')
    print(f'✓ Archivo original: {src} ({src.stat().st_size} bytes)')
else:
    print('✗ Error: No se encontró modelo de backup')
