"""Ejecutar 11 tests rápidos consecutivos"""
import subprocess
import sys

python_exe = r"C:/Users/jordy/OneDrive/Desktop/iaaaa/iajordy2/.venv/Scripts/python.exe"
test_script = "quick_test_mlflow_model.py"

print("=" * 80)
print("EJECUTANDO 11 REENTRENAMIENTOS DE PRUEBA")
print("=" * 80)
print()

successful = 0
failed = 0

for i in range(1, 12):
    print(f"\n{'='*80}")
    print(f"TEST {i}/11")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [python_exe, test_script],
            capture_output=False,
            text=True,
            cwd=r"C:\Users\jordy\OneDrive\Desktop\iaaaa\iajordy2"
        )
        
        if result.returncode== 0:
            successful += 1
            print(f"\n✅ Test {i} completado exitosamente")
        else:
            failed += 1
            print(f"\n❌ Test {i} falló con código {result.returncode}")
            
    except Exception as e:
        failed += 1
        print(f"\n❌ Error en test {i}: {e}")
    
    print(f"Progreso: {successful}/{i} exitosos")

print("\n" + "=" * 80)
print("RESUMEN FINAL")
print("=" * 80)
print(f"✅ Exitosos: {successful}")
print(f"❌ Fallidos: {failed}")
print(f"📊 Tasa de éxito: {(successful/11)*100:.1f}%")
print()
print("VERIFICA EN MLFLOW UI: http://localhost:5001")
print("Experimento → Pestaña 'Models'")
print(f"Deberías ver 'yolo_reentrenado'con versiones 2 a {2+successful}")
print("=" * 80)
