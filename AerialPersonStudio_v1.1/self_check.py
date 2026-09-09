from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

print("Aerial Person Studio - Self Check")
print("=" * 44)
print("Python:", sys.version.replace("\n", " "))
print("App root:", ROOT)

checks = {}
for name in ["PySide6", "numpy", "PIL", "cv2", "yaml", "ultralytics", "torch"]:
    try:
        mod = __import__(name)
        checks[name] = getattr(mod, "__version__", "OK")
    except Exception as exc:
        checks[name] = f"ERROR: {exc}"
for k,v in checks.items():
    print(f"{k:12s}: {v}")

try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception:
    pass

model = ROOT / "models" / "prelabel" / "yolo26s_visdrone_best.pt"
print("Default model:", "FOUND" if model.exists() else "not placed yet")
print("\nIf PySide6, Ultralytics and Torch are OK, the application runtime is ready.")
