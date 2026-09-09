# PyInstaller build recipe. Build on Windows after START_HERE.bat.
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ["ultralytics", "torch", "torchvision", "PySide6"]:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass

a = Analysis(
    ["app.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas + [("models/prelabel/PUT_MODEL_HERE.txt", "models/prelabel")],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AerialPersonStudio", debug=False, bootloader_ignore_signals=False, strip=False, upx=True, console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, name="AerialPersonStudio")
