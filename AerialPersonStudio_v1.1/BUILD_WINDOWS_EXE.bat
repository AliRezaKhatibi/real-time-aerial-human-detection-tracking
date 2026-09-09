@echo off
setlocal
cd /d "%~dp0"
set "VENV=%LOCALAPPDATA%\AerialPersonStudio\venv-py312"
if not exist "%VENV%\Scripts\python.exe" (
  echo Run START_HERE.bat first.
  pause
  exit /b 1
)
"%VENV%\Scripts\python.exe" -m pip install --upgrade pyinstaller
"%VENV%\Scripts\pyinstaller.exe" --noconfirm --clean AerialPersonStudio.spec
if errorlevel 1 (
  echo Build failed. See console output above.
  pause
  exit /b 1
)
echo.
echo Build finished under dist\AerialPersonStudio\
pause
