@echo off
setlocal
cd /d "%~dp0"
set "VENV=%LOCALAPPDATA%\AerialPersonStudio\venv-py312"
if not exist "%VENV%\Scripts\python.exe" (
  echo Runtime not found. Run START_HERE.bat first.
  pause
  exit /b 1
)
"%VENV%\Scripts\python.exe" "%~dp0self_check.py"
pause
