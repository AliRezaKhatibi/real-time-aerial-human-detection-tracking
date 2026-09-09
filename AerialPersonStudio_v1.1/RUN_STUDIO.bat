@echo off
setlocal
cd /d "%~dp0"
set "VENV=%LOCALAPPDATA%\AerialPersonStudio\venv-py312"
if not exist "%VENV%\Scripts\python.exe" (
  echo Runtime not found. Starting first-time setup...
  call "%~dp0START_HERE.bat"
  exit /b
)
"%VENV%\Scripts\python.exe" "%~dp0app.py"
if errorlevel 1 pause
