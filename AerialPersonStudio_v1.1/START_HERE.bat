@echo off
setlocal
cd /d "%~dp0"
title Aerial Person Studio - Setup
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" -RunAfterSetup
if errorlevel 1 (
  echo.
  echo Setup did not complete successfully.
  pause
)
