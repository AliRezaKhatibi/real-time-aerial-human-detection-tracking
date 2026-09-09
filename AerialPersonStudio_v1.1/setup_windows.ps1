param([switch]$RunAfterSetup)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "        Aerial Person Studio Setup" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Application: $Root"

function Find-Python312 {
    try {
        $p = (& py -3.12 -c "import sys; print(sys.executable)" 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() }
    } catch {}
    try {
        $lines = (& python -c "import sys; print(sys.version_info.major, sys.version_info.minor); print(sys.executable)" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $lines[0] -eq "3 12") { return $lines[1].Trim() }
    } catch {}
    return $null
}

$Python = Find-Python312
if (-not $Python) {
    Write-Host "Python 3.12 was not found." -ForegroundColor Yellow
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        $answer = Read-Host "Install Python 3.12 using winget? [Y/n]"
        if ($answer -eq "" -or $answer -match "^[Yy]") {
            winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
            $Python = Find-Python312
        }
    }
}
if (-not $Python) {
    Write-Host "Please install Python 3.12 (64-bit), then run START_HERE.bat again." -ForegroundColor Red
    exit 1
}
Write-Host "Python: $Python" -ForegroundColor Green

# Keep the runtime outside the project folder. This avoids Windows long-path
# failures in PyTorch and keeps the environment valid if the app folder moves.
$RuntimeRoot = Join-Path $env:LOCALAPPDATA "AerialPersonStudio"
$Venv = Join-Path $RuntimeRoot "venv-py312"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating runtime: $Venv"
    & $Python -m venv $Venv
}

Write-Host "Runtime: $Venv" -ForegroundColor Green
Write-Host "Upgrading pip tools..."
& $VenvPython -m pip install --upgrade pip setuptools wheel
Write-Host "Installing Aerial Person Studio dependencies..."
& $VenvPython -m pip install -r (Join-Path $Root "requirements.txt")

# Store runtime location next to the app for diagnostics.
$Venv | Set-Content -Encoding UTF8 (Join-Path $Root ".runtime_path.txt")

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Model location: models\prelabel\yolo26s_visdrone_best.pt"
Write-Host "You can also select any .pt file from inside the application."

if ($RunAfterSetup) {
    Write-Host "Launching Aerial Person Studio..."
    & $VenvPython (Join-Path $Root "app.py")
}
