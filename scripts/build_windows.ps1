param(
  [switch]$UseCuda
)

$ErrorActionPreference = "Stop"

# Build in an isolated environment so a workstation's existing CUDA-enabled
# PyTorch installation is not silently copied into the distributable. The CPU
# build is portable and dramatically smaller; maintainers can pass -UseCuda
# when producing a GPU-specific institutional build.
$buildEnvironment = Join-Path $PSScriptRoot "..\.build-venv"
if (-not (Test-Path $buildEnvironment)) {
  python -m venv $buildEnvironment
}

$buildPython = Join-Path $buildEnvironment "Scripts\python.exe"
& $buildPython -m pip install --upgrade pip

if (-not $UseCuda) {
  & $buildPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

& $buildPython -m pip install -e ".[build]"
& $buildPython -m PyInstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name FinFrame `
  --collect-all ultralytics `
  --hidden-import cv2 `
  finframe_launcher.py

Write-Host "FinFrame desktop build created in dist/FinFrame"
