@echo off
setlocal
cd /d "%~dp0\.."

set "CUDA_WHL=%~1"
if "%CUDA_WHL%"=="" set "CUDA_WHL=cu126"

where uv >nul 2>nul
if errorlevel 1 (
  echo uv was not found. Install it from https://docs.astral.sh/uv/getting-started/installation/
  pause
  exit /b 1
)

echo Synchronizing the standard project environment...
uv sync
if errorlevel 1 (
  echo Environment setup failed.
  pause
  exit /b 1
)

echo Installing the PyTorch CUDA build %CUDA_WHL%...
uv pip install --python .venv\Scripts\python.exe --upgrade torch --index-url https://download.pytorch.org/whl/%CUDA_WHL%
if errorlevel 1 (
  echo CUDA PyTorch installation failed. Choose a CUDA wheel supported by the NVIDIA driver.
  pause
  exit /b 1
)

uv run --no-sync python verify_torch.py --expect cuda
if errorlevel 1 (
  echo CUDA verification failed. The CPU environment can be restored with: uv sync --reinstall
  pause
  exit /b 1
)

echo CUDA environment is ready.
