@echo off
setlocal
cd /d "%~dp0\.."

where uv >nul 2>nul
if errorlevel 1 (
  echo uv was not found. Install it from https://docs.astral.sh/uv/getting-started/installation/
  pause
  exit /b 1
)

echo Synchronizing the Python environment...
uv sync
if errorlevel 1 (
  echo Environment setup failed.
  pause
  exit /b 1
)

echo Starting JupyterLab...
uv run --no-sync jupyter lab notebooks
