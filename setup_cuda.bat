@echo off
setlocal
cd /d "%~dp0"
python -m venv .venv || exit /b 1
".venv\Scripts\python.exe" -m pip install --upgrade pip || exit /b 1
echo Install the PyTorch CUDA build recommended for this machine from https://pytorch.org/get-started/locally/
echo Then run: .venv\Scripts\python.exe -m pip install -r requirements.txt
echo Finally run: .venv\Scripts\python.exe verify_torch.py --expect cuda
endlocal
