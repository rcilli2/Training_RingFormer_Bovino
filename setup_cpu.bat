@echo off
setlocal
cd /d "%~dp0"
python -m venv .venv || exit /b 1
".venv\Scripts\python.exe" -m pip install --upgrade pip || exit /b 1
".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1
".venv\Scripts\python.exe" verify_torch.py --expect cpu || exit /b 1
endlocal
