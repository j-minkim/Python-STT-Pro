@echo off
setlocal
cd /d %~dp0

if not exist .venv (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

echo [INFO] Installing/Updating dependencies...
pip install -r requirements.txt

echo [INFO] Starting Python STT Pro...
python main.py %*

pause
