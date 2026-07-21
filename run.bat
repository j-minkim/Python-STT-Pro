@echo off
setlocal
cd /d %~dp0

:: Set UTF-8 encoding to fix rich library rendering on Windows
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if not exist .venv (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

echo [INFO] Installing/Updating dependencies...
pip install -r requirements.txt --quiet

echo [INFO] Starting Python STT Pro...
if "%1"=="web" (
    python web_app.py
) else (
    python main.py %*
)

pause
