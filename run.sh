#!/bin/bash

# Get the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv .venv
fi

echo "[INFO] Activating virtual environment..."
source .venv/bin/activate

echo "[INFO] Installing/Updating dependencies..."
pip install -r requirements.txt

echo "[INFO] Starting Python STT Pro..."
python3 main.py "$@"
