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
if [ "$1" = "web-instance" ]; then
    if [ -z "$2" ] || [ -z "$3" ]; then
        echo "Usage: ./run.sh web-instance <name> <port>"
        exit 2
    fi
    export STT_INSTANCE="$2"
    export STT_PORT="$3"
    shift 3
    python3 web_app.py "$@"
elif [ "$1" = "web" ]; then
    shift
    python3 web_app.py "$@"
else
    python3 main.py "$@"
fi
