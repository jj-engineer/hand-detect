#!/usr/bin/env bash
set -e

# Change to script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Find Python interpreter
if command -v python3 >/dev/null 2>&1; then
    PY="python3"
elif command -v python >/dev/null 2>&1; then
    PY="python"
elif [ -x "/usr/bin/python3" ]; then
    PY="/usr/bin/python3"
else
    echo "[ERROR] Python 3 not found in PATH." >&2
    exit 1
fi

# Check if .venv exists, otherwise create it
if [ ! -d ".venv" ]; then
    echo "[SETUP] Creating Python virtual environment..."
    "$PY" -m venv .venv
    if [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
    else
        source .venv/bin/activate
    fi
    pip install --upgrade pip
    pip install --no-compile -r requirements.txt
else
    if [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
    else
        source .venv/bin/activate
    fi
fi

echo "[START] Launching CyberPuppet 3D Robot Hand Controller..."
python main.py "$@"
