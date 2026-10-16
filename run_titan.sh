#!/usr/bin/env bash
# TITAN-9 launcher:  ./run_titan.sh            (webcam hand control)
#                    ./run_titan.sh --demo     (scripted showcase, no camera needed)
#                    ./run_titan.sh --no-camera (keyboard + mouse only)
set -e
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

if [ ! -d ".venv" ]; then
    echo "[SETUP] creating virtual environment..."
    "$PY" -m venv .venv
    if [ -f ".venv/Scripts/pip" ] || [ -f ".venv/Scripts/pip.exe" ]; then
        VENV_PIP="./.venv/Scripts/pip"
    else
        VENV_PIP="./.venv/bin/pip"
    fi
    "$VENV_PIP" install --upgrade pip
    "$VENV_PIP" install -r requirements.txt
fi

if [ -f ".venv/Scripts/python" ] || [ -f ".venv/Scripts/python.exe" ]; then
    VENV_PY="./.venv/Scripts/python"
else
    VENV_PY="./.venv/bin/python3"
fi

exec "$VENV_PY" -m titan "$@"
