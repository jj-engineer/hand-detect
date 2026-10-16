@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv" (
    echo [SETUP] Creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo Please ensure Python 3.9+ is installed and 'python' is added to your PATH.
        exit /b 1
    )
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        exit /b 1
    )
) else (
    call .venv\Scripts\activate.bat
)

echo [START] Launching CyberPuppet 3D Robot Hand Controller...
python main.py %*
