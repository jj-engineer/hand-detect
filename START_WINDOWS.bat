@echo off
rem ============================================================================
rem  TITAN-9 - Windows launcher.  Just double-click this file.
rem
rem  First run: installs Python, the Visual C++ runtime and every package the
rem  project needs, checks the laptop camera, then starts the game.
rem  Next runs: starts in a few seconds.
rem
rem  Options (type them after the file name in a Command Prompt, optional):
rem     START_WINDOWS.bat --reinstall      rebuild the Python environment from scratch
rem     START_WINDOWS.bat --setup-only     install everything but do not start the game
rem     START_WINDOWS.bat --check          only test the camera / graphics
rem     START_WINDOWS.bat --demo           scripted showcase, no camera needed
rem     START_WINDOWS.bat --no-camera      keyboard + mouse only
rem     START_WINDOWS.bat --quality medium slower laptops (or: low)
rem     START_WINDOWS.bat --camera 1       use another camera (USB webcam)
rem ============================================================================
setlocal EnableExtensions
title TITAN-9 - Windows launcher
cd /d "%~dp0"

rem All the real work happens in setup_windows.ps1. Options are handed over in an
rem environment variable so PowerShell does not try to interpret "--demo" etc.
set "TITAN_ARGS=%*"

where powershell >nul 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Windows PowerShell was not found. It is part of Windows 10 and 11 -
    echo         please run Windows Update or repair Windows, then try again.
    echo.
    pause
    exit /b 1
)

if not exist "%~dp0setup_windows.ps1" (
    echo.
    echo [ERROR] setup_windows.ps1 is missing next to this file.
    echo         If you received a ZIP, right-click it, choose "Extract All", and start
    echo         START_WINDOWS.bat from the extracted folder - not from inside the ZIP.
    echo.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
set "RC=%errorlevel%"

if not "%RC%"=="0" (
    echo.
    echo ============================================================================
    echo  TITAN-9 did not finish correctly ^(code %RC%^).
    echo  Everything that happened was saved in:
    echo      %~dp0setup_windows.log
    echo  Send that file to the person who gave you this project and they can fix it.
    echo ============================================================================
    echo.
    pause
)
exit /b %RC%
