@echo off
rem Same as START_WINDOWS.bat (installs everything needed on first run, then starts TITAN-9).
rem Kept so the documented command "run_titan.bat [--demo | --no-camera | ...]" keeps working.
call "%~dp0START_WINDOWS.bat" %*
exit /b %errorlevel%
