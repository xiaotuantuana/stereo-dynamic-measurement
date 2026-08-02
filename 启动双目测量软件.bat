@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 goto python_missing

python -m stereo_research.gui
if errorlevel 1 goto launch_failed
exit /b 0

:python_missing
echo Python was not found in PATH.
echo Install Python and run: python -m pip install -r requirements_research.txt
pause
exit /b 1

:launch_failed
echo.
echo Stereo Measurement Studio failed to start.
echo Run this command for details: python -m stereo_research.gui
pause
exit /b 1
