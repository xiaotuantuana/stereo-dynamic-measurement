@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m stereo_research.gui
if errorlevel 1 (
    echo.
    echo 软件启动失败，请先运行：python -m pip install -r requirements_research.txt
    pause
)
