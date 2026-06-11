@echo off
cd /d "%~dp0"
pythonw app.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR al iniciar la app. Detalle:
    python app.py
    pause
)
