@echo off
title Personal Wealth Engine - Quantitative OS
cd /d "%~dp0"

echo ====================================================================
echo   Starting Personal Wealth Engine (Quantitative OS)
echo ====================================================================
echo.

:: 1. Check and activate virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment: .venv
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment: venv
    call "venv\Scripts\activate.bat"
)

:: 2. Verify Python is installed and accessible
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python 3.10, 3.11, or 3.12 and ensure "Add Python to PATH" is checked during installation.
    echo.
    pause
    exit /b 1
)

:: 3. Verify Streamlit is installed
python -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] Streamlit is not installed in the active environment.
    echo [*] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install dependencies. Please run 'pip install -r requirements.txt' manually.
        echo.
        pause
        exit /b 1
    )
)

:: 4. Launch Streamlit Web Application
echo [*] Launching Streamlit Web Application...
echo ====================================================================
echo   Dashboard URL: http://localhost:8501
echo ====================================================================
echo.
python -m streamlit run app.py --server.port=8501 --server.headless=false

echo.
echo ====================================================================
echo App exited. Check above for any error details.
echo ====================================================================
pause