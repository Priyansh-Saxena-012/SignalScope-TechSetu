@echo off
REM SignalScope One-Click Windows Runner
setlocal

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found.
    echo Please create the Python 3.11 virtual environment first.
    exit /b 1
)

IF "%~1"=="" (
    echo ============================================================
    echo  SignalScope - Generative Media Forensics
    echo ============================================================
    echo  Usage:
    echo    CLI Inference:    run_demo.bat path\to\image.jpg
    echo    Streamlit Web UI: run_demo.bat --ui
    echo.
    echo  No image specified. Launching Streamlit Web UI...
    echo ============================================================
    ".venv\Scripts\python.exe" -m streamlit run src\ui\app.py
    exit /b 0
)

IF /I "%~1"=="--ui" (
    echo Starting SignalScope Streamlit Web UI...
    ".venv\Scripts\python.exe" -m streamlit run src\ui\app.py
    exit /b 0
)

".venv\Scripts\python.exe" src\model\predict.py --image "%~1"
