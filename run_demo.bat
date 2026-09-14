@echo off
REM SignalScope One-Click Windows Runner
setlocal

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found.
    echo Please create the Python 3.11 virtual environment first.
    exit /b 1
)

IF "%~1"=="" (
    echo [USAGE] run_demo.bat path\to\image.jpg
    exit /b 1
)

".venv\Scripts\python.exe" src\model\predict.py --image "%~1"
