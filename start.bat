@echo off
setlocal

echo ========================================================
echo               Starting VeriLens AI System
echo ========================================================

cd /d "%~dp0"
set PYTHONPATH=src

echo [1/3] Checking and installing dependencies...
python -m pip install -r requirements.txt --quiet

echo [2/3] Starting FastAPI Backend on http://localhost:8000 ...
start "VeriLens AI - FastAPI Backend" cmd /k "python -m uvicorn verilens.api.app:app --reload --port 8000 --app-dir src"

:: Wait 3 seconds for FastAPI to initialize
timeout /t 3 /nobreak >nul

echo [3/3] Starting Streamlit Dashboard on http://localhost:8501 ...
start "VeriLens AI - Streamlit UI" cmd /k "streamlit run ui/app.py"

echo.
echo ========================================================
echo VeriLens AI services are up and running!
echo  - API Docs:    http://localhost:8000/docs
echo  - Frontend UI: http://localhost:8501
echo To shut down all services, run stop.bat
echo ========================================================
echo.
pause
