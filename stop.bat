@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo               Stopping VeriLens AI System
echo ========================================================

echo Stopping services listening on port 8000 (FastAPI / Uvicorn)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo Terminating PID %%a on port 8000...
    taskkill /f /pid %%a >nul 2>&1
)

echo Stopping services listening on port 8501 (Streamlit)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501 ^| findstr LISTENING') do (
    echo Terminating PID %%a on port 8501...
    taskkill /f /pid %%a >nul 2>&1
)

:: Terminate cmd windows created with custom titles
taskkill /fi "WINDOWTITLE eq VeriLens AI - FastAPI Backend*" /f >nul 2>&1
taskkill /fi "WINDOWTITLE eq VeriLens AI - Streamlit UI*" /f >nul 2>&1

echo.
echo ========================================================
echo VeriLens AI services have been stopped.
echo ========================================================
echo.
pause
