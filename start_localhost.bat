@echo off
title FinAgent RAG — Local Host
echo ========================================================
echo   FinAgent RAG - Financial Intelligence Local Server
echo ========================================================
echo.

cd /d "%~dp0"

IF EXIST "venv\Scripts\python.exe" (
    set PYTHON_CMD=venv\Scripts\python.exe
) ELSE (
    set PYTHON_CMD=python
)

echo Starting FinAgent server on http://localhost:8000 ...
echo Opening default web browser...
start http://localhost:8000

"%PYTHON_CMD%" run_server.py
pause
