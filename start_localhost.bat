@echo off
title FinSight AI — Financial Intelligence Agent
echo ========================================================
echo   FinSight AI - Financial Intelligence Agent
echo   Turn financial documents into business decisions.
echo ========================================================
echo.

cd /d "%~dp0"

IF EXIST "venv\Scripts\python.exe" (
    set PYTHON_CMD=venv\Scripts\python.exe
) ELSE (
    set PYTHON_CMD=python
)

echo Starting FinSight AI server on http://localhost:8000 ...

echo Opening default web browser...
start http://localhost:8000

"%PYTHON_CMD%" run_server.py
pause
