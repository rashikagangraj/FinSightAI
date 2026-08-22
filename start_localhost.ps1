# FinSight AI - PowerShell Local Host Runner
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  FinSight AI - Financial Intelligence Agent" -ForegroundColor Cyan
Write-Host "  Turn financial documents into business decisions." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$PythonPath = "python"
if (Test-Path ".\venv\Scripts\python.exe") {
    $PythonPath = ".\venv\Scripts\python.exe"
}

Write-Host "Starting FinSight AI server on http://localhost:8000 ..." -ForegroundColor Green

Start-Process "http://localhost:8000"

& $PythonPath run_server.py
