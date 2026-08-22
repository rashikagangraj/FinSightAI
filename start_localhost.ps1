# FinAgent RAG - PowerShell Local Host Runner
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  FinAgent RAG - Financial Intelligence Local Server" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$PythonPath = "python"
if (Test-Path ".\venv\Scripts\python.exe") {
    $PythonPath = ".\venv\Scripts\python.exe"
}

Write-Host "Starting FinAgent server on http://localhost:8000 ..." -ForegroundColor Green
Start-Process "http://localhost:8000"

& $PythonPath run_server.py
