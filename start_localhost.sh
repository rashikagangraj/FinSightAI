#!/usr/bin/env bash
echo "========================================================"
echo "  FinAgent RAG - Financial Intelligence Local Server"
echo "========================================================"
echo ""

if [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
else
    PYTHON_CMD="python3"
fi

echo "Starting FinAgent server on http://localhost:8000 ..."
$PYTHON_CMD run_server.py
