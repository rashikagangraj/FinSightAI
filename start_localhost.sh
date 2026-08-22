#!/usr/bin/env bash
echo "========================================================"
echo "  FinSight AI - Financial Intelligence Agent"
echo "  Turn financial documents into business decisions."
echo "========================================================"
echo ""

if [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
else
    PYTHON_CMD="python3"
fi

echo "Starting FinSight AI server on http://localhost:8000 ..."

$PYTHON_CMD run_server.py
