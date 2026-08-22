#!/usr/bin/env bash
# FinSight AI — Ollama model setup script
# Run: bash scripts/setup_ollama.sh

set -e

echo "=================================================="
echo " FinSight AI — Financial Intelligence Agent"
echo " Turn financial documents into business decisions."
echo "=================================================="


# Check if ollama is installed
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "Ollama not found. Install it from: https://ollama.com/download"
    echo ""
    echo "Quick install (macOS/Linux):"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

echo ""
echo "Ollama found: $(ollama --version)"
echo ""

# Pull CPU-friendly LLM model
echo "[1/2] Pulling LLM: llama3.2:3b (CPU-friendly, ~2GB)"
echo "      Alternatively use: qwen2.5:3b for better financial text quality"
ollama pull llama3.2:3b

echo ""
echo "[2/2] Pulling embedding model: nomic-embed-text (~274MB)"
ollama pull nomic-embed-text

echo ""
echo "=================================================="
echo " Setup complete! Models available:"
ollama list
echo ""
echo " Quick start:"
echo "   cp .env.example .env"
echo "   python scripts/ingest_sample_docs.py"
echo "   python examples/demo_cli.py demo"
echo "=================================================="
