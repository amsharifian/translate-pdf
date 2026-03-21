#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-qwen3:8b}

echo "Pulling Ollama model: ${MODEL}"
ollama pull "${MODEL}"

echo "Done. You can now run:"
echo "  translate-pdf --provider ollama --model ${MODEL} -i ./docs/*.pdf"
