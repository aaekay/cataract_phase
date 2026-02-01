#!/bin/bash
# Minimal environment setup for the phase recognition experiments

set -e

echo "=========================================="
echo "Cataract Phase Experiments - Setup"
echo "=========================================="

if ! command -v uv &> /dev/null; then
  echo "Error: 'uv' is not installed. Install it with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

uv venv venv
source venv/bin/activate
uv pip install -r requirements.txt

echo ""
echo "Setup complete. Activate with:"
echo "  source venv/bin/activate"
