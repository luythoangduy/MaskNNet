#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required but was not found in PATH."
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

echo "Creating .venv with Python 3.10..."
uv venv .venv --python 3.10

PYTHON="$ROOT_DIR/.venv/bin/python"

echo "Installing PyTorch CUDA 12.8 packages..."
uv pip install --python "$PYTHON" \
  torch==2.7.0 \
  torchvision==0.22.0 \
  torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

echo "Installing project requirements..."
uv pip install --python "$PYTHON" -r requirements.txt

echo "Installing project in editable mode..."
uv pip install --python "$PYTHON" -e .

echo "Ensuring pip is available inside .venv..."
"$PYTHON" -m ensurepip --upgrade
if [ ! -e "$ROOT_DIR/.venv/bin/pip" ] && [ -e "$ROOT_DIR/.venv/bin/pip3" ]; then
  ln -s pip3 "$ROOT_DIR/.venv/bin/pip"
fi

echo "Running smoke test..."
"$PYTHON" - <<'PY'
import numpy
import torch
import torchvision

print("python ok")
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("numpy", numpy.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("cuda_arch_list", torch.cuda.get_arch_list() if torch.cuda.is_available() else [])
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

echo
echo "Setup complete."
echo "Activate with: source .venv/bin/activate"
echo "Run help with: python foundad/main.py --help"
