#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-.venv/bin/python}

"$PYTHON" -m pip install --upgrade "transformers>=4.56.0" "safetensors"

cat <<'MSG'

Hugging Face DINOv3 support installed.

If the model is gated for your account, run:
  huggingface-cli login

Then train/evaluate with:
  app=train_dinov3
  app.meta.dinov3_source=hf
  app.meta.dinov3_hf_model=facebook/dinov3-vitb16-pretrain-lvd1689m
MSG
