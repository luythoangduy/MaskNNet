#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${DINOV3_REPO:-third_party/dinov3}

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "[ok] DINOv3 repo already exists: $REPO_DIR"
  git -C "$REPO_DIR" pull --ff-only
else
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone https://github.com/facebookresearch/dinov3.git "$REPO_DIR"
fi

echo
echo "DINOv3 repo ready: $REPO_DIR"
echo
echo "Next, download an official checkpoint after accepting the license, then pass it as:"
echo "  app.meta.dinov3_repo=$REPO_DIR"
echo "  app.meta.dinov3_weights=/path/to/dinov3_vitb16_pretrain_lvd1689m.pth"
echo
echo "You can also set env vars:"
echo "  export DINOV3_REPO=$REPO_DIR"
echo "  export DINOV3_WEIGHTS=/path/to/checkpoint.pth"
