#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/train_channel_cases.sh <case|all|synth_sweep|topk_sweep|lambda_sweep|mask_sweep|temperature_sweep>

Cases:
  baseline              FoundAD baseline, no channel routing
  top_delta             sensitivity-only channel routing
  reliability           masked reliability gate
  background            reliability + background preservation
  full                  reliability + background preservation + delta contrastive

Sweep groups:
  synth_sweep           feature synthetic modes: dropout, noise, replacement, mixed
  topk_sweep            channel_topk: 16, 32, 64, 128
  lambda_sweep          lambda_selected/background/contrastive small sweep
  mask_sweep            token mask modes and mask ratios
  temperature_sweep     delta contrastive temperatures
  all                   baseline, top_delta, reliability, background, full

Environment variables:
  PYTHON                Python runner. Default: python
  APP                   Hydra app config. Default: train_dinov2
  DATASET               Dataset name. Default: visa
  DATA_NAME             Few-shot folder name. Default: visa_1shot
  DATA_PATH             Parent folder containing DATA_NAME. Default: data/fewshot
  BATCH_SIZE            Training batch size. Default: 8
  EPOCHS                Number of epochs. Default: 2000
  DEVICES               Hydra devices list. Default: [cuda:0]
  CROP_SIZE             Optional crop override. Empty by default
  DIY_PREFIX            Prefix for diy_name. Default: _cr
  EXTRA                 Extra Hydra overrides appended to every run.

Example:
  DATASET=mvtec DATA_NAME=mvtec_1shot DATA_PATH=/data/fewshot EPOCHS=2000 scripts/train_channel_cases.sh full
USAGE
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PYTHON=${PYTHON:-python}
APP=${APP:-train_dinov2}
DATASET=${DATASET:-visa}
DATA_NAME=${DATA_NAME:-visa_1shot}
DATA_PATH=${DATA_PATH:-data/fewshot}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-2000}
DEVICES=${DEVICES:-"[cuda:0]"}
CROP_SIZE=${CROP_SIZE:-}
DIY_PREFIX=${DIY_PREFIX:-_cr}
EXTRA=${EXTRA:-}

BASE_OVERRIDES=(
  "mode=train"
  "app=${APP}"
  "data.dataset=${DATASET}"
  "data.data_name=${DATA_NAME}"
  "data.data_path=${DATA_PATH}"
  "data.batch_size=${BATCH_SIZE}"
  "optimization.epochs=${EPOCHS}"
  "devices=${DEVICES}"
)

if [[ -n "${CROP_SIZE}" ]]; then
  BASE_OVERRIDES+=("app.meta.crop_size=${CROP_SIZE}")
fi

run_train() {
  local name="$1"
  shift

  local overrides=("${BASE_OVERRIDES[@]}" "diy_name=${DIY_PREFIX}_${name}" "$@")

  if [[ -n "${EXTRA}" ]]; then
    # Intentional word splitting lets EXTRA contain multiple Hydra overrides.
    # shellcheck disable=SC2206
    local extra_args=(${EXTRA})
    overrides+=("${extra_args[@]}")
  fi

  echo
  echo "==> Training case: ${name}"
  printf '    %q' "${PYTHON}" foundad/main.py "${overrides[@]}"
  echo
  "${PYTHON}" foundad/main.py "${overrides[@]}"
}

case_baseline() {
  run_train "baseline" \
    "app.meta.use_channel_routing=false" \
    "app.meta.use_channel_weighted_loss=false" \
    "app.meta.lambda_background_preservation=0.0" \
    "app.meta.lambda_delta_contrastive=0.0"
}

case_top_delta() {
  run_train "top_delta" \
    "app.meta.use_channel_routing=true" \
    "app.meta.use_channel_weighted_loss=true" \
    "app.meta.use_masked_channel_reliability=false" \
    "app.meta.lambda_background_preservation=0.0" \
    "app.meta.lambda_delta_contrastive=0.0"
}

case_reliability() {
  run_train "reliability" \
    "app.meta.use_channel_routing=true" \
    "app.meta.use_channel_weighted_loss=true" \
    "app.meta.use_masked_channel_reliability=true" \
    "app.meta.lambda_background_preservation=0.0" \
    "app.meta.lambda_delta_contrastive=0.0"
}

case_background() {
  run_train "background" \
    "app.meta.use_channel_routing=true" \
    "app.meta.use_channel_weighted_loss=true" \
    "app.meta.use_masked_channel_reliability=true" \
    "app.meta.lambda_background_preservation=0.5" \
    "app.meta.lambda_delta_contrastive=0.0"
}

case_full() {
  run_train "full" \
    "app.meta.use_channel_routing=true" \
    "app.meta.use_channel_weighted_loss=true" \
    "app.meta.use_masked_channel_reliability=true" \
    "app.meta.lambda_background_preservation=0.5" \
    "app.meta.lambda_delta_contrastive=0.1" \
    "app.meta.delta_contrastive_temperature=0.2"
}

synth_sweep() {
  for mode in dropout noise replacement mixed; do
    run_train "synth_${mode}" \
      "app.meta.use_channel_routing=true" \
      "app.meta.use_channel_weighted_loss=true" \
      "app.meta.use_masked_channel_reliability=true" \
      "app.meta.feature_synth_mode=${mode}" \
      "app.meta.lambda_background_preservation=0.5" \
      "app.meta.lambda_delta_contrastive=0.1"
  done
}

topk_sweep() {
  for topk in 16 32 64 128; do
    run_train "topk_${topk}" \
      "app.meta.use_channel_routing=true" \
      "app.meta.use_channel_weighted_loss=true" \
      "app.meta.use_masked_channel_reliability=true" \
      "app.meta.channel_topk=${topk}" \
      "app.meta.lambda_background_preservation=0.5" \
      "app.meta.lambda_delta_contrastive=0.1"
  done
}

lambda_sweep() {
  for value in 0.25 0.5 1.0 2.0; do
    run_train "lambda_selected_${value}" \
      "app.meta.use_channel_routing=true" \
      "app.meta.use_channel_weighted_loss=true" \
      "app.meta.use_masked_channel_reliability=true" \
      "app.meta.lambda_selected=${value}"
  done

  for value in 0.0 0.1 0.5 1.0; do
    run_train "lambda_bg_${value}" \
      "app.meta.use_channel_routing=true" \
      "app.meta.use_channel_weighted_loss=true" \
      "app.meta.use_masked_channel_reliability=true" \
      "app.meta.lambda_background_preservation=${value}" \
      "app.meta.lambda_delta_contrastive=0.0"
  done

  for value in 0.0 0.05 0.1 0.2; do
    run_train "lambda_delta_${value}" \
      "app.meta.use_channel_routing=true" \
      "app.meta.use_channel_weighted_loss=true" \
      "app.meta.use_masked_channel_reliability=true" \
      "app.meta.lambda_background_preservation=0.5" \
      "app.meta.lambda_delta_contrastive=${value}"
  done
}

mask_sweep() {
  for mode in rectangle random_blob random_tokens; do
    run_train "mask_${mode}" \
      "app.meta.use_channel_routing=true" \
      "app.meta.use_channel_weighted_loss=true" \
      "app.meta.use_masked_channel_reliability=true" \
      "app.meta.token_mask_mode=${mode}" \
      "app.meta.lambda_background_preservation=0.5" \
      "app.meta.lambda_delta_contrastive=0.1"
  done

  run_train "mask_ratio_002_010" \
    "app.meta.use_channel_routing=true" \
    "app.meta.use_channel_weighted_loss=true" \
    "app.meta.use_masked_channel_reliability=true" \
    "app.meta.token_mask_min_ratio=0.02" \
    "app.meta.token_mask_max_ratio=0.10" \
    "app.meta.lambda_background_preservation=0.5" \
    "app.meta.lambda_delta_contrastive=0.1"

  run_train "mask_ratio_005_025" \
    "app.meta.use_channel_routing=true" \
    "app.meta.use_channel_weighted_loss=true" \
    "app.meta.use_masked_channel_reliability=true" \
    "app.meta.token_mask_min_ratio=0.05" \
    "app.meta.token_mask_max_ratio=0.25" \
    "app.meta.lambda_background_preservation=0.5" \
    "app.meta.lambda_delta_contrastive=0.1"

  run_train "mask_ratio_010_035" \
    "app.meta.use_channel_routing=true" \
    "app.meta.use_channel_weighted_loss=true" \
    "app.meta.use_masked_channel_reliability=true" \
    "app.meta.token_mask_min_ratio=0.10" \
    "app.meta.token_mask_max_ratio=0.35" \
    "app.meta.lambda_background_preservation=0.5" \
    "app.meta.lambda_delta_contrastive=0.1"
}

temperature_sweep() {
  for value in 0.07 0.1 0.2 0.5; do
    run_train "temperature_${value}" \
      "app.meta.use_channel_routing=true" \
      "app.meta.use_channel_weighted_loss=true" \
      "app.meta.use_masked_channel_reliability=true" \
      "app.meta.lambda_background_preservation=0.5" \
      "app.meta.lambda_delta_contrastive=0.1" \
      "app.meta.delta_contrastive_temperature=${value}"
  done
}

case "$1" in
  baseline) case_baseline ;;
  top_delta) case_top_delta ;;
  reliability) case_reliability ;;
  background) case_background ;;
  full) case_full ;;
  synth_sweep) synth_sweep ;;
  topk_sweep) topk_sweep ;;
  lambda_sweep) lambda_sweep ;;
  mask_sweep) mask_sweep ;;
  temperature_sweep) temperature_sweep ;;
  all)
    case_baseline
    case_top_delta
    case_reliability
    case_background
    case_full
    ;;
  *)
    echo "Unknown case: $1" >&2
    usage >&2
    exit 2
    ;;
esac
