#!/usr/bin/env bash
set -euo pipefail

# Activate conda environment
export CONDA_NO_PLUGINS=true
source /home/dellio/anaconda3/etc/profile.d/conda.sh
conda activate signbart_tf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/lsa64-63kpts.yaml"
DATA_ROOT="data/lsa64-63kpts"
HOLDOUT="user0001"
DATA_LOSO="${DATA_ROOT}_LOSO_${HOLDOUT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-"$SCRIPT_DIR/outputs"}"
CKPT_DIR="${OUTPUT_ROOT}/lsa64/loso/checkpoints/lsa64_LOSO_${HOLDOUT}"
FP32_TFLITE="${CKPT_DIR}/final_model_fp32.tflite"
PTQ_DIR="${OUTPUT_ROOT}/lsa64/loso/exports/ptq/${HOLDOUT}"
PTQ_TFLITE="${PTQ_DIR}/model_dynamic_int8.tflite"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: Missing dataset folder: $DATA_ROOT" >&2
  exit 1
fi

printf "\n== Step 1: LOSO split creation ==\n"
python fix_loso.py --base_root "$DATA_ROOT" --holdouts all

printf "\n== Step 2: Smoke test training (LOSO %s) ==\n" "$HOLDOUT"
if [[ -f "$FP32_TFLITE" ]]; then
  echo "FP32 TFLite exists at $FP32_TFLITE; skipping training."
else
  python train_loso_functional.py \
    --config_path "$CONFIG" \
    --base_data_path "$DATA_ROOT" \
    --holdouts all \
    --holdout_only "$HOLDOUT" \
    --epochs 1 \
    --dataset_name lsa64 \
    --output_root "$OUTPUT_ROOT"
fi

printf "\n== Step 3: PTQ export (LOSO %s) ==\n" "$HOLDOUT"
if [[ -f "$PTQ_TFLITE" ]]; then
  echo "PTQ TFLite exists at $PTQ_TFLITE; skipping PTQ export."
else
  python ptq_export_batch.py \
    --config_path "$CONFIG" \
    --base_data_path "$DATA_ROOT" \
    --holdouts all \
    --holdout_only "$HOLDOUT" \
    --dataset_name lsa64 \
    --output_root "$OUTPUT_ROOT"
fi

printf "\n== Step 4: Evaluation grouping (FP32 vs PTQ) ==\n"
python test_tflite_models.py \
  --config_path "$CONFIG" \
  --data_path "$DATA_LOSO" \
  --fp32_tflite "$FP32_TFLITE" \
  --ptq_tflite "$PTQ_TFLITE"
