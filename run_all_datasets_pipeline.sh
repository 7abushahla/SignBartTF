#!/usr/bin/env bash
set -euo pipefail

# Run the standard end-to-end pipeline across multiple datasets using the same
# settings. The only per-dataset differences are DATASET_NAME/DATA_DIR/CONFIG_PATH.
#
# Default datasets:
#   arabic_asl (63kpts), lsa64 (63kpts), karsl100 (63kpts), karsl502 (63kpts)
#
# You can override the dataset list:
#   export DATASETS="karsl502,karsl100"
#
# Common knobs are inherited from the environment (with safe defaults):
#   OUTPUT_ROOT, HOLDOUTS, NO_VALIDATION, SEED, EPOCHS, QAT_EPOCHS,
#   RUN_FULL_DATASET, RUN_LOSO, RUN_PTQ, RUN_QAT, RUN_COLLECT,
#   SKIP_FLIP, SKIP_EXTRACT

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Common defaults (can be overridden by the caller)
export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-"$ROOT_DIR/outputs"}"
export HOLDOUTS="${HOLDOUTS:-all}"
export NO_VALIDATION="${NO_VALIDATION:-1}"
export SEED="${SEED:-379}"
export EPOCHS="${EPOCHS:-2}"
export QAT_EPOCHS="${QAT_EPOCHS:-1}"
export RUN_FULL_DATASET="${RUN_FULL_DATASET:-1}"
export RUN_LOSO="${RUN_LOSO:-1}"
export RUN_PTQ="${RUN_PTQ:-1}"
export RUN_QAT="${RUN_QAT:-1}"
export RUN_COLLECT="${RUN_COLLECT:-1}"
export SKIP_FLIP="${SKIP_FLIP:-1}"
export SKIP_EXTRACT="${SKIP_EXTRACT:-1}"

DATASETS="${DATASETS:-arabic_asl,lsa64,karsl100,karsl502}"

log() {
  echo "[MULTI] $*"
}

ensure_karsl100_root() {
  local src="$ROOT_DIR/data/karsl502-63kpts"
  local dst="$ROOT_DIR/data/karsl100-63kpts"
  local mode="${KARSL100_MODE:-symlink}"

  if [[ -d "$dst/all" && -f "$dst/label2id.json" && -f "$dst/id2label.json" ]]; then
    return 0
  fi

  if [[ ! -d "$src/all" ]]; then
    echo "ERROR: KArSL-502 source root not found: $src" >&2
    echo "       Cannot auto-build KArSL-100 without KArSL-502." >&2
    exit 1
  fi

  log "Building KArSL-100 subset dataset at $dst (mode=$mode)"
  python "$ROOT_DIR/build_karsl_subset.py" \
    --src_root "$src" \
    --dst_root "$dst" \
    --start 71 --end 170 \
    --mode "$mode"
}

run_one_dataset() {
  local ds="$1"
  case "$ds" in
    arabic_asl)
      export DATASET_NAME="arabic_asl"
      export DATA_DIR="$ROOT_DIR/data/arabic-asl-63kpts"
      export CONFIG_PATH="$ROOT_DIR/configs/arabic-asl-63kpts.yaml"
      ;;
    lsa64)
      export DATASET_NAME="lsa64"
      export DATA_DIR="$ROOT_DIR/data/lsa64-63kpts"
      export CONFIG_PATH="$ROOT_DIR/configs/lsa64-63kpts.yaml"
      ;;
    karsl100)
      ensure_karsl100_root
      export DATASET_NAME="karsl100"
      export DATA_DIR="$ROOT_DIR/data/karsl100-63kpts"
      export CONFIG_PATH="$ROOT_DIR/configs/karsl100-63kpts.yaml"
      ;;
    karsl502)
      export DATASET_NAME="karsl502"
      export DATA_DIR="$ROOT_DIR/data/karsl502-63kpts"
      export CONFIG_PATH="$ROOT_DIR/configs/karsl502-63kpts.yaml"
      ;;
    *)
      echo "ERROR: Unknown dataset key: $ds" >&2
      echo "Valid: arabic_asl, lsa64, karsl100, karsl502" >&2
      exit 1
      ;;
  esac

  log "============================================================"
  log "Running pipeline for dataset: $DATASET_NAME"
  log "  DATA_DIR     : $DATA_DIR"
  log "  CONFIG_PATH  : $CONFIG_PATH"
  log "  OUTPUT_ROOT  : $OUTPUT_ROOT"
  log "  HOLDOUTS     : $HOLDOUTS"
  log "  EPOCHS       : $EPOCHS"
  log "  QAT_EPOCHS   : $QAT_EPOCHS"
  log "  NO_VALIDATION: $NO_VALIDATION"
  log "============================================================"

  bash "$ROOT_DIR/run_end_to_end_pipeline.sh"
}

IFS=',' read -r -a dataset_list <<< "$DATASETS"
for ds in "${dataset_list[@]}"; do
  ds="$(echo "$ds" | xargs)"  # trim whitespace
  [[ -n "$ds" ]] || continue
  run_one_dataset "$ds"
done

log "All requested datasets completed."

