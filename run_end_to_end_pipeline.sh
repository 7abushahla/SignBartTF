#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths
RAW_DIR="${RAW_DIR:-"$ROOT_DIR/MLR511-ArabicSignLanguage-Dataset-MP4"}"
FLIPPED_DIR="${FLIPPED_DIR:-"$ROOT_DIR/MLR511-ArabicSignLanguage-Dataset-MP4_FLIPPED"}"
KEYPOINTS_COUNT="${KEYPOINTS_COUNT:-63}"
DATASET_NAME="${DATASET_NAME:-arabic_asl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-"$ROOT_DIR/outputs"}"

if [[ -z "${DATA_DIR:-}" ]]; then
  case "$KEYPOINTS_COUNT" in
    63) DATA_DIR="$ROOT_DIR/data/arabic-asl-63kpts" ;;
    65) DATA_DIR="$ROOT_DIR/data/arabic-asl-65kpts" ;;
    90) DATA_DIR="$ROOT_DIR/data/arabic-asl-90kpts" ;;
    *)
      echo "Unsupported KEYPOINTS_COUNT: $KEYPOINTS_COUNT" >&2
      exit 1
      ;;
  esac
fi

if [[ -z "${CONFIG_PATH:-}" ]]; then
  case "$KEYPOINTS_COUNT" in
    63) CONFIG_PATH="$ROOT_DIR/configs/arabic-asl-63kpts.yaml" ;;
    65) CONFIG_PATH="$ROOT_DIR/configs/arabic-asl-65kpts.yaml" ;;
    90) CONFIG_PATH="$ROOT_DIR/configs/arabic-asl-90kpts.yaml" ;;
    *)
      echo "Unsupported KEYPOINTS_COUNT: $KEYPOINTS_COUNT" >&2
      exit 1
      ;;
  esac
fi

# Training settings
HOLDOUT_ONLY="${HOLDOUT_ONLY:-""}"
EPOCHS="${EPOCHS:-80}"
LR="${LR:-2e-4}"
SEED="${SEED:-42}"
NO_VALIDATION="${NO_VALIDATION:-1}"
RUN_LOSO="${RUN_LOSO:-1}"

# Full-dataset (non-LOSO) training settings
RUN_FULL_DATASET="${RUN_FULL_DATASET:-1}"
FULL_EPOCHS="${FULL_EPOCHS:-$EPOCHS}"
FULL_LR="${FULL_LR:-$LR}"
FULL_SEED="${FULL_SEED:-$SEED}"
FULL_EXP_NAME="${FULL_EXP_NAME:-${DATASET_NAME}_full}"

# Quantization / conversion settings
RUN_PTQ="${RUN_PTQ:-1}"
RUN_QAT="${RUN_QAT:-1}"
RUN_COLLECT="${RUN_COLLECT:-1}"

# Data prep toggles (useful for keypoints-only datasets)
SKIP_FLIP="${SKIP_FLIP:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"

QAT_EPOCHS="${QAT_EPOCHS:-20}"
QAT_BATCH_SIZE="${QAT_BATCH_SIZE:-4}"
QAT_LR="${QAT_LR:-5e-5}"

PTQ_LOSO_DIR="${PTQ_LOSO_DIR:-$OUTPUT_ROOT/$DATASET_NAME/loso/exports/ptq}"
PTQ_FULL_DIR="${PTQ_FULL_DIR:-$OUTPUT_ROOT/$DATASET_NAME/full/exports/ptq}"
QAT_LOSO_DIR="${QAT_LOSO_DIR:-$OUTPUT_ROOT/$DATASET_NAME/loso/exports/qat}"
QAT_FULL_DIR="${QAT_FULL_DIR:-$OUTPUT_ROOT/$DATASET_NAME/full/exports/qat}"

# Flip settings
FLIP_USERS="${FLIP_USERS:-"user01,user02"}"
HOLDOUTS="${HOLDOUTS:-"user01,user08,user11"}"

log() {
  echo "[PIPELINE] $*"
}

ensure_ffmpeg() {
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found. Please install ffmpeg and retry." >&2
    exit 1
  fi
}

# Try to activate conda environment `signbart_tf` if available.
ensure_conda_env() {
  local env_name="signbart_tf"
  if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" == "$env_name" ]]; then
    log "Conda env '$env_name' already active."
    return 0
  fi

  # Prefer sourcing conda.sh from common locations to avoid `conda info` issues in
  # some environments (plugins / sandbox restrictions).
  local -a conda_sh_candidates=()
  if [[ -n "${CONDA_SH:-}" ]]; then
    conda_sh_candidates+=("${CONDA_SH}")
  fi
  conda_sh_candidates+=(
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/mambaforge/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
  )

  for conda_sh in "${conda_sh_candidates[@]}"; do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck disable=SC1090
      source "$conda_sh"
      if conda activate "$env_name" >/dev/null 2>&1; then
        log "Activated conda env: $env_name"
        return 0
      fi
    fi
  done

  # Fallback: try to locate conda.sh via `conda info --base`
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base=$(conda info --base 2>/dev/null || true)
    if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$conda_base/etc/profile.d/conda.sh"
      if conda activate "$env_name" >/dev/null 2>&1; then
        log "Activated conda env: $env_name"
        return 0
      fi
    fi
  fi

  echo "Failed to activate conda env '$env_name'. Please activate it manually and retry." >&2
  exit 1
}

should_flip_user() {
  local user="$1"
  IFS=',' read -r -a users <<< "$FLIP_USERS"
  for u in "${users[@]}"; do
    if [[ "$user" == "$u" ]]; then
      return 0
    fi
  done
  return 1
}

flip_videos_if_needed() {
  if [[ -f "$FLIPPED_DIR/.flip_complete" ]]; then
    log "Flip step: already complete, skipping."
    return 0
  fi

  ensure_ffmpeg
  mkdir -p "$FLIPPED_DIR"

  log "Flip step: preparing normalized dataset in $FLIPPED_DIR"
  shopt -s nullglob

  for user_dir in "$RAW_DIR"/user*; do
    [[ -d "$user_dir" ]] || continue
    user="$(basename "$user_dir")"

    for gesture_dir in "$user_dir"/G*; do
      [[ -d "$gesture_dir" ]] || continue

      for video in "$gesture_dir"/*.mp4 "$gesture_dir"/*.avi; do
        [[ -f "$video" ]] || continue
        rel="${video#"$RAW_DIR"/}"
        out="$FLIPPED_DIR/$rel"
        mkdir -p "$(dirname "$out")"

        if [[ -f "$out" ]]; then
          continue
        fi

        if should_flip_user "$user"; then
          log "Flipping $rel"
          ffmpeg -y -i "$video" -vf hflip -c:v libx264 -preset veryfast -crf 18 -c:a copy "$out" >/dev/null 2>&1
        else
          log "Copying $rel"
          cp -n "$video" "$out"
        fi
      done
    done
  done

  touch "$FLIPPED_DIR/.flip_complete"
  log "Flip step: complete."
}

extract_keypoints_if_needed() {
  if [[ -d "$DATA_DIR/all" ]] && find "$DATA_DIR/all" -type f -name "*.pkl" -print -quit | grep -q .; then
    log "Keypoint extraction: output exists, skipping."
    return 0
  fi

  local extractor="${EXTRACT_SCRIPT:-}"
  if [[ -z "$extractor" ]]; then
    case "$KEYPOINTS_COUNT" in
      63) extractor="$ROOT_DIR/extract_63_keypoints.py" ;;
      65) extractor="$ROOT_DIR/extract_65_keypoints.py" ;;
      90) extractor="$ROOT_DIR/extract_90_keypoints.py" ;;
      *)
        echo "Unsupported KEYPOINTS_COUNT: $KEYPOINTS_COUNT" >&2
        exit 1
        ;;
    esac
  fi

  log "Keypoint extraction: running $(basename "$extractor")"
  python "$extractor" \
    --input_dir "$FLIPPED_DIR" \
    --output_dir "$DATA_DIR"
}

create_loso_if_needed() {
  local holdouts_csv
  if [[ "$HOLDOUTS" == "all" ]]; then
    if [[ ! -d "$DATA_DIR/all" ]]; then
      echo "Missing $DATA_DIR/all for holdout discovery" >&2
      exit 1
    fi
    holdouts_csv=$(find "$DATA_DIR/all" -type f -name "*.pkl" -maxdepth 2 -print0 \
      | xargs -0 -n1 basename \
      | awk -F_ '{print $1}' \
      | sort -u \
      | tr '\n' ',' \
      | sed 's/,$//')
  else
    holdouts_csv="$HOLDOUTS"
  fi

  if [[ -z "$holdouts_csv" ]]; then
    echo "No holdouts found for LOSO." >&2
    exit 1
  fi

  IFS=',' read -r -a holdouts <<< "$holdouts_csv"
  for holdout in "${holdouts[@]}"; do
    loso_dir="${DATA_DIR}_LOSO_${holdout}"
    if [[ ! -d "$loso_dir/train" ]] || [[ ! -d "$loso_dir/test" ]]; then
      log "LOSO splits: missing ${holdout}, regenerating."
      python "$ROOT_DIR/fix_loso.py" \
        --base_root "$DATA_DIR" \
        --holdouts "$holdouts_csv"
      return 0
    fi
  done

  log "LOSO splits: already present for all holdouts, skipping."
}

train_model() {
  if [[ "$RUN_LOSO" != "1" ]]; then
    log "LOSO training: disabled (RUN_LOSO=$RUN_LOSO)."
    return 0
  fi

  log "Training: starting"
  cmd=(
    python "$ROOT_DIR/train_loso_functional.py"
    --config_path "$CONFIG_PATH"
    --base_data_path "$DATA_DIR"
    --epochs "$EPOCHS"
    --lr "$LR"
    --seed "$SEED"
    --dataset_name "$DATASET_NAME"
    --output_root "$OUTPUT_ROOT"
  )

  if [[ -n "$HOLDOUT_ONLY" ]]; then
    cmd+=(--holdout_only "$HOLDOUT_ONLY")
  fi
  if [[ -n "$HOLDOUTS" ]]; then
    cmd+=(--holdouts "$HOLDOUTS")
  fi

  if [[ "$NO_VALIDATION" == "1" ]]; then
    cmd+=(--no_validation)
  fi

  "${cmd[@]}"
}

train_full_dataset() {
  if [[ "$RUN_FULL_DATASET" != "1" ]]; then
    log "Full dataset training: disabled (RUN_FULL_DATASET=$RUN_FULL_DATASET)."
    return 0
  fi

  log "Full dataset training: starting"
  python "$ROOT_DIR/train_full_dataset.py" \
    --config_path "$CONFIG_PATH" \
    --base_data_path "$DATA_DIR" \
    --epochs "$FULL_EPOCHS" \
    --lr "$FULL_LR" \
    --seed "$FULL_SEED" \
    --exp_name "$FULL_EXP_NAME" \
    --dataset_name "$DATASET_NAME" \
    --output_root "$OUTPUT_ROOT"
}

evaluate_model() {
  if [[ "$RUN_COLLECT" != "1" ]]; then
    log "Evaluation: disabled (RUN_COLLECT=$RUN_COLLECT)."
    return 0
  fi
  if [[ "$RUN_LOSO" != "1" ]]; then
    log "Evaluation: skipped (RUN_LOSO=$RUN_LOSO)."
    return 0
  fi

  log "Evaluation: running collect_results.py (default LOSO users)"
  python "$ROOT_DIR/collect_results.py" \
    --run_evaluation \
    --config_path "$CONFIG_PATH" \
    --base_data_path "$DATA_DIR" \
    --dataset_name "$DATASET_NAME" \
    --run_type "loso" \
    --output_root "$OUTPUT_ROOT" \
    --ptq_base_dir "$PTQ_LOSO_DIR" \
    --qat_base_dir "$QAT_LOSO_DIR"
}

run_ptq() {
  if [[ "$RUN_PTQ" != "1" ]]; then
    log "PTQ: disabled (RUN_PTQ=$RUN_PTQ)."
    return 0
  fi

  if [[ "$RUN_LOSO" == "1" ]]; then
    log "PTQ (LOSO): exporting dynamic-range INT8 models"
    python "$ROOT_DIR/ptq_export_batch.py" \
      --config_path "$CONFIG_PATH" \
      --base_data_path "$DATA_DIR" \
      --holdouts "$HOLDOUTS" \
      --dataset_name "$DATASET_NAME" \
      --output_root "$OUTPUT_ROOT" \
      --output_base_dir "$PTQ_LOSO_DIR"
  else
    log "PTQ (LOSO): skipped (RUN_LOSO=$RUN_LOSO)."
  fi

  log "PTQ (Full dataset): exporting dynamic-range INT8 model"
  python "$ROOT_DIR/ptq_export.py" \
    --config_path "$CONFIG_PATH" \
    --checkpoint "$OUTPUT_ROOT/$DATASET_NAME/full/checkpoints/${FULL_EXP_NAME}/final_model.h5" \
    --output_dir "$PTQ_FULL_DIR"
}

run_qat() {
  if [[ "$RUN_QAT" != "1" ]]; then
    log "QAT: disabled (RUN_QAT=$RUN_QAT)."
    return 0
  fi

  if [[ "$RUN_LOSO" == "1" ]]; then
    log "QAT (LOSO): fine-tuning and export"
    cmd=(
      python "$ROOT_DIR/train_loso_functional_qat_batch.py"
      --config_path "$CONFIG_PATH"
      --base_data_path "$DATA_DIR"
      --holdouts "$HOLDOUTS"
      --dataset_name "$DATASET_NAME"
      --output_root "$OUTPUT_ROOT"
      --output_base_dir "$QAT_LOSO_DIR"
      --qat_epochs "$QAT_EPOCHS"
      --batch_size "$QAT_BATCH_SIZE"
      --lr "$QAT_LR"
      --seed "$SEED"
    )

    # No validation by default (matches FP32 training behavior).
    if [[ "$NO_VALIDATION" == "1" ]]; then
      cmd+=(--no_validation)
    fi

    "${cmd[@]}"
  else
    log "QAT (LOSO): skipped (RUN_LOSO=$RUN_LOSO)."
  fi

  log "QAT (Full dataset): fine-tuning and export"
  python "$ROOT_DIR/train_loso_functional_qat.py" \
    --config_path "$CONFIG_PATH" \
    --data_path "$DATA_DIR" \
    --checkpoint "$OUTPUT_ROOT/$DATASET_NAME/full/checkpoints/${FULL_EXP_NAME}/final_model.h5" \
    --output_dir "$QAT_FULL_DIR" \
    --qat_epochs "$QAT_EPOCHS" \
    --batch_size "$QAT_BATCH_SIZE" \
    --lr "$QAT_LR" \
    --seed "$SEED" \
    --no_validation
}

main() {
  ensure_conda_env
  if [[ "$SKIP_FLIP" == "1" ]]; then
    log "Flip step: skipped (SKIP_FLIP=$SKIP_FLIP)."
  else
    flip_videos_if_needed
  fi

  if [[ "$SKIP_EXTRACT" == "1" ]]; then
    if [[ -d "$DATA_DIR/all" ]] && find "$DATA_DIR/all" -type f -name "*.pkl" -print -quit | grep -q .; then
      log "Keypoint extraction: skipped (SKIP_EXTRACT=$SKIP_EXTRACT)."
    else
      echo "SKIP_EXTRACT=1 but no keypoints found in $DATA_DIR/all. Aborting." >&2
      exit 1
    fi
  else
    if [[ "$SKIP_FLIP" == "1" && ! -d "$FLIPPED_DIR" ]]; then
      echo "SKIP_FLIP=1 but FLIPPED_DIR does not exist: $FLIPPED_DIR" >&2
      exit 1
    fi
    extract_keypoints_if_needed
  fi
  create_loso_if_needed
  train_model
  train_full_dataset
  run_ptq
  run_qat
  evaluate_model
  log "Pipeline complete."
}

main "$@"
