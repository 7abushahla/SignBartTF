#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_NAME="${DATASET_NAME:-arabic_asl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-"$ROOT_DIR/outputs"}"

# Paths (shared dataset)
RAW_DIR="${RAW_DIR:-"$ROOT_DIR/MLR511-ArabicSignLanguage-Dataset-MP4"}"
FLIPPED_DIR="${FLIPPED_DIR:-"$ROOT_DIR/MLR511-ArabicSignLanguage-Dataset-MP4_FLIPPED"}"
DATA_DIR_65="${DATA_DIR_65:-"$ROOT_DIR/data/arabic-asl-65kpts"}"
DATA_DIR_90="${DATA_DIR_90:-"$ROOT_DIR/data/arabic-asl-90kpts"}"

# Data prep toggles (useful for keypoints-only datasets)
SKIP_FLIP="${SKIP_FLIP:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"

# Training settings
HOLDOUT_ONLY="${HOLDOUT_ONLY:-""}"
EPOCHS="${EPOCHS:-80}"
LR="${LR:-2e-4}"
SEED="${SEED:-42}"
SEED_LIST="${SEED_LIST:-"42,511,999983,1000003,324528439"}"
NO_VALIDATION="${NO_VALIDATION:-1}"
RUN_LOSO="${RUN_LOSO:-1}"

# Full-dataset (non-LOSO) training settings
RUN_FULL_DATASET="${RUN_FULL_DATASET:-1}"
FULL_EPOCHS="${FULL_EPOCHS:-$EPOCHS}"
FULL_LR="${FULL_LR:-$LR}"
FULL_SEED="${FULL_SEED:-$SEED}"

# PTQ/QAT/Eval toggles
RUN_PTQ="${RUN_PTQ:-1}"
RUN_QAT="${RUN_QAT:-1}"
RUN_COLLECT="${RUN_COLLECT:-1}"

# QAT settings
QAT_EPOCHS="${QAT_EPOCHS:-20}"
QAT_BATCH="${QAT_BATCH:-4}"
QAT_LR="${QAT_LR:-5e-5}"

# Flip settings
FLIP_USERS="${FLIP_USERS:-"user01,user02"}"
HOLDOUTS="${HOLDOUTS:-"all"}"

# Variants to run (space-separated list)
VARIANTS="${VARIANTS:-"42kpts-hands 44kpts-hands-wrists 44kpts-hands-shoulders 46kpts-hands-shoulders-wrists 48kpts-hands-shoulders-elbows-wrists 65kpts 90kpts"}"

log() {
  echo "[VARIANT-PIPELINE] $*"
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
  local data_dir="$1"
  local extract_script="$2"
  if [[ -d "$data_dir/all" ]] && find "$data_dir/all" -type f -name "*.pkl" -print -quit | grep -q .; then
    log "Keypoint extraction: output exists, skipping."
    return 0
  fi

  log "Keypoint extraction: running $extract_script"
  python "$ROOT_DIR/$extract_script" \
    --input_dir "$FLIPPED_DIR" \
    --output_dir "$data_dir"
}

create_loso_if_needed() {
  local data_dir="$1"
  local holdouts_csv
  if [[ "$HOLDOUTS" == "all" ]]; then
    if [[ ! -d "$data_dir/all" ]]; then
      echo "Missing $data_dir/all for holdout discovery" >&2
      exit 1
    fi
    holdouts_csv=$(find "$data_dir/all" -type f -name "*.pkl" -maxdepth 2 -print0 \
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
    loso_dir="${data_dir}_LOSO_${holdout}"
    if [[ ! -d "$loso_dir/train" ]] || [[ ! -d "$loso_dir/test" ]]; then
      log "LOSO splits: missing ${holdout}, regenerating."
      python "$ROOT_DIR/fix_loso.py" \
        --base_root "$data_dir" \
        --holdouts "$holdouts_csv"
      return 0
    fi
  done

  log "LOSO splits: already present for all holdouts, skipping."
}

resolve_variant() {
  local variant="$1"
  local config_path
  local exp_prefix
  local data_dir
  local extract_script

  case "$variant" in
    42kpts-hands)
      config_path="$ROOT_DIR/configs/arabic-asl-42kpts-hands.yaml"
      exp_prefix="arabic_asl_42kpts_hands"
      data_dir="$DATA_DIR_65"
      extract_script="extract_65_keypoints.py"
      ;;
    44kpts-hands-wrists)
      config_path="$ROOT_DIR/configs/arabic-asl-44kpts-hands-wrists.yaml"
      exp_prefix="arabic_asl_44kpts_hands_wrists"
      data_dir="$DATA_DIR_65"
      extract_script="extract_65_keypoints.py"
      ;;
    44kpts-hands-shoulders)
      config_path="$ROOT_DIR/configs/arabic-asl-44kpts-hands-shoulders.yaml"
      exp_prefix="arabic_asl_44kpts_hands_shoulders"
      data_dir="$DATA_DIR_65"
      extract_script="extract_65_keypoints.py"
      ;;
    46kpts-hands-shoulders-wrists)
      config_path="$ROOT_DIR/configs/arabic-asl-46kpts-hands-shoulders-wrists.yaml"
      exp_prefix="arabic_asl_46kpts_hands_shoulders_wrists"
      data_dir="$DATA_DIR_65"
      extract_script="extract_65_keypoints.py"
      ;;
    48kpts-hands-shoulders-elbows-wrists)
      config_path="$ROOT_DIR/configs/arabic-asl-48kpts-hands-shoulders-elbows-wrists.yaml"
      exp_prefix="arabic_asl_48kpts_hands_shoulders_elbows_wrists"
      data_dir="$DATA_DIR_65"
      extract_script="extract_65_keypoints.py"
      ;;
    65kpts)
      config_path="$ROOT_DIR/configs/arabic-asl-65kpts.yaml"
      exp_prefix="arabic_asl_65kpts_pose_hands"
      data_dir="$DATA_DIR_65"
      extract_script="extract_65_keypoints.py"
      ;;
    90kpts)
      config_path="$ROOT_DIR/configs/arabic-asl-90kpts.yaml"
      exp_prefix="arabic_asl_90kpts_full"
      data_dir="$DATA_DIR_90"
      extract_script="extract_90_keypoints.py"
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      exit 1
      ;;
  esac

  echo "$config_path|$exp_prefix|$data_dir|$extract_script"
}

run_variant() {
  local variant="$1"
  IFS='|' read -r config_path exp_prefix data_dir extract_script <<< "$(resolve_variant "$variant")"

  log "Running variant: $variant"
  log "  Config: $config_path"
  log "  Exp prefix: $exp_prefix"
  log "  Data dir: $data_dir"

  if [[ "$SKIP_EXTRACT" == "1" ]]; then
    if [[ -d "$data_dir/all" ]] && find "$data_dir/all" -type f -name "*.pkl" -print -quit | grep -q .; then
      log "Keypoint extraction: skipped (SKIP_EXTRACT=$SKIP_EXTRACT)."
    else
      echo "SKIP_EXTRACT=1 but no keypoints found in $data_dir/all. Aborting." >&2
      exit 1
    fi
  else
    if [[ "$SKIP_FLIP" == "1" && ! -d "$FLIPPED_DIR" ]]; then
      echo "SKIP_FLIP=1 but FLIPPED_DIR does not exist: $FLIPPED_DIR" >&2
      exit 1
    fi
    extract_keypoints_if_needed "$data_dir" "$extract_script"
  fi
  if [[ "$RUN_LOSO" == "1" ]]; then
    create_loso_if_needed "$data_dir"

    IFS=',' read -r -a seeds <<< "$SEED_LIST"
    for seed in "${seeds[@]}"; do
      seed_prefix="${exp_prefix}_seed${seed}"
      cmd=(
        python "$ROOT_DIR/train_loso_functional.py"
        --config_path "$config_path"
        --base_data_path "$data_dir"
        --epochs "$EPOCHS"
        --lr "$LR"
        --seed "$seed"
        --exp_prefix "$seed_prefix"
        --holdouts "$HOLDOUTS"
        --dataset_name "$DATASET_NAME"
        --output_root "$OUTPUT_ROOT"
      )

      if [[ -n "$HOLDOUT_ONLY" ]]; then
        cmd+=(--holdout_only "$HOLDOUT_ONLY")
      fi

      if [[ "$NO_VALIDATION" == "1" ]]; then
        cmd+=(--no_validation)
      fi

      "${cmd[@]}"

      if [[ "$RUN_PTQ" == "1" ]]; then
        python "$ROOT_DIR/ptq_export_batch.py" \
          --config_path "$config_path" \
          --base_data_path "$data_dir" \
          --exp_prefix "$seed_prefix" \
          --holdouts "$HOLDOUTS" \
          --dataset_name "$DATASET_NAME" \
          --output_root "$OUTPUT_ROOT" \
          --output_base_dir "$OUTPUT_ROOT/$DATASET_NAME/loso/exports/ptq_${seed_prefix}"
      else
        log "PTQ export: disabled (RUN_PTQ=$RUN_PTQ)."
      fi

      if [[ "$RUN_QAT" == "1" ]]; then
        python "$ROOT_DIR/train_loso_functional_qat_batch.py" \
          --config_path "$config_path" \
          --base_data_path "$data_dir" \
          --exp_prefix "$seed_prefix" \
          --holdouts "$HOLDOUTS" \
          --dataset_name "$DATASET_NAME" \
          --output_root "$OUTPUT_ROOT" \
          --output_base_dir "$OUTPUT_ROOT/$DATASET_NAME/loso/exports/qat_${seed_prefix}" \
          --qat_epochs "$QAT_EPOCHS" \
          --batch_size "$QAT_BATCH" \
          --lr "$QAT_LR" \
          --no_validation
      else
        log "QAT export: disabled (RUN_QAT=$RUN_QAT)."
      fi
    done
  else
    log "LOSO training: disabled (RUN_LOSO=$RUN_LOSO)."
  fi

  if [[ "$RUN_FULL_DATASET" == "1" ]]; then
    python "$ROOT_DIR/train_full_dataset.py" \
      --config_path "$config_path" \
      --base_data_path "$data_dir" \
      --epochs "$FULL_EPOCHS" \
      --lr "$FULL_LR" \
      --seed "$FULL_SEED" \
      --exp_name "${exp_prefix}_full" \
      --dataset_name "$DATASET_NAME" \
      --output_root "$OUTPUT_ROOT"
  else
    log "Full dataset training: disabled (RUN_FULL_DATASET=$RUN_FULL_DATASET)."
  fi

  if [[ "$RUN_COLLECT" == "1" && "$RUN_LOSO" == "1" ]]; then
    python "$ROOT_DIR/collect_results.py" \
      --run_evaluation \
      --config_path "$config_path" \
      --base_data_path "$data_dir" \
      --seed_list "$SEED_LIST" \
      --exp_prefix_base "$exp_prefix" \
      --dataset_name "$DATASET_NAME" \
      --run_type "loso" \
      --output_root "$OUTPUT_ROOT" \
      --ptq_base_dir "$OUTPUT_ROOT/$DATASET_NAME/loso/exports/ptq_${exp_prefix}_seed{seed}" \
      --qat_base_dir "$OUTPUT_ROOT/$DATASET_NAME/loso/exports/qat_${exp_prefix}_seed{seed}"
  else
    log "Collect results: disabled (RUN_COLLECT=$RUN_COLLECT)."
  fi
}

main() {
  ensure_conda_env
  if [[ "$SKIP_FLIP" == "1" ]]; then
    log "Flip step: skipped (SKIP_FLIP=$SKIP_FLIP)."
  else
    flip_videos_if_needed
  fi
  for variant in $VARIANTS; do
    run_variant "$variant"
  done

  log "Variant pipeline complete."
}

main "$@"
