#!/usr/bin/env python3
"""
ptq_export_batch.py

Batch Post-Training Quantization (PTQ) export for all LOSO models.
Automatically exports dynamic-range INT8 TFLite models for all 3 LOSO splits.

This script:
1. Loops through each LOSO user (user01, user08, user11)
2. Loads the trained model from checkpoints_arabic_asl_LOSO_userXX/final_model.h5
3. Exports dynamic-range INT8 TFLite model (weights INT8, activations FP32)
4. Saves to organized directories

Example:
    python ptq_export_batch.py \
        --config_path configs/arabic-asl-90kpts.yaml \
        --base_data_path ~/signbart_tf/data/arabic-asl-90kpts
"""
import argparse
import os
import sys
import glob
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
import yaml

from model_functional import build_signbart_functional_with_dict_inputs, ExtractLastValidToken
from utils import resolve_checkpoint_dir
from layers import Projection, ClassificationHead, PositionalEmbedding
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from attention import SelfAttention, CrossAttention, CausalSelfAttention

# Default LOSO users - can be overridden via --holdouts or auto-discovery
DEFAULT_HOLDOUTS = "user01,user08,user11"
EXPERIMENT_PREFIX = "arabic_asl_LOSO_"
MAX_SEQ_LEN = 64


@tf.keras.utils.register_keras_serializable()
class Top5Accuracy(keras.metrics.SparseTopKCategoricalAccuracy):
    """Top-5 accuracy metric compatible with saved models (k configurable)."""

    def __init__(self, name="top5_accuracy", **kwargs):
        k = kwargs.pop("k", 5)
        super().__init__(k=k, name=name, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch PTQ export for all LOSO models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Export PTQ for all 3 LOSO models:
  python ptq_export_batch.py \\
      --config_path configs/arabic-asl-90kpts.yaml \\
      --base_data_path ~/signbart_tf/data/arabic-asl-90kpts

  # Test with single user first:
  python ptq_export_batch.py \\
      --config_path configs/arabic-asl-90kpts.yaml \\
      --base_data_path ~/signbart_tf/data/arabic-asl-90kpts \\
      --holdout_only user01
        """
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model config YAML")
    parser.add_argument("--base_data_path", type=str, required=True,
                        help="Base path to processed data (without _LOSO suffix)")
    parser.add_argument("--holdout_only", type=str, default="",
                        help="Export PTQ only for specific user (e.g., 'user01') - USE THIS TO TEST FIRST!")
    parser.add_argument("--holdouts", type=str, default=DEFAULT_HOLDOUTS,
                        help="Comma-separated holdout users or 'all' to use all users")
    parser.add_argument("--output_base_dir", type=str, default="exports/ptq_loso",
                        help="Base directory for PTQ outputs (will create subdirs per user)")
    parser.add_argument("--exp_prefix", type=str, default="",
                        help="Experiment prefix used during training (optional)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def discover_users(base_data_path):
    """Discover user IDs from filenames in base_data_path/all/G??/*.pkl."""
    all_glob = os.path.join(base_data_path, "all", "G??", "*.pkl")
    users = set()
    for p in glob.glob(all_glob):
        bn = os.path.basename(p)
        m = re.match(r"(user\d{2})_G\d{2}_R\d{2}\.pkl$", bn)
        if m:
            users.add(m.group(1))
    return sorted(users)


def build_experiment_prefix(config, exp_prefix=""):
    if exp_prefix:
        return f"{exp_prefix}_arabic_asl_LOSO_"
    joint_idx = config.get("joint_idx") if isinstance(config, dict) else None
    if isinstance(joint_idx, list) and len(joint_idx) > 0:
        return f"arabic_asl_{len(joint_idx)}kpts_LOSO_"
    return "arabic_asl_LOSO_"


def get_custom_objects():
    return {
        "Projection": Projection,
        "ClassificationHead": ClassificationHead,
        "PositionalEmbedding": PositionalEmbedding,
        "Encoder": Encoder,
        "EncoderLayer": EncoderLayer,
        "Decoder": Decoder,
        "DecoderLayer": DecoderLayer,
        "SelfAttention": SelfAttention,
        "CrossAttention": CrossAttention,
        "CausalSelfAttention": CausalSelfAttention,
        "ExtractLastValidToken": ExtractLastValidToken,
        "Top5Accuracy": Top5Accuracy,
    }


def load_trained_model(checkpoint_path):
    """Load trained model from checkpoint."""
    print(f"  [LOAD] Loading trained model from {checkpoint_path}")
    custom_objects = get_custom_objects()
    model = keras.models.load_model(checkpoint_path, custom_objects=custom_objects)
    print("  ✓ Trained model loaded.")
    return model


def export_tflite(model, config, output_path, dynamic_range=False):
    """Export model to TFLite format."""
    num_keypoints = len(config["joint_idx"])

    @tf.function(input_signature=[
        {
            "keypoints": tf.TensorSpec(shape=[1, MAX_SEQ_LEN, num_keypoints, 2], dtype=tf.float32),
            "attention_mask": tf.TensorSpec(shape=[1, MAX_SEQ_LEN], dtype=tf.float32),
        }
    ])
    def serving_fn(inputs):
        return model(inputs, training=False)

    concrete_fn = serving_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])

    if dynamic_range:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("  [INT8] Dynamic-range quantization enabled (weights INT8, activations FP32).")

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()
    from utils import ensure_dir_safe
    ensure_dir_safe(Path(output_path).parent)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"  ✓ Saved TFLite model to {output_path} ({size_mb:.2f} MB)")


def export_ptq_for_user(config, user, checkpoint_path, output_dir):
    """Export PTQ models for a single LOSO user."""
    print(f"\n{'='*80}")
    print(f"Exporting PTQ for {user.upper()}")
    print(f"{'='*80}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output dir: {output_dir}")
    print()
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"  ✗ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # Load model
        trained_model = load_trained_model(checkpoint_path)
        
        # Warm up model (build graph)
        num_keypoints = len(config["joint_idx"])
        dummy = {
            "keypoints": tf.random.normal((1, 10, num_keypoints, 2)),
            "attention_mask": tf.ones((1, 10)),
        }
        _ = trained_model(dummy, training=False)
        
        # Create output directory
        output_path = Path(output_dir)
        from utils import ensure_dir_safe
        ensure_dir_safe(output_path)
        
        # Export dynamic-range INT8 TFLite
        dynamic_path = output_path / "model_dynamic_int8.tflite"
        print("  [INT8] Exporting dynamic-range INT8 TFLite model...")
        export_tflite(trained_model, config, dynamic_path, dynamic_range=True)
        
        print(f"\n  ✓ PTQ export complete for {user}")
        dynamic_size = os.path.getsize(dynamic_path) / (1024**2)
        print(f"    INT8DR : {dynamic_path} ({dynamic_size:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error during PTQ export: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config_path)
    global EXPERIMENT_PREFIX
    EXPERIMENT_PREFIX = build_experiment_prefix(config, exp_prefix=args.exp_prefix)
    
    # Resolve users to process
    if args.holdouts.strip().lower() == "all":
        users_to_run = discover_users(args.base_data_path)
    else:
        users_to_run = [u.strip() for u in args.holdouts.split(",") if u.strip()]

    # If using holdouts=all, skip users without a trained checkpoint.
    skipped_users = []
    if args.holdouts.strip().lower() == "all":
        filtered_users = []
        for user in users_to_run:
            ck_dir = resolve_checkpoint_dir(f"{EXPERIMENT_PREFIX}{user}")
            ckpt = os.path.join(ck_dir, "final_model.h5")
            if os.path.exists(ckpt):
                filtered_users.append(user)
            else:
                skipped_users.append(user)
        users_to_run = filtered_users
        if skipped_users:
            print(f"[WARN] Skipping users without checkpoints: {', '.join(skipped_users)}")
        if not users_to_run:
            print("[ERROR] No users with checkpoints were found for PTQ export.")
            sys.exit(1)

    # Filter users if holdout_only specified
    if args.holdout_only:
        if args.holdout_only not in users_to_run:
            print(f"ERROR: Invalid user '{args.holdout_only}'")
            print(f"Valid options: {', '.join(users_to_run)}")
            sys.exit(1)
        users_to_run = [args.holdout_only]
        print(f"[INFO] TESTING MODE: Exporting PTQ only for {args.holdout_only}")
        print(f"       Once this works, remove --holdout_only to run all 3 LOSO models\n")
    
    print("="*80)
    print("BATCH PTQ EXPORT FOR LOSO MODELS")
    print("="*80)
    print(f"Config: {args.config_path}")
    print(f"Base data path: {args.base_data_path}")
    print(f"Output base dir: {args.output_base_dir}")
    print(f"Users to process: {len(users_to_run)} ({', '.join(users_to_run)})")
    print("="*80)
    print()
    
    results = []
    for i, user in enumerate(users_to_run, 1):
        print(f"\n{'#'*80}")
        print(f"# Processing {i}/{len(users_to_run)}: {user.upper()}")
        print(f"{'#'*80}")
        
        # Set up paths (resolve actual checkpoint directory)
        ck_dir = resolve_checkpoint_dir(f"{EXPERIMENT_PREFIX}{user}")
        checkpoint_path = os.path.join(ck_dir, "final_model.h5")
        output_dir = f"{args.output_base_dir}/{user}"
        
        # Export PTQ
        success = export_ptq_for_user(
            config=config,
            user=user,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir
        )
        
        results.append({
            "user": user,
            "success": success
        })
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH PTQ EXPORT SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"  ✓ {r['user']}")
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"  ✗ {r['user']}")
    if skipped_users:
        print(f"\nSkipped (no checkpoint): {len(skipped_users)}")
        for user in skipped_users:
            print(f"  - {user}")
    
    print("\n" + "="*80)
    print("Output directories:")
    for user in users_to_run:
        output_dir = f"{args.output_base_dir}/{user}"
        print(f"  {user}: {output_dir}/")
        if os.path.exists(f"{output_dir}/model_dynamic_int8.tflite"):
            size_mb = os.path.getsize(f"{output_dir}/model_dynamic_int8.tflite") / (1024**2)
            print(f"    - model_dynamic_int8.tflite ({size_mb:.2f} MB)")
    
    print("="*80)
    
    if len(successful) == len(results):
        print("\n✓ All PTQ exports completed successfully!")
        if args.holdout_only:
            print("\nNext steps:")
            print(f"  1. Check results in {args.output_base_dir}/{args.holdout_only}/")
            print(f"  2. If satisfied, run for all users:")
            print(f"     python ptq_export_batch.py \\")
            print(f"         --config_path {args.config_path} \\")
            print(f"         --base_data_path {args.base_data_path}")
            print(f"     (Remove --holdout_only flag)")
    else:
        print(f"\n⚠️  Some PTQ exports failed. Check logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

