#!/usr/bin/env python3
"""
train_loso_functional_qat_batch.py

Batch QAT fine-tuning for all LOSO models.
Automatically runs QAT for all 3 LOSO splits (user01, user08, user11).

This script:
1. Loops through each LOSO user
2. Loads the trained model from checkpoints_arabic_asl_LOSO_userXX/final_model.h5
3. Applies QAT annotation and fine-tunes for N epochs
4. Exports dynamic-range INT8 TFLite model
5. Optionally evaluates the TFLite model

Example:
    python train_loso_functional_qat_batch.py \
        --config_path configs/arabic-asl-90kpts.yaml \
        --base_data_path ~/signbart_tf/data/arabic-asl-90kpts \
        --qat_epochs 3 \
        --batch_size 1 \
        --lr 5e-5 \
        --seed 42
"""
import argparse
import os
import sys
import subprocess
import glob
import re
import yaml
from pathlib import Path
from datetime import datetime
from utils import resolve_output_base

# Default LOSO users - can be overridden via --holdouts or auto-discovery
DEFAULT_HOLDOUTS = "user01,user08,user11"
EXPERIMENT_PREFIX = "arabic_asl_LOSO_"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch QAT fine-tuning for all LOSO models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Run QAT for all 3 LOSO models:
  python train_loso_functional_qat_batch.py \\
      --config_path configs/arabic-asl-90kpts.yaml \\
      --base_data_path ~/signbart_tf/data/arabic-asl-90kpts \\
      --qat_epochs 3 \\
      --batch_size 1 \\
      --lr 5e-5

  # Test with single user first:
  python train_loso_functional_qat_batch.py \\
      --config_path configs/arabic-asl-90kpts.yaml \\
      --base_data_path ~/signbart_tf/data/arabic-asl-90kpts \\
      --holdout_only user01 \\
      --qat_epochs 1
        """
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model config YAML")
    parser.add_argument("--base_data_path", type=str, required=True,
                        help="Base path to processed data (without _LOSO suffix)")
    parser.add_argument("--qat_epochs", type=int, default=20,
                        help="Number of QAT fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for QAT training")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for QAT fine-tuning")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--holdout_only", type=str, default="",
                        help="Run QAT only for specific user (e.g., 'user01') - USE THIS TO TEST FIRST!")
    parser.add_argument("--holdouts", type=str, default=DEFAULT_HOLDOUTS,
                        help="Comma-separated holdout users or 'all' to use all users")
    parser.add_argument("--output_base_dir", type=str, default="",
                        help="Base directory for QAT outputs (default: outputs/<dataset>/loso/exports/qat)")
    parser.add_argument("--exp_prefix", type=str, default="",
                        help="Experiment prefix used during training (optional)")
    parser.add_argument("--dataset_name", type=str, default="arabic_asl",
                        help="Dataset name prefix for experiments (default: arabic_asl)")
    parser.add_argument("--output_root", type=str, default="",
                        help="Base output directory (default: outputs or SIGNBART_OUTPUT_ROOT)")
    parser.add_argument("--quantize_dense_names", nargs="*", default=None,
                        help="Dense layer name substrings to quantize (default: fc1,fc2,proj,q_proj,k_proj,v_proj,out_proj)")
    parser.add_argument("--skip_tflite_eval", action="store_true",
                        help="Skip TFLite model evaluation after export")
    parser.add_argument("--no_validation", action="store_true",
                        help="Disable validation during QAT training (monitor training loss for scheduler)")
    parser.add_argument("--scheduler_factor", type=float, default=0.1,
                        help="Factor for ReduceLROnPlateau scheduler (default: 0.1)")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for ReduceLROnPlateau scheduler (default: 5)")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="Patience for EarlyStopping (default: 10, should be > scheduler_patience)")
    return parser.parse_args()


def build_experiment_prefix(config, exp_prefix="", dataset_name="arabic_asl"):
    if exp_prefix:
        return f"{exp_prefix}_{dataset_name}_LOSO_"
    return f"{dataset_name}_LOSO_"


def discover_users(base_data_path):
    """Discover user IDs from filenames in base_data_path/all/G*/*.pkl."""
    all_glob = os.path.join(base_data_path, "all", "G*", "*.pkl")
    users = set()
    for p in glob.glob(all_glob):
        bn = os.path.basename(p)
        user = bn.split("_", 1)[0]
        if re.match(r"^user\d+$", user):
            users.add(user)
    return sorted(users)


def check_prerequisites(base_data_path, user, dataset_name=None, output_root=None):
    """Check if required files exist for a LOSO user."""
    data_path = f"{base_data_path}_LOSO_{user}"
    from utils import resolve_checkpoint_dir
    ck_dir = resolve_checkpoint_dir(
        f"{EXPERIMENT_PREFIX}{user}",
        dataset_name=dataset_name,
        run_type="loso",
        output_root=output_root,
    )
    checkpoint_path = f"{ck_dir}/final_model.h5"
    
    missing = []
    if not os.path.exists(data_path):
        missing.append(f"Data directory: {data_path}")
    if not os.path.exists(checkpoint_path):
        missing.append(f"Checkpoint: {checkpoint_path}")
    
    return missing


def run_qat_for_user(args, user):
    """Run QAT fine-tuning for a single LOSO user."""
    print(f"\n{'#'*80}")
    print(f"# QAT Fine-tuning for {user.upper()}")
    print(f"{'#'*80}\n")
    
    # Check prerequisites
    missing = check_prerequisites(
        args.base_data_path,
        user,
        dataset_name=args.dataset_name,
        output_root=args.output_root or None,
    )
    if missing:
        print(f"✗ Missing prerequisites for {user}:")
        for item in missing:
            print(f"  - {item}")
        print(f"  Skipping {user}...\n")
        return False
    
    # Set up paths
    data_path = f"{args.base_data_path}_LOSO_{user}"
    from utils import resolve_checkpoint_dir
    ck_dir = resolve_checkpoint_dir(
        f"{EXPERIMENT_PREFIX}{user}",
        dataset_name=args.dataset_name,
        run_type="loso",
        output_root=args.output_root or None,
    )
    checkpoint_path = f"{ck_dir}/final_model.h5"
    output_dir = f"{args.output_base_dir}/{user}"
    
    print(f"[INFO] User: {user}")
    print(f"  Data path: {data_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output dir: {output_dir}")
    print()
    
    # Build command to run train_loso_functional_qat.py (do not depend on CWD)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    qat_script = os.path.join(script_dir, "train_loso_functional_qat.py")
    cmd = [
        sys.executable,
        qat_script,
        "--config_path", args.config_path,
        "--data_path", data_path,
        "--checkpoint", checkpoint_path,
        "--output_dir", output_dir,
        "--batch_size", str(args.batch_size),
        "--qat_epochs", str(args.qat_epochs),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--scheduler_factor", str(args.scheduler_factor),
        "--scheduler_patience", str(args.scheduler_patience),
        "--early_stop_patience", str(args.early_stop_patience),
    ]
    
    if args.quantize_dense_names:
        cmd.extend(["--quantize_dense_names"] + args.quantize_dense_names)
    
    if args.no_validation:
        cmd.append("--no_validation")
    
    print(f"[RUN] Executing QAT for {user}...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = datetime.now()
    result = subprocess.run(cmd)
    end_time = datetime.now()
    
    success = result.returncode == 0
    duration = (end_time - start_time).total_seconds()
    
    if success:
        print(f"\n✓ QAT completed for {user} in {duration:.1f} seconds")
        
        # Check if outputs were created
        qat_model_path = f"{output_dir}/qat_model.keras"
        tflite_path = f"{output_dir}/qat_dynamic_int8.tflite"
        
        outputs_ok = True
        if not os.path.exists(qat_model_path):
            print(f"  ⚠️  Warning: QAT model not found at {qat_model_path}")
            outputs_ok = False
        if not os.path.exists(tflite_path):
            print(f"  ⚠️  Warning: TFLite model not found at {tflite_path}")
            outputs_ok = False
        
        if outputs_ok:
            tflite_size_mb = os.path.getsize(tflite_path) / (1024**2)
            print(f"  ✓ QAT model: {qat_model_path}")
            print(f"  ✓ TFLite model: {tflite_path} ({tflite_size_mb:.2f} MB)")
    else:
        print(f"\n✗ QAT failed for {user} after {duration:.1f} seconds")
    
    print()
    return success


def main():
    args = parse_args()

    try:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}

    global EXPERIMENT_PREFIX
    EXPERIMENT_PREFIX = build_experiment_prefix(config, exp_prefix=args.exp_prefix, dataset_name=args.dataset_name)

    if not args.output_base_dir:
        output_base = resolve_output_base(
            dataset_name=args.dataset_name,
            run_type="loso",
            output_root=args.output_root or None,
        )
        if output_base:
            args.output_base_dir = str(output_base / "exports" / "qat")
        else:
            args.output_base_dir = "exports/qat_loso"
    
    # Resolve users to process
    if args.holdouts.strip().lower() == "all":
        users_to_run = discover_users(args.base_data_path)
    else:
        users_to_run = [u.strip() for u in args.holdouts.split(",") if u.strip()]

    # Filter users if holdout_only specified
    if args.holdout_only:
        if args.holdout_only not in users_to_run:
            print(f"ERROR: Invalid user '{args.holdout_only}'")
            print(f"Valid options: {', '.join(users_to_run)}")
            sys.exit(1)
        users_to_run = [args.holdout_only]
        print(f"[INFO] TESTING MODE: Running QAT only for {args.holdout_only}")
        print(f"       Once this works, remove --holdout_only to run all 3 LOSO models\n")
    
    print("="*80)
    print("BATCH QAT FINE-TUNING FOR LOSO MODELS")
    print("="*80)
    print(f"Config: {args.config_path}")
    print(f"Base data path: {args.base_data_path}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"QAT epochs: {args.qat_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")
    print(f"Output base dir: {args.output_base_dir}")
    print(f"Validation: {'DISABLED' if args.no_validation else 'ENABLED'}")
    print(f"Scheduler: ReduceLROnPlateau (factor={args.scheduler_factor}, patience={args.scheduler_patience})")
    print(f"  Monitor: {'loss' if args.no_validation else 'val_loss'}")
    print(f"Users to process: {len(users_to_run)} ({', '.join(users_to_run)})")
    print("="*80)
    print()
    
    # Check if train_loso_functional_qat.py exists (do not depend on CWD)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    qat_script = os.path.join(script_dir, "train_loso_functional_qat.py")
    if not os.path.exists(qat_script):
        print(f"ERROR: train_loso_functional_qat.py not found: {qat_script}")
        sys.exit(1)
    
    results = []
    for i, user in enumerate(users_to_run, 1):
        print(f"\n{'='*80}")
        print(f"Processing {i}/{len(users_to_run)}: {user.upper()}")
        print(f"{'='*80}")
        
        success = run_qat_for_user(args, user)
        results.append({
            "user": user,
            "success": success
        })
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH QAT SUMMARY")
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
    
    print("\n" + "="*80)
    print("Output directories:")
    for user in users_to_run:
        output_dir = f"{args.output_base_dir}/{user}"
        print(f"  {user}: {output_dir}/")
        if os.path.exists(f"{output_dir}/qat_dynamic_int8.tflite"):
            size_mb = os.path.getsize(f"{output_dir}/qat_dynamic_int8.tflite") / (1024**2)
            print(f"    - qat_dynamic_int8.tflite ({size_mb:.2f} MB)")
        if os.path.exists(f"{output_dir}/qat_model.keras"):
            print(f"    - qat_model.keras")
    
    print("="*80)
    
    if len(successful) == len(results):
        print("\n✓ All QAT fine-tuning completed successfully!")
        if args.holdout_only:
            print("\nNext steps:")
            print(f"  1. Check results in {args.output_base_dir}/{args.holdout_only}/")
            print(f"  2. If satisfied, run for all users:")
            print(f"     python train_loso_functional_qat_batch.py \\")
            print(f"         --config_path {args.config_path} \\")
            print(f"         --base_data_path {args.base_data_path} \\")
            print(f"         --qat_epochs {args.qat_epochs}")
            print(f"     (Remove --holdout_only flag)")
    else:
        print(f"\n⚠️  Some QAT runs failed. Check logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

