#!/usr/bin/env python
"""
train_loso.py - Train models using Leave-One-Subject-Out cross-validation
for Arabic Sign Language dataset (TensorFlow version).

Features:
1. Save training metadata with timestamps
2. Run test set evaluation after training when --no_validation is used
3. Save structured training history for easy retrieval
4. Include model configuration, parameters, and size in metadata
5. Test on single LOSO case first with --holdout_only
"""

import os
import argparse
import subprocess
import sys
import json
import yaml
import glob
import re
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from utils import resolve_output_paths

DEFAULT_HOLDOUTS = "user01,user08,user11"


def check_data_exists(base_data_path, holdouts):
    """Check if LOSO data directories exist."""
    missing = []
    for holdout in holdouts:
        loso_path = f"{base_data_path}_LOSO_{holdout}"
        if not os.path.exists(loso_path):
            missing.append(loso_path)
    return missing


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


def build_loso_configs(holdouts, all_users, dataset_name="arabic_asl"):
    configs = []
    for holdout in holdouts:
        train_users = [u for u in all_users if u != holdout]
        configs.append({
            "holdout_user": holdout,
            "train_users": ",".join(train_users),
            "experiment_name": f"{dataset_name}_LOSO_{holdout}"
        })
    return configs


def load_model_config(config_path):
    """Load model configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Warning: Could not load config file {config_path}: {e}")
        return None


def count_model_parameters(checkpoint_path):
    """
    Count model parameters from a checkpoint file (TensorFlow .h5).
    Returns dict with total, trainable, and non-trainable parameter counts.
    """
    try:
        # Find the best checkpoint
        checkpoint_files = glob.glob(os.path.join(checkpoint_path, "*.h5"))
        
        if not checkpoint_files:
            return None
        
        # Use the most recent checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        
        # Load model to count parameters
        # Note: We need to reconstruct the model first
        # For simplicity, we'll extract from the config
        # Alternatively, we could load the actual model
        
        # Try to get file size as approximation
        file_size_bytes = os.path.getsize(latest_checkpoint)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Estimate parameters (rough approximation: 4 bytes per param for float32)
        estimated_params = file_size_bytes // 4
        
        return {
            "estimated_parameters": estimated_params,
            "estimated_parameters_M": round(estimated_params / 1e6, 2),
            "model_size_mb": round(file_size_mb, 2),
            "checkpoint_file": os.path.basename(latest_checkpoint),
            "note": "Parameters estimated from checkpoint file size"
        }
    except Exception as e:
        print(f"Warning: Could not count model parameters: {e}")
        return None


def extract_model_info_from_log(log_file):
    """
    Extract model information from training log file.
    Looks for lines containing parameter counts or model architecture info.
    """
    model_info = {}
    
    if not os.path.exists(log_file):
        return model_info
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Look for common patterns in logs
                if "parameters" in line.lower() or "params" in line.lower():
                    if "total" in line.lower():
                        model_info['log_params_info'] = line.strip()
                if "model size" in line.lower():
                    model_info['log_size_info'] = line.strip()
    except Exception as e:
        print(f"Warning: Could not parse log file {log_file}: {e}")
    
    return model_info


def save_run_metadata(experiment_name, config, args, start_time, end_time, success,
                      model_config=None, checkpoint_path=None,
                      dataset_name=None, run_type=None, output_root=None):
    """
    Save metadata about this training run for later retrieval.
    This ensures we can always identify the most recent run and its configuration.
    Now includes model configuration, parameters, and size information.
    """
    output_paths = resolve_output_paths(
        experiment_name,
        dataset_name=dataset_name,
        run_type=run_type,
        output_root=output_root,
    )
    metadata_dir = str(output_paths["training_metadata_dir"])
    os.makedirs(metadata_dir, exist_ok=True)
    
    metadata = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name or args.dataset_name,
        "run_type": run_type or "loso",
        "holdout_user": config["holdout_user"],
        "train_users": config["train_users"],
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "config_path": args.config_path,
        "data_path": f"{args.base_data_path}_LOSO_{config['holdout_user']}",
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "seed": args.seed,
        "no_validation": args.no_validation,
        "success": success,
        "log_file": str(Path(output_paths["logs_dir"]) / f"{experiment_name}.log"),
        "eval_log_file": str(Path(output_paths["logs_dir"]) / f"{experiment_name}_eval.log") if success else None,
        "checkpoint_dir": str(output_paths["checkpoints_dir"]),
        "timestamp": end_time.timestamp(),
        "framework": "tensorflow"
    }
    
    # Add model configuration from YAML file
    if model_config:
        metadata["model_config"] = model_config
        
        # Extract key model parameters for easy access
        metadata["d_model"] = model_config.get("d_model", "unknown")
        metadata["encoder_layers"] = model_config.get("encoder_layers", "unknown")
        metadata["decoder_layers"] = model_config.get("decoder_layers", "unknown")
    
    # Add model parameter counts and size
    if success and checkpoint_path and os.path.exists(checkpoint_path):
        param_info = count_model_parameters(checkpoint_path)
        if param_info:
            metadata["model_parameters"] = param_info
            
            # Create a formatted summary
            summary = (f"Model has ~{param_info['estimated_parameters_M']}M parameters "
                      f"({param_info['model_size_mb']} MB)")
            metadata["model_summary"] = summary
            
            print(f"[ℹ] Model info: {summary}")
    
    # Try to extract additional info from log file
    log_info = extract_model_info_from_log(metadata["log_file"])
    if log_info:
        metadata["log_extracted_info"] = log_info
    
    # Save with timestamp in filename for history
    timestamp_str = end_time.strftime("%Y%m%d_%H%M%S")
    metadata_file = os.path.join(metadata_dir, f"{experiment_name}_{timestamp_str}.json")
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Also save as "latest" for easy access
    latest_file = os.path.join(metadata_dir, f"{experiment_name}_latest.json")
    with open(latest_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[✓] Metadata saved: {metadata_file}")
    
    # Print model configuration summary
    if model_config:
        print(f"[ℹ] Model: d_model={metadata.get('d_model')}, "
              f"encoder_layers={metadata.get('encoder_layers')}, "
              f"decoder_layers={metadata.get('decoder_layers')}")
    
    return metadata_file


def run_training(config_path, data_path, experiment_name, epochs, lr, seed,
                 pretrained_path="", no_validation=False, use_functional=False,
                 dataset_name=None, run_type=None, output_root=None):
    """Run training for one LOSO configuration."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(
        script_dir,
        "main_functional.py" if use_functional else "main.py"
    )
    cmd = [
        sys.executable, main_script,
        "--experiment_name", experiment_name,
        "--config_path", config_path,
        "--task", "train",
        "--data_path", data_path,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--seed", str(seed)
    ]

    if dataset_name:
        cmd.extend(["--dataset_name", dataset_name])
    if run_type:
        cmd.extend(["--run_type", run_type])
    if output_root:
        cmd.extend(["--output_root", output_root])
    
    if pretrained_path:
        cmd.extend(["--pretrained_path", pretrained_path])
    
    if no_validation:
        cmd.append("--no_validation")
    
    print(f"\n{'='*80}")
    print(f"Starting training: {experiment_name}")
    if no_validation:
        print(f"Validation: DISABLED (training only, test evaluation will run after)")
    else:
        print(f"Validation: ENABLED (using test set)")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    result = subprocess.run(cmd)
    end_time = datetime.now()
    
    success = result.returncode == 0
    
    if not success:
        print(f"\n{'='*80}")
        print(f"WARNING: Training failed for {experiment_name}")
        print(f"{'='*80}\n")
    
    return success, start_time, end_time


# REMOVED: run_evaluation() function
# Evaluation is now performed automatically at the end of training in main.py
# (Both Keras and TFLite models are evaluated inline)


def main():
    parser = argparse.ArgumentParser(description="Train Arabic ASL models with LOSO cross-validation (TensorFlow)")
    parser.add_argument("--config_path", type=str, default="configs/arabic-asl.yaml",
                        help="Path to model config file")
    parser.add_argument("--base_data_path", type=str, default="data/arabic-asl",
                        help="Base path to processed data (without _LOSO suffix)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="Path to pretrained model (.h5 file, optional)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and only run evaluation")
    parser.add_argument("--holdout_only", type=str, default="",
                        help="Train only specific holdout user (e.g., 'user01', 'user02') - USE THIS TO TEST FIRST!")
    parser.add_argument("--holdouts", type=str, default=DEFAULT_HOLDOUTS,
                        help="Comma-separated holdout users or 'all' to use all users")
    parser.add_argument("--no_validation", action="store_true",
                        help="Disable validation during training (test evaluation will run after)")
    parser.add_argument("--skip_final_eval", action="store_true",
                        help="Skip final evaluation on test set after training")
    parser.add_argument("--exp_prefix", type=str, default="",
                        help="Prefix for experiment names (e.g., 'pose_hands' or 'hands_only')")
    parser.add_argument("--dataset_name", type=str, default="arabic_asl",
                        help="Dataset name prefix for experiments (default: arabic_asl)")
    parser.add_argument("--output_root", type=str, default="",
                        help="Base output directory (default: outputs or SIGNBART_OUTPUT_ROOT)")
    parser.add_argument("--use_functional", action="store_true",
                        help="Use functional API model instead of nested model (for QAT compatibility)")
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config_path):
        print(f"ERROR: Config file not found: {args.config_path}")
        print(f"Please create the config file first.")
        sys.exit(1)
    
    # Load model configuration
    model_config = load_model_config(args.config_path)
    if model_config:
        print(f"[✓] Loaded model configuration from {args.config_path}")
        d_model = model_config.get("d_model", "unknown")
        enc_layers = model_config.get("encoder_layers", "unknown")
        dec_layers = model_config.get("decoder_layers", "unknown")
        print(f"[ℹ] Model: d_model={d_model}, encoder_layers={enc_layers}, decoder_layers={dec_layers}")
    else:
        print(f"[!] Warning: Could not load model configuration")
    
    # Determine holdouts list
    if args.holdouts.strip().lower() == "all":
        all_users = discover_users(args.base_data_path)
        holdouts = all_users
    else:
        holdouts = [u.strip() for u in args.holdouts.split(",") if u.strip()]
        all_users = discover_users(args.base_data_path)

    # Check if data directories exist
    missing = check_data_exists(args.base_data_path, holdouts)
    if missing:
        print(f"ERROR: Missing LOSO data directories:")
        for path in missing:
            print(f"  - {path}")
        print(f"\nPlease run the data preparation script first to create LOSO splits.")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"{args.dataset_name} LOSO Training (TensorFlow) - {len(holdouts)} users")
    print(f"{'='*80}")
    print(f"Config: {args.config_path}")
    print(f"Base data path: {args.base_data_path}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")
    print(f"Model type: {'FUNCTIONAL API (QAT-ready)' if args.use_functional else 'NESTED (standard)'}")
    if args.exp_prefix:
        print(f"Experiment prefix: {args.exp_prefix}")
    if args.pretrained_path:
        print(f"Pretrained model: {args.pretrained_path}")
    
    if args.no_validation:
        print(f"Validation: DISABLED (training only)")
        print(f"Final evaluation: {'DISABLED' if args.skip_final_eval else 'ENABLED (will run on test set after training)'}")
    else:
        print(f"Validation: ENABLED (using test set)")
        print(f"Final evaluation: {'DISABLED' if args.skip_final_eval else 'ENABLED'}")
    
    print(f"{'='*80}\n")
    
    # Filter configurations if specific holdout requested
    configs_to_run = build_loso_configs(holdouts, all_users, dataset_name=args.dataset_name)
    if args.holdout_only:
        configs_to_run = [c for c in configs_to_run if c["holdout_user"] == args.holdout_only]
        if not configs_to_run:
            print(f"ERROR: Invalid holdout user: {args.holdout_only}")
            print(f"Valid options: {', '.join(holdouts)}")
            sys.exit(1)
        print(f"[ℹ] TESTING MODE: Running only {args.holdout_only}")
        print(f"    Once this works, remove --holdout_only to run all 3 LOSO experiments\n")
    
    results = []
    
    for i, config in enumerate(configs_to_run, 1):
        holdout = config["holdout_user"]
        
        # Add optional prefix to experiment name
        if args.exp_prefix:
            exp_name = f"{args.exp_prefix}_{config['experiment_name']}"
        else:
            exp_name = config["experiment_name"]
        
        loso_data_path = f"{args.base_data_path}_LOSO_{holdout}"
        
        print(f"\n{'#'*80}")
        print(f"# LOSO Experiment {i}/{len(configs_to_run)}: Test on {holdout}")
        print(f"# Training users: {config['train_users']}")
        if args.exp_prefix:
            print(f"# Experiment: {exp_name}")
        print(f"{'#'*80}\n")
        
        success = True
        train_start = train_end = None
        output_paths = resolve_output_paths(
            exp_name,
            dataset_name=args.dataset_name,
            run_type="loso",
            output_root=args.output_root or None,
        )
        checkpoint_dir = str(output_paths["checkpoints_dir"])
        
        if not args.skip_training:
            # Train model
            success, train_start, train_end = run_training(
                config_path=args.config_path,
                data_path=loso_data_path,
                experiment_name=exp_name,
                epochs=args.epochs,
                lr=args.lr,
                seed=args.seed,
                pretrained_path=args.pretrained_path,
                no_validation=args.no_validation,
                use_functional=args.use_functional,
                dataset_name=args.dataset_name,
                run_type="loso",
                output_root=args.output_root or None,
            )
            
            # Save metadata after training (includes model config and parameters)
            if train_start and train_end:
                save_run_metadata(
                    exp_name, config, args, train_start, train_end, success,
                    model_config=model_config,
                    checkpoint_path=checkpoint_dir if success else None,
                    dataset_name=args.dataset_name,
                    run_type="loso",
                    output_root=args.output_root or None,
                )
        
        # NOTE: Evaluation is already performed inline at the end of training in main.py
        # (Both Keras and TFLite models are evaluated automatically)
        # No need for a separate evaluation subprocess
        
        results.append({
            "holdout": holdout,
            "experiment": exp_name,
            "success": success
        })
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Training Summary")
    print(f"{'='*80}")
    for result in results:
        status = "[✓] SUCCESS" if result["success"] else "[X] FAILED"
        print(f"{status}: {result['experiment']} (test on {result['holdout']})")
    print(f"{'='*80}\n")
    
    # Check logs and results
    print(f"Results saved to:")
    for result in results:
        exp_name = result["experiment"]
        output_paths = resolve_output_paths(
            exp_name,
            dataset_name=args.dataset_name,
            run_type="loso",
            output_root=args.output_root or None,
        )
        log_dir = output_paths["logs_dir"]
        metadata_dir = output_paths["training_metadata_dir"]
        checkpoints_dir = output_paths["checkpoints_dir"]
        plots_dir = output_paths["out_imgs_dir"]
        print(f"\n{exp_name}:")
        print(f"  - Log: {log_dir}/{exp_name}.log")
        if not args.skip_training:
            print(f"  - Evaluation log: {log_dir}/{exp_name}_eval.log")
        print(f"  - Metadata: {metadata_dir}/{exp_name}_latest.json")
        print(f"  - Checkpoints: {checkpoints_dir}/")
        print(f"  - Plots: {plots_dir}/training_curves.png")
    
    # Remind about next steps
    print(f"\n{'='*80}")
    if args.holdout_only:
        print(f"✓ Single LOSO test completed successfully!")
        print(f"\nNext steps:")
        print(f"  1. Check the results in {results[0]['experiment']}.log")
        print(f"  2. If satisfied, run ALL LOSO experiments:")
        print(f"     python train_loso.py \\")
        print(f"         --config_path {args.config_path} \\")
        print(f"         --base_data_path {args.base_data_path} \\")
        print(f"         --epochs {args.epochs} \\")
        print(f"         --lr {args.lr}")
        print(f"     (Remove --holdout_only flag)")
    else:
        print(f"To collect and analyze all results:")
        print(f"  python collect_results.py  # (if you have this script)")
        print(f"\nTo convert best model to TFLite:")
        print(f"  python convert_to_tflite.py \\")
        print(f"      --config {args.config_path} \\")
        print(f"      --checkpoint outputs/{args.dataset_name}/loso/checkpoints/<exp_name>/checkpoint_*_best_val.h5 \\")
        print(f"      --quantization float16")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

