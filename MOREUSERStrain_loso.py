#!/usr/bin/env python
"""
train_loso.py - Train models using Leave-One-Subject-Out cross-validation
for Arabic Sign Language dataset.

Updated to:
1. Save training metadata with timestamps
2. Ensure test set evaluation runs after training when --no_validation is used
3. Save structured training history for easy retrieval
4. Include model configuration, parameters, and size in metadata
"""

import os
import argparse
import subprocess
import sys
import json
import yaml
import glob
from pathlib import Path
from datetime import datetime

# Define LOSO configurations - includes 4 users
LOSO_CONFIGS = [
    {
        "holdout_user": "user01",
        "train_users": "user02-user12",
        "experiment_name": "arabic_asl_LOSO_user01"
    },
    {
        "holdout_user": "user02",
        "train_users": "user01,user03-12",
        "experiment_name": "arabic_asl_LOSO_user02"
    },
    {
        "holdout_user": "user08",
        "train_users": "user01-07,user09-12",
        "experiment_name": "arabic_asl_LOSO_user08"
    },
    {
        "holdout_user": "user11",
        "train_users": "user01-10,user12",
        "experiment_name": "arabic_asl_LOSO_user11"
    }
]

def check_data_exists(base_data_path):
    """Check if LOSO data directories exist."""
    missing = []
    for config in LOSO_CONFIGS:
        holdout = config["holdout_user"]
        loso_path = f"{base_data_path}_LOSO_{holdout}"
        if not os.path.exists(loso_path):
            missing.append(loso_path)
    return missing

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
    Count model parameters from a checkpoint file.
    Returns dict with total, trainable, and non-trainable parameter counts.
    """
    try:
        import torch
        
        # Find the best checkpoint
        checkpoint_files = glob.glob(os.path.join(checkpoint_path, "*.pth"))
        if not checkpoint_files:
            checkpoint_files = glob.glob(os.path.join(checkpoint_path, "*.pt"))
        
        if not checkpoint_files:
            return None
        
        # Use the most recent checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        
        # Load checkpoint
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        # Extract state dict (handle different checkpoint formats)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Count parameters
        total_params = 0
        for param_name, param_tensor in state_dict.items():
            param_count = param_tensor.numel()
            total_params += param_count
        
        # Calculate size in MB
        total_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32 (4 bytes)
        
        return {
            "total_parameters": total_params,
            "total_parameters_M": round(total_params / 1e6, 2),
            "model_size_mb": round(total_size_mb, 2),
            "checkpoint_file": os.path.basename(latest_checkpoint)
        }
    except ImportError:
        print("Warning: PyTorch not available, cannot count parameters")
        return None
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
                      model_config=None, checkpoint_path=None):
    """
    Save metadata about this training run for later retrieval.
    This ensures we can always identify the most recent run and its configuration.
    Now includes model configuration, parameters, and size information.
    """
    metadata_dir = "training_metadata"
    os.makedirs(metadata_dir, exist_ok=True)
    
    metadata = {
        "experiment_name": experiment_name,
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
        "log_file": f"{experiment_name}.log",
        "eval_log_file": f"{experiment_name}_eval.log" if success else None,
        "checkpoint_dir": f"checkpoints_{experiment_name}",
        "timestamp": end_time.timestamp()
    }
    
    # Add model configuration from YAML file
    if model_config:
        metadata["model_config"] = model_config
        
        # Extract key model parameters for easy access
        if "model" in model_config:
            model_section = model_config["model"]
            metadata["model_type"] = model_section.get("type", "unknown")
            metadata["model_args"] = model_section.get("args", {})
    
    # Add model parameter counts and size
    if success and checkpoint_path and os.path.exists(checkpoint_path):
        param_info = count_model_parameters(checkpoint_path)
        if param_info:
            metadata["model_parameters"] = param_info
            
            # Create a formatted summary
            summary = (f"Model has {param_info['total_parameters_M']}M parameters "
                      f"({param_info['model_size_mb']} MB)")
            metadata["model_summary"] = summary
            
            print(f"[ℹ] Model info: {summary}")
    
    # Try to extract additional info from log file
    log_file = f"{experiment_name}.log"
    log_info = extract_model_info_from_log(log_file)
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
    if model_config and "model" in model_config:
        print(f"[ℹ] Model type: {metadata.get('model_type', 'unknown')}")
    
    return metadata_file

def run_training(config_path, data_path, experiment_name, epochs, lr, seed, 
                 pretrained_path="", no_validation=False):
    """Run training for one LOSO configuration."""
    cmd = [
        sys.executable, "main.py",
        "--experiment_name", experiment_name,
        "--config_path", config_path,
        "--task", "train",
        "--data_path", data_path,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--seed", str(seed)
    ]
    
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

def run_evaluation(config_path, data_path, experiment_name, checkpoint_path):
    """Run evaluation on test set."""
    cmd = [
        sys.executable, "main.py",
        "--experiment_name", f"{experiment_name}_eval",
        "--config_path", config_path,
        "--task", "eval",
        "--data_path", data_path,
        "--resume_checkpoints", checkpoint_path
    ]
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    result = subprocess.run(cmd)
    end_time = datetime.now()
    
    success = result.returncode == 0
    
    if not success:
        print(f"\n{'='*80}")
        print(f"WARNING: Evaluation failed for {experiment_name}")
        print(f"{'='*80}\n")
    
    return success, start_time, end_time

def main():
    parser = argparse.ArgumentParser(description="Train Arabic ASL models with LOSO cross-validation")
    parser.add_argument("--config_path", type=str, default="configs/arabic-asl.yaml",
                        help="Path to model config file")
    parser.add_argument("--base_data_path", type=str, default="data/arabic-asl",
                        help="Base path to processed data (without _LOSO suffix)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=379,
                        help="Random seed")
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="Path to pretrained model (optional)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and only run evaluation")
    parser.add_argument("--holdout_only", type=str, default="",
                        help="Train only specific holdout user (e.g., 'user01', 'user02')")
    parser.add_argument("--no_validation", action="store_true",
                        help="Disable validation during training (test evaluation will run after)")
    parser.add_argument("--skip_final_eval", action="store_true",
                        help="Skip final evaluation on test set after training")
    parser.add_argument("--exp_prefix", type=str, default="",
                        help="Prefix for experiment names (e.g., 'pose_hands' or 'hands_only')")
    
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
        if "model" in model_config:
            model_type = model_config["model"].get("type", "unknown")
            print(f"[ℹ] Model type: {model_type}")
    else:
        print(f"[!] Warning: Could not load model configuration")
    
    # Check if data directories exist
    missing = check_data_exists(args.base_data_path)
    if missing:
        print(f"ERROR: Missing LOSO data directories:")
        for path in missing:
            print(f"  - {path}")
        print(f"\nPlease run MOREUSERSprepare_arabic_asl.py first to create LOSO splits.")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Arabic ASL LOSO Training (4 users)")
    print(f"{'='*80}")
    print(f"Config: {args.config_path}")
    print(f"Base data path: {args.base_data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")
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
    configs_to_run = LOSO_CONFIGS
    if args.holdout_only:
        configs_to_run = [c for c in LOSO_CONFIGS if c["holdout_user"] == args.holdout_only]
        if not configs_to_run:
            print(f"ERROR: Invalid holdout user: {args.holdout_only}")
            print(f"Valid options: user01, user02, user08, user11")
            sys.exit(1)
    
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
        checkpoint_dir = f"checkpoints_{exp_name}"
        
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
                no_validation=args.no_validation
            )
            
            # Save metadata after training (includes model config and parameters)
            if train_start and train_end:
                save_run_metadata(
                    exp_name, config, args, train_start, train_end, success,
                    model_config=model_config,
                    checkpoint_path=checkpoint_dir if success else None
                )
        
        # CRITICAL FIX: When --no_validation is used, we MUST run evaluation to get test metrics
        # Otherwise, evaluation is optional based on --skip_final_eval
        should_evaluate = success and (args.no_validation or not args.skip_final_eval)
        
        if should_evaluate:
            # Find best checkpoint
            if os.path.exists(checkpoint_dir):
                # Evaluate on test set
                eval_success, eval_start, eval_end = run_evaluation(
                    config_path=args.config_path,
                    data_path=loso_data_path,
                    experiment_name=exp_name,
                    checkpoint_path=checkpoint_dir
                )
                if not eval_success:
                    success = False
                
                # Update metadata with evaluation info
                if eval_start and eval_end and (train_start and train_end):
                    metadata_file = f"training_metadata/{exp_name}_latest.json"
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        metadata['eval_start_time'] = eval_start.isoformat()
                        metadata['eval_end_time'] = eval_end.isoformat()
                        metadata['eval_duration_seconds'] = (eval_end - eval_start).total_seconds()
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
            else:
                print(f"WARNING: Checkpoint directory not found: {checkpoint_dir}")
                success = False
        elif args.no_validation and not success:
            print(f"WARNING: Training failed, cannot evaluate test set")
        
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
        print(f"\n{exp_name}:")
        print(f"  - Log: {exp_name}.log")
        if not args.skip_training:
            print(f"  - Evaluation log: {exp_name}_eval.log")
        print(f"  - Metadata: training_metadata/{exp_name}_latest.json")
        print(f"  - Checkpoints: checkpoints_{exp_name}/")
        print(f"  - Plots: out-imgs/{exp_name}_loss.png, out-imgs/{exp_name}_lr.png")
    
    # Remind user about collect_results.py
    print(f"\n{'='*80}")
    print(f"To generate a comprehensive report, run:")
    if args.exp_prefix:
        print(f"  python EVALUATEcollect_results.py")
        print(f"  (Note: Update TEST_USERS and EXPERIMENT_PREFIX in collect_results.py)")
    else:
        print(f"  python EVALUATEcollect_results.py")
    print(f"\n  For direct model evaluation (slower but most accurate):")
    print(f"  python EVALUATEcollect_results.py --run_evaluation")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()