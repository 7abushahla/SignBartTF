#!/usr/bin/env python
"""
train_loso.py - Train models using Leave-One-Subject-Out cross-validation
for Arabic Sign Language dataset.

Updated to ensure test set evaluation runs after training when --no_validation is used.
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path

# Define LOSO configurations
LOSO_CONFIGS = [
    {
        "holdout_user": "user01",
        "train_users": "user02-user12",
        "experiment_name": "arabic_asl_LOSO_user01"
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
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n{'='*80}")
        print(f"WARNING: Training failed for {experiment_name}")
        print(f"{'='*80}\n")
        return False
    
    return True

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
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n{'='*80}")
        print(f"WARNING: Evaluation failed for {experiment_name}")
        print(f"{'='*80}\n")
        return False
    
    return True

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
                        help="Train only specific holdout user (e.g., 'user01')")
    parser.add_argument("--no_validation", action="store_true",
                        help="Disable validation during training (test evaluation will run after)")
    parser.add_argument("--skip_final_eval", action="store_true",
                        help="Skip final evaluation on test set after training")
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config_path):
        print(f"ERROR: Config file not found: {args.config_path}")
        print(f"Please create the config file first.")
        sys.exit(1)
    
    # Check if data directories exist
    missing = check_data_exists(args.base_data_path)
    if missing:
        print(f"ERROR: Missing LOSO data directories:")
        for path in missing:
            print(f"  - {path}")
        print(f"\nPlease run prepare_arabic_asl.py first to create LOSO splits.")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Arabic ASL LOSO Training")
    print(f"{'='*80}")
    print(f"Config: {args.config_path}")
    print(f"Base data path: {args.base_data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")
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
            print(f"Valid options: user01, user08, user11")
            sys.exit(1)
    
    results = []
    
    for i, config in enumerate(configs_to_run, 1):
        holdout = config["holdout_user"]
        exp_name = config["experiment_name"]
        loso_data_path = f"{args.base_data_path}_LOSO_{holdout}"
        
        print(f"\n{'#'*80}")
        print(f"# LOSO Experiment {i}/{len(configs_to_run)}: Test on {holdout}")
        print(f"# Training users: {config['train_users']}")
        print(f"{'#'*80}\n")
        
        success = True
        
        if not args.skip_training:
            # Train model
            success = run_training(
                config_path=args.config_path,
                data_path=loso_data_path,
                experiment_name=exp_name,
                epochs=args.epochs,
                lr=args.lr,
                seed=args.seed,
                pretrained_path=args.pretrained_path,
                no_validation=args.no_validation
            )
        
        # When --no_validation is used, we MUST run evaluation to get test metrics
        # Otherwise, evaluation is optional based on --skip_final_eval
        should_evaluate = success and (args.no_validation or not args.skip_final_eval)
        
        if should_evaluate:
            # Find best checkpoint
            checkpoint_dir = f"checkpoints_{exp_name}"
            if os.path.exists(checkpoint_dir):
                # Evaluate on test set
                eval_success = run_evaluation(
                    config_path=args.config_path,
                    data_path=loso_data_path,
                    experiment_name=exp_name,
                    checkpoint_path=checkpoint_dir
                )
                if not eval_success:
                    success = False
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
        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
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
        print(f"  - Checkpoints: checkpoints_{exp_name}/")
        print(f"  - Plots: out-imgs/{exp_name}_loss.png, out-imgs/{exp_name}_lr.png")
    
    # Remind user about collect_results.py
    print(f"\n{'='*80}")
    print(f"To generate a comprehensive report, run:")
    print(f"  python collect_results.py")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()