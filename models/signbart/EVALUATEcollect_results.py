#!/usr/bin/env python
"""
EVALUATEcollect_results.py - Collect and report all metrics for the paper
ENHANCED VERSION: Can either parse logs OR load models and run evaluation
This script generates a comprehensive report with:
1. Network architecture details
2. Training configuration and hyperparameters
3. Model size and number of parameters
4. Recognition accuracy per test signer
5. Per-class accuracy statistics across all users
6. Training and inference times per test signer

Usage:
  Parse logs only (fast):
    python EVALUATEcollect_results.py
  
  Load models and evaluate (slow but accurate):
    python EVALUATEcollect_results.py --run_evaluation
"""

import os
import re
import json
import yaml
import torch
import numpy as np
import argparse
import time
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

from model import SignBart
from dataset import Datasets
from utils import evaluate

# LOSO test users - update this list based on your experiments
TEST_USERS = ["user01", "user02", "user08", "user11"]  # For 4 users
EXPERIMENT_PREFIX = "arabic_asl_LOSO_"
GESTURE_CLASSES = [f"G{i:02d}" for i in range(1, 11)]  # G01-G10

class DualWriter:
    """Write to both file and console simultaneously."""
    
    def __init__(self, filename, console=True):
        self.file = open(filename, 'w', encoding='utf-8')
        self.console = console
        self.filename = filename
    
    def write(self, text):
        """Write text to file and optionally console."""
        self.file.write(text)
        if self.console:
            print(text, end='')
    
    def writeln(self, text=''):
        """Write line to file and optionally console."""
        self.write(text + '\n')
    
    def close(self):
        """Close the file."""
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def load_config(config_path="configs/arabic-asl.yaml"):
    """Load model configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def count_model_parameters(config):
    """Count model parameters."""
    model = SignBart(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024**2)  # float32
    return {
        'total': total_params,
        'trainable': trainable_params,
        'size_mb': model_size_mb
    }

def determine_keypoint_groups(config_joint_idx):
    """
    Determine how to group keypoints for normalization.
    Returns a list of lists for normalization groups.
    """
    groups = []
    
    body_kpts = []
    left_hand_kpts = []
    right_hand_kpts = []
    face_kpts = []
    
    for idx in config_joint_idx:
        if idx < 33:
            body_kpts.append(idx)
        elif idx < 54:
            left_hand_kpts.append(idx)
        elif idx < 75:
            right_hand_kpts.append(idx)
        else:  # idx >= 75
            face_kpts.append(idx)
    
    if body_kpts:
        groups.append(body_kpts)
    if left_hand_kpts:
        groups.append(left_hand_kpts)
    if right_hand_kpts:
        groups.append(right_hand_kpts)
    if face_kpts:
        groups.append(face_kpts)
    
    return groups

def evaluate_model_from_checkpoint(config_path, data_path, checkpoint_dir, user):
    """
    Load model from checkpoint and evaluate on test set.
    
    Args:
        config_path: Path to config YAML
        data_path: Path to LOSO data directory (e.g., data/arabic-asl_LOSO_user01)
        checkpoint_dir: Path to checkpoint directory
        user: User ID (e.g., 'user01')
    
    Returns:
        dict: Evaluation results including accuracy, per-class accuracy, timing, etc.
    """
    print(f"\n{'='*80}")
    print(f"Evaluating model for {user.upper()}")
    print(f"  Config: {config_path}")
    print(f"  Data: {data_path}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"{'='*80}\n")
    
    # Check if checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        print(f"  ✗ Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # Check if data directory exists
    if not os.path.exists(data_path):
        print(f"  ✗ Data directory not found: {data_path}")
        return None
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine keypoint groups for normalization
    if 'joint_idx' in config and config['joint_idx']:
        joint_idx = determine_keypoint_groups(config['joint_idx'])
    else:
        print(f"  ⚠️  Warning: No joint_idx in config, using default")
        joint_idx = None
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # Create model
    model = SignBart(config)
    model.to(device)
    model.eval()
    
    # Find best checkpoint
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if not checkpoint_files:
        print(f"  ✗ No checkpoint files found in {checkpoint_dir}")
        return None
    
    # Prioritize: best_val > best_train > latest
    best_val = [f for f in checkpoint_files if 'best_val' in f]
    best_train = [f for f in checkpoint_files if 'best_train' in f]
    latest = [f for f in checkpoint_files if 'latest' in f]
    
    if best_val:
        checkpoint_file = best_val[0]
        checkpoint_type = "best_val"
    elif best_train:
        checkpoint_file = best_train[0]
        checkpoint_type = "best_train"
    elif latest:
        checkpoint_file = latest[0]
        checkpoint_type = "latest"
    else:
        checkpoint_file = checkpoint_files[0]
        checkpoint_type = "first_available"
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    print(f"  Loading checkpoint: {checkpoint_file} ({checkpoint_type})")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"  ✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading checkpoint: {e}")
        return None
    
    # Load test dataset
    print(f"  Loading test dataset...")
    try:
        test_datasets = Datasets(data_path, "test", shuffle=False, joint_idxs=joint_idx)
        test_loader = DataLoader(
            test_datasets, 
            shuffle=False, 
            batch_size=1, 
            collate_fn=test_datasets.data_collator
        )
        print(f"  ✓ Test dataset loaded: {len(test_datasets)} samples")
    except Exception as e:
        print(f"  ✗ Error loading test dataset: {e}")
        return None
    
    # Run evaluation
    print(f"  Running evaluation...")
    start_time = time.time()
    
    try:
        with torch.no_grad():
            test_loss, test_acc, test_top5 = evaluate(
                model, 
                test_loader, 
                epoch=0, 
                epochs=0, 
                log_per_class=True
            )
        
        inference_time = time.time() - start_time
        print(f"  ✓ Evaluation complete in {inference_time:.2f}s")
        
    except Exception as e:
        print(f"  ✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Collect per-class accuracies
    print(f"  Collecting per-class accuracies...")
    
    per_class_acc = {}
    all_preds = []
    all_labels = []
    
    try:
        with torch.no_grad():
            for data in test_loader:
                data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in data.items()}
                labels = data["labels"]
                _, logits = model(**data)
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate per-class accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        for class_idx in sorted(np.unique(all_labels)):
            class_mask = all_labels == class_idx
            class_correct = (all_preds[class_mask] == all_labels[class_mask]).sum()
            class_total = class_mask.sum()
            class_acc = class_correct / class_total if class_total > 0 else 0
            class_name = f"G{class_idx+1:02d}"
            per_class_acc[class_name] = class_acc
        
        print(f"  ✓ Per-class accuracies collected")
        
    except Exception as e:
        print(f"  ⚠️  Warning: Could not collect per-class accuracies: {e}")
    
    # Return results in same format as parse_log_file()
    results = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': [],
        'train_top5': [],
        'val_top5': [],
        'best_train_acc': 0,
        'best_val_acc': 0,
        'total_training_time': None,
        'avg_epoch_time': None,
        'inference_time': inference_time,
        'time_per_sample': inference_time / len(test_datasets) if len(test_datasets) > 0 else 0,
        'num_test_samples': len(test_datasets),
        'has_validation': False,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'test_top5': test_top5,
        'per_class_acc': per_class_acc,
        'evaluated_from_checkpoint': True,
        'checkpoint_used': checkpoint_file
    }
    
    print(f"\n  Results:")
    print(f"    Test Accuracy: {test_acc*100:.2f}%")
    print(f"    Test Top-5 Accuracy: {test_top5*100:.2f}%")
    print(f"    Test Loss: {test_loss:.4f}")
    print(f"    Inference Time: {inference_time:.2f}s")
    print(f"    Time per sample: {results['time_per_sample']*1000:.2f}ms")
    print(f"{'='*80}\n")
    
    return results

def parse_log_file(log_path):
    """Extract metrics from log file (original function - kept as fallback)."""
    if not os.path.exists(log_path):
        return None
    
    results = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': [],
        'train_top5': [],
        'val_top5': [],
        'best_train_acc': 0,
        'best_val_acc': 0,
        'total_training_time': None,
        'avg_epoch_time': None,
        'inference_time': None,
        'time_per_sample': None,
        'has_validation': True,
        'test_acc': None,
        'test_loss': None,
        'test_top5': None,
        'per_class_acc': {},
        'evaluated_from_checkpoint': False
    }
    
    with open(log_path, 'r') as f:
        content = f.read()
        
        if 'Validation: DISABLED' in content:
            results['has_validation'] = False
        
        for line in content.split('\n'):
            train_match = re.search(r'\[(\d+)/\d+\] TRAIN\s+loss:\s+([\d.]+)\s+acc:\s+([\d.]+)\s+top5:\s+([\d.]+)', line)
            if train_match:
                results['train_loss'].append(float(train_match.group(2)))
                results['train_acc'].append(float(train_match.group(3)))
                results['train_top5'].append(float(train_match.group(4)))
            
            if results['has_validation']:
                val_match = re.search(r'\[(\d+)/\d+\] VAL\s+loss:\s+([\d.]+)\s+acc:\s+([\d.]+)\s+top5:\s+([\d.]+)', line)
                if val_match:
                    results['val_loss'].append(float(val_match.group(2)))
                    results['val_acc'].append(float(val_match.group(3)))
                    results['val_top5'].append(float(val_match.group(4)))
            
            time_match = re.search(r'Training complete - Total time:\s+([\d.]+)s', line)
            if time_match:
                results['total_training_time'] = float(time_match.group(1))
            
            best_val_match = re.search(r'Best val acc:\s+([\d.]+)', line)
            if best_val_match:
                results['best_val_acc'] = float(best_val_match.group(1))
            
            eval_match = re.search(r'Evaluation - Loss:\s+([\d.]+),\s+Acc:\s+([\d.]+),\s+Top-5:\s+([\d.]+),\s+Time:\s+([\d.]+)s', line)
            if eval_match:
                results['test_loss'] = float(eval_match.group(1))
                results['test_acc'] = float(eval_match.group(2))
                results['test_top5'] = float(eval_match.group(3))
                results['inference_time'] = float(eval_match.group(4))
            
            class_acc_match = re.search(r'Class\s+(G\d+):\s+Acc\s+=\s+([\d.]+)', line)
            if class_acc_match:
                class_name = class_acc_match.group(1)
                class_acc = float(class_acc_match.group(2))
                results['per_class_acc'][class_name] = class_acc
    
    if results['train_acc']:
        results['best_train_acc'] = max(results['train_acc'])
    if results['val_acc']:
        results['best_val_acc'] = max(results['val_acc'])
    
    if results['total_training_time'] and results['train_acc']:
        results['avg_epoch_time'] = results['total_training_time'] / len(results['train_acc'])
    
    # Check eval log
    eval_log_path = log_path.replace('.log', '_eval.log')
    if os.path.exists(eval_log_path):
        with open(eval_log_path, 'r') as f:
            eval_content = f.read()
            
            # Get LAST occurrence (most recent evaluation)
            eval_matches = list(re.finditer(
                r'Evaluation - Loss:\s+([\d.]+),\s+Acc:\s+([\d.]+),\s+Top-5:\s+([\d.]+),\s+Time:\s+([\d.]+)s',
                eval_content
            ))
            if eval_matches:
                last_match = eval_matches[-1]  # Get last match
                results['test_loss'] = float(last_match.group(1))
                results['test_acc'] = float(last_match.group(2))
                results['test_top5'] = float(last_match.group(3))
                results['inference_time'] = float(last_match.group(4))
            
            # Get per-class accuracy from last evaluation section
            lines = eval_content.split('\n')
            # Find last "Per-class accuracy:" occurrence
            last_per_class_idx = -1
            for i, line in enumerate(lines):
                if 'Per-class accuracy:' in line:
                    last_per_class_idx = i
            
            if last_per_class_idx >= 0:
                # Parse per-class from that section onwards
                results['per_class_acc'] = {}
                for line in lines[last_per_class_idx:]:
                    class_acc_match = re.search(r'Class\s+(G\d+):\s+Acc\s+=\s+([\d.]+)', line)
                    if class_acc_match:
                        class_name = class_acc_match.group(1)
                        class_acc = float(class_acc_match.group(2))
                        results['per_class_acc'][class_name] = class_acc
    
    return results

def calculate_per_class_statistics(all_results):
    """Calculate detailed statistics for each gesture class across all users."""
    class_stats = {}
    
    for gesture in GESTURE_CLASSES:
        accuracies = []
        
        for user in TEST_USERS:
            results = all_results.get(user)
            if results and gesture in results.get('per_class_acc', {}):
                acc = results['per_class_acc'][gesture] * 100
                accuracies.append(acc)
        
        if accuracies:
            class_stats[gesture] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'count': len(accuracies),
                'values': accuracies
            }
    
    return class_stats

def save_csv_summary(all_results, output_dir="results"):
    """Save summary tables as CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary table
    summary_file = os.path.join(output_dir, "summary_table.csv")
    with open(summary_file, 'w') as f:
        has_validation = any(r.get('has_validation', True) for r in all_results.values())
        show_test = any(r.get('test_acc') is not None for r in all_results.values())
        
        header = "Test Signer,Train Acc (%)"
        if has_validation:
            header += ",Val Acc (%)"
        if show_test:
            header += ",Test Acc (%)"
        header += ",Train Time (h),Inference (ms)\n"
        f.write(header)
        
        for user in TEST_USERS:
            results = all_results.get(user)
            if results:
                train_acc = results['best_train_acc'] * 100
                val_acc = results['best_val_acc'] * 100 if has_validation else 0
                test_acc = results['test_acc'] * 100 if results['test_acc'] is not None else 0
                train_time = results['total_training_time'] / 3600 if results['total_training_time'] else 0
                inference_ms = (results.get('time_per_sample')*1000) if results.get('time_per_sample') else ((results['inference_time']/results['num_test_samples']*1000) if (results.get('num_test_samples') and results.get('inference_time')) else 0)
                
                row = f"{user},{train_acc:.2f}"
                if has_validation:
                    row += f",{val_acc:.2f}"
                if show_test:
                    row += f",{test_acc:.2f}"
                row += f",{train_time:.2f},{inference_ms:.2f}\n"
                f.write(row)
    
    # Per-class accuracy table
    has_per_class = any(r.get('per_class_acc') for r in all_results.values())
    if has_per_class:
        perclass_file = os.path.join(output_dir, "per_class_accuracy.csv")
        with open(perclass_file, 'w') as f:
            header = "Class," + ",".join(TEST_USERS) + ",Average\n"
            f.write(header)
            
            for gesture in GESTURE_CLASSES:
                row = f"{gesture}"
                class_accs = []
                
                for user in TEST_USERS:
                    results = all_results.get(user)
                    if results and gesture in results.get('per_class_acc', {}):
                        acc = results['per_class_acc'][gesture] * 100
                        row += f",{acc:.2f}"
                        class_accs.append(acc)
                    else:
                        row += ",N/A"
                
                if class_accs:
                    avg_acc = sum(class_accs) / len(class_accs)
                    row += f",{avg_acc:.2f}"
                else:
                    row += ",N/A"
                
                row += "\n"
                f.write(row)

def merge_training_metrics_from_logs(exp_name, results):
    """Merge training metrics parsed from <exp_name>.log into 'results'."""
    log_path = f"{exp_name}.log"
    parsed = parse_log_file(log_path)
    if not parsed:
        return results
    # Copy epoch-wise curves
    for key in ["train_acc", "val_acc", "train_loss", "val_loss", "train_top5", "val_top5"]:
        if parsed.get(key):
            results[key] = parsed[key]
    # Copy best metrics/timings/flags
    for key in ["best_train_acc", "best_val_acc", "total_training_time", "avg_epoch_time", "has_validation"]:
        if parsed.get(key) is not None:
            results[key] = parsed[key]
    # Preserve freshly computed test metrics unless missing
    for key in ["test_acc", "test_loss", "test_top5", "inference_time", "time_per_sample"]:
        if results.get(key) is None and parsed.get(key) is not None:
            results[key] = parsed[key]
    return results


def save_training_curves(all_results, output_dir="results"):
    """Save epoch-wise training (and validation) curves per user to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    for user, res in all_results.items():
        has_any = any(len(res.get(k, [])) > 0 for k in ["train_loss", "train_acc", "train_top5","val_loss", "val_acc", "val_top5"])
        if not has_any:
            continue
        max_len = max(len(res.get("train_loss", [])),
                      len(res.get("train_acc", [])),
                      len(res.get("train_top5", [])),
                      len(res.get("val_loss", [])),
                      len(res.get("val_acc", [])),
                      len(res.get("val_top5", [])))
        if max_len == 0:
            continue
        fp = os.path.join(output_dir, f"training_curves_{user}.csv")
        with open(fp, "w") as f:
            headers = ["epoch", "train_loss", "train_acc", "train_top5"]
            has_val = res.get("has_validation", True) and any(len(res.get(k, [])) > 0 for k in ["val_loss", "val_acc", "val_top5"])
            if has_val:
                headers += ["val_loss", "val_acc", "val_top5"]
            f.write(",".join(headers) + "\n")
            for e in range(max_len):
                row = [str(e+1)]
                row.append(f"{res.get('train_loss', [None]*max_len)[e]:.6f}" if e < len(res.get('train_loss', [])) else "")
                row.append(f"{res.get('train_acc', [None]*max_len)[e]:.6f}" if e < len(res.get('train_acc', [])) else "")
                row.append(f"{res.get('train_top5', [None]*max_len)[e]:.6f}" if e < len(res.get('train_top5', [])) else "")
                if has_val:
                    row.append(f"{res.get('val_loss', [None]*max_len)[e]:.6f}" if e < len(res.get('val_loss', [])) else "")
                    row.append(f"{res.get('val_acc', [None]*max_len)[e]:.6f}" if e < len(res.get('val_acc', [])) else "")
                    row.append(f"{res.get('val_top5', [None]*max_len)[e]:.6f}" if e < len(res.get('val_top5', [])) else "")
                f.write(",".join(row) + "\n")


def generate_report(output_file=None, console=True, save_csv=True, run_evaluation=False,
                   config_path="configs/arabic-asl.yaml", base_data_path="data/arabic-asl"):
    """
    Generate comprehensive report.
    
    Args:
        output_file: Path to save report
        console: Whether to print to console
        save_csv: Whether to save CSV files
        run_evaluation: If True, load models and evaluate. If False, parse logs.
        config_path: Path to config file (for run_evaluation mode)
        base_data_path: Base path to data (for run_evaluation mode)
    """
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_evaluated" if run_evaluation else "_from_logs"
        output_file = f"results/experiment_results_{timestamp}{mode_suffix}.txt"
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"REPORT GENERATION MODE: {'MODEL EVALUATION' if run_evaluation else 'LOG PARSING'}")
    print(f"{'='*80}\n")
    
    with DualWriter(output_file, console=console) as writer:
        writer.writeln("="*80)
        writer.writeln("ARABIC SIGN LANGUAGE RECOGNITION - EXPERIMENTAL RESULTS")
        writer.writeln(f"LOSO Cross-Validation with {len(TEST_USERS)} Test Signers")
        writer.writeln(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if run_evaluation:
            writer.writeln(f"Mode: DIRECT MODEL EVALUATION (from checkpoints)")
        else:
            writer.writeln(f"Mode: LOG FILE PARSING")
        writer.writeln("="*80)
        writer.writeln()
        
        # 1. Network Architecture
        writer.writeln("1. NETWORK ARCHITECTURE")
        writer.writeln("-" * 80)
        writer.writeln()
        writer.writeln("Model: SignBart (Encoder-Decoder Transformer for Sign Language Recognition)")
        writer.writeln()
        writer.writeln("Architecture:")
        writer.writeln("  Input: Video sequences → MediaPipe Holistic → Hand keypoints")
        writer.writeln("  Preprocessing: Keypoint normalization to [0,1] with local bounding box")
        writer.writeln("  ")
        writer.writeln("  Projection Layer:")
        writer.writeln("    - Separates X and Y coordinates")
        writer.writeln("    - Projects keypoints → d_model dimension")
        writer.writeln("  ")
        writer.writeln("  Encoder (processes X coordinates):")
        writer.writeln("    - Multi-layer transformer encoder")
        writer.writeln("    - Self-attention mechanism")
        writer.writeln("    - Position embeddings (learned)")
        writer.writeln("    - Feed-forward network with GELU activation")
        writer.writeln("  ")
        writer.writeln("  Decoder (processes Y coordinates):")
        writer.writeln("    - Multi-layer transformer decoder")
        writer.writeln("    - Causal self-attention")
        writer.writeln("    - Cross-attention to encoder outputs")
        writer.writeln("    - Position embeddings (learned)")
        writer.writeln("    - Feed-forward network with GELU activation")
        writer.writeln("  ")
        writer.writeln("  Classification Head:")
        writer.writeln("    - Takes last decoder time-step output")
        writer.writeln("    - Dropout for regularization")
        writer.writeln("    - Linear projection to class logits")
        writer.writeln("  ")
        writer.writeln("  Output: 10 gesture classes (G01-G10)")
        writer.writeln()
        writer.writeln("Reference:")
        writer.writeln("  SignBart implementation based on BART architecture")
        writer.writeln("  Original: https://github.com/tinh2044/SignBart")
        writer.writeln("  Adapted for: Arabic Sign Language")
        writer.writeln()
        
        # 2. Configuration and Hyperparameters
        writer.writeln("2. TRAINING CONFIGURATION")
        writer.writeln("-" * 80)
        writer.writeln()
        
        config = load_config(config_path)
        
        writer.writeln("Model Hyperparameters:")
        writer.writeln(f"  Hidden dimension (d_model): {config.get('d_model', 'N/A')}")
        writer.writeln(f"  Encoder layers: {config.get('encoder_layers', 'N/A')}")
        writer.writeln(f"  Decoder layers: {config.get('decoder_layers', 'N/A')}")
        writer.writeln(f"  Encoder attention heads: {config.get('encoder_attention_heads', 'N/A')}")
        writer.writeln(f"  Decoder attention heads: {config.get('decoder_attention_heads', 'N/A')}")
        writer.writeln(f"  Encoder FFN dimension: {config.get('encoder_ffn_dim', 'N/A')}")
        writer.writeln(f"  Decoder FFN dimension: {config.get('decoder_ffn_dim', 'N/A')}")
        writer.writeln(f"  Dropout: {config.get('dropout', 'N/A')}")
        writer.writeln(f"  Activation dropout: {config.get('activation_dropout', 'N/A')}")
        writer.writeln(f"  Attention dropout: {config.get('attention_dropout', 'N/A')}")
        writer.writeln(f"  Classifier dropout: {config.get('classifier_dropout', 'N/A')}")
        writer.writeln(f"  Encoder layerdrop: {config.get('encoder_layerdrop', 'N/A')}")
        writer.writeln(f"  Decoder layerdrop: {config.get('decoder_layerdrop', 'N/A')}")
        writer.writeln(f"  Max position embeddings: {config.get('max_position_embeddings', 'N/A')}")
        writer.writeln(f"  Number of classes: {config.get('num_labels', 'N/A')}")
        writer.writeln(f"  Input keypoints: {len(config.get('joint_idx', []))}")
        writer.writeln()
        
        writer.writeln("Training Hyperparameters:")
        writer.writeln("  Optimizer: AdamW")
        writer.writeln("  Learning rate: 2e-4")
        writer.writeln("  Batch size: 1")
        writer.writeln("  Number of epochs: 80")
        writer.writeln("  LR Scheduler: ReduceLROnPlateau")
        writer.writeln("    - Factor: 0.1")
        writer.writeln("    - Patience: 5 epochs")
        writer.writeln("  Random seed: 379")
        writer.writeln("  Loss function: Cross-Entropy")
        writer.writeln()
        
        writer.writeln("Data Augmentation:")
        writer.writeln("  Applied with 40% probability during training:")
        writer.writeln("  - Random rotation: ±15 degrees")
        writer.writeln("  - Gaussian noise injection: σ ∈ [0.01, 0.2]")
        writer.writeln("  - Temporal clipping: 50-100% of frames")
        writer.writeln("  - Time warping: speed ∈ [1.1, 1.5]")
        writer.writeln("  - Horizontal flipping")
        writer.writeln()
        
        # 3. Model Size and Parameters
        writer.writeln("3. MODEL SIZE AND PARAMETERS")
        writer.writeln("-" * 80)
        writer.writeln()
        
        params = count_model_parameters(config)
        writer.writeln(f"Total parameters: {params['total']:,}")
        writer.writeln(f"Trainable parameters: {params['trainable']:,}")
        writer.writeln(f"Model size: {params['size_mb']:.2f} MB (FP32)")
        writer.writeln(f"Model size: {params['size_mb']/2:.2f} MB (FP16)")
        writer.writeln()
        
        # 4. Results per Test Signer
        writer.writeln("4. RECOGNITION ACCURACY PER TEST SIGNER")
        writer.writeln("-" * 80)
        writer.writeln()
        
        all_results = {}
        has_validation = True
        
        for user in TEST_USERS:
            exp_name = f"{EXPERIMENT_PREFIX}{user}"
            
            writer.writeln(f"Test Signer: {user.upper()}")
            writer.writeln(f"  Training set: All users except {user}")
            
            if run_evaluation:
                # NEW: Load model and evaluate
                loso_data_path = f"{base_data_path}_LOSO_{user}"
                checkpoint_dir = f"checkpoints_{exp_name}"
                
                results = evaluate_model_from_checkpoint(
                    config_path, 
                    loso_data_path, 
                    checkpoint_dir,
                    user
                )
                
                if results:
                    writer.writeln(f"  ✓ Evaluated from checkpoint: {results.get('checkpoint_used', 'N/A')}")
            else:
                # ORIGINAL: Parse log files
                log_path = f"{exp_name}.log"
                results = parse_log_file(log_path)
                
                if results:
                    writer.writeln(f"  ✓ Parsed from log file: {log_path}")
            
            if results:
                results = merge_training_metrics_from_logs(exp_name, results)
                all_results[user] = results
                
                if not results['has_validation']:
                    has_validation = False
                
                writer.writeln(f"  Training samples: ~1,100")
                writer.writeln(f"  Test samples: ~100")
                writer.writeln(f"  ")
                
                # Training metrics
                if results['train_acc']:
                    writer.writeln(f"  Final Training Accuracy: {results['train_acc'][-1]*100:.2f}%")
                if results['train_loss']:
                    writer.writeln(f"  Final Training Loss: {results['train_loss'][-1]:.4f}")
                writer.writeln(f"  Best Training Accuracy: {results['best_train_acc']*100:.2f}%")
                
                # Validation metrics
                if results['has_validation']:
                    if results['val_acc']:
                        writer.writeln(f"  Final Validation Accuracy: {results['val_acc'][-1]*100:.2f}%")
                    if results['val_loss']:
                        writer.writeln(f"  Final Validation Loss: {results['val_loss'][-1]:.4f}")
                    writer.writeln(f"  Best Validation Accuracy: {results['best_val_acc']*100:.2f}%")
                    if results['val_top5']:
                        writer.writeln(f"  Top-5 Validation Accuracy: {results['val_top5'][-1]*100:.2f}%")
                
                # Test set metrics
                if results['test_acc'] is not None:
                    writer.writeln(f"  Test Set Accuracy: {results['test_acc']*100:.2f}%")
                    if results['test_top5'] is not None:
                        writer.writeln(f"  Test Set Top-5 Accuracy: {results['test_top5']*100:.2f}%")
            else:
                writer.writeln(f"  Status: Data not found - training may not have completed")
            writer.writeln()
        
        # Average accuracy
        if all_results:
            if has_validation:
                avg_val_acc = sum(r['best_val_acc'] for r in all_results.values()) / len(all_results)
                writer.writeln(f"Average Validation Accuracy (across all test signers): {avg_val_acc*100:.2f}%")
            
            test_accs = [r['test_acc'] for r in all_results.values() if r['test_acc'] is not None]
            if test_accs:
                avg_test_acc = sum(test_accs) / len(test_accs)
                writer.writeln(f"Average Test Set Accuracy (across all test signers): {avg_test_acc*100:.2f}%")
            writer.writeln()
        
        # 5. Per-Class Accuracy Table
        writer.writeln("5. PER-CLASS ACCURACY (Test Set)")
        writer.writeln("-" * 80)
        writer.writeln()
        
        # Check if we have per-class data
        has_per_class = any(r.get('per_class_acc') for r in all_results.values())
        
        if has_per_class:
            # Create table header
            header = f"{'Class':<10}"
            for user in TEST_USERS:
                header += f" {user:<12}"
            header += f" {'Average':<12}"
            writer.writeln(header)
            writer.writeln("-" * 80)
            
            # Calculate per-class averages
            class_averages = {}
            for gesture in GESTURE_CLASSES:
                row = f"{gesture:<10}"
                class_accs = []
                
                for user in TEST_USERS:
                    results = all_results.get(user)
                    if results and gesture in results.get('per_class_acc', {}):
                        acc = results['per_class_acc'][gesture] * 100
                        row += f" {acc:<12.2f}"
                        class_accs.append(acc)
                    else:
                        row += f" {'N/A':<12}"
                
                # Average for this class
                if class_accs:
                    avg_acc = sum(class_accs) / len(class_accs)
                    class_averages[gesture] = avg_acc
                    row += f" {avg_acc:<12.2f}"
                else:
                    row += f" {'N/A':<12}"
                
                writer.writeln(row)
            
            writer.writeln("-" * 80)
            
            # Overall average
            if class_averages:
                overall_avg = sum(class_averages.values()) / len(class_averages)
                overall_row = f"{'Overall Avg':<10}"
                overall_row += f" {' ':<{12 * len(TEST_USERS)}}"
                overall_row += f" {overall_avg:<12.2f}"
                writer.writeln(overall_row)
            
            writer.writeln()
            
            # 5b. Detailed Per-Class Statistics
            writer.writeln("5b. DETAILED PER-CLASS STATISTICS ACROSS ALL USERS")
            writer.writeln("-" * 80)
            writer.writeln()
            
            class_stats = calculate_per_class_statistics(all_results)
            
            if class_stats:
                # Table header
                writer.writeln(f"{'Class':<10} {'Mean (%)':<12} {'Std Dev':<12} {'Min (%)':<12} {'Max (%)':<12} {'# Users':<10}")
                writer.writeln("-" * 80)
                
                for gesture in GESTURE_CLASSES:
                    if gesture in class_stats:
                        stats = class_stats[gesture]
                        writer.writeln(f"{gesture:<10} {stats['mean']:<12.2f} {stats['std']:<12.2f} "
                              f"{stats['min']:<12.2f} {stats['max']:<12.2f} {stats['count']:<10}")
                
                writer.writeln("-" * 80)
                
                # Overall statistics
                all_means = [stats['mean'] for stats in class_stats.values()]
                if all_means:
                    writer.writeln(f"{'Overall':<10} {np.mean(all_means):<12.2f} {np.std(all_means):<12.2f} "
                          f"{np.min(all_means):<12.2f} {np.max(all_means):<12.2f} {len(TEST_USERS):<10}")
                
                writer.writeln()
                
                # Analysis
                writer.writeln("Per-Class Performance Analysis:")
                writer.writeln()
                
                best_class = max(class_stats.items(), key=lambda x: x[1]['mean'])
                worst_class = min(class_stats.items(), key=lambda x: x[1]['mean'])
                most_consistent = min(class_stats.items(), key=lambda x: x[1]['std'])
                least_consistent = max(class_stats.items(), key=lambda x: x[1]['std'])
                
                writer.writeln(f"  Best performing class:")
                writer.writeln(f"    {best_class[0]}: {best_class[1]['mean']:.2f}% (±{best_class[1]['std']:.2f}%)")
                writer.writeln(f"    Range: {best_class[1]['min']:.2f}% - {best_class[1]['max']:.2f}%")
                writer.writeln()
                
                writer.writeln(f"  Worst performing class:")
                writer.writeln(f"    {worst_class[0]}: {worst_class[1]['mean']:.2f}% (±{worst_class[1]['std']:.2f}%)")
                writer.writeln(f"    Range: {worst_class[1]['min']:.2f}% - {worst_class[1]['max']:.2f}%")
                writer.writeln()
                
                writer.writeln(f"  Most consistent class (lowest std dev):")
                writer.writeln(f"    {most_consistent[0]}: {most_consistent[1]['mean']:.2f}% (±{most_consistent[1]['std']:.2f}%)")
                writer.writeln()
                
                writer.writeln(f"  Least consistent class (highest std dev):")
                writer.writeln(f"    {least_consistent[0]}: {least_consistent[1]['mean']:.2f}% (±{least_consistent[1]['std']:.2f}%)")
                writer.writeln()
                
                # Performance categories
                high_perf = [g for g, s in class_stats.items() if s['mean'] >= 90]
                medium_perf = [g for g, s in class_stats.items() if 70 <= s['mean'] < 90]
                low_perf = [g for g, s in class_stats.items() if s['mean'] < 70]
                
                writer.writeln("Performance Categories:")
                if high_perf:
                    writer.writeln(f"  High (≥90%): {', '.join(high_perf)} ({len(high_perf)} classes)")
                if medium_perf:
                    writer.writeln(f"  Medium (70-89%): {', '.join(medium_perf)} ({len(medium_perf)} classes)")
                if low_perf:
                    writer.writeln(f"  Low (<70%): {', '.join(low_perf)} ({len(low_perf)} classes)")
                writer.writeln()
                
        else:
            writer.writeln("No per-class accuracy data found in evaluation logs.")
            writer.writeln("To get per-class accuracies, ensure evaluation logs contain per-class metrics.")
            writer.writeln()
        
        # 6. Training and Inference Times
        writer.writeln("6. TRAINING AND INFERENCE TIMES")
        writer.writeln("-" * 80)
        writer.writeln()
        
        for user in TEST_USERS:
            exp_name = f"{EXPERIMENT_PREFIX}{user}"
            
            writer.writeln(f"Test Signer: {user.upper()}")
            
            results = all_results.get(user)
            if results:
                if results['total_training_time']:
                    hours = results['total_training_time'] / 3600
                    writer.writeln(f"  Total training time: {hours:.2f} hours ({results['total_training_time']:.1f} seconds)")
                if results['avg_epoch_time']:
                    writer.writeln(f"  Average time per epoch: {results['avg_epoch_time']:.1f} seconds")
                if results['inference_time']:
                    writer.writeln(f"  Total inference time (test set): {results['inference_time']:.2f} seconds")
                    time_per_sample = results.get('time_per_sample')
                    if time_per_sample is None and results.get('num_test_samples'):
                        time_per_sample = results['inference_time'] / results['num_test_samples']
                    if time_per_sample is not None:
                        writer.writeln(f"  Time per sample: {time_per_sample:.4f} seconds ({time_per_sample*1000:.2f} ms)")
            else:
                writer.writeln(f"  Status: Timing data not available")
            writer.writeln()
        
        # 7. Summary Table
        writer.writeln("7. SUMMARY TABLE")
        writer.writeln("-" * 80)
        writer.writeln()
        
        show_val = has_validation
        show_test = any(r.get('test_acc') is not None for r in all_results.values())
        
        header = f"{'Test Signer':<15} {'Train Acc (%)':<15}"
        if show_val:
            header += f" {'Val Acc (%)':<15}"
        if show_test:
            header += f" {'Test Acc (%)':<15}"
        header += f" {'Train Time (h)':<18} {'Inference (ms)':<15}"
        
        writer.writeln(header)
        writer.writeln("-" * 80)
        
        for user in TEST_USERS:
            results = all_results.get(user)
            if results:
                train_acc = results['best_train_acc'] * 100
                val_acc = results['best_val_acc'] * 100 if show_val else 0
                test_acc = results['test_acc'] * 100 if results['test_acc'] is not None else 0
                train_time = results['total_training_time'] / 3600 if results['total_training_time'] else 0
                inference_ms = (results.get('time_per_sample')*1000) if results.get('time_per_sample') else ((results['inference_time']/results['num_test_samples']*1000) if (results.get('num_test_samples') and results.get('inference_time')) else 0)
                
                row = f"{user:<15} {train_acc:<15.2f}"
                if show_val:
                    row += f" {val_acc:<15.2f}"
                if show_test:
                    row += f" {test_acc:<15.2f}"
                row += f" {train_time:<18.2f} {inference_ms:<15.2f}"
                writer.writeln(row)
            else:
                row = f"{user:<15} {'N/A':<15}"
                if show_val:
                    row += f" {'N/A':<15}"
                if show_test:
                    row += f" {'N/A':<15}"
                row += f" {'N/A':<18} {'N/A':<15}"
                writer.writeln(row)
        
        writer.writeln("-" * 80)
        
        # Calculate averages
        if all_results:
            avg_train_acc = sum(r['best_train_acc'] for r in all_results.values()) / len(all_results) * 100
            
            train_times = [r['total_training_time'] for r in all_results.values() if r['total_training_time']]
            avg_train_time = (sum(train_times) / len(train_times) / 3600) if train_times else 0
            
            time_per_samples = [r.get('time_per_sample') or ((r['inference_time']/r['num_test_samples']) if (r.get('inference_time') and r.get('num_test_samples')) else None) for r in all_results.values()]
            valid_tps = [x for x in time_per_samples if x is not None]
            avg_inference = (sum(valid_tps)/len(valid_tps)*1000) if valid_tps else 0
            
            row = f"{'AVERAGE':<15} {avg_train_acc:<15.2f}"
            
            if show_val:
                avg_val_acc = sum(r['best_val_acc'] for r in all_results.values()) / len(all_results) * 100
                row += f" {avg_val_acc:<15.2f}"
            
            if show_test:
                test_accs = [r['test_acc'] for r in all_results.values() if r['test_acc'] is not None]
                avg_test_acc = (sum(test_accs) / len(test_accs) * 100) if test_accs else 0
                row += f" {avg_test_acc:<15.2f}"
            
            row += f" {avg_train_time:<18.2f} {avg_inference:<15.2f}"
            writer.writeln(row)
            writer.writeln("=" * 80)
            writer.writeln()
        
        writer.writeln("="*80)
        writer.writeln("REPORT GENERATION COMPLETE")
        writer.writeln("="*80)
        writer.writeln()
        writer.writeln("Summary:")
        if all_results:
            writer.writeln(f"✓ {len(all_results)} LOSO experiments completed")
            if show_test:
                test_accs = [r['test_acc'] for r in all_results.values() if r['test_acc'] is not None]
                if test_accs:
                    avg_test_acc = sum(test_accs) / len(test_accs) * 100
                    writer.writeln(f"✓ Average test set accuracy: {avg_test_acc:.2f}%")
            writer.writeln(f"✓ Total training time: {avg_train_time * len(all_results):.2f} hours")
            writer.writeln(f"✓ Average inference time: {avg_inference:.2f} ms per sample")
            
            # Per-class summary
            if has_per_class:
                class_stats = calculate_per_class_statistics(all_results)
                if class_stats:
                    all_means = [s['mean'] for s in class_stats.values()]
                    writer.writeln(f"✓ Per-class accuracy range: {min(all_means):.2f}% - {max(all_means):.2f}%")
                    writer.writeln(f"✓ Per-class accuracy mean: {np.mean(all_means):.2f}% (±{np.std(all_means):.2f}%)")
        else:
            writer.writeln("✗ No training results found")
            writer.writeln("  Please run training or ensure checkpoints/logs are available")
        writer.writeln()
        
        writer.writeln("="*80)
        writer.writeln(f"Report saved to: {output_file}")
        if save_csv:
            writer.writeln(f"CSV files saved to: results/")
        writer.writeln("="*80)
    
    # Save CSV files
    if save_csv and all_results:
        save_csv_summary(all_results)
        save_training_curves(all_results)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Collect and report LOSO cross-validation results")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (default: results/experiment_results_TIMESTAMP.txt)")
    parser.add_argument("--no-console", action="store_true",
                        help="Don't print to console (only save to file)")
    parser.add_argument("--no-csv", action="store_true",
                        help="Don't save CSV summary files")
    parser.add_argument("--run_evaluation", action="store_true",
                        help="Load models from checkpoints and run evaluation (slow but accurate)")
    parser.add_argument("--config_path", type=str, default="configs/arabic-asl.yaml",
                        help="Path to config file (for --run_evaluation mode)")
    parser.add_argument("--base_data_path", type=str, default="data/arabic-asl",
                        help="Base path to data (for --run_evaluation mode)")
    
    args = parser.parse_args()
    
    output_file = generate_report(
        output_file=args.output,
        console=not args.no_console,
        save_csv=not args.no_csv,
        run_evaluation=args.run_evaluation,
        config_path=args.config_path,
        base_data_path=args.base_data_path
    )
    
    if not args.no_console:
        print(f"\n✓ Report saved to: {output_file}")
        if not args.no_csv:
            print(f"✓ CSV files saved to: results/")

if __name__ == "__main__":
    main()



