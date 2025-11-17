#!/usr/bin/env python
"""
collect_results.py - Collect and report all metrics for the paper
Updated to properly handle --no_validation mode and save results to file
This script generates a comprehensive report with:
1. Network architecture details
2. Training configuration and hyperparameters
3. Model size and number of parameters
4. Recognition accuracy per test signer
5. Per-class accuracy statistics across all users
6. Training and inference times per test signer

Output is saved to a timestamped file and optionally displayed on console.
"""

import os
import re
import json
import yaml
import torch
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from model import SignBart

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

def parse_log_file(log_path):
    """Extract metrics from log file."""
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
        'per_class_acc': {}  # Per-class accuracies
    }
    
    with open(log_path, 'r') as f:
        content = f.read()
        
        # Check if validation was disabled
        if 'Validation: DISABLED' in content:
            results['has_validation'] = False
        
        # Parse line by line
        for line in content.split('\n'):
            # Training metrics
            train_match = re.search(r'\[(\d+)/\d+\] TRAIN\s+loss:\s+([\d.]+)\s+acc:\s+([\d.]+)\s+top5:\s+([\d.]+)', line)
            if train_match:
                results['train_loss'].append(float(train_match.group(2)))
                results['train_acc'].append(float(train_match.group(3)))
                results['train_top5'].append(float(train_match.group(4)))
            
            # Validation metrics
            if results['has_validation']:
                val_match = re.search(r'\[(\d+)/\d+\] VAL\s+loss:\s+([\d.]+)\s+acc:\s+([\d.]+)\s+top5:\s+([\d.]+)', line)
                if val_match:
                    results['val_loss'].append(float(val_match.group(2)))
                    results['val_acc'].append(float(val_match.group(3)))
                    results['val_top5'].append(float(val_match.group(4)))
            
            # Training time
            time_match = re.search(r'Training complete - Total time:\s+([\d.]+)s', line)
            if time_match:
                results['total_training_time'] = float(time_match.group(1))
            
            # Best accuracies
            best_val_match = re.search(r'Best val acc:\s+([\d.]+)', line)
            if best_val_match:
                results['best_val_acc'] = float(best_val_match.group(1))
            
            # Evaluation metrics
            eval_match = re.search(r'Evaluation - Loss:\s+([\d.]+),\s+Acc:\s+([\d.]+),\s+Top-5:\s+([\d.]+),\s+Time:\s+([\d.]+)s', line)
            if eval_match:
                results['test_loss'] = float(eval_match.group(1))
                results['test_acc'] = float(eval_match.group(2))
                results['test_top5'] = float(eval_match.group(3))
                results['inference_time'] = float(eval_match.group(4))
            
            # Per-class accuracy
            class_acc_match = re.search(r'Class\s+(G\d+):\s+Acc\s+=\s+([\d.]+)', line)
            if class_acc_match:
                class_name = class_acc_match.group(1)
                class_acc = float(class_acc_match.group(2))
                results['per_class_acc'][class_name] = class_acc
    
    # Calculate best accuracies
    if results['train_acc']:
        results['best_train_acc'] = max(results['train_acc'])
    if results['val_acc']:
        results['best_val_acc'] = max(results['val_acc'])
    
    # Calculate average epoch time
    if results['total_training_time'] and results['train_acc']:
        results['avg_epoch_time'] = results['total_training_time'] / len(results['train_acc'])
    
    # Check eval log for test set performance
    eval_log_path = log_path.replace('.log', '_eval.log')
    if os.path.exists(eval_log_path):
        with open(eval_log_path, 'r') as f:
            eval_content = f.read()
            
            # Evaluation metrics
            eval_match = re.search(r'Evaluation - Loss:\s+([\d.]+),\s+Acc:\s+([\d.]+),\s+Top-5:\s+([\d.]+),\s+Time:\s+([\d.]+)s', eval_content)
            if eval_match:
                results['test_loss'] = float(eval_match.group(1))
                results['test_acc'] = float(eval_match.group(2))
                results['test_top5'] = float(eval_match.group(3))
                results['inference_time'] = float(eval_match.group(4))
            
            # Per-class accuracy from eval log
            for line in eval_content.split('\n'):
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
        
        # Header
        header = "Test Signer,Train Acc (%)"
        if has_validation:
            header += ",Val Acc (%)"
        if show_test:
            header += ",Test Acc (%)"
        header += ",Train Time (h),Inference (ms)\n"
        f.write(header)
        
        # Data rows
        for user in TEST_USERS:
            results = all_results.get(user)
            if results:
                train_acc = results['best_train_acc'] * 100
                val_acc = results['best_val_acc'] * 100 if has_validation else 0
                test_acc = results['test_acc'] * 100 if results['test_acc'] is not None else 0
                train_time = results['total_training_time'] / 3600 if results['total_training_time'] else 0
                inference_ms = (results['inference_time'] / 100 * 1000) if results['inference_time'] else 0
                
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
            # Header
            header = "Class," + ",".join(TEST_USERS) + ",Average\n"
            f.write(header)
            
            # Data rows
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
                
                # Average for this class
                if class_accs:
                    avg_acc = sum(class_accs) / len(class_accs)
                    row += f",{avg_acc:.2f}"
                else:
                    row += ",N/A"
                
                row += "\n"
                f.write(row)

def generate_report(output_file=None, console=True, save_csv=True):
    """Generate comprehensive report."""
    
    # Determine output filename
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/experiment_results_{timestamp}.txt"
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    with DualWriter(output_file, console=console) as writer:
        writer.writeln("="*80)
        writer.writeln("ARABIC SIGN LANGUAGE RECOGNITION - EXPERIMENTAL RESULTS")
        writer.writeln(f"LOSO Cross-Validation with {len(TEST_USERS)} Test Signers")
        writer.writeln(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        
        config = load_config()
        
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
            log_path = f"{exp_name}.log"
            
            writer.writeln(f"Test Signer: {user.upper()}")
            writer.writeln(f"  Training set: All users except {user}")
            
            results = parse_log_file(log_path)
            if results:
                all_results[user] = results
                
                if not results['has_validation']:
                    has_validation = False
                
                writer.writeln(f"  Training samples: ~1,100")
                writer.writeln(f"  Test samples: ~100")
                writer.writeln(f"  ")
                
                # Training metrics
                if results['train_acc']:
                    writer.writeln(f"  Final Training Accuracy: {results['train_acc'][-1]*100:.2f}%")
                writer.writeln(f"  Best Training Accuracy: {results['best_train_acc']*100:.2f}%")
                
                # Validation metrics
                if results['has_validation']:
                    if results['val_acc']:
                        writer.writeln(f"  Final Validation Accuracy: {results['val_acc'][-1]*100:.2f}%")
                    writer.writeln(f"  Best Validation Accuracy: {results['best_val_acc']*100:.2f}%")
                    if results['val_top5']:
                        writer.writeln(f"  Top-5 Validation Accuracy: {results['val_top5'][-1]*100:.2f}%")
                
                # Test set metrics
                if results['test_acc'] is not None:
                    writer.writeln(f"  Test Set Accuracy: {results['test_acc']*100:.2f}%")
                    if results['test_top5'] is not None:
                        writer.writeln(f"  Test Set Top-5 Accuracy: {results['test_top5']*100:.2f}%")
            else:
                writer.writeln(f"  Status: Log file not found - training may not have completed")
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
                    time_per_sample = results['inference_time'] / 100
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
                inference_ms = (results['inference_time'] / 100 * 1000) if results['inference_time'] else 0
                
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
            
            inference_times = [r['inference_time'] for r in all_results.values() if r['inference_time']]
            avg_inference = (sum(inference_times) / len(inference_times) / 100 * 1000) if inference_times else 0
            
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
            if inference_times:
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
            writer.writeln("  Please run: python tools/MOREUSERStrain_loso.py --epochs 80 --lr 2e-4 --no_validation")
        writer.writeln()
        
        writer.writeln("="*80)
        writer.writeln(f"Report saved to: {output_file}")
        if save_csv:
            writer.writeln(f"CSV files saved to: results/")
        writer.writeln("="*80)
    
    # Save CSV files
    if save_csv and all_results:
        save_csv_summary(all_results)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Collect and report LOSO cross-validation results")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (default: results/experiment_results_TIMESTAMP.txt)")
    parser.add_argument("--no-console", action="store_true",
                        help="Don't print to console (only save to file)")
    parser.add_argument("--no-csv", action="store_true",
                        help="Don't save CSV summary files")
    
    args = parser.parse_args()
    
    output_file = generate_report(
        output_file=args.output,
        console=not args.no_console,
        save_csv=not args.no_csv
    )
    
    if not args.no_console:
        print(f"\n✓ Report saved to: {output_file}")
        if not args.no_csv:
            print(f"✓ CSV files saved to: results/")

if __name__ == "__main__":
    main()