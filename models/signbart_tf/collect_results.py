#!/usr/bin/env python3
"""
collect_results.py - Collect and report all metrics for TensorFlow SignBART models
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
    python collect_results.py
  
  Load models and evaluate (slow but accurate):
    python collect_results.py --run_evaluation
"""

import os
import re
import json
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
import time
from pathlib import Path
from datetime import datetime

from dataset import SignDataset
from model_functional import build_signbart_functional_with_dict_inputs
from layers import Projection, ClassificationHead, PositionalEmbedding
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from attention import SelfAttention, CrossAttention, CausalSelfAttention
from model_functional import ExtractLastValidToken

# LOSO test users - matches train_loso.py configuration
TEST_USERS = ["user01", "user08", "user11"]  # 3 users (user02 removed)
EXPERIMENT_PREFIX = "arabic_asl_LOSO_"
GESTURE_CLASSES = [f"G{i:02d}" for i in range(1, 11)]  # G01-G10


@tf.keras.utils.register_keras_serializable()
class Top5Accuracy(keras.metrics.Metric):
    """Custom Top-5 accuracy metric for loading models."""
    def __init__(self, name="Top5Accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.top5_correct = self.add_weight(name="top5_correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        top5_preds = tf.nn.top_k(y_pred, k=5).indices
        y_true_expanded = tf.expand_dims(tf.cast(y_true, tf.int32), axis=1)
        top5_preds = tf.cast(top5_preds, tf.int32)
        correct = tf.reduce_any(tf.equal(top5_preds, y_true_expanded), axis=1)
        self.top5_correct.assign_add(tf.reduce_sum(tf.cast(correct, tf.float32)))
        self.total.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.top5_correct / self.total

    def reset_state(self):
        self.top5_correct.assign(0.0)
        self.total.assign(0.0)


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
    """Count model parameters by building the model."""
    model = build_signbart_functional_with_dict_inputs(config)
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
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


def get_custom_objects():
    """Get custom objects for loading Keras models."""
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
    print(f"  Device: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
    
    # Find best checkpoint
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
    
    if not checkpoint_files:
        print(f"  ✗ No checkpoint files found in {checkpoint_dir}")
        return None
    
    # Prioritize: best_model > final_model > latest
    best_model = [f for f in checkpoint_files if 'best_model' in f]
    final_model = [f for f in checkpoint_files if 'final_model' in f]
    
    if best_model:
        checkpoint_file = best_model[0]
        checkpoint_type = "best_model"
    elif final_model:
        checkpoint_file = final_model[0]
        checkpoint_type = "final_model"
    else:
        checkpoint_file = checkpoint_files[0]
        checkpoint_type = "first_available"
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    print(f"  Loading checkpoint: {checkpoint_file} ({checkpoint_type})")
    
    # Load checkpoint
    try:
        custom_objects = get_custom_objects()
        model = keras.models.load_model(checkpoint_path, custom_objects=custom_objects)
        print(f"  ✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Load test dataset
    print(f"  Loading test dataset...")
    try:
        test_datasets = SignDataset(data_path, "test", shuffle=False, joint_idxs=joint_idx)
        test_loader_raw = test_datasets.create_tf_dataset(batch_size=1, drop_remainder=False)
        
        # Transform dataset to separate inputs and labels for functional model
        def separate_labels(batch):
            inputs = {
                'keypoints': batch['keypoints'],
                'attention_mask': batch['attention_mask']
            }
            labels = batch['labels']
            return inputs, labels
        
        test_loader = test_loader_raw.map(separate_labels)
        num_samples = len(test_datasets)
        print(f"  ✓ Test dataset loaded: {num_samples} samples")
    except Exception as e:
        print(f"  ✗ Error loading test dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Run evaluation
    print(f"  Running evaluation...")
    start_time = time.time()
    
    try:
        results = model.evaluate(test_loader, return_dict=True, verbose=1)
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
        for batch in test_loader:
            inputs, labels = batch
            logits = model(inputs, training=False)
            
            preds = tf.argmax(logits, axis=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
        
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
    results_dict = {
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
        'time_per_sample': inference_time / num_samples if num_samples > 0 else 0,
        'num_test_samples': num_samples,
        'has_validation': False,
        'test_acc': results.get('accuracy', 0),
        'test_loss': results.get('loss', 0),
        'test_top5': results.get('top5_accuracy', 0),
        'per_class_acc': per_class_acc,
        'evaluated_from_checkpoint': True,
        'checkpoint_used': checkpoint_file
    }
    
    print(f"\n  Results:")
    print(f"    Test Accuracy: {results_dict['test_acc']*100:.2f}%")
    print(f"    Test Top-5 Accuracy: {results_dict['test_top5']*100:.2f}%")
    print(f"    Test Loss: {results_dict['test_loss']:.4f}")
    print(f"    Inference Time: {inference_time:.2f}s")
    print(f"    Time per sample: {results_dict['time_per_sample']*1000:.2f}ms")
    print(f"{'='*80}\n")
    
    return results_dict


def parse_log_file(log_path):
    """Extract metrics from TensorFlow/Keras log file."""
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
        
        # Parse Keras training progress (Epoch X/Y format)
        # Example: "Epoch 1/80" followed by metrics
        epoch_pattern = r'Epoch (\d+)/(\d+)'
        val_pattern = r'val_loss: ([\d.]+).*?val_accuracy: ([\d.]+)'
        train_pattern = r'loss: ([\d.]+).*?accuracy: ([\d.]+).*?top5_accuracy: ([\d.]+)'
        
        lines = content.split('\n')
        current_epoch = None
        
        for i, line in enumerate(lines):
            # Check for epoch start
            epoch_match = re.search(epoch_pattern, line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            # Parse training metrics (from progress bar or summary)
            if 'loss:' in line and 'accuracy:' in line:
                train_match = re.search(r'loss:\s+([\d.]+).*?accuracy:\s+([\d.]+)', line)
                if train_match:
                    loss = float(train_match.group(1))
                    acc = float(train_match.group(2))
                    if len(results['train_loss']) < current_epoch if current_epoch else True:
                        results['train_loss'].append(loss)
                        results['train_acc'].append(acc)
                
                # Try to get top5
                top5_match = re.search(r'top5_accuracy:\s+([\d.]+)', line)
                if top5_match:
                    top5 = float(top5_match.group(1))
                    if len(results['train_top5']) < current_epoch if current_epoch else True:
                        results['train_top5'].append(top5)
            
            # Parse validation metrics
            if 'val_loss:' in line and 'val_accuracy:' in line:
                val_match = re.search(r'val_loss:\s+([\d.]+).*?val_accuracy:\s+([\d.]+)', line)
                if val_match:
                    val_loss = float(val_match.group(1))
                    val_acc = float(val_match.group(2))
                    if len(results['val_loss']) < current_epoch if current_epoch else True:
                        results['val_loss'].append(val_loss)
                        results['val_acc'].append(val_acc)
            
            # Parse evaluation results
            eval_match = re.search(r'Evaluation.*?Loss:\s+([\d.]+).*?Top-1 Accuracy:\s+([\d.]+).*?Top-5 Accuracy:\s+([\d.]+)', content, re.DOTALL)
            if eval_match:
                results['test_loss'] = float(eval_match.group(1))
                results['test_acc'] = float(eval_match.group(2))
                results['test_top5'] = float(eval_match.group(3))
            
            # Parse inference time
            time_match = re.search(r'Inference time:\s+([\d.]+)\s+seconds', line)
            if time_match:
                results['inference_time'] = float(time_match.group(1))
    
    if results['train_acc']:
        results['best_train_acc'] = max(results['train_acc'])
    if results['val_acc']:
        results['best_val_acc'] = max(results['val_acc'])
    
    # Try to get training time from log
    time_match = re.search(r'Training complete.*?Total time:\s+([\d.]+)s', content, re.IGNORECASE)
    if time_match:
        results['total_training_time'] = float(time_match.group(1))
    elif results['train_acc']:
        # Estimate from epochs (rough)
        results['total_training_time'] = len(results['train_acc']) * 60  # Rough estimate
    
    if results['total_training_time'] and results['train_acc']:
        results['avg_epoch_time'] = results['total_training_time'] / len(results['train_acc'])
    
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
        writer.writeln(f"LOSO Cross-Validation with {len(TEST_USERS)} Test Signers (TensorFlow)")
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
        writer.writeln("Framework: TensorFlow/Keras (Functional API - QAT-ready)")
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
        writer.writeln(f"  Number of classes: {config.get('num_labels', 'N/A')}")
        writer.writeln(f"  Input keypoints: {len(config.get('joint_idx', []))}")
        writer.writeln()
        
        writer.writeln("Training Hyperparameters:")
        writer.writeln("  Optimizer: AdamW")
        writer.writeln("  Learning rate: 2e-4")
        writer.writeln("  Batch size: 1")
        writer.writeln("  Number of epochs: 80")
        writer.writeln("  LR Scheduler: ReduceLROnPlateau")
        writer.writeln("  Random seed: 379")
        writer.writeln()
        
        # 3. Model Size and Parameters
        writer.writeln("3. MODEL SIZE AND PARAMETERS")
        writer.writeln("-" * 80)
        writer.writeln()
        
        params = count_model_parameters(config)
        writer.writeln(f"Total parameters: {params['total']:,}")
        writer.writeln(f"Trainable parameters: {params['trainable']:,}")
        writer.writeln(f"Model size: {params['size_mb']:.2f} MB (FP32)")
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
        
        # 5. Per-Class Accuracy Table (if available)
        writer.writeln("5. PER-CLASS ACCURACY (Test Set)")
        writer.writeln("-" * 80)
        writer.writeln()
        
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
        else:
            writer.writeln("No per-class accuracy data available.")
            writer.writeln("Run with --run_evaluation to get per-class accuracies.")
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
    parser = argparse.ArgumentParser(description="Collect and report LOSO cross-validation results (TensorFlow)")
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

