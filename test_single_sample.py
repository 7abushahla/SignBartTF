#!/usr/bin/env python
"""
Test TFLite inference on full test set and a single pickle file.

This script:
1. Evaluates TFLite model on the full test set (like after training)
2. Tests on a specific sample to match phone preprocessing

Usage:
    python test_single_sample.py \
        --test_dir data/arabic-asl-90kpts_LOSO_user01/test \
        --tflite_model /path/to/final_model_fp32.tflite \
        --config_path configs/arabic-asl-90kpts.yaml \
        --sample_file data/arabic-asl-90kpts_LOSO_user01/test/G10/user01_G10_R10.pkl \
        --sample_label G10
"""

import argparse
import json
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
import glob
import os
from collections import defaultdict
import yaml

# Class mapping (0-9 for G01-G10)
CLASS_NAMES = [
    'G01', 'G02', 'G03', 'G04', 'G05',
    'G06', 'G07', 'G08', 'G09', 'G10'
]


def load_pickle_file(pickle_path, verbose=True):
    """
    Load keypoints from pickle file.
    
    Args:
        pickle_path: path to pickle file
        verbose: if True, print detailed info
    
    Returns:
        keypoints: numpy array [num_frames, 90, 2]
    """
    if verbose:
        print(f"Loading pickle file: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract keypoints (should be [T, 90, 2])
    if isinstance(data, dict):
        if 'keypoints' in data:
            keypoints = data['keypoints']
        elif 'data' in data:
            keypoints = data['data']
        else:
            # Try to find the keypoints array
            for key, value in data.items():
                if isinstance(value, np.ndarray) and value.ndim == 3:
                    keypoints = value
                    if verbose:
                        print(f"  Found keypoints in key: '{key}'")
                    break
            else:
                raise ValueError(f"Could not find keypoints in pickle file. Keys: {list(data.keys())}")
    elif isinstance(data, np.ndarray):
        keypoints = data
    else:
        raise ValueError(f"Unexpected pickle format: {type(data)}")
    
    if verbose:
        print(f"  Keypoints shape: {keypoints.shape}")
        print(f"  Keypoints dtype: {keypoints.dtype}")
        print(f"  Keypoints range: [{keypoints.min():.3f}, {keypoints.max():.3f}]")
    
    # Validate shape
    if keypoints.ndim != 3:
        raise ValueError(f"Expected 3D array [T, 90, 2/3], got shape {keypoints.shape}")
    
    num_frames, num_kpts, num_coords = keypoints.shape
    if verbose:
        print(f"  Frames: {num_frames}")
        print(f"  Keypoints per frame: {num_kpts}")
        print(f"  Coordinates: {num_coords}")
    
    if num_kpts != 90:
        raise ValueError(f"Expected 90 keypoints, got {num_kpts}")
    
    # Handle 3D coordinates (x, y, z/visibility) - extract only x, y
    if num_coords == 3:
        if verbose:
            print(f"  Extracting only x, y coordinates (dropping 3rd dimension)")
        keypoints = keypoints[:, :, :2]
        if verbose:
            print(f"  New shape: {keypoints.shape}")
    elif num_coords != 2:
        raise ValueError(f"Expected 2 or 3 coordinates, got {num_coords}")
    
    return keypoints


def dump_keypoint_frames(keypoints, dump_path):
    """
    Dump per-frame keypoints (split by groups) to JSON for inspection.
    """
    dump_data = {
        'shape': keypoints.shape,
        'num_frames': int(keypoints.shape[0]),
        'frames': []
    }
    for idx, frame in enumerate(keypoints):
        dump_data['frames'].append({
            'frame_index': idx,
            'pose': frame[0:23].tolist(),
            'left_hand': frame[23:44].tolist(),
            'right_hand': frame[44:65].tolist(),
            'face': frame[65:90].tolist(),
        })
    with open(dump_path, 'w') as f:
        json.dump(dump_data, f, indent=2)
    print(f"📝 Dumped raw keypoints for {keypoints.shape[0]} frames to {dump_path}")


def log_raw_keypoints_sample(keypoints, frame_idx=0):
    """
    Log raw keypoints before any preprocessing (to match phone app's RAW output).
    
    Args:
        keypoints: numpy array [T, 90, 2]
        frame_idx: which frame to show (default: 0)
    """
    if keypoints.shape[0] <= frame_idx:
        print(f"⚠️ Frame {frame_idx} not available (only {keypoints.shape[0]} frames)")
        return
    
    frame = keypoints[frame_idx]
    
    print(f"\n🔍 RAW (pre-clip) - frame {frame_idx} sample:")
    print(f"   Pose (first 5): {frame[0:5].tolist()}")
    print(f"   Left hand (first 5): {frame[23:28].tolist()}")
    print(f"   Right hand (first 5): {frame[44:49].tolist()}")
    print(f"   Face (first 5): {frame[65:70].tolist()}")


def log_bounding_box_stats(keypoints, frame_idx=0):
    """
    Log bounding box statistics for each keypoint group (to match phone app's output).
    
    Args:
        keypoints: numpy array [T, 90, 2]
        frame_idx: which frame to analyze (default: 0)
    """
    if keypoints.shape[0] <= frame_idx:
        print(f"⚠️ Frame {frame_idx} not available (only {keypoints.shape[0]} frames)")
        return
    
    frame = keypoints[frame_idx]
    
    print(f"\n📐 Bounding Box Stats (frame {frame_idx}, for comparison with phone):")
    
    groups = [
        ('Pose', 0, 22),
        ('L-Hand', 23, 43),
        ('R-Hand', 44, 64),
        ('Face', 65, 89),
    ]
    
    for name, start, end in groups:
        group_kpts = frame[start:end+1]
        
        # Find valid (non-zero) keypoints
        valid_mask = np.any(group_kpts != 0.0, axis=1)
        valid_kpts = group_kpts[valid_mask]
        
        if len(valid_kpts) == 0:
            print(f"   {name}: all zeros (no valid keypoints)")
            continue
        
        minX = valid_kpts[:, 0].min()
        maxX = valid_kpts[:, 0].max()
        minY = valid_kpts[:, 1].min()
        maxY = valid_kpts[:, 1].max()
        w = maxX - minX
        h = maxY - minY
        
        print(f"   {name}: minX={minX:.3f}, maxX={maxX:.3f}, "
              f"minY={minY:.3f}, maxY={maxY:.3f}, "
              f"w={w:.3f}, h={h:.3f}")


def log_frame_summary(keypoints, label, max_frames=3, sample_interval_ms=150):
    """
    Print summary of frames sampled at specified intervals (to match phone sampling).
    
    Args:
        keypoints: numpy array [T, 90, 2]
        label: string label for the summary
        max_frames: max number of frames to show details for
        sample_interval_ms: sampling interval in milliseconds (default 150ms to match phone)
    """
    print(f"\n🔍 {label} frame summary (sampled every {sample_interval_ms}ms to match phone):")
    total_frames = keypoints.shape[0]
    print(f"   Total frames: {total_frames}")
    
    # Calculate frame indices at specified intervals (assuming 15fps for this dataset)
    fps = 15  # Actual video frame rate
    frames_per_sample = int(round((sample_interval_ms / 1000.0) * fps))
    frames_per_sample = max(1, frames_per_sample)  # At least 1
    
    sampled_indices = list(range(0, total_frames, frames_per_sample))
    print(f"   Sampling every {frames_per_sample} frames: {len(sampled_indices)} samples")
    print(f"   Frame indices: {sampled_indices[:10]}{'...' if len(sampled_indices) > 10 else ''}")
    
    group_defs = {
        'pose': (0, 23),
        'left_hand': (23, 44),
        'right_hand': (44, 65),
        'face': (65, 90),
    }
    
    for list_idx, frame_idx in enumerate(sampled_indices[:21]):  # Match phone's typical 21 frames
        frame = keypoints[frame_idx]
        counts = {}
        for name, (start, end) in group_defs.items():
            segment = frame[start:end]
            counts[name] = int(np.count_nonzero(np.any(segment != 0.0, axis=1)))
        
        time_ms = frame_idx * (1000 / fps)
        print(f"   Frame {frame_idx} (~{int(time_ms)}ms): pose {counts['pose']}/23, "
              f"L-hand {counts['left_hand']}/21, "
              f"R-hand {counts['right_hand']}/21, "
              f"face {counts['face']}/25")
        
        # Show details for first few frames
        if list_idx < max_frames:
            for name, (start, end) in group_defs.items():
                segment = np.round(frame[start:end][:5], 3).tolist()
                print(f"     {name} first 5: {segment}")


def normalize_part(keypoint, debug=False, group_name=""):
    """
    Normalize a group of keypoints using THE SAME method as training (dataset.py).
    This adds padding and makes bounding box square.
    
    Args:
        keypoint: numpy array of shape (N, 2)
        debug: if True, print bounding box details
        group_name: name for debug output
    
    Returns:
        normalized keypoints of same shape
    """
    assert keypoint.shape[-1] == 2, "Keypoints must have x, y"
    
    x_coords = keypoint[:, 0]
    y_coords = keypoint[:, 1]
    
    min_x, min_y = np.min(x_coords), np.min(y_coords)
    max_x, max_y = np.max(x_coords), np.max(y_coords)
    
    w = max_x - min_x
    h = max_y - min_y
    
    # Add padding to bounding box (THIS IS THE KEY DIFFERENCE!)
    if w > h:
        delta_x = 0.05 * w
        delta_y = delta_x + ((w - h) / 2)
    else:
        delta_y = 0.05 * h
        delta_x = delta_y + ((h - w) / 2)
    
    s_point = [max(0, min(min_x - delta_x, 1)), max(0, min(min_y - delta_y, 1))]
    e_point = [max(0, min(max_x + delta_x, 1)), max(0, min(max_y + delta_y, 1))]
    
    if debug:
        print(f"    {group_name}: minX={min_x:.3f}, maxX={max_x:.3f}, minY={min_y:.3f}, maxY={max_y:.3f}, "
              f"w={w:.3f}, h={h:.3f}, sX={s_point[0]:.3f}, eX={e_point[0]:.3f}, sY={s_point[1]:.3f}, eY={e_point[1]:.3f}")
    
    # Normalize keypoints
    result = keypoint.copy()
    if (e_point[0] - s_point[0]) != 0.0:
        result[:, 0] = (keypoint[:, 0] - s_point[0]) / (e_point[0] - s_point[0])
    if (e_point[1] - s_point[1]) != 0.0:
        result[:, 1] = (keypoint[:, 1] - s_point[1]) / (e_point[1] - s_point[1])
    
    return result


def normalize_keypoints(keypoints, joint_idxs):
    """
    Normalize keypoints by groups using SAME method as training (dataset.py).
    
    Args:
        keypoints: numpy array of shape (T, K, 2)
        joint_idxs: list of lists of keypoint indices for each group
    
    Returns:
        normalized keypoints of same shape
    """
    if joint_idxs is None:
        return keypoints
    
    # Get flat list of indices
    flat_joint_idxs = []
    for group in joint_idxs:
        flat_joint_idxs.extend(group)
    flat_joint_idxs = sorted(flat_joint_idxs)
    
    # Create mapping from original indices to filtered positions
    idx_to_pos = {idx: pos for pos, idx in enumerate(flat_joint_idxs)}
    
    # Normalize each group
    group_names = ["Pose", "L-Hand", "R-Hand", "Face"]
    for i in range(keypoints.shape[0]):  # for each frame
        if i == 0:
            print(f"\n  🔧 Normalization debug (frame 0):")
        for gidx, group in enumerate(joint_idxs):  # for each group
            # Map original indices to positions in filtered array
            filtered_positions = [idx_to_pos[idx] for idx in group if idx in idx_to_pos]
            
            if len(filtered_positions) > 0:
                # Get keypoints for this group
                group_keypoints = keypoints[i, filtered_positions, :]
                # Normalize the group (with debug for frame 0)
                debug_mode = (i == 0)
                group_name = group_names[gidx] if gidx < len(group_names) else f"Group {gidx}"
                normalized = normalize_part(group_keypoints, debug=debug_mode, group_name=group_name)
                # Put back
                keypoints[i, filtered_positions, :] = normalized
    
    return keypoints


def preprocess_for_tflite_training_style(keypoints, joint_idxs, max_len=64):
    """
    Preprocess keypoints using THE SAME method as training (dataset.py).
    
    Args:
        keypoints: numpy array [T, 90, 2]
        joint_idxs: list of lists of keypoint indices for each group
        max_len: max sequence length (default 64)
    
    Returns:
        keypoints_padded: [1, 64, 90, 2]
        attention_mask: [1, 64]
    """
    num_frames = keypoints.shape[0]
    
    # Step 1: Clip values to [0, 1] (like dataset.py line 90)
    keypoints = np.clip(keypoints, 0, 1)
    
    # Step 2: Clip to max 64 frames (like dataset.py lines 92-94)
    if num_frames > max_len:
        # Uniform sampling
        indices = np.linspace(0, num_frames - 1, max_len, dtype=int)
        keypoints = keypoints[indices]
        num_frames = max_len
    
    # Step 3: Normalize by groups (like dataset.py line 101)
    keypoints = normalize_keypoints(keypoints, joint_idxs)
    
    # Step 4: Pad to max_len if needed
    if num_frames < max_len:
        pad_len = max_len - num_frames
        keypoints = np.pad(
            keypoints,
            pad_width=((0, pad_len), (0, 0), (0, 0)),
            mode='constant',
            constant_values=0.0
        )
        mask = np.concatenate([
            np.ones(num_frames, dtype=np.float32),
            np.zeros(pad_len, dtype=np.float32)
        ])
    else:
        mask = np.ones(max_len, dtype=np.float32)
    
    # Step 5: Add batch dimension
    keypoints = keypoints[np.newaxis, ...]  # [1, 64, 90, 2]
    mask = mask[np.newaxis, ...]            # [1, 64]
    
    return keypoints.astype(np.float32), mask.astype(np.float32)


def postprocess_tflite_output(logits, return_top_k=5):
    """
    Process TFLite model output to get predictions.
    
    Args:
        logits: numpy array [1, 10] or [10] - raw model output
        return_top_k: return top K predictions (default: 5)
    
    Returns:
        dict with predicted class, confidence, etc.
    """
    # Remove batch dimension if present
    if logits.ndim == 2:
        logits = logits[0]
    
    # Apply softmax
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    # Get predicted class
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])
    
    # Get top-k
    top_k_indices = np.argsort(probabilities)[-return_top_k:][::-1]
    top_k_probs = probabilities[top_k_indices]
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top_k_classes': top_k_indices.tolist(),
        'top_k_confidences': top_k_probs.tolist(),
        'all_probabilities': probabilities
    }


def load_test_set(test_dir):
    """
    Load all test samples from the test directory.
    
    Args:
        test_dir: path to test directory (e.g., data/arabic-asl-90kpts_LOSO_user01/test)
    
    Returns:
        samples: list of (keypoints, label, filename) tuples
    """
    print(f"\nLoading test set from: {test_dir}")
    
    samples = []
    class_counts = defaultdict(int)
    
    # Find all pickle files
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"  Warning: Class directory not found: {class_dir}")
            continue
        
        pickle_files = glob.glob(os.path.join(class_dir, "*.pkl"))
        
        for pkl_file in pickle_files:
            try:
                keypoints = load_pickle_file(pkl_file, verbose=False)
                label = CLASS_NAMES.index(class_name)
                filename = os.path.basename(pkl_file)
                samples.append((keypoints, label, filename, class_name))
                class_counts[class_name] += 1
            except Exception as e:
                print(f"  Error loading {pkl_file}: {e}")
    
    print(f"  Total samples: {len(samples)}")
    print(f"  Class distribution:")
    for class_name in CLASS_NAMES:
        count = class_counts[class_name]
        if count > 0:
            print(f"    {class_name}: {count} samples")
    
    return samples


def load_config(config_path):
    """Load model configuration and create joint_idx groups."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_joint_idx_groups(joint_idx_flat):
    """
    Create groups from flat joint_idx list.
    Groups: Pose (0-22), Left Hand (23-43), Right Hand (44-64), Face (65-89)
    
    Args:
        joint_idx_flat: flat list of 90 keypoint indices
    
    Returns:
        list of lists (groups)
    """
    if not joint_idx_flat:
        return []
    
    sorted_idx = sorted(joint_idx_flat)
    total_kpts = len(sorted_idx)
    
    # Known structure for 90 keypoints:
    # - Pose: 0-22 (23 points)
    # - Left Hand: 23-43 (21 points)
    # - Right Hand: 44-64 (21 points)
    # - Face: 65-89 (25 points)
    
    if total_kpts >= 67:  # At least some pose + 2 hands + face
        # Last 25: face
        face_kpts = sorted_idx[-25:]
        # Previous 21: right hand  
        right_hand_kpts = sorted_idx[-46:-25]
        # Previous 21: left hand
        left_hand_kpts = sorted_idx[-67:-46]
        # Everything else: pose/body
        body_kpts = sorted_idx[:-67]
        
        # Create groups (non-empty only)
        groups = []
        if body_kpts:
            groups.append(body_kpts)
        if left_hand_kpts:
            groups.append(left_hand_kpts)
        if right_hand_kpts:
            groups.append(right_hand_kpts)
        if face_kpts:
            groups.append(face_kpts)
        
        return groups
    else:
        # Fallback: just use all as one group
        return [sorted_idx]


def evaluate_test_set(tflite_path, test_dir, joint_idxs):
    """
    Evaluate TFLite model on full test set using CORRECT preprocessing.
    
    Args:
        tflite_path: path to TFLite model
        test_dir: path to test directory
        joint_idxs: list of lists of keypoint indices for each group
    
    Returns:
        metrics: dict with accuracy, per-class metrics, confusion matrix
    """
    print("\n" + "="*80)
    print("EVALUATING ON FULL TEST SET (using training preprocessing)")
    print("="*80)
    
    # Load test samples
    samples = load_test_set(test_dir)
    
    if len(samples) == 0:
        print("No test samples found!")
        return None
    
    # Load TFLite model
    print(f"\nLoading TFLite model: {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Find keypoints and mask inputs
    keypoints_idx = None
    mask_idx = None
    for i, detail in enumerate(input_details):
        if len(detail['shape']) == 4:
            keypoints_idx = i
        elif len(detail['shape']) == 2:
            mask_idx = i
    
    # Run inference on all samples
    print(f"\nRunning inference on {len(samples)} samples...")
    
    predictions = []
    true_labels = []
    confidences = []
    
    for i, (keypoints, label, filename, class_name) in enumerate(samples):
        # Preprocess using TRAINING method
        kpts_input, mask_input = preprocess_for_tflite_training_style(keypoints, joint_idxs)
        
        # Set inputs
        interpreter.set_tensor(input_details[keypoints_idx]['index'], kpts_input.astype(np.float32))
        interpreter.set_tensor(input_details[mask_idx]['index'], mask_input.astype(np.float32))
        
        # Inference
        interpreter.invoke()
        logits = interpreter.get_tensor(output_details[0]['index'])
        
        # Postprocess
        result = postprocess_tflite_output(logits)
        
        predictions.append(result['predicted_class'])
        true_labels.append(label)
        confidences.append(result['confidence'])
        
        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processed {i+1}/{len(samples)} samples...")
    
    # Calculate metrics
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    confidences = np.array(confidences)
    
    # Overall accuracy
    accuracy = np.mean(predictions == true_labels)
    
    # Per-class accuracy
    per_class_acc = {}
    per_class_counts = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = true_labels == i
        if class_mask.sum() > 0:
            class_acc = np.mean(predictions[class_mask] == true_labels[class_mask])
            per_class_acc[class_name] = class_acc
            per_class_counts[class_name] = class_mask.sum()
    
    # Confusion matrix
    num_classes = len(CLASS_NAMES)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(true_labels, predictions):
        confusion_matrix[true_label, pred_label] += 1
    
    # Display results
    print("\n" + "="*80)
    print("TEST SET EVALUATION RESULTS")
    print("="*80)
    print(f"Total samples: {len(samples)}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Mean Confidence: {confidences.mean():.4f}")
    
    print(f"\nPer-class Accuracy:")
    for class_name in CLASS_NAMES:
        if class_name in per_class_acc:
            acc = per_class_acc[class_name]
            count = per_class_counts[class_name]
            print(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%) - {count} samples")
    
    print(f"\nConfusion Matrix:")
    print(f"{'':6s} " + " ".join([f"{c:>5s}" for c in CLASS_NAMES]))
    for i, class_name in enumerate(CLASS_NAMES):
        row = confusion_matrix[i]
        if row.sum() > 0:
            print(f"{class_name:6s} " + " ".join([f"{v:5d}" for v in row]))
    
    print("="*80)
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': confusion_matrix,
        'num_samples': len(samples)
    }


def run_tflite_inference(tflite_path, keypoints, joint_idxs):
    """
    Run TFLite inference on keypoints using the SAME preprocessing as training.
    
    Args:
        tflite_path: path to TFLite model
        keypoints: numpy array [T, 90, 2]
        joint_idxs: list of lists of keypoint indices for each group
    
    Returns:
        result: dict with predictions
    """
    print(f"\nLoading TFLite model: {tflite_path}")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Model inputs: {len(input_details)}")
    for i, detail in enumerate(input_details):
        print(f"    [{i}] {detail['name']}: {detail['dtype']} {detail['shape']}")
    
    print(f"  Model outputs: {len(output_details)}")
    for i, detail in enumerate(output_details):
        print(f"    [{i}] {detail['name']}: {detail['dtype']} {detail['shape']}")
    
    # Preprocess keypoints (SAME as training)
    print(f"\nPreprocessing keypoints (using training method)...")
    kpts_input, mask_input = preprocess_for_tflite_training_style(keypoints, joint_idxs)
    
    print(f"  Preprocessed keypoints shape: {kpts_input.shape}")
    print(f"  Preprocessed mask shape: {mask_input.shape}")
    print(f"  Valid frames in mask: {int(mask_input.sum())} / {mask_input.shape[1]}")
    print(f"  Keypoints range: [{kpts_input.min():.3f}, {kpts_input.max():.3f}]")
    
    # Show sample of preprocessed data
    print(f"\n  Sample preprocessed keypoints (frame 0):")
    # Show first 5 points from each group
    pose_pts = kpts_input[0, 0, :5, :]
    lhand_pts = kpts_input[0, 0, 23:28, :]
    rhand_pts = kpts_input[0, 0, 44:49, :]
    face_pts = kpts_input[0, 0, 65:70, :]
    
    print(f"    Pose (first 5): {[[f'{x:.3f}', f'{y:.3f}'] for x, y in pose_pts]}")
    print(f"    L-hand (first 5): {[[f'{x:.3f}', f'{y:.3f}'] for x, y in lhand_pts]}")
    print(f"    R-hand (first 5): {[[f'{x:.3f}', f'{y:.3f}'] for x, y in rhand_pts]}")
    print(f"    Face (first 5): {[[f'{x:.3f}', f'{y:.3f}'] for x, y in face_pts]}")
    
    # Find keypoints and mask inputs by shape
    keypoints_idx = None
    mask_idx = None
    
    for i, detail in enumerate(input_details):
        shape = detail['shape']
        if len(shape) == 4:  # keypoints: [1, 64, 90, 2]
            keypoints_idx = i
        elif len(shape) == 2:  # attention_mask: [1, 64]
            mask_idx = i
    
    if keypoints_idx is None or mask_idx is None:
        raise ValueError("Could not identify keypoints and mask inputs")
    
    print(f"\n  Input mapping: keypoints=input[{keypoints_idx}], mask=input[{mask_idx}]")
    
    # Set input tensors
    interpreter.set_tensor(input_details[keypoints_idx]['index'], kpts_input.astype(np.float32))
    interpreter.set_tensor(input_details[mask_idx]['index'], mask_input.astype(np.float32))
    
    # Run inference
    print(f"\nRunning inference...")
    interpreter.invoke()
    
    # Get output
    logits = interpreter.get_tensor(output_details[0]['index'])
    print(f"  Raw logits shape: {logits.shape}")
    print(f"  Raw logits: {logits[0]}")
    
    # Postprocess
    result = postprocess_tflite_output(logits)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Test TFLite on full test set and single sample')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to test directory (e.g., data/arabic-asl-90kpts_LOSO_user01/test)')
    parser.add_argument('--tflite_model', type=str, required=True,
                        help='Path to TFLite model (e.g., final_model_fp32.tflite)')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to model config file (e.g., configs/arabic-asl-90kpts.yaml)')
    parser.add_argument('--sample_file', type=str, default=None,
                        help='Path to specific sample pickle file (optional)')
    parser.add_argument('--sample_label', type=str, default=None,
                        help='Ground truth label for sample (e.g., G10)')
    parser.add_argument('--skip_test_set', action='store_true',
                        help='Skip full test set evaluation (only test sample)')
    parser.add_argument('--dump_raw_sample', type=str, default="",
                        help='Optional path to dump raw sample keypoints as JSON')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TFLite Inference Test (using TRAINING preprocessing)")
    print("="*80)
    print(f"Test directory: {args.test_dir}")
    print(f"TFLite model: {args.tflite_model}")
    print(f"Config file: {args.config_path}")
    if args.sample_file:
        print(f"Sample file: {args.sample_file}")
        if args.sample_label:
            print(f"Sample ground truth: {args.sample_label}")
    print("="*80)
    
    # Load config to get joint_idxs
    print(f"\nLoading config from: {args.config_path}")
    config = load_config(args.config_path)
    joint_idx_flat = config['joint_idx']
    
    # Check if it's already grouped or flat
    if joint_idx_flat and isinstance(joint_idx_flat[0], list):
        # Already grouped
        joint_idxs = joint_idx_flat
    else:
        # Flat list - create groups
        joint_idxs = create_joint_idx_groups(joint_idx_flat)
    
    print(f"  Total keypoints: {len(joint_idx_flat)}")
    print(f"  Grouped into: {len(joint_idxs)} groups")
    for i, group in enumerate(joint_idxs):
        print(f"    Group {i}: {len(group)} keypoints")
    
    # 1. Evaluate on full test set (unless skipped)
    if not args.skip_test_set:
        test_metrics = evaluate_test_set(args.tflite_model, args.test_dir, joint_idxs)
        
        if test_metrics:
            print("\n✓ Full test set evaluation complete!")
            print(f"  Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    
    # 2. Test on specific sample (if provided)
    if args.sample_file:
        print("\n" + "="*80)
        print("TESTING SPECIFIC SAMPLE")
        print("="*80)
        print(f"Sample file: {args.sample_file}")
        if args.sample_label:
            print(f"Ground truth: {args.sample_label}")
        
        # Load keypoints from pickle
        keypoints = load_pickle_file(args.sample_file)
        
        # Log RAW keypoints (before any preprocessing) - matching phone app
        print("\n" + "="*80)
        print("RAW KEYPOINTS (before any preprocessing)")
        print("="*80)
        log_raw_keypoints_sample(keypoints, frame_idx=0)
        log_bounding_box_stats(keypoints, frame_idx=0)
        
        log_frame_summary(keypoints, "RAW sample (all frames)")
        if args.dump_raw_sample:
            dump_keypoint_frames(keypoints, args.dump_raw_sample)
        
        # Subsample to match phone's 150ms sampling rate
        # Video is 15fps, so 150ms = 0.15s × 15fps = 2.25 frames ≈ 2 frames
        fps = 15  # Actual video frame rate
        frames_per_sample = int(round((150 / 1000.0) * fps))  # ~2 frames per 150ms at 15fps
        frames_per_sample = max(1, frames_per_sample)
        sampled_indices = list(range(0, keypoints.shape[0], frames_per_sample))
        keypoints_sampled = keypoints[sampled_indices[:21]]  # Match phone's typical 21 frames
        
        print(f"\n📱 Subsampled to match phone (150ms intervals, up to 21 frames): {keypoints_sampled.shape[0]} frames")
        log_frame_summary(keypoints_sampled, "Subsampled (phone-matched)", max_frames=21)
        
        # Log RAW subsampled keypoints (still before clip/normalize)
        print("\n" + "="*80)
        print("RAW SUBSAMPLED KEYPOINTS (before clip/normalize, matching phone)")
        print("="*80)
        log_raw_keypoints_sample(keypoints_sampled, frame_idx=0)
        log_bounding_box_stats(keypoints_sampled, frame_idx=0)
        
        # Run TFLite inference on SUBSAMPLED keypoints (matching phone)
        print("\n" + "="*80)
        print("INFERENCE ON SUBSAMPLED FRAMES (matching phone's 150ms sampling)")
        print("="*80)
        result = run_tflite_inference(args.tflite_model, keypoints_sampled, joint_idxs)
        
        # Display sample results
        print("\n" + "="*80)
        print("SAMPLE RESULTS")
        print("="*80)
        
        predicted_class = result['predicted_class']
        predicted_label = CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else f"Class_{predicted_class}"
        
        print(f"Predicted class: {predicted_class} ({predicted_label})")
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        
        if args.sample_label:
            # Check if prediction matches ground truth
            if args.sample_label in CLASS_NAMES:
                true_class = CLASS_NAMES.index(args.sample_label)
                is_correct = (predicted_class == true_class)
                print(f"Ground truth: {true_class} ({args.sample_label})")
                print(f"Correct: {'✓ YES' if is_correct else '✗ NO'}")
                
                if not is_correct:
                    true_confidence = result['all_probabilities'][true_class]
                    print(f"Confidence for correct class ({args.sample_label}): {true_confidence:.4f} ({true_confidence*100:.2f}%)")
        
        print(f"\nTop 5 predictions:")
        for i, (cls, conf) in enumerate(zip(result['top_k_classes'], result['top_k_confidences']), 1):
            label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class_{cls}"
            marker = "← PREDICTED" if i == 1 else ""
            if args.sample_label and label == args.sample_label:
                marker = "← GROUND TRUTH"
            print(f"  {i}. {label:4s} (class {cls}): {conf:.4f} ({conf*100:.2f}%) {marker}")
        
        print("\nAll class probabilities:")
        for i, (label, prob) in enumerate(zip(CLASS_NAMES, result['all_probabilities'])):
            marker = ""
            if i == predicted_class:
                marker = "← PREDICTED"
            if args.sample_label and label == args.sample_label:
                marker += " ← GROUND TRUTH"
            print(f"  {label:4s} (class {i}): {prob:.4f} ({prob*100:.2f}%) {marker}")
        
        print("="*80)
        print("\nComparison with phone app:")
        print("  - If this matches phone output → TFLite model is correct")
        print("  - If this differs from phone → Phone preprocessing is different")
        print("="*80)


if __name__ == "__main__":
    main()

