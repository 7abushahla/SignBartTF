"""
TensorFlow/Keras utilities for SignBART.
Converted from PyTorch implementation.
"""
import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import glob


def setup_gpu(logger=None):
    """Configure GPU usage if available and return (has_gpu, device_label)."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            message = f"Using GPU(s): {gpus}"
            if logger:
                logger.info(message)
            else:
                print(message)
            return True, f"GPU: {gpus}"
        except RuntimeError as e:
            message = f"GPU setup failed, using CPU: {e}"
            if logger:
                logger.error(message)
            else:
                print(message)
            return False, "CPU"

    message = "No GPU found, using CPU"
    if logger:
        logger.info(message)
    else:
        print(message)
    return False, "CPU"

# MediaPipe Holistic keypoint structure
total_body_idx = 33  # Pose landmarks: 0-32
total_hand = 21      # Hand landmarks per hand: 21 each
total_face = 25      # Selected face landmarks: 75-99

# ============================================================================
# Keypoint Index Definitions
# ============================================================================

# POSE KEYPOINTS (0-32) - Full body pose from MediaPipe
pose_idx = list(range(0, 33))

# Upper body pose only (shoulders, elbows, wrists) - 6 keypoints
upper_body_idx = list(range(11, 17))

# HAND KEYPOINTS
# Left hand: 33-53 (21 keypoints)
lefthand_idx = [x + total_body_idx for x in range(0, 21)]
# Right hand: 54-74 (21 keypoints)  
righthand_idx = [x + 21 for x in lefthand_idx]

# FACE KEYPOINTS (75-99) - 25 selected face landmarks
face_idx = list(range(75, 100))

# ============================================================================
# Common Keypoint Configurations
# ============================================================================

# Hands only (42 keypoints) - Original approach
hands_only_idx = lefthand_idx + righthand_idx

# Upper body + hands (48 keypoints) - Body context with hands
pose_hands_idx = upper_body_idx + lefthand_idx + righthand_idx

# Full body 75 keypoints - All MediaPipe Holistic (pose + hands)
full_75_idx = pose_idx + lefthand_idx + righthand_idx

# Full body 100 keypoints - MediaPipe Holistic + selected face landmarks
full_100_idx = pose_idx + lefthand_idx + righthand_idx + face_idx

# Grouped versions (for normalization) - each sublist is normalized independently
hands_only_groups = [lefthand_idx, righthand_idx]
pose_hands_groups = [upper_body_idx, lefthand_idx, righthand_idx]
full_75_groups = [pose_idx, lefthand_idx, righthand_idx]
full_100_groups = [pose_idx, lefthand_idx, righthand_idx, face_idx]

# Legacy aliases for backward compatibility
body_idx = upper_body_idx
total_idx = pose_hands_idx
all_keypoints_idx = full_75_idx
all_keypoints_groups = full_75_groups


# ============================================================================
# Training and Evaluation Functions (TensorFlow version)
# ============================================================================

def accuracy(logits, labels):
    """
    Calculate accuracy from logits and labels.
    
    Args:
        logits: (batch_size, num_classes) tensor
        labels: (batch_size,) tensor
    
    Returns:
        accuracy: float
    """
    preds = tf.argmax(logits, axis=1)
    # Cast both to same type to avoid type mismatch
    preds = tf.cast(preds, tf.int32)
    labels = tf.cast(labels, tf.int32)
    correct = tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.float32))
    total = tf.cast(tf.shape(labels)[0], tf.float32)
    return (correct / total).numpy()


def top_k_accuracy(logits, labels, k=5):
    """
    Calculate top-k accuracy.
    
    Args:
        logits: (batch_size, num_classes) tensor
        labels: (batch_size,) tensor
        k: int, top-k predictions to consider
    
    Returns:
        top_k_accuracy: float
    """
    top_k_preds = tf.nn.top_k(logits, k=k).indices
    # Cast both to same type to avoid type mismatch
    top_k_preds = tf.cast(top_k_preds, tf.int32)
    labels_expanded = tf.expand_dims(tf.cast(labels, tf.int32), axis=1)
    correct = tf.reduce_any(tf.equal(top_k_preds, labels_expanded), axis=1)
    return tf.reduce_mean(tf.cast(correct, tf.float32)).numpy()


def save_checkpoint(model, optimizer, epoch, path_dir, name=None):
    """
    Save model and optimizer checkpoint.
    
    Args:
        model: Keras model
        optimizer: Keras optimizer
        epoch: current epoch number
        path_dir: directory to save checkpoint
        name: optional name suffix
    """
    if not os.path.exists(path_dir):
        print(f"Making directory {path_dir}")
        os.makedirs(path_dir)
    
    if name is None:
        checkpoint_path = f'{path_dir}/checkpoint_{epoch}'
    else:
        checkpoint_path = f'{path_dir}/checkpoint_{epoch}_{name}'
    
    # Save model weights
    model.save_weights(f'{checkpoint_path}.h5')
    
    # Save optimizer state (if needed)
    # Note: TF optimizer state saving is different from PyTorch
    print(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(model, optimizer, path, resume=True):
    """
    Load model checkpoint.
    
    Args:
        model: Keras model
        optimizer: Keras optimizer
        path: path to checkpoint directory or file
        resume: whether to resume training (return epoch number)
    
    Returns:
        start_epoch: epoch to start from
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    if os.path.isdir(path):
        # Find best checkpoint
        checkpoint_files = [f for f in os.listdir(path) if f.endswith('.h5')]
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {path}")
        
        # Try to find best_val checkpoint first, then best_train, then latest
        best_val = [f for f in checkpoint_files if 'best_val' in f]
        best_train = [f for f in checkpoint_files if 'best_train' in f]
        
        if best_val:
            filename = f'{path}/{best_val[0]}'
            print(f'Loading best validation checkpoint: {best_val[0]}')
        elif best_train:
            filename = f'{path}/{best_train[0]}'
            print(f'Loading best training checkpoint: {best_train[0]}')
        else:
            # Find latest by epoch number
            epochs = []
            for f in checkpoint_files:
                try:
                    epoch_str = f.replace('checkpoint_', '').replace('.h5', '')
                    epoch_num = int(epoch_str.split('_')[0])
                    epochs.append((epoch_num, f))
                except (ValueError, IndexError):
                    continue
            
            if not epochs:
                raise FileNotFoundError(f"No valid checkpoint files found in {path}")
            
            latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
            filename = f'{path}/{latest_file}'
            print(f'Loading latest checkpoint: epoch {latest_epoch}')
    else:
        filename = path
        print(f"Loading checkpoint from file: {path}")
    
    # Load weights
    model.load_weights(filename)
    
    # Extract epoch number if resuming
    if resume:
        try:
            epoch_str = os.path.basename(filename).replace('checkpoint_', '').replace('.h5', '')
            epoch = int(epoch_str.split('_')[0])
            return epoch + 1
        except:
            return 1
    else:
        return 1


def count_model_parameters(model):
    """
    Count trainable and total parameters in model.
    
    Args:
        model: Keras model
    
    Returns:
        dict with 'total' and 'trainable' parameter counts
    """
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_params = sum([tf.size(w).numpy() for w in model.weights])
    
    return {"total": total_params, "trainable": trainable_params}


# ============================================================================
# Utility function to get keypoint configuration by name
# ============================================================================

def get_keypoint_config(config_name):
    """
    Get keypoint indices and groups by configuration name.
    
    Args:
        config_name: One of 'hands_only', 'pose_hands', 'full_75', 'full_100', 
                     'all_keypoints' (legacy, refers to 75)
    
    Returns:
        tuple: (flat_indices, grouped_indices)
    """
    configs = {
        'hands_only': (hands_only_idx, hands_only_groups),
        'pose_hands': (pose_hands_idx, pose_hands_groups),
        'full_75': (full_75_idx, full_75_groups),
        'full_100': (full_100_idx, full_100_groups),
        'all_keypoints': (full_75_idx, full_75_groups),  # Legacy alias
        'full': (full_75_idx, full_75_groups),  # Legacy alias
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Valid options: {list(configs.keys())}")
    
    return configs[config_name]


def print_keypoint_summary():
    """Print a summary of available keypoint configurations."""
    print("=" * 80)
    print("Keypoint Configuration Summary")
    print("=" * 80)
    print(f"MediaPipe Holistic Structure:")
    print(f"  Pose landmarks:       {total_body_idx} keypoints (indices 0-32)")
    print(f"  Left hand landmarks:  {total_hand} keypoints (indices 33-53)")
    print(f"  Right hand landmarks: {total_hand} keypoints (indices 54-74)")
    print(f"  Face landmarks:       {total_face} keypoints (indices 75-99)")
    print()
    print(f"Available Configurations:")
    print(f"  'hands_only':   {len(hands_only_idx):3d} keypoints in {len(hands_only_groups)} groups")
    print(f"  'pose_hands':   {len(pose_hands_idx):3d} keypoints in {len(pose_hands_groups)} groups")
    print(f"  'full_75':      {len(full_75_idx):3d} keypoints in {len(full_75_groups)} groups")
    print(f"  'full_100':     {len(full_100_idx):3d} keypoints in {len(full_100_groups)} groups")
    print("=" * 80)


if __name__ == "__main__":
    # Print keypoint configuration summary
    print_keypoint_summary()
    
    # Test keypoint configurations
    print("\nTesting get_keypoint_config():")
    for config_name in ['hands_only', 'pose_hands', 'full_75', 'full_100']:
        flat, groups = get_keypoint_config(config_name)
        print(f"  {config_name:15s}: {len(flat):3d} keypoints, {len(groups)} groups")
    
    # Test accuracy metrics
    print("\nTesting accuracy metrics:")
    test_logits = tf.random.normal((32, 64))
    test_labels = tf.random.uniform((32,), minval=0, maxval=64, dtype=tf.int32)
    
    acc = accuracy(test_logits, test_labels)
    top5_acc = top_k_accuracy(test_logits, test_labels, k=5)
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Top-5 Accuracy: {top5_acc:.4f}")
    
    print("\n✓ All utility tests passed!")



# =============================================================================
# 65 Keypoint Configuration (NO FACE) - Added for MLR511_Project compatibility
# =============================================================================

# Upper body pose (23) + Left hand (21) + Right hand (21) = 65 keypoints
# This matches on-device preprocessing in:
#   - MLR511_Project/lib/preprocessing/keypoint_preprocessing.dart
upper_body_65_idx = list(range(0, 65))
upper_body_65_groups = [list(range(0, 23)), list(range(23, 44)), list(range(44, 65))]

# v2.1: Pose subset (15) + Left hand (21) + Right hand (21) + Face subset (6) = 63 keypoints
# Pose indices correspond to selected pose landmarks; face indices are a small FaceMesh subset.
upper_body_63_idx = list(range(0, 63))
upper_body_63_groups = [
    list(range(0, 15)),   # Pose subset (15)
    list(range(15, 36)),  # Left hand (21)
    list(range(36, 57)),  # Right hand (21)
    list(range(57, 63)),  # Face subset (6)
]

# Add to the get_keypoint_config function by monkey-patching
_original_get_keypoint_config = get_keypoint_config

def get_keypoint_config(config_name):
    """
    Get keypoint indices and groups by configuration name.
    
    Extended to support 'upper_body_65' for 65-keypoint no-face configuration.
    """
    if config_name == 'upper_body_65':
        return (upper_body_65_idx, upper_body_65_groups)
    if config_name == 'upper_body_63':
        return (upper_body_63_idx, upper_body_63_groups)
    if config_name == 'v2_1_63':
        return (upper_body_63_idx, upper_body_63_groups)
    return _original_get_keypoint_config(config_name)


def ensure_dir_safe(path):
    """
    Ensure a directory exists at `path`. If a non-directory file exists at the
    path, rename it with a timestamp suffix and then create the directory.

    Accepts either a string or a pathlib.Path.
    """
    from pathlib import Path
    p = Path(path)
    if p.exists() and not p.is_dir():
        backup_name = str(p) + ".bak_" + datetime.now().strftime("%Y%m%dT%H%M%S")
        print(f"Warning: A file exists at '{p}'. Renaming it to '{backup_name}' to create directory.")
        try:
            os.rename(str(p), backup_name)
        except Exception as e:
            raise
    p.mkdir(parents=True, exist_ok=True)


def resolve_checkpoint_dir(exp_name_or_prefix_user: str):
    """
    Resolve a checkpoint directory for a given experiment name or prefix+user token.
    Tries the direct `checkpoints_{exp_name_or_prefix_user}` first, then falls
    back to any existing directory matching `checkpoints_*<user>` (useful when
    experiments were named with/without keypoint-count prefixes).

    Returns the directory path string (not guaranteed to exist).
    """
    from pathlib import Path
    # Direct candidate
    candidate = f"checkpoints_{exp_name_or_prefix_user}"
    if Path(candidate).exists():
        return candidate

    # Try to extract trailing user token (e.g., user01) and find any checkpoints_* that ends with it
    parts = exp_name_or_prefix_user.split("_")
    if parts:
        user_token = parts[-1]
        matches = glob.glob(f"checkpoints_*{user_token}")
        if matches:
            return matches[0]

    # As a last resort, return the original candidate (caller will handle missing files)
    return candidate
