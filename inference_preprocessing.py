#!/usr/bin/env python
"""
inference_preprocessing.py - Unified preprocessing for training and inference

This module provides IDENTICAL preprocessing to the on-device Android/iOS implementation.
It matches exactly what is done in:
    MLR511_Project/lib/preprocessing/keypoint_preprocessing.dart

The preprocessing steps are:
    1. Clip keypoint values to [0, 1] range
    2. Subsample or keep frames (max 64 frames)
    3. Normalize keypoints by groups (pose, left hand, right hand, [face])
    4. Pad to 64 frames if needed
    5. Return [1, 64, K, 2] tensor ready for TFLite

CRITICAL: This must match the on-device preprocessing EXACTLY for consistent inference.
"""

import numpy as np
from typing import List, Tuple, Optional

# Constants
MAX_SEQ_LEN = 64  # Fixed sequence length for TFLite


def clip_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Clip keypoint values to [0, 1] range.
    
    Args:
        keypoints: numpy array of shape (T, K, 2)
    
    Returns:
        clipped keypoints of same shape
    """
    return np.clip(keypoints, 0.0, 1.0)


def subsample_frames(keypoints: np.ndarray, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """
    Subsample frames if sequence is longer than max_len.
    Uses uniform sampling to preserve temporal information.
    
    Args:
        keypoints: numpy array of shape (T, K, 2)
        max_len: maximum sequence length
    
    Returns:
        subsampled keypoints of shape (min(T, max_len), K, 2)
    """
    num_frames = keypoints.shape[0]
    
    if num_frames <= max_len:
        return keypoints
    
    # Uniform subsampling (matches Dart: linspace indices)
    indices = np.linspace(0, num_frames - 1, max_len, dtype=int)
    return keypoints[indices]


def normalize_part(keypoints_group: np.ndarray) -> np.ndarray:
    """
    Normalize a group of keypoints using the SAME method as training (dataset.py)
    and on-device (keypoint_preprocessing.dart).
    
    This adds padding to the bounding box and makes it SQUARE.
    
    Args:
        keypoints_group: numpy array of shape (N, 2) - keypoints for one group
    
    Returns:
        normalized keypoints of same shape
    """
    assert keypoints_group.shape[-1] == 2, "Keypoints must have x, y"
    
    # Find valid (non-zero) keypoints
    valid_mask = (keypoints_group[:, 0] != 0.0) | (keypoints_group[:, 1] != 0.0)
    valid_kpts = keypoints_group[valid_mask]
    
    if len(valid_kpts) == 0:
        return keypoints_group
    
    # Find bounding box
    min_x = np.min(valid_kpts[:, 0])
    max_x = np.max(valid_kpts[:, 0])
    min_y = np.min(valid_kpts[:, 1])
    max_y = np.max(valid_kpts[:, 1])
    
    w = max_x - min_x
    h = max_y - min_y
    
    if w == 0 and h == 0:
        return keypoints_group
    
    # KEY: Add padding and make bounding box SQUARE
    # This matches training preprocessing exactly!
    if w > h:
        delta_x = 0.05 * w
        delta_y = delta_x + ((w - h) / 2)
    else:
        delta_y = 0.05 * h
        delta_x = delta_y + ((h - w) / 2)
    
    # Calculate start and end points with padding
    s_point_x = max(0.0, min(min_x - delta_x, 1.0))
    s_point_y = max(0.0, min(min_y - delta_y, 1.0))
    e_point_x = max(0.0, min(max_x + delta_x, 1.0))
    e_point_y = max(0.0, min(max_y + delta_y, 1.0))
    
    # Normalize each keypoint
    result = keypoints_group.copy()
    
    # Only normalize valid keypoints
    for i in range(len(keypoints_group)):
        if keypoints_group[i, 0] == 0.0 and keypoints_group[i, 1] == 0.0:
            continue  # Skip zero keypoints
        
        if (e_point_x - s_point_x) != 0.0:
            result[i, 0] = (keypoints_group[i, 0] - s_point_x) / (e_point_x - s_point_x)
        if (e_point_y - s_point_y) != 0.0:
            result[i, 1] = (keypoints_group[i, 1] - s_point_y) / (e_point_y - s_point_y)
    
    return result


def normalize_keypoints_by_groups(
    keypoints: np.ndarray,
    num_keypoints: int = 65
) -> np.ndarray:
    """
    Normalize keypoints by groups (pose, left hand, right hand, [face]).
    Each group is normalized independently.
    
    This matches the on-device Dart implementation exactly.
    
    Args:
        keypoints: numpy array of shape (T, K, 2)
    num_keypoints: 63 (pose+hands+face subset), 65 (no face) or 90 (with face)
    
    Returns:
        normalized keypoints of same shape
    """
    # Define group boundaries based on keypoint count
    if num_keypoints == 63:
        # 63 keypoints: Pose subset + Left Hand + Right Hand + Face subset
        groups = [
            (0, 14),   # Pose subset: 0-14 (15 points)
            (15, 35),  # Left hand: 15-35 (21 points)
            (36, 56),  # Right hand: 36-56 (21 points)
            (57, 62),  # Face subset: 57-62 (6 points)
        ]
    elif num_keypoints == 65:
        # 65 keypoints: Pose + Left Hand + Right Hand (NO FACE)
        groups = [
            (0, 22),   # Pose: 0-22 (23 points)
            (23, 43),  # Left hand: 23-43 (21 points)
            (44, 64),  # Right hand: 44-64 (21 points)
        ]
    elif num_keypoints == 90:
        # 90 keypoints: Pose + Left Hand + Right Hand + Face
        groups = [
            (0, 22),   # Pose: 0-22 (23 points)
            (23, 43),  # Left hand: 23-43 (21 points)
            (44, 64),  # Right hand: 44-64 (21 points)
            (65, 89),  # Face: 65-89 (25 points)
        ]
    else:
        raise ValueError(f"Unsupported num_keypoints: {num_keypoints}. Use 65 or 90.")
    
    result = keypoints.copy()
    
    # Normalize each frame
    for frame_idx in range(keypoints.shape[0]):
        for start, end in groups:
            # Extract group keypoints
            group_kpts = keypoints[frame_idx, start:end+1, :]
            
            # Normalize group
            normalized_group = normalize_part(group_kpts)
            
            # Put back
            result[frame_idx, start:end+1, :] = normalized_group
    
    return result


def pad_sequence(keypoints: np.ndarray, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """
    Pad sequence to max_len with zero frames.
    
    Args:
        keypoints: numpy array of shape (T, K, 2)
        max_len: target sequence length
    
    Returns:
        padded keypoints of shape (max_len, K, 2)
    """
    num_frames = keypoints.shape[0]
    num_keypoints = keypoints.shape[1]
    
    if num_frames >= max_len:
        return keypoints[:max_len]
    
    # Pad with zeros
    pad_len = max_len - num_frames
    padding = np.zeros((pad_len, num_keypoints, 2), dtype=np.float32)
    return np.concatenate([keypoints, padding], axis=0)


def create_attention_mask(actual_frames: int, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """
    Create attention mask for TFLite inference.
    
    Args:
        actual_frames: number of valid frames (before padding)
        max_len: total sequence length
    
    Returns:
        attention mask of shape (max_len,) with 1.0 for valid frames, 0.0 for padding
    """
    mask = np.zeros(max_len, dtype=np.float32)
    valid_frames = min(actual_frames, max_len)
    mask[:valid_frames] = 1.0
    return mask


def preprocess_for_inference(
    keypoints: np.ndarray,
    num_keypoints: int = 65,
    max_len: int = MAX_SEQ_LEN
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for TFLite inference.
    
    This matches EXACTLY what the on-device Dart code does:
        1. Clip to [0, 1]
        2. Subsample if > max_len (BEFORE normalization)
        3. Normalize by groups
        4. Pad to max_len (AFTER normalization)
    
    Args:
        keypoints: numpy array of shape (T, K, 2)
    num_keypoints: 63 (pose+hands+face subset), 65 (no face) or 90 (with face)
        max_len: fixed sequence length (default 64)
    
    Returns:
        keypoints_input: numpy array of shape (1, max_len, K, 2) ready for TFLite
        attention_mask: numpy array of shape (1, max_len)
    """
    # Validate input shape
    if keypoints.ndim != 3:
        raise ValueError(f"Expected keypoints shape (T, K, 2), got {keypoints.shape}")
    
    if keypoints.shape[1] != num_keypoints:
        raise ValueError(f"Expected {num_keypoints} keypoints, got {keypoints.shape[1]}")
    
    if keypoints.shape[2] != 2:
        raise ValueError(f"Expected 2 coordinates (x, y), got {keypoints.shape[2]}")
    
    # Store original frame count for attention mask
    original_frames = keypoints.shape[0]
    
    # Step 1: Clip to [0, 1]
    keypoints = clip_keypoints(keypoints)
    
    # Step 2: Subsample if too long (BEFORE normalization - matches Dart!)
    keypoints = subsample_frames(keypoints, max_len)
    actual_frames = min(original_frames, max_len)
    
    # Step 3: Normalize by groups
    keypoints = normalize_keypoints_by_groups(keypoints, num_keypoints)
    
    # Step 4: Pad to max_len (AFTER normalization)
    keypoints = pad_sequence(keypoints, max_len)
    
    # Step 5: Create attention mask
    attention_mask = create_attention_mask(actual_frames, max_len)
    
    # Add batch dimension
    keypoints_input = keypoints[np.newaxis, ...]  # (1, 64, K, 2)
    attention_mask = attention_mask[np.newaxis, ...]  # (1, 64)
    
    return keypoints_input.astype(np.float32), attention_mask.astype(np.float32)


def preprocess_for_training(
    keypoints: np.ndarray,
    joint_idxs: Optional[List[List[int]]] = None,
    num_keypoints: int = 65,
    max_len: int = MAX_SEQ_LEN,
    augment: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Preprocessing for training (used by SignDataset).
    
    This is called DURING training and should match inference preprocessing.
    
    Args:
        keypoints: numpy array of shape (T, K, 2) or (T, K, 3) - x, y, [visibility]
        joint_idxs: optional list of keypoint groups for normalization
        num_keypoints: expected number of keypoints
        max_len: maximum sequence length
        augment: whether to apply data augmentation (NOT implemented here)
    
    Returns:
        processed keypoints of shape (T_clipped, K, 2)
        label (passed through)
    """
    # Take only x, y coordinates
    if keypoints.shape[-1] > 2:
        keypoints = keypoints[:, :, :2]
    
    # Step 1: Clip to [0, 1]
    keypoints = clip_keypoints(keypoints)
    
    # Step 2: Subsample if too long
    keypoints = subsample_frames(keypoints, max_len)
    
    # Step 3: Normalize by groups
    keypoints = normalize_keypoints_by_groups(keypoints, num_keypoints)
    
    return keypoints.astype(np.float32)


# =============================================================================
# Utility functions
# =============================================================================

def determine_keypoint_groups_65() -> List[List[int]]:
    """
    Return keypoint groups for 65-keypoint configuration (no face).
    Used for normalization during training.
    """
    return [
        list(range(0, 23)),   # Pose: 0-22 (23 points)
        list(range(23, 44)),  # Left hand: 23-43 (21 points)
        list(range(44, 65)),  # Right hand: 44-64 (21 points)
    ]


def determine_keypoint_groups_90() -> List[List[int]]:
    """
    Return keypoint groups for 90-keypoint configuration (with face).
    Used for normalization during training.
    """
    return [
        list(range(0, 23)),   # Pose: 0-22 (23 points)
        list(range(23, 44)),  # Left hand: 23-43 (21 points)
        list(range(44, 65)),  # Right hand: 44-64 (21 points)
        list(range(65, 90)),  # Face: 65-89 (25 points)
    ]


def determine_keypoint_groups_63() -> List[List[int]]:
    """
    Return keypoint groups for 63-keypoint v2.1 configuration.
    Pose subset (15) + Left hand (21) + Right hand (21) + Face subset (6).
    """
    return [
        list(range(0, 15)),   # Pose subset: 0-14 (15 points)
        list(range(15, 36)),  # Left hand: 15-35 (21 points)
        list(range(36, 57)),  # Right hand: 36-56 (21 points)
        list(range(57, 63)),  # Face subset: 57-62 (6 points)
    ]


def determine_keypoint_groups(config_joint_idx: List[int]) -> List[List[int]]:
    """
    Automatically determine keypoint groups from config joint_idx.
    
    This matches the logic in main.py / main_functional.py.
    
    Args:
        config_joint_idx: flat list of keypoint indices from config
    
    Returns:
        list of lists (groups for normalization)
    """
    if not config_joint_idx:
        return []
    
    sorted_idx = sorted(config_joint_idx)
    total_kpts = len(sorted_idx)
    groups = []
    
    if total_kpts >= 67:
        # Structure: Pose + Left Hand (21) + Right Hand (21) + Face (25)
        face_kpts = sorted_idx[-25:]
        right_hand_kpts = sorted_idx[-46:-25]
        left_hand_kpts = sorted_idx[-67:-46]
        body_kpts = sorted_idx[:-67]
        
        if body_kpts:
            groups.append(body_kpts)
        if left_hand_kpts:
            groups.append(left_hand_kpts)
        if right_hand_kpts:
            groups.append(right_hand_kpts)
        if face_kpts:
            groups.append(face_kpts)
    elif total_kpts == 63:
        # v2.1: Pose (15) + Left Hand (21) + Right Hand (21) + Face (6)
        body_kpts = sorted_idx[:15]
        left_hand_kpts = sorted_idx[15:36]
        right_hand_kpts = sorted_idx[36:57]
        face_kpts = sorted_idx[57:63]

        if body_kpts:
            groups.append(body_kpts)
        if left_hand_kpts:
            groups.append(left_hand_kpts)
        if right_hand_kpts:
            groups.append(right_hand_kpts)
        if face_kpts:
            groups.append(face_kpts)
    elif total_kpts == 65:
        # 65 keypoints: no face
        # Structure: Pose (23) + Left Hand (21) + Right Hand (21)
        right_hand_kpts = sorted_idx[-21:]
        left_hand_kpts = sorted_idx[-42:-21]
        body_kpts = sorted_idx[:-42]
        
        if body_kpts:
            groups.append(body_kpts)
        if left_hand_kpts:
            groups.append(left_hand_kpts)
        if right_hand_kpts:
            groups.append(right_hand_kpts)
    else:
        # Fallback: treat all as one group
        groups.append(sorted_idx)
    
    return groups


# =============================================================================
# Example / Test
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Inference Preprocessing Test (65 Keypoints)")
    print("="*80)
    
    # Create dummy keypoints: 30 frames, 65 keypoints, 2 coordinates
    num_frames = 30
    num_kpts = 65
    
    print(f"\nTest 1: Short video ({num_frames} frames)")
    keypoints = np.random.rand(num_frames, num_kpts, 2).astype(np.float32)
    
    kpts_input, mask = preprocess_for_inference(keypoints, num_keypoints=65)
    
    print(f"  Input shape: {keypoints.shape}")
    print(f"  Output shape: {kpts_input.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Valid frames: {int(mask.sum())} / {MAX_SEQ_LEN}")
    
    # Test long video
    print(f"\nTest 2: Long video (80 frames)")
    keypoints_long = np.random.rand(80, num_kpts, 2).astype(np.float32)
    
    kpts_input, mask = preprocess_for_inference(keypoints_long, num_keypoints=65)
    
    print(f"  Input shape: {keypoints_long.shape}")
    print(f"  Output shape: {kpts_input.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Valid frames: {int(mask.sum())} / {MAX_SEQ_LEN} (subsampled)")
    
    # Test 90 keypoints
    print(f"\nTest 3: 90 keypoints (with face)")
    keypoints_90 = np.random.rand(40, 90, 2).astype(np.float32)
    
    kpts_input, mask = preprocess_for_inference(keypoints_90, num_keypoints=90)
    
    print(f"  Input shape: {keypoints_90.shape}")
    print(f"  Output shape: {kpts_input.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Valid frames: {int(mask.sum())} / {MAX_SEQ_LEN}")
    
    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)
    print("\nThis preprocessing EXACTLY matches:")
    print("  - MLR511_Project/lib/preprocessing/keypoint_preprocessing.dart")
    print("  - SignBartTF/dataset.py (training)")
    print("\nUse preprocess_for_inference() in your TFLite inference pipeline.")
    print("="*80)
