"""
Data augmentation functions for keypoint sequences.
Matches SignBart augmentation strategies.
"""
import numpy as np


def rotate_keypoints(frames, origin, angle_degrees):
    """
    Rotate keypoints around origin by angle_degrees.
    
    Args:
        frames: (T, K, 2) array of keypoints
        origin: (x, y) rotation center
        angle_degrees: rotation angle in degrees (±15)
    
    Returns:
        rotated keypoints
    """
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    
    shifted_points = frames - np.array(origin)
    rotated_points = np.einsum('ij,klj->kli', rotation_matrix, shifted_points)
    rotated_frames = rotated_points + np.array(origin)
    
    return rotated_frames


def noise_injection(frames, noise_level):
    """
    Add Gaussian noise to keypoints.
    
    Args:
        frames: (T, K, 2) array of keypoints
        noise_level: standard deviation of noise (σ ∈ [0.01, 0.2])
    
    Returns:
        noisy keypoints
    """
    noise = np.random.normal(0, noise_level, frames.shape)
    noisy_frames = frames + noise
    return noisy_frames


def clip_frame(frames, tgt_frame, is_uniform):
    """
    Temporal clipping: select subset of frames.
    
    Args:
        frames: (T, K, 2) array of keypoints
        tgt_frame: target number of frames (50-100% of original)
        is_uniform: if True, uniformly sample; if False, randomly sample
    
    Returns:
        clipped keypoints
    """
    t = frames.shape[0]
    if is_uniform:
        indices = np.clip(np.linspace(0, t - 1, tgt_frame).astype(int), 0, t - 1)
    else:
        indices = np.sort(np.random.choice(np.arange(t), size=tgt_frame, replace=True))
    selected_frames = frames[indices]
    return selected_frames


def time_warp_uniform(frames, scale):
    """
    Time warping: speed up/slow down the sequence.
    
    Args:
        frames: (T, K, 2) array of keypoints
        scale: speed factor (speed ∈ [1.1, 1.5] = faster)
    
    Returns:
        time-warped keypoints
    """
    t = frames.shape[0]
    new_t = int(t * scale)
    indices = np.clip(np.linspace(0, t - 1, new_t).astype(int), 0, t - 1)
    warped_frames = frames[indices]
    return warped_frames


def flip_keypoints(keypoints):
    """
    Horizontal flip: mirror keypoints along x-axis.
    
    Args:
        keypoints: (T, K, 2) array of keypoints
    
    Returns:
        flipped keypoints
    """
    flipped_keypoints = keypoints.copy()
    flipped_keypoints[..., 0] = 1 - flipped_keypoints[..., 0]
    return flipped_keypoints


def apply_augmentations(keypoints, augment_prob=0.4):
    """
    Apply random augmentations to keypoint sequence.
    Matches SignBart augmentation strategy.
    
    Args:
        keypoints: (T, K, 2) array of keypoints
        augment_prob: probability of applying augmentation (default: 0.4)
    
    Returns:
        augmented keypoints
    """
    if np.random.uniform(0, 1) > augment_prob:
        return keypoints
    
    aug = False
    while not aug:
        # Random rotation: ±15 degrees
        if np.random.uniform(0, 1) < 0.5:
            angle = np.random.uniform(-15, 15)
            keypoints = rotate_keypoints(keypoints, (0.5, 0.5), angle)
            aug = True
        
        # Gaussian noise injection: σ ∈ [0.01, 0.2]
        if np.random.uniform(0, 1) < 0.5:
            random_noise = np.random.uniform(0.01, 0.2)
            keypoints = noise_injection(keypoints, random_noise)
            aug = True
        
        # Temporal clipping: 50-100% of frames
        if np.random.uniform(0, 1) < 0.5:
            n_f = keypoints.shape[0]
            tgt = np.random.randint(n_f // 2, n_f)
            is_uniform = np.random.uniform(0, 1) < 0.5
            keypoints = clip_frame(keypoints, tgt, is_uniform)
            aug = True
        
        # Time warping: speed ∈ [1.1, 1.5]
        if np.random.uniform(0, 1) < 0.5:
            speed = np.random.uniform(1.1, 1.5)
            keypoints = time_warp_uniform(keypoints, speed)
            aug = True
        
        # Horizontal flip
        if np.random.uniform(0, 1) < 0.5:
            keypoints = flip_keypoints(keypoints)
            aug = True
    
    return keypoints

