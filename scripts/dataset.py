"""
Dataset loader for Arabic Sign Language Recognition
Handles keypoint data with grouped normalization
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple


class ArabicASLDataset:
    """
    Dataset class for Arabic Sign Language with 90 keypoints per frame
    
    Keypoint structure:
    - 23 pose (upper body)
    - 21 left hand
    - 21 right hand
    - 25 face (symmetric selection)
    """
    
    def __init__(self, config_path: str, data_dir: str):
        """
        Initialize dataset
        
        Args:
            config_path: Path to arabic-asl.yaml config
            data_dir: Directory containing processed keypoint files
        """
        self.config = self._load_config(config_path)
        self.data_dir = Path(data_dir)
        
        # Extract keypoint configuration
        self.total_keypoints = self.config['keypoints']['total']
        self.pose_indices = self.config['keypoints']['pose']['indices']
        self.left_hand_indices = self.config['keypoints']['left_hand']['indices']
        self.right_hand_indices = self.config['keypoints']['right_hand']['indices']
        self.face_indices = self.config['keypoints']['face']['indices']
        
        # Normalization groups
        self.norm_groups = self.config['normalization']['groups']
        
        print(f"✅ Dataset initialized: {self.total_keypoints} keypoints per frame")
        print(f"   Pose: {len(self.pose_indices)}, Left Hand: {len(self.left_hand_indices)}, "
              f"Right Hand: {len(self.right_hand_indices)}, Face: {len(self.face_indices)}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Apply grouped normalization to keypoints
        
        Args:
            keypoints: Array of shape (num_frames, 90, 2) - [x, y] coordinates
        
        Returns:
            Normalized keypoints of same shape
        """
        normalized = keypoints.copy()
        
        for group in self.norm_groups:
            indices = group['indices']
            group_kpts = keypoints[:, indices, :]  # (frames, group_size, 2)
            
            # Compute bounding box for this group
            valid_mask = (group_kpts != 0).any(axis=-1)  # (frames, group_size)
            
            if valid_mask.any():
                # Get min/max for valid keypoints
                valid_kpts = group_kpts[valid_mask]
                x_min, y_min = valid_kpts.min(axis=0)
                x_max, y_max = valid_kpts.max(axis=0)
                
                # Normalize to [0, 1] within bounding box
                width = x_max - x_min
                height = y_max - y_min
                
                if width > 0 and height > 0:
                    normalized[:, indices, 0] = (group_kpts[:, :, 0] - x_min) / width
                    normalized[:, indices, 1] = (group_kpts[:, :, 1] - y_min) / height
        
        return normalized
    
    def load_sample(self, video_path: str) -> Tuple[np.ndarray, int]:
        """
        Load a single video sample
        
        Args:
            video_path: Path to .npy keypoint file
        
        Returns:
            keypoints: (num_frames, 90, 2)
            label: class label
        """
        # Load keypoints
        keypoints = np.load(video_path)  # (frames, 90, 2)
        
        # Extract label from filename (adjust based on your naming scheme)
        label = int(Path(video_path).stem.split('_')[0])
        
        # Apply normalization
        keypoints = self.normalize_keypoints(keypoints)
        
        return keypoints, label
    
    def __len__(self) -> int:
        """Return number of samples"""
        return len(list(self.data_dir.glob('*.npy')))
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get a single sample"""
        files = sorted(list(self.data_dir.glob('*.npy')))
        return self.load_sample(str(files[idx]))


if __name__ == "__main__":
    # Example usage
    config_path = "../configs/arabic-asl.yaml"
    data_dir = "../data/processed/"
    
    dataset = ArabicASLDataset(config_path, data_dir)
    print(f"Dataset size: {len(dataset)} samples")

