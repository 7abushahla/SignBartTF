"""
TensorFlow/Keras dataset implementation for SignBART.
Converted from PyTorch implementation.
"""
import json
import pickle
import numpy as np
import tensorflow as tf
import os
from augmentations import (
    rotate_keypoints, noise_injection, clip_frame, 
    time_warp_uniform, flip_keypoints
)


class SignDataset:
    """
    Sign language dataset for TensorFlow.
    Uses tf.data.Dataset for efficient data loading.
    """
    def __init__(self, root, split, shuffle=True, joint_idxs=None, augment=False):
        self.root = root
        self.split = split
        self.joint_idxs = joint_idxs
        self.augment = augment
        self.shuffle = shuffle
        
        # Flatten joint indices for filtering
        if joint_idxs is not None:
            self.flat_joint_idxs = []
            for group in joint_idxs:
                self.flat_joint_idxs.extend(group)
            self.flat_joint_idxs = sorted(self.flat_joint_idxs)
        else:
            self.flat_joint_idxs = None
        
        # Load label mappings
        with open(f"{self.root}/label2id.json", 'r') as f:
            self.label2id = json.load(f)
        
        with open(f"{self.root}/id2label.json", 'r') as f:
            self.id2label = json.load(f)
        
        self.data_dir = f"{root}/{split}"
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Split directory not found: {self.data_dir}")

        # Get all pickle paths under: {root}/{split}/{class_dir}/*.pkl
        # Be defensive against stray files like .DS_Store at any level.
        list_key = []
        for x in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, x)
            if not os.path.isdir(class_dir):
                continue
            for y in os.listdir(class_dir):
                if not y.endswith(".pkl"):
                    continue
                p = os.path.join(class_dir, y)
                if os.path.isfile(p):
                    list_key.append(p)
        self.list_key = list_key
        
        if shuffle:
            np.random.shuffle(self.list_key)
    
    def __len__(self):
        return len(self.list_key)
    
    def load_sample(self, file_path):
        """
        Load a single sample from pickle file.
        
        Args:
            file_path: path to pickle file (as bytes for tf.py_function)
        
        Returns:
            keypoints: (T, K, 2) array
            label: int
        """
        # If file_path is a tensor (from tf.data), convert to string
        if hasattr(file_path, 'numpy'):
            file_path = file_path.numpy().decode('utf-8')
        elif isinstance(file_path, bytes):
            file_path = file_path.decode('utf-8')
        
        with open(file_path, "rb") as f:
            sample = pickle.load(f)
        
        keypoints = np.array(sample['keypoints'])[:, :, :2]  # Only x, y coordinates
        
        # Get label
        class_name = sample['class']
        label = self.label2id[class_name]
        
        # Filter keypoints to only include specified indices
        if self.flat_joint_idxs is not None:
            keypoints = keypoints[:, self.flat_joint_idxs, :]
        
        # Clip values to [0, 1]
        keypoints = np.clip(keypoints, 0, 1)
        
        # Clip to max 64 frames
        if keypoints.shape[0] > 64:
            keypoints = clip_frame(keypoints, 64, True)
        
        # Apply augmentation
        if self.augment and np.random.uniform(0, 1) < 0.4:
            keypoints = self.apply_augment(keypoints)
        
        # Normalize keypoints
        keypoints = self.normalize_keypoints(keypoints)
        
        return keypoints.astype(np.float32), np.int32(label)
    
    def apply_augment(self, keypoints):
        """Apply data augmentation to keypoints."""
        aug = False
        while not aug:
            if np.random.uniform(0, 1) < 0.5:
                angle = np.random.uniform(-15, 15)
                keypoints = rotate_keypoints(keypoints, (0, 0), angle)
                aug = True
            if np.random.uniform(0, 1) < 0.5:
                random_noise = np.random.uniform(0.01, 0.2)
                keypoints = noise_injection(keypoints, random_noise)
                aug = True
            if np.random.uniform(0, 1) < 0.5:
                n_f = keypoints.shape[0]
                tgt = np.random.randint(n_f // 2, n_f)
                is_uniform = np.random.uniform(0, 1) < 0.5
                keypoints = clip_frame(keypoints, tgt, is_uniform)
                aug = True
            if np.random.uniform(0, 1) < 0.5:
                speed = np.random.uniform(1.1, 1.5)
                keypoints = time_warp_uniform(keypoints, speed)
                aug = True
            if np.random.uniform(0, 1) < 0.5:
                keypoints = flip_keypoints(keypoints)
                aug = True
        
        return keypoints
    
    def normalize_keypoints(self, keypoints):
        """
        Normalize keypoints by groups.
        
        Args:
            keypoints: numpy array of shape (T, K_filtered, 2)
        
        Returns:
            normalized keypoints of same shape
        """
        if self.joint_idxs is None:
            return keypoints
        
        # Create mapping from original indices to filtered positions
        if self.flat_joint_idxs is not None:
            idx_to_pos = {idx: pos for pos, idx in enumerate(self.flat_joint_idxs)}
        
        # Normalize each group
        for i in range(keypoints.shape[0]):  # for each frame
            for group in self.joint_idxs:  # for each group
                # Map original indices to positions in filtered array
                if self.flat_joint_idxs is not None:
                    filtered_positions = [idx_to_pos[idx] for idx in group if idx in idx_to_pos]
                else:
                    filtered_positions = group
                
                if len(filtered_positions) > 0:
                    # Get keypoints for this group
                    group_keypoints = keypoints[i, filtered_positions, :]
                    # Normalize the group
                    normalized = self._normalize_part(group_keypoints)
                    # Put back
                    keypoints[i, filtered_positions, :] = normalized
        
        return keypoints
    
    @staticmethod
    def _normalize_part(keypoint):
        """
        Normalize a group of keypoints to [0, 1] range based on their bounding box.
        
        Args:
            keypoint: numpy array of shape (N, 2)
        
        Returns:
            normalized keypoints of same shape
        """
        assert keypoint.shape[-1] == 2, "Keypoints must have x, y"
        
        # Ignore missing keypoints (0, 0) to match on-device preprocessing
        valid_mask = (keypoint[:, 0] != 0.0) | (keypoint[:, 1] != 0.0)
        valid_kpts = keypoint[valid_mask]
        if valid_kpts.size == 0:
            return keypoint
        
        x_coords = valid_kpts[:, 0]
        y_coords = valid_kpts[:, 1]
        
        min_x, min_y = np.min(x_coords), np.min(y_coords)
        max_x, max_y = np.max(x_coords), np.max(y_coords)
        
        w = max_x - min_x
        h = max_y - min_y
        
        # Add padding to bounding box
        if w > h:
            delta_x = 0.05 * w
            delta_y = delta_x + ((w - h) / 2)
        else:
            delta_y = 0.05 * h
            delta_x = delta_y + ((h - w) / 2)
        
        s_point = [max(0, min(min_x - delta_x, 1)), max(0, min(min_y - delta_y, 1))]
        e_point = [max(0, min(max_x + delta_x, 1)), max(0, min(max_y + delta_y, 1))]
        
        # Normalize keypoints (only for valid points)
        result = keypoint.copy()
        if (e_point[0] - s_point[0]) != 0.0:
            result[valid_mask, 0] = (keypoint[valid_mask, 0] - s_point[0]) / (e_point[0] - s_point[0])
        if (e_point[1] - s_point[1]) != 0.0:
            result[valid_mask, 1] = (keypoint[valid_mask, 1] - s_point[1]) / (e_point[1] - s_point[1])
        
        return result
    
    def create_tf_dataset(self, batch_size, drop_remainder=False):
        """
        Create a tf.data.Dataset from the file list.
        
        Args:
            batch_size: batch size
            drop_remainder: whether to drop last incomplete batch
        
        Returns:
            tf.data.Dataset
        """
        print(f"Creating dataset with {len(self.list_key)} samples, batch_size={batch_size}")
        
        # Create dataset from file paths
        dataset = tf.data.Dataset.from_tensor_slices(self.list_key)
        
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.list_key))
        
        # Load samples - Use single-threaded processing to avoid deadlock with tf.py_function
        def load_sample_wrapper(file_path):
            keypoints, label = tf.py_function(
                func=self.load_sample,
                inp=[file_path],
                Tout=[tf.float32, tf.int32]
            )
            # Set shapes explicitly to help TensorFlow
            keypoints.set_shape([None, None, 2])
            label.set_shape([])
            return keypoints, label
        
        dataset = dataset.map(
            load_sample_wrapper,
            num_parallel_calls=1  # Use single thread to avoid tf.py_function deadlock
        )
        
        # Batch with padding
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([None, None, 2], []),  # Pad sequences to max length in batch
            padding_values=(0.0, 0),
            drop_remainder=drop_remainder
        )
        
        # Add attention masks and format
        dataset = dataset.map(
            self.add_attention_masks,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Prefetch for performance
        dataset = dataset.prefetch(buffer_size=2)
        
        print(f"✓ Dataset created successfully")
        
        return dataset
    
    @staticmethod
    def add_attention_masks(keypoints, labels):
        """
        Add attention masks to batch.
        
        Args:
            keypoints: (batch_size, max_seq_len, num_joints, 2)
            labels: (batch_size,)
        
        Returns:
            dict with 'keypoints', 'attention_mask', 'labels'
        """
        # Create attention mask based on non-zero keypoints
        # Check if all coordinates are zero (padding)
        mask = tf.reduce_any(
            tf.not_equal(keypoints, 0.0),
            axis=[2, 3]  # Check across joints and coordinates
        )
        mask = tf.cast(mask, tf.float32)
        
        return {
            'keypoints': keypoints,
            'attention_mask': mask,
            'labels': labels
        }


def create_data_loaders(root, batch_size, joint_idxs=None, augment_train=True):
    """
    Create train and validation data loaders.
    
    Args:
        root: root directory of dataset
        batch_size: batch size
        joint_idxs: joint indices configuration
        augment_train: whether to apply augmentation to training set
    
    Returns:
        train_dataset, val_dataset: TensorFlow datasets
    """
    train_ds = SignDataset(
        root=root,
        split='train',
        shuffle=True,
        joint_idxs=joint_idxs,
        augment=augment_train
    )
    
    val_ds = SignDataset(
        root=root,
        split='val',
        shuffle=False,
        joint_idxs=joint_idxs,
        augment=False
    )
    
    train_dataset = train_ds.create_tf_dataset(batch_size, drop_remainder=False)
    val_dataset = val_ds.create_tf_dataset(batch_size, drop_remainder=False)
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    print("Testing TensorFlow SignBART Dataset...")
    
    # Note: You'll need to adjust the path to your actual dataset
    # This is just for demonstration
    
    # Example usage:
    # from utils import full_100_groups
    # 
    # dataset = SignDataset(
    #     root="/path/to/dataset",
    #     split="train",
    #     shuffle=True,
    #     joint_idxs=full_100_groups,
    #     augment=True
    # )
    # 
    # tf_dataset = dataset.create_tf_dataset(batch_size=8)
    # 
    # # Iterate
    # for batch in tf_dataset.take(1):
    #     print(f"Keypoints shape: {batch['keypoints'].shape}")
    #     print(f"Attention mask shape: {batch['attention_mask'].shape}")
    #     print(f"Labels shape: {batch['labels'].shape}")
    
    print("\n✓ Dataset module created!")
    print("   Note: Actual testing requires a dataset at the specified path.")

