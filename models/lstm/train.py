"""
Training script for e-LSTM model on Arabic Sign Language dataset

Usage:
    python train.py --data_dir ../../data/processed --config config.yaml
"""

import argparse
import pickle
import numpy as np
import yaml
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from datetime import datetime
import random
import os

from model_bilstm import build_elstm_model, compile_model  # Using improved BiLSTM model
from augmentations import apply_augmentations


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Note: NOT forcing complete GPU determinism
    # Full determinism can hurt performance by restricting optimization paths
    # We keep data/augmentation seeded but allow GPU flexibility
    
    print(f"🌱 Random seed set to {seed} for reproducibility")
    print(f"⚡ GPU operations: Non-deterministic (allows better optimization)")


class KeypointDataLoader:
    """Load and preprocess keypoint data from .pkl files"""
    
    def __init__(self, data_dir, seq_len=64, num_keypoints=90, config=None, augment=False):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.num_keypoints = num_keypoints
        self.augment = augment
        
        # Grouped normalization configuration (90 keypoints)
        # Matches arabic-asl-new.yaml
        if config and 'normalization' in config:
            self.norm_groups = config['normalization']['groups']
        else:
            # Default groups for 90 keypoints
            self.norm_groups = [
                {'name': 'pose', 'indices': list(range(0, 23))},        # 0-22: upper body pose
                {'name': 'left_hand', 'indices': list(range(23, 44))},  # 23-43: left hand
                {'name': 'right_hand', 'indices': list(range(44, 65))}, # 44-64: right hand
                {'name': 'face', 'indices': list(range(65, 90))}        # 65-89: face
            ]
    
    def load_data(self):
        """
        Load all .pkl files from data directory.
        Handles LOSO split structure: data_dir/train/G01/*.pkl, G02/*.pkl, etc.
        
        Returns:
            X: Array of shape (num_samples, seq_len, 90, 2)
            y: Array of labels (num_samples,)
            label_to_id: Dictionary mapping class names to indices
        """
        # Build label mapping (G01 -> 0, G02 -> 1, ..., G10 -> 9)
        gesture_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('G')])
        label_to_id = {gesture.name: idx for idx, gesture in enumerate(gesture_dirs)}
        
        print(f"Found {len(gesture_dirs)} gesture classes: {list(label_to_id.keys())}")
        
        # Collect all .pkl files from all gesture directories
        pkl_files = []
        for gesture_dir in gesture_dirs:
            pkl_files.extend(list(gesture_dir.glob('*.pkl')))
        
        print(f"Found {len(pkl_files)} total .pkl files")
        
        X_list = []
        y_list = []
        
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract keypoints (frames, 90, 3) where 3 = [x, y, visibility]
            keypoints = data['keypoints'][:, :, :2]  # Take only x, y (drop visibility)
            
            # Get class label (e.g., "G01") and convert to numeric ID
            gesture_class = data['class']
            label = label_to_id[gesture_class]
            
            # Clip to max 64 frames if longer (before augmentation)
            if keypoints.shape[0] > 64:
                # Uniform sampling like SignBart
                indices = np.clip(np.linspace(0, keypoints.shape[0] - 1, 64).astype(int), 0, keypoints.shape[0] - 1)
                keypoints = keypoints[indices]
            
            # Apply augmentation (before normalization, like SignBart)
            if self.augment:
                keypoints = apply_augmentations(keypoints, augment_prob=0.4)
            
            # Pad or trim to fixed sequence length
            keypoints = self._pad_or_trim(keypoints)
            
            # Apply grouped normalization
            keypoints = self._normalize(keypoints)
            
            X_list.append(keypoints)
            y_list.append(label)
        
        X = np.array(X_list)  # (num_samples, seq_len, 90, 2)
        y = np.array(y_list)  # (num_samples,)
        
        print(f"Loaded data: X shape = {X.shape}, y shape = {y.shape}")
        print(f"Label mapping: {label_to_id}")
        
        return X, y
    
    def _pad_or_trim(self, sequence):
        """Pad with zeros or trim to fixed sequence length"""
        current_len = sequence.shape[0]
        
        if current_len < self.seq_len:
            # Pad with zeros
            padding = np.zeros((self.seq_len - current_len, self.num_keypoints, 2))
            sequence = np.vstack([sequence, padding])
        elif current_len > self.seq_len:
            # Trim to seq_len
            sequence = sequence[:self.seq_len]
        
        return sequence
    
    def _normalize(self, keypoints):
        """
        Apply GROUPED normalization to keypoints.
        Each group (pose, left_hand, right_hand, face) is normalized independently
        based on its own bounding box.
        
        This matches SignBart's normalization strategy and preserves spatial
        relationships within each body part.
        
        Args:
            keypoints: numpy array of shape (T, 90, 2) - sequence of keypoints
        
        Returns:
            normalized keypoints of same shape
        """
        normalized = keypoints.copy()
        
        # Normalize each group independently
        for group in self.norm_groups:
            indices = group['indices']
            group_kpts = keypoints[:, indices, :]  # (T, num_keypoints_in_group, 2)
            
            # Get valid (non-zero) keypoints in this group across all frames
            valid_mask = (group_kpts != 0).any(axis=-1)  # (T, num_keypoints_in_group)
            
            if valid_mask.any():
                valid_kpts = group_kpts[valid_mask]  # (num_valid, 2)
                
                # Compute bounding box for this group
                min_x, min_y = valid_kpts[:, 0].min(), valid_kpts[:, 1].min()
                max_x, max_y = valid_kpts[:, 0].max(), valid_kpts[:, 1].max()
                
                w = max_x - min_x
                h = max_y - min_y
                
                # Add padding to bounding box (same as SignBart)
                if w > h:
                    delta_x = 0.05 * w
                    delta_y = delta_x + ((w - h) / 2)
                else:
                    delta_y = 0.05 * h
                    delta_x = delta_y + ((h - w) / 2)
                
                s_point_x = max(0, min(min_x - delta_x, 1))
                s_point_y = max(0, min(min_y - delta_y, 1))
                e_point_x = max(0, min(max_x + delta_x, 1))
                e_point_y = max(0, min(max_y + delta_y, 1))
                
                # Normalize this group to [0, 1] based on its bounding box
                if (e_point_x - s_point_x) > 0:
                    normalized[:, indices, 0] = (group_kpts[:, :, 0] - s_point_x) / (e_point_x - s_point_x)
                if (e_point_y - s_point_y) > 0:
                    normalized[:, indices, 1] = (group_kpts[:, :, 1] - s_point_y) / (e_point_y - s_point_y)
        
        return normalized


def train_elstm(config_path, data_dir, output_dir, test_dir=None, seed=42, no_validation=False):
    """
    Train e-LSTM model with LOSO setup
    
    Args:
        config_path: Path to config YAML file
        data_dir: Directory containing training .pkl files (e.g., LOSO_user01/train)
        output_dir: Directory to save model checkpoints and logs
        test_dir: Directory containing test .pkl files (e.g., LOSO_user01/test)
        seed: Random seed for reproducibility
        no_validation: If True, don't use test_dir for validation during training (only evaluate at end)
    """
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Check GPU availability
    print("="*80)
    print("GPU/Device Information")
    print("="*80)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ {len(gpus)} GPU(s) detected:")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        print(f"   TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"   GPU available: {tf.test.is_gpu_available()}" if hasattr(tf.test, 'is_gpu_available') else "")
    else:
        print("⚠️  No GPU detected - training will use CPU")
        print("   This will be significantly slower")
    print("="*80 + "\n")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract hyperparameters
    num_classes = config['model']['num_classes']
    seq_len = config['model']['seq_len']
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    learning_rate = config['training']['learning_rate']
    
    print("="*80)
    print("e-LSTM Training Configuration")
    print("="*80)
    print(f"Num Classes: {num_classes}")
    print(f"Sequence Length: {seq_len}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Random Seed: {seed}")
    print("="*80 + "\n")
    
    # Check if augmentation is enabled
    augment_enabled = config.get('data', {}).get('augmentation', {}).get('enabled', False)
    
    # Load training data WITH augmentation
    print("Loading training data...")
    print(f"  Augmentation: {'ENABLED (40% prob)' if augment_enabled else 'DISABLED'}")
    train_loader = KeypointDataLoader(data_dir, seq_len=seq_len, config=config, augment=augment_enabled)
    X_train, y_train = train_loader.load_data()
    
    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, num_classes=num_classes)
    
    print(f"Training data: X shape = {X_train.shape}, y shape = {y_train.shape}")
    print(f"Train: {X_train.shape[0]} samples")
    
    # Load test data
    if test_dir:
        print(f"\nLoading test data...")
        test_loader = KeypointDataLoader(test_dir, seq_len=seq_len, config=config, augment=False)
        X_test, y_test = test_loader.load_data()
        y_test_cat = keras.utils.to_categorical(y_test, num_classes=num_classes)
        print(f"Test data: X shape = {X_test.shape}, y shape = {y_test.shape}")
        print(f"Test: {X_test.shape[0]} samples")
        
        if no_validation:
            print("  ⚠️  --no_validation flag set: Test data will NOT be used for validation during training")
            print("     (Will only evaluate at the end)\n")
            X_val, y_val_cat = None, None
        else:
            print("  ℹ️  Test data WILL be used for validation during training (early stopping, etc.)\n")
            X_val, y_val_cat = X_test, y_test_cat
    else:
        print("No test directory provided - training without validation\n")
        X_val, y_val_cat = None, None
        X_test, y_test_cat = None, None
    
    # Build model
    print("Building e-LSTM model...")
    arch = config['model']['architecture']
    model = build_elstm_model(
        num_classes=num_classes,
        seq_len=seq_len,
        num_keypoints=arch['num_keypoints'],
        coords_per_keypoint=arch['coords_per_keypoint'],
        lstm1_units=arch['lstm1_units'],
        lstm2_units=arch['lstm2_units'],
        attention_units=arch['attention_units'],
        fc_units=arch['fc_units'],
        dropout_rate=arch['dropout_rate']
    )
    
    # Compile with label smoothing and gradient clipping
    model = compile_model(
        model, 
        learning_rate=learning_rate,
        label_smoothing=0.1,  # Helps with generalization
        clipnorm=1.0          # Prevents exploding gradients
    )
    model.summary()
    
    # Setup callbacks
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup callbacks (adjust based on validation availability)
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=str(output_path / f'logs/{timestamp}'),
            histogram_freq=1
        ),
        keras.callbacks.CSVLogger(
            str(output_path / f'logs/training_{timestamp}.csv')
        )
    ]
    
    # Determine if we should use validation-based callbacks
    use_validation_callbacks = (X_val is not None) and not no_validation
    
    if use_validation_callbacks:
        # With validation: use validation-based callbacks (early stopping, best model selection)
        print("Using validation-based callbacks (early stopping, best model checkpoint)")
        callbacks.extend([
            keras.callbacks.ModelCheckpoint(
                filepath=str(output_path / f'checkpoints/elstm_{timestamp}_epoch{{epoch:02d}}_val{{val_accuracy:.4f}}.h5'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config['training']['early_stopping']['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=config['training']['reduce_lr']['factor'],
                patience=config['training']['reduce_lr']['patience'],
                min_lr=config['training']['reduce_lr']['min_lr'],
                verbose=1
            )
        ])
    else:
        # Without validation: save checkpoints periodically (no early stopping, no val monitoring)
        print("No validation callbacks - training for full epochs, saving periodically")
        callbacks.insert(0, keras.callbacks.ModelCheckpoint(
            filepath=str(output_path / f'checkpoints/elstm_{timestamp}_epoch{{epoch:02d}}.h5'),
            save_freq=10 * len(X_train) // batch_size,  # Every 10 epochs
            verbose=1
        ))
    
    # Train
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80 + "\n")
    
    history = model.fit(
        X_train, y_train_cat,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val_cat) if test_dir else None,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = output_path / f'checkpoints/elstm_final_{timestamp}.h5'
    model.save(final_model_path)
    print(f"\n✅ Final model saved to {final_model_path}")
    
    # Note: Model can now be loaded without custom_objects due to @register_keras_serializable
    # loaded_model = keras.models.load_model(final_model_path)
    
    # For TFLite conversion:
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    # with open('elstm_model.tflite', 'wb') as f:
    #     f.write(tflite_model)
    
    # Evaluate on test set (if available)
    if test_dir and X_test is not None:
        print("\n" + "="*80)
        print("Final Evaluation on Test Set (LOSO holdout)")
        print("="*80)
        if no_validation:
            print("✅ Proper LOSO: Model did NOT see test data during training")
        else:
            print("⚠️  Model used test data for validation during training (early stopping, etc.)")
        print("="*80)
        test_loss, test_acc, test_top5 = model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Top-5 Accuracy: {test_top5:.4f}")
    else:
        print("\n" + "="*80)
        print("Training completed (no test set provided)")
        print("="*80)
        print("Provide --test_dir to evaluate on LOSO holdout user.")
    
    # Save training history
    history_path = output_path / f'results/training_history_{timestamp}.pkl'
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"\n✅ Training history saved to {history_path}")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train e-LSTM model with LOSO")
    parser.add_argument("--data_dir", required=True, help="Training directory (e.g., data/arabic-asl-90kpts_LOSO_user01/train)")
    parser.add_argument("--test_dir", default=None, help="Test directory to use as validation (e.g., data/arabic-asl-90kpts_LOSO_user01/test)")
    parser.add_argument("--config", default="config.yaml", help="Config YAML file")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory for checkpoints/logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no_validation", action="store_true", help="Disable validation during training (only evaluate at end)")
    
    args = parser.parse_args()
    
    # Pass no_validation flag to training function
    train_elstm(args.config, args.data_dir, args.output_dir, args.test_dir, args.seed, args.no_validation)

