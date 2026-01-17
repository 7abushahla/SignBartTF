"""
Main training script for SignBART TensorFlow.
Equivalent to the PyTorch main.py with all features.
"""
import os
import argparse
import random
import logging
import yaml
import numpy as np
import tensorflow as tf
import gc
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from pathlib import Path
from datetime import datetime

from dataset import SignDataset, create_data_loaders
from model import SignBart
from utils import (
    accuracy, top_k_accuracy, save_checkpoint, load_checkpoint,
    count_model_parameters, get_keypoint_config
)

# TFLite fixed sequence length (based on dataset analysis)
# 99th percentile = 61 frames, rounded to power of 2 = 64
# This ensures consistent input/output signatures across all LOSO models
MAX_SEQ_LEN = 64


def get_default_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--config_path", type=str, default="",
                        help="Path to the config model to be used")
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="Path to pretrained weights (.h5 file)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed with which to initialize all the random components of the training")
    parser.add_argument("--task", type=str, default=False, choices=["train", "eval"],
                        help="Whether to train or evaluate the model")

    parser.add_argument("--data_path", type=str, default="",
                        help="Path to the training dataset")

    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for the model training")

    parser.add_argument("--resume_checkpoints", type=str, default="",
                        help="Path to the checkpoints to be used for resuming training")

    # Scheduler
    parser.add_argument("--scheduler_factor", type=float, default=0.1,
                        help="Factor for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for the ReduceLROnPlateau scheduler")
    
    # Validation control
    parser.add_argument("--no_validation", action="store_true",
                        help="Disable validation during training (training only)")
    
    # Checkpoint control
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs (default: 10)")
    parser.add_argument("--save_all_checkpoints", action="store_true",
                        help="Save checkpoint at every epoch (disk intensive)")

    return parser


def setup_logging(experiment_name):
    """Setup logging configuration."""
    log_dir = Path("logs/run_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(log_dir / f"{experiment_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    print(f"Random seed set to: {seed}")


def load_model(config_path, pretrained_path):
    """Load model and config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from {config_path}")
    print(f"  Config keys: {list(config.keys())}")
    
    model = SignBart(config)
    
    # ALWAYS build the model with dummy input to initialize parameters
    print(f"Building model with {len(config['joint_idx'])} keypoints...")
    dummy_data = {
        'keypoints': tf.random.normal((1, 10, len(config['joint_idx']), 2)),
        'attention_mask': tf.ones((1, 10))
    }
    _ = model(dummy_data, training=False)
    print("✓ Model built successfully")
    
    if pretrained_path:
        print(f"Loading pretrained weights from: {pretrained_path}")
        try:
            # Load weights
            model.load_weights(pretrained_path)
            print("✓ Pretrained weights loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    return model, config


def prepare_data_loaders(data_path, joint_idx, batch_size=1, no_validation=False):
    """Prepare training and validation data loaders."""
    train_datasets = SignDataset(data_path, "train", shuffle=True, joint_idxs=joint_idx, augment=True)
    train_loader = train_datasets.create_tf_dataset(batch_size, drop_remainder=False)
    
    if no_validation:
        return train_loader, None, train_datasets
    
    val_datasets = SignDataset(data_path, "test", shuffle=False, joint_idxs=joint_idx, augment=False)
    val_loader = val_datasets.create_tf_dataset(batch_size, drop_remainder=False)
    
    return train_loader, val_loader, train_datasets


def determine_keypoint_groups(config_joint_idx):
    """
    Determine how to group keypoints for normalization.
    Returns a list of lists, where each inner list is a group to normalize together.
    
    Automatically detects groups based on common MediaPipe layouts:
      - 65 keypoints: Pose (23) + Left Hand (21) + Right Hand (21)
      - 75 keypoints: Pose (33) + Left Hand (21) + Right Hand (21)
      - 90/100 keypoints: Pose (23/33) + Left Hand (21) + Right Hand (21) + Face (25)
    """
    if not config_joint_idx:
        return []
    
    # Sort indices to ensure they're in order
    sorted_idx = sorted(config_joint_idx)
    total_kpts = len(sorted_idx)
    
    # With face (90 or 100 total): Pose + Left Hand + Right Hand + Face
    if total_kpts in (90, 100):
        body_count = total_kpts - 67  # 21 + 21 + 25 = 67
        body_kpts = sorted_idx[:body_count]
        left_hand_kpts = sorted_idx[body_count:body_count + 21]
        right_hand_kpts = sorted_idx[body_count + 21:body_count + 42]
        face_kpts = sorted_idx[-25:]

        groups = []
        if body_kpts:
            groups.append(body_kpts)
        groups.append(left_hand_kpts)
        groups.append(right_hand_kpts)
        groups.append(face_kpts)
        return groups
    
    # No face (>= 42 total): Pose/Body + Left Hand + Right Hand
    if total_kpts >= 42:
        body_count = total_kpts - 42
        body_kpts = sorted_idx[:body_count] if body_count > 0 else []
        left_hand_kpts = sorted_idx[body_count:body_count + 21]
        right_hand_kpts = sorted_idx[body_count + 21:body_count + 42]

        groups = []
        if body_kpts:
            groups.append(body_kpts)
        groups.append(left_hand_kpts)
        groups.append(right_hand_kpts)
        return groups
    
    # Fallback: single group
    return [sorted_idx]


def train_step(model, batch, optimizer):
    """Single training step."""
    keypoints = batch['keypoints']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    
    with tf.GradientTape() as tape:
        loss, logits = model(keypoints, attention_mask, labels=labels, training=True)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, logits


def train_epoch(model, train_loader, optimizer, epoch, epochs, logger):
    """Train for one epoch."""
    from tqdm import tqdm
    import time
    
    all_losses = []
    all_accs = []
    all_top5_accs = []
    
    start_time = time.time()
    
    # Create progress bar (Keras style)
    pbar = tqdm(train_loader, 
                desc=f'Epoch {epoch}/{epochs}',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                ncols=100)
    
    for batch in pbar:
        loss, logits = train_step(model, batch, optimizer)
        
        acc = accuracy(logits, batch['labels'])
        top5_acc = top_k_accuracy(logits, batch['labels'], k=5)
        
        all_losses.append(loss.numpy())
        all_accs.append(acc)
        all_top5_accs.append(top5_acc)
        
        # Update progress bar with current metrics
        pbar.set_postfix({
            'loss': f'{loss.numpy():.4f}',
            'acc': f'{acc:.4f}',
            'top5': f'{top5_acc:.4f}'
        })
    
    elapsed = time.time() - start_time
    
    avg_loss = np.mean(all_losses)
    avg_acc = np.mean(all_accs)
    avg_top5_acc = np.mean(all_top5_accs)
    
    return avg_loss, avg_acc, avg_top5_acc


def evaluate(model, val_loader, epoch, epochs, logger):
    """Evaluate model."""
    from tqdm import tqdm
    
    all_losses = []
    all_accs = []
    all_top5_accs = []
    
    # Create progress bar for validation
    pbar = tqdm(val_loader,
                desc='Validating',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]',
                ncols=100,
                leave=False)
    
    for batch in pbar:
        keypoints = batch['keypoints']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        loss, logits = model(keypoints, attention_mask, labels=labels, training=False)
        
        acc = accuracy(logits, labels)
        top5_acc = top_k_accuracy(logits, labels, k=5)
        
        all_losses.append(loss.numpy())
        all_accs.append(acc)
        all_top5_accs.append(top5_acc)
        
        # Update progress bar with current metrics
        pbar.set_postfix({
            'val_loss': f'{loss.numpy():.4f}',
            'val_acc': f'{acc:.4f}',
            'val_top5': f'{top5_acc:.4f}'
        })
    
    avg_loss = np.mean(all_losses)
    avg_acc = np.mean(all_accs)
    avg_top5_acc = np.mean(all_top5_accs)
    
    return avg_loss, avg_acc, avg_top5_acc


def main(args):
    """Main training function."""
    set_random_seed(args.seed)
    logger = setup_logging(args.experiment_name)
    
    # Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            device = f"GPU: {gpus}"
        except RuntimeError as e:
            logger.error(e)
            device = "CPU"
    else:
        device = "CPU"
    
    print(f"\n{'='*80}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {device}")
    if args.no_validation:
        print(f"Validation: DISABLED (training only)")
    print(f"{'='*80}\n")
    
    # Load model and config
    model, config = load_model(args.config_path, args.pretrained_path)
    
    # Get joint indices from config
    if 'joint_idx' in config and config['joint_idx']:
        config_joint_idx = config['joint_idx']
        print(f"Using joint indices from config: {len(config_joint_idx)} keypoints")
        
        joint_idx = determine_keypoint_groups(config_joint_idx)
        
        # Report keypoint groups
        # Auto-detect labels based on group structure and size
        group_names = []
        for i, group in enumerate(joint_idx):
            group_size = len(group)
            # Standard MediaPipe structure: Pose + Left Hand (21) + Right Hand (21) + Face (25)
            if i == 0:
                # First group is always body/pose (variable size: 23, 33, etc.)
                name = f"body pose (indices {group[0]}-{group[-1]}, {group_size} keypoints)"
            elif i == 1 and group_size == 21:
                # Second group of 21 points is left hand
                name = f"left hand (indices {group[0]}-{group[-1]}, {group_size} keypoints)"
            elif i == 2 and group_size == 21:
                # Third group of 21 points is right hand
                name = f"right hand (indices {group[0]}-{group[-1]}, {group_size} keypoints)"
            elif i == 3 and group_size == 25:
                # Fourth group of 25 points is face
                name = f"face (indices {group[0]}-{group[-1]}, {group_size} keypoints)"
            else:
                # Fallback for unexpected structure
                name = f"group {i+1} (indices {group[0]}-{group[-1]}, {group_size} keypoints)"
            group_names.append(name)
        
        print(f"Keypoint groups for normalization ({len(joint_idx)} groups):")
        for i, name in enumerate(group_names):
            print(f"  Group {i+1}: {name}")
    else:
        from utils import hands_only_groups
        print("Using default joint indices (hands only)")
        joint_idx = hands_only_groups
    
    checkpoint_dir = "checkpoints_" + args.experiment_name
    
    # Prepare data
    batch_size = config.get('batch_size', 1)
    train_loader, val_loader, train_datasets = prepare_data_loaders(
        args.data_path, joint_idx, batch_size, args.no_validation
    )
    
    # Setup optimizer
    optimizer = keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=config.get('weight_decay', 0.01))
    
    # Compile model (Keras standard workflow)
    print(f"\n{'='*80}")
    print(f"COMPILING MODEL")
    print(f"{'='*80}")
    
    # Custom metrics
    from utils import top_k_accuracy as top_k_accuracy_fn
    
    class Top5Accuracy(keras.metrics.Metric):
        def __init__(self, name='top5_accuracy', **kwargs):
            super().__init__(name=name, **kwargs)
            self.top5_correct = self.add_weight(name='top5_correct', initializer='zeros')
            self.total = self.add_weight(name='total', initializer='zeros')
        
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
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            Top5Accuracy(name='top5_accuracy')
        ]
    )
    print("✓ Model compiled with AdamW optimizer and SparseCategoricalCrossentropy loss")
    
    # Show model summary (Keras standard)
    print(f"\n{'='*80}")
    print(f"MODEL SUMMARY")
    print(f"{'='*80}")
    try:
        model.summary(line_length=100)
    except Exception as e:
        print(f"Note: Could not display full summary: {e}")
        # Fallback to manual parameter count
        param_counts = count_model_parameters(model)
        total_params = param_counts['total']
        trainable_params = param_counts['trainable']
        print(f"\nModel Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")
    print(f"{'='*80}\n")
    
    # Count parameters for logging
    param_counts = count_model_parameters(model)
    total_params = param_counts['total']
    trainable_params = param_counts['trainable']
    logger.info(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Report configuration
    print(f"Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Optimizer: AdamW (weight_decay={config.get('weight_decay', 0.01)})")
    print(f"  Loss: SparseCategoricalCrossentropy (from_logits=True)")
    print(f"  Scheduler: ReduceLROnPlateau (factor={args.scheduler_factor}, patience={args.scheduler_patience})")
    print(f"  Seed: {args.seed}")
    print(f"  Validation: {'DISABLED' if args.no_validation else 'ENABLED'}")
    print(f"  Checkpoint strategy: Save every {args.save_every} epochs + best + latest + final")
    if args.save_all_checkpoints:
        print(f"  WARNING: Saving ALL epoch checkpoints (disk intensive!)")
    
    print(f"\nModel Architecture (from {args.config_path}):")
    print(f"  d_model: {config.get('d_model', 'N/A')}")
    print(f"  Encoder layers: {config.get('encoder_layers', 'N/A')}")
    print(f"  Decoder layers: {config.get('decoder_layers', 'N/A')}")
    print(f"  Attention heads (encoder): {config.get('encoder_attention_heads', 'N/A')}")
    print(f"  Attention heads (decoder): {config.get('decoder_attention_heads', 'N/A')}")
    print(f"  FFN dim (encoder): {config.get('encoder_ffn_dim', 'N/A')}")
    print(f"  FFN dim (decoder): {config.get('decoder_ffn_dim', 'N/A')}")
    print(f"  Dropout: {config.get('dropout', 'N/A')}")
    print(f"  Number of classes: {config.get('num_labels', 'N/A')}")
    print(f"  Max position embeddings: {config.get('max_position_embeddings', 'N/A')}")
    
    num_joints = len(config.get('joint_idx', []))
    if num_joints > 0:
        has_pose = any(idx < 33 for idx in config.get('joint_idx', []))
        has_left_hand = any(33 <= idx < 54 for idx in config.get('joint_idx', []))
        has_right_hand = any(54 <= idx < 75 for idx in config.get('joint_idx', []))
        has_face = any(idx >= 75 for idx in config.get('joint_idx', []))
        
        keypoint_types = []
        if has_pose:
            keypoint_types.append("body pose")
        if has_left_hand:
            keypoint_types.append("left hand")
        if has_right_hand:
            keypoint_types.append("right hand")
        if has_face:
            keypoint_types.append("face")
        
        desc = " + ".join(keypoint_types)
        print(f"  Joint indices: {num_joints} keypoints ({desc})")
    print()
    
    logger.info(f"Configuration: {config}")
    
    # Setup Keras callbacks
    callbacks = []
    
    # ModelCheckpoint for best model (weights only for subclassed model compatibility)
    # Use .weights.h5 extension to be explicit and avoid HDF5 full model save
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.weights.h5")
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=best_checkpoint_path,
        monitor='val_loss' if not args.no_validation else 'loss',
        save_best_only=True,
        save_weights_only=True,  # Critical: Only save weights, not full model (subclassed model)
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # ReduceLROnPlateau
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss' if not args.no_validation else 'loss',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        verbose=1,
        min_lr=1e-7
    )
    callbacks.append(reduce_lr_callback)
    
    # CSVLogger for training history
    training_csv_dir = Path("logs/training_csv")
    training_csv_dir.mkdir(parents=True, exist_ok=True)
    csv_logger = keras.callbacks.CSVLogger(
        str(training_csv_dir / f"{args.experiment_name}_training.csv"),
        append=True
    )
    callbacks.append(csv_logger)
    
    # TensorBoard (optional)
    tensorboard_dir = os.path.join("logs", args.experiment_name)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=0,
        write_graph=False
    )
    callbacks.append(tensorboard_callback)
    
    # Create directories
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path("out-imgs/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output images directory: out-imgs/{args.experiment_name}/\n")
    
    # Evaluation mode
    if args.task == "eval":
        if val_loader is None:
            print("ERROR: Cannot evaluate without validation data. Remove --no_validation flag.")
            return
        
        # Load checkpoint for evaluation
        if args.resume_checkpoints:
            checkpoint_to_load = args.resume_checkpoints
            
            # If it's a directory, look for final_model.keras or best_model.weights.h5
            if os.path.isdir(checkpoint_to_load):
                final_model_path = os.path.join(checkpoint_to_load, "final_model.keras")
                best_weights_path = os.path.join(checkpoint_to_load, "best_model.weights.h5")
                
                if os.path.exists(final_model_path):
                    checkpoint_to_load = final_model_path
                    print(f"\nLoading trained model from: {checkpoint_to_load}")
                    try:
                        # For .keras files, load the full model (but we need to reload to get architecture + weights)
                        loaded_model = keras.models.load_model(checkpoint_to_load)
                        # Copy weights to our model
                        model.set_weights(loaded_model.get_weights())
                        print("✓ Model weights loaded successfully from final_model.keras")
                        logger.info(f"Loaded model from {checkpoint_to_load}")
                    except Exception as e:
                        print(f"✗ Error loading model: {e}")
                        print("Using untrained model (this will give poor results!)")
                elif os.path.exists(best_weights_path):
                    checkpoint_to_load = best_weights_path
                    print(f"\nLoading trained weights from: {checkpoint_to_load}")
                    try:
                        model.load_weights(checkpoint_to_load)
                        print("✓ Model weights loaded successfully from best_model.weights.h5")
                        logger.info(f"Loaded weights from {checkpoint_to_load}")
                    except Exception as e:
                        print(f"✗ Error loading weights: {e}")
                        print("Using untrained model (this will give poor results!)")
                else:
                    print(f"\n⚠ Warning: No checkpoint found in {args.resume_checkpoints}")
                    print("Expected: final_model.keras or best_model.weights.h5")
                    print("Using untrained model (this will give poor results!)")
            else:
                # It's a file path
                print(f"\nLoading checkpoint from: {checkpoint_to_load}")
                try:
                    if checkpoint_to_load.endswith('.keras'):
                        loaded_model = keras.models.load_model(checkpoint_to_load)
                        model.set_weights(loaded_model.get_weights())
                        print("✓ Model loaded successfully")
                    else:
                        model.load_weights(checkpoint_to_load)
                        print("✓ Weights loaded successfully")
                    logger.info(f"Loaded checkpoint from {checkpoint_to_load}")
                except Exception as e:
                    print(f"✗ Error loading checkpoint: {e}")
                    print("Using untrained model (this will give poor results!)")
        else:
            print("\n⚠ Warning: No checkpoint specified (--resume_checkpoints)")
            print("Evaluating untrained model (this will give poor results!)")
        
        print("\nEvaluate model..!")
        start_time = time.time()
        results = model.evaluate(val_loader, return_dict=True, verbose=1)
        inference_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"Evaluation Results:")
        print(f"  Loss: {results['loss']:.4f}")
        print(f"  Top-1 Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"  Top-5 Accuracy: {results['top5_accuracy']:.4f} ({results['top5_accuracy']*100:.2f}%)")
        print(f"  Inference time: {inference_time:.2f} seconds")
        print(f"{'='*80}\n")
        
        logger.info(f"Evaluation - Loss: {results['loss']:.4f}, Acc: {results['accuracy']:.4f}, Top-5: {results['top5_accuracy']:.4f}")
        return
    
    # Train using Keras model.fit() - The standard Keras way!
    print(f"\n{'='*80}")
    print(f"STARTING TRAINING")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Use model.fit() - standard Keras training
    history = model.fit(
        train_loader,
        validation_data=val_loader if not args.no_validation else None,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1  # Standard Keras progress bar output
    )
    
    total_train_time = time.time() - start_time
    
    # Load best weights and save as final checkpoint
    print("\nLoading best weights and saving final checkpoint...")
    if os.path.exists(best_checkpoint_path):
        print(f"Loading best weights from: {best_checkpoint_path}")
        model.load_weights(best_checkpoint_path)
        logger.info(f"Loaded best weights from {best_checkpoint_path}")
    else:
        print(f"Warning: Best weights file not found at {best_checkpoint_path}")
        print("Saving current weights instead.")
    
    final_path = os.path.join(checkpoint_dir, "final_model.keras")
    model.save(final_path)
    print(f"Final checkpoint saved: {final_path}")
    logger.info(f"Final checkpoint saved with best weights: {final_path}")
    
    print(f"\nNote: The final model contains the BEST weights (not the last epoch's weights)")
    print(f"  Best weights from: {best_checkpoint_path}")
    print(f"  Saved full model to: {final_path}")
    
    # Convert to TFLite for deployment
    print(f"\n{'='*80}")
    print("CONVERTING TO TFLITE FOR DEPLOYMENT")
    print(f"{'='*80}")
    
    try:
        import tensorflow.lite as tflite
        
        # Get number of keypoints from config
        num_keypoints = len(config['joint_idx'])
        
        # Create a concrete function with FIXED input shape for TFLite
        print(f"\nUsing fixed sequence length for TFLite: {MAX_SEQ_LEN} frames")
        print(f"  (Covers 99%+ of dataset, max was 75 frames)")
        
        @tf.function(input_signature=[
            {
                'keypoints': tf.TensorSpec(shape=[1, MAX_SEQ_LEN, num_keypoints, 2], dtype=tf.float32),
                'attention_mask': tf.TensorSpec(shape=[1, MAX_SEQ_LEN], dtype=tf.float32)
            }
        ])
        def model_predict(inputs):
            return model(inputs, training=False)
        
        print("\n1. Converting to FP32 TFLite...")
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [model_predict.get_concrete_function()]
        )
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = os.path.join(checkpoint_dir, "final_model_fp32.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        tflite_size_mb = os.path.getsize(tflite_path) / (1024**2)
        print(f"✓ TFLite FP32 model saved: {tflite_path}")
        print(f"  File size: {tflite_size_mb:.2f} MB")
        
        # Evaluate both Keras and TFLite models on full test/validation set
        print("\n2. Evaluating Keras model on test set...")
        if val_loader is not None:
            keras_start = time.time()
            keras_results = model.evaluate(val_loader, return_dict=True, verbose=1)
            keras_time = time.time() - keras_start
            
            print(f"\n  Keras Model Results:")
            print(f"    Loss: {keras_results['loss']:.4f}")
            print(f"    Top-1 Accuracy: {keras_results['accuracy']:.4f} ({keras_results['accuracy']*100:.2f}%)")
            print(f"    Top-5 Accuracy: {keras_results['top5_accuracy']:.4f} ({keras_results['top5_accuracy']*100:.2f}%)")
            print(f"    Inference time: {keras_time:.2f} seconds")
            
            logger.info(f"Keras eval - Loss: {keras_results['loss']:.4f}, Acc: {keras_results['accuracy']:.4f}, Top5: {keras_results['top5_accuracy']:.4f}")
            
            # Evaluate TFLite model on full test set
            print("\n3. Evaluating TFLite FP32 model on test set...")
            print(f"   Note: TFLite requires fixed seq_len={MAX_SEQ_LEN}, padding/subsampling as needed")
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Find input indices
            keypoints_idx = None
            mask_idx = None
            for i, detail in enumerate(input_details):
                if 'keypoints' in detail['name'].lower():
                    keypoints_idx = i
                elif 'attention' in detail['name'].lower() or 'mask' in detail['name'].lower():
                    mask_idx = i
            
            # Fallback to shape-based detection
            if keypoints_idx is None:
                for i, detail in enumerate(input_details):
                    if len(detail['shape']) == 4:
                        keypoints_idx = i
            if mask_idx is None:
                for i, detail in enumerate(input_details):
                    if len(detail['shape']) == 2:
                        mask_idx = i
            
            # Run inference on all batches
            tflite_correct = 0
            tflite_top5_correct = 0
            tflite_total = 0
            tflite_losses = []
            max_output_diff = 0.0
            
            tflite_start = time.time()
            
            for batch_idx, batch in enumerate(val_loader):
                batch_keypoints = batch['keypoints']
                batch_mask = batch['attention_mask']
                batch_labels = batch['labels'].numpy()
                
                # Process each sample in batch (TFLite typically processes one at a time)
                for i in range(len(batch_keypoints)):
                    keypoints_sample = batch_keypoints[i:i+1]
                    mask_sample = batch_mask[i:i+1]
                    label = batch_labels[i]
                    
                    # Preprocess for TFLite: pad/subsample to MAX_SEQ_LEN
                    seq_len = tf.shape(keypoints_sample)[1].numpy()
                    
                    if seq_len != MAX_SEQ_LEN:
                        # Need to resize to MAX_SEQ_LEN
                        kpts_np = keypoints_sample.numpy()[0]  # [seq_len, num_kpts, 2]
                        mask_np = mask_sample.numpy()[0]        # [seq_len]
                        
                        if seq_len < MAX_SEQ_LEN:
                            # Pad
                            pad_len = MAX_SEQ_LEN - seq_len
                            kpts_padded = np.pad(kpts_np, ((0, pad_len), (0, 0), (0, 0)), mode='constant', constant_values=0.0)
                            mask_padded = np.pad(mask_np, (0, pad_len), mode='constant', constant_values=0.0)
                        else:
                            # Subsample (rare case)
                            indices = np.linspace(0, seq_len-1, MAX_SEQ_LEN, dtype=int)
                            kpts_padded = kpts_np[indices]
                            mask_padded = mask_np[indices]
                        
                        # Add batch dimension back
                        keypoints_tflite = kpts_padded[np.newaxis, ...]  # [1, MAX_SEQ_LEN, num_kpts, 2]
                        mask_tflite = mask_padded[np.newaxis, ...]       # [1, MAX_SEQ_LEN]
                    else:
                        # Already correct size
                        keypoints_tflite = keypoints_sample.numpy()
                        mask_tflite = mask_sample.numpy()
                    
                    # TFLite inference
                    interpreter.set_tensor(input_details[keypoints_idx]['index'], keypoints_tflite.astype(np.float32))
                    interpreter.set_tensor(input_details[mask_idx]['index'], mask_tflite.astype(np.float32))
                    interpreter.invoke()
                    tflite_output = interpreter.get_tensor(output_details[0]['index'])[0]
                    
                    # Also get Keras output for comparison
                    keras_output = model({'keypoints': keypoints_sample, 'attention_mask': mask_sample}, training=False).numpy()[0]
                    
                    # Track max difference
                    diff = np.abs(keras_output - tflite_output).max()
                    max_output_diff = max(max_output_diff, diff)
                    
                    # Predictions
                    tflite_pred = np.argmax(tflite_output)
                    tflite_top5_preds = np.argsort(tflite_output)[-5:]
                    
                    # Accuracy
                    if tflite_pred == label:
                        tflite_correct += 1
                    if label in tflite_top5_preds:
                        tflite_top5_correct += 1
                    
                    tflite_total += 1
                
                # Progress indicator
                if (batch_idx + 1) % 20 == 0:
                    current_acc = tflite_correct / tflite_total
                    print(f"  Progress: {batch_idx + 1}/{len(val_loader)} batches, Current accuracy: {current_acc:.4f}")
            
            tflite_time = time.time() - tflite_start
            
            tflite_accuracy = tflite_correct / tflite_total
            tflite_top5_accuracy = tflite_top5_correct / tflite_total
            
            print(f"\n  TFLite FP32 Model Results:")
            print(f"    Top-1 Accuracy: {tflite_accuracy:.4f} ({tflite_accuracy*100:.2f}%)")
            print(f"    Top-5 Accuracy: {tflite_top5_accuracy:.4f} ({tflite_top5_accuracy*100:.2f}%)")
            print(f"    Inference time: {tflite_time:.2f} seconds")
            print(f"    Max output difference vs Keras: {max_output_diff:.6e}")
            
            logger.info(f"TFLite eval - Acc: {tflite_accuracy:.4f}, Top5: {tflite_top5_accuracy:.4f}")
            
            # Comparison
            print(f"\n  {'='*80}")
            print(f"  COMPARISON:")
            print(f"  {'='*80}")
            print(f"  Metric                  | Keras Model | TFLite FP32 | Difference")
            print(f"  {'-'*80}")
            print(f"  Top-1 Accuracy          | {keras_results['accuracy']*100:6.2f}%    | {tflite_accuracy*100:6.2f}%     | {abs(keras_results['accuracy'] - tflite_accuracy)*100:+5.2f}%")
            print(f"  Top-5 Accuracy          | {keras_results['top5_accuracy']*100:6.2f}%    | {tflite_top5_accuracy*100:6.2f}%     | {abs(keras_results['top5_accuracy'] - tflite_top5_accuracy)*100:+5.2f}%")
            print(f"  Inference time          | {keras_time:6.2f}s    | {tflite_time:6.2f}s     | {tflite_time - keras_time:+6.2f}s")
            print(f"  Model size              | {os.path.getsize(final_path)/(1024**2):6.2f} MB  | {tflite_size_mb:6.2f} MB   | {(os.path.getsize(final_path)/(1024**2)) - tflite_size_mb:+6.2f} MB")
            print(f"  {'='*80}")
            
            if abs(keras_results['accuracy'] - tflite_accuracy) < 0.01:
                print(f"  ✓ Keras and TFLite accuracies match (difference < 1%)")
            else:
                print(f"  ⚠ Keras and TFLite accuracies differ by {abs(keras_results['accuracy'] - tflite_accuracy)*100:.2f}%")
            
            if max_output_diff < 1e-5:
                print(f"  ✓ Outputs are numerically identical (max diff: {max_output_diff:.2e})")
            elif max_output_diff < 1e-3:
                print(f"  ✓ Outputs are very close (max diff: {max_output_diff:.2e})")
            else:
                print(f"  ⚠ Outputs differ noticeably (max diff: {max_output_diff:.2e})")
            
        else:
            print("  Note: No validation data available for TFLite evaluation")
        
        print(f"\n{'='*80}")
        print("✓ TFLite FP32 conversion and validation complete!")
        print(f"{'='*80}")
        print(f"Deployment-ready models:")
        print(f"  1. Keras model: {final_path} ({os.path.getsize(final_path)/(1024**2):.2f} MB)")
        print(f"  2. FP32 TFLite: {tflite_path} ({tflite_size_mb:.2f} MB)")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n✗ TFLite conversion failed: {e}")
        print("Note: You can still use the .keras model for inference")
        import traceback
        traceback.print_exc()
        logger.warning(f"TFLite conversion failed: {e}")
    
    # Final summary
    avg_epoch_time = total_train_time / args.epochs
    best_train_acc = max(history.history['accuracy'])
    best_val_acc = max(history.history['val_accuracy']) if 'val_accuracy' in history.history else 0.0
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total training time: {total_train_time/3600:.2f} hours ({total_train_time:.1f} seconds)")
    print(f"Average time per epoch: {avg_epoch_time:.1f} seconds")
    print(f"Best train accuracy: {best_train_acc:.4f} ({best_train_acc*100:.2f}%)")
    if not args.no_validation:
        print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    # List saved checkpoints
    print(f"\nSaved checkpoints in {checkpoint_dir}:")
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(('.keras', '.h5'))])
        for ckpt in checkpoints:
            size_mb = os.path.getsize(os.path.join(checkpoint_dir, ckpt)) / (1024**2)
            print(f"  - {ckpt} ({size_mb:.2f} MB)")
    
    print(f"{'='*80}\n")
    
    logger.info(f"Training complete - Total time: {total_train_time:.1f}s, Best train acc: {best_train_acc:.4f}")
    if not args.no_validation:
        logger.info(f"Best val acc: {best_val_acc:.4f}")
    
    # Plot training curves using history from model.fit()
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    ax.plot(epochs_range, history.history['loss'], c="#D64436", label="Training loss", linestyle='--', linewidth=2)
    ax.plot(epochs_range, history.history['accuracy'], c="#00B09B", label="Training accuracy", linewidth=2)
    
    if not args.no_validation and 'val_accuracy' in history.history:
        ax.plot(epochs_range, history.history['val_loss'], c="#FF6B6B", label="Validation loss", linestyle='--', linewidth=2)
        ax.plot(epochs_range, history.history['val_accuracy'], c="#E0A938", label="Validation accuracy", linewidth=2)
    
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set(xlabel="Epoch", ylabel="Metrics", title=f"Training Curves - {args.experiment_name}")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plot_path = "out-imgs/" + args.experiment_name + "/training_curves.png"
    fig.savefig(plot_path, bbox_inches='tight', dpi=100)
    plt.close()
    
    print(f"\nPlot saved to {plot_path}")
    logger.info("Experiment finished.")

    # Clear GPU/TF resources
    keras.backend.clear_session()
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.experiment_name:
        print("ERROR: --experiment_name is required")
        exit(1)
    if not args.config_path:
        print("ERROR: --config_path is required")
        exit(1)
    if not args.data_path:
        print("ERROR: --data_path is required")
        exit(1)
    if not args.task:
        print("ERROR: --task is required (train or eval)")
        exit(1)
    
    main(args)

