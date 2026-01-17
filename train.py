"""
TensorFlow/Keras training script for SignBART.
Converted from PyTorch implementation.
"""
import os
import yaml
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from tqdm import tqdm

from model import SignBart
from dataset import create_data_loaders
from utils import (
    accuracy, top_k_accuracy, save_checkpoint, load_checkpoint,
    count_model_parameters, get_keypoint_config
)


def setup_logging(log_dir):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_step(model, batch, optimizer, loss_fn):
    """
    Single training step.
    
    Args:
        model: SignBart model
        batch: dict with 'keypoints', 'attention_mask', 'labels'
        optimizer: Keras optimizer
        loss_fn: loss function
    
    Returns:
        loss, logits
    """
    keypoints = batch['keypoints']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    
    with tf.GradientTape() as tape:
        loss, logits = model(
            keypoints,
            attention_mask,
            labels=labels,
            training=True
        )
    
    # Compute gradients and update weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, logits


def train_epoch(model, dataset, optimizer, loss_fn, epoch, total_epochs, logger):
    """
    Train for one epoch.
    
    Args:
        model: SignBart model
        dataset: tf.data.Dataset
        optimizer: Keras optimizer
        loss_fn: loss function
        epoch: current epoch
        total_epochs: total number of epochs
        logger: logger instance
    
    Returns:
        avg_loss, avg_acc, avg_top5_acc
    """
    all_losses = []
    all_accs = []
    all_top5_accs = []
    
    # Progress bar
    pbar = tqdm(dataset, desc=f"Training epoch {epoch+1}/{total_epochs}")
    
    for batch in pbar:
        loss, logits = train_step(model, batch, optimizer, loss_fn)
        
        # Calculate metrics
        acc = accuracy(logits, batch['labels'])
        top5_acc = top_k_accuracy(logits, batch['labels'], k=5)
        
        all_losses.append(loss.numpy())
        all_accs.append(acc)
        all_top5_accs.append(top5_acc)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.numpy():.4f}",
            'acc': f"{acc:.4f}",
            'top5': f"{top5_acc:.4f}"
        })
    
    avg_loss = np.mean(all_losses)
    avg_acc = np.mean(all_accs)
    avg_top5_acc = np.mean(all_top5_accs)
    
    logger.info(f"Train Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, Top5={avg_top5_acc:.4f}")
    
    return avg_loss, avg_acc, avg_top5_acc


@tf.function
def eval_step(model, batch):
    """
    Single evaluation step (with tf.function for speed).
    
    Args:
        model: SignBart model
        batch: dict with 'keypoints', 'attention_mask', 'labels'
    
    Returns:
        loss, logits
    """
    keypoints = batch['keypoints']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    
    loss, logits = model(
        keypoints,
        attention_mask,
        labels=labels,
        training=False
    )
    
    return loss, logits


def evaluate(model, dataset, epoch, total_epochs, logger):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: SignBart model
        dataset: tf.data.Dataset
        epoch: current epoch
        total_epochs: total number of epochs
        logger: logger instance
    
    Returns:
        avg_loss, avg_acc, avg_top5_acc
    """
    all_losses = []
    all_accs = []
    all_top5_accs = []
    
    # Progress bar
    pbar = tqdm(dataset, desc=f"Evaluation epoch {epoch+1}/{total_epochs}")
    
    for batch in pbar:
        loss, logits = eval_step(model, batch)
        
        # Calculate metrics
        acc = accuracy(logits, batch['labels'])
        top5_acc = top_k_accuracy(logits, batch['labels'], k=5)
        
        all_losses.append(loss.numpy())
        all_accs.append(acc)
        all_top5_accs.append(top5_acc)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.numpy():.4f}",
            'acc': f"{acc:.4f}",
            'top5': f"{top5_acc:.4f}"
        })
    
    avg_loss = np.mean(all_losses)
    avg_acc = np.mean(all_accs)
    avg_top5_acc = np.mean(all_top5_accs)
    
    logger.info(f"Val Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, Top5={avg_top5_acc:.4f}")
    
    return avg_loss, avg_acc, avg_top5_acc


def main(args):
    """Main training function."""
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logging(config['log_dir'])
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Config: {config}")
    
    # Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Using GPU: {gpus}")
        except RuntimeError as e:
            logger.error(e)
    else:
        logger.info("No GPU found, using CPU")
    
    # Get keypoint configuration
    joint_idx, joint_groups = get_keypoint_config(config['keypoint_config'])
    config['joint_idx'] = joint_idx
    logger.info(f"Using keypoint config: {config['keypoint_config']} ({len(joint_idx)} keypoints)")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_data_loaders(
        root=config['data_root'],
        batch_size=config['batch_size'],
        joint_idxs=joint_groups,
        augment_train=config.get('augment', True)
    )
    
    # Create model
    logger.info("Creating model...")
    model = SignBart(config)
    
    # Count parameters
    param_counts = count_model_parameters(model)
    logger.info(f"Model parameters - Total: {param_counts['total']:,}, Trainable: {param_counts['trainable']:,}")
    
    # Setup optimizer and loss
    if config['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(
            learning_rate=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0)
        )
    elif config['optimizer'] == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        try:
            start_epoch = load_checkpoint(model, optimizer, config['checkpoint_dir'], resume=True)
            logger.info(f"Resumed from epoch {start_epoch}")
        except FileNotFoundError as e:
            logger.warning(f"Could not load checkpoint: {e}")
            logger.info("Starting from scratch")
    
    # Training loop
    logger.info("Starting training...")
    best_val_acc = 0.0
    best_train_acc = 0.0
    
    for epoch in range(start_epoch, config['epochs']):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch+1}/{config['epochs']}")
        logger.info(f"{'='*80}")
        
        # Train
        train_loss, train_acc, train_top5 = train_epoch(
            model, train_dataset, optimizer, loss_fn, 
            epoch, config['epochs'], logger
        )
        
        # Validate
        val_loss, val_acc, val_top5 = evaluate(
            model, val_dataset, epoch, config['epochs'], logger
        )
        
        # Save checkpoint
        if (epoch + 1) % config.get('save_every', 5) == 0:
            save_checkpoint(model, optimizer, epoch, config['checkpoint_dir'])
        
        # Save best models
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, config['checkpoint_dir'], name='best_val')
            logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
        
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            save_checkpoint(model, optimizer, epoch, config['checkpoint_dir'], name='best_train')
        
        # Log summary
        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Top5: {train_top5:.4f}")
        logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Top5: {val_top5:.4f}")
        logger.info(f"  Best Val Acc: {best_val_acc:.4f}, Best Train Acc: {best_train_acc:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Best training accuracy: {best_train_acc:.4f}")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SignBART model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    main(args)

