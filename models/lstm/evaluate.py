"""
Evaluate trained e-LSTM model on test set (for proper LOSO evaluation)

Usage:
    python evaluate.py --model_path outputs/user01/checkpoints/elstm_final.h5 \
                       --test_dir ../../data/arabic-asl-90kpts_LOSO_user01/test \
                       --config config.yaml
"""

import argparse
import numpy as np
import yaml
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# Import data loader from train.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import KeypointDataLoader


def evaluate_model(model_path, test_dir, config_path):
    """
    Evaluate trained model on test set
    
    Args:
        model_path: Path to trained model (.h5 file)
        test_dir: Directory containing test .pkl files
        config_path: Path to config YAML file
    """
    print("="*80)
    print("e-LSTM Model Evaluation (LOSO)")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Test data: {test_dir}")
    print("="*80 + "\n")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    num_classes = config['model']['num_classes']
    seq_len = config['model']['seq_len']
    
    # Load test data (NO augmentation!)
    print("Loading test data...")
    test_loader = KeypointDataLoader(
        test_dir,
        seq_len=seq_len,
        config=config,
        augment=False  # Never augment test data
    )
    X_test, y_test = test_loader.load_data()
    y_test_cat = keras.utils.to_categorical(y_test, num_classes=num_classes)
    
    print(f"Test data: X shape = {X_test.shape}, y shape = {y_test.shape}")
    print(f"Number of test samples: {X_test.shape[0]}\n")
    
    # Load model
    print("Loading trained model...")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully\n")
    
    # Evaluate
    print("="*80)
    print("Evaluating on Test Set...")
    print("="*80)
    
    results = model.evaluate(X_test, y_test_cat, verbose=1)
    test_loss = results[0]
    test_acc = results[1]
    test_top5 = results[2] if len(results) > 2 else None
    
    print("\n" + "="*80)
    print("Final Results")
    print("="*80)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    if test_top5:
        print(f"Test Top-5 Accuracy: {test_top5:.4f} ({test_top5*100:.2f}%)")
    print("="*80)
    
    # Per-class accuracy
    print("\n" + "="*80)
    print("Per-Class Analysis")
    print("="*80)
    
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Compute per-class accuracy
    for class_id in range(num_classes):
        mask = (y_test == class_id)
        if mask.sum() == 0:
            continue
        
        class_acc = (y_pred_classes[mask] == class_id).mean()
        class_name = f"G{class_id+1:02d}"
        print(f"  {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%) - {mask.sum()} samples")
    
    print("="*80)
    
    return test_acc, test_top5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained e-LSTM model")
    parser.add_argument("--model_path", required=True, help="Path to trained model (.h5)")
    parser.add_argument("--test_dir", required=True, help="Test directory (e.g., data/LOSO_user01/test)")
    parser.add_argument("--config", default="config.yaml", help="Config YAML file")
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_dir, args.config)

