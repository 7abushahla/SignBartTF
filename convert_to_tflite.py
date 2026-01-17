"""
Convert trained SignBART TensorFlow model to TFLite with quantization support.
"""
import os
import yaml
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

from model import SignBart
from dataset import SignDataset
from utils import get_keypoint_config


def representative_dataset_gen(dataset_obj, num_samples=100):
    """
    Generator function for representative dataset (needed for full integer quantization).
    
    Args:
        dataset_obj: SignDataset instance
        num_samples: number of samples to use for calibration
    
    Yields:
        List of input tensors for the model
    """
    print(f"Generating representative dataset with {num_samples} samples...")
    
    for i in range(min(num_samples, len(dataset_obj))):
        keypoints, label = dataset_obj.load_sample(dataset_obj.list_key[i])
        
        # Add batch dimension and create attention mask
        keypoints = np.expand_dims(keypoints, axis=0)  # (1, T, K, 2)
        attention_mask = np.ones((1, keypoints.shape[1]), dtype=np.float32)
        
        # Yield as a list (model expects multiple inputs)
        yield [keypoints, attention_mask]


def convert_to_tflite(model, output_path, quantization='none', 
                      representative_dataset=None, input_shapes=None):
    """
    Convert Keras model to TFLite format with optional quantization.
    
    Args:
        model: Keras model
        output_path: path to save TFLite model
        quantization: one of ['none', 'dynamic', 'float16', 'int8', 'int8_full']
        representative_dataset: generator for representative dataset (needed for int8)
        input_shapes: dict of input shapes for model
    
    Returns:
        Path to saved TFLite model
    """
    print(f"\n{'='*80}")
    print(f"Converting model to TFLite with quantization: {quantization}")
    print(f"{'='*80}\n")
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization based on quantization type
    if quantization == 'none':
        print("No quantization - full float32 model")
        pass  # No optimizations
    
    elif quantization == 'dynamic':
        print("Dynamic range quantization - weights int8, activations float32")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    elif quantization == 'float16':
        print("Float16 quantization - all float16")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    elif quantization == 'int8':
        print("Integer quantization - weights and activations int8 (hybrid)")
        if representative_dataset is None:
            raise ValueError("representative_dataset required for int8 quantization")
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
    
    elif quantization == 'int8_full':
        print("Full integer quantization - all int8 (for edge TPU/embedded)")
        if representative_dataset is None:
            raise ValueError("representative_dataset required for int8_full quantization")
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
    
    else:
        raise ValueError(f"Unknown quantization type: {quantization}")
    
    # Convert
    print("\nConverting... (this may take a while)")
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"Conversion failed: {e}")
        print("\nTrying with experimental features...")
        converter.experimental_new_converter = True
        tflite_model = converter.convert()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get file size
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"\n✓ Model saved to: {output_path}")
    print(f"  File size: {size_mb:.2f} MB")
    
    return output_path


def test_tflite_model(tflite_path, test_input_keypoints, test_input_mask):
    """
    Test the TFLite model with sample input.
    
    Args:
        tflite_path: path to TFLite model
        test_input_keypoints: test keypoints (batch_size, seq_len, num_joints, 2)
        test_input_mask: test attention mask (batch_size, seq_len)
    
    Returns:
        output predictions
    """
    print(f"\n{'='*80}")
    print(f"Testing TFLite model: {tflite_path}")
    print(f"{'='*80}\n")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Input details:")
    for i, detail in enumerate(input_details):
        print(f"  Input {i}: {detail['name']}, shape={detail['shape']}, dtype={detail['dtype']}")
    
    print("\nOutput details:")
    for i, detail in enumerate(output_details):
        print(f"  Output {i}: {detail['name']}, shape={detail['shape']}, dtype={detail['dtype']}")
    
    # Prepare inputs
    # Note: TFLite may require specific shapes or types
    keypoints_input = test_input_keypoints.astype(np.float32)
    mask_input = test_input_mask.astype(np.float32)
    
    # Set input tensors
    interpreter.set_tensor(input_details[0]['index'], keypoints_input)
    interpreter.set_tensor(input_details[1]['index'], mask_input)
    
    # Run inference
    print("\nRunning inference...")
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"✓ Inference successful!")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output sample: {output[0, :5]}...")
    
    return output


def main(args):
    """Main conversion function."""
    
    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Get keypoint configuration
    joint_idx, joint_groups = get_keypoint_config(config['keypoint_config'])
    config['joint_idx'] = joint_idx
    print(f"Keypoint config: {config['keypoint_config']} ({len(joint_idx)} keypoints)")
    
    # Create model
    print("\nCreating model...")
    model = SignBart(config)
    
    # Build model with dummy input to initialize weights
    dummy_keypoints = tf.random.normal((1, 10, len(joint_idx), 2))
    dummy_mask = tf.ones((1, 10))
    _ = model(dummy_keypoints, dummy_mask, training=False)
    
    print(f"Model created with {sum([tf.size(w).numpy() for w in model.trainable_weights]):,} parameters")
    
    # Load weights
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        try:
            model.load_weights(args.checkpoint)
            print("✓ Checkpoint loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Using random weights (for testing only)")
    else:
        print("\nNo checkpoint provided - using random weights (for testing only)")
    
    # Prepare representative dataset if needed
    representative_dataset = None
    if args.quantization in ['int8', 'int8_full']:
        print("\nPreparing representative dataset for quantization calibration...")
        dataset_obj = SignDataset(
            root=config['data_root'],
            split='train',
            shuffle=True,
            joint_idxs=joint_groups,
            augment=False
        )
        representative_dataset = lambda: representative_dataset_gen(dataset_obj, args.num_calibration_samples)
    
    # Convert model
    output_path = args.output or f"signbart_{args.quantization}.tflite"
    
    convert_to_tflite(
        model=model,
        output_path=output_path,
        quantization=args.quantization,
        representative_dataset=representative_dataset,
        input_shapes={
            'keypoints': (1, None, len(joint_idx), 2),
            'attention_mask': (1, None)
        }
    )
    
    # Test the converted model
    if args.test:
        print("\nTesting converted model...")
        
        # Create test input
        test_keypoints = np.random.randn(1, 20, len(joint_idx), 2).astype(np.float32)
        test_mask = np.ones((1, 20), dtype=np.float32)
        
        # Test TFLite model
        tflite_output = test_tflite_model(output_path, test_keypoints, test_mask)
        
        # Compare with original model (if not int8_full)
        if args.quantization != 'int8_full':
            print("\nComparing with original TensorFlow model...")
            tf_output = model(
                tf.constant(test_keypoints),
                tf.constant(test_mask),
                training=False
            ).numpy()
            
            diff = np.abs(tf_output - tflite_output).mean()
            print(f"  Mean absolute difference: {diff:.6f}")
            
            if diff < 1.0:
                print("  ✓ Outputs are similar")
            else:
                print("  ⚠ Outputs differ significantly (expected for quantized models)")
    
    print(f"\n{'='*80}")
    print("Conversion complete!")
    print(f"Model saved to: {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert SignBART to TFLite with quantization')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to model config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (.h5 file)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for TFLite model')
    parser.add_argument('--quantization', type=str, default='dynamic',
                        choices=['none', 'dynamic', 'float16', 'int8', 'int8_full'],
                        help='Quantization type')
    parser.add_argument('--num-calibration-samples', type=int, default=100,
                        help='Number of samples for quantization calibration (for int8)')
    parser.add_argument('--test', action='store_true',
                        help='Test the converted model')
    
    args = parser.parse_args()
    main(args)

