"""
Convert trained e-LSTM model to TFLite for mobile deployment

Usage:
    python convert_to_tflite.py --model checkpoints/elstm_final_TIMESTAMP.h5 --output elstm_model.tflite
"""

import argparse
import tensorflow as tf
from tensorflow import keras
from pathlib import Path


def convert_to_tflite(model_path, output_path, quantize=False):
    """
    Convert Keras model to TFLite
    
    Args:
        model_path: Path to saved .h5 model
        output_path: Output path for .tflite model
        quantize: Whether to apply dynamic range quantization (smaller model)
    """
    print(f"Loading model from {model_path}...")
    
    # Load model (BahdanauAttention is registered, no custom_objects needed)
    model = keras.models.load_model(model_path)
    
    print("Model loaded successfully!")
    model.summary()
    
    # Convert to TFLite
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optional: Apply optimizations
    if quantize:
        print("Applying dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Print size info
    original_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
    tflite_size = len(tflite_model) / (1024 * 1024)  # MB
    
    print(f"\n✅ Conversion complete!")
    print(f"Original model size: {original_size:.2f} MB")
    print(f"TFLite model size: {tflite_size:.2f} MB")
    print(f"Compression ratio: {original_size / tflite_size:.2f}x")
    print(f"Saved to: {output_path}")
    
    return tflite_model


def test_tflite_model(tflite_path, test_input):
    """
    Test TFLite model with sample input
    
    Args:
        tflite_path: Path to .tflite model
        test_input: Test input array (seq_len, 90, 2)
    """
    print(f"\nTesting TFLite model from {tflite_path}...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # Prepare test input
    test_input = test_input.astype('float32')
    test_input = test_input[None, ...]  # Add batch dimension
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"✅ Inference successful!")
    print(f"Output shape: {output.shape}")
    print(f"Predicted class: {output.argmax()}")
    print(f"Confidence: {output.max():.4f}")
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert e-LSTM to TFLite")
    parser.add_argument("--model", required=True, help="Path to .h5 model")
    parser.add_argument("--output", default="elstm_model.tflite", help="Output .tflite path")
    parser.add_argument("--quantize", action="store_true", help="Apply quantization")
    parser.add_argument("--test", action="store_true", help="Test converted model")
    
    args = parser.parse_args()
    
    # Convert
    tflite_model = convert_to_tflite(args.model, args.output, args.quantize)
    
    # Test if requested
    if args.test:
        import numpy as np
        
        # Create dummy test input (seq_len, 90, 2)
        # Replace with actual test data
        test_input = np.random.rand(64, 90, 2).astype('float32')
        test_tflite_model(args.output, test_input)

