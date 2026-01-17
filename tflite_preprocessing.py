"""
Preprocessing utilities for TFLite model deployment.

This module provides functions to prepare keypoint data for inference
with the SignBART TFLite models (all LOSO variants).

All LOSO models have IDENTICAL input/output signatures:
  - Input: float32[1, 64, 90, 2] (keypoints) + float32[1, 64] (attention_mask)
  - Output: float32[1, 10] (class logits)
"""

import numpy as np

# TFLite fixed sequence length (based on dataset analysis)
# - 99th percentile = 61 frames
# - Rounded to power of 2 = 64
# - Covers 99%+ of training data (max was 75 frames)
MAX_SEQ_LEN = 64

# Number of keypoints per frame
NUM_KEYPOINTS = 90  # 23 pose + 21 left hand + 21 right hand + 25 face

# Number of output classes
NUM_CLASSES = 10


def preprocess_for_tflite(keypoints, max_len=MAX_SEQ_LEN):
    """
    Prepare keypoints for TFLite inference with fixed sequence length.
    
    This function handles both padding (for short sequences) and subsampling
    (for sequences longer than max_len).
    
    Args:
        keypoints: numpy array of shape [num_frames, 90, 2]
                   - num_frames: actual video length (variable)
                   - 90: number of keypoints (23 pose + 21 left hand + 21 right hand + 25 face)
                   - 2: (x, y) coordinates (normalized)
        max_len: fixed sequence length for TFLite (default: 64)
    
    Returns:
        keypoints_padded: numpy array [1, 64, 90, 2] ready for TFLite
        attention_mask: numpy array [1, 64] (1.0 for valid frames, 0.0 for padding)
    
    Example:
        >>> # Your video has 38 frames
        >>> keypoints = extract_keypoints_from_video(video)  # shape: [38, 90, 2]
        >>> kpts_input, mask_input = preprocess_for_tflite(keypoints)
        >>> # Now kpts_input shape: [1, 64, 90, 2]
        >>> # mask_input shape: [1, 64]
        >>> # Ready for TFLite inference!
    """
    # Validate input shape
    if keypoints.ndim != 3:
        raise ValueError(f"Expected keypoints shape (num_frames, 90, 2), got {keypoints.shape}")
    
    num_frames, num_kpts, num_coords = keypoints.shape
    
    if num_kpts != NUM_KEYPOINTS:
        raise ValueError(f"Expected {NUM_KEYPOINTS} keypoints per frame, got {num_kpts}")
    
    if num_coords != 2:
        raise ValueError(f"Expected 2 coordinates (x, y) per keypoint, got {num_coords}")
    
    # Handle sequence length
    if num_frames <= max_len:
        # PAD short sequences
        pad_len = max_len - num_frames
        keypoints_padded = np.pad(
            keypoints,
            pad_width=((0, pad_len), (0, 0), (0, 0)),
            mode='constant',
            constant_values=0.0
        )
        # Create attention mask: 1.0 for valid frames, 0.0 for padding
        mask = np.concatenate([
            np.ones(num_frames, dtype=np.float32),
            np.zeros(pad_len, dtype=np.float32)
        ])
    else:
        # SUBSAMPLE long sequences (only ~1% of samples)
        # Use linear interpolation to preserve temporal information
        indices = np.linspace(0, num_frames - 1, max_len, dtype=int)
        keypoints_padded = keypoints[indices]
        mask = np.ones(max_len, dtype=np.float32)
    
    # Add batch dimension (TFLite expects batch size 1)
    keypoints_padded = keypoints_padded[np.newaxis, ...]  # [1, 64, 90, 2]
    mask = mask[np.newaxis, ...]                          # [1, 64]
    
    return keypoints_padded, mask


def postprocess_tflite_output(logits, return_top_k=5):
    """
    Process TFLite model output to get predictions.
    
    Args:
        logits: numpy array [1, 10] or [10] - raw model output
        return_top_k: return top K predictions (default: 5)
    
    Returns:
        dict with:
            - predicted_class: int (0-9)
            - confidence: float (softmax probability of predicted class)
            - top_k_classes: list of int (top K class indices)
            - top_k_confidences: list of float (top K probabilities)
            - all_probabilities: numpy array [10] (softmax of all classes)
    
    Example:
        >>> logits = interpreter.get_tensor(output_details[0]['index'])
        >>> result = postprocess_tflite_output(logits)
        >>> print(f"Predicted class: {result['predicted_class']}")
        >>> print(f"Confidence: {result['confidence']:.2%}")
    """
    # Remove batch dimension if present
    if logits.ndim == 2:
        logits = logits[0]
    
    # Apply softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits))  # subtract max for numerical stability
    probabilities = exp_logits / np.sum(exp_logits)
    
    # Get predicted class (argmax)
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])
    
    # Get top-k predictions
    top_k_indices = np.argsort(probabilities)[-return_top_k:][::-1]
    top_k_probs = probabilities[top_k_indices]
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top_k_classes': top_k_indices.tolist(),
        'top_k_confidences': top_k_probs.tolist(),
        'all_probabilities': probabilities
    }


def load_tflite_model(model_path):
    """
    Load a TFLite model and return interpreter ready for inference.
    
    Args:
        model_path: path to .tflite file
    
    Returns:
        interpreter: TFLite interpreter
        input_details: model input details
        output_details: model output details
    
    Example:
        >>> interpreter, input_details, output_details = load_tflite_model('model.tflite')
        >>> # Ready for inference!
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required for TFLite inference")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Verify expected input shape
    expected_shape = (1, MAX_SEQ_LEN, NUM_KEYPOINTS, 2)
    keypoints_input = next((d for d in input_details if len(d['shape']) == 4), None)
    
    if keypoints_input is None:
        raise ValueError("Could not find keypoints input in TFLite model")
    
    actual_shape = tuple(keypoints_input['shape'])
    if actual_shape != expected_shape:
        print(f"Warning: Expected input shape {expected_shape}, got {actual_shape}")
    
    return interpreter, input_details, output_details


def run_inference(interpreter, input_details, output_details, keypoints, attention_mask):
    """
    Run inference on TFLite model.
    
    Args:
        interpreter: TFLite interpreter
        input_details: model input details
        output_details: model output details
        keypoints: preprocessed keypoints [1, 64, 90, 2]
        attention_mask: preprocessed mask [1, 64]
    
    Returns:
        logits: raw model output [1, 10]
    
    Example:
        >>> kpts_input, mask_input = preprocess_for_tflite(keypoints)
        >>> logits = run_inference(interpreter, input_details, output_details, kpts_input, mask_input)
        >>> result = postprocess_tflite_output(logits)
    """
    # Find keypoints and mask inputs by shape
    keypoints_idx = None
    mask_idx = None
    
    for i, detail in enumerate(input_details):
        shape = detail['shape']
        if len(shape) == 4:  # keypoints: [1, 64, 90, 2]
            keypoints_idx = i
        elif len(shape) == 2:  # attention_mask: [1, 64]
            mask_idx = i
    
    if keypoints_idx is None or mask_idx is None:
        raise ValueError("Could not identify keypoints and mask inputs")
    
    # Set input tensors
    interpreter.set_tensor(input_details[keypoints_idx]['index'], keypoints.astype(np.float32))
    interpreter.set_tensor(input_details[mask_idx]['index'], attention_mask.astype(np.float32))
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    logits = interpreter.get_tensor(output_details[0]['index'])
    
    return logits


def inference_pipeline(model_path, keypoints):
    """
    Complete inference pipeline: preprocess → inference → postprocess.
    
    This is the main function you'll use in your app!
    
    Args:
        model_path: path to .tflite model (any LOSO variant)
        keypoints: raw keypoints from video [num_frames, 90, 2]
    
    Returns:
        result: dict with predicted class, confidence, and probabilities
    
    Example:
        >>> # Extract keypoints from video (your code)
        >>> keypoints = extract_keypoints_from_video('sign_video.mp4')  # [T, 90, 2]
        >>> 
        >>> # Run inference with ANY LOSO model
        >>> result = inference_pipeline('final_model_fp32.tflite', keypoints)
        >>> 
        >>> print(f"Predicted sign: {result['predicted_class']}")
        >>> print(f"Confidence: {result['confidence']:.2%}")
        >>> print(f"Top 5: {result['top_k_classes']}")
    """
    # Load model
    interpreter, input_details, output_details = load_tflite_model(model_path)
    
    # Preprocess
    kpts_input, mask_input = preprocess_for_tflite(keypoints)
    
    # Inference
    logits = run_inference(interpreter, input_details, output_details, kpts_input, mask_input)
    
    # Postprocess
    result = postprocess_tflite_output(logits)
    
    return result


# Example usage
if __name__ == "__main__":
    print("TFLite Preprocessing Utilities")
    print("="*80)
    print(f"MAX_SEQ_LEN: {MAX_SEQ_LEN} frames")
    print(f"NUM_KEYPOINTS: {NUM_KEYPOINTS} per frame")
    print(f"NUM_CLASSES: {NUM_CLASSES}")
    print()
    
    # Simulate keypoint extraction
    print("Example: Short video (38 frames)")
    keypoints_short = np.random.randn(38, 90, 2).astype(np.float32)
    kpts_padded, mask = preprocess_for_tflite(keypoints_short)
    print(f"  Input shape: {keypoints_short.shape}")
    print(f"  Output shape: {kpts_padded.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Valid frames: {int(mask.sum())} / {MAX_SEQ_LEN}")
    print()
    
    print("Example: Long video (80 frames)")
    keypoints_long = np.random.randn(80, 90, 2).astype(np.float32)
    kpts_padded, mask = preprocess_for_tflite(keypoints_long)
    print(f"  Input shape: {keypoints_long.shape}")
    print(f"  Output shape: {kpts_padded.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Valid frames: {int(mask.sum())} / {MAX_SEQ_LEN} (subsampled)")
    print()
    
    print("✓ All LOSO models accept this EXACT input shape!")
    print("  - Model input: float32[1, 64, 90, 2]")
    print("  - Model output: float32[1, 10]")
    print("="*80)

