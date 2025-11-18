#!/usr/bin/env python3
"""
verify_qat.py

Diagnostic script to verify that QAT is actually being applied correctly.
This will:
1. Check if QuantizeWrapper layers exist in the model
2. Verify fake quantization nodes are in the graph
3. Compare weight statistics between QAT and non-QAT models
4. Test if quantization affects forward pass
"""
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from train_loso_functional_qat import (
    get_custom_objects,
    build_qat_model,
    load_config,
)


def count_quantize_wrappers(model):
    """Count QuantizeWrapper layers in model."""
    count = 0
    wrapper_names = []
    for layer in model.layers:
        if isinstance(layer, QuantizeWrapper):
            count += 1
            wrapper_names.append(layer.name)
    return count, wrapper_names


def check_fake_quant_in_graph(model):
    """Check if fake quantization ops exist in the computation graph."""
    # Try to find FakeQuant operations in the graph
    concrete_func = tf.function(lambda x: model(x, training=True)).get_concrete_function(
        {
            "keypoints": tf.TensorSpec(shape=[1, 10, 90, 2], dtype=tf.float32),
            "attention_mask": tf.TensorSpec(shape=[1, 10], dtype=tf.float32),
        }
    )
    
    graph_def = concrete_func.graph.as_graph_def()
    fake_quant_ops = [node.name for node in graph_def.node if "FakeQuant" in node.op]
    
    return len(fake_quant_ops) > 0, fake_quant_ops


def test_quantization_effect(base_model, qat_model, config):
    """Test if QAT model produces different outputs than base model."""
    num_keypoints = len(config["joint_idx"])
    
    # Create dummy input
    dummy_input = {
        "keypoints": tf.random.normal((2, 10, num_keypoints, 2), seed=42),
        "attention_mask": tf.ones((2, 10)),
    }
    
    # Get outputs
    base_output = base_model(dummy_input, training=False)
    qat_output = qat_model(dummy_input, training=False)
    
    # Check if outputs are different
    max_diff = tf.reduce_max(tf.abs(base_output - qat_output)).numpy()
    mean_diff = tf.reduce_mean(tf.abs(base_output - qat_output)).numpy()
    
    return max_diff, mean_diff


def inspect_dense_layer_weights(model, layer_name_filter="fc1"):
    """Inspect weights of Dense layers to see if they show quantization effects."""
    dense_weights = []
    for layer in model.layers:
        if isinstance(layer, QuantizeWrapper):
            inner_layer = layer.layer
            if hasattr(inner_layer, 'name') and layer_name_filter in inner_layer.name:
                if isinstance(inner_layer, keras.layers.Dense):
                    weights = inner_layer.get_weights()[0]  # Get kernel weights
                    dense_weights.append({
                        'name': inner_layer.name,
                        'shape': weights.shape,
                        'mean': np.mean(weights),
                        'std': np.std(weights),
                        'min': np.min(weights),
                        'max': np.max(weights),
                        'unique_values': len(np.unique(np.round(weights, 6))),
                    })
    return dense_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained FP32 model checkpoint")
    args = parser.parse_args()
    
    print("="*80)
    print("QAT VERIFICATION DIAGNOSTIC")
    print("="*80)
    print()
    
    # Load config and base model
    config = load_config(args.config_path)
    print(f"[1/5] Loading base model from {args.checkpoint}...")
    custom_objects = get_custom_objects()
    base_model = keras.models.load_model(args.checkpoint, custom_objects=custom_objects)
    print(f"      ✓ Base model loaded: {len(base_model.layers)} layers")
    
    # Check if base model has QAT
    base_qat_count, _ = count_quantize_wrappers(base_model)
    if base_qat_count > 0:
        print(f"      ⚠️  WARNING: Base model already has {base_qat_count} QuantizeWrapper layers!")
        print("          This checkpoint may be from a previous QAT run.")
        return
    else:
        print(f"      ✓ Base model is FP32 (no QuantizeWrappers)")
    print()
    
    # Build QAT model
    print("[2/5] Building QAT model with annotation...")
    dense_filters = ["fc1", "fc2"]
    qat_model, dense_log, container_log = build_qat_model(base_model, dense_filters)
    print(f"      ✓ QAT model built: {len(qat_model.layers)} layers")
    print(f"      ✓ Dense layers matched by filters: {len(set(dense_log))}")
    print()
    
    # Count QuantizeWrapper layers
    print("[3/5] Checking for QuantizeWrapper layers...")
    qat_count, wrapper_names = count_quantize_wrappers(qat_model)
    if qat_count > 0:
        print(f"      ✓ VERIFIED: Found {qat_count} QuantizeWrapper layers in QAT model")
        print(f"      ✓ This confirms QAT annotation was applied!")
        print(f"\n      First 10 QuantizeWrapper layers:")
        for name in wrapper_names[:10]:
            print(f"        • {name}")
        if len(wrapper_names) > 10:
            print(f"        ... and {len(wrapper_names) - 10} more")
    else:
        print(f"      ✗ ERROR: No QuantizeWrapper layers found!")
        print(f"      ✗ QAT annotation may have failed!")
        return
    print()
    
    # Check for FakeQuant ops in graph
    print("[4/5] Checking for FakeQuant operations in computation graph...")
    try:
        has_fake_quant, fake_quant_ops = check_fake_quant_in_graph(qat_model)
        if has_fake_quant:
            print(f"      ✓ VERIFIED: Found {len(fake_quant_ops)} FakeQuant operations in graph")
            print(f"      ✓ This confirms quantization simulation is active during training!")
            print(f"\n      First 5 FakeQuant ops:")
            for op in fake_quant_ops[:5]:
                print(f"        • {op}")
        else:
            print(f"      ⚠️  WARNING: No FakeQuant operations found in graph")
            print(f"          QAT may not be working correctly")
    except Exception as e:
        print(f"      ⚠️  Could not check graph: {e}")
    print()
    
    # Test quantization effect on forward pass
    print("[5/5] Testing if QAT affects model outputs...")
    try:
        max_diff, mean_diff = test_quantization_effect(base_model, qat_model, config)
        print(f"      Output difference (base vs QAT):")
        print(f"        Max difference:  {max_diff:.6f}")
        print(f"        Mean difference: {mean_diff:.6f}")
        
        if max_diff > 1e-5:
            print(f"      ✓ VERIFIED: QAT model produces different outputs!")
            print(f"      ✓ Quantization simulation IS affecting the forward pass!")
        else:
            print(f"      ⚠️  WARNING: Outputs are nearly identical")
            print(f"          QAT may not be properly configured")
    except Exception as e:
        print(f"      ⚠️  Could not compare outputs: {e}")
    print()
    
    # Summary
    print("="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print()
    
    if qat_count > 0 and has_fake_quant and max_diff > 1e-5:
        print("✓ QAT IS WORKING CORRECTLY:")
        print("  • QuantizeWrapper layers found in model")
        print("  • FakeQuant operations present in graph")
        print("  • Quantization affects forward pass")
        print()
        print("⚠️  HOWEVER, your training is collapsing (95% → 11% accuracy)!")
        print()
        print("POSSIBLE CAUSES OF TRAINING COLLAPSE:")
        print("  1. Learning rate too high for QAT (try 1e-5 or 5e-6 instead of 5e-5)")
        print("  2. Batch size too small (batch_size=1 may be unstable for QAT)")
        print("  3. ReduceLROnPlateau scheduler too aggressive (patience=5 may be too short)")
        print("  4. Need more epochs to stabilize (QAT often needs 10-20 epochs)")
        print("  5. Quantization filters may still be catching sensitive layers")
        print()
        print("RECOMMENDATIONS:")
        print("  1. Reduce learning rate: --lr 1e-5 (10x lower)")
        print("  2. Increase batch size: --batch_size 4 or 8")
        print("  3. Increase scheduler patience: --scheduler_patience 10")
        print("  4. Disable scheduler entirely for first few epochs to see raw QAT behavior")
        print("  5. Try --qat_epochs 10 to give model time to recover")
    else:
        print("⚠️  QAT MAY NOT BE WORKING PROPERLY:")
        if qat_count == 0:
            print("  ✗ No QuantizeWrapper layers found")
        if not has_fake_quant:
            print("  ✗ No FakeQuant operations in graph")
        if max_diff <= 1e-5:
            print("  ✗ Quantization not affecting outputs")
        print()
        print("Check the logs above for details.")
    
    print("="*80)


if __name__ == "__main__":
    main()

