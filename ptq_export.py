#!/usr/bin/env python3
"""
ptq_export.py

Post-training dynamic-range quantization for trained SignBART models.
Loads a trained .h5/.keras checkpoint, converts to TFLite FP32 and dynamic INT8 (weights only),
and saves both artifacts for deployment.

Example:
    python ptq_export.py \
        --config_path configs/arabic-asl-90kpts.yaml \
        --checkpoint checkpoints_arabic_asl_LOSO_user01/final_model.h5 \
        --output_dir exports/ptq_arabic_asl_user01
"""
import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import yaml

from model_functional import build_signbart_functional_with_dict_inputs, ExtractLastValidToken
from utils import ensure_dir_safe, resolve_output_base
from layers import Projection, ClassificationHead, PositionalEmbedding
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from attention import SelfAttention, CrossAttention, CausalSelfAttention


@tf.keras.utils.register_keras_serializable()
class Top5Accuracy(keras.metrics.SparseTopKCategoricalAccuracy):
    """Top-5 accuracy metric compatible with saved models (k configurable)."""

    def __init__(self, k=5, name="top5_accuracy", **kwargs):
        # Accept k from config during deserialization; default to 5
        super().__init__(k=k, name=name, **kwargs)


MAX_SEQ_LEN = 64


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load trained SignBART model and export FP32 + dynamic-range INT8 TFLite."
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model (.h5 or .keras).")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Output directory for TFLite models (default: outputs/<dataset>/<run_type>/exports/ptq).")
    parser.add_argument("--dataset_name", type=str, default="",
                        help="Dataset name for output separation (e.g., arabic_asl, lsa64).")
    parser.add_argument("--run_type", type=str, default="full",
                        help="Run type for output separation (default: full).")
    parser.add_argument("--output_root", type=str, default="",
                        help="Base output directory (default: outputs or SIGNBART_OUTPUT_ROOT).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_custom_objects():
    return {
        "Projection": Projection,
        "ClassificationHead": ClassificationHead,
        "PositionalEmbedding": PositionalEmbedding,
        "Encoder": Encoder,
        "EncoderLayer": EncoderLayer,
        "Decoder": Decoder,
        "DecoderLayer": DecoderLayer,
        "SelfAttention": SelfAttention,
        "CrossAttention": CrossAttention,
        "CausalSelfAttention": CausalSelfAttention,
        "ExtractLastValidToken": ExtractLastValidToken,
        "Top5Accuracy": Top5Accuracy,
    }


def load_trained_model(checkpoint_path):
    print(f"[LOAD] Loading trained model from {checkpoint_path}")
    custom_objects = get_custom_objects()
    model = keras.models.load_model(checkpoint_path, custom_objects=custom_objects)
    print("✓ Trained model loaded.")
    return model


def export_tflite(model, config, output_path, dynamic_range=False):
    num_keypoints = len(config["joint_idx"])

    @tf.function(input_signature=[
        {
            "keypoints": tf.TensorSpec(shape=[1, MAX_SEQ_LEN, num_keypoints, 2], dtype=tf.float32),
            "attention_mask": tf.TensorSpec(shape=[1, MAX_SEQ_LEN], dtype=tf.float32),
        }
    ])
    def serving_fn(inputs):
        return model(inputs, training=False)

    concrete_fn = serving_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])

    if dynamic_range:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("[INT8] Dynamic-range quantization enabled (weights INT8, activations FP32).")

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()
    ensure_dir_safe(Path(output_path).parent)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"✓ Saved TFLite model to {output_path} ({size_mb:.2f} MB)")


def main():
    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.config_path)
    trained_model = load_trained_model(args.checkpoint)

    if not args.output_dir:
        output_base = resolve_output_base(
            dataset_name=args.dataset_name or None,
            run_type=args.run_type or None,
            output_root=args.output_root or None,
        )
        if output_base:
            args.output_dir = str(output_base / "exports" / "ptq")
        else:
            args.output_dir = "exports/ptq_export"

    num_keypoints = len(config["joint_idx"])
    dummy = {
        "keypoints": tf.random.normal((1, 10, num_keypoints, 2)),
        "attention_mask": tf.ones((1, 10)),
    }
    _ = trained_model(dummy, training=False)

    output_dir = Path(args.output_dir)
    ensure_dir_safe(output_dir)

    fp32_path = output_dir / "model_fp32.tflite"
    export_tflite(trained_model, config, fp32_path, dynamic_range=False)

    dynamic_path = output_dir / "model_dynamic_int8.tflite"
    export_tflite(trained_model, config, dynamic_path, dynamic_range=True)

    print("\nExport complete!")
    print(f"  FP32   : {fp32_path}")
    print(f"  INT8DR : {dynamic_path}")


if __name__ == "__main__":
    main()


