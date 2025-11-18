#!/usr/bin/env python3
"""
train_loso_functional_qat.py

QAT fine-tuning pipeline:
1. Load trained Functional SignBART model (.h5/.keras) for a LOSO split
2. Annotate selectively for QAT (Dense layers only)
3. Fine-tune for N epochs on the same LOSO train split
4. Evaluate on test split and export dynamic-range INT8 TFLite

Example:
    python train_loso_functional_qat.py \
        --config_path configs/arabic-asl-90kpts.yaml \
        --data_path ~/signbart_tf/data/arabic-asl-90kpts_LOSO_user01 \
        --checkpoint checkpoints_arabic_asl_LOSO_user01/final_model.h5 \
        --qat_epochs 3 \
        --output_dir exports/qat_finetune_user01
"""
import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import yaml

from dataset import SignDataset
from model_functional import build_signbart_functional_with_dict_inputs, ExtractLastValidToken
from model_functional_tflite import build_signbart_functional_tflite, ExtractLastValidTokenTFLite
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from layers import Projection, ClassificationHead, PositionalEmbedding
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from attention import SelfAttention, CrossAttention, CausalSelfAttention

MAX_SEQ_LEN = 64


@tf.keras.utils.register_keras_serializable()
class Top5Accuracy(keras.metrics.SparseTopKCategoricalAccuracy):
    """Top-5 accuracy metric using Keras built-in."""
    def __init__(self, name="top5_accuracy", **kwargs):
        super().__init__(k=5, name=name, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune trained SignBART model with QAT and export TFLite."
    )
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to LOSO dataset (e.g., .../arabic-asl-90kpts_LOSO_user01)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Trained Functional model (.h5/.keras)")
    parser.add_argument("--output_dir", type=str, default="exports/qat_finetune")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for QAT (default: 4, larger than training for stability)")
    parser.add_argument("--qat_epochs", type=int, default=10,
                        help="Number of QAT fine-tuning epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for QAT (default: 5e-5, ~4x lower than FP32 training)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize_dense_names", nargs="*", default=None,
                        help="Dense layer name substrings to quantize. "
                             "Default: all Dense layers including FFN (fc1,fc2), attention projections "
                             "(q_proj,k_proj,v_proj,out_proj), and projection layers (proj_x1,proj_y1). "
                             "NOTE: The Projection container itself is excluded (not wrapped), but its "
                             "internal Dense layers ARE quantized.")
    parser.add_argument("--no_validation", action="store_true",
                        help="Disable validation during QAT training (monitor training loss for scheduler)")
    parser.add_argument("--scheduler_factor", type=float, default=0.1,
                        help="Factor for ReduceLROnPlateau scheduler (default: 0.1)")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for ReduceLROnPlateau scheduler (default: 5)")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="Patience for EarlyStopping (default: 10, should be > scheduler_patience)")
    parser.add_argument(
        "--freeze_mode",
        type=str,
        choices=["none", "head", "all"],
        default="none",
        help=(
            "Layer freezing strategy during QAT fine-tuning: "
            "'none' = all layers trainable (original behavior), "
            "'head' = only classification head trainable, "
            "'all' = all layers frozen (calibration-only QAT)."
        ),
    )
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def determine_keypoint_groups(config_joint_idx):
    if not config_joint_idx:
        return None
    sorted_idx = sorted(config_joint_idx)
    total = len(sorted_idx)
    groups = []
    if total >= 67:
        face = sorted_idx[-25:]
        right_hand = sorted_idx[-46:-25]
        left_hand = sorted_idx[-67:-46]
        body = sorted_idx[:-67]
        if body:
            groups.append(body)
        if left_hand:
            groups.append(left_hand)
        if right_hand:
            groups.append(right_hand)
        if face:
            groups.append(face)
    else:
        groups.append(sorted_idx)
    return groups


def create_dataset(root, joint_groups, batch_size, split, augment):
    dataset = SignDataset(
        root=root,
        split=split,
        shuffle=(split == "train"),
        joint_idxs=joint_groups,
        augment=augment
    )
    ds = dataset.create_tf_dataset(batch_size=batch_size, drop_remainder=False)

    def split_batch(batch):
        inputs = {
            "keypoints": batch["keypoints"],
            "attention_mask": batch["attention_mask"],
        }
        labels = batch["labels"]
        return inputs, labels

    ds = ds.map(split_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


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
        "ExtractLastValidTokenTFLite": ExtractLastValidTokenTFLite,
        "Top5Accuracy": Top5Accuracy,
    }


class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


CUSTOM_LAYER_TYPES = (
    Projection,
    Encoder,
    Decoder,
    ClassificationHead,
    ExtractLastValidToken,
    PositionalEmbedding,
    EncoderLayer,
    DecoderLayer,
    SelfAttention,
    CrossAttention,
    CausalSelfAttention,
)

# Layers that should NEVER be wrapped with quantization wrappers
# CRITICAL FINDING: The Projection CONTAINER must not be wrapped (causes training collapse)
# However, its internal Dense layers (proj_x1, proj_y1) CAN and SHOULD be quantized
# via the quantize_dense_names filters - they're safe when the container is unwrapped
EXCLUDE_FROM_QAT = (
    Projection,  # Container must NOT be wrapped (tuple output handling issue)
                 # But proj_x1, proj_y1 Dense layers inside ARE quantized via filters
)


def record_internal_dense_layers(layer, parent_name, dense_filters, dense_log, skipped_dense_log, visited, seen_dense_layers):
    """Track Dense layers, deduplicating by layer object ID to avoid counting same layer multiple times."""
    if layer is None:
        return
    obj_id = id(layer)
    if obj_id in visited:
        return
    visited.add(obj_id)

    if isinstance(layer, (list, tuple)):
        for idx, sub in enumerate(layer):
            record_internal_dense_layers(sub, f"{parent_name}/{idx}" if parent_name else str(idx),
                                         dense_filters, dense_log, skipped_dense_log, visited, seen_dense_layers)
        return
    if isinstance(layer, dict):
        for key, sub in layer.items():
            record_internal_dense_layers(sub, f"{parent_name}/{key}" if parent_name else str(key),
                                         dense_filters, dense_log, skipped_dense_log, visited, seen_dense_layers)
        return

    if hasattr(layer, "layers") and layer.layers:
        for sublayer in layer.layers:
            record_internal_dense_layers(
                sublayer,
                f"{parent_name}/{sublayer.name}" if parent_name else sublayer.name,
                dense_filters,
                dense_log,
                skipped_dense_log,
                visited,
                seen_dense_layers,
            )
        return

    for attr_name in dir(layer):
        if attr_name.startswith("_"):
            continue
        # Skip graph traversal attributes that cause duplicates
        if attr_name in ["inbound_nodes", "outbound_nodes", "input", "output", "inputs", "outputs"]:
            continue
        try:
            attr = getattr(layer, attr_name)
        except Exception:
            continue

        if isinstance(attr, keras.layers.Dense):
            # Deduplicate by layer object ID, not just name
            dense_id = id(attr)
            if dense_id not in seen_dense_layers:
                seen_dense_layers.add(dense_id)
                name = f"{parent_name}/{attr.name}" if parent_name else attr.name
                if any(f in attr.name for f in dense_filters):
                    dense_log.append(name)
                else:
                    skipped_dense_log.append(name)
        elif isinstance(attr, (list, tuple, dict)):
            record_internal_dense_layers(
                attr, f"{parent_name}/{attr_name}" if parent_name else attr_name,
                dense_filters, dense_log, skipped_dense_log, visited, seen_dense_layers,
            )
        elif isinstance(attr, keras.layers.Layer) and not isinstance(attr, keras.Model):
            record_internal_dense_layers(
                attr, f"{parent_name}/{attr.name}" if parent_name else attr.name,
                dense_filters, dense_log, skipped_dense_log, visited, seen_dense_layers,
            )


def annotate_dense_layers(layer, dense_filters, dense_log, skipped_dense_log, container_log):
    # First check if this layer should be completely excluded from QAT
    if isinstance(layer, EXCLUDE_FROM_QAT):
        print(f"  ℹ️  EXCLUDING {layer.__class__.__name__} container from wrapping (its Dense layers will still be quantized)")
        # Don't wrap it at all, but still record its internal layers for quantization via filters
        container_name = layer.name or layer.__class__.__name__
        record_internal_dense_layers(
            layer,
            container_name,
            dense_filters,
            dense_log,
            skipped_dense_log,
            visited=set(),
            seen_dense_layers=set()
        )
        return layer  # Return unwrapped
    
    if isinstance(layer, keras.layers.Dense):
        if any(name in layer.name for name in dense_filters):
            dense_log.append(layer.name)
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        else:
            skipped_dense_log.append(layer.name)
        return layer

    # Wrap custom containers with NoOpQuantizeConfig so TF-MOT can traverse into them
    # The NoOp means: "don't quantize the container itself, but DO traverse and quantize marked Dense layers inside"
    # This is necessary because clone_model doesn't automatically recurse into custom layer internals
    if isinstance(layer, CUSTOM_LAYER_TYPES):
        container_name = layer.name or layer.__class__.__name__
        container_log.append(container_name)
        
        # NoOpQuantizeConfig = container wrapper is transparent (FP32 passthrough)
        # But TF-MOT will still find and quantize individual Dense layers we annotated inside
        wrapped = tfmot.quantization.keras.quantize_annotate_layer(
            layer,
            quantize_config=NoOpQuantizeConfig(),
        )
        
        record_internal_dense_layers(
            layer,
            container_name,
            dense_filters,
            dense_log,
            skipped_dense_log,
            visited=set(),
            seen_dense_layers=set()
        )
        return wrapped
    
    return layer


def build_qat_model(base_model, dense_filters):
    dense_log = []
    skipped_dense_log = []
    container_log = []
    custom_objects = get_custom_objects()
    custom_objects["NoOpQuantizeConfig"] = NoOpQuantizeConfig

    with keras.utils.custom_object_scope(custom_objects):
        annotated = keras.models.clone_model(
            base_model,
            clone_function=lambda layer: annotate_dense_layers(layer, dense_filters, dense_log, skipped_dense_log, container_log)
        )
    with keras.utils.custom_object_scope(custom_objects):
        qat_model = tfmot.quantization.keras.quantize_apply(annotated)
    return qat_model, dense_log, container_log


def copy_weights(source, target):
    source_layers = {layer.name: layer for layer in source.layers}
    for layer in target.layers:
        name = layer.name
        src = source_layers.get(name) or source_layers.get(f"quant_{name}")
        if src is None:
            continue
        if isinstance(src, QuantizeWrapper):
            weights = src.layer.get_weights()
        else:
            weights = src.get_weights()
        if weights:
            layer.set_weights(weights)


def export_dynamic_tflite(model, config, output_path):
    print("[INT8] Building TFLite-friendly model for export...")
    dummy = {
        "keypoints": tf.random.normal((1, 10, len(config["joint_idx"]), 2)),
        "attention_mask": tf.ones((1, 10)),
    }

    float_model = build_signbart_functional_with_dict_inputs(config)
    _ = float_model(dummy, training=False)
    copy_weights(model, float_model)

    tflite_model = build_signbart_functional_tflite(config)
    _ = tflite_model(dummy, training=False)
    copy_weights(float_model, tflite_model)
    print("✓ Weights copied to TFLite-friendly graph.")

    num_keypoints = len(config["joint_idx"])

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, MAX_SEQ_LEN, num_keypoints, 2], dtype=tf.float32, name="keypoints"),
        tf.TensorSpec(shape=[1, MAX_SEQ_LEN], dtype=tf.float32, name="attention_mask"),
    ])
    def serving_fn(keypoints, attention_mask):
        return tflite_model({"keypoints": keypoints, "attention_mask": attention_mask}, training=False)

    concrete_fn = serving_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]

    tflite_bytes = converter.convert()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_bytes)

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"✓ Saved dynamic-range TFLite to {output_path} ({size_mb:.2f} MB)")


def main():
    args = parse_args()
    set_seed(args.seed)
    config = load_config(args.config_path)
    joint_groups = determine_keypoint_groups(config.get("joint_idx", []))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[DATA] Preparing datasets...")
    train_ds = create_dataset(args.data_path, joint_groups, args.batch_size, "train", augment=True)
    test_ds = create_dataset(args.data_path, joint_groups, args.batch_size, "test", augment=False)

    print(f"[LOAD] Loading base model from {args.checkpoint}")
    custom_objects = get_custom_objects()
    base_model = keras.models.load_model(args.checkpoint, custom_objects=custom_objects)
    print("✓ Base model loaded.")
    
    # Verify checkpoint is FP32 (not already QAT)
    from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
    has_qat_layers = any(isinstance(layer, QuantizeWrapper) for layer in base_model.layers)
    if has_qat_layers:
        print("  ⚠️  WARNING: Checkpoint appears to already have QAT layers!")
        print("     This checkpoint may be from a previous QAT run.")
        print("     Please use the original FP32 checkpoint (final_model.h5) instead.")
    else:
        print("  ✓ Checkpoint verified as FP32 (no QAT layers found)")

    # Default: Quantize ALL Dense layers (proven stable through experimentation)
    # Key finding: The Projection CONTAINER must be excluded (see EXCLUDE_FROM_QAT),
    # but its internal Dense layers (proj_x1, proj_y1) can be safely quantized
    dense_filters = args.quantize_dense_names or [
        "fc1", "fc2",                                    # FFN layers
        "q_proj", "k_proj", "v_proj", "out_proj",       # Attention projections (safe!)
        "proj_x1", "proj_y1",                            # Projection Dense layers (safe!)
    ]
    print(f"[QAT] Quantization filters: {dense_filters}")
    print(f"  ℹ️  Quantizing all Dense layers (FFN + attention projections + projection layers)")
    print(f"  ✓  Projection container excluded from wrapping (critical for stability)")
    print("[SUMMARY] Base model before QAT annotation:")
    try:
        base_model.summary(line_length=96)
    except Exception as e:
        print(f"  Unable to display summary: {e}")

    print("[QAT] Annotating model...")
    qat_model, dense_log, container_log = build_qat_model(base_model, dense_filters)
    
    dense_summary = sorted(set(dense_log))
    
    # Verify that quantization is working as expected
    # NOTE: Attention projections ARE safe to quantize (experimentally verified)
    # The critical requirement is that the Projection CONTAINER is not wrapped
    attention_layers = [n for n in dense_summary if any(x in n for x in ["q_proj", "k_proj", "v_proj"])]
    if attention_layers:
        print(f"\n  ℹ️  Quantizing {len(attention_layers)} attention projection layers (safe, verified)")
    
    projection_layers = [n for n in dense_summary if any(x in n for x in ["proj_x1", "proj_y1"])]
    if projection_layers:
        print(f"  ℹ️  Quantizing {len(projection_layers)} projection Dense layers (safe, container excluded)")
    
    if dense_summary:
        print(f"\n✓ QUANTIZED Dense layers ({len(dense_summary)} unique layers):")
        # Group by layer type for clarity
        ffn_layers = [n for n in dense_summary if any(x in n for x in ["fc1", "fc2"])]
        other_layers = [n for n in dense_summary if n not in ffn_layers]
        
        if ffn_layers:
            print(f"  FFN layers ({len(ffn_layers)}):")
            for name in sorted(ffn_layers):
                print(f"    • {name}")
        if other_layers:
            print(f"  Other layers ({len(other_layers)}):")
            for name in sorted(other_layers):
                print(f"    • {name}")
    else:
        print("\n⚠️  WARNING: No Dense layers matched filters!")
    
    if container_log:
        print(f"\n  Containers wrapped ({len(set(container_log))}):")
        for name in sorted(set(container_log)):
            print(f"    - {name}")

    print("\n[SUMMARY] QAT model after annotation:")
    try:
        qat_model.summary(line_length=96)
    except Exception as e:
        print(f"  Unable to display QAT summary: {e}")

    # Optionally freeze layers to stabilize QAT fine-tuning
    if args.freeze_mode == "all":
        for layer in qat_model.layers:
            layer.trainable = False
        print("[QAT] Freeze mode: all layers frozen (calibration-only QAT).")
    elif args.freeze_mode == "head":
        frozen, trainable = 0, 0
        for layer in qat_model.layers:
            # Keep only the classification head trainable
            if "classification_head" in layer.name:
                layer.trainable = True
                trainable += 1
            else:
                layer.trainable = False
                frozen += 1
        print(f"[QAT] Freeze mode: head-only "
              f"(trainable layers: {trainable}, frozen layers: {frozen}).")
    else:
        print("[QAT] Freeze mode: none (all layers trainable).")

    # Add gradient clipping to make QAT updates more stable
    optimizer = keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0)
    qat_model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            Top5Accuracy(name="top5_accuracy"),
        ]
    )

    # Setup callbacks
    callbacks = []
    
    # Learning rate scheduler
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss' if not args.no_validation else 'loss',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        verbose=1,
        min_lr=1e-7
    )
    callbacks.append(reduce_lr_callback)

    # Early stopping with best-weight restoration
    # Uses longer patience than scheduler to allow training with reduced LR
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss' if not args.no_validation else 'loss',
        patience=args.early_stop_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)
    
    print("\n[TRAIN] Starting QAT fine-tuning...")
    print(f"  Learning rate scheduler: ReduceLROnPlateau")
    print(f"    Monitor: {'val_loss' if not args.no_validation else 'loss'}")
    print(f"    Factor: {args.scheduler_factor}")
    print(f"    Patience: {args.scheduler_patience}")
    print(f"  Early stopping:")
    print(f"    Monitor: {'val_loss' if not args.no_validation else 'loss'}")
    print(f"    Patience: {args.early_stop_patience} (gives model chance to train with reduced LR)")
    
    history = qat_model.fit(
        train_ds,
        validation_data=None if args.no_validation else test_ds,
        epochs=args.qat_epochs,
        verbose=1,
        callbacks=callbacks
    )

    print("\n[EVAL] Evaluating QAT model on test set...")
    results = qat_model.evaluate(test_ds, return_dict=True)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    qat_path = output_dir / "qat_model.keras"
    qat_model.save(qat_path)
    print(f"✓ Saved QAT Keras model to {qat_path}")

    tflite_path = output_dir / "qat_dynamic_int8.tflite"
    export_dynamic_tflite(qat_model, config, tflite_path)

    print("\n[SUMMARY]")
    print(f"  Checkpoint used : {args.checkpoint}")
    print(f"  QAT epochs      : {args.qat_epochs}")
    print(f"  QAT model       : {qat_path}")
    print(f"  Dynamic TFLite  : {tflite_path}")


if __name__ == "__main__":
    main()


