#!/usr/bin/env python3
"""
evaluate_tflite_single.py

Evaluate a single TFLite model on any split of any dataset directory.

This is a more flexible version of test_tflite_models.py:
- You specify exactly ONE TFLite model to evaluate
- You specify which dataset root and which split name to use

Examples:
  # Evaluate FP32 full-dataset model on a LOSO test set
  python evaluate_tflite_single.py \
      --config_path configs/arabic-asl-90kpts.yaml \
      --data_path data/arabic-asl-90kpts_LOSO_user01 \
      --split test \
      --tflite_path checkpoints_arabic_asl_full/final_model_fp32.tflite

  # Evaluate any TFLite model on a custom split (e.g., "all")
  python evaluate_tflite_single.py \
      --config_path configs/arabic-asl-90kpts.yaml \
      --data_path data/arabic-asl-90kpts \
      --split all \
      --tflite_path exports/ptq_full/model_dynamic_int8.tflite
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf
import yaml

from dataset import SignDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a single TFLite model on a specified dataset split."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to model config YAML (used for joint_idx / keypoint groups).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to processed dataset root (e.g., data/arabic-asl-90kpts or *_LOSO_user01).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help='Dataset split to use (default: "test", e.g., "all", "test").',
    )
    parser.add_argument(
        "--tflite_path",
        type=str,
        required=True,
        help="Path to TFLite model to evaluate.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def determine_keypoint_groups(config_joint_idx):
    """
    Match the grouping logic used during training (body, left hand, right hand, face).
    Copied from test_tflite_models.py to ensure identical preprocessing.
    """
    if not config_joint_idx:
        return None

    sorted_idx = sorted(config_joint_idx)
    total_kpts = len(sorted_idx)
    groups = []

    if total_kpts >= 67:
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


def load_split_dataset(data_path, split, joint_groups, max_samples=None):
    """
    Load a specific split using SignDataset.

    This assumes the standard structure:
      {data_path}/{split}/G01, G02, ...
    """
    print(f"[DATA] Loading split '{split}' from root: {data_path}")
    dataset = SignDataset(
        root=data_path,
        split=split,
        shuffle=False,
        joint_idxs=joint_groups,
        augment=False,
    )
    if max_samples is not None:
        dataset.list_key = dataset.list_key[: max_samples]
    return dataset


def preprocess_sample(dataset, file_path, max_seq_len):
    keypoints, label = dataset.load_sample(file_path)
    seq_len = min(keypoints.shape[0], max_seq_len)

    padded_keypoints = np.zeros((max_seq_len, keypoints.shape[1], 2), dtype=np.float32)
    padded_keypoints[:seq_len] = keypoints[:seq_len]

    attention_mask = np.zeros((max_seq_len,), dtype=np.float32)
    attention_mask[:seq_len] = 1.0

    return padded_keypoints, attention_mask, label


def evaluate_tflite(model_path, dataset, max_seq_len):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    keypoints_idx = None
    mask_idx = None
    for idx, detail in enumerate(input_details):
        if len(detail["shape"]) == 4:
            keypoints_idx = idx
        elif len(detail["shape"]) in (2, 3):
            mask_idx = idx

    total = 0
    correct_top1 = 0
    correct_top5 = 0
    start_time = time.time()

    for file_path in dataset.list_key:
        keypoints, mask, label = preprocess_sample(dataset, file_path, max_seq_len)

        interpreter.set_tensor(input_details[keypoints_idx]["index"], keypoints[None, ...])
        interpreter.set_tensor(input_details[mask_idx]["index"], mask[None, ...])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]

        pred_top1 = np.argmax(output)
        pred_top5 = np.argsort(output)[-5:]

        correct_top1 += int(pred_top1 == label)
        correct_top5 += int(label in pred_top5)
        total += 1

    elapsed = time.time() - start_time
    acc_top1 = correct_top1 / total if total > 0 else 0.0
    acc_top5 = correct_top5 / total if total > 0 else 0.0
    size_mb = os.path.getsize(model_path) / (1024 ** 2)
    return acc_top1, acc_top5, elapsed, size_mb, total


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    joint_groups = determine_keypoint_groups(config.get("joint_idx", []))

    max_seq_len = 64  # fixed for TFLite models
    dataset = load_split_dataset(args.data_path, args.split, joint_groups, args.max_samples)
    num_samples = len(dataset.list_key)
    print(f"[DATA] Loaded {num_samples} samples from {args.data_path}/{args.split}")

    print(f"\n[MODEL] Evaluating TFLite model: {args.tflite_path}")
    top1, top5, runtime, size_mb, total = evaluate_tflite(args.tflite_path, dataset, max_seq_len)

    print("\n[RESULTS]")
    print(f"  Samples evaluated: {total}")
    print(f"  Top-1 Accuracy: {top1:.4f} ({top1*100:.2f}%)")
    print(f"  Top-5 Accuracy: {top5:.4f} ({top5*100:.2f}%)")
    print(f"  Inference time (total): {runtime:.2f} seconds")
    if total > 0:
        print(f"  Inference time per sample: {runtime / total * 1000:.2f} ms")
    print(f"  File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()


