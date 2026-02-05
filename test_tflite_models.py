#!/usr/bin/env python3
"""
test_tflite_models.py

Evaluate FP32 and dynamic-range INT8 TFLite models on a test split and compare accuracy.

Usage example:
    python test_tflite_models.py \
        --config_path configs/arabic-asl-90kpts.yaml \
        --data_path ~/signbart_tf/data/arabic-asl-90kpts_LOSO_user01 \
        --fp32_tflite checkpoints_arabic_asl_LOSO_user01/final_model_fp32.tflite \
        --dynamic_tflite checkpoints_arabic_asl_LOSO_user01/signbart_dynamic_range.tflite
"""
import argparse
import os
import time
import numpy as np
import tensorflow as tf
import yaml
from dataset import SignDataset
from utils import determine_keypoint_groups, load_label_maps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare FP32 vs PTQ vs QAT TFLite models on accuracy."
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model config YAML (unused but kept for parity)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to processed LOSO data split (e.g., *_LOSO_user01)")
    parser.add_argument("--fp32_tflite", type=str, required=True,
                        help="Path to FP32 TFLite model (final_model_fp32.tflite)")
    parser.add_argument("--ptq_tflite", type=str, required=True,
                        help="Path to PTQ dynamic-range TFLite model")
    parser.add_argument("--qat_tflite", type=str, default=None,
                        help="Optional path to QAT dynamic-range TFLite model")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of test samples to evaluate (default: all)")
    parser.add_argument("--seed", type=int, default=379,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_test_dataset(data_path, joint_groups, max_samples=None):
    dataset = SignDataset(
        root=data_path,
        split="test",
        shuffle=False,
        joint_idxs=joint_groups,
        augment=False
    )
    if max_samples is not None:
        dataset.list_key = dataset.list_key[:max_samples]
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
    acc_top1 = correct_top1 / total
    acc_top5 = correct_top5 / total
    size_mb = os.path.getsize(model_path) / (1024 ** 2)
    return acc_top1, acc_top5, elapsed, size_mb


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    joint_groups = determine_keypoint_groups(config.get("joint_idx", []))
    _, _, label2name = load_label_maps(args.data_path)
    if label2name:
        example_key = sorted(label2name.keys())[0]
        print(f"[LABELS] Loaded label names (e.g., {example_key} -> {label2name[example_key]})")

    max_seq_len = 64  # fixed for TFLite models
    dataset = load_test_dataset(args.data_path, joint_groups, args.max_samples)
    print(f"[DATA] Loaded {len(dataset.list_key)} test samples from {args.data_path}")

    results = []

    print("\n[FP32] Evaluating FP32 TFLite model...")
    fp32 = evaluate_tflite(args.fp32_tflite, dataset, max_seq_len)
    results.append(("FP32", args.fp32_tflite, *fp32))
    print(f"  Top-1 Accuracy: {fp32[0]:.4f} ({fp32[0]*100:.2f}%)")
    print(f"  Top-5 Accuracy: {fp32[1]:.4f} ({fp32[1]*100:.2f}%)")
    print(f"  Inference time (total): {fp32[2]:.2f} seconds")
    print(f"  File size: {fp32[3]:.2f} MB")

    print("\n[PTQ] Evaluating dynamic-range TFLite model...")
    ptq = evaluate_tflite(args.ptq_tflite, dataset, max_seq_len)
    results.append(("PTQ", args.ptq_tflite, *ptq))
    print(f"  Top-1 Accuracy: {ptq[0]:.4f} ({ptq[0]*100:.2f}%)")
    print(f"  Top-5 Accuracy: {ptq[1]:.4f} ({ptq[1]*100:.2f}%)")
    print(f"  Inference time (total): {ptq[2]:.2f} seconds")
    print(f"  File size: {ptq[3]:.2f} MB")

    if args.qat_tflite:
        print("\n[QAT] Evaluating QAT dynamic-range TFLite model...")
        qat = evaluate_tflite(args.qat_tflite, dataset, max_seq_len)
        results.append(("QAT", args.qat_tflite, *qat))
        print(f"  Top-1 Accuracy: {qat[0]:.4f} ({qat[0]*100:.2f}%)")
        print(f"  Top-5 Accuracy: {qat[1]:.4f} ({qat[1]*100:.2f}%)")
        print(f"  Inference time (total): {qat[2]:.2f} seconds")
        print(f"  File size: {qat[3]:.2f} MB")

    print("\n[SUMMARY]")
    for tag, path, top1, top5, runtime, size in results:
        print(f"  {tag:<4} | Top-1 {top1*100:6.2f}% | Top-5 {top5*100:6.2f}% | "
              f"Time {runtime:6.2f}s | Size {size:5.2f} MB | {path}")


if __name__ == "__main__":
    main()


