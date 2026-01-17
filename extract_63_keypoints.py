#!/usr/bin/env python
"""
extract_63_keypoints.py - Extract 63 keypoints for v2.1 (Pose subset + Hands + Face subset)

Keypoint Structure (per frame, ordered):
    Indices 0-14:  Pose subset (15 keypoints)
    Indices 15-35: Left hand (21 keypoints)
    Indices 36-56: Right hand (21 keypoints)
    Indices 57-62: Face subset (6 keypoints)

This matches READMEv2.1 / TODOlistv2.1 and on-device ordering:
    Pose(15) + L-Hand(21) + R-Hand(21) + Face(6)

Usage:
    python extract_63_keypoints.py --input_dir videos/ --output_dir data/arabic-asl-63kpts/
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import json
from pathlib import Path
import argparse
from tqdm import tqdm

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Pose subset indices (15 total) - in this exact order
POSE_SUBSET_INDICES = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16]

# Face subset indices (6 total) - in this exact order
FACE_SUBSET_INDICES = [10, 338, 297, 67, 234, 454]

# Total keypoints: 63
NUM_KEYPOINTS = 63


def extract_keypoints_from_results(results):
    """
    Extract 63 keypoints from MediaPipe Holistic results.

    Returns:
        keypoints: numpy array of shape (63, 2) containing [x, y] normalized to [0,1]
    """
    keypoints = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)

    # Pose subset (15 keypoints: indices 0-14)
    if results.pose_landmarks:
        for out_i, pose_i in enumerate(POSE_SUBSET_INDICES):
            if pose_i < len(results.pose_landmarks.landmark):
                landmark = results.pose_landmarks.landmark[pose_i]
                keypoints[out_i] = [landmark.x, landmark.y]

    # Left hand (21 keypoints: indices 15-35)
    if results.left_hand_landmarks:
        for i, landmark in enumerate(results.left_hand_landmarks.landmark):
            keypoints[15 + i] = [landmark.x, landmark.y]

    # Right hand (21 keypoints: indices 36-56)
    if results.right_hand_landmarks:
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            keypoints[36 + i] = [landmark.x, landmark.y]

    # Face subset (6 keypoints: indices 57-62)
    if results.face_landmarks:
        face_landmarks = results.face_landmarks.landmark
        for i, face_idx in enumerate(FACE_SUBSET_INDICES):
            if face_idx < len(face_landmarks):
                landmark = face_landmarks[face_idx]
                keypoints[57 + i] = [landmark.x, landmark.y]

    return keypoints


def process_video(video_path, holistic):
    """
    Process a single video and extract keypoints from all frames.

    Returns:
        keypoints_sequence: numpy array of shape (num_frames, 63, 2)
    """
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = holistic.process(image_rgb)

        keypoints = extract_keypoints_from_results(results)
        keypoints_sequence.append(keypoints)

    cap.release()
    return np.array(keypoints_sequence, dtype=np.float32)


def save_keypoints(keypoints, output_path, gesture_class, user_id, metadata=None):
    """
    Save keypoints to pickle file with metadata.
    """
    data = {
        'keypoints': keypoints,
        'class': gesture_class,
        'user': user_id,
        'num_keypoints': NUM_KEYPOINTS,
        'keypoint_structure': {
            'pose_subset': '0-14 (15 pose keypoints, selected indices) ',
            'left_hand': '15-35 (21 keypoints)',
            'right_hand': '36-56 (21 keypoints)',
            'face_subset': '57-62 (6 keypoints)',
        },
        'pose_indices': POSE_SUBSET_INDICES,
        'face_indices': FACE_SUBSET_INDICES,
        'pose_note': 'Pose subset excludes ears and pose finger tips',
        'face_note': 'Face subset is 6-point FaceMesh (forehead + side-face)'
    }

    if metadata:
        data.update(metadata)

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def write_label_maps(output_root, gesture_classes):
    """
    Write label2id.json and id2label.json to output root.
    """
    labels = sorted(gesture_classes)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {str(idx): label for label, idx in label2id.items()}

    with open(output_root / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)
    with open(output_root / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)


def process_dataset(input_dir, output_dir, min_confidence=0.5):
    """
    Process all videos in the input directory.
    """
    input_path = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "all"
    output_path.mkdir(parents=True, exist_ok=True)

    user_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])

    if len(user_dirs) == 0:
        print(f"Error: No user directories found in {input_dir}")
        return

    total_videos = 0
    gesture_classes = set()
    for user_dir in user_dirs:
        for gesture_dir in user_dir.iterdir():
            if gesture_dir.is_dir() and gesture_dir.name.startswith('G'):
                total_videos += len(list(gesture_dir.glob('*.mp4'))) + len(list(gesture_dir.glob('*.avi')))
                gesture_classes.add(gesture_dir.name)

    print(f"Found {len(user_dirs)} users, {total_videos} videos total")

    with mp_holistic.Holistic(
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
        model_complexity=1
    ) as holistic:

        with tqdm(total=total_videos, desc="Extracting 63 keypoints") as pbar:
            for user_dir in user_dirs:
                user_id = user_dir.name

                gesture_dirs = sorted([d for d in user_dir.iterdir()
                                     if d.is_dir() and d.name.startswith('G')])

                for gesture_dir in gesture_dirs:
                    gesture_class = gesture_dir.name
                    gesture_output = output_path / gesture_class
                    gesture_output.mkdir(parents=True, exist_ok=True)

                    video_files = list(gesture_dir.glob('*.mp4')) + list(gesture_dir.glob('*.avi'))

                    for video_file in video_files:
                        keypoints = process_video(str(video_file), holistic)

                        output_filename = f"{user_id}_{gesture_class}_{video_file.stem}.pkl"
                        output_file = gesture_output / output_filename

                        metadata = {
                            'source_video': str(video_file),
                            'total_frames': keypoints.shape[0]
                        }

                        save_keypoints(keypoints, output_file, gesture_class, user_id, metadata)
                        pbar.update(1)

    write_label_maps(output_root, gesture_classes)
    print(f"\n✓ Extraction complete! Data saved to {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract 63 keypoints (v2.1) from videos")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with raw videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for keypoints")
    parser.add_argument("--min_confidence", type=float, default=0.5, help="Min detection/tracking confidence")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Extracting 63 keypoints (v2.1)")
    print("Pose subset (15) + Left hand (21) + Right hand (21) + Face subset (6)")
    print("=" * 80)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Min confidence:   {args.min_confidence}")
    print("=" * 80)

    process_dataset(args.input_dir, args.output_dir, args.min_confidence)


if __name__ == "__main__":
    main()
