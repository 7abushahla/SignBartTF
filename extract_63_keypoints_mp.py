#!/usr/bin/env python
"""
extract_63_keypoints_mp.py - Multiprocess version of 63-keypoint extraction.

This is a drop-in variant of extract_63_keypoints.py that can process
multiple clips in parallel using Python multiprocessing (one process
per clip), so you can better utilize multi-core CPUs.

Keypoint Structure (per frame, ordered):
    Indices 0-14:  Pose subset (15 keypoints)
    Indices 15-35: Left hand (21 keypoints)
    Indices 36-56: Right hand (21 keypoints)
    Indices 57-62: Face subset (6 keypoints)

Usage (example):
    python extract_63_keypoints_mp.py \\
        --input_dir videos/ \\
        --output_dir data/arabic-asl-63kpts/ \\
        --num_workers 4
"""

import argparse
import json
import os
from pathlib import Path
import pickle
from typing import Iterable, List, Tuple
import warnings
import contextlib
import sys
import os as _os

import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp_ctx

# -----------------------------------------------------------------------------
# Logging / warning suppression (for cleaner multiprocessing output)
# -----------------------------------------------------------------------------

# Reduce TensorFlow / Mediapipe C++ log spam
os.environ.setdefault("GLOG_minloglevel", "2")      # 0=INFO,1=WARNING,2=ERROR,3=FATAL
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

# Suppress known protobuf deprecation warnings from Mediapipe
warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype\\(\\) is deprecated",
    category=UserWarning,
)

# NOTE: We intentionally do NOT import mediapipe at module import time.
# In multiprocessing ("spawn"), each worker imports this module; MediaPipe/TFLite may emit
# noisy C++ logs during import/initialization. To reliably silence those logs, we need
# to redirect the OS-level stderr FD *before* importing/initializing MediaPipe inside
# the worker (see `_redirect_stderr_fd` + `_get_mediapipe_modules`).

# Pose subset indices (15 total) - in this exact order
POSE_SUBSET_INDICES = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16]

# Face subset indices (6 total) - in this exact order
FACE_SUBSET_INDICES = [10, 338, 297, 67, 234, 454]

# Total keypoints: 63
NUM_KEYPOINTS = 63


@contextlib.contextmanager
def _suppress_stderr(enabled: bool):
    """
    Best-effort suppression of noisy C++ logs that bypass Python logging.

    WARNING: When enabled, this will also hide genuine errors printed to stderr
    during the suppressed region. Keep it OFF unless you're sure things are stable.
    """
    if not enabled:
        yield
        return

    try:
        with open(os.devnull, "w") as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            yield
    finally:
        try:
            sys.stderr = old_stderr  # type: ignore[name-defined]
        except Exception:
            pass


@contextlib.contextmanager
def _redirect_stderr_fd(enabled: bool):
    """
    Redirect OS-level stderr (fd=2) to /dev/null for native/C++ logs.

    This is the *only* reliable way to silence some MediaPipe/TFLite logs.
    WARNING: This will also hide genuine native errors printed to stderr during
    the suppressed region.
    """
    if not enabled:
        yield
        return

    saved_fd = None
    devnull_fd = None
    try:
        saved_fd = _os.dup(2)
        devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
        _os.dup2(devnull_fd, 2)
        yield
    finally:
        try:
            if saved_fd is not None:
                _os.dup2(saved_fd, 2)
        finally:
            if devnull_fd is not None:
                _os.close(devnull_fd)
            if saved_fd is not None:
                _os.close(saved_fd)


def _get_mediapipe_modules(suppress_stderr: bool):
    """
    Import mediapipe and configure absl verbosity, optionally with stderr suppression.

    Returns:
        mp_holistic: mp.solutions.holistic
    """
    with _redirect_stderr_fd(suppress_stderr), _suppress_stderr(suppress_stderr):
        try:
            from absl import logging as absl_logging

            absl_logging.set_verbosity(absl_logging.ERROR)
        except Exception:
            pass

        import mediapipe as mp  # local import on purpose

        return mp.solutions.holistic


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


def process_video(video_path: str, min_confidence: float, suppress_stderr: bool) -> np.ndarray:
    """
    Process a single video and extract keypoints from all frames.

    Returns:
        keypoints_sequence: numpy array of shape (num_frames, 63, 2)
    """
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence: List[np.ndarray] = []

    # Import/init MediaPipe Holistic inside the worker, optionally suppressing native stderr
    # so we catch the noisy C++ logs that happen during initialization.
    mp_holistic = _get_mediapipe_modules(suppress_stderr=suppress_stderr)

    # Some MediaPipe/TFLite warnings are emitted to stderr from C++ during runtime too;
    # wrap the whole processing section.
    with _redirect_stderr_fd(suppress_stderr), _suppress_stderr(suppress_stderr):
        with mp_holistic.Holistic(
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            model_complexity=1,
        ) as holistic:
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
        "keypoints": keypoints,
        "class": gesture_class,
        "user": user_id,
        "num_keypoints": NUM_KEYPOINTS,
        "keypoint_structure": {
            "pose_subset": "0-14 (15 pose keypoints, selected indices) ",
            "left_hand": "15-35 (21 keypoints)",
            "right_hand": "36-56 (21 keypoints)",
            "face_subset": "57-62 (6 keypoints)",
        },
        "pose_indices": POSE_SUBSET_INDICES,
        "face_indices": FACE_SUBSET_INDICES,
        "pose_note": "Pose subset excludes ears and pose finger tips",
        "face_note": "Face subset is 6-point FaceMesh (forehead + side-face)",
    }

    if metadata:
        data.update(metadata)

    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def write_label_maps(output_root: Path, gesture_classes: Iterable[str]):
    """
    Write label2id.json and id2label.json to output root.
    """
    labels = sorted(set(gesture_classes))
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {str(idx): label for label, idx in label2id.items()}

    with open(output_root / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)
    with open(output_root / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)


# --------------------------- Multiprocessing --------------------------- #

def _worker_process_video(args: Tuple[str, str, str, str, float]) -> Tuple[str, str, str, int]:
    """
    Worker function: process a single video and save its keypoints.

    Args tuple:
        video_path, user_id, gesture_class, output_root_str, min_confidence

    Returns:
        (user_id, gesture_class, video_filename, num_frames)
    """
    video_path, user_id, gesture_class, output_root_str, min_confidence, suppress_stderr = args
    output_root = Path(output_root_str)
    output_path = output_root / "all" / gesture_class
    output_path.mkdir(parents=True, exist_ok=True)

    # If requested, redirect stderr FD for the entire worker task so we also suppress
    # logs emitted during MediaPipe import/initialization in this worker.
    with _redirect_stderr_fd(bool(suppress_stderr)), _suppress_stderr(bool(suppress_stderr)):
        keypoints = process_video(video_path, min_confidence, suppress_stderr=bool(suppress_stderr))

    video_file = Path(video_path)
    output_filename = f"{user_id}_{gesture_class}_{video_file.stem}.pkl"
    output_file = output_path / output_filename

    metadata = {
        "source_video": str(video_file),
        "total_frames": int(keypoints.shape[0]),
    }

    save_keypoints(keypoints, output_file, gesture_class, user_id, metadata)

    return user_id, gesture_class, video_file.name, int(keypoints.shape[0])


def process_dataset_mp(
    input_dir: str,
    output_dir: str,
    min_confidence: float,
    num_workers: int,
    suppress_stderr: bool,
):
    """
    Process all videos in the input directory using multiprocessing.
    """
    input_path = Path(input_dir)
    output_root = Path(output_dir)
    (output_root / "all").mkdir(parents=True, exist_ok=True)

    user_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])

    if len(user_dirs) == 0:
        print(f"Error: No user directories found in {input_dir}")
        return

    jobs: List[Tuple[str, str, str, str, float, bool]] = []
    gesture_classes = set()

    for user_dir in user_dirs:
        user_id = user_dir.name
        gesture_dirs = sorted(
            [d for d in user_dir.iterdir() if d.is_dir() and d.name.startswith("G")]
        )
        for gesture_dir in gesture_dirs:
            gesture_class = gesture_dir.name
            gesture_classes.add(gesture_class)
            video_files = list(gesture_dir.glob("*.mp4")) + list(gesture_dir.glob("*.avi"))
            for video_file in video_files:
                jobs.append(
                    (
                        str(video_file),
                        user_id,
                        gesture_class,
                        str(output_root),
                        float(min_confidence),
                        bool(suppress_stderr),
                    )
                )

    total_videos = len(jobs)
    print(f"Found {len(user_dirs)} users, {total_videos} videos total")
    if total_videos == 0:
        print("No video files found.")
        return

    # Use a process pool to handle videos in parallel
    num_workers = max(1, int(num_workers))
    print(f"Using {num_workers} worker processes")

    with mp_ctx.Pool(processes=num_workers) as pool:
        with tqdm(total=total_videos, desc="Extracting 63 keypoints (mp)") as pbar:
            for user_id, gesture_class, video_name, n_frames in pool.imap_unordered(
                _worker_process_video, jobs
            ):
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "user": user_id,
                        "gesture": gesture_class,
                        "file": video_name,
                        "frames": n_frames,
                    }
                )

    write_label_maps(output_root, gesture_classes)
    print(f"\n✓ Multiprocess extraction complete! Data saved to {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract 63 keypoints (v2.1) with multiprocessing")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory with raw videos",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for keypoints",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="Min detection/tracking confidence",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4)",
    )
    parser.add_argument(
        "--suppress_stderr",
        action="store_true",
        help="HARD-quiet mode: redirect stderr in workers to silence MediaPipe/TFLite C++ warnings (may hide real errors).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Extracting 63 keypoints (v2.1) - MULTIPROCESS VERSION")
    print("Pose subset (15) + Left hand (21) + Right hand (21) + Face subset (6)")
    print("=" * 80)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Min confidence:   {args.min_confidence}")
    print(f"Num workers:      {args.num_workers}")
    print(f"Suppress stderr:  {bool(args.suppress_stderr)}")
    print("=" * 80)

    process_dataset_mp(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_confidence=args.min_confidence,
        num_workers=args.num_workers,
        suppress_stderr=bool(args.suppress_stderr),
    )


if __name__ == "__main__":
    # On macOS, 'spawn' is the default; being explicit can avoid surprises.
    try:
        mp_ctx.set_start_method("spawn")
    except RuntimeError:
        # Start method already set; ignore.
        pass
    main()

