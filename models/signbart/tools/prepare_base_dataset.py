#!/usr/bin/env python
"""
prepare_base_dataset.py - Extract keypoints from Arabic ASL videos
Processes videos using MediaPipe Holistic and saves keypoints as .pkl files

Input structure:
  data/MLR511-ArabicSignLanguage-Dataset-MP4/
    user01/
      G01/
        R01.mp4
        R02.mp4
        ...
      G02/
        ...

Output structure:
  data/arabic-asl/
    all/
      G01/
        user01_G01_R01.pkl
        user01_G01_R02.pkl
        ...
      G02/
        ...
"""

import os
import cv2
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Try to import mediapipe
try:
    import mediapipe as mp
except ImportError:
    print("ERROR: MediaPipe not installed!")
    print("Please install it with: pip install mediapipe")
    exit(1)

def extract_keypoints_from_video(video_path, holistic):
    """
    Extract keypoints from a video using MediaPipe Holistic.
    
    Args:
        video_path: Path to video file
        holistic: MediaPipe Holistic object
        
    Returns:
        numpy array of shape (num_frames, 75, 3) containing keypoints
        or None if video cannot be processed
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return None
    
    keypoints_sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = holistic.process(frame_rgb)
        
        # Extract keypoints (75 total: 33 pose + 21 left hand + 21 right hand)
        keypoints = np.zeros((75, 3))  # x, y, visibility/confidence
        
        # Pose landmarks (0-32)
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints[idx] = [landmark.x, landmark.y, landmark.visibility]
        
        # Left hand landmarks (33-53)
        if results.left_hand_landmarks:
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                keypoints[33 + idx] = [landmark.x, landmark.y, landmark.z]  # z as confidence
        
        # Right hand landmarks (54-74)
        if results.right_hand_landmarks:
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                keypoints[54 + idx] = [landmark.x, landmark.y, landmark.z]  # z as confidence
        
        keypoints_sequence.append(keypoints)
    
    cap.release()
    
    if len(keypoints_sequence) == 0:
        return None
    
    return np.array(keypoints_sequence)

def process_dataset(input_dir, output_dir, max_videos=None):
    """
    Process all videos in the dataset.
    
    Args:
        input_dir: Root directory containing user folders
        output_dir: Output directory for processed data
        max_videos: Maximum number of videos to process (for testing)
    """
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Create output directory structure
    all_dir = os.path.join(output_dir, "all")
    os.makedirs(all_dir, exist_ok=True)
    
    # Find all video files
    video_files = []
    for user_dir in sorted(os.listdir(input_dir)):
        user_path = os.path.join(input_dir, user_dir)
        if not os.path.isdir(user_path) or not user_dir.startswith("user"):
            continue
        
        for gesture_dir in sorted(os.listdir(user_path)):
            gesture_path = os.path.join(user_path, gesture_dir)
            if not os.path.isdir(gesture_path) or not gesture_dir.startswith("G"):
                continue
            
            for video_file in sorted(os.listdir(gesture_path)):
                if not video_file.endswith(('.mp4', '.avi', '.mov')):
                    continue
                
                video_path = os.path.join(gesture_path, video_file)
                video_files.append({
                    'path': video_path,
                    'user': user_dir,
                    'gesture': gesture_dir,
                    'repetition': os.path.splitext(video_file)[0]
                })
    
    if max_videos:
        video_files = video_files[:max_videos]
    
    print(f"Found {len(video_files)} videos to process")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process each video
    processed = 0
    failed = 0
    
    for video_info in tqdm(video_files, desc="Processing videos"):
        user = video_info['user']
        gesture = video_info['gesture']
        repetition = video_info['repetition']
        
        # Create output directory for this gesture
        gesture_output_dir = os.path.join(all_dir, gesture)
        os.makedirs(gesture_output_dir, exist_ok=True)
        
        # Output filename: user01_G01_R01.pkl
        output_filename = f"{user}_{gesture}_{repetition}.pkl"
        output_path = os.path.join(gesture_output_dir, output_filename)
        
        # Skip if already processed
        if os.path.exists(output_path):
            processed += 1
            continue
        
        # Extract keypoints
        keypoints = extract_keypoints_from_video(video_info['path'], holistic)
        
        if keypoints is None or len(keypoints) == 0:
            print(f"\nWARNING: Failed to process {video_info['path']}")
            failed += 1
            continue
        
        # Save as pickle file
        data = {
            'keypoints': keypoints,
            'class': gesture,
            'user': user,
            'repetition': repetition,
            'num_frames': len(keypoints)
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        processed += 1
    
    holistic.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Processing Complete!")
    print(f"{'='*80}")
    print(f"Successfully processed: {processed} videos")
    print(f"Failed: {failed} videos")
    print(f"Output directory: {output_dir}")
    
    # Count files per gesture
    print(f"\nFiles per gesture class:")
    for gesture_dir in sorted(os.listdir(all_dir)):
        gesture_path = os.path.join(all_dir, gesture_dir)
        if os.path.isdir(gesture_path):
            num_files = len([f for f in os.listdir(gesture_path) if f.endswith('.pkl')])
            print(f"  {gesture_dir}: {num_files} files")
    
    print(f"\n{'='*80}")
    print(f"Next steps:")
    print(f"  1. Verify the data looks correct")
    print(f"  2. Run: python MOREUSERSprepare_arabic_asl.py")
    print(f"  3. Then train: python MOREUSERStrain_loso.py --epochs 80 --lr 2e-4 --no_validation")
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Extract keypoints from Arabic ASL videos using MediaPipe Holistic"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/MLR511-ArabicSignLanguage-Dataset-MP4",
        help="Input directory containing user folders with videos"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/arabic-asl",
        help="Output directory for processed keypoint files"
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (for testing)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 10 videos"
    )
    
    args = parser.parse_args()
    
    if args.test:
        args.max_videos = 10
        print("TEST MODE: Processing only 10 videos")
        print()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory not found: {args.input_dir}")
        print(f"Please ensure the video dataset is available.")
        return
    
    # Check for at least one user directory
    user_dirs = [d for d in os.listdir(args.input_dir) 
                 if os.path.isdir(os.path.join(args.input_dir, d)) and d.startswith("user")]
    
    if not user_dirs:
        print(f"ERROR: No user directories found in {args.input_dir}")
        print(f"Expected directory structure:")
        print(f"  {args.input_dir}/")
        print(f"    user01/")
        print(f"      G01/")
        print(f"        R01.mp4")
        print(f"        R02.mp4")
        return
    
    print(f"{'='*80}")
    print(f"Arabic ASL Dataset Preparation")
    print(f"{'='*80}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Found {len(user_dirs)} users: {', '.join(sorted(user_dirs))}")
    print(f"{'='*80}\n")
    
    # Process the dataset
    process_dataset(args.input_dir, args.output_dir, args.max_videos)

if __name__ == "__main__":
    main()