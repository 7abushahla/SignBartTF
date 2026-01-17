#!/usr/bin/env python
"""
extract_90_keypoints.py - Extract 90 keypoints from MediaPipe Holistic

This script processes videos to extract 90 keypoints (upper body only):
- 23 pose landmarks (UPPER BODY ONLY - no hips, knees, ankles, feet)
- 21 left hand landmarks
- 21 right hand landmarks
- 25 face landmarks (selected from 468 available)

The face landmarks are strategically selected for sign language:
- 10 face contour points (head orientation) - SYMMETRIC
- 7 right eye points (gaze and expression)
- 7 left eye points (gaze and expression)
- 1 nose tip (reference point)

Usage:
    python extract_90_keypoints.py --input_dir videos/ --output_dir data/
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from pathlib import Path
import argparse
from tqdm import tqdm

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Face landmark indices to extract (25 total) - CORRECTED SYMMETRIC VERSION
# These are indices in MediaPipe Face Mesh (468 total landmarks)
FACE_LANDMARK_INDICES = [
    # Face contour (10 points) - SYMMETRIC for head orientation
    10,   # Forehead center (top)
    338,  # Right temple
    297,  # Right upper cheek
    332,  # Right lower cheek
    172,  # Right jaw chin
    152,  # Chin center (bottom)
    397,  # Left jaw chin (symmetric to 172)
    103,  # Left lower cheek (symmetric to 332)
    67,   # Left upper cheek (symmetric to 297)
    109,  # Left temple (symmetric to 338)
    # Right eye (7 points)
    33, 133, 160, 159, 158, 144, 145,
    # Left eye (7 points)
    362, 263, 387, 386, 385, 373, 374,
    # Nose (1 point)
    1
]

# Upper body pose landmark indices (23 total: 0-22)
# Excludes: hips (23-24), knees (25-26), ankles (27-28), heels (29-30), feet (31-32)
UPPER_BODY_POSE_INDICES = list(range(23))  # [0, 1, 2, ..., 22]

def extract_keypoints_from_results(results, image_width, image_height):
    """
    Extract 90 keypoints from MediaPipe Holistic results.
    
    Returns:
        keypoints: numpy array of shape (90, 3) containing [x, y, visibility/presence]
                   Indices 0-22:  Pose (UPPER BODY ONLY - 23 keypoints)
                   Indices 23-43: Left hand (21 keypoints)
                   Indices 44-64: Right hand (21 keypoints)
                   Indices 65-89: Face (25 selected symmetric keypoints)
    """
    keypoints = np.zeros((90, 3))
    
    # Extract UPPER BODY pose landmarks (23 keypoints: indices 0-22)
    # MediaPipe Pose indices 0-22 = nose, eyes, ears, mouth, shoulders, elbows, wrists, hands
    if results.pose_landmarks:
        for i in UPPER_BODY_POSE_INDICES:
            if i < len(results.pose_landmarks.landmark):
                landmark = results.pose_landmarks.landmark[i]
                keypoints[i] = [
                    landmark.x,
                    landmark.y,
                    landmark.visibility
                ]
    
    # Extract left hand landmarks (21 keypoints: indices 23-43)
    if results.left_hand_landmarks:
        for i, landmark in enumerate(results.left_hand_landmarks.landmark):
            keypoints[23 + i] = [
                landmark.x,
                landmark.y,
                1.0  # Hand landmarks don't have visibility, use 1.0
            ]
    
    # Extract right hand landmarks (21 keypoints: indices 44-64)
    if results.right_hand_landmarks:
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            keypoints[44 + i] = [
                landmark.x,
                landmark.y,
                1.0  # Hand landmarks don't have visibility, use 1.0
            ]
    
    # Extract selected face landmarks (25 keypoints: indices 65-89)
    if results.face_landmarks:
        face_landmarks = results.face_landmarks.landmark
        for i, face_idx in enumerate(FACE_LANDMARK_INDICES):
            if face_idx < len(face_landmarks):
                landmark = face_landmarks[face_idx]
                keypoints[65 + i] = [
                    landmark.x,
                    landmark.y,
                    1.0  # Face landmarks don't have visibility, use presence or 1.0
                ]
    
    return keypoints

def process_video(video_path, holistic):
    """
    Process a single video and extract keypoints from all frames.
    
    Returns:
        keypoints_sequence: numpy array of shape (num_frames, 90, 3)
    """
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process with MediaPipe
        results = holistic.process(image_rgb)
        
        # Extract keypoints
        keypoints = extract_keypoints_from_results(
            results, 
            image.shape[1], 
            image.shape[0]
        )
        keypoints_sequence.append(keypoints)
    
    cap.release()
    
    return np.array(keypoints_sequence)

def save_keypoints(keypoints, output_path, gesture_class, user_id, metadata=None):
    """
    Save keypoints to pickle file with metadata.
    
    Args:
        keypoints: numpy array of shape (num_frames, 90, 3)
        output_path: path to save pickle file
        gesture_class: gesture class label (e.g., 'G01')
        user_id: user identifier (e.g., 'user01')
        metadata: optional dictionary with additional metadata
    """
    data = {
        'keypoints': keypoints,
        'class': gesture_class,
        'user': user_id,
        'num_keypoints': 90,
        'keypoint_structure': {
            'pose': '0-22 (23 upper body keypoints)',
            'left_hand': '23-43 (21 keypoints)',
            'right_hand': '44-64 (21 keypoints)',
            'face': '65-89 (25 symmetric selected keypoints)'
        },
        'face_mesh_indices': FACE_LANDMARK_INDICES,
        'pose_note': 'Upper body only - excludes hips, knees, ankles, heels, feet'
    }
    
    if metadata:
        data.update(metadata)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

def process_dataset(input_dir, output_dir, min_confidence=0.5):
    """
    Process all videos in the input directory.
    
    Expected structure:
        input_dir/
            user01/
                G01/
                    R01.mp4
                    R02.mp4
                    ...
                G02/
                    R01.mp4
                    ...
            user02/
                G01/
                    ...
    
    Output structure:
        output_dir/
            all/
                G01/
                    user01_G01_R01.pkl
                    user01_G01_R02.pkl
                    user02_G01_R01.pkl
                    ...
                G02/
                    ...
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) / "all"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all user directories
    user_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    
    if len(user_dirs) == 0:
        print(f"Error: No user directories found in {input_dir}")
        return
    
    print(f"Found {len(user_dirs)} user directories")
    print(f"Output directory: {output_path}")
    
    # Count total videos
    total_videos = 0
    for user_dir in user_dirs:
        gesture_dirs = [d for d in user_dir.iterdir() if d.is_dir() and d.name.startswith('G')]
        for gesture_dir in gesture_dirs:
            total_videos += len(list(gesture_dir.glob('*.mp4'))) + len(list(gesture_dir.glob('*.avi')))
    
    print(f"Total videos to process: {total_videos}")
    
    # Initialize MediaPipe Holistic
    with mp_holistic.Holistic(
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
        model_complexity=1
    ) as holistic:
        
        with tqdm(total=total_videos, desc="Processing videos") as pbar:
            for user_dir in user_dirs:
                user_id = user_dir.name
                
                # Find all gesture directories for this user
                gesture_dirs = sorted([d for d in user_dir.iterdir() 
                                     if d.is_dir() and d.name.startswith('G')])
                
                for gesture_dir in gesture_dirs:
                    gesture_class = gesture_dir.name
                    
                    # Create output directory for this gesture (if not exists)
                    gesture_output = output_path / gesture_class
                    gesture_output.mkdir(parents=True, exist_ok=True)
                    
                    # Process all videos in this gesture directory
                    video_files = list(gesture_dir.glob('*.mp4')) + list(gesture_dir.glob('*.avi'))
                    
                    for video_file in video_files:
                        # Extract keypoints
                        keypoints = process_video(str(video_file), holistic)
                        
                        if len(keypoints) == 0:
                            print(f"Warning: No frames extracted from {video_file}")
                            pbar.update(1)
                            continue
                        
                        # Create output filename: user_id_gesture_class_recording.pkl
                        # e.g., user01_G01_R01.pkl
                        output_filename = f"{user_id}_{gesture_class}_{video_file.stem}.pkl"
                        output_file = gesture_output / output_filename
                        
                        # Save to pickle file
                        save_keypoints(
                            keypoints,
                            str(output_file),
                            gesture_class,
                            user_id,
                            metadata={
                                'video_file': video_file.name,
                                'num_frames': len(keypoints),
                                'original_path': str(video_file.relative_to(input_path))
                            }
                        )
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'user': user_id,
                            'gesture': gesture_class,
                            'file': video_file.name,
                            'frames': len(keypoints)
                        })
    
    print(f"\nProcessing complete!")
    print(f"Keypoints saved to: {output_path}")
    print(f"\nKeypoint structure (90 total - UPPER BODY ONLY):")
    print(f"  Indices 0-22:   Pose UPPER BODY (23 keypoints)")
    print(f"                  ✓ nose, eyes, ears, mouth (0-10)")
    print(f"                  ✓ shoulders, elbows, wrists (11-16)")
    print(f"                  ✓ hand orientation points (17-22)")
    print(f"                  ✗ NO hips, knees, ankles, heels, feet")
    print(f"  Indices 23-43:  Left Hand (21 keypoints)")
    print(f"  Indices 44-64:  Right Hand (21 keypoints)")
    print(f"  Indices 65-89:  Face (25 symmetric selected keypoints)")
    print(f"\nFace landmarks selected (SYMMETRIC):")
    print(f"  65-74:  Face contour (10 symmetric points)")
    print(f"          [10, 338, 297, 332, 172, 152, 397, 103, 67, 109]")
    print(f"  75-81:  Right eye (7 points)")
    print(f"  82-88:  Left eye (7 points)")
    print(f"  89:     Nose tip (1 point)")

def visualize_keypoints(video_path, output_video_path=None):
    """
    Visualize the extracted keypoints on video frames.
    Useful for debugging and verification.
    Only draws upper body pose landmarks.
    """
    cap = cv2.VideoCapture(video_path)
    
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # Process with MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            # Draw landmarks
            image.flags.writeable = True
            
            # Draw UPPER BODY pose only (indices 0-22)
            if results.pose_landmarks:
                # Draw only upper body landmarks
                for i in UPPER_BODY_POSE_INDICES:
                    if i < len(results.pose_landmarks.landmark):
                        landmark = results.pose_landmarks.landmark[i]
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            
            # Draw hands
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
                )
            
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
                )
            
            # Draw selected face landmarks (symmetric)
            if results.face_landmarks:
                for idx in FACE_LANDMARK_INDICES:
                    if idx < len(results.face_landmarks.landmark):
                        landmark = results.face_landmarks.landmark[idx]
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
            
            # Display
            cv2.imshow('MediaPipe Holistic (90 keypoints - Upper Body)', image)
            
            if output_video_path:
                out.write(image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    cap.release()
    if output_video_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description="Extract 90 keypoints (upper body only) from videos using MediaPipe Holistic"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing user subdirectories with gesture folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save extracted keypoints"
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="Minimum detection/tracking confidence (default: 0.5)"
    )
    parser.add_argument(
        "--visualize",
        type=str,
        default="",
        help="Path to a single video to visualize keypoints (debugging)"
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="",
        help="Path to save visualization video (requires --visualize)"
    )
    
    args = parser.parse_args()
    
    if args.visualize:
        print(f"Visualizing keypoints for: {args.visualize}")
        print(f"Press 'q' to quit")
        visualize_keypoints(args.visualize, args.output_video)
    else:
        print("="*80)
        print("MediaPipe Holistic - 90 Keypoint Extraction (Upper Body Only)")
        print("="*80)
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Min confidence: {args.min_confidence}")
        print("="*80)
        print()
        
        process_dataset(args.input_dir, args.output_dir, args.min_confidence)
        
        print("\n" + "="*80)
        print("Next steps:")
        print("="*80)
        print("1. Create LOSO splits:")
        print("   python MOREUSERSprepare_arabic_asl.py --base_data_path data/arabic-asl-90kpts")
        print()
        print("2. Train models:")
        print("   python MOREUSERStrain_loso.py \\")
        print("       --config_path configs/arabic-asl-new.yaml \\")
        print("       --epochs 80 --lr 2e-4 --no_validation")
        print()
        print("3. Collect results:")
        print("   python collect_results.py")
        print("="*80)

if __name__ == "__main__":
    main()

