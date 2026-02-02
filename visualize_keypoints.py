#!/usr/bin/env python3
"""
Visualize extracted keypoints overlaid on the original video.

This script takes:
1. A video file (e.g., .mp4)
2. A corresponding .pkl file with extracted keypoints

And produces a new video with keypoints drawn on each frame.

Usage:
    python visualize_keypoints.py \\
        --video_path path/to/video.mp4 \\
        --pkl_path path/to/keypoints.pkl \\
        --output_path visualization.mp4
"""

import argparse
import pickle
import cv2
import numpy as np
from pathlib import Path


# Color scheme for different body parts (BGR format for OpenCV)
COLORS = {
    'pose': (0, 255, 0),        # Green
    'left_hand': (255, 0, 0),   # Blue
    'right_hand': (0, 0, 255),  # Red
    'face': (255, 255, 0),      # Cyan
}


def load_keypoints(pkl_path):
    """
    Load keypoints from pickle file.
    
    Returns:
        keypoints: numpy array [num_frames, num_keypoints, 2]
        metadata: dict with additional info
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        keypoints = data.get('keypoints', data.get('data', None))
        metadata = {k: v for k, v in data.items() if k not in ['keypoints', 'data']}
    else:
        keypoints = data
        metadata = {}
    
    print(f"Loaded keypoints: {keypoints.shape}")
    print(f"Metadata keys: {list(metadata.keys())}")
    
    return keypoints, metadata


def get_keypoint_groups(num_keypoints):
    """
    Determine keypoint groups based on total count.
    
    Returns:
        dict mapping group name to (start_idx, end_idx, color)
    """
    if num_keypoints == 90:
        # 90 keypoints: 23 pose + 21 left hand + 21 right hand + 25 face
        return {
            'pose': (0, 23, COLORS['pose']),
            'left_hand': (23, 44, COLORS['left_hand']),
            'right_hand': (44, 65, COLORS['right_hand']),
            'face': (65, 90, COLORS['face']),
        }
    elif num_keypoints == 63:
        # 63 keypoints: 15 pose + 21 left hand + 21 right hand + 6 face
        return {
            'pose': (0, 15, COLORS['pose']),
            'left_hand': (15, 36, COLORS['left_hand']),
            'right_hand': (36, 57, COLORS['right_hand']),
            'face': (57, 63, COLORS['face']),
        }
    else:
        # Unknown format - treat all as one group
        return {
            'all': (0, num_keypoints, COLORS['pose'])
        }


def draw_keypoints_on_frame(frame, keypoints_2d, groups, radius=3, thickness=-1):
    """
    Draw keypoints on a frame.
    
    Args:
        frame: OpenCV image (H, W, 3)
        keypoints_2d: (num_keypoints, 2) - normalized coordinates [0, 1]
        groups: dict of group definitions
        radius: circle radius
        thickness: -1 for filled, positive for outline
    
    Returns:
        frame with keypoints drawn
    """
    h, w = frame.shape[:2]
    
    for group_name, (start_idx, end_idx, color) in groups.items():
        for i in range(start_idx, end_idx):
            if i < len(keypoints_2d):
                x_norm, y_norm = keypoints_2d[i]
                
                # Skip invalid keypoints (zeros or out of bounds)
                if x_norm == 0 and y_norm == 0:
                    continue
                if not (0 <= x_norm <= 1 and 0 <= y_norm <= 1):
                    continue
                
                # Convert normalized to pixel coordinates
                x = int(x_norm * w)
                y = int(y_norm * h)
                
                # Draw circle
                cv2.circle(frame, (x, y), radius, color, thickness)
    
    return frame


def visualize_video(video_path, pkl_path, output_path, show_preview=False):
    """
    Create visualization video with keypoints overlaid.
    
    Args:
        video_path: path to original video
        pkl_path: path to keypoints pkl file
        output_path: path to save output video
        show_preview: if True, show frames during processing (slower)
    """
    # Load keypoints
    keypoints, metadata = load_keypoints(pkl_path)
    
    # Handle 3D keypoints (extract only x, y)
    if keypoints.shape[-1] == 3:
        print(f"Detected 3D keypoints, extracting x,y only")
        keypoints = keypoints[:, :, :2]
    
    num_frames, num_keypoints, num_coords = keypoints.shape
    assert num_coords == 2, f"Expected 2D keypoints, got shape {keypoints.shape}"
    
    # Get keypoint groups
    groups = get_keypoint_groups(num_keypoints)
    print(f"Keypoint groups: {list(groups.keys())}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_video_frames}")
    print(f"  Keypoint frames: {num_frames}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_idx = 0
    frames_written = 0
    
    print(f"\nProcessing frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw keypoints if available for this frame
        if frame_idx < num_frames:
            frame = draw_keypoints_on_frame(frame, keypoints[frame_idx], groups)
            
            # Add frame number and info text
            cv2.putText(
                frame, 
                f"Frame {frame_idx + 1}/{num_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        else:
            # No keypoints for this frame (shouldn't happen normally)
            cv2.putText(
                frame,
                f"Frame {frame_idx + 1} (no keypoints)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        # Write frame
        out.write(frame)
        frames_written += 1
        
        # Show preview if requested
        if show_preview:
            cv2.imshow('Keypoints Visualization', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break
        
        # Progress update
        if frame_idx % 10 == 0 or frame_idx == num_frames - 1:
            print(f"  Processed {frame_idx + 1}/{num_frames} frames", end='\r')
        
        frame_idx += 1
    
    # Cleanup
    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()
    
    print(f"\n\n✓ Visualization complete!")
    print(f"  Frames written: {frames_written}")
    print(f"  Output saved to: {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize keypoints overlaid on video"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to original video file"
    )
    parser.add_argument(
        "--pkl_path",
        type=str,
        required=True,
        help="Path to keypoints pickle file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="keypoints_visualization.mp4",
        help="Path to save output video (default: keypoints_visualization.mp4)"
    )
    parser.add_argument(
        "--show_preview",
        action="store_true",
        help="Show frames during processing (press 'q' to stop)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Keypoints Visualization")
    print("=" * 80)
    print(f"Video:  {args.video_path}")
    print(f"PKL:    {args.pkl_path}")
    print(f"Output: {args.output_path}")
    print("=" * 80)
    print()
    
    # Check files exist
    if not Path(args.video_path).exists():
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    if not Path(args.pkl_path).exists():
        print(f"Error: PKL file not found: {args.pkl_path}")
        return
    
    # Run visualization
    visualize_video(args.video_path, args.pkl_path, args.output_path, args.show_preview)
    
    print("\n" + "=" * 80)
    print("Color coding:")
    print("  Green:  Pose keypoints")
    print("  Blue:   Left hand keypoints")
    print("  Red:    Right hand keypoints")
    print("  Cyan:   Face keypoints")
    print("=" * 80)


if __name__ == "__main__":
    main()
