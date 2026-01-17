#!/usr/bin/env python
"""
Convert a pickle file to JSON for Flutter testing.
This allows us to load the exact same keypoints in the Flutter app.

Usage:
    python convert_pkl_to_json.py \
        --pickle_file data/arabic-asl-90kpts_LOSO_user01/test/G10/user01_G10_R10.pkl \
        --output test_keypoints.json \
        --subsample_to 21
"""

import argparse
import json
import pickle
import numpy as np


def load_pickle_file(pickle_path):
    """Load keypoints from pickle file."""
    print(f"Loading pickle file: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract keypoints (should be [T, 90, 2] or [T, 90, 3])
    if isinstance(data, dict):
        if 'keypoints' in data:
            keypoints = data['keypoints']
        elif 'data' in data:
            keypoints = data['data']
        else:
            # Try to find the keypoints array
            for key, value in data.items():
                if isinstance(value, np.ndarray) and value.ndim == 3:
                    keypoints = value
                    print(f"  Found keypoints in key: '{key}'")
                    break
            else:
                raise ValueError(f"Could not find keypoints in pickle file. Keys: {list(data.keys())}")
    elif isinstance(data, np.ndarray):
        keypoints = data
    else:
        raise ValueError(f"Unexpected pickle format: {type(data)}")
    
    print(f"  Keypoints shape: {keypoints.shape}")
    print(f"  Keypoints dtype: {keypoints.dtype}")
    print(f"  Keypoints range: [{keypoints.min():.3f}, {keypoints.max():.3f}]")
    
    # Handle 3D coordinates (x, y, z/visibility) - extract only x, y
    if keypoints.shape[-1] == 3:
        print(f"  Extracting only x, y coordinates (dropping 3rd dimension)")
        keypoints = keypoints[:, :, :2]
        print(f"  New shape: {keypoints.shape}")
    
    return keypoints


def subsample_keypoints(keypoints, target_frames, fps=15, interval_ms=150):
    """Subsample keypoints to match phone's sampling rate."""
    total_frames = keypoints.shape[0]
    
    # Calculate frame indices at specified intervals
    frames_per_sample = int(round((interval_ms / 1000.0) * fps))
    frames_per_sample = max(1, frames_per_sample)
    
    sampled_indices = list(range(0, total_frames, frames_per_sample))
    sampled_indices = sampled_indices[:target_frames]  # Limit to target
    
    print(f"  Subsampling from {total_frames} frames to {len(sampled_indices)} frames")
    print(f"  Sampling every {frames_per_sample} frame(s) at {interval_ms}ms intervals")
    print(f"  Frame indices: {sampled_indices}")
    
    return keypoints[sampled_indices]


def convert_to_json(keypoints, output_path):
    """Convert numpy keypoints to JSON format."""
    # Convert to list format
    keypoints_list = keypoints.tolist()
    
    # Create JSON structure
    data = {
        "metadata": {
            "num_frames": len(keypoints_list),
            "num_keypoints": len(keypoints_list[0]),
            "num_coords": 2,
            "description": "Keypoints from pickle file for testing"
        },
        "keypoints": keypoints_list
    }
    
    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Converted to JSON: {output_path}")
    print(f"   Frames: {len(keypoints_list)}")
    print(f"   Keypoints per frame: {len(keypoints_list[0])}")
    print(f"   File size: {len(json.dumps(data)) / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description='Convert pickle to JSON for Flutter testing')
    parser.add_argument('--pickle_file', type=str, required=True,
                        help='Path to pickle file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--subsample_to', type=int, default=None,
                        help='Subsample to N frames (default: no subsampling)')
    parser.add_argument('--fps', type=int, default=15,
                        help='Video FPS for subsampling (default: 15)')
    parser.add_argument('--interval_ms', type=int, default=150,
                        help='Sampling interval in ms (default: 150)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PICKLE TO JSON CONVERTER")
    print("="*80)
    print(f"Input: {args.pickle_file}")
    print(f"Output: {args.output}")
    if args.subsample_to:
        print(f"Subsample to: {args.subsample_to} frames @ {args.interval_ms}ms intervals")
    print("="*80)
    
    # Load pickle
    keypoints = load_pickle_file(args.pickle_file)
    
    # Subsample if requested
    if args.subsample_to:
        keypoints = subsample_keypoints(keypoints, args.subsample_to, args.fps, args.interval_ms)
    
    # Convert to JSON
    convert_to_json(keypoints, args.output)
    
    # Show first frame sample
    print("\n📊 First frame sample (first 5 keypoints):")
    for i in range(min(5, keypoints.shape[1])):
        print(f"   Keypoint {i}: {keypoints[0, i].tolist()}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

