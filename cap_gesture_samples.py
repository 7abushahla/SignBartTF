#!/usr/bin/env python3
"""
Cap gesture samples to maximum 50 per gesture per user.

This script:
1. Scans all user/gesture folders in video directory
2. For folders with >50 videos, randomly selects 50 to keep (seed=379)
3. Moves excess videos to extras folder
4. Deletes corresponding pkl files

Usage:
    python cap_gesture_samples.py \
        --video_dir /Volumes/LenovoPS8/MLR511/KArSL-502 \
        --pkl_dir /Volumes/LenovoPS8/MLR511/data/karsl502-63kpts \
        --max_samples 50 \
        --seed 379
"""

import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm


def cap_gesture_samples(video_dir, pkl_dir, extras_dir, max_samples=50, seed=379, dry_run=False):
    """
    Cap samples per gesture to max_samples by randomly selecting which to keep.
    
    Args:
        video_dir: Directory with videos (e.g., KArSL-502)
        pkl_dir: Directory with pkl files (e.g., data/karsl502-63kpts)
        extras_dir: Directory to move excess videos
        max_samples: Maximum samples per gesture (default: 50)
        seed: Random seed for reproducibility (default: 379)
        dry_run: If True, only preview without moving/deleting
    """
    video_path = Path(video_dir)
    pkl_path = Path(pkl_dir)
    extras_path = Path(extras_dir)
    
    # Set random seed
    random.seed(seed)
    
    # Get all user directories
    user_dirs = sorted([d for d in video_path.iterdir() 
                        if d.is_dir() and d.name.startswith('user')])
    
    print(f"Found {len(user_dirs)} user directories")
    print(f"Max samples per gesture: {max_samples}")
    print(f"Random seed: {seed}")
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be moved/deleted\n")
    
    stats = {
        'gestures_capped': 0,
        'videos_moved': 0,
        'pkls_deleted': 0,
        'gestures_checked': 0
    }
    
    # Scan all gesture folders
    for user_dir in tqdm(user_dirs, desc="Processing users"):
        user_name = user_dir.name
        
        # Get all gesture directories
        gesture_dirs = sorted([d for d in user_dir.iterdir() 
                              if d.is_dir() and d.name.startswith('G')])
        
        for gesture_dir in gesture_dirs:
            gesture_name = gesture_dir.name
            stats['gestures_checked'] += 1
            
            # Get all video files
            video_files = sorted(list(gesture_dir.glob('*.mp4')) + list(gesture_dir.glob('*.avi')))
            num_videos = len(video_files)
            
            if num_videos <= max_samples:
                # No need to cap
                continue
            
            # Need to cap - randomly select which to keep
            num_to_remove = num_videos - max_samples
            
            print(f"\n{user_name}/{gesture_name}: {num_videos} videos → capping to {max_samples}")
            
            # Shuffle and select
            shuffled_videos = video_files.copy()
            random.shuffle(shuffled_videos)
            
            videos_to_keep = set(shuffled_videos[:max_samples])
            videos_to_remove = [v for v in video_files if v not in videos_to_keep]
            
            stats['gestures_capped'] += 1
            
            # Process each video to remove
            for video_file in videos_to_remove:
                video_name = video_file.name
                
                # Construct pkl path
                # video: user01/G0005/01_01_0005_(...).mp4
                # pkl:   data/karsl502-63kpts/all/G0005/user01_G0005_01_01_0005_(...).pkl
                pkl_filename = f"{user_name}_{gesture_name}_{video_file.stem}.pkl"
                pkl_file = pkl_path / "all" / gesture_name / pkl_filename
                
                if not dry_run:
                    # Move video to extras
                    # Create extras structure: extras/session/gesture_num/
                    # Extract session from filename (e.g., 01_01_... → session=01)
                    parts = video_name.split('_')
                    if len(parts) >= 2:
                        session = parts[1]  # e.g., "01", "02", "03"
                        gesture_num = gesture_name[1:]  # G0005 → 0005
                        
                        extras_dest_dir = extras_path / session / gesture_num
                        extras_dest_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Move video
                        shutil.move(str(video_file), str(extras_dest_dir / video_name))
                        stats['videos_moved'] += 1
                    
                    # Delete pkl if it exists
                    if pkl_file.exists():
                        pkl_file.unlink()
                        stats['pkls_deleted'] += 1
                else:
                    # Dry run - just count
                    stats['videos_moved'] += 1
                    if pkl_file.exists():
                        stats['pkls_deleted'] += 1
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"Gestures checked:  {stats['gestures_checked']}")
    print(f"Gestures capped:   {stats['gestures_capped']}")
    print(f"Videos moved:      {stats['videos_moved']}")
    print(f"PKL files deleted: {stats['pkls_deleted']}")
    
    if dry_run:
        print("\n⚠️  This was a DRY RUN. Run without --dry_run to apply changes.")
    else:
        print(f"\n✓ Capping complete!")
        print(f"  Excess videos moved to: {extras_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Cap gesture samples to maximum per user/gesture"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/Volumes/LenovoPS8/MLR511/KArSL-502",
        help="Video directory (default: /Volumes/LenovoPS8/MLR511/KArSL-502)"
    )
    parser.add_argument(
        "--pkl_dir",
        type=str,
        default="/Volumes/LenovoPS8/MLR511/data/karsl502-63kpts",
        help="PKL directory (default: /Volumes/LenovoPS8/MLR511/data/karsl502-63kpts)"
    )
    parser.add_argument(
        "--extras_dir",
        type=str,
        default="/Volumes/LenovoPS8/MLR511/KArSL-502/extras",
        help="Directory to move excess videos (default: KArSL-502/extras)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Maximum samples per gesture (default: 50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=379,
        help="Random seed for reproducibility (default: 379)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview changes without applying them"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Cap Gesture Samples")
    print("=" * 80)
    print(f"Video directory: {args.video_dir}")
    print(f"PKL directory:   {args.pkl_dir}")
    print(f"Extras directory: {args.extras_dir}")
    print(f"Max samples:     {args.max_samples}")
    print(f"Random seed:     {args.seed}")
    if args.dry_run:
        print("Mode:            DRY RUN")
    print("=" * 80)
    print()
    
    if not args.dry_run:
        response = input("This will move videos and delete pkl files. Continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled.")
            return
        print()
    
    cap_gesture_samples(
        video_dir=args.video_dir,
        pkl_dir=args.pkl_dir,
        extras_dir=args.extras_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
