#!/usr/bin/env python3
"""
Merge two pkl directories from split processing.

This script safely merges pkl files from two directories:
- Source directory (e.g., karsl502-63kpts-rest): new pkl files to merge
- Target directory (e.g., karsl502-63kpts): existing pkl files

It will:
1. Find all pkl files in source
2. Check for conflicts in target
3. Copy non-conflicting files
4. Report statistics and any issues

Usage:
    python merge_pkl_directories.py \
        --source data/karsl502-63kpts-rest \
        --target data/karsl502-63kpts \
        --dry_run  # Optional: preview without copying
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import pickle


def get_all_pkl_files(directory):
    """
    Recursively find all .pkl files in a directory.
    
    Returns:
        dict mapping relative path to absolute path
    """
    base_path = Path(directory)
    pkl_files = {}
    
    for pkl_file in base_path.rglob("*.pkl"):
        rel_path = pkl_file.relative_to(base_path)
        pkl_files[str(rel_path)] = pkl_file
    
    return pkl_files


def verify_pkl_file(pkl_path):
    """
    Verify a pkl file is valid and get its frame count.
    
    Returns:
        (is_valid, num_frames, error_msg)
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                keypoints = data.get('keypoints', data.get('data', None))
            else:
                keypoints = data
            
            if keypoints is None:
                return False, 0, "No keypoints found in pkl"
            
            num_frames = keypoints.shape[0]
            return True, num_frames, None
    except Exception as e:
        return False, 0, str(e)


def merge_pkl_directories(source_dir, target_dir, dry_run=False, verify=False):
    """
    Merge pkl files from source into target directory.
    
    Args:
        source_dir: Directory with new pkl files
        target_dir: Target directory to merge into
        dry_run: If True, only preview without copying
        verify: If True, verify pkl files are valid before copying
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return
    
    if not target_path.exists():
        print(f"Error: Target directory does not exist: {target_dir}")
        return
    
    print("Scanning directories...")
    source_files = get_all_pkl_files(source_dir)
    target_files = get_all_pkl_files(target_dir)
    
    print(f"\nSource directory: {source_dir}")
    print(f"  Total pkl files: {len(source_files)}")
    
    print(f"\nTarget directory: {target_dir}")
    print(f"  Total pkl files: {len(target_files)}")
    
    # Analyze files
    to_copy = []
    conflicts = []
    
    for rel_path, source_file in source_files.items():
        target_file = target_path / rel_path
        
        if target_file.exists():
            # Conflict: file exists in both
            conflicts.append((rel_path, source_file, target_file))
        else:
            # Safe to copy
            to_copy.append((rel_path, source_file, target_file))
    
    print(f"\n{'='*80}")
    print("Analysis:")
    print(f"{'='*80}")
    print(f"Files to copy:   {len(to_copy)}")
    print(f"Conflicts found: {len(conflicts)}")
    print(f"Total after merge: {len(target_files) + len(to_copy)}")
    
    # Report conflicts
    if conflicts:
        print(f"\n⚠️  Warning: {len(conflicts)} conflicts found!")
        print("These files exist in both directories:")
        for i, (rel_path, src, tgt) in enumerate(conflicts[:10], 1):
            print(f"  {i}. {rel_path}")
            print(f"     Source size: {src.stat().st_size:,} bytes")
            print(f"     Target size: {tgt.stat().st_size:,} bytes")
        
        if len(conflicts) > 10:
            print(f"  ... and {len(conflicts) - 10} more")
        
        print("\nConflicts will be SKIPPED (target files preserved).")
    
    if len(to_copy) == 0:
        print("\n✓ No files to copy. Directories already merged.")
        return
    
    # Verify files if requested
    if verify:
        print(f"\nVerifying {len(to_copy)} pkl files...")
        invalid_files = []
        
        for rel_path, source_file, target_file in tqdm(to_copy, desc="Verifying"):
            is_valid, num_frames, error = verify_pkl_file(source_file)
            if not is_valid:
                invalid_files.append((rel_path, error))
        
        if invalid_files:
            print(f"\n⚠️  Warning: {len(invalid_files)} invalid pkl files found!")
            for rel_path, error in invalid_files[:10]:
                print(f"  {rel_path}: {error}")
            
            if len(invalid_files) > 10:
                print(f"  ... and {len(invalid_files) - 10} more")
            
            response = input("\nContinue copying anyway? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("Cancelled.")
                return
    
    # Preview or execute
    if dry_run:
        print(f"\n{'='*80}")
        print("DRY RUN MODE - No files will be copied")
        print(f"{'='*80}")
        print("\nFiles that would be copied:")
        for i, (rel_path, src, tgt) in enumerate(to_copy[:20], 1):
            print(f"  {i}. {rel_path}")
        
        if len(to_copy) > 20:
            print(f"  ... and {len(to_copy) - 20} more")
        
        print(f"\nRun without --dry_run to perform the merge.")
    else:
        # Confirm
        print(f"\n{'='*80}")
        print(f"Ready to copy {len(to_copy)} files")
        print(f"{'='*80}")
        response = input("\nProceed with merge? (yes/no): ").strip().lower()
        
        if response not in ['yes', 'y']:
            print("Cancelled.")
            return
        
        # Copy files
        print("\nCopying files...")
        copied_count = 0
        failed_count = 0
        
        for rel_path, source_file, target_file in tqdm(to_copy, desc="Copying"):
            try:
                # Create parent directory if needed
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source_file, target_file)
                copied_count += 1
            except Exception as e:
                print(f"\nError copying {rel_path}: {e}")
                failed_count += 1
        
        print(f"\n{'='*80}")
        print("Merge Complete!")
        print(f"{'='*80}")
        print(f"Files copied:  {copied_count}")
        print(f"Files failed:  {failed_count}")
        print(f"Files skipped (conflicts): {len(conflicts)}")
        print(f"\nTarget directory now has: {len(target_files) + copied_count} pkl files")
        
        if failed_count > 0:
            print(f"\n⚠️  {failed_count} files failed to copy. Check errors above.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge pkl directories from split processing"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory with new pkl files (e.g., data/karsl502-63kpts-rest)"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target directory to merge into (e.g., data/karsl502-63kpts)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview merge without copying files"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify pkl files are valid before copying (slower)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PKL Directory Merge")
    print("=" * 80)
    print(f"Source: {args.source}")
    print(f"Target: {args.target}")
    if args.dry_run:
        print("Mode:   DRY RUN (preview only)")
    if args.verify:
        print("Verify: ENABLED")
    print("=" * 80)
    print()
    
    merge_pkl_directories(args.source, args.target, args.dry_run, args.verify)


if __name__ == "__main__":
    main()
