#!/usr/bin/env python
"""
MOREUSERSprepare_arabic_asl.py - Prepare Arabic ASL dataset for LOSO cross-validation
Creates separate data directories for each LOSO configuration (4 users version)
Handles data organized by gesture classes (G01-G10) with user files inside
"""

import os
import shutil
import argparse
from pathlib import Path

# LOSO configurations for 4 users
LOSO_CONFIGS = [
    {
        "holdout_user": "user01",
        "train_users": ["user02", "user03", "user04", "user05", "user06", "user07", 
                       "user08", "user09", "user10", "user11", "user12"]
    },
    {
        "holdout_user": "user02",
        "train_users": ["user01", "user03", "user04", "user05", "user06", "user07",
                       "user08", "user09", "user10", "user11", "user12"]
    },
    {
        "holdout_user": "user08",
        "train_users": ["user01", "user02", "user03", "user04", "user05", "user06", "user07",
                       "user09", "user10", "user11", "user12"]
    },
    {
        "holdout_user": "user11",
        "train_users": ["user01", "user02", "user03", "user04", "user05", "user06", "user07",
                       "user08", "user09", "user10", "user12"]
    }
]

def check_existing_splits(base_data_path):
    """Check which LOSO splits already exist."""
    existing = []
    missing = []
    
    for config in LOSO_CONFIGS:
        holdout = config["holdout_user"]
        loso_path = f"{base_data_path}_LOSO_{holdout}"
        
        if os.path.exists(loso_path):
            existing.append((holdout, loso_path))
        else:
            missing.append((holdout, loso_path))
    
    return existing, missing

def get_user_confirmation(holdout_user, loso_path):
    """Ask user if they want to regenerate existing split."""
    print(f"\nLOSO split for {holdout_user} already exists at: {loso_path}")
    while True:
        response = input(f"Do you want to regenerate this split? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please answer 'yes' or 'no'")

def copy_metadata_files(data_source, loso_path):
    """Copy metadata files (label2id.json, etc.) to LOSO directory."""
    metadata_files = ['label2id.json', 'id2label.json', 'vocab.json', 'config.json']
    copied_files = []
    
    for filename in metadata_files:
        src_file = os.path.join(data_source, filename)
        if os.path.exists(src_file):
            dst_file = os.path.join(loso_path, filename)
            shutil.copy2(src_file, dst_file)
            copied_files.append(filename)
    
    return copied_files

def create_loso_split(base_data_path, holdout_user, train_users, force=False):
    """Create LOSO split for one holdout user."""
    loso_path = f"{base_data_path}_LOSO_{holdout_user}"
    
    # Check if it already exists
    if os.path.exists(loso_path):
        if not force:
            print(f"✓ LOSO split for {holdout_user} already exists, skipping...")
            return True
        else:
            print(f"Removing existing split for {holdout_user}...")
            shutil.rmtree(loso_path)
    
    print(f"\nCreating LOSO split for test user: {holdout_user}")
    print(f"  Training users: {', '.join(train_users)}")
    print(f"  Output path: {loso_path}")
    
    # Check if base data exists
    if not os.path.exists(base_data_path):
        print(f"ERROR: Base data path does not exist: {base_data_path}")
        return False
    
    # Look for 'all' subdirectory (common structure)
    data_source = os.path.join(base_data_path, "all")
    if not os.path.exists(data_source):
        # Try base path directly
        data_source = base_data_path
    
    print(f"  Data source: {data_source}")
    
    # Create LOSO directory first
    os.makedirs(loso_path, exist_ok=True)
    
    # Copy metadata files (label2id.json, etc.)
    print(f"  Copying metadata files...")
    copied_files = copy_metadata_files(data_source, loso_path)
    if copied_files:
        print(f"    ✓ Copied: {', '.join(copied_files)}")
    else:
        print(f"    ⚠ Warning: No metadata files found in {data_source}")
        print(f"    Looking for: label2id.json, id2label.json, vocab.json, config.json")
    
    # Get gesture directories (G01, G02, etc.)
    gesture_dirs = []
    for item in os.listdir(data_source):
        item_path = os.path.join(data_source, item)
        if os.path.isdir(item_path) and item.startswith("G"):
            gesture_dirs.append(item)
    
    if not gesture_dirs:
        print(f"ERROR: No gesture directories (G01, G02, ...) found in {data_source}")
        return False
    
    gesture_dirs.sort()
    print(f"  Found {len(gesture_dirs)} gesture classes: {', '.join(gesture_dirs)}")
    
    # Create directory structure
    train_dir = os.path.join(loso_path, "train")
    test_dir = os.path.join(loso_path, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    train_count = 0
    test_count = 0
    
    # Process each gesture directory
    for gesture in gesture_dirs:
        src_gesture_path = os.path.join(data_source, gesture)
        
        # Get all files in this gesture directory
        files = [f for f in os.listdir(src_gesture_path) if f.endswith('.pkl')]
        
        if not files:
            print(f"    ⚠ Warning: No .pkl files found in {gesture}")
            continue
        
        # Create gesture directories in train and test
        train_gesture_dir = os.path.join(train_dir, gesture)
        test_gesture_dir = os.path.join(test_dir, gesture)
        os.makedirs(train_gesture_dir, exist_ok=True)
        os.makedirs(test_gesture_dir, exist_ok=True)
        
        gesture_train_count = 0
        gesture_test_count = 0
        
        # Split files based on user
        for file in files:
            src_file = os.path.join(src_gesture_path, file)
            
            # Check which user this file belongs to
            # Files are named like: user02_G01_R01.pkl
            user_match = None
            for user in train_users + [holdout_user]:
                if file.startswith(user + "_"):
                    user_match = user
                    break
            
            if not user_match:
                continue
            
            # Copy to appropriate directory
            if user_match == holdout_user:
                # Test set
                dst_file = os.path.join(test_gesture_dir, file)
                shutil.copy2(src_file, dst_file)
                test_count += 1
                gesture_test_count += 1
            elif user_match in train_users:
                # Training set
                dst_file = os.path.join(train_gesture_dir, file)
                shutil.copy2(src_file, dst_file)
                train_count += 1
                gesture_train_count += 1
        
        print(f"    ✓ {gesture}: {gesture_train_count} train, {gesture_test_count} test files")
    
    print(f"  Summary: {train_count} training files, {test_count} test files")
    
    if test_count == 0:
        print(f"    ✗ ERROR: No test files found for {holdout_user}")
        return False
    
    if train_count == 0:
        print(f"    ✗ ERROR: No training files found")
        return False
    
    print(f"✓ LOSO split created successfully!\n")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Arabic ASL dataset for LOSO cross-validation (4 users)"
    )
    parser.add_argument(
        "--base_data_path", 
        type=str, 
        default="data/arabic-asl",
        help="Base path to processed data (without _LOSO suffix)"
    )
    parser.add_argument(
        "--force_all",
        action="store_true",
        help="Force regeneration of all LOSO splits without asking"
    )
    parser.add_argument(
        "--user_only",
        type=str,
        default="",
        help="Create split for specific user only (e.g., 'user02')"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Arabic ASL LOSO Data Preparation (4 Users)")
    print("="*80)
    print(f"Base data path: {args.base_data_path}")
    print(f"LOSO configurations: {len(LOSO_CONFIGS)} users")
    print("="*80)
    print()
    
    # Check if base data exists
    if not os.path.exists(args.base_data_path):
        print(f"ERROR: Base data directory not found: {args.base_data_path}")
        print(f"Please ensure the data is prepared first.")
        return
    
    # Check existing splits
    existing, missing = check_existing_splits(args.base_data_path)
    
    if existing:
        print("Existing LOSO splits found:")
        for holdout, path in existing:
            print(f"  ✓ {holdout}: {path}")
        print()
    
    if missing:
        print("Missing LOSO splits:")
        for holdout, path in missing:
            print(f"  ✗ {holdout}: {path}")
        print()
    
    # Determine which splits to create
    configs_to_create = []
    
    if args.user_only:
        # Create only for specific user
        config = next((c for c in LOSO_CONFIGS if c["holdout_user"] == args.user_only), None)
        if not config:
            print(f"ERROR: Invalid user: {args.user_only}")
            print(f"Valid options: user01, user02, user08, user11")
            return
        
        loso_path = f"{args.base_data_path}_LOSO_{args.user_only}"
        if os.path.exists(loso_path):
            if args.force_all or get_user_confirmation(args.user_only, loso_path):
                configs_to_create.append((config, True))  # force=True
            else:
                print(f"Skipping {args.user_only}")
        else:
            configs_to_create.append((config, False))
    else:
        # Process all configs
        for config in LOSO_CONFIGS:
            holdout = config["holdout_user"]
            loso_path = f"{args.base_data_path}_LOSO_{holdout}"
            
            if os.path.exists(loso_path):
                # Existing split - ask or force
                if args.force_all:
                    configs_to_create.append((config, True))  # force=True
                else:
                    if get_user_confirmation(holdout, loso_path):
                        configs_to_create.append((config, True))  # force=True
                    else:
                        print(f"Keeping existing split for {holdout}")
            else:
                # Missing split - create by default
                configs_to_create.append((config, False))
    
    # Create the splits
    if not configs_to_create:
        print("\n" + "="*80)
        print("No splits to create. All requested splits already exist.")
        print("="*80)
        return
    
    print("\n" + "="*80)
    print(f"Creating {len(configs_to_create)} LOSO split(s)...")
    print("="*80)
    
    success_count = 0
    for config, force in configs_to_create:
        if create_loso_split(
            args.base_data_path,
            config["holdout_user"],
            config["train_users"],
            force=force
        ):
            success_count += 1
    
    # Final summary
    print("="*80)
    print("LOSO Data Preparation Complete")
    print("="*80)
    print(f"Successfully created: {success_count}/{len(configs_to_create)} splits")
    print()
    
    # Show all available splits
    existing, missing = check_existing_splits(args.base_data_path)
    
    if existing:
        print("Available LOSO splits:")
        for holdout, path in existing:
            print(f"  ✓ {holdout}: {path}")
        print()
    
    if missing:
        print("Still missing:")
        for holdout, path in missing:
            print(f"  ✗ {holdout}: {path}")
        print()
    
    if not missing:
        print("All LOSO splits are ready!")
        print("\nYou can now run training with:")
        print("  python tools/MOREUSERStrain_loso.py --epochs 80 --lr 2e-4 --no_validation")
    else:
        print("Some splits are still missing. Run this script again to create them.")
    
    print("="*80)

if __name__ == "__main__":
    main()