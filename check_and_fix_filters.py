#!/usr/bin/env python3
"""
check_and_fix_filters.py

Check if train_loso_functional_qat.py has the correct dense_filters configuration.
Optionally fix it if misconfigured.
"""
import re
import sys
from pathlib import Path

def check_file(filepath):
    """Check the dense_filters configuration in the file."""
    with open(filepath, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Find the dense_filters line
    filter_start = None
    for i, line in enumerate(lines):
        if 'dense_filters = args.quantize_dense_names or [' in line:
            filter_start = i
            break
    
    if filter_start is None:
        print("ERROR: Could not find 'dense_filters = args.quantize_dense_names or [' in file")
        return None, None, None
    
    # Extract the full list (find closing bracket)
    filter_end = None
    for i in range(filter_start, min(filter_start + 10, len(lines))):
        if ']' in lines[i]:
            filter_end = i
            break
    
    if filter_end is None:
        print("ERROR: Could not find closing bracket for dense_filters")
        return None, None, None
    
    # Get the filter definition
    filter_lines = lines[filter_start:filter_end+1]
    filter_text = '\n'.join(filter_lines)
    
    print("Current configuration (lines {}–{})".format(filter_start+1, filter_end+1))
    print("="*80)
    for i, line in enumerate(filter_lines, start=filter_start+1):
        print(f"{i:3d} | {line}")
    print("="*80)
    print()
    
    # Parse active filters (not commented)
    active_filters = []
    commented_filters = []
    
    for line in filter_lines:
        # Check if line contains a string in quotes
        matches = re.findall(r'"([^"]+)"', line)
        if matches:
            # Check if the line is commented
            stripped = line.lstrip()
            if stripped.startswith('#'):
                commented_filters.extend(matches)
            else:
                active_filters.extend(matches)
    
    print(f"Active filters (will be quantized): {active_filters}")
    print(f"Commented filters (excluded): {commented_filters}")
    print()
    
    # Check for sensitive layers
    sensitive = ['q_proj', 'k_proj', 'v_proj', 'out_proj']
    dangerous_active = [f for f in active_filters if f in sensitive]
    
    if dangerous_active:
        print(f"⚠️  DANGER: Sensitive attention layers in active filters: {dangerous_active}")
        print(f"⚠️  This will cause training collapse!")
        print()
        return 'UNSAFE', filter_start, filter_end
    else:
        print(f"✓ SAFE: No sensitive attention layers in active filters")
        print(f"✓ Only safe FFN layers will be quantized")
        print()
        return 'SAFE', filter_start, filter_end


def fix_file(filepath, filter_start, filter_end, dry_run=True):
    """Fix the dense_filters configuration."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Replace the problematic section
    correct_config = [
        '    dense_filters = args.quantize_dense_names or [\n',
        '        "fc1", "fc2",          # FFN layers only\n',
        '        # "proj_x1", "proj_y1",  # Projection (optional, can add if needed)\n',
        '        # "q_proj", "k_proj", "v_proj", "out_proj",  # ATTENTION PROJECTIONS - DO NOT QUANTIZE!\n',
        '    ]\n',
    ]
    
    new_lines = lines[:filter_start] + correct_config + lines[filter_end+1:]
    
    if dry_run:
        print("="*80)
        print("PROPOSED FIX (dry run - no changes made)")
        print("="*80)
        print()
        print(f"Would replace lines {filter_start+1}–{filter_end+1} with:")
        print()
        for i, line in enumerate(correct_config, start=filter_start+1):
            print(f"{i:3d} | {line}", end='')
        print()
        print("="*80)
        print()
        print("To apply this fix, run:")
        print(f"  python {sys.argv[0]} --fix")
        print()
    else:
        # Backup original
        backup_path = filepath + '.backup'
        with open(backup_path, 'w') as f:
            f.writelines(lines)
        print(f"✓ Created backup: {backup_path}")
        
        # Write fixed version
        with open(filepath, 'w') as f:
            f.writelines(new_lines)
        print(f"✓ Fixed configuration in {filepath}")
        print()
        
        # Re-check
        print("Verifying fix...")
        status, _, _ = check_file(filepath)
        if status == 'SAFE':
            print()
            print("="*80)
            print("✓ SUCCESS: Configuration fixed and verified!")
            print("="*80)
        else:
            print()
            print("⚠️  WARNING: Fix may not have worked correctly")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Check and fix dense_filters configuration in train_loso_functional_qat.py"
    )
    parser.add_argument("--file", type=str, default="train_loso_functional_qat.py",
                       help="Path to file to check (default: train_loso_functional_qat.py)")
    parser.add_argument("--fix", action="store_true",
                       help="Actually fix the file (default: dry run only)")
    args = parser.parse_args()
    
    filepath = Path(args.file)
    
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        print(f"Current directory: {Path.cwd()}")
        sys.exit(1)
    
    print("="*80)
    print("CHECKING DENSE_FILTERS CONFIGURATION")
    print("="*80)
    print(f"File: {filepath}")
    print()
    
    status, filter_start, filter_end = check_file(filepath)
    
    if status is None:
        print("ERROR: Could not parse file")
        sys.exit(1)
    
    if status == 'UNSAFE':
        print()
        print("="*80)
        print("⚠️  UNSAFE CONFIGURATION DETECTED")
        print("="*80)
        print()
        print("Your code will quantize sensitive attention layers!")
        print("This explains your training collapse (95% → 11%).")
        print()
        
        if not args.fix:
            fix_file(filepath, filter_start, filter_end, dry_run=True)
        else:
            response = input("Apply fix? This will modify the file (backup will be created) [y/N]: ")
            if response.lower() == 'y':
                fix_file(filepath, filter_start, filter_end, dry_run=False)
            else:
                print("Fix cancelled.")
    else:
        print("="*80)
        print("✓ CONFIGURATION IS SAFE")
        print("="*80)
        print()
        print("Your dense_filters are correctly configured.")
        print("The training collapse is likely due to:")
        print("  • Learning rate too high (try --lr 1e-5)")
        print("  • Batch size too small (try --batch_size 4)")
        print()


if __name__ == "__main__":
    main()

