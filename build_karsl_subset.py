#!/usr/bin/env python3
"""
build_karsl_subset.py

Create a smaller KArSL dataset root from a larger one by selecting a gesture range.

This is mainly intended to build KArSL-100 (G0071..G0170) from KArSL-502.

The output folder will contain:
  - all/Gxxxx/ (directories copied or symlinked)
  - label2id.json / id2label.json (remapped to 0..N-1)
  - subset_info.json (metadata)

Examples:
  # Build KArSL-100 from KArSL-502 using symlinks (fast, minimal disk):
  python build_karsl_subset.py \
      --src_root data/karsl502-63kpts \
      --dst_root data/karsl100-63kpts \
      --start 71 --end 170 \
      --mode symlink
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Build a KArSL subset dataset root (e.g., KArSL-100).")
    p.add_argument("--src_root", type=str, required=True, help="Source dataset root (contains all/, label2id.json).")
    p.add_argument("--dst_root", type=str, required=True, help="Destination dataset root to create/update.")
    p.add_argument("--start", type=int, default=71, help="Start gesture id (inclusive), e.g. 71 -> G0071.")
    p.add_argument("--end", type=int, default=170, help="End gesture id (inclusive), e.g. 170 -> G0170.")
    p.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to materialize gesture directories in dst_root/all (default: symlink).",
    )
    return p.parse_args()


def gesture_code(i: int) -> str:
    return f"G{i:04d}"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def main():
    args = parse_args()

    src_root = Path(args.src_root).expanduser().resolve()
    dst_root = Path(args.dst_root).expanduser().resolve()

    if not (src_root / "all").is_dir():
        raise SystemExit(f"ERROR: Missing source directory: {src_root / 'all'}")
    if not (src_root / "label2id.json").is_file() or not (src_root / "id2label.json").is_file():
        raise SystemExit(f"ERROR: Missing label files in source root: {src_root}")

    labels = [gesture_code(i) for i in range(int(args.start), int(args.end) + 1)]
    if not labels:
        raise SystemExit("ERROR: Empty label selection (check --start/--end).")

    ensure_dir(dst_root / "all")

    missing = []
    created = 0
    for lbl in labels:
        src_dir = src_root / "all" / lbl
        dst_dir = dst_root / "all" / lbl

        if not src_dir.is_dir():
            missing.append(lbl)
            continue

        if dst_dir.exists():
            continue

        if args.mode == "symlink":
            # Create a relative symlink for portability within the repo
            rel_target = os.path.relpath(src_dir, start=dst_dir.parent)
            dst_dir.symlink_to(rel_target, target_is_directory=True)
        else:
            shutil.copytree(src_dir, dst_dir)
        created += 1

    if missing:
        print(f"WARNING: {len(missing)} gesture dirs were missing in source (showing up to 10): {missing[:10]}")

    # Remap labels to contiguous ids
    label2id = {lbl: idx for idx, lbl in enumerate(labels)}
    id2label = {str(idx): lbl for idx, lbl in enumerate(labels)}

    write_json(dst_root / "label2id.json", label2id)
    write_json(dst_root / "id2label.json", id2label)

    info = {
        "source_root": str(src_root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "subset_range": {"start": int(args.start), "end": int(args.end)},
        "num_labels": len(labels),
        "mode": args.mode,
        "created_gesture_dirs": created,
        "missing_gesture_dirs": missing,
    }
    write_json(dst_root / "subset_info.json", info)

    print("=" * 80)
    print("KArSL SUBSET BUILD COMPLETE")
    print("=" * 80)
    print(f"Source: {src_root}")
    print(f"Dest  : {dst_root}")
    print(f"Range : {labels[0]}..{labels[-1]} ({len(labels)} labels)")
    print(f"Mode  : {args.mode}")
    print(f"Created gesture dirs: {created}")
    if missing:
        print(f"Missing in source: {len(missing)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

