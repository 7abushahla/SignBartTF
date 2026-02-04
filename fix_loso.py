import os
import shutil
import glob
import re
import argparse
import json
import pickle


def parse_gestures(gestures_arg):
    gestures_arg = gestures_arg.strip()
    if "-" in gestures_arg and gestures_arg.startswith("G"):
        start, end = gestures_arg.split("-", 1)
        start_i = int(start[1:])
        end_i = int(end[1:])
        return [f"G{i:02d}" for i in range(start_i, end_i + 1)]
    return [g.strip() for g in gestures_arg.split(",") if g.strip()]


def discover_users(base_root):
    """Discover user IDs from filenames in base_root/all/G??/*.pkl."""
    all_glob = os.path.join(base_root, "all", "G??", "*.pkl")
    users = set()
    for p in glob.glob(all_glob):
        bn = os.path.basename(p)
        m = re.match(r"(user\d+)_G\d{2}_.*\.pkl$", bn)
        if m:
            users.add(m.group(1))
    return sorted(users)


def discover_gestures(base_root):
    """Discover gesture IDs from label2id.json or all/G?? directories."""
    label_path = os.path.join(base_root, "label2id.json")
    if os.path.exists(label_path):
        try:
            with open(label_path, "r") as f:
                label2id = json.load(f)
            gestures = sorted(label2id.keys(), key=lambda x: int(x[1:]) if x.startswith("G") else x)
            if gestures:
                return gestures
        except Exception:
            pass
    # Fallback: discover from directories
    all_dir = os.path.join(base_root, "all")
    if os.path.isdir(all_dir):
        gestures = [d for d in os.listdir(all_dir) if re.match(r"^G\d{2}$", d)]
        return sorted(gestures, key=lambda x: int(x[1:]))
    return []


def parse_sample_metadata(path):
    """Fallback: read user/class from pickle if filename parsing fails."""
    try:
        with open(path, "rb") as f:
            sample = pickle.load(f)
        user = sample.get("user")
        gesture = sample.get("class")
        return user, gesture
    except Exception:
        return None, None


def parse_filename(bn):
    """Parse user + gesture from filename, supports Arabic ASL and LSA64 formats."""
    patterns = [
        r"^(user\d+)_((G\d{2}))_R\d{2}\.pkl$",          # user01_G01_R01.pkl
        r"^(user\d+)_((G\d{2}))_\d+_\d+_\d+\.pkl$",     # user0001_G01_001_001_001.pkl
        r"^(user\d+)_((G\d{2}))_.*\.pkl$",              # fallback
    ]
    for pat in patterns:
        m = re.match(pat, bn)
        if m:
            user = m.group(1)
            gesture = m.group(2)
            return user, gesture
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Create LOSO splits from extracted keypoints")
    parser.add_argument(
        "--base_root",
        type=str,
        default="data/arabic-asl-90kpts",
        help="Base dataset root (contains 'all/' and label2id.json)"
    )
    parser.add_argument(
        "--holdouts",
        type=str,
        default="user01,user08,user11",
        help="Comma-separated holdout users or 'all' to use all users"
    )
    parser.add_argument(
        "--gestures",
        type=str,
        default="auto",
        help="Gesture list (e.g., 'G01-G10' or 'G01,G02,G03' or 'auto')"
    )
    args = parser.parse_args()

    base_root = args.base_root
    if args.holdouts.strip().lower() == "all":
        holdouts = discover_users(base_root)
    else:
        holdouts = [u.strip() for u in args.holdouts.split(",") if u.strip()]
    if args.gestures.strip().lower() == "auto":
        gestures = discover_gestures(base_root)
    else:
        gestures = parse_gestures(args.gestures)
    if not gestures:
        raise RuntimeError(f"Could not discover gestures for base_root: {base_root}")

    # Recreate LOSO directories with label files
    for hu in holdouts:
        loso_root = f"{base_root}_LOSO_{hu}"

        # Create directory structure
        for split in ("train", "test"):
            for g in gestures:
                os.makedirs(os.path.join(loso_root, split, g), exist_ok=True)

        # Copy label files
        for fname in ["label2id.json", "id2label.json"]:
            src = os.path.join(base_root, fname)
            dst = os.path.join(loso_root, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"Copied {fname} to {loso_root}")
            else:
                print(f"Warning: {fname} not found in {base_root}")

        # Copy pickle files
        all_glob = os.path.join(base_root, "all", "G??", "*.pkl")
        copied_train = 0
        copied_test = 0

        for p in glob.glob(all_glob):
            bn = os.path.basename(p)
            user, gesture = parse_filename(bn)
            if not user or not gesture:
                # Fallback to metadata inside pickle
                user, gesture = parse_sample_metadata(p)
            if not user or not gesture:
                continue
            if gesture not in gestures:
                continue

            dst_split = "test" if user == hu else "train"
            dst = os.path.join(loso_root, dst_split, gesture, bn)

            if not os.path.exists(dst):
                shutil.copy2(p, dst)
                if dst_split == "train":
                    copied_train += 1
                else:
                    copied_test += 1

        print(f"LOSO {hu}: {copied_train} train samples, {copied_test} test samples\n")

    print("LOSO directories fixed!")


if __name__ == "__main__":
    main()
