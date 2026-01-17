import os
import shutil
import glob
import re
import argparse


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
        m = re.match(r"(user\d{2})_G\d{2}_R\d{2}\.pkl$", bn)
        if m:
            users.add(m.group(1))
    return sorted(users)


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
        default="G01-G10",
        help="Gesture list (e.g., 'G01-G10' or 'G01,G02,G03')"
    )
    args = parser.parse_args()

    base_root = args.base_root
    if args.holdouts.strip().lower() == "all":
        holdouts = discover_users(base_root)
    else:
        holdouts = [u.strip() for u in args.holdouts.split(",") if u.strip()]
    gestures = parse_gestures(args.gestures)

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
            m = re.match(r"(user\d{2})_(G\d{2})_(R\d{2})\.pkl$", bn)
            if not m:
                continue
            user, gesture, repeat = m.group(1), m.group(2), m.group(3)

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