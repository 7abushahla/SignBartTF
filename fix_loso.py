import os
import shutil
import glob
import re

BASE_ROOT = "data/arabic-asl"
HOLDOUTS = ["user01", "user08", "user11"]
GESTURES = [f"G{i:02d}" for i in range(1, 11)]

# Recreate LOSO directories with label files
for hu in HOLDOUTS:
    loso_root = f"{BASE_ROOT}_LOSO_{hu}"
    
    # Create directory structure
    for split in ("train", "test"):
        for g in GESTURES:
            os.makedirs(os.path.join(loso_root, split, g), exist_ok=True)
    
    # Copy label files
    for fname in ["label2id.json", "id2label.json"]:
        src = os.path.join(BASE_ROOT, fname)
        dst = os.path.join(loso_root, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {fname} to {loso_root}")
    
    # Copy pickle files
    all_glob = os.path.join(BASE_ROOT, "all", "G??", "*.pkl")
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