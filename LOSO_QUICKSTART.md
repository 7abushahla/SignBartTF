# LOSO Training - Quick Start

## 🎯 What You Need to Know

**LOSO (Leave-One-Subject-Out)** = Train on 11 users, test on 1 held-out user

For Arabic ASL, we test on 4 representative users: **user01, user02, user08, user11**

**Output layout update**: results now live under `outputs/<dataset>/<run_type>/...` by default (e.g., `outputs/arabic_asl/loso/checkpoints/arabic_asl_LOSO_user01`). Legacy paths like `checkpoints_*`, `exports/`, `results/`, and `logs/` may still exist for older runs.

## 🚀 Two-Step Process

### Step 1: Test Single LOSO First (30 min - 2 hours)

```bash
cd ml_training/models/signbart_tf

python train_loso.py \
    --config_path configs/arabic-asl.yaml \
    --base_data_path /path/to/arabic-asl \
    --holdout_only user01 \
    --epochs 80 \
    --lr 2e-4 \
    --seed 42
```

**What this does:**
- ✅ Tests training on LOSO setup with user01 held out
- ✅ Verifies your data paths are correct
- ✅ Checks model trains without errors
- ✅ Takes 2-4 hours instead of 8-16 hours

**Check it worked:**
```bash
ls checkpoints_arabic_asl_LOSO_user01/
cat arabic_asl_LOSO_user01.log
```

### Step 2: Run All 4 LOSO Experiments (8-16 hours)

Once Step 1 works successfully:

```bash
python train_loso.py \
    --config_path configs/arabic-asl.yaml \
    --base_data_path /path/to/arabic-asl \
    --epochs 80 \
    --lr 2e-4 \
    --seed 42
```

**Note:** Just remove `--holdout_only`! That's it!

**What this does:**
- Runs 4 experiments (user01, user02, user08, user11 as holdouts)
- Saves results for each in separate directories
- Creates metadata and logs for analysis

## 📁 Data Structure Required

Your data should be organized like this:

```
/path/to/arabic-asl_LOSO_user01/
├── train/          # user02-12 samples
│   ├── G01/
│   ├── G02/
│   └── ...
└── test/           # user01 samples
    ├── G01/
    ├── G02/
    └── ...

/path/to/arabic-asl_LOSO_user02/
├── train/          # user01, user03-12 samples
└── test/           # user02 samples

... (and so on for user08, user11)
```

## 📊 What You Get

For each LOSO experiment (e.g., `arabic_asl_LOSO_user01`):

```
arabic_asl_LOSO_user01.log              # Training log
arabic_asl_LOSO_user01_eval.log         # Evaluation results
checkpoints_arabic_asl_LOSO_user01/     # Model checkpoints (.h5 files)
out-imgs/arabic_asl_LOSO_user01_*.png   # Training curves
training_metadata/...                    # Structured metadata
```

## 🔍 Check Results

```bash
# View training log
tail -50 arabic_asl_LOSO_user01.log

# View evaluation results
grep "Accuracy\|Loss" arabic_asl_LOSO_user01_eval.log

# Check best checkpoint
ls -lh checkpoints_arabic_asl_LOSO_user01/*best*
```

## ⚙️ Common Options

```bash
# Quick test (fewer epochs)
python train_loso.py \
    --holdout_only user01 \
    --epochs 20 \
    --lr 2e-4

# Add experiment prefix
python train_loso.py \
    --exp_prefix "100kp_6L" \
    --epochs 80

# Train without validation (faster)
python train_loso.py \
    --no_validation \
    --epochs 80
```

## 🐛 Troubleshooting

### "ERROR: Missing LOSO data directories"
→ Run your data preparation script first to create `_LOSO_user*` splits

### "Out of memory"
→ Reduce `batch_size` in your config file to 8 or 4

### Takes too long
→ Use `--epochs 20` for testing, then increase for final run

## 📈 After LOSO Training

### Convert best model to TFLite:

```bash
python convert_to_tflite.py \
    --config configs/arabic-asl.yaml \
    --checkpoint checkpoints_arabic_asl_LOSO_user01/checkpoint_50_best_val.h5 \
    --quantization float16 \
    --output signbart_loso_user01.tflite
```

### Compare all results:

```bash
# View all evaluation logs
grep -A 5 "Evaluation Results" *_eval.log
```

## 📚 More Information

- Detailed guide: `LOSO_GUIDE.md`
- General training: `MAIN_PY_GUIDE.md`
- Quick start: `QUICKSTART.md`
- Full docs: `README.md`

---

## 💡 Remember

1. **Always test with `--holdout_only user01` first!**
2. Check results before running all 4 LOSO experiments
3. Each LOSO experiment takes 2-4 hours on GPU
4. All 4 experiments take 8-16 hours total
5. Use best_val checkpoint for deployment

**Start here:** `python train_loso.py --holdout_only user01 --epochs 80 ...`

