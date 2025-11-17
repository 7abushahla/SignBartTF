# LOSO (Leave-One-Subject-Out) Training Guide

## Overview

The `train_loso.py` script automates Leave-One-Subject-Out cross-validation for the Arabic ASL dataset. This is the recommended approach for evaluating model performance on unseen subjects.

## What is LOSO?

**Leave-One-Subject-Out (LOSO)** cross-validation:
- Trains on all subjects except one
- Tests on the held-out subject
- Repeats for each subject
- Provides robust evaluation of generalization to new users

For Arabic ASL with 12 users, we use 4 representative holdout users: user01, user02, user08, user11.

## Quick Start

### Step 1: Test with Single LOSO (Recommended First!)

Before running all 4 LOSO experiments, test with just one:

```bash
python train_loso.py \
    --config_path configs/arabic-asl.yaml \
    --base_data_path /path/to/arabic-asl \
    --holdout_only user01 \
    --epochs 80 \
    --lr 2e-4 \
    --seed 42
```

**Why test first?**
- Verifies your data paths are correct
- Checks model trains without errors
- Takes ~2-4 hours instead of 8-16 hours for all
- You can stop if something is wrong

### Step 2: Run All LOSO Experiments

Once the single test works:

```bash
python train_loso.py \
    --config_path configs/arabic-asl.yaml \
    --base_data_path /path/to/arabic-asl \
    --epochs 80 \
    --lr 2e-4 \
    --seed 42
```

**Note**: Remove `--holdout_only` to run all 4 experiments!

## Command Line Arguments

### Required
- `--config_path`: Path to model config YAML file
- `--base_data_path`: Base path to data (without `_LOSO_user*` suffix)

### Important
- `--holdout_only`: **Test on single user first!** (e.g., `user01`)
- `--epochs`: Number of training epochs (default: 80)
- `--lr`: Learning rate (default: 2e-4)
- `--seed`: Random seed for reproducibility (default: 379)

### Optional
- `--exp_prefix`: Add prefix to experiment names
- `--pretrained_path`: Start from pretrained weights (.h5 file)
- `--no_validation`: Disable validation during training
- `--skip_final_eval`: Skip final test evaluation
- `--skip_training`: Only run evaluation (if models exist)

## LOSO Configuration

The script runs experiments for 4 holdout users:

| Holdout | Training Users | Test User |
|---------|---------------|-----------|
| user01 | user02-12 (11 users) | user01 |
| user02 | user01, user03-12 (11 users) | user02 |
| user08 | user01-07, user09-12 (11 users) | user08 |
| user11 | user01-10, user12 (11 users) | user11 |

## Data Preparation

Your data directory structure should be:

```
/path/to/arabic-asl_LOSO_user01/
├── train/
│   ├── G01/
│   ├── G02/
│   └── ...
└── test/
    ├── G01/
    ├── G02/
    └── ...

/path/to/arabic-asl_LOSO_user02/
├── train/
└── test/

... (and so on for user08, user11)
```

## Output Files

For each LOSO experiment (e.g., `arabic_asl_LOSO_user01`):

### Training Outputs
- `arabic_asl_LOSO_user01.log` - Training log
- `checkpoints_arabic_asl_LOSO_user01/` - Model checkpoints
  - `checkpoint_*_best_train.h5` - Best training accuracy
  - `checkpoint_*_best_val.h5` - Best validation accuracy
  - `checkpoint_*_latest.h5` - Most recent epoch
  - `checkpoint_*_final.h5` - Final epoch
- `out-imgs/arabic_asl_LOSO_user01_loss.png` - Training curves
- `out-imgs/arabic_asl_LOSO_user01_lr.png` - Learning rate schedule

### Evaluation Outputs
- `arabic_asl_LOSO_user01_eval.log` - Evaluation results
- `training_metadata/arabic_asl_LOSO_user01_latest.json` - Structured metadata

### Metadata JSON Example

```json
{
  "experiment_name": "arabic_asl_LOSO_user01",
  "holdout_user": "user01",
  "train_users": "user02-user12",
  "start_time": "2025-11-15T10:00:00",
  "end_time": "2025-11-15T12:30:00",
  "duration_seconds": 9000,
  "epochs": 80,
  "learning_rate": 0.0002,
  "success": true,
  "model_parameters": {
    "estimated_parameters_M": 12.34,
    "model_size_mb": 47.12
  },
  "framework": "tensorflow"
}
```

## Usage Examples

### Example 1: Quick Test (Single User)

```bash
# Test with user01 first
python train_loso.py \
    --config_path configs/arabic-asl.yaml \
    --base_data_path data/arabic-asl \
    --holdout_only user01 \
    --epochs 10 \
    --lr 2e-4
```

### Example 2: Full LOSO (All Users)

```bash
# After testing works, run all 4
python train_loso.py \
    --config_path configs/arabic-asl.yaml \
    --base_data_path data/arabic-asl \
    --epochs 80 \
    --lr 2e-4 \
    --seed 42
```

### Example 3: With Experiment Prefix

```bash
# Add prefix to distinguish experiments
python train_loso.py \
    --config_path configs/arabic-asl.yaml \
    --base_data_path data/arabic-asl \
    --exp_prefix "100kp_6L" \
    --epochs 80
```

Creates experiments like: `100kp_6L_arabic_asl_LOSO_user01`

### Example 4: Evaluation Only

```bash
# If models already trained, just evaluate
python train_loso.py \
    --config_path configs/arabic-asl.yaml \
    --base_data_path data/arabic-asl \
    --skip_training
```

### Example 5: Training Without Validation

```bash
# Train only, evaluate on test after
python train_loso.py \
    --config_path configs/arabic-asl.yaml \
    --base_data_path data/arabic-asl \
    --no_validation \
    --epochs 80
```

## Workflow Recommendations

### Best Practice: Progressive Testing

**1. Single User Test (30 min - 2 hours)**
```bash
python train_loso.py --holdout_only user01 --epochs 10 ...
```
✓ Verify data loads correctly  
✓ Check model trains without errors  
✓ Confirm checkpoints save properly  

**2. Full Single User (2-4 hours)**
```bash
python train_loso.py --holdout_only user01 --epochs 80 ...
```
✓ Verify convergence  
✓ Check final accuracy  
✓ Inspect training curves  

**3. All LOSO Users (8-16 hours)**
```bash
python train_loso.py --epochs 80 ...  # Remove --holdout_only
```
✓ Full cross-validation  
✓ Robust performance metrics  

### Hardware Considerations

| Hardware | Recommended Batch Size | Expected Time/User |
|----------|----------------------|-------------------|
| CPU only | 4-8 | 6-8 hours |
| GPU (8GB) | 16-32 | 2-3 hours |
| GPU (16GB+) | 32-64 | 1-2 hours |

Adjust `batch_size` in your config file.

## Monitoring Progress

### Check Training Progress
```bash
# Watch log file
tail -f arabic_asl_LOSO_user01.log

# Check training curves
open out-imgs/arabic_asl_LOSO_user01_loss.png
```

### Check Checkpoints
```bash
ls -lh checkpoints_arabic_asl_LOSO_user01/
```

### Check Metadata
```bash
cat training_metadata/arabic_asl_LOSO_user01_latest.json | python -m json.tool
```

## Troubleshooting

### Issue: "ERROR: Missing LOSO data directories"
**Solution**: Run your data preparation script first to create LOSO splits with format:
- `{base_path}_LOSO_user01/`
- `{base_path}_LOSO_user02/`
- etc.

### Issue: "ERROR: Config file not found"
**Solution**: Check the path to your config file:
```bash
ls configs/arabic-asl.yaml
```

### Issue: Training fails with OOM (Out of Memory)
**Solution**: Reduce batch size in config:
```yaml
batch_size: 8  # or even 4
```

### Issue: Training takes too long
**Solution**: 
1. Reduce epochs for testing: `--epochs 20`
2. Use a smaller model (edit config)
3. Use GPU if available

### Issue: Checkpoints not saving
**Solution**: Check disk space:
```bash
df -h .
```

## Converting LOSO Results to TFLite

After LOSO training, convert the best model to TFLite:

```bash
# Find best checkpoint
ls -lh checkpoints_arabic_asl_LOSO_user01/*best_val*

# Convert to TFLite
python convert_to_tflite.py \
    --config configs/arabic-asl.yaml \
    --checkpoint checkpoints_arabic_asl_LOSO_user01/checkpoint_50_best_val.h5 \
    --quantization float16 \
    --output models/signbart_loso_user01.tflite \
    --test
```

## Comparing LOSO Results

After running all LOSO experiments, you can compare results:

```bash
# Check all metadata files
for user in user01 user02 user08 user11; do
    echo "=== $user ==="
    cat training_metadata/arabic_asl_LOSO_${user}_latest.json | \
        python -m json.tool | grep -A3 "success\|duration"
done
```

Or create a collection script (like `collect_results.py` from PyTorch version).

## Next Steps

1. **Start with single user**: `--holdout_only user01`
2. **Verify it works**: Check logs and checkpoints
3. **Run all LOSO**: Remove `--holdout_only` flag
4. **Analyze results**: Compare accuracy across all users
5. **Convert to TFLite**: Use best checkpoint for deployment

## Summary

| Command | Purpose | Time |
|---------|---------|------|
| `train_loso.py --holdout_only user01 --epochs 10` | Quick test | 30min-1hr |
| `train_loso.py --holdout_only user01` | Full single LOSO | 2-4 hrs |
| `train_loso.py` (no --holdout_only) | All LOSO (4 users) | 8-16 hrs |

**Always start with `--holdout_only` to test first!**

---

For more details, see:
- `QUICKSTART.md` - General training guide
- `MAIN_PY_GUIDE.md` - Detailed main.py documentation
- `README.md` - Full SignBART TensorFlow documentation

