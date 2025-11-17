# main.py - Full-Featured Training Script

## Overview

`main.py` is the comprehensive training script that mirrors all features from the PyTorch version. It provides advanced training capabilities, detailed logging, and training visualization.

## Key Differences: main.py vs train.py

| Feature | main.py | train.py |
|---------|---------|----------|
| **Argument Parsing** | ✅ Extensive (20+ args) | ⚠️ Basic (2 args) |
| **Learning Rate Scheduler** | ✅ ReduceLROnPlateau | ❌ Fixed LR |
| **Best Checkpoint Tracking** | ✅ Train + Val | ⚠️ Basic |
| **Training Curves** | ✅ Auto-generated plots | ❌ No plots |
| **Detailed Logging** | ✅ File + Console | ⚠️ Basic console |
| **Evaluation Mode** | ✅ Separate eval task | ❌ Only training |
| **Seed Control** | ✅ Full reproducibility | ❌ Not set |
| **Config Validation** | ✅ Extensive validation | ⚠️ Basic |
| **Resume Training** | ✅ Full state restore | ⚠️ Weights only |
| **Progress Tracking** | ✅ Detailed metrics | ⚠️ Basic metrics |
| **Checkpoint Strategy** | ✅ Multiple types | ⚠️ Basic |

**Recommendation**: Use `main.py` for serious experiments and `train.py` for quick tests.

## Complete Usage Examples

### Basic Training

```bash
python main.py \
    --experiment_name arabic_asl_exp1 \
    --config_path configs/arabic-asl.yaml \
    --data_path /path/to/arabic-asl-dataset \
    --task train \
    --epochs 200 \
    --lr 2e-5 \
    --seed 42
```

### Training with Custom Settings

```bash
python main.py \
    --experiment_name my_experiment \
    --config_path configs/your_config.yaml \
    --data_path /path/to/dataset \
    --task train \
    --epochs 100 \
    --lr 1e-4 \
    --scheduler_factor 0.5 \
    --scheduler_patience 10 \
    --save_every 5 \
    --seed 42
```

### Resume Training

```bash
python main.py \
    --experiment_name my_experiment \
    --config_path configs/your_config.yaml \
    --data_path /path/to/dataset \
    --task train \
    --resume_checkpoints checkpoints_my_experiment/checkpoint_50_latest.h5 \
    --epochs 200
```

### Training with Pretrained Weights

```bash
python main.py \
    --experiment_name finetune_exp \
    --config_path configs/your_config.yaml \
    --data_path /path/to/dataset \
    --task train \
    --pretrained_path models/pretrained_signbart.h5 \
    --epochs 100 \
    --lr 1e-5
```

### Evaluation Only

```bash
python main.py \
    --experiment_name eval_best_model \
    --config_path configs/your_config.yaml \
    --data_path /path/to/dataset \
    --task eval \
    --pretrained_path checkpoints_my_experiment/checkpoint_100_best_val.h5
```

### Training Without Validation (Training Set Only)

```bash
python main.py \
    --experiment_name train_only_exp \
    --config_path configs/your_config.yaml \
    --data_path /path/to/dataset \
    --task train \
    --no_validation \
    --epochs 200
```

### Save All Checkpoints (Disk Intensive)

```bash
python main.py \
    --experiment_name detailed_exp \
    --config_path configs/your_config.yaml \
    --data_path /path/to/dataset \
    --task train \
    --save_all_checkpoints \
    --epochs 50
```

## Command Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--experiment_name` | Name for logs and checkpoints | `my_experiment` |
| `--config_path` | Path to YAML config file | `configs/arabic-asl.yaml` |
| `--data_path` | Path to dataset root | `/data/arabic-asl` |
| `--task` | Task to perform | `train` or `eval` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 200 | Number of training epochs |
| `--lr` | 2e-5 | Initial learning rate |
| `--seed` | 379 | Random seed for reproducibility |
| `--pretrained_path` | "" | Path to pretrained weights (.h5) |
| `--resume_checkpoints` | "" | Path to checkpoint to resume from |
| `--scheduler_factor` | 0.1 | LR reduction factor |
| `--scheduler_patience` | 5 | Epochs before LR reduction |
| `--save_every` | 10 | Save checkpoint every N epochs |
| `--save_all_checkpoints` | False | Save every epoch (disk intensive) |
| `--no_validation` | False | Disable validation during training |

## Output Files

### Checkpoints

Saved in `checkpoints_{experiment_name}/`:

- `checkpoint_{epoch}_best_train.h5` - Best training accuracy
- `checkpoint_{epoch}_best_val.h5` - Best validation accuracy  
- `checkpoint_{epoch}_latest.h5` - Most recent epoch
- `checkpoint_{epoch}_final.h5` - Last epoch
- `checkpoint_{epoch}_epoch_{N}.h5` - Periodic saves

### Logs

- `{experiment_name}.log` - Training log file with timestamps
- Console output with detailed progress

### Plots

Saved in `out-imgs/{experiment_name}/`:

- `{experiment_name}_loss.png` - Training/validation curves
- `{experiment_name}_lr.png` - Learning rate schedule

## Training Output Example

```
================================================================================
Experiment: arabic_asl_exp1
Device: GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
================================================================================

Loaded config from configs/arabic-asl.yaml
  Config keys: ['d_model', 'encoder_layers', ...]
Using joint indices from config: 100 keypoints
Keypoint groups for normalization (4 groups):
  Group 1: body pose (indices 0-32, 33 keypoints)
  Group 2: left hand (indices 33-53, 21 keypoints)
  Group 3: right hand (indices 54-74, 21 keypoints)
  Group 4: face (indices 75-99, 25 keypoints)

Model Parameters:
  Total: 12,345,678
  Trainable: 12,345,678
  Size: ~47.12 MB (float32)

Training Configuration:
  Epochs: 200
  Learning rate: 2e-05
  Batch size: 16
  Optimizer: AdamW
  Scheduler: ReduceLROnPlateau (factor=0.1, patience=5)
  Seed: 42
  Validation: ENABLED
  Checkpoint strategy: Save every 10 epochs + best + latest + final

Model Architecture (from configs/arabic-asl.yaml):
  d_model: 256
  Encoder layers: 6
  Decoder layers: 6
  ...

[1/200] TRAIN  loss: 3.8765 acc: 0.1234 top5: 0.4567 | time: 234.5s
  → Saved best training checkpoint (acc: 0.1234)
[1/200] VAL    loss: 3.7654 acc: 0.1456 top5: 0.4789
  → Saved best validation checkpoint (acc: 0.1456)
...
```

## Best Practices

### 1. Experiment Naming

Use descriptive names:
```bash
--experiment_name signbart_100kp_6L_256d_lr2e5_seed42
```

### 2. Seed for Reproducibility

Always set a seed:
```bash
--seed 42
```

### 3. Learning Rate Schedule

Start with default, adjust if needed:
```bash
--scheduler_factor 0.5      # Less aggressive LR reduction
--scheduler_patience 10     # More patient before reducing
```

### 4. Checkpoint Strategy

For experiments:
```bash
--save_every 10             # Save every 10 epochs
```

For final training:
```bash
--save_every 5              # More frequent saves
--save_all_checkpoints      # Keep all epochs (if space allows)
```

### 5. Monitor Training

Watch the log file in real-time:
```bash
tail -f {experiment_name}.log
```

Check plots periodically:
```bash
open out-imgs/{experiment_name}_loss.png
```

## Comparison with PyTorch main.py

| Feature | PyTorch | TensorFlow (Ours) | Notes |
|---------|---------|-------------------|-------|
| Argument parsing | ✅ Same | ✅ Same | Identical interface |
| Seed setting | ✅ torch.manual_seed | ✅ tf.random.set_seed | Different APIs, same effect |
| LR scheduler | ✅ PyTorch ReduceLROnPlateau | ✅ Custom implementation | Same behavior |
| Checkpoint format | .pth | .h5 | Framework difference |
| Training loop | PyTorch autograd | GradientTape | Different APIs, same result |
| Plotting | ✅ matplotlib | ✅ matplotlib | Identical |
| Logging | ✅ Python logging | ✅ Python logging | Identical |
| Output format | ✅ Console + files | ✅ Console + files | Identical |

**Result**: Functionally equivalent with identical user experience!

## Troubleshooting

### Issue: "ERROR: --experiment_name is required"
**Solution**: Provide all required arguments:
```bash
python main.py \
    --experiment_name my_exp \
    --config_path configs/config.yaml \
    --data_path /path/to/data \
    --task train
```

### Issue: Out of memory during training
**Solution**: Reduce batch size in config file:
```yaml
batch_size: 8  # or even 4
```

### Issue: Learning rate not decreasing
**Solution**: Check scheduler patience:
```bash
--scheduler_patience 5  # Reduce for faster LR adaptation
```

### Issue: Checkpoints taking too much space
**Solution**: Don't save all checkpoints:
```bash
--save_every 20  # Less frequent saves
# Don't use --save_all_checkpoints
```

### Issue: Training too slow
**Solution**: 
1. Check GPU usage: `nvidia-smi` (Linux) or Activity Monitor (Mac)
2. Reduce model size in config
3. Enable mixed precision (add to code)

## Integration with TFLite Conversion

After training with main.py, convert to TFLite:

```bash
# 1. Train
python main.py --experiment_name my_exp --config_path configs/config.yaml --data_path /data --task train

# 2. Identify best checkpoint
ls -lh checkpoints_my_exp/*best_val*

# 3. Convert to TFLite
python convert_to_tflite.py \
    --config configs/config.yaml \
    --checkpoint checkpoints_my_exp/checkpoint_100_best_val.h5 \
    --quantization float16 \
    --output models/signbart_mobile.tflite \
    --test
```

## Summary

`main.py` is the **recommended training script** for:
- Production experiments
- Research work
- Reproducible results
- Detailed analysis
- Model comparison

It provides the same experience as the PyTorch version while leveraging TensorFlow's advantages for mobile deployment.

---

**Quick Start**: Copy a command from the examples above, adjust paths, and run!

