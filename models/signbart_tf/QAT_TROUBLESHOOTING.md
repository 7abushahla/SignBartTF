# QAT Troubleshooting Guide

## Problem: Training Collapse During QAT

Your training logs show a catastrophic collapse pattern:
- Epoch 1-3: 95% accuracy ✓
- Epoch 4: 86% accuracy ⚠️
- Epoch 5: 56% accuracy ✗
- Epoch 6: 11% accuracy ✗✗✗

**This proves QAT IS working** - if it was just printing, accuracy would stay at 95%.

## Root Causes

### 1. Learning Rate Too High
- **Your setting**: `--lr 5e-5`
- **Problem**: QAT introduces fake quantization noise that makes gradients unstable
- **Fix**: Reduce to `--lr 1e-5` or `--lr 5e-6` (5-10x lower)

### 2. Batch Size Too Small  
- **Your setting**: `--batch_size 1`
- **Problem**: Single-sample batches have high variance, amplified by quantization noise
- **Fix**: Use `--batch_size 4` or `--batch_size 8` (if memory allows)

### 3. Scheduler Too Aggressive
- **Your setting**: `--scheduler_patience 5`
- **Problem**: Reduces LR after only 5 epochs of no improvement
- **Fix**: Use `--scheduler_patience 10` or disable scheduler initially

### 4. Not Enough Epochs
- **Your setting**: Default (likely 40 epochs)
- **Problem**: Model needs time to adapt to quantization constraints
- **Fix**: QAT often needs 10-20 epochs to stabilize

## Recommended Solutions (Try in Order)

### Solution 1: Conservative QAT Settings (RECOMMENDED)
```bash
python train_loso_functional_qat_batch.py \
    --config_path configs/arabic-asl-90kpts.yaml \
    --base_data_path ~/signbart_tf/data/arabic-asl-90kpts \
    --holdout_only user01 \
    --qat_epochs 15 \
    --batch_size 4 \
    --lr 1e-5 \
    --scheduler_patience 10 \
    --seed 42
```

**Why this should work:**
- Lower LR (1e-5) reduces gradient instability
- Larger batch size (4) smooths out gradient variance
- More patience (10) prevents premature LR reduction
- More epochs (15) gives model time to adapt

### Solution 2: Disable Scheduler Initially
```bash
python train_loso_functional_qat_batch.py \
    --config_path configs/arabic-asl-90kpts.yaml \
    --base_data_path ~/signbart_tf/data/arabic-asl-90kpts \
    --holdout_only user01 \
    --qat_epochs 10 \
    --batch_size 4 \
    --lr 5e-6 \
    --no_validation \
    --seed 42
```

**Why this helps:**
- `--no_validation` disables scheduler monitoring
- Even lower LR (5e-6) for maximum stability
- Focus on seeing if QAT can work without scheduler interference

### Solution 3: Gradual QAT (Multi-Stage)

**Stage 1: Stabilization** (very low LR, no scheduler)
```bash
python train_loso_functional_qat.py \
    --config_path configs/arabic-asl-90kpts.yaml \
    --data_path ~/signbart_tf/data/arabic-asl-90kpts_LOSO_user01 \
    --checkpoint checkpoints_arabic_asl_LOSO_user01/final_model.h5 \
    --output_dir exports/qat_stage1_user01 \
    --qat_epochs 5 \
    --batch_size 4 \
    --lr 5e-6 \
    --no_validation \
    --seed 42
```

**Stage 2: Fine-tuning** (slightly higher LR, with scheduler)
```bash
python train_loso_functional_qat.py \
    --config_path configs/arabic-asl-90kpts.yaml \
    --data_path ~/signbart_tf/data/arabic-asl-90kpts_LOSO_user01 \
    --checkpoint exports/qat_stage1_user01/qat_model.keras \
    --output_dir exports/qat_stage2_user01 \
    --qat_epochs 10 \
    --batch_size 4 \
    --lr 1e-5 \
    --scheduler_patience 10 \
    --seed 42
```

## Verification Steps

### 1. Monitor Early Epochs Carefully
```
Epoch 1/15 - loss: 0.16, accuracy: 0.95  ← Should stay near initial
Epoch 2/15 - loss: 0.17, accuracy: 0.94  ← Small degradation OK
Epoch 3/15 - loss: 0.18, accuracy: 0.93  ← Should stabilize
Epoch 4/15 - loss: 0.17, accuracy: 0.94  ← Starting to recover
```

**Red flags:**
- Accuracy drops > 5% in single epoch
- Loss increases dramatically
- Early collapse in first 3 epochs

### 2. Check for QuantizeWrapper Layers
```python
# Add this debug code after QAT model is built:
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

qat_layer_count = sum(1 for layer in qat_model.layers if isinstance(layer, QuantizeWrapper))
print(f"QuantizeWrapper layers: {qat_layer_count}")

if qat_layer_count == 0:
    print("ERROR: No QAT layers found!")
else:
    print(f"✓ QAT is active on {qat_layer_count} layers")
```

### 3. Compare Model Sizes
```bash
# Base FP32 model
ls -lh checkpoints_arabic_asl_LOSO_user01/final_model.h5

# QAT model (should be similar size)
ls -lh exports/qat_finetune_user01/qat_model.keras

# TFLite model (should be ~4x smaller)
ls -lh exports/qat_finetune_user01/qat_dynamic_int8.tflite
```

Expected sizes:
- FP32 model: ~20-40 MB
- QAT model: ~20-40 MB (same as FP32)
- TFLite INT8: ~5-10 MB (4x smaller)

## Advanced Debugging

### Check What Layers Are Being Quantized
```bash
# Look for this output during QAT annotation:
grep "QUANTIZED Dense layers" output.log

# Should show only FFN layers (fc1, fc2):
# ✓ QUANTIZED Dense layers (28 total):
#   • encoder/encoder_layers/0/fc1
#   • encoder/encoder_layers/0/fc2
#   • decoder/decoder_layers/0/fc1
#   • decoder/decoder_layers/0/fc2
#   ...

# WARNING: If you see attention projections, that's the problem:
# ⚠️  q_proj, k_proj, v_proj, out_proj should NOT appear!
```

### Verify Attention Layers Are NOT Quantized
```bash
# These should appear in SKIPPED layers, not quantized:
grep -A 50 "QUANTIZED Dense layers" output.log | grep -E "proj"

# If you see q_proj/k_proj/v_proj/out_proj in the quantized list,
# that's causing the collapse!
```

## What NOT to Do

❌ **Don't quantize attention projections**
```bash
# BAD - will cause collapse:
--quantize_dense_names fc1 fc2 q_proj k_proj v_proj out_proj
```

❌ **Don't use high learning rates**
```bash
# BAD - too unstable for QAT:
--lr 1e-4  # or higher
```

❌ **Don't use tiny batches if possible**
```bash
# BAD - too much variance:
--batch_size 1
```

❌ **Don't give up after 3 epochs**
```bash
# BAD - not enough time to adapt:
--qat_epochs 3
```

## Success Criteria

Your QAT training is successful if:

1. **Accuracy stays within 5% of original** (e.g., 95% → 90%)
2. **Model stabilizes after initial epochs** (not continuous collapse)
3. **TFLite model is significantly smaller** (~4x reduction)
4. **TFLite accuracy is close to QAT accuracy** (within 2-3%)

## Next Steps

1. **Start with Solution 1** (recommended settings above)
2. **Monitor first 5 epochs** - if collapse happens, stop and lower LR further
3. **If successful**, proceed to all 3 LOSO models
4. **If still failing**, try Solution 2 (no scheduler) or Solution 3 (gradual QAT)

## Questions to Ask Yourself

1. **Is the checkpoint FP32?**
   - Run: `grep -c "QuantizeWrapper" checkpoints_*/final_model.h5` 
   - Should be 0 (no QAT layers)

2. **Are you using the right checkpoint?**
   - Use `final_model.h5` (original FP32)
   - NOT `qat_model.keras` (already QAT'd)

3. **Do you have enough memory for larger batches?**
   - Try `--batch_size 2` first
   - Then `--batch_size 4` if possible

4. **Are your filters correct?**
   - Default: `["fc1", "fc2"]` ✓
   - Should NOT include attention layers

## References

- TensorFlow Model Optimization: https://www.tensorflow.org/model_optimization
- QAT Best Practices: https://www.tensorflow.org/model_optimization/guide/quantization/training
- Debugging QAT: https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/g3doc/guide/quantization/training_comprehensive_guide.md

