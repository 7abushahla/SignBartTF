# SignBART TensorFlow - Quick Start Guide

Get up and running with SignBART TensorFlow in 5 minutes!

## 🚀 Installation (2 minutes)

```bash
# 1. Create environment
conda create -n signbart_tf python=3.9 -y
conda activate signbart_tf

# 2. Install TensorFlow
pip install tensorflow>=2.12.0

# 3. Install dependencies
cd ml_training/models/signbart_tf
pip install -r requirements.txt
```

## 📝 Configuration (1 minute)

Edit `configs/example_config.yaml` or use an existing one:

```yaml
data_root: "/path/to/your/dataset"
keypoint_config: "full_100"
num_labels: 64
batch_size: 16
epochs: 100
learning_rate: 0.0001
```

## 🎯 Training (1 command)

```bash
python train.py --config configs/your_config.yaml
```

To resume from a checkpoint:
```bash
python train.py --config configs/your_config.yaml --resume
```

## 📱 Convert to TFLite (1 command)

```bash
python convert_to_tflite.py \
    --config configs/your_config.yaml \
    --checkpoint checkpoints/checkpoint_50_best_val.h5 \
    --quantization float16 \
    --output signbart_mobile.tflite
```

## ✅ Verify Everything Works

```bash
# Test individual components
python layers.py      # Should print "✓ All layer tests passed!"
python attention.py   # Should print "✓ All attention tests passed!"
python encoder.py     # Should print "✓ All encoder tests passed!"
python decoder.py     # Should print "✓ All decoder tests passed!"
python model.py       # Should print "✓ All model tests passed!"
python utils.py       # Should print "✓ All utility tests passed!"
```

## 📊 Monitor Training

Training logs are saved to `logs/` directory. View with TensorBoard:

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

## 🎓 What's Next?

1. **Read the full README**: `README.md` for detailed documentation
2. **Migration from PyTorch**: `MIGRATION_GUIDE.md` if you're coming from PyTorch
3. **Detailed setup**: `SETUP.md` for advanced installation options
4. **Experiment**: Try different model configurations and quantization methods

## 🆘 Troubleshooting Quick Fixes

### Issue: Module not found
```bash
pip install tensorflow numpy pyyaml tqdm
```

### Issue: Out of memory
Reduce `batch_size` in config:
```yaml
batch_size: 8  # or even 4
```

### Issue: Slow training
Enable mixed precision (edit train.py):
```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

### Issue: TFLite conversion fails
Try simpler quantization:
```bash
python convert_to_tflite.py ... --quantization dynamic
```

## 📁 Important Files

| File | Purpose |
|------|---------|
| `train.py` | Main training script |
| `convert_to_tflite.py` | Convert trained model to TFLite |
| `model.py` | SignBART model definition |
| `configs/*.yaml` | Configuration files |
| `README.md` | Complete documentation |
| `MIGRATION_GUIDE.md` | PyTorch to TensorFlow guide |

## 💡 Pro Tips

1. **Start small**: Use fewer epochs first to verify everything works
2. **Save often**: Set `save_every: 5` in config
3. **Watch metrics**: Monitor both train and validation accuracy
4. **Quantize wisely**: float16 is best balance of size and accuracy
5. **Test TFLite**: Always test the converted model before deployment

## 🎉 Success Checklist

- [ ] TensorFlow installed and working
- [ ] Dataset prepared and accessible
- [ ] Config file created/edited
- [ ] Training runs without errors
- [ ] Model checkpoints are saved
- [ ] TFLite conversion succeeds
- [ ] TFLite model tested and works
- [ ] Ready for mobile deployment!

## 🔗 Quick Links

- Full Documentation: `README.md`
- Setup Guide: `SETUP.md`
- Migration Guide: `MIGRATION_GUIDE.md`
- Example Config: `configs/example_config.yaml`
- PyTorch Version: `../signbart/`

---

**Need help?** Check the detailed guides or review the code comments!

