# Migration Guide: PyTorch SignBART → TensorFlow SignBART

This guide helps you migrate from the PyTorch implementation to the TensorFlow implementation.

## 📋 Quick Comparison

| Aspect | PyTorch Version | TensorFlow Version |
|--------|----------------|-------------------|
| **Location** | `models/signbart/` | `models/signbart_tf/` |
| **Model Class** | `nn.Module` | `keras.Model` |
| **Dataset** | `torch.utils.data.Dataset` | `tf.data.Dataset` |
| **Training Loop** | Manual loop with `.backward()` | `tf.GradientTape` or `.fit()` |
| **Checkpoints** | `.pth` files | `.h5` files |
| **Deployment** | ONNX → ai_edge_torch | Direct TFLite |

## 🔄 Step-by-Step Migration

### Step 1: Install TensorFlow

```bash
# Remove PyTorch environment (optional)
conda deactivate

# Create new TensorFlow environment
conda create -n signbart_tf python=3.9
conda activate signbart_tf

# Install TensorFlow (see SETUP.md for details)
pip install tensorflow>=2.12.0
pip install -r requirements.txt
```

### Step 2: Understand Directory Structure

```
PyTorch (models/signbart/)          TensorFlow (models/signbart_tf/)
├── model.py                        ├── model.py           (converted)
├── encoder.py                      ├── encoder.py         (converted)
├── decoder.py                      ├── decoder.py         (converted)
├── attention.py                    ├── attention.py       (converted)
├── layers.py                       ├── layers.py          (converted)
├── dataset.py                      ├── dataset.py         (converted)
├── utils.py                        ├── utils.py           (converted)
├── train_loso.py                   ├── train.py           (simplified)
└── augmentations.py                └── augmentations.py   (same!)
```

### Step 3: Convert Your Config File

Your existing config files should work with minimal changes. Update paths if needed:

```yaml
# Old (PyTorch)
data_root: "/path/to/dataset"
checkpoint_dir: "signbart/checkpoints"

# New (TensorFlow)
data_root: "/path/to/dataset"  # Same
checkpoint_dir: "signbart_tf/checkpoints"  # Update path
```

### Step 4: Training

#### PyTorch Training:
```bash
cd models/signbart
python train_loso.py --config configs/your_config.yaml
```

#### TensorFlow Training:
```bash
cd models/signbart_tf
python train.py --config configs/your_config.yaml
```

### Step 5: Convert Existing PyTorch Checkpoints (Optional)

If you have trained PyTorch models and want to use them in TensorFlow:

```python
import torch
import tensorflow as tf
import numpy as np

# Load PyTorch checkpoint
pt_checkpoint = torch.load('pytorch_model.pth')

# Create TensorFlow model
from model import SignBart
tf_model = SignBart(config)

# Build model
dummy_input = tf.random.normal((1, 10, 100, 2))
dummy_mask = tf.ones((1, 10))
_ = tf_model(dummy_input, dummy_mask)

# Map and transfer weights (you'll need to write custom mapping)
# This is non-trivial due to different naming conventions
# Easier to retrain from scratch
```

**Recommendation**: It's usually easier to retrain the TensorFlow model from scratch rather than converting weights.

### Step 6: Inference

#### PyTorch Inference:
```python
import torch
from model import SignBart

model = SignBart(config)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

with torch.no_grad():
    outputs = model(keypoints, mask)
```

#### TensorFlow Inference:
```python
import tensorflow as tf
from model import SignBart

model = SignBart(config)
model.load_weights('checkpoint.h5')

outputs = model(keypoints, mask, training=False)
```

### Step 7: Convert to TFLite (The Main Advantage!)

This is where TensorFlow really shines:

```bash
# PyTorch: Complex, often breaks
python convert_to_onnx.py ...
# Then use ai_edge_torch (experimental, often fails)

# TensorFlow: Clean and simple!
python convert_to_tflite.py \
    --config configs/your_config.yaml \
    --checkpoint checkpoints/best.h5 \
    --quantization float16 \
    --output models/signbart.tflite
```

## 🔍 Key Differences in Code

### Model Definition

#### PyTorch:
```python
class SignBart(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        
    def forward(self, keypoints, mask, labels=None):
        x = self.encoder(keypoints, mask)
        return x
```

#### TensorFlow:
```python
class SignBart(keras.Model):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        
    def call(self, keypoints, mask, labels=None, training=None):
        x = self.encoder(keypoints, mask, training=training)
        return x
```

### Training Loop

#### PyTorch:
```python
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(**batch)
    loss.backward()
    optimizer.step()
```

#### TensorFlow:
```python
for batch in dataset:
    with tf.GradientTape() as tape:
        loss = model(**batch, training=True)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

### Attention Masks

#### PyTorch:
```python
# Additive mask with torch.finfo
mask = mask.masked_fill(mask.bool(), torch.finfo(dtype).min)
```

#### TensorFlow:
```python
# Additive mask with tf.where
mask = tf.where(tf.cast(mask, tf.bool), -1e9, 0.0)
```

## ⚠️ Common Pitfalls

### 1. Training Mode
- **PyTorch**: `model.train()` and `model.eval()`
- **TensorFlow**: `training=True/False` parameter

### 2. Dimension Order
- Both use same order for sequences: `(batch, seq_len, features)`
- No major differences here!

### 3. Gradient Computation
- **PyTorch**: Automatic with `loss.backward()`
- **TensorFlow**: Manual with `tf.GradientTape()`

### 4. Device Management
- **PyTorch**: Explicit `.to(device)` or `.cuda()`
- **TensorFlow**: Automatic device placement

### 5. Model Saving
- **PyTorch**: `torch.save(model.state_dict(), 'model.pth')`
- **TensorFlow**: `model.save_weights('model.h5')`

## 📊 Performance Comparison

| Metric | PyTorch | TensorFlow |
|--------|---------|------------|
| Training Speed | ~Baseline | ~Similar (±10%) |
| Memory Usage | ~Baseline | ~Similar |
| Model Size (float32) | 100 MB | 100 MB |
| **TFLite (quantized)** | **Difficult** | **25 MB (easy!)** |
| Mobile Inference | Needs conversion | Native support |
| Debugging | Good | Good (eager mode) |

## ✅ Why Migrate?

1. **Clean TFLite Conversion**: No ONNX headaches
2. **Better Mobile Support**: TFLite is production-ready
3. **Easy Quantization**: Built-in quantization tools
4. **Stable Deployment**: TensorFlow has better mobile ecosystem
5. **Same Accuracy**: Model architecture is identical

## 🚀 Next Steps After Migration

1. **Retrain** your model with TensorFlow
2. **Validate** that accuracy matches PyTorch version
3. **Convert** to TFLite with quantization
4. **Integrate** into your mobile app
5. **Deploy** with confidence!

## 🔗 Additional Resources

- [TensorFlow Migration Guide](https://www.tensorflow.org/guide/migrate)
- [PyTorch to TensorFlow Cheat Sheet](https://www.tensorflow.org/guide/migrate/model_mapping)
- [TFLite Converter Guide](https://www.tensorflow.org/lite/convert)

## 🆘 Getting Help

If you encounter issues during migration:

1. Check `SETUP.md` for installation issues
2. Check `README.md` for usage examples
3. Compare with PyTorch version to verify behavior
4. Test components individually (run `python model.py`, etc.)

---

**Remember**: The TensorFlow version is a complete reimplementation that maintains the same architecture and behavior as the PyTorch version, but optimized for deployment!

