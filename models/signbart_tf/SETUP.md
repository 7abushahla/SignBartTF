# Setup Guide for SignBART TensorFlow

## Quick Start

### 1. Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n signbart_tf python=3.9
conda activate signbart_tf

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install TensorFlow

#### For Linux/Windows with NVIDIA GPU:
```bash
pip install tensorflow[and-cuda]>=2.12.0
```

#### For macOS (Intel):
```bash
pip install tensorflow>=2.12.0
```

#### For macOS (Apple Silicon):
```bash
pip install tensorflow-macos>=2.12.0
pip install tensorflow-metal>=0.8.0
```

### 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"
```

Expected output:
```
TensorFlow version: 2.12.0 (or higher)
GPU available: [List of GPUs] or []
```

## Testing the Installation

### Test Individual Components

```bash
# Test layers
python layers.py

# Test attention mechanisms
python attention.py

# Test encoder
python encoder.py

# Test decoder
python decoder.py

# Test full model
python model.py

# Test utilities
python utils.py
```

If all tests pass, you should see output like:
```
✓ All [component] tests passed!
```

## Troubleshooting

### Issue: "No module named 'tensorflow'"
**Solution**: Install TensorFlow using the commands above

### Issue: ImportError with tensorflow-metal
**Solution**: Make sure you're on macOS with Apple Silicon and have installed tensorflow-macos first

### Issue: CUDA errors on Linux
**Solution**: 
```bash
# Check CUDA version
nvidia-smi

# Install appropriate TensorFlow version
pip install tensorflow[and-cuda]==2.12.0  # Match your CUDA version
```

### Issue: Out of memory errors
**Solution**: Add this to your code:
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

## Next Steps

After successful installation:

1. **Prepare your dataset** in the format expected by `dataset.py`
2. **Create a config file** (see `configs/` for examples)
3. **Start training**: `python train.py --config your_config.yaml`
4. **Convert to TFLite**: `python convert_to_tflite.py --config your_config.yaml --checkpoint your_checkpoint.h5`

## Performance Tips

### For Training:
- Use mixed precision for faster training:
  ```python
  from tensorflow.keras import mixed_precision
  mixed_precision.set_global_policy('mixed_float16')
  ```

### For Inference:
- Use TFLite with quantization (see README.md)
- Enable XLA compilation:
  ```python
  @tf.function(jit_compile=True)
  def model_call(...):
      ...
  ```

## Environment Variables

For better performance, set these before running:

```bash
# Use all CPU cores
export TF_NUM_INTRAOP_THREADS=0
export TF_NUM_INTEROP_THREADS=0

# Enable XLA (experimental)
export TF_XLA_FLAGS=--tf_xla_enable_xla_devices

# For debugging
export TF_CPP_MIN_LOG_LEVEL=0  # 0=all, 1=info, 2=warning, 3=error
```

## Additional Resources

- [TensorFlow Installation Guide](https://www.tensorflow.org/install)
- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization)

