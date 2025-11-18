# SignBART TensorFlow/Keras Implementation

This directory contains a complete TensorFlow/Keras implementation of SignBART, converted from the original PyTorch version. The primary motivation for this conversion is to enable **clean quantization and TFLite conversion** for mobile deployment.

## 📁 Directory Structure

```
signbart_tf/
├── attention.py              # Attention mechanisms (Self, Cross, Causal)
├── augmentations.py          # Data augmentations (copied from PyTorch, pure NumPy)
├── configs/                  # Configuration files
├── convert_to_tflite.py      # TFLite conversion with quantization support
├── dataset.py                # TensorFlow dataset implementation
├── decoder.py                # Decoder layers and architecture
├── encoder.py                # Encoder layers and architecture
├── layers.py                 # Core layers (Positional, Projection, etc.)
├── main.py                   # Main training script (full-featured, like PyTorch)
├── model.py                  # Main SignBART model
├── train.py                  # Training script (simplified version)
├── train_loso.py             # LOSO cross-validation training script
├── utils.py                  # Utility functions
└── README.md                 # This file
```

## 🚀 Key Features

### ✅ Fully Functional TensorFlow Implementation
- All PyTorch components converted to TensorFlow/Keras
- Maintains same architecture and behavior as original
- Uses native TF operations for best performance

### ✅ Multiple Quantization Options
- **No quantization**: Full float32 model
- **Dynamic quantization**: Weights int8, activations float32
- **Float16 quantization**: All operations in float16
- **Int8 quantization**: Hybrid int8 (weights and activations)
- **Full int8 quantization**: All int8 for edge TPU/embedded devices

### ✅ Clean TFLite Conversion
- Direct conversion from Keras model
- No ONNX intermediary needed
- Supports quantization-aware training path (if needed)

### ✅ Efficient Data Pipeline
- Uses `tf.data.Dataset` for optimal performance
- Automatic batching with padding
- Support for data augmentation

## 🛠️ Installation

```bash
# Create conda environment (or use existing)
conda create -n signbart_tf python=3.9
conda activate signbart_tf

# Install dependencies
pip install tensorflow>=2.12.0
pip install numpy pyyaml tqdm
```

## 📖 Usage

### 1. Training

#### Option A: Using main.py (Recommended - full-featured)

```bash
python main.py \
    --experiment_name my_experiment \
    --config_path configs/your_config.yaml \
    --data_path /path/to/dataset \
    --task train \
    --epochs 200 \
    --lr 2e-5
```

Features:
- Learning rate scheduling (ReduceLROnPlateau)
- Best checkpoint tracking (train and validation)
- Training curve plotting
- Detailed logging
- Evaluation mode
- Resume from checkpoint

To resume training:
```bash
python main.py \
    --experiment_name my_experiment \
    --config_path configs/your_config.yaml \
    --data_path /path/to/dataset \
    --task train \
    --resume_checkpoints checkpoints_my_experiment/checkpoint_50_latest.h5
```

To evaluate:
```bash
python main.py \
    --experiment_name my_experiment \
    --config_path configs/your_config.yaml \
    --data_path /path/to/dataset \
    --task eval
```

#### Option B: Using train.py (Simplified)

```bash
python train.py --config configs/your_config.yaml
```

To resume from checkpoint:
```bash
python train.py --config configs/your_config.yaml --resume
```

#### Option C: LOSO Cross-Validation (Recommended for Research)

For Leave-One-Subject-Out cross-validation:

```bash
# Test with single user first
python train_loso.py \
    --config_path configs/your_config.yaml \
    --base_data_path /path/to/dataset \
    --holdout_only user01 \
    --epochs 80 \
    --lr 2e-4

# Then run all LOSO experiments
python train_loso.py \
    --config_path configs/your_config.yaml \
    --base_data_path /path/to/dataset \
    --epochs 80 \
    --lr 2e-4
```

See `LOSO_GUIDE.md` for detailed instructions.

### 2. Converting to TFLite

#### Basic Conversion (Dynamic Quantization)
```bash
python convert_to_tflite.py \
    --config configs/your_config.yaml \
    --checkpoint checkpoints/checkpoint_50_best_val.h5 \
    --quantization dynamic \
    --output models/signbart_dynamic.tflite
```

#### Float16 Quantization (Recommended for mobile)
```bash
python convert_to_tflite.py \
    --config configs/your_config.yaml \
    --checkpoint checkpoints/checkpoint_50_best_val.h5 \
    --quantization float16 \
    --output models/signbart_float16.tflite
```

#### Full Int8 Quantization (Maximum compression)
```bash
python convert_to_tflite.py \
    --config configs/your_config.yaml \
    --checkpoint checkpoints/checkpoint_50_best_val.h5 \
    --quantization int8 \
    --num-calibration-samples 200 \
    --output models/signbart_int8.tflite
```

#### Test the Converted Model
```bash
python convert_to_tflite.py \
    --config configs/your_config.yaml \
    --checkpoint checkpoints/checkpoint_50_best_val.h5 \
    --quantization dynamic \
    --test
```

### 3. Configuration

Example config file (`configs/example.yaml`):

```yaml
# Data
data_root: "/path/to/dataset"
keypoint_config: "full_100"  # Options: hands_only, pose_hands, full_75, full_100

# Model architecture
d_model: 256
encoder_layers: 6
encoder_attention_heads: 8
encoder_ffn_dim: 1024
decoder_layers: 6
decoder_attention_heads: 8
decoder_ffn_dim: 1024
max_position_embeddings: 512
num_labels: 64  # Number of sign classes
classifier_dropout: 0.1

# Regularization
dropout: 0.1
attention_dropout: 0.1
activation_dropout: 0.1
encoder_layerdrop: 0.0
decoder_layerdrop: 0.0

# Training
batch_size: 16
epochs: 100
learning_rate: 0.0001
optimizer: "adamw"
weight_decay: 0.01
augment: true

# Paths
checkpoint_dir: "checkpoints"
log_dir: "logs"
save_every: 5
```

## 🔄 Differences from PyTorch Version

### Architecture
- **Same**: Model architecture is identical
- **Same**: Attention mechanisms work the same way
- **Different**: Uses Keras `Layer` instead of `nn.Module`
- **Different**: Uses `tf.function` for graph optimization

### Data Loading
- **Different**: Uses `tf.data.Dataset` instead of `torch.utils.data.Dataset`
- **Same**: Augmentations are identical (pure NumPy)
- **Different**: Automatic batching and padding in TF data pipeline

### Training
- **Different**: Uses `tf.GradientTape` for gradient computation
- **Same**: Training loop logic is the same
- **Different**: Checkpoint format (.h5 instead of .pth)

### Performance
- **Similar**: Should have comparable training speed
- **Better**: TFLite inference is optimized for mobile
- **Better**: Quantization is cleaner and more reliable

## 📊 Model Size Comparison

| Quantization | Model Size | Inference Speed | Accuracy Loss |
|--------------|------------|-----------------|---------------|
| None (float32) | ~100 MB | Baseline | 0% |
| Dynamic (int8) | ~25 MB | 2-3x faster | <1% |
| Float16 | ~50 MB | 1.5-2x faster | <0.5% |
| Int8 | ~25 MB | 3-4x faster | 1-2% |
| Int8 Full | ~25 MB | 4x+ faster | 2-3% |

*Sizes and speeds are approximate and depend on model architecture*

## 🧪 Testing

Test individual components:

```bash
# Test layers
python layers.py

# Test attention
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

## 🐛 Troubleshooting

### Issue: "Out of memory" during conversion
**Solution**: Reduce `num_calibration_samples` or use a smaller model

### Issue: "Conversion failed" error
**Solution**: Try with `experimental_new_converter=True` (already in code)

### Issue: Large difference between TF and TFLite outputs
**Solution**: This is normal for int8 quantization. Use float16 for better accuracy.

### Issue: Model not loading weights
**Solution**: Make sure model is built before loading weights (call model once)

## 📝 Citation

If you use this TensorFlow implementation, please cite the original SignBART paper and acknowledge this conversion.

## 🤝 Contributing

This is a direct conversion from PyTorch to TensorFlow. If you find bugs or want to add features:
1. Test thoroughly
2. Ensure consistency with PyTorch version
3. Update documentation

## 📄 License

Same license as the original SignBART implementation.

## 🔗 Related Files

- Original PyTorch implementation: `../signbart/`
- LSTM TensorFlow implementation: `../lstm/`
- Mobile app integration: `../../../mobile_app/`

## ✨ Advantages of This Implementation

1. **Clean quantization**: No need for ONNX or ai_edge_torch
2. **Better mobile support**: Native TFLite is optimized for mobile
3. **Easier debugging**: TensorFlow's eager execution
4. **Better ecosystem**: Integrates with TensorFlow Lite Model Maker
5. **Production ready**: TFLite is battle-tested on mobile devices

---

**Note**: This implementation is functionally equivalent to the PyTorch version but optimized for deployment. For research and experimentation, you may prefer the original PyTorch version.

