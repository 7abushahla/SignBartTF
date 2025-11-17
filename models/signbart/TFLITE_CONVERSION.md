# SignBart TFLite Conversion Guide

This guide shows how to convert your trained SignBart PyTorch model to TFLite for mobile deployment.

## Prerequisites

```bash
pip install ai-edge-torch
pip install tensorflow  # For testing TFLite model
```

## Quick Start

### Basic Conversion

```bash
python convert_to_tflite.py \
    --config configs/arabic-asl.yaml \
    --checkpoint checkpoints/best_model.pth \
    --output signbart.tflite
```

### With Verification and Testing

```bash
python convert_to_tflite.py \
    --config configs/arabic-asl.yaml \
    --checkpoint checkpoints/best_model.pth \
    --output signbart.tflite \
    --seq_len 64 \
    --test
```

## Equivalent to ResNet Example

Your question referenced this ResNet example:

```python
# ResNet example
resnet18 = torchvision.models.resnet18(weights.IMAGENET1K_V1).eval()
sample_inputs = (torch.randn(1, 3, 224, 224),)
```

**For SignBart, the equivalent is:**

```python
# SignBart equivalent
from model import SignBart
import yaml

# 1. Load config
with open("configs/arabic-asl.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Create model
signbart = SignBart(config)
signbart.eval()

# 3. Load trained weights
state_dict = torch.load("checkpoints/best_model.pth")
signbart.load_state_dict(state_dict)

# 4. Create sample inputs
sample_inputs = (
    torch.randn(1, 64, 90, 2),      # keypoints: (batch, seq_len, 90, 2)
    torch.ones(1, 64, dtype=torch.long)  # attention_mask: (batch, seq_len)
)

# 5. Convert to TFLite
edge_model = ai_edge_torch.convert(signbart.eval(), sample_inputs)
edge_model.export('signbart.tflite')
```

## Conversion Process

The script follows these steps:

### 1. Load Model
```python
model, config = load_signbart_model(
    config_path="configs/arabic-asl.yaml",
    checkpoint_path="checkpoints/best_model.pth"
)
```

### 2. Create Sample Inputs
```python
keypoints = torch.randn(1, 64, 90, 2)      # (batch, frames, keypoints, coords)
attention_mask = torch.ones(1, 64)          # (batch, frames)
sample_inputs = (keypoints, attention_mask)
```

### 3. Get PyTorch Output (for verification)
```python
with torch.no_grad():
    loss, logits = model(*sample_inputs)
    torch_output = logits  # Shape: (batch, num_classes)
```

### 4. Convert to TFLite
```python
edge_model = ai_edge_torch.convert(model.eval(), sample_inputs)
```

### 5. Verify Accuracy
```python
edge_output = edge_model(*sample_inputs)

if np.allclose(torch_output.numpy(), edge_output, atol=1e-5, rtol=1e-5):
    print("✅ Conversion successful!")
```

### 6. Export
```python
edge_model.export('signbart.tflite')
```

## Input Format

SignBart expects two inputs:

| Input | Shape | Type | Description |
|-------|-------|------|-------------|
| **keypoints** | `(batch, seq_len, 90, 2)` | `float32` | 90 keypoints × 2 coords (x, y) |
| **attention_mask** | `(batch, seq_len)` | `int64` | 1 for valid frames, 0 for padding |

**Example:**
- `seq_len=64`: 64 frames per video
- `90`: Number of keypoints per frame
- `2`: x, y coordinates

## Output Format

| Output | Shape | Type | Description |
|--------|-------|------|-------------|
| **logits** | `(batch, num_classes)` | `float32` | Class predictions |

## Testing Converted Model

The script can test the TFLite model:

```bash
python convert_to_tflite.py \
    --config configs/arabic-asl.yaml \
    --checkpoint checkpoints/best_model.pth \
    --output signbart.tflite \
    --test
```

This will:
1. Load the TFLite model
2. Run inference with random inputs
3. Print predicted class and confidence

## Using TFLite Model in Your App

### Python (for testing)

```python
import tensorflow as tf
import numpy as np

# Load model
interpreter = tf.lite.Interpreter(model_path="signbart.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare inputs
keypoints = np.random.randn(1, 64, 90, 2).astype(np.float32)
attention_mask = np.ones((1, 64), dtype=np.int32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], keypoints)
interpreter.set_tensor(input_details[1]['index'], attention_mask)
interpreter.invoke()

# Get output
output = interpreter.get_tensor(output_details[0]['index'])
predicted_class = output.argmax()
```

### iOS/Swift

```swift
import TensorFlowLite

// Load model
let interpreter = try Interpreter(modelPath: "signbart.tflite")
try interpreter.allocateTensors()

// Prepare inputs
var keypoints: [Float] = ... // Shape: [1, 64, 90, 2]
var attentionMask: [Int32] = ... // Shape: [1, 64]

// Run inference
try interpreter.copy(Data(bytes: &keypoints, count: keypoints.count * 4), toInputAt: 0)
try interpreter.copy(Data(bytes: &attentionMask, count: attentionMask.count * 4), toInputAt: 1)
try interpreter.invoke()

// Get output
let outputTensor = try interpreter.output(at: 0)
let results: [Float] = [Float](unsafeData: outputTensor.data) ?? []
```

## Troubleshooting

### Issue: Conversion Fails

**Solution:**
1. Ensure `ai-edge-torch` is installed:
   ```bash
   pip install ai-edge-torch
   ```

2. Check if all operations are supported:
   - Some PyTorch operations may not be supported
   - Try simplifying the model

3. Use `torch.jit.script` if needed:
   ```python
   scripted_model = torch.jit.script(model)
   edge_model = ai_edge_torch.convert(scripted_model, sample_inputs)
   ```

### Issue: Output Mismatch

**Solution:**
- Small differences (< 1e-5) are normal due to floating-point precision
- Large differences may indicate:
  - Wrong input format
  - Model not in eval mode
  - Incorrect checkpoint loaded

### Issue: Large Model Size

**Solution:**
Apply quantization to reduce size:

```python
# This will be added in a future update
# Currently, ai-edge-torch handles optimization automatically
```

## Model Size

Typical SignBart model sizes:
- **PyTorch checkpoint**: ~50-100 MB
- **TFLite model**: ~30-70 MB (automatically optimized)
- **With quantization**: ~10-20 MB (future feature)

## Next Steps

After converting to TFLite:
1. Test model accuracy on validation set
2. Integrate into mobile app
3. Profile inference speed on target device
4. Consider quantization if size/speed is critical

## References

- ai-edge-torch: https://github.com/google-ai-edge/ai-edge-torch
- TFLite documentation: https://www.tensorflow.org/lite
- PyTorch Mobile: https://pytorch.org/mobile/

