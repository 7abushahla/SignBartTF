# e-LSTM Model for Sign Language Recognition

Implementation of an **enhanced LSTM (e-LSTM)** model with Bahdanau-style additive attention for Arabic Sign Language recognition using 2D keypoints.

## Architecture

### Original Paper Design
```
Input: 75 keypoints × 3 (x,y,z) = 225 features
  ↓
LSTM(1024) - return sequences
  ↓
LSTM(512) - return sequences
  ↓
Additive Attention (64 hidden units)
  ↓
Dense(64, ReLU)
  ↓
Softmax(num_classes)
```

### Our Adaptation
```
Input: 90 keypoints × 2 (x,y) = 180 features
  ↓
Flatten → (seq_len, 180)
  ↓
LSTM(1024) - return sequences
  ↓
LSTM(512) - return sequences
  ↓
Bahdanau Attention (64 hidden units)
  ↓
Dropout(0.3)
  ↓
Dense(64, ReLU)
  ↓
Dropout(0.3)
  ↓
Softmax(num_classes)
```

## Key Differences from Original Paper

| Aspect | Original Paper | Our Implementation |
|--------|---------------|-------------------|
| **Keypoints** | 75 | 90 |
| **Coordinates** | 3D (x, y, z) | 2D (x, y) |
| **Input Features** | 225 | 180 |
| **Keypoint Source** | (Not specified) | MediaPipe Holistic |
| **LSTM Units** | 1024 → 512 | ✅ Same |
| **Attention** | Additive (64 units) | ✅ Same (Bahdanau) |
| **FC Layer** | Dense(64, ReLU) | ✅ Same |
| **Dropout** | (Not specified) | 0.3 (added for regularization) |

## Attention Mechanism

The model uses **Bahdanau-style additive attention** with 64 hidden units. This attention mechanism:
- Concentrates on particular time steps with the most discriminative information
- Computes attention weights over all LSTM hidden states
- Creates a context vector as a weighted sum of hidden states
- Formula: `score = V * tanh(W1*values + W2*query)`

### Implementation
We provide **two attention implementations**:

1. **Custom BahdanauAttention** (Recommended) ✅
   - Faithful to the paper's design
   - Explicit 64-unit hidden dimension
   - Located in `model.py`

2. **tf.keras.layers.AdditiveAttention** (Alternative)
   - Uses Keras built-in attention
   - Simpler but may not exactly match paper's 64-unit hidden layer
   - Provided for comparison

## Files

```
lstm/
├── model.py           # e-LSTM architecture with attention
├── train.py           # Training script
├── config.yaml        # Hyperparameters and settings
├── checkpoints/       # Saved model weights
├── logs/             # TensorBoard logs
└── results/          # Training history, predictions
```

## Training

### Hyperparameters (from paper)

| Parameter | Value |
|-----------|-------|
| LSTM 1 Units | 1024 |
| LSTM 2 Units | 512 |
| Attention Hidden Units | 64 |
| FC Units | 64 |
| Batch Size | 16 |
| Epochs | 70 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |

### Usage

```bash
# Train model
python train.py \
    --data_dir ../../data/processed \
    --config config.yaml \
    --output_dir .

# Training will save:
# - checkpoints/elstm_TIMESTAMP_epochXX_valACCURACY.h5
# - logs/TIMESTAMP/ (TensorBoard logs)
# - results/training_history_TIMESTAMP.pkl
```

### Data Format

The training script expects `.pkl` files with the following structure:

```python
{
    'keypoints': np.array(shape=(num_frames, 90, 2)),  # 2D keypoints
    'label': int  # Class label
}
```

**Keypoint structure (90 × 2 = 180 features):**
- **Pose (23)**: Upper body landmarks (nose through thumbs)
- **Left Hand (21)**: MediaPipe hand landmarks
- **Right Hand (21)**: MediaPipe hand landmarks
- **Face (25)**: Symmetric face landmarks (10 contour + eyes + nose)

### Preprocessing

Keypoints are:
1. **Padded or trimmed** to fixed `seq_len` (default: 64 frames)
2. **Grouped Normalization**: Each body part (pose, left hand, right hand, face) is normalized independently by its own bounding box
   - Preserves spatial relationships within each body part
   - Matches SignBart's normalization strategy
   - Prevents one body part from dominating the normalization
3. Fed as shape `(batch, seq_len, 90, 2)` → flattened to `(batch, seq_len, 180)`

### Grouped Normalization Details

For each group (pose, left_hand, right_hand, face):
- Compute bounding box from valid keypoints
- Add 5% padding + centering
- Normalize to [0, 1] within that bounding box

This ensures hands, face, and pose maintain their relative spatial structure independently.

## Model Summary

```
Input: (batch, 64, 90, 2)
  ↓ Reshape
(batch, 64, 180)
  ↓ LSTM(1024)
(batch, 64, 1024)
  ↓ LSTM(512)
(batch, 64, 512)
  ↓ Attention
(batch, 512)
  ↓ Dense(64)
(batch, 64)
  ↓ Softmax
(batch, num_classes)
```

**Total Parameters:** ~7-8M (depending on num_classes)

## Evaluation Metrics

- **Accuracy**: Top-1 classification accuracy
- **Top-5 Accuracy**: Top-5 classification accuracy
- **Loss**: Categorical cross-entropy

## Callbacks

- **ModelCheckpoint**: Save best model based on validation accuracy
- **TensorBoard**: Log training metrics
- **EarlyStopping**: Stop if val_loss doesn't improve for 15 epochs
- **ReduceLROnPlateau**: Reduce LR by 0.5 if val_loss plateaus for 5 epochs
- **CSVLogger**: Save training history to CSV

## TensorBoard

View training progress:

```bash
tensorboard --logdir logs/
```

## Model Serialization

The `BahdanauAttention` layer is decorated with `@tf.keras.utils.register_keras_serializable()`, which enables:

✅ **Easy saving/loading** (no `custom_objects` needed)
✅ **TFLite conversion** for mobile deployment
✅ **Model portability**

### Loading a Saved Model

```python
from tensorflow import keras

# Load without custom_objects (thanks to @register_keras_serializable)
model = keras.models.load_model('checkpoints/elstm_final_TIMESTAMP.h5')
```

### TFLite Conversion

```bash
# Convert to TFLite
python convert_to_tflite.py \
    --model checkpoints/elstm_final_TIMESTAMP.h5 \
    --output elstm_model.tflite \
    --quantize  # Optional: smaller model size

# Test converted model
python convert_to_tflite.py \
    --model checkpoints/elstm_final_TIMESTAMP.h5 \
    --output elstm_model.tflite \
    --test
```

## Testing

```python
from tensorflow import keras
import numpy as np

# Load trained model
model = keras.models.load_model('checkpoints/elstm_final_TIMESTAMP.h5')

# Predict
keypoints = np.load('test_sample.npy')  # Shape: (seq_len, 90, 2)
keypoints = np.expand_dims(keypoints, axis=0)  # Add batch dimension
prediction = model.predict(keypoints)
predicted_class = np.argmax(prediction)
```

## Comparison with Other Models

| Model | Input | Parameters | Architecture |
|-------|-------|------------|--------------|
| **e-LSTM** | Keypoints (180 features) | ~7-8M | LSTM + Attention |
| **SignBart** | Keypoints (180 features) | ~XX M | Transformer (BART) |
| **CNN-LSTM** | Video frames | ~XX M | CNN (spatial) + LSTM (temporal) |

## References

- Original e-LSTM paper: (Add paper citation)
- MediaPipe Holistic: https://google.github.io/mediapipe/solutions/holistic.html
- Bahdanau Attention: "Neural Machine Translation by Jointly Learning to Align and Translate"

## Notes

- The model expects **fixed-length sequences**. Longer/shorter videos are padded/trimmed.
- 2D keypoints (x, y) are used instead of 3D (x, y, z) from the original paper.
- Dropout layers added for regularization (not in original paper).
- The attention mechanism helps identify which frames contain the most discriminative information for each sign.

