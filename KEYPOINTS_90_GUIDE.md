# 90 Keypoints Configuration Guide

## Overview

The 90-keypoint version removes lower body keypoints (hips, knees, ankles, heels, feet) to focus on upper body, hands, and face - the most relevant features for sign language recognition.

## Keypoint Structure

### Total: 90 keypoints

1. **Upper Body Pose**: 23 keypoints (indices 0-22)
   - Face landmarks: nose, eyes, ears, mouth
   - Upper body: shoulders, elbows, wrists
   - Hand orientation points
   - **Excluded**: hips, knees, ankles, heels, feet

2. **Left Hand**: 21 keypoints (indices 23-43)
   - MediaPipe hand landmarks
   - Wrist + 4 fingers × 5 joints each

3. **Right Hand**: 21 keypoints (indices 44-64)
   - MediaPipe hand landmarks
   - Wrist + 4 fingers × 5 joints each

4. **Face**: 25 keypoints (indices 65-89)
   - Selected symmetric landmarks from MediaPipe Face Mesh
   - Face contour, eyes, nose

## Configuration File

Use: `configs/arabic-asl-90kpts.yaml`

## Training Command

```bash
python train_loso.py \
    --config_path configs/arabic-asl-90kpts.yaml \
    --base_data_path ~/signbart_tf/data/arabic-asl \
    --holdout_only user01 \
    --epochs 30 \
    --lr 2e-4 \
    --seed 42
```

## Benefits

✅ **Reduced complexity**: 10 fewer keypoints (90 vs 100)
✅ **Focused features**: Only upper body relevant to sign language
✅ **Faster training**: Fewer input dimensions
✅ **Better for edge deployment**: Smaller models, less computation

## Automatic Grouping

The code automatically detects keypoint groups for normalization:
- Detects MediaPipe structure: Pose + Hand (21) + Hand (21) + Face (25)
- No need to manually specify group boundaries
- Works with both 90 and 100 keypoint configs

## Model Size

Expected model sizes with 90 keypoints:
- **Keras model**: ~9 MB
- **TFLite FP32**: ~3 MB
- **TFLite INT8** (dynamic-range): ~0.7 MB

## Data Requirements

Your data must be extracted with MediaPipe Holistic and have:
- At least 90 keypoints per frame
- PKL format with 'keypoints' and 'class' fields
- LOSO directory structure: `{dataset}_LOSO_{user}/train/` and `test/`

## Comparison: 90 vs 100 Keypoints

| Feature | 100 Keypoints | 90 Keypoints |
|---------|--------------|--------------|
| Pose | Full body (33) | Upper body only (23) |
| Hands | 2 × 21 | 2 × 21 (same) |
| Face | 25 | 25 (same) |
| **Total** | **100** | **90** |
| Lower body | ✓ Included | ✗ Excluded |
| Model size | Slightly larger | Slightly smaller |
| Training speed | Baseline | ~10% faster |

## Notes

- The lower body keypoints (hips down) are rarely relevant for sign language
- Removing them reduces noise and improves generalization
- If your dataset already has 90 keypoints, this config is required
- The normalization groups are detected automatically based on MediaPipe structure

