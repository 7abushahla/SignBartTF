# SignBART TensorFlow - Arabic Sign Language Recognition

TensorFlow/Keras implementation of SignBART for Arabic sign language gesture recognition with full quantization support (PTQ and QAT).

## 🎯 Features

- **Functional API Model**: QAT-ready architecture using Keras Functional API
- **LOSO Cross-Validation**: Leave-One-Signer-Out evaluation across 3 users
- **Full Dataset Training**: Train on all 12 users combined
- **Quantization Support**: 
  - Post-Training Quantization (PTQ)
  - Quantization-Aware Training (QAT) with optimized hyperparameters
  - Dynamic-range INT8 quantization (weights INT8, activations FP32)
- **TFLite Export**: Optimized models for mobile/edge deployment
- **Comprehensive Evaluation**: Accuracy metrics, confusion matrices, FLOPs calculation

---

## 📁 Project Structure

```
signbart_tf/
├── configs/
│   ├── arabic-asl-65kpts.yaml       # 65 keypoints (upper body + hands, no face)
│   ├── arabic-asl-42kpts-hands.yaml # 42 keypoints (hands only)
│   ├── arabic-asl-44kpts-hands-wrists.yaml
│   ├── arabic-asl-44kpts-hands-shoulders.yaml
│   ├── arabic-asl-46kpts-hands-shoulders-wrists.yaml
│   ├── arabic-asl-48kpts-hands-shoulders-elbows-wrists.yaml
│   └── arabic-asl-90kpts.yaml       # 90 keypoints (body + hands + face)
├── data/
│   ├── arabic-asl-90kpts/           # Full dataset (all users)
│   │   ├── all/                     # All samples for full training
│   │   │   ├── G01/ ... G10/
│   │   ├── label2id.json
│   │   └── id2label.json
│   ├── arabic-asl-90kpts_LOSO_user01/  # LOSO split for user01
│   │   ├── train/                   # Training samples (users 08, 11)
│   │   ├── test/                    # Test samples (user01)
│   │   └── ...
│   └── ...
├── outputs/                         # Dataset/run-type separated outputs (default)
│   └── <dataset>/                   # e.g., arabic_asl, lsa64
│       ├── loso/
│       │   ├── checkpoints/         # FP32 checkpoints + FP32 TFLite
│       │   ├── exports/ptq/         # PTQ models (per user)
│       │   ├── exports/qat/         # QAT models (per user)
│       │   ├── logs/run_logs/       # Training logs
│       │   ├── out-imgs/            # Training curves
│       │   ├── training_metadata/   # Structured metadata
│       │   └── results/             # Evaluation results
│       └── full/
│           ├── checkpoints/
│           ├── exports/ptq/
│           ├── exports/qat/
│           ├── logs/run_logs/
│           ├── out-imgs/
│           └── results/
```

**Output root**: defaults to `outputs/` (or `SIGNBART_OUTPUT_ROOT` if set). You can override with `--output_root`.
Legacy folders (`checkpoints_*`, `exports/`, `logs/`, `results/`, `out-imgs/`, `training_metadata/`) may still exist for older runs.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n signbart_tf python=3.10
conda activate signbart_tf
pip install tensorflow tensorflow-model-optimization keras pyyaml numpy matplotlib seaborn
```

**MediaPipe extractor compatibility (required for keypoint extraction):**

```bash
# MediaPipe solutions API (used by extract_65_keypoints.py)
pip install mediapipe==0.10.14

# TensorFlow 2.15 compatibility pins
pip install numpy==1.26.4 ml-dtypes==0.3.2

# OpenCV contrib (MediaPipe dependency) pinned to numpy<2
pip install opencv-contrib-python==4.9.0.80
```

### 2. Prepare 65-Keypoint Dataset (No Face)

The raw videos are already available in:

- MLR511-ArabicSignLanguage-Dataset-MP4/

Use the 65-keypoint extractor to build the training dataset:

```bash
python extract_65_keypoints.py \
    --input_dir MLR511-ArabicSignLanguage-Dataset-MP4 \
    --output_dir data/arabic-asl-65kpts
```

This writes:

- data/arabic-asl-65kpts/all/G01...G10/*.pkl
- data/arabic-asl-65kpts/label2id.json
- data/arabic-asl-65kpts/id2label.json

Create LOSO splits for training/evaluation:

```bash
python fix_loso.py \
    --base_root data/arabic-asl-65kpts \
    --holdouts user01,user08,user11
```

#### LSA64 (63 keypoints) Quickstart

LSA64 uses 63-keypoint pickles and 64 gesture classes (G01-G64). Use `--holdouts all` to run LOSO across all 10 users.

1) Create LOSO splits:
```bash
python fix_loso.py \
    --base_root data/lsa64-63kpts \
    --holdouts all
```

2) FP32 LOSO training (single user smoke test):
```bash
python train_loso_functional.py \
    --config_path configs/lsa64-63kpts.yaml \
    --base_data_path data/lsa64-63kpts \
    --holdout_only user0001 \
    --epochs 1 \
    --dataset_name lsa64 \
    --output_root outputs
```

3) PTQ / QAT exports (all users):
```bash
python ptq_export_batch.py \
    --config_path configs/lsa64-63kpts.yaml \
    --base_data_path data/lsa64-63kpts \
    --holdouts all \
    --dataset_name lsa64 \
    --output_root outputs

python train_loso_functional_qat_batch.py \
    --config_path configs/lsa64-63kpts.yaml \
    --base_data_path data/lsa64-63kpts \
    --holdouts all \
    --dataset_name lsa64 \
    --output_root outputs
```

4) Evaluate TFLite models (example user0001):
```bash
python test_tflite_models.py \
    --config_path configs/lsa64-63kpts.yaml \
    --data_path data/lsa64-63kpts_LOSO_user0001 \
    --fp32_tflite outputs/lsa64/loso/checkpoints/lsa64_LOSO_user0001/final_model_fp32.tflite \
    --ptq_tflite outputs/lsa64/loso/exports/ptq/user0001/model_dynamic_int8.tflite
```

Then train with the 65-keypoint config:

```bash
python train_loso_functional.py \
    --config_path configs/arabic-asl-65kpts.yaml \
    --base_data_path data/arabic-asl-65kpts \
    --epochs 80 \
    --lr 2e-4 \
    --no_validation \
    --dataset_name arabic_asl \
    --output_root outputs
```

Default seed is 42 (override with --seed as needed).

---

### 3. Training Workflows

#### **LOSO Training (Recommended for Research)**

Train on 3 LOSO splits (leave-one-signer-out):

```bash
# All 3 users (user01, user08, user11) - 65 keypoints (no face)
python train_loso_functional.py \
    --config_path configs/arabic-asl-65kpts.yaml \
    --base_data_path data/arabic-asl-65kpts \
    --epochs 80 \
    --lr 2e-4 \
    --no_validation \
    --dataset_name arabic_asl \
    --output_root outputs
```

**Output**: FP32 models in `outputs/arabic_asl/loso/checkpoints/arabic_asl_LOSO_user01/`, `user08/`, `user11/`

**Quick test** (single user, 2 epochs):

```bash
python train_loso_functional.py \
    --config_path configs/arabic-asl-65kpts.yaml \
    --base_data_path data/arabic-asl-65kpts \
    --holdout_only user01 \
    --epochs 2 \
    --lr 2e-4 \
    --no_validation \
    --dataset_name arabic_asl \
    --output_root outputs
```

If you want the 90-keypoint variant (with face), replace the config and data paths with:

- configs/arabic-asl-90kpts.yaml
- data/arabic-asl-90kpts

---

#### **Full Dataset Training**

Train on all 12 users:

```bash
python train_full_dataset.py \
    --config_path configs/arabic-asl-65kpts.yaml \
    --base_data_path data/arabic-asl-65kpts \
    --epochs 80 \
    --lr 2e-4 \
    --seed 42 \
    --dataset_name arabic_asl \
    --output_root outputs
```

**Output**: `outputs/arabic_asl/full/checkpoints/arabic_asl_full/final_model.h5` and `final_model_fp32.tflite`

---

### 3b. End-to-End Pipeline (Flip → Extract → LOSO → Full Dataset)

The pipeline runs dataset flipping, 65-keypoint extraction, LOSO training, and (optionally) full-dataset training:

```bash
bash run_end_to_end_pipeline.sh
```

Control full-dataset training:

```bash
# Skip full dataset training if already done
RUN_FULL_DATASET=0 bash run_end_to_end_pipeline.sh
```

Optional overrides:

```bash
FULL_EPOCHS=80 FULL_LR=2e-4 FULL_SEED=42 FULL_EXP_NAME=arabic_asl_full \
    bash run_end_to_end_pipeline.sh
```

### 3c. Variant Pipeline (Multiple Keypoint Subsets)

Run multiple keypoint variants (hands-only, shoulders, wrists, etc.) and include 90-keypoint training:

```bash
bash run_end_to_end_variants.sh
```

Control which variants run:

```bash
VARIANTS="42kpts-hands 44kpts-hands-wrists 46kpts-hands-shoulders-wrists 65kpts 90kpts" \
    bash run_end_to_end_variants.sh
```

Defaults for LOSO runs:

- SEED_LIST=42,511,999983,1000003,324528439 (seed sweep)
- HOLDOUTS=all (all users)
- RUN_LOSO=1 (set RUN_LOSO=0 to skip LOSO for other datasets)

Other dataset support (optional):

```bash
# Point to a different dataset root and skip LOSO if labels/users differ
RAW_DIR=/path/to/your_dataset \
FLIPPED_DIR=/path/to/your_dataset_flipped \
DATA_DIR_65=/path/to/output_keypoints_65 \
DATA_DIR_90=/path/to/output_keypoints_90 \
RUN_LOSO=0 \
bash run_end_to_end_variants.sh
```

Note: HOLDOUTS=all requires filenames like userXX_GYY_RZZ.pkl to discover users.

The variant pipeline will:

- Reuse the 65-keypoint dataset
- Train LOSO models with an experiment prefix per variant
- Export PTQ/QAT models to variant-specific export directories
- Run collect_results with the correct prefix and export paths

---

### 4. Quantization

#### **Post-Training Quantization (PTQ)**

Dynamic-range INT8 quantization (weights only):

```bash
# For LOSO models (all 3 users)
python ptq_export_batch.py \
    --config_path configs/arabic-asl-65kpts.yaml \
    --base_data_path data/arabic-asl-65kpts

# For single LOSO model (e.g., user01)
python ptq_export.py \
    --config_path configs/arabic-asl-65kpts.yaml \
    --checkpoint outputs/arabic_asl/loso/checkpoints/arabic_asl_LOSO_user01/final_model.h5 \
    --output_dir outputs/arabic_asl/loso/exports/ptq/user01

# For full dataset model
python ptq_export.py \
    --config_path configs/arabic-asl-65kpts.yaml \
    --checkpoint outputs/arabic_asl/full/checkpoints/arabic_asl_full/final_model.h5 \
    --output_dir outputs/arabic_asl/full/exports/ptq

# For a keypoint variant (example)
python ptq_export_batch.py \
    --config_path configs/arabic-asl-42kpts-hands.yaml \
    --base_data_path data/arabic-asl-65kpts \
    --exp_prefix arabic_asl_42kpts_hands \
    --output_base_dir outputs/arabic_asl/loso/exports/ptq_arabic_asl_42kpts_hands
```

---

#### **Quantization-Aware Training (QAT)**

Fine-tune with simulated quantization (better accuracy than PTQ):

```bash
# For LOSO models (all 3 users)
python train_loso_functional_qat_batch.py \
    --config_path configs/arabic-asl-65kpts.yaml \
    --base_data_path data/arabic-asl-65kpts \
    --batch_size 4 \
    --qat_epochs 20 \
    --lr 5e-5 \
    --no_validation

# For single LOSO model (e.g., user01)
python train_loso_functional_qat.py \
    --config_path configs/arabic-asl-65kpts.yaml \
    --data_path data/arabic-asl-65kpts_LOSO_user01 \
    --checkpoint outputs/arabic_asl/loso/checkpoints/arabic_asl_LOSO_user01/final_model.h5 \
    --output_dir outputs/arabic_asl/loso/exports/qat/user01 \
    --batch_size 4 \
    --qat_epochs 20 \
    --lr 5e-5

# For full dataset model
python train_loso_functional_qat.py \
    --config_path configs/arabic-asl-65kpts.yaml \
    --data_path data/arabic-asl-65kpts \
    --checkpoint outputs/arabic_asl/full/checkpoints/arabic_asl_full/final_model.h5 \
    --output_dir outputs/arabic_asl/full/exports/qat \
    --batch_size 4 \
    --qat_epochs 20 \
    --lr 5e-5 \
    --no_validation

# For a keypoint variant (example)
python train_loso_functional_qat_batch.py \
    --config_path configs/arabic-asl-42kpts-hands.yaml \
    --base_data_path data/arabic-asl-65kpts \
    --exp_prefix arabic_asl_42kpts_hands \
    --output_base_dir outputs/arabic_asl/loso/exports/qat_arabic_asl_42kpts_hands \
    --batch_size 4 \
    --qat_epochs 20 \
    --lr 5e-5 \
    --no_validation
```

**QAT Configuration**:
- **Learning Rate**: 5e-5 (~4× lower than FP32 training)
- **Batch Size**: 4 (larger than training for stability)
- **Epochs**: 20 (short fine-tuning)
- **Quantized Layers**: All Dense layers (FFN, attention projections, projection layers)
- **Excluded**: Projection container (tuple output handling issue)
- **Gradient Clipping**: clipnorm=1.0
- **Early Stopping**: Patience 10 (restores best weights)

**Quick QAT Demo** (builds toy model, applies QAT, exports):

```bash
python run_qat_export.py \
    --config_path configs/arabic-asl-90kpts.yaml \
    --output_dir exports/qat_demo \
    --save_keras \
    --seed 42
```

---

### 5. Evaluation

#### **Single Model Evaluation**

Evaluate any TFLite model on any dataset split:

```bash
python evaluate_tflite_single.py \
    --config_path configs/arabic-asl-65kpts.yaml \
    --data_path data/arabic-asl-65kpts_LOSO_user01 \
    --split test \
    --tflite_path outputs/arabic_asl/full/checkpoints/arabic_asl_full/final_model_fp32.tflite
```

---

#### **Compare FP32 vs PTQ vs QAT**

Side-by-side comparison of all three quantization approaches:

```bash
# Compare all three models
python test_tflite_models.py \
    --config_path configs/arabic-asl-65kpts.yaml \
    --data_path data/arabic-asl-65kpts_LOSO_user01 \
    --fp32_tflite outputs/arabic_asl/loso/checkpoints/arabic_asl_LOSO_user01/final_model_fp32.tflite \
    --ptq_tflite outputs/arabic_asl/loso/exports/ptq/user01/model_dynamic_int8.tflite \
    --qat_tflite outputs/arabic_asl/loso/exports/qat/user01/qat_dynamic_int8.tflite

# Compare FP32 vs PTQ only
python test_tflite_models.py \
    --config_path configs/arabic-asl-65kpts.yaml \
    --data_path data/arabic-asl-65kpts_LOSO_user01 \
    --fp32_tflite outputs/arabic_asl/loso/checkpoints/arabic_asl_LOSO_user01/final_model_fp32.tflite \
    --ptq_tflite outputs/arabic_asl/loso/exports/ptq/user01/model_dynamic_int8.tflite
```

---

#### **Test Single Sample**

Detailed analysis of a single prediction (with raw keypoint dump):

```bash
python test_single_sample.py \
    --test_dir data/arabic-asl-65kpts_LOSO_user01/test \
    --tflite_model outputs/arabic_asl/loso/checkpoints/arabic_asl_LOSO_user01/final_model_fp32.tflite \
    --config_path configs/arabic-asl-65kpts.yaml \
    --sample_file data/arabic-asl-65kpts_LOSO_user01/test/G10/user01_G10_R10.pkl \
    --sample_label G10 \
    --dump_raw_sample raw_keypoints.json
```

---

#### **Comprehensive Results Collection**

Generate full report with confusion matrices, FLOPs, and accuracy tables:

```bash
python collect_results.py \
    --run_evaluation \
    --config_path configs/arabic-asl-65kpts.yaml \
    --base_data_path data/arabic-asl-65kpts

# Multi-seed LOSO summary (mean/std across seeds)
python collect_results.py \
    --run_evaluation \
    --config_path configs/arabic-asl-65kpts.yaml \
    --base_data_path data/arabic-asl-65kpts \
    --seed_list "42,511,999983,1000003,324528439" \
    --exp_prefix_base arabic_asl_65kpts_pose_hands \
    --ptq_base_dir "outputs/arabic_asl/loso/exports/ptq_arabic_asl_65kpts_pose_hands_seed{seed}" \
    --qat_base_dir "outputs/arabic_asl/loso/exports/qat_arabic_asl_65kpts_pose_hands_seed{seed}"

# Variant example (with exp_prefix + custom PTQ/QAT export dirs)
python collect_results.py \
    --run_evaluation \
    --config_path configs/arabic-asl-42kpts-hands.yaml \
    --base_data_path data/arabic-asl-65kpts \
    --exp_prefix arabic_asl_42kpts_hands \
    --ptq_base_dir outputs/arabic_asl/loso/exports/ptq_arabic_asl_42kpts_hands \
    --qat_base_dir outputs/arabic_asl/loso/exports/qat_arabic_asl_42kpts_hands
```

**Output** (default):
- `outputs/arabic_asl/loso/results/experiment_results_YYYYMMDD_HHMMSS_*.txt` - Full text report
- `outputs/arabic_asl/loso/results/confusion_matrices/*.png` - 9 confusion matrices (3 users × 3 models)
- `outputs/arabic_asl/loso/results/model_info.csv` - Parameters, FLOPs
- `outputs/arabic_asl/loso/results/summary_table.csv` - FP32 vs PTQ vs QAT comparison
- `outputs/arabic_asl/loso/results/per_class_accuracy.csv` - Per-gesture accuracy

---

## 📱 Deploy to Mobile (Android/iOS)

Copy the exported TFLite model into the Flutter app assets and rebuild the app:

```bash
# Example: use QAT INT8 model for user01
cp outputs/arabic_asl/loso/exports/qat/user01/qat_dynamic_int8.tflite \
    ../assets/models/final_model_qat_int8.tflite
```

For keypoint variants (custom exp_prefix export dirs), update the source path accordingly:

```bash
cp outputs/arabic_asl/loso/exports/qat_arabic_asl_42kpts_hands/user01/qat_dynamic_int8.tflite \
    ../assets/models/final_model_qat_int8.tflite
```

Then rebuild the Flutter app (Android/iOS) so the asset bundle includes the new model.

---

## 📊 Model Architecture

```
Input: Keypoints [T, K, 2] (K from config, e.g., 65)
  ↓
Projection Layer (proj_x1, proj_y1) → [T, d_model=144]
  ↓
Positional Embeddings (learned)
  ↓
Encoder (2 layers, 4 heads, FFN 576)
  ├─ Self-Attention (q_proj, k_proj, v_proj, out_proj)
  ├─ LayerNorm + Residual
  ├─ Feed-Forward (fc1, fc2)
  └─ LayerNorm + Residual
  ↓
Decoder (2 layers, 4 heads, FFN 576)
  ├─ Causal Self-Attention
  ├─ Cross-Attention to Encoder
  ├─ Feed-Forward (fc1, fc2)
  └─ LayerNorm + Residual
  ↓
Extract Last Valid Token
  ↓
Classification Head → [10 classes]
```

**Parameters**: 773,578 total  
**FLOPs**: Calculated per forward pass  

### Model Size Comparison

| Format | Size | Compression | Use Case |
|--------|------|-------------|----------|
| Keras .h5 (FP32) | ~9.2 MB | - | Training/Fine-tuning |
| TFLite FP32 | ~3.0 MB | 3.1× | CPU inference (high accuracy) |
| TFLite INT8 (PTQ/QAT) | ~1.03 MB | 12.3× | Mobile/Edge (optimized) |  

---

## 🔬 Quantization Details

### What Gets Quantized

✅ **Quantized** (Weights + Activations during training, Weights-only in TFLite):
- FFN Dense layers: `fc1`, `fc2` (in encoder & decoder)
- Attention projections: `q_proj`, `k_proj`, `v_proj`, `out_proj`
- Input projections: `proj_x1`, `proj_y1`
- Classification head: `out_proj`

❌ **Not Quantized**:
- Embeddings (positional)
- Normalization layers (LayerNorm)
- Activation functions (GELU, Softmax)
- Dropout
- Structural operations (residual connections, masking)

🚫 **Excluded from Wrapping** (Critical):
- `Projection` container (causes collapse if wrapped, but internal Dense layers ARE quantized)

### Why Dynamic-Range Quantization?

We use **weights-only INT8 quantization** (dynamic-range) instead of full INT8 because:
- ✅ Significant model size reduction (~75% smaller)
- ✅ Numerically stable (avoids INF/NaN in attention & normalization)
- ✅ No calibration dataset needed
- ❌ Full INT8 (with calibration) caused numerical instability → INF values

---

## 🎓 Key Findings (QAT Optimization)

### Training Stability Issues Solved

**Problem**: Model collapse after 3-4 QAT epochs (accuracy dropped from 95% → 11%)

**Root Cause**: The `Projection` container layer (tuple output) was sensitive to `QuantizeWrapper`, even with `NoOpQuantizeConfig`.

**Solution**: 
1. Exclude `Projection` container from wrapping entirely
2. Still quantize its internal Dense layers (`proj_x1`, `proj_y1`) via filters
3. Use lower LR (5e-5 vs 2e-4 for FP32 training)
4. Increase batch size (4 vs 1 for FP32 training)
5. Add gradient clipping (clipnorm=1.0)
6. Early stopping with best-weight restoration

**Result**: Stable QAT training reaching 95% accuracy ✅

### Attention Layers Are Safe to Quantize

**Myth**: Attention projections are too sensitive for quantization  
**Reality**: `q_proj`, `k_proj`, `v_proj`, `out_proj` can be safely quantized with proper hyperparameters

---

## 📈 Expected Results

### LOSO Cross-Validation (3 users)

| Model Type | Accuracy | Top-5 Acc | Size (MB) | Speedup |
|------------|----------|-----------|-----------|---------|
| FP32       | 94-96%   | 99-100%   | 3.00      | 1.0×    |
| INT8-PTQ   | 93-95%   | 99-100%   | 0.75      | 2-3×    |
| INT8-QAT   | 94-96%   | 99-100%   | 0.75      | 2-3×    |

**QAT advantage**: +1-2% accuracy over PTQ while maintaining same size/speed.

---

## 🛠️ Key Scripts Reference

### Training
- `train_loso_functional.py` - LOSO training (3 users)
- `train_full_dataset.py` - Full dataset training (12 users)
- `main_functional.py` - Core training logic (called by above)

### Quantization
- `ptq_export.py` - PTQ for single model
- `ptq_export_batch.py` - PTQ for all LOSO models
- `train_loso_functional_qat.py` - QAT for single model
- `train_loso_functional_qat_batch.py` - QAT for all LOSO models

### Evaluation
- `evaluate_tflite_single.py` - Evaluate any TFLite model on any dataset
- `collect_results.py` - Comprehensive report generation
- `test_tflite_models.py` - Compare FP32/PTQ/QAT side-by-side

### Utilities
- `dataset.py` - Dataset loading & preprocessing
- `model_functional.py` - Functional API model definition
- `layers.py` - Custom layers (Projection, ClassificationHead, etc.)
- `encoder.py`, `decoder.py`, `attention.py` - Architecture components

---

## 🐛 Troubleshooting

### Issue: "FileNotFoundError: train split not found"

**Cause**: Using LOSO script on full dataset (or vice versa)

**Solution**:
- LOSO: Use `train_loso_functional_qat.py` with `data/arabic-asl-90kpts_LOSO_userXX`
- Full: Use `train_loso_functional_qat.py` with `data/arabic-asl-90kpts` (auto-detects `all` split)

---

### Issue: "Top5Accuracy deserialization error"

**Cause**: Mismatch between saved model config and metric definition

**Solution**: Already fixed in latest code (extracts `k` from kwargs)

---

### Issue: QAT model collapse

**Cause**: One of:
1. Wrapping `Projection` container
2. Learning rate too high
3. Batch size too small

**Solution**: Use provided QAT hyperparameters (lr=5e-5, batch=4)

---

## 📚 Citation and Reaching Out

### Citation
If you use this repository or its contents in your work, please cite this paper:
```bibtex

```

### Contact
If you have any questions, please feel free to reach out to me through email ([b00090279@alumni.aus.edu](mailto:b00090279@alumni.aus.edu)) or by connecting with me on [LinkedIn](https://www.linkedin.com/in/hamza-abushahla/).





