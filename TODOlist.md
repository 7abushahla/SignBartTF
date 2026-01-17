# TODOlist.md - SignBartTF Pipeline (Aligned with TODOlistv2.md)

This checklist tracks training-pipeline work that corresponds to the app-side refactor in TODOlistv2.md.

---

## Goal 1 (Model Retrain for 65 Keypoints)

- [ ] **1.7.a** Flip left-handed users (user01, user02) to right-handed
  - Output: MLR511-ArabicSignLanguage-Dataset-MP4_FLIPPED/
  - Skip if .flip_complete marker exists

- [ ] **1.7.b** Extract 65 keypoints from normalized dataset
  - Script: extract_65_keypoints.py
  - Output: data/arabic-asl-65kpts/all/*.pkl
  - Label maps: data/arabic-asl-65kpts/label2id.json, id2label.json

- [ ] **1.7.c** Create LOSO splits for training/evaluation
  - Script: fix_loso.py
  - Output: data/arabic-asl-65kpts_LOSO_userXX/{train,test}

- [ ] **1.7.d** Train FP32 LOSO models (65 keypoints)
  - Script: train_loso_functional.py
  - Config: configs/arabic-asl-65kpts.yaml

- [ ] **1.7.e** Evaluate trained models (test split)
  - Output: *_eval.log

- [ ] **1.7.f** (Optional) QAT fine-tuning and TFLite export
  - Script: train_loso_functional_qat.py
  - Output: exports/qat_65kpts_userXX/qat_dynamic_int8.tflite

- [ ] **1.7.g** Deploy TFLite model to mobile app
  - Copy to: ../assets/models/final_model_qat_int8.tflite

---

## Notes

- This list mirrors Goal 1.7 (model retraining) in TODOlistv2.md.
- Use run_end_to_end_pipeline.sh for automated execution.