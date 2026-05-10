# Phase 3 Implementation: Step-by-Step Execution Guide

**Total Time Required:**
- Option A (Quick Baseline): ~10 minutes (LR only)
- Option B (Standard): ~60 minutes (LR + SVM)
- Option C (Full Tuning): ~3-4 hours (LR + SVM + hyperparameter tuning)

---

## STEP 1: Verify Preprocessing Completed (1 min)

Before running Phase 3, ensure Phase 2 preprocessing generated all required files.

### Check File Existence

Run this in PowerShell/Terminal:

```bash
# Check if processed data exists
ls data/processed/

# Expected files:
# - X_train_ohe.npz
# - X_dev_ohe.npz
# - X_test_ohe.npz
# - X_train_lexical.npy
# - X_dev_lexical.npy
# - X_test_lexical.npy
# - y_train.npy
# - y_dev.npy
# - y_test.npy
# - row_ids_train.npy
# - row_ids_dev.npy
# - row_ids_test.npy

# Check if vectorizer exists
ls models/model_a/traditional/ohe_vectorizer.pkl
```

**If files missing:**
```bash
# Re-run preprocessing
python src/preprocessing.py

# Wait for completion (takes 10-15 minutes)
```

---

## STEP 2: Create Required Directories (1 min)

```bash
# Create models directory for Phase 3 outputs
mkdir -p models/model_a/traditional
```

---

## STEP 3: Run Training (Choose One Option)

### OPTION A: Quick Baseline — Logistic Regression Only (⏱️ 5 minutes)

**Best for:** Quick testing, understanding the pipeline

```bash
python src/model_a_train.py --model lr
```

**What happens:**
1. Loads processed data from Phase 2 (~30 sec)
2. Scales and combines features (~30 sec)
3. Trains Logistic Regression with C=1.0 (~2-3 minutes)
4. Evaluates on dev set (~10 sec)
5. Evaluates on test set (~10 sec)
6. Saves model and metrics

**Expected output:**
```
======================================================================
PHASE 3: TRADITIONAL ML MODELS FOR ANSWER VERIFICATION
======================================================================

======================================================================
LOADING DATA
======================================================================

Train OHE shape:     (281168, 10000)
Train Lexical shape: (281168, 23)
Train labels shape:  (281168,)
Train row_ids shape: (281168,)

[... more diagnostic info ...]

✓ Data integrity checks passed

======================================================================
PREPARING FEATURES
======================================================================

Combined feature shape: (281168, 10023)
  OHE: 10,000 dims
  Lexical (scaled): 23 dims
  Total: 10,023 dims

======================================================================
TRAINING LOGISTIC REGRESSION
======================================================================

C (regularization): 1.0
Training set size: 281,168 examples
Features: 10,023 dimensions
Solver: saga | Max iterations: 1000 | Class weight: balanced

✓ Training completed in 142.34 seconds (2.37 minutes)

======================================================================
EVALUATION: Logistic Regression (Dev)
======================================================================

Accuracy:   0.6243
Macro F1:   0.6001
Precision:  0.5892
Recall:     0.6428

[... confusion matrix and classification report ...]

Exact Match (EM):    0.6024
  Questions: 5,298 / 8,786 correct

Per-position accuracy (A/B/C/D):
  Option A: 0.5892
  Option B: 0.6145
  Option C: 0.6134
  Option D: 0.5987

✓ Saved model: models/model_a/traditional/logistic_regression.pkl
✓ Saved scaler: models/model_a/traditional/scaler_lexical.pkl
✓ Saved metrics: models/model_a/traditional/logistic_regression_metrics.json
```

---

### OPTION B: Standard — Both Models (⏱️ 45-60 minutes)

**Best for:** Getting full comparison, submitting coursework

```bash
python src/model_a_train.py --model all
```

**What happens:**
1. Loads data and prepares features (as Option A)
2. Trains Logistic Regression (~2-3 min)
3. Evaluates LR on dev + test (~1 min)
4. Trains SVM with calibration (~30-40 min)  ⚠️ **LONGEST STEP**
5. Evaluates SVM on dev + test (~1-2 min)
6. Creates comparison table
7. Saves both models

**Expected output after SVM training:**
```
======================================================================
EVALUATION: SVM (Dev)
======================================================================

Accuracy:   0.6412
Macro F1:   0.6234
Precision:  0.6089
Recall:     0.6578

[... metrics ...]

Exact Match (EM):    0.6234

======================================================================
PHASE 3: MODEL COMPARISON
======================================================================

                       Accuracy  Macro F1  Precision    Recall  Exact Match
Logistic Regression     0.6243     0.6001     0.5892    0.6428      0.6024
SVM                     0.6412     0.6234     0.6089    0.6578      0.6234

✓ SVM achieves +1.7% better accuracy than LR
```

---

### OPTION C: Full Analysis — With Hyperparameter Tuning (⏱️ 3-4 hours)

**Best for:** Thorough analysis, finding optimal hyperparameters

```bash
python src/model_a_train.py --model all --tune
```

**What happens:**
1. Loads data and prepares features
2. **LR Tuning:** Tests C ∈ [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
   - Each C value: ~2-3 min training + evaluation
   - Total: ~15 minutes
3. Trains best LR model
4. **SVM Tuning:** Tests C ∈ [0.01, 0.1, 1.0, 10.0, 100.0]
   - Each C value: ~30-45 min training (with calibration) + evaluation
   - **Total: ~2.5-3 hours** ⚠️ ⚠️ ⚠️
5. Trains best SVM model
6. Creates detailed comparison

**Expected LR tuning output:**
```
======================================================================
HYPERPARAMETER TUNING: Logistic Regression
======================================================================

C              Accuracy        Macro F1         EM              Time (sec)
0.001          0.5234          0.5012           0.5123          142.12
0.01           0.5512          0.5289           0.5401          142.45
0.1            0.6015          0.5823           0.5901          142.89
1.0            0.6243          0.6001           0.6024          142.34
10.0           0.6189          0.5934           0.5967          142.56
100.0          0.6156          0.5896           0.5923          142.78

======================================================================
✓ Best C: 1.0 (Accuracy: 0.6243)
======================================================================
```

---

## STEP 4: Check Output Files (2 min)

After training completes, verify all files were saved:

```bash
# Check saved models
ls -la models/model_a/traditional/

# Expected files:
# - logistic_regression.pkl (100-300 MB)
# - svm_calibrated.pkl (100-300 MB) [if trained SVM]
# - scaler_lexical.pkl (1 KB)
# - logistic_regression_metrics.json
# - svm_calibrated_metrics.json [if trained SVM]
```

**View metrics:**
```bash
# View LR metrics
cat models/model_a/traditional/logistic_regression_metrics.json

# Output:
# {
#   "timestamp": "2026-05-06T14:35:22.123456",
#   "model_name": "logistic_regression",
#   "model_type": "lr",
#   "metrics": {
#     "accuracy": 0.6243,
#     "macro_f1": 0.6001,
#     "precision": 0.5892,
#     "recall": 0.6428,
#     "exact_match": 0.6024,
#     ...
#   }
# }
```

---

## STEP 5: Document Results in Jupyter Notebook (5-10 min)

Create a cell in `notebooks/experiments.ipynb` to load and display results:

```python
# Phase 3: Traditional ML Results
import json
import pandas as pd

# Load metrics
with open('models/model_a/traditional/logistic_regression_metrics.json', 'r') as f:
    lr_metrics = json.load(f)

with open('models/model_a/traditional/svm_calibrated_metrics.json', 'r') as f:
    svm_metrics = json.load(f)

# Create comparison table
results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Macro F1', 'Precision', 'Recall', 'Exact Match'],
    'Logistic Regression': [
        lr_metrics['metrics']['accuracy'],
        lr_metrics['metrics']['macro_f1'],
        lr_metrics['metrics']['precision'],
        lr_metrics['metrics']['recall'],
        lr_metrics['metrics']['exact_match']
    ],
    'SVM': [
        svm_metrics['metrics']['accuracy'],
        svm_metrics['metrics']['macro_f1'],
        svm_metrics['metrics']['precision'],
        svm_metrics['metrics']['recall'],
        svm_metrics['metrics']['exact_match']
    ]
})

print("Phase 3: Model Comparison")
print(results_df.to_string(index=False))

# Calculate improvements
print(f"\nSVM vs LR Improvements:")
print(f"  Accuracy: +{(svm_metrics['metrics']['accuracy'] - lr_metrics['metrics']['accuracy'])*100:.2f}%")
print(f"  Macro F1: +{(svm_metrics['metrics']['macro_f1'] - lr_metrics['metrics']['macro_f1'])*100:.2f}%")
print(f"  Exact Match: +{(svm_metrics['metrics']['exact_match'] - lr_metrics['metrics']['exact_match'])*100:.2f}%")
```

---

## STEP 6: Troubleshooting Common Issues

### Issue: "ModuleNotFoundError: No module named 'sklearn'"

**Solution:**
```bash
pip install scikit-learn pandas numpy scipy joblib
```

### Issue: "FileNotFoundError: data/processed/X_train_ohe.npz"

**Solution:** Re-run preprocessing from Phase 2:
```bash
python src/preprocessing.py
```

### Issue: "MemoryError" during training

**Solution:** The model requires ~5-10x the data size in RAM.
- For 10GB+ feature matrices, SVM may not fit
- Try:
  ```bash
  # Use only LR (less memory)
  python src/model_a_train.py --model lr
  
  # Or subsample data in preprocessing (edit preprocessing.py)
  ```

### Issue: SVM training seems stuck / no output for 30 minutes

**This is normal!** SVM with 5-fold calibration is very slow. Monitor process:

```bash
# Check if Python process still running
tasklist | grep python  # Windows
ps aux | grep python    # Mac/Linux

# Check disk I/O
# (SVM may be writing to disk during calibration)
```

---

## STEP 7: Verify Results Make Sense

### Expected Metrics (Baseline)

| Metric | LR | SVM | Notes |
|--------|-----|-----|-------|
| Accuracy | ~62% | ~64% | Good; both beat 50% random |
| Macro F1 | ~60% | ~62% | Similar to accuracy (balanced class) |
| Exact Match | ~60% | ~62% | Per-question accuracy slightly lower |
| Precision | ~59% | ~61% | Not all predicted correct answers are actually correct |
| Recall | ~64% | ~66% | Can identify ~65% of actual correct answers |

### Red Flags (Something Wrong)

| Symptom | Likely Cause | Fix |
|---------|------------|-----|
| Accuracy < 55% | Bad features or wrong labels | Check preprocessing output |
| Accuracy > 75% | Overfitting or label leakage | Verify train/dev/test split |
| All predictions = 0 | Model predicts all "wrong" | Check class weight or threshold |
| OOM error | Sparse matrix not working | Check feature preparation code |

---

## STEP 8: Next Steps (Phase 4)

Once Phase 3 models are trained and saved, you're ready for Phase 4:

```bash
# Phase 4 Preview: Apply PCA to model features
python src/model_a_phase4.py --method pca --components 50

# Then run K-Means clustering, Label Propagation, GMM
```

---

## Timeline Summary

```
START
  ↓
[5 min] Verify Phase 2 outputs exist
  ↓
[1 min] Create model directory
  ↓
CHOOSE ONE PATH:
  ├─ FAST: [5 min] Train LR only
  ├─ STANDARD: [45-60 min] Train LR + SVM
  └─ THOROUGH: [3-4 hours] Train LR + SVM + hyperparameter tuning
  ↓
[2 min] Verify saved files
  ↓
[5-10 min] Document in Jupyter
  ↓
COMPLETE ✓
```

---

## Running Multiple Experiments (Advanced)

If you want to experiment with different settings, create a bash script:

```bash
# save as run_phase3_all.sh

#!/bin/bash

echo "Running Phase 3 experiments..."

echo "1. LR baseline..."
python src/model_a_train.py --model lr

echo "2. SVM baseline..."
python src/model_a_train.py --model svm

echo "3. Both with tuning..."
python src/model_a_train.py --model all --tune

echo "All experiments complete!"
```

Then run:
```bash
bash run_phase3_all.sh
# (This will run sequentially, taking ~4 hours total)
```

---

## Key Commands for Reference

```bash
# Train LR (FAST - 5 min)
python src/model_a_train.py --model lr

# Train SVM (MEDIUM - 45 min)
python src/model_a_train.py --model svm

# Train both (STANDARD - 1 hour)
python src/model_a_train.py --model all

# Train both with tuning (THOROUGH - 3-4 hours)
python src/model_a_train.py --model all --tune

# Tune only LR (QUICK TUNING - 20 min)
python src/model_a_train.py --model lr --tune

# View saved metrics
cat models/model_a/traditional/logistic_regression_metrics.json
cat models/model_a/traditional/svm_calibrated_metrics.json

# View all models saved
ls -lah models/model_a/traditional/
```

---

## Success Criteria ✅

You've successfully completed Phase 3 when:

- [ ] Both LR and SVM models trained and saved
- [ ] Accuracy > 60% on dev set
- [ ] Exact Match > 60% on dev set
- [ ] Metrics JSON files exist and contain all metrics
- [ ] SVM accuracy > LR accuracy (SVM should be better)
- [ ] Results documented in experiments.ipynb
- [ ] Code committed to git with meaningful message

Example commit:
```bash
git add models/model_a/traditional/*.pkl models/model_a/traditional/*.json
git add src/model_a_train.py
git commit -m "phase 3: train LR and SVM - LR acc 62.4%, SVM acc 64.1%"
```

---

**Ready? Start with Step 1!**
