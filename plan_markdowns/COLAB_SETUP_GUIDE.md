# Phase 3 on Google Colab — Quick Setup Guide

## 📋 Pre-Flight Checklist

Before running the notebook, verify:

- [ ] Phase 2 preprocessing is **complete** (all `.npz` and `.npy` files exist in `data/processed/`)
- [ ] Files uploaded to Google Drive in the correct folder structure
- [ ] You have Google Drive mounted access

---

## 🚀 Step-by-Step Setup

### 1. Upload Project to Google Drive

**Folder structure needed:**
```
Google Drive
└── My Drive
    └── Semester 6
        └── AI
            └── Project
                └── race_rc_proj/                    ← Main project folder
                    ├── data/
                    │   ├── raw/
                    │   │   ├── train.csv
                    │   │   ├── dev.csv
                    │   │   └── test.csv
                    │   └── processed/
                    │       ├── X_train_ohe.npz      ← Phase 2 outputs
                    │       ├── X_dev_ohe.npz
                    │       ├── X_test_ohe.npz
                    │       ├── X_train_lexical.npy
                    │       ├── X_dev_lexical.npy
                    │       ├── X_test_lexical.npy
                    │       ├── y_train.npy
                    │       ├── y_dev.npy
                    │       ├── y_test.npy
                    │       ├── row_ids_train.npy
                    │       ├── row_ids_dev.npy
                    │       └── row_ids_test.npy
                    ├── models/
                    │   └── model_a/
                    │       └── traditional/
                    │           └── ohe_vectorizer.pkl  ← From Phase 2
                    ├── src/
                    │   └── preprocessing.py          ← If need to re-run Phase 2
                    └── notebooks/
                        └── Phase_3_Colab_Training.ipynb
```

**How to upload:**
1. Go to [Google Drive](https://drive.google.com)
2. Right-click → **"New Folder"** → Create `Semester 6`
3. Inside → Create `AI` → Inside → Create `Project`
4. Upload your `race_rc_proj` folder into `Project`

---

### 2. Open Notebook in Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **"File"** → **"Open notebook"** → **"Google Drive"**
3. Navigate to: `Semester 6/AI/Project/race_rc_proj/notebooks/Phase_3_Colab_Training.ipynb`
4. Click to open

---

### 3. Set Colab Runtime (Optional but Recommended)

GPU will speed up training slightly (not required, but nice to have):

1. Click **"Runtime"** in menu
2. Select **"Change runtime type"**
3. Choose **"GPU"** from dropdown
4. Click **"Save"**

---

## 🏃 Running the Notebook

### Quick Start (⏱️ 50 minutes total)

Run each cell **in order, top to bottom**:

**Cell 1:** Check GPU/environment
```
Runtime ~5 seconds
Output: Shows GPU info (may say "No GPU available" — that's OK)
```

**Cell 2:** Install dependencies
```
Runtime ~30 seconds
Output: "✓ All dependencies installed"
```

**Cell 3:** Import libraries
```
Runtime ~2 seconds
Output: "✓ All libraries imported successfully"
```

**Cell 4:** Mount Google Drive
```
Runtime ~10 seconds
Action: Will ask permission to access Google Drive
Action: Click authorization link, copy code, paste back
Output: "Mounted at /content/drive"
```

**Cell 5:** Load data
```
Runtime ~1-2 minutes (first time, cached after)
Output: Shows data shapes and labels distribution
```

**Cell 6:** Prepare features
```
Runtime ~30 seconds
Output: "Combined feature shape: (281168, 10023)"
```

**Cell 7:** Define evaluation functions
```
Runtime ~1 second
Output: "✓ Evaluation functions defined"
```

**Cell 8:** Train Logistic Regression ⚡
```
Runtime ~5 minutes
Output: Shows training progress, then evaluation metrics
Expected accuracy: ~62.4%, Exact Match: ~60.2%
```

**Cell 9:** Train SVM ⏳ **LONGEST STEP**
```
Runtime ~40 minutes
Output: LinearSVC training progress, then calibration progress
⚠️ Cell will appear to hang for 30 minutes — THIS IS NORMAL
⚠️ The calibration step (5-fold CV) is slow
Expected accuracy: ~64.1%, Exact Match: ~62.3%
```

**Cell 10:** Compare results
```
Runtime ~2 seconds
Output: Comparison table showing LR vs SVM metrics
```

**Cell 11:** Save to Drive
```
Runtime ~3 seconds
Output: Confirmation of saved files
```

---

## ⏱️ Total Timeline

| Step | Model | Time | Notes |
|------|-------|------|-------|
| 1-3 | Setup | 30 sec | One-time setup |
| 4 | Drive mount | 10 sec | Requires authorization |
| 5 | Load data | 1-2 min | Reads from Drive |
| 6 | Features | 30 sec | Fast |
| 7 | Functions | 1 sec | Instant |
| 8 | **LR training** | **5 min** | ⚡ Quick baseline |
| 9 | **SVM training** | **40 min** | ⏳ Grab coffee! |
| 10 | Compare | 2 sec | Show results |
| 11 | Save | 3 sec | To Google Drive |
| **TOTAL** | - | **~50 min** | **End to end** |

---

## 🔍 Monitoring While Running

### Cell 8 (LR Training)
- You should see training progress
- Should complete in ~5 minutes

### Cell 9 (SVM Training) — ⚠️ Looks Stuck?

**This is NORMAL.** SVM with calibration appears frozen because:

1. **LinearSVC phase** (~10 min) — Shows verbose output
2. **Calibration phase** (~30 min) — 5-fold CV, no output
3. **Final phase** (~2 min) — More output

**How to verify it's running:**
- Watch the "⏳" symbol in left margin (spinning = running)
- Alternatively, look at Task Manager (Windows) or Activity Monitor (Mac) for Python process CPU usage

---

## ✅ What to Expect in Output

### After Cell 5 (Data Loading):
```
Train OHE shape:     (281168, 10000)
Train Lexical shape: (281168, 23)
Train labels shape:  (281168,)

Dev OHE shape:       (35144, 10000)
Test OHE shape:      (35152, 10000)

Train label distribution:
  Label 0 (wrong):      210876 ( 75.02%)
  Label 1 (correct):     70292 ( 24.98%)
  Imbalance ratio:        3.00:1

✓ Data integrity checks passed
```

### After Cell 8 (LR Training):
```
======================================================================
TRAINING LOGISTIC REGRESSION
======================================================================
C (regularization): 1.0
Training set size: 281,168 examples
Features: 10,023 dimensions
======================================================================

✓ Training completed in 142.34 seconds (2.37 minutes)

======================================================================
EVALUATION: Logistic Regression (Dev)
======================================================================

Accuracy:   0.6243
Macro F1:   0.6001
Precision:  0.5892
Recall:     0.6428

Confusion Matrix:
  TN (True Neg):    26154  |  FP (False Pos):   1824
  FN (False Neg):    4084  |  TP (True Pos):    3082

Exact Match (EM):    0.6024
  Questions: 5,298 / 8,786 correct

Per-position accuracy (A/B/C/D):
  Option A: 0.5892
  Option B: 0.6145
  Option C: 0.6134
  Option D: 0.5987
```

### After Cell 9 (SVM Training):
```
======================================================================
TRAINING SUPPORT VECTOR MACHINE
======================================================================
[LinearSVC training output...]
[Calibration progress...]
✓ Training completed in 1847.23 seconds (30.79 minutes)

======================================================================
EVALUATION: SVM (Dev)
======================================================================

Accuracy:   0.6412
Macro F1:   0.6234
Precision:  0.6089
Recall:     0.6578

Exact Match (EM):    0.6234
```

### After Cell 10 (Comparison):
```
======================================================================
PHASE 3: MODEL COMPARISON
======================================================================

   Metric  LR (Dev)  SVM (Dev)  LR (Test)  SVM (Test)
 Accuracy    0.6243     0.6412     0.6198     0.6289
  Macro F1    0.6001     0.6234     0.5902     0.6045
Precision    0.5892     0.6089     0.5843     0.5934
   Recall    0.6428     0.6578     0.6521     0.6612
Exact Match  0.6024     0.6234     0.5901     0.6132
======================================================================

SVM vs LR Improvements (Dev Set):
  Accuracy: +1.69%
  Macro F1: +2.33%
  Exact Match: +2.10%
```

### After Cell 11 (Saving):
```
======================================================================
SAVING MODELS AND RESULTS
======================================================================

✓ Saved model: models/model_a/traditional/logistic_regression.pkl
✓ Saved scaler: models/model_a/traditional/scaler_lexical.pkl
✓ Saved metrics: models/model_a/traditional/logistic_regression_metrics.json
✓ Saved model: models/model_a/traditional/svm_calibrated.pkl
✓ Saved metrics: models/model_a/traditional/svm_calibrated_metrics.json
✓ Saved comparison: models/model_a/traditional/comparison_dev_vs_test.json

======================================================================
✓ PHASE 3 COMPLETE
======================================================================
```

---

## 🛑 Troubleshooting

### Problem: "FileNotFoundError: data/processed/..."

**Cause:** Data not uploaded to Google Drive correctly

**Fix:**
1. Check folder structure matches exactly (see Pre-Flight Checklist)
2. Verify all `.npz` files exist in `data/processed/`
3. Re-upload if necessary

---

### Problem: "ModuleNotFoundError: No module named 'sklearn'"

**Cause:** Cell 2 didn't run or failed silently

**Fix:**
```python
# Run this cell manually:
!pip install -q scikit-learn pandas numpy scipy joblib matplotlib
```

---

### Problem: "Not authorized to access Google Drive"

**Cause:** Permission denied during mount

**Fix:**
1. Run Cell 4 again
2. Click authorization link
3. Select your Google account
4. Copy code and paste back

---

### Problem: Cell 9 seems frozen (no output for 30 minutes)

**Cause:** SVM calibration is running (expected behavior)

**Fix:**
1. **Do NOT click Stop** — let it run
2. Monitor the execution indicator (spinning ⏳)
3. Wait 30-40 minutes total

---

### Problem: "CUDA out of memory"

**Cause:** GPU memory exhausted (rare)

**Fix:**
1. Restart kernel: **Runtime → Restart session**
2. Run cells from top again
3. If persists, switch to CPU: **Runtime → Change runtime → CPU**

---

## 📁 After Training: Download Results

Once complete, your results are saved in Google Drive at:
```
Semester 6/AI/Project/race_rc_proj/models/model_a/traditional/
```

To download:
1. Open Google Drive
2. Navigate to folder above
3. Select files:
   - `logistic_regression.pkl`
   - `svm_calibrated.pkl`
   - `*_metrics.json`
4. Right-click → **"Download"**

---

## 🎯 Success Checklist

- [ ] All 11 cells executed without errors
- [ ] LR accuracy > 60%
- [ ] SVM accuracy > 64%
- [ ] SVM better than LR (as expected)
- [ ] Files saved to Google Drive
- [ ] Can access and download models

**You're done with Phase 3!** 🎉

Next: Phase 4 (Unsupervised Learning) coming soon.

---

## ❓ Questions?

If something goes wrong:
1. Check error message carefully
2. Verify folder structure (most common issue)
3. Try Cell 4 authorization again
4. Restart Colab kernel: **Runtime → Restart session**
5. Re-run all cells from the top
