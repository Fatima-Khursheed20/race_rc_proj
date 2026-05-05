# Phase 3: Model A Traditional ML — Detailed Implementation Guide

**Marks at stake:** 15/100  
**Time estimate:** 2-3 days (training + hyperparameter tuning)  
**Deliverable:** Trained models, comparison table, confusion matrices in `notebooks/experiments.ipynb`

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Feature Engineering Pipeline](#2-feature-engineering-pipeline)
3. [Data Loading & Preprocessing](#3-data-loading--preprocessing)
4. [Model 1: Logistic Regression](#4-model-1-logistic-regression)
5. [Model 2: Support Vector Machine](#5-model-2-support-vector-machine)
6. [Hyperparameter Tuning Strategy](#6-hyperparameter-tuning-strategy)
7. [Evaluation & Metrics](#7-evaluation--metrics)
8. [File Structure & Outputs](#8-file-structure--outputs)
9. [Troubleshooting & Performance](#9-troubleshooting--performance)
10. [Code Template & Integration](#10-code-template--integration)

---

## 1. Overview & Architecture

### 1.1 Task Definition

**Model A Sub-task 1: Answer Verification (Binary Classification)**

```
Input:  article text + question text + option text → 1 feature vector
Output: probability (0 to 1) that this option is the correct answer

During inference:
  For each of the 4 options (A, B, C, D):
    - Create feature vector
    - Get probability from model
    - Pick the option with highest probability as the system's answer
```

**Expansion reminder:**
- Original RACE row: 1 question with 4 options → 1 label (A, B, C, or D)
- Training examples: 4 binary classification examples per original row
  - Each example: (concatenated text, binary label 0/1)
- ~281,168 training examples (4× expansion from ~70,292 original RACE rows)
- ~35,144 dev examples (~8,786 original questions × 4)
- Class distribution: 75% negative (wrong answer), 25% positive (correct answer)

### 1.2 Why These Models?

| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| **Logistic Regression** | Fast, interpretable, handles sparse data | Linear only | Baseline, quick experiments |
| **SVM (LinearSVC)** | Excellent margin separation, robust | Slower, no native probabilities | High-dimensional sparse text |
| **Random Forest** | Feature importance, non-linear | Slow on sparse data, memory-hungry | Understanding feature importance |
| **XGBoost** | Best performance often | Hyperparameter tuning complex | Production final model [BONUS] |

**For Phase 3, implement:** LR + SVM (required), optionally RF for feature importance

---

## 2. Feature Engineering Pipeline

### 2.1 Feature Composition

Your processed data from Phase 2 provides:

**Sparse Features (One-Hot Encoding):**
```
X_train_ohe.npz     → (281168, 10000) sparse matrix
  Each row: binary vector of length 10,000
  1 = word from vocabulary appears in concatenated text
  0 = word doesn't appear
  Vocabulary: top 10,000 most frequent words from training corpus
```

**Dense Features (Handcrafted Lexical):**
```
X_train_lexical.npy → (281168, 23) dense matrix
  
  Columns 0-8: Overlap & Length Features
    0. word_overlap_article_option    (0.0-1.0)
    1. word_overlap_question_option   (0.0-1.0)
    2. option_length                  (raw count)
    3. article_length                 (raw count)
    4. question_length                (raw count)
    5. option_position_in_article     (0 or 1, binary)
    6. char_length_ratio              (0.0+)
    7. unique_words_in_option         (raw count)
    8. [reserved]                     (not used or combined metric)
  
  Columns 9-20: Question Type Flags (Binary)
    9.  question_type_who             (0 or 1)
    10. question_type_what            (0 or 1)
    11. question_type_where           (0 or 1)
    12. question_type_when            (0 or 1)
    13. question_type_why             (0 or 1)
    14. question_type_how             (0 or 1)
    15. question_type_which           (0 or 1)
    16. question_type_other           (0 or 1)
    17. has_blank_in_question         (0 or 1)
    18. starts_with_aux_verb          (0 or 1)
    19. starts_with_article           (0 or 1)
    20. [additional structural flag]  (0 or 1)
  
  Columns 21-22: Semantic Features (New in Phase 2)
    21. cosine_similarity_within_question (0.0-1.0)
    22. option_diversity_variance         (0.0+)
```

### 2.2 Feature Scaling Strategy

**Why scale?**
- OHE vectors are naturally bounded (0 or 1)
- Lexical features have wildly different ranges:
  - Overlaps: 0.0-1.0
  - Lengths: 10-500 (raw counts)
  - Flags: 0 or 1
- When combined, a length feature (value 300) dominates an overlap feature (value 0.7)
- StandardScaler normalizes: `x_scaled = (x - mean) / std`

**Implementation:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# FIT ONLY ON TRAINING DATA
X_lex_train_scaled = scaler.fit_transform(X_lex_train)

# TRANSFORM dev and test (never refit)
X_lex_dev_scaled = scaler.transform(X_lex_dev)
X_lex_test_scaled = scaler.transform(X_lex_test)

# Save scaler for inference
joblib.dump(scaler, 'models/model_a/traditional/scaler_lexical.pkl')
```

**Before scaling (example single row):**
```
[0.5, 0.3, 245, 450, 15, 1, 0.54, 12, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0.87, 0.12]
```

**After scaling (same row):**
```
[0.3, -0.1, 0.8, 0.9, -0.2, 1.0, 0.2, 0.5, -0.5, 0.6, ..., 0.4, -0.1]
```

---

## 3. Data Loading & Preprocessing

### 3.1 Load All Splits

```python
import numpy as np
from scipy import sparse
import joblib

def load_data(data_dir='data/processed'):
    """Load preprocessed features, labels, row IDs, and vectorizer."""
    
    # Load OHE sparse matrices
    X_train_ohe = sparse.load_npz(f'{data_dir}/X_train_ohe.npz')
    X_dev_ohe = sparse.load_npz(f'{data_dir}/X_dev_ohe.npz')
    X_test_ohe = sparse.load_npz(f'{data_dir}/X_test_ohe.npz')
    
    # Load lexical dense matrices
    X_train_lex = np.load(f'{data_dir}/X_train_lexical.npy')
    X_dev_lex = np.load(f'{data_dir}/X_dev_lexical.npy')
    X_test_lex = np.load(f'{data_dir}/X_test_lexical.npy')
    
    # Load labels
    y_train = np.load(f'{data_dir}/y_train.npy')
    y_dev = np.load(f'{data_dir}/y_dev.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')
    
    # Load row IDs (for Exact Match computation)
    row_ids_train = np.load(f'{data_dir}/row_ids_train.npy')
    row_ids_dev = np.load(f'{data_dir}/row_ids_dev.npy')
    row_ids_test = np.load(f'{data_dir}/row_ids_test.npy')
    
    # Load vectorizer
    vectorizer = joblib.load(f'{data_dir}/../models/model_a/traditional/ohe_vectorizer.pkl')
    
    return {
        'X_train_ohe': X_train_ohe, 'X_dev_ohe': X_dev_ohe, 'X_test_ohe': X_test_ohe,
        'X_train_lex': X_train_lex, 'X_dev_lex': X_dev_lex, 'X_test_lex': X_test_lex,
        'y_train': y_train, 'y_dev': y_dev, 'y_test': y_test,
        'row_ids_train': row_ids_train, 'row_ids_dev': row_ids_dev, 'row_ids_test': row_ids_test,
        'vectorizer': vectorizer
    }
```

### 3.2 Sanity Checks

```python
# Verify data integrity
print(f"Train shape OHE: {X_train_ohe.shape}, Lex: {X_train_lex.shape}, Labels: {y_train.shape}")
print(f"Dev shape OHE: {X_dev_ohe.shape}, Lex: {X_dev_lex.shape}, Labels: {y_dev.shape}")

# Check label distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"Train label distribution: 0={counts[0]}, 1={counts[1]}")
print(f"  Imbalance ratio: {counts[0]/counts[1]:.2f}:1")

# Verify row_ids divisibility (should be groups of 4)
assert len(y_train) % 4 == 0, "Train examples not divisible by 4!"
print(f"✓ Data integrity checks passed")
```

---

## 4. Model 1: Logistic Regression

### 4.1 Why Logistic Regression?

- **Speed:** ~2-3 minutes on full training set (good for rapid iteration)
- **Interpretability:** Can extract top feature weights to understand predictions
- **Sparse-friendly:** Designed for high-dimensional data
- **Baseline:** Establishes a performance floor that more complex models should beat

### 4.2 Feature Combination for LR

```python
from scipy import sparse
from sklearn.preprocessing import StandardScaler

def prepare_features_for_lr(X_ohe, X_lex, is_training=False, scaler=None):
    """Combine OHE and lexical features for Logistic Regression."""
    
    # Step 1: Scale lexical features
    if is_training:
        scaler = StandardScaler()
        X_lex_scaled = scaler.fit_transform(X_lex)
    else:
        assert scaler is not None, "Must provide scaler for test data"
        X_lex_scaled = scaler.transform(X_lex)
    
    # Step 2: Convert lexical to sparse (for efficient storage and computation)
    X_lex_sparse = sparse.csr_matrix(X_lex_scaled)
    
    # Step 3: Horizontally stack
    X_combined = sparse.hstack([X_ohe, X_lex_sparse])
    
    return X_combined, scaler

# Usage
X_train_combined, scaler = prepare_features_for_lr(
    X_train_ohe, X_train_lex, 
    is_training=True
)
X_dev_combined, _ = prepare_features_for_lr(
    X_dev_ohe, X_dev_lex, 
    is_training=False, 
    scaler=scaler
)

print(f"Combined training feature shape: {X_train_combined.shape}")
# Output: (281168, 10023) — 10000 OHE + 23 lexical
```

### 4.3 Training Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
import time

def train_logistic_regression(X_train, y_train, C=1.0, verbose=True):
    """Train Logistic Regression with specified C parameter."""
    
    lr = LogisticRegression(
        C=C,                        # Inverse regularization strength
        max_iter=1000,              # Iterations for solver convergence
        solver='saga',              # Best for sparse data
        class_weight='balanced',    # Handle class imbalance (75% vs 25%)
        n_jobs=-1,                  # Use all CPU cores
        verbose=1 if verbose else 0,
        random_state=42
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training Logistic Regression (C={C})")
        print(f"{'='*60}")
        print(f"Training set size: {X_train.shape[0]:,} examples")
        print(f"Feature dimensions: {X_train.shape[1]:,}")
        print(f"Solver: saga | Max iterations: 1000 | Class weight: balanced")
        print(f"{'='*60}\n")
    
    start_time = time.time()
    lr.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    if verbose:
        print(f"✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return lr, training_time

# Train with default C=1.0
lr_model, train_time = train_logistic_regression(
    X_train_combined, y_train, 
    C=1.0, 
    verbose=True
)
```

### 4.4 Prediction & Probability Output

```python
# Get predictions
y_pred = lr_model.predict(X_dev_combined)

# Get probabilities (soft scores)
y_proba = lr_model.predict_proba(X_dev_combined)
# Shape: (35144, 2) — column 0 = P(y=0), column 1 = P(y=1)

# We care about P(correct answer)
y_score = y_proba[:, 1]

print(f"Sample predictions:")
print(f"  First 10 predictions: {y_pred[:10]}")
print(f"  First 10 probabilities: {y_score[:10]}")
print(f"  Prediction range: [{y_score.min():.4f}, {y_score.max():.4f}]")
```

---

## 5. Model 2: Support Vector Machine

### 5.1 Why LinearSVC?

- **High-dimensional advantage:** Finds the best separating hyperplane even in 10k+ dimensions
- **Theory:** Maximizes margin (distance between decision boundary and nearest points)
- **Text classification standard:** Proven effective for sparse bag-of-words features
- **Challenge:** Doesn't output probabilities natively → need calibration

### 5.2 SVM with Probability Calibration

```python
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import time

def train_svm(X_train, y_train, C=1.0, verbose=True):
    """Train LinearSVC with probability calibration."""
    
    # LinearSVC: the core SVM
    svc = LinearSVC(
        C=C,
        max_iter=2000,              # More iterations than LR
        loss='squared_hinge',       # Smooth loss for calibration
        class_weight='balanced',
        random_state=42,
        verbose=1 if verbose else 0
    )
    
    # CalibratedClassifierCV: wraps SVC to add probability output
    # Uses 5-fold cross-validation internally to calibrate probabilities
    svm = CalibratedClassifierCV(
        svc,
        cv=5,                       # 5-fold cross-validation for calibration
        method='sigmoid'            # Platt scaling for probability conversion
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training SVM with Probability Calibration (C={C})")
        print(f"{'='*60}")
        print(f"Training set size: {X_train.shape[0]:,} examples")
        print(f"Feature dimensions: {X_train.shape[1]:,}")
        print(f"SVM: LinearSVC | Calibration: CalibratedClassifierCV (5-fold)")
        print(f"Note: Training may take 15-30 minutes due to 5-fold calibration")
        print(f"{'='*60}\n")
    
    start_time = time.time()
    svm.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    if verbose:
        print(f"✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return svm, training_time

# Train SVM
svm_model, svm_train_time = train_svm(
    X_train_combined, y_train, 
    C=1.0, 
    verbose=True
)
```

**⚠️ Important:** SVM training with calibration is slower than LR (15-45 minutes). Plan accordingly.

### 5.3 Understanding LinearSVC Parameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `C` | 1.0 | Regularization strength (inverse). Lower = stronger regularization = simpler model |
| `max_iter` | 2000 | Optimization iterations. Increase if convergence warning appears |
| `loss` | 'squared_hinge' | Smooth hinge loss; better for calibration than standard hinge |
| `class_weight` | 'balanced' | Auto-adjust for 75% vs 25% label imbalance |
| `random_state` | 42 | Reproducibility seed |

**Calibration Details:**
- `cv=5`: Split training data 5 ways, train 4 folds, calibrate on the held-out fold
- `method='sigmoid'`: Fits a sigmoid curve to map SVM decision scores → probabilities [0,1]

---

## 6. Hyperparameter Tuning Strategy

### 6.1 Manual Grid Search (Fast Iteration)

```python
def manual_grid_search(X_train, y_train, X_dev, y_dev, model_type='lr'):
    """Quick hyperparameter search with manual iteration."""
    
    results = []
    
    if model_type == 'lr':
        param_values = {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    elif model_type == 'svm':
        param_values = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}
    
    print(f"\n{'='*70}")
    print(f"Manual Grid Search: {model_type.upper()}")
    print(f"{'='*70}\n")
    
    for param_name, param_list in param_values.items():
        for param_val in param_list:
            print(f"Testing {param_name}={param_val}...", end=" ", flush=True)
            
            if model_type == 'lr':
                model, _ = train_logistic_regression(X_train, y_train, C=param_val, verbose=False)
            else:  # svm
                model, _ = train_svm(X_train, y_train, C=param_val, verbose=False)
            
            # Evaluate
            y_pred = model.predict(X_dev)
            acc = (y_pred == y_dev).mean()
            results.append({
                'param_name': param_name,
                'param_value': param_val,
                'accuracy': acc,
                'model': model
            })
            print(f"Accuracy: {acc:.4f}")
    
    # Find best
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\n{'='*70}")
    print(f"✓ Best {param_name}: {best_result['param_value']} (Accuracy: {best_result['accuracy']:.4f})")
    print(f"{'='*70}\n")
    
    return results, best_result

# Usage
lr_results, lr_best = manual_grid_search(
    X_train_combined, y_train, 
    X_dev_combined, y_dev, 
    model_type='lr'
)
```

**Expected output:**
```
Manual Grid Search: LR
======================================================================

Testing C=0.001... Accuracy: 0.5234
Testing C=0.01... Accuracy: 0.5512
Testing C=0.1... Accuracy: 0.6015
Testing C=1.0... Accuracy: 0.6243
Testing C=10.0... Accuracy: 0.6189
Testing C=100.0... Accuracy: 0.6156

======================================================================
✓ Best C: 1.0 (Accuracy: 0.6243)
======================================================================
```

### 6.2 GridSearchCV for Automated Tuning [BONUS]

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def automated_grid_search(X_train, y_train, model_type='lr', cv_folds=5):
    """Automated hyperparameter search with cross-validation."""
    
    if model_type == 'lr':
        model = LogisticRegression(
            max_iter=1000, 
            solver='saga', 
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }
        scoring = 'f1_macro'  # Better for imbalanced data than accuracy
    
    else:  # svm
        svc = LinearSVC(
            max_iter=2000,
            loss='squared_hinge',
            class_weight='balanced',
            random_state=42
        )
        model = CalibratedClassifierCV(svc, cv=5, method='sigmoid')
        param_grid = {
            'estimator__C': [0.01, 0.1, 1.0, 10.0, 100.0]  # Note: estimator__C for wrapped model
        }
        scoring = 'f1_macro'
    
    gs = GridSearchCV(
        model,
        param_grid,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=-1,
        verbose=2
    )
    
    print(f"\n{'='*70}")
    print(f"Automated GridSearchCV: {model_type.upper()}")
    print(f"Parameters: {param_grid}")
    print(f"CV: {cv_folds}-fold | Scoring: {scoring}")
    print(f"{'='*70}\n")
    
    gs.fit(X_train, y_train)
    
    # Display results
    results_df = pd.DataFrame(gs.cv_results_)
    print(results_df[['param_C' if model_type == 'lr' else 'param_estimator__C', 
                      'mean_test_score', 'std_test_score']].to_string())
    
    print(f"\n{'='*70}")
    print(f"✓ Best params: {gs.best_params_}")
    print(f"✓ Best {scoring}: {gs.best_score_:.4f}")
    print(f"{'='*70}\n")
    
    return gs

# Usage
gs_result = automated_grid_search(X_train_combined, y_train, model_type='lr', cv_folds=5)
best_lr_model = gs_result.best_estimator_
```

---

## 7. Evaluation & Metrics

### 7.1 Binary-Level Metrics

```python
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
import matplotlib.pyplot as plt

def evaluate_binary_classification(y_true, y_pred, y_proba=None, model_name="Model"):
    """Compute and display all binary classification metrics."""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
    }
    
    print(f"\n{'='*60}")
    print(f"Binary Classification Metrics: {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Macro F1:   {metrics['macro_f1']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"{'='*60}\n")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:")
    print(f"  TN (True Neg):  {cm[0,0]:>6}  |  FP (False Pos): {cm[0,1]:>6}")
    print(f"  FN (False Neg): {cm[1,0]:>6}  |  TP (True Pos):  {cm[1,1]:>6}")
    print()
    
    # Classification report
    print(classification_report(y_true, y_pred, 
                               target_names=['Wrong Answer (0)', 'Correct Answer (1)'],
                               digits=4))
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=['Wrong (0)', 'Correct (1)'])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.show()
    
    return metrics, cm

# Usage
lr_metrics, lr_cm = evaluate_binary_classification(
    y_dev, 
    lr_pred, 
    y_proba=lr_proba,
    model_name="Logistic Regression"
)
```

**Sample output:**
```
============================================================
Binary Classification Metrics: Logistic Regression
============================================================
Accuracy:   0.6243
Macro F1:   0.6001
Precision:  0.5892
Recall:     0.6428
============================================================

Confusion Matrix:
  TN (True Neg):  26154  |  FP (False Pos):  1824
  FN (False Neg):  4084  |  TP (True Pos):   3082

              precision    recall  f1-score   support

Wrong Ans (0)     0.8646    0.9346    0.8982     27978
Correct Ans (1)   0.6282    0.4303    0.5106      7166

   accuracy                         0.6243     35144
   macro avg      0.7464    0.6824    0.7044     35144
weighted avg     0.8213    0.6243    0.7084     35144
```

### 7.2 Exact Match (EM) Metric

```python
def compute_exact_match(y_proba, row_ids, y_true):
    """
    Compute Exact Match: for each original question,
    pick the option with highest predicted probability.
    EM = fraction of questions where top-ranked option is correct.
    """
    
    n_questions = len(np.unique(row_ids))
    
    # Reshape into groups of 4 (one group per original question)
    y_proba_grouped = y_proba.reshape(n_questions, 4)
    y_true_grouped = y_true.reshape(n_questions, 4)
    
    # For each question, get the argmax (index of highest probability)
    pred_option_indices = np.argmax(y_proba_grouped, axis=1)  # (n_questions,)
    true_option_indices = np.argmax(y_true_grouped, axis=1)   # (n_questions,)
    
    # EM = fraction where predicted matches true
    em = (pred_option_indices == true_option_indices).mean()
    
    # Per-position accuracy
    position_acc = {}
    for pos in range(4):
        mask = (true_option_indices == pos)
        if mask.sum() > 0:
            position_acc[pos] = (pred_option_indices[mask] == pos).mean()
    
    print(f"\nExact Match (EM) Analysis:")
    print(f"{'='*60}")
    print(f"Overall EM: {em:.4f} ({int(em * n_questions):,} / {n_questions:,} questions correct)")
    print(f"\nPer-position accuracy:")
    for pos, acc in position_acc.items():
        print(f"  Option {chr(65+pos)}: {acc:.4f}")
    print(f"{'='*60}\n")
    
    return em, position_acc

# Usage
lr_em, lr_pos_acc = compute_exact_match(
    lr_proba[:, 1],  # Probabilities for positive class
    row_ids_dev,
    y_dev
)
```

**Sample output:**
```
Exact Match (EM) Analysis:
============================================================
Overall EM: 0.6024 (5,298 / 8,786 questions correct)

Per-position accuracy:
  Option A: 0.5892
  Option B: 0.6145
  Option C: 0.6134
  Option D: 0.5987
============================================================
```

---

## 8. File Structure & Outputs

### 8.1 Directory Layout After Phase 3

```
race_rc_project/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   └── test.csv
│   └── processed/
│       ├── X_train_ohe.npz
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
│
├── models/
│   └── model_a/
│       └── traditional/
│           ├── ohe_vectorizer.pkl
│           ├── scaler_lexical.pkl                    ← NEW
│           ├── logistic_regression.pkl               ← NEW
│           ├── svm_calibrated.pkl                    ← NEW
│           ├── random_forest.pkl                     ← NEW [BONUS]
│           └── model_a_metrics.json                  ← NEW
│
├── src/
│   ├── preprocessing.py              (existing)
│   ├── model_a_train.py              ← NEW (main file)
│   ├── evaluate.py                   (update with new metrics)
│   └── inference.py
│
└── notebooks/
    └── experiments.ipynb              ← NEW (run training, show results)
```

### 8.2 Model Checkpoint Saving

```python
import json
from datetime import datetime

def save_model_and_artifacts(model, scaler, metrics, model_name, output_dir='models/model_a/traditional'):
    """Save trained model, scaler, and metadata."""
    
    # Save model
    model_path = f"{output_dir}/{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"✓ Saved model: {model_path}")
    
    # Save scaler (if provided)
    if scaler is not None:
        scaler_path = f"{output_dir}/scaler_lexical.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"✓ Saved scaler: {scaler_path}")
    
    # Save metrics to JSON
    metrics_path = f"{output_dir}/{model_name}_metrics.json"
    metrics_with_metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'metrics': metrics
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_with_metadata, f, indent=2)
    print(f"✓ Saved metrics: {metrics_path}\n")
    
    return model_path, scaler_path, metrics_path

# Usage
save_model_and_artifacts(
    lr_model, 
    scaler, 
    {
        'accuracy': lr_metrics['accuracy'],
        'macro_f1': lr_metrics['macro_f1'],
        'precision': lr_metrics['precision'],
        'recall': lr_metrics['recall'],
        'exact_match': lr_em,
        'training_time_seconds': train_time
    },
    model_name='logistic_regression'
)
```

### 8.3 Metrics JSON Structure

```json
{
  "timestamp": "2026-05-06T14:35:22.123456",
  "model_name": "logistic_regression",
  "metrics": {
    "accuracy": 0.6243,
    "macro_f1": 0.6001,
    "precision": 0.5892,
    "recall": 0.6428,
    "exact_match": 0.6024,
    "training_time_seconds": 142.3,
    "hyperparameters": {
      "C": 1.0,
      "solver": "saga",
      "max_iter": 1000,
      "class_weight": "balanced"
    }
  }
}
```

---

## 9. Troubleshooting & Performance

### 9.1 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| MemoryError when combining OHE + lexical | Sparse matrices converted to dense | Keep as sparse: use `sparse.hstack()` instead of `np.hstack()` |
| Training takes too long (>1 hour) | Too many iterations or data too large | Reduce `max_iter` to 500, or subsample training data for initial experiments |
| Accuracy doesn't improve with tuning | Model hitting data quality ceiling | Features may be insufficient; focus on next phase (ensemble/unsupervised) |
| SVM calibration is very slow | CalibratedClassifierCV uses 5-fold CV | Reduce to `cv=3` for quick experiments, then increase for final model |
| Predictions all near 0.5 probability | Model uncertain; features not discriminative | Check feature engineering; ensure OHE vocabulary is appropriate |
| Very imbalanced predictions (mostly 0s) | Threshold isn't optimal at 0.5 | Analyze precision-recall curve; try custom threshold |

### 9.2 Performance Profiling

```python
import time

def profile_model_pipeline(X_train, y_train, X_dev, y_dev, model_type='lr'):
    """Measure timing of each phase."""
    
    timings = {}
    
    # Training time
    start = time.time()
    if model_type == 'lr':
        model, train_time = train_logistic_regression(X_train, y_train, verbose=False)
    else:
        model, train_time = train_svm(X_train, y_train, verbose=False)
    timings['training'] = train_time
    
    # Prediction time (dev set)
    start = time.time()
    y_pred = model.predict(X_dev)
    timings['predict_binary'] = time.time() - start
    
    # Probability time
    start = time.time()
    y_proba = model.predict_proba(X_dev)
    timings['predict_proba'] = time.time() - start
    
    # EM computation
    start = time.time()
    em, _ = compute_exact_match(y_proba[:, 1], row_ids_dev, y_dev)
    timings['exact_match'] = time.time() - start
    
    print(f"\nTiming Analysis: {model_type.upper()}")
    print(f"{'='*60}")
    for phase, time_seconds in timings.items():
        print(f"  {phase:20s}: {time_seconds:8.2f} sec ({time_seconds/60:6.2f} min)")
    print(f"{'='*60}\n")
    
    return timings

# Usage
lr_timings = profile_model_pipeline(X_train_combined, y_train, X_dev_combined, y_dev, 'lr')
svm_timings = profile_model_pipeline(X_train_combined, y_train, X_dev_combined, y_dev, 'svm')
```

**Expected output:**
```
Timing Analysis: LR
============================================================
  training             :     142.34 sec (    2.37 min)
  predict_binary       :       3.21 sec (    0.05 min)
  predict_proba        :       3.45 sec (    0.06 min)
  exact_match          :       0.12 sec (    0.00 min)
============================================================

Timing Analysis: SVM
============================================================
  training             :    1847.23 sec (   30.79 min)
  predict_binary       :       8.92 sec (    0.15 min)
  predict_proba        :       9.15 sec (    0.15 min)
  exact_match          :       0.12 sec (    0.00 min)
============================================================
```

### 9.3 Memory Usage Optimization

```python
def estimate_memory_usage(X_ohe, X_lex):
    """Estimate memory for combined features."""
    
    ohe_sparse_bytes = X_ohe.data.nbytes + X_ohe.indices.nbytes + X_ohe.indptr.nbytes
    lex_bytes = X_lex.nbytes
    combined_estimate = ohe_sparse_bytes + lex_bytes
    
    print(f"Memory Estimation:")
    print(f"  OHE sparse:        {ohe_sparse_bytes / 1e9:.2f} GB")
    print(f"  Lexical dense:     {lex_bytes / 1e9:.2f} GB")
    print(f"  Combined (est):    {combined_estimate / 1e9:.2f} GB")
    
    # Rule of thumb: model training requires ~5-10× feature matrix size
    print(f"  Model training ~:  {(combined_estimate * 5) / 1e9:.2f} GB")

estimate_memory_usage(X_train_ohe, X_train_lex)
```

---

## 10. Code Template & Integration

### 10.1 Complete model_a_train.py Structure

```python
"""
Phase 3: Model A Traditional ML Training

Usage:
  python src/model_a_train.py --model lr --mode train
  python src/model_a_train.py --model svm --mode train
  python src/model_a_train.py --model all --mode train --tune
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_data(data_dir='data/processed'):
    """Load preprocessed features and labels."""
    pass  # (See section 3.1 for implementation)

def prepare_features(X_ohe, X_lex, is_training=False, scaler=None):
    """Combine and scale features."""
    pass  # (See section 4.2 for implementation)

# ============================================================================
# 2. MODEL TRAINING
# ============================================================================

def train_logistic_regression(X_train, y_train, C=1.0, verbose=True):
    """Train Logistic Regression."""
    pass  # (See section 4.3 for implementation)

def train_svm(X_train, y_train, C=1.0, verbose=True):
    """Train SVM with calibration."""
    pass  # (See section 5.2 for implementation)

def train_random_forest(X_train, y_train, n_estimators=100, verbose=True):
    """Train Random Forest."""
    pass  # (Implementation for bonus)

# ============================================================================
# 3. EVALUATION
# ============================================================================

def evaluate_model(y_true, y_pred, y_proba, row_ids, model_name):
    """Compute all evaluation metrics."""
    pass  # (See section 7 for implementation)

def compute_exact_match(y_proba, row_ids, y_true):
    """Compute Exact Match metric."""
    pass  # (See section 7.2 for implementation)

# ============================================================================
# 4. HYPERPARAMETER TUNING
# ============================================================================

def tune_hyperparameters(X_train, y_train, X_dev, y_dev, model_type='lr'):
    """Grid search over hyperparameters."""
    pass  # (See section 6 for implementation)

# ============================================================================
# 5. SAVING & REPORTING
# ============================================================================

def save_artifacts(model, scaler, metrics, model_name):
    """Save trained model and results."""
    pass  # (See section 8.2 for implementation)

def create_comparison_table(results_dict):
    """Create comparison table of all models."""
    df = pd.DataFrame(results_dict).T
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(df.to_string())
    print("="*80 + "\n")
    return df

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main(args):
    """Main training pipeline."""
    
    # Load data
    print("Loading data...")
    data = load_data()
    X_train_ohe = data['X_train_ohe']
    # ... (rest of loading)
    
    # Prepare features
    X_train, scaler = prepare_features(X_train_ohe, X_train_lex, is_training=True)
    X_dev, _ = prepare_features(X_dev_ohe, X_dev_lex, is_training=False, scaler=scaler)
    
    results = {}
    
    # Train selected models
    if args.model in ['lr', 'all']:
        print("\n" + "="*80)
        print("LOGISTIC REGRESSION")
        print("="*80)
        
        if args.tune:
            lr_model = tune_hyperparameters(X_train, y_train, X_dev, y_dev, 'lr')
        else:
            lr_model, _ = train_logistic_regression(X_train, y_train, verbose=True)
        
        lr_results = evaluate_model(y_dev, lr_pred, lr_proba, row_ids_dev, "LR")
        results['Logistic Regression'] = lr_results
        save_artifacts(lr_model, scaler, lr_results, 'logistic_regression')
    
    if args.model in ['svm', 'all']:
        print("\n" + "="*80)
        print("SUPPORT VECTOR MACHINE")
        print("="*80)
        
        if args.tune:
            svm_model = tune_hyperparameters(X_train, y_train, X_dev, y_dev, 'svm')
        else:
            svm_model, _ = train_svm(X_train, y_train, verbose=True)
        
        svm_results = evaluate_model(y_dev, svm_pred, svm_proba, row_ids_dev, "SVM")
        results['SVM'] = svm_results
        save_artifacts(svm_model, scaler, svm_results, 'svm_calibrated')
    
    # Comparison
    if len(results) > 1:
        create_comparison_table(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 3: Model A Training')
    parser.add_argument('--model', choices=['lr', 'svm', 'all'], default='lr',
                       help='Which model(s) to train')
    parser.add_argument('--tune', action='store_true',
                       help='Run hyperparameter tuning')
    parser.add_argument('--data-dir', default='data/processed',
                       help='Path to processed data directory')
    args = parser.parse_args()
    
    main(args)
```

### 10.2 Running from Jupyter Notebook

```python
# In notebooks/experiments.ipynb

%load_ext autoreload
%autoreload 2

# Add src to path
import sys
sys.path.insert(0, 'src')

# Import training functions
from model_a_train import (
    load_data, prepare_features, 
    train_logistic_regression, train_svm,
    evaluate_model, create_comparison_table
)

# Load data
data = load_data()
X_train, X_dev, X_test = data['X_train_ohe'], data['X_dev_ohe'], data['X_test_ohe']
X_lex_train, X_lex_dev = data['X_train_lex'], data['X_dev_lex']
y_train, y_dev = data['y_train'], data['y_dev']

# Prepare features
X_train_combined, scaler = prepare_features(
    X_train, X_lex_train, is_training=True
)
X_dev_combined, _ = prepare_features(
    X_dev, X_lex_dev, is_training=False, scaler=scaler
)

# Train LR
lr_model, lr_time = train_logistic_regression(X_train_combined, y_train, verbose=True)
lr_pred = lr_model.predict(X_dev_combined)
lr_proba = lr_model.predict_proba(X_dev_combined)

# Evaluate
lr_results = evaluate_model(y_dev, lr_pred, lr_proba, row_ids_dev, "LR")
```

---

## Summary Checklist

- [ ] **Understand data:** 281k train examples, 23 features per example
- [ ] **Feature preparation:** OHE + lexical, scale lexical, combine with sparse.hstack()
- [ ] **Implement LR:** Train, predict, get probabilities
- [ ] **Implement SVM:** Train with calibration, handle 30-minute training time
- [ ] **Hyperparameter tuning:** Test C values [0.001, 0.01, 0.1, 1, 10, 100]
- [ ] **Evaluate:** Binary metrics + Exact Match metric
- [ ] **Save models:** .pkl files + JSON metrics
- [ ] **Compare:** Create comparison table in notebook
- [ ] **Report results:** ~62% accuracy, ~60% EM expected

---

**Next:** Once you have LR + SVM results, move to Phase 4 (Unsupervised/Semi-Supervised learning with K-Means, Label Propagation, GMM).
