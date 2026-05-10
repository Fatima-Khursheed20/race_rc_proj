"""
Phase 3: Model A Traditional ML Training

This script trains and evaluates Logistic Regression and SVM models for
binary answer verification on the RACE dataset.

Usage Examples:
  # Quick baseline LR (2-3 minutes)
  python src/model_a_train.py --model lr
  
  # SVM with calibration (30-45 minutes)
  python src/model_a_train.py --model svm
  
  # Both models with hyperparameter tuning
  python src/model_a_train.py --model all --tune
  
  # LR with manual grid search for C parameter
  python src/model_a_train.py --model lr --tune

Output:
  - Trained models saved to: models/model_a/traditional/
  - Metrics saved to: models/model_a/traditional/*_metrics.json
  - Results summary printed to console
"""

import argparse
import json
import time
import sys
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = 'data/processed'
MODEL_DIR = 'models/model_a/traditional'
RANDOM_STATE = 42

# Ensure model directory exists
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_data(data_dir=DATA_DIR, verbose=True):
    """
    Load preprocessed features, labels, and row IDs from Phase 2.
    
    Returns:
        dict: Contains OHE matrices, lexical features, labels, row_ids, vectorizer
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"LOADING DATA")
        print(f"{'='*70}\n")
    
    try:
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
        vectorizer = joblib.load(f'{MODEL_DIR}/ohe_vectorizer.pkl')
        
        # Verify data integrity
        if verbose:
            print(f"Train OHE shape:     {X_train_ohe.shape}")
            print(f"Train Lexical shape: {X_train_lex.shape}")
            print(f"Train labels shape:  {y_train.shape}")
            print(f"Train row_ids shape: {row_ids_train.shape}\n")
            
            print(f"Dev OHE shape:       {X_dev_ohe.shape}")
            print(f"Dev labels shape:    {y_dev.shape}\n")
            
            print(f"Test OHE shape:      {X_test_ohe.shape}")
            print(f"Test labels shape:   {y_test.shape}\n")
            
            # Check label distribution
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"Train label distribution:")
            print(f"  Label 0 (wrong):    {counts[0]:>10,} ({counts[0]/len(y_train)*100:>5.2f}%)")
            print(f"  Label 1 (correct):  {counts[1]:>10,} ({counts[1]/len(y_train)*100:>5.2f}%)")
            print(f"  Imbalance ratio:    {counts[0]/counts[1]:>10.2f}:1\n")
            
            # Verify row_ids divisibility
            assert len(y_train) % 4 == 0, "Train examples not divisible by 4!"
            n_questions = len(np.unique(row_ids_train))
            print(f"Train questions:     {n_questions:>10,}")
            print(f"Train examples (4×): {len(y_train):>10,}\n")
            print(f"✓ Data integrity checks passed")
            print(f"{'='*70}\n")
        
        return {
            'X_train_ohe': X_train_ohe, 'X_dev_ohe': X_dev_ohe, 'X_test_ohe': X_test_ohe,
            'X_train_lex': X_train_lex, 'X_dev_lex': X_dev_lex, 'X_test_lex': X_test_lex,
            'y_train': y_train, 'y_dev': y_dev, 'y_test': y_test,
            'row_ids_train': row_ids_train, 'row_ids_dev': row_ids_dev, 'row_ids_test': row_ids_test,
            'vectorizer': vectorizer
        }
    
    except FileNotFoundError as e:
        print(f"ERROR: Could not load data. Make sure Phase 2 preprocessing is complete.")
        print(f"Missing file: {e}")
        sys.exit(1)


# ============================================================================
# 2. FEATURE PREPARATION
# ============================================================================

def prepare_features(X_ohe, X_lex, is_training=False, scaler=None, verbose=False):
    """
    Combine OHE and lexical features for model training.
    
    Steps:
      1. Scale lexical features (StandardScaler)
      2. Convert to sparse matrix
      3. Horizontally stack with OHE
    
    Args:
        X_ohe: Sparse OHE matrix
        X_lex: Dense lexical feature matrix
        is_training: If True, fit scaler; if False, only transform
        scaler: Fitted scaler (required if is_training=False)
        verbose: Print debug info
    
    Returns:
        tuple: (combined_sparse_matrix, fitted_scaler)
    """
    
    # Step 1: Scale lexical features
    if is_training:
        scaler = StandardScaler()
        X_lex_scaled = scaler.fit_transform(X_lex)
        if verbose:
            print(f"Fitted StandardScaler on lexical features")
    else:
        if scaler is None:
            raise ValueError("Must provide scaler for non-training data")
        X_lex_scaled = scaler.transform(X_lex)
    
    # Step 2: Convert to sparse
    X_lex_sparse = sparse.csr_matrix(X_lex_scaled)
    
    # Step 3: Stack horizontally
    X_combined = sparse.hstack([X_ohe, X_lex_sparse])
    
    if verbose:
        print(f"Combined feature shape: {X_combined.shape}")
        print(f"  OHE: {X_ohe.shape[1]:,} dims")
        print(f"  Lexical (scaled): {X_lex.shape[1]} dims")
        print(f"  Total: {X_combined.shape[1]:,} dims\n")
    
    return X_combined, scaler


# ============================================================================
# 3. MODEL TRAINING
# ============================================================================

def train_logistic_regression(X_train, y_train, C=1.0, verbose=True):
    """
    Train Logistic Regression model for answer verification.
    
    Args:
        X_train: Combined feature matrix (sparse)
        y_train: Binary labels (0=wrong, 1=correct)
        C: Regularization parameter (inverse strength). Lower = stronger regularization
        verbose: Print training info
    
    Returns:
        tuple: (trained_model, training_time_seconds)
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING LOGISTIC REGRESSION")
        print(f"{'='*70}")
        print(f"C (regularization): {C}")
        print(f"Training set size: {X_train.shape[0]:,} examples")
        print(f"Features: {X_train.shape[1]:,} dimensions")
        print(f"Solver: saga | Max iterations: 1000 | Class weight: balanced")
        print(f"{'='*70}\n")
    
    lr = LogisticRegression(
        C=C,
        max_iter=1000,
        solver='saga',              # Good for sparse data
        class_weight='balanced',    # Handle 75% vs 25% imbalance
        n_jobs=-1,                  # Use all CPU cores
        verbose=1 if verbose else 0,
        random_state=RANDOM_STATE
    )
    
    start_time = time.time()
    lr.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    if verbose:
        print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n")
    
    return lr, training_time


def train_svm(X_train, y_train, C=1.0, verbose=True):
    """
    Train SVM with probability calibration for answer verification.
    
    Notes:
      - LinearSVC doesn't output probabilities natively
      - CalibratedClassifierCV adds calibration using 5-fold CV
      - This makes training slower (~30-45 min) but gives probability estimates
    
    Args:
        X_train: Combined feature matrix (sparse)
        y_train: Binary labels (0=wrong, 1=correct)
        C: Regularization parameter
        verbose: Print training info
    
    Returns:
        tuple: (trained_model, training_time_seconds)
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING SUPPORT VECTOR MACHINE")
        print(f"{'='*70}")
        print(f"C (regularization): {C}")
        print(f"Training set size: {X_train.shape[0]:,} examples")
        print(f"Features: {X_train.shape[1]:,} dimensions")
        print(f"Base model: LinearSVC")
        print(f"Calibration: CalibratedClassifierCV (5-fold)")
        print(f"⚠️  WARNING: This may take 30-45 minutes")
        print(f"{'='*70}\n")
    
    # LinearSVC (core SVM)
    svc = LinearSVC(
        C=C,
        max_iter=2000,
        loss='squared_hinge',       # Smooth loss for calibration
        class_weight='balanced',
        random_state=RANDOM_STATE,
        verbose=1 if verbose else 0
    )
    
    # Calibration wrapper
    svm = CalibratedClassifierCV(
        svc,
        cv=5,                       # 5-fold CV for calibration
        method='sigmoid'            # Platt scaling
    )
    
    start_time = time.time()
    svm.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    if verbose:
        print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n")
    
    return svm, training_time


# ============================================================================
# 4. EVALUATION
# ============================================================================

def compute_exact_match(y_proba, row_ids, y_true):
    """
    Compute Exact Match metric for question-level evaluation.
    
    For each original question (4 options):
      1. Get probability for each option
      2. Pick option with highest probability
      3. Check if it matches the ground truth
    
    Args:
        y_proba: Probabilities for positive class (shape: n_examples)
        row_ids: Original question IDs (shape: n_examples)
        y_true: True labels (shape: n_examples)
    
    Returns:
        tuple: (em_score, position_accuracies_dict)
    """
    
    n_questions = len(np.unique(row_ids))
    
    # Reshape into groups of 4
    y_proba_grouped = y_proba.reshape(n_questions, 4)
    y_true_grouped = y_true.reshape(n_questions, 4)
    
    # Argmax for each question
    pred_option_indices = np.argmax(y_proba_grouped, axis=1)
    true_option_indices = np.argmax(y_true_grouped, axis=1)
    
    # EM = fraction correct
    em = (pred_option_indices == true_option_indices).mean()
    
    # Per-position accuracy
    position_acc = {}
    for pos in range(4):
        mask = (true_option_indices == pos)
        if mask.sum() > 0:
            position_acc[pos] = (pred_option_indices[mask] == pos).mean()
    
    return em, position_acc


def evaluate_model(y_true, y_pred, y_proba, row_ids, model_name="Model", verbose=True):
    """
    Compute comprehensive evaluation metrics for binary classification.
    
    Metrics computed:
      - Accuracy, Macro F1, Precision, Recall
      - Confusion Matrix
      - Classification Report
      - Exact Match (question-level)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_proba: Predicted probabilities for positive class
        row_ids: Original question IDs (for EM computation)
        model_name: Name for output messages
        verbose: Print results
    
    Returns:
        dict: Dictionary with all metrics
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATION: {model_name}")
        print(f"{'='*70}\n")
    
    # Binary classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    if verbose:
        print(f"Accuracy:   {accuracy:.4f}")
        print(f"Macro F1:   {macro_f1:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}\n")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if verbose:
        print(f"Confusion Matrix:")
        print(f"  TN (True Neg):  {cm[0,0]:>8,}  |  FP (False Pos): {cm[0,1]:>8,}")
        print(f"  FN (False Neg): {cm[1,0]:>8,}  |  TP (True Pos):  {cm[1,1]:>8,}\n")
    
    # Classification report
    if verbose:
        print("Detailed Classification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Wrong Answer (0)', 'Correct Answer (1)'],
                                   digits=4))
    
    # Exact Match
    em, position_acc = compute_exact_match(y_proba, row_ids, y_true)
    
    if verbose:
        print(f"Exact Match (EM):    {em:.4f}")
        print(f"  Questions: {int(em * len(np.unique(row_ids))):,} / {len(np.unique(row_ids)):,} correct")
        print(f"\nPer-position accuracy (A/B/C/D):")
        for pos, acc in sorted(position_acc.items()):
            print(f"  Option {chr(65+pos)}: {acc:.4f}")
        print(f"{'='*70}\n")
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall,
        'exact_match': em,
        'position_accuracies': position_acc,
        'confusion_matrix': cm.tolist()
    }


# ============================================================================
# 5. HYPERPARAMETER TUNING
# ============================================================================

def tune_logistic_regression(X_train, y_train, X_dev, y_dev, row_ids_dev):
    """
    Manual grid search for Logistic Regression C parameter.
    
    Tests C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    Returns:
        dict: Best model, C value, and metrics
    """
    
    c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    results = []
    
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER TUNING: Logistic Regression")
    print(f"{'='*70}\n")
    print(f"{'C':<10} {'Accuracy':<15} {'Macro F1':<15} {'EM':<15} {'Time (sec)':<15}")
    print(f"{'-'*70}")
    
    for c in c_values:
        model, train_time = train_logistic_regression(X_train, y_train, C=c, verbose=False)
        
        y_pred = model.predict(X_dev)
        y_proba = model.predict_proba(X_dev)[:, 1]
        
        metrics = evaluate_model(y_dev, y_pred, y_proba, row_ids_dev, 
                                 model_name=f"LR (C={c})", verbose=False)
        
        print(f"{c:<10} {metrics['accuracy']:<15.4f} {metrics['macro_f1']:<15.4f} "
              f"{metrics['exact_match']:<15.4f} {train_time:<15.2f}")
        
        results.append({
            'C': c,
            'model': model,
            'metrics': metrics,
            'train_time': train_time
        })
    
    # Find best by accuracy
    best = max(results, key=lambda x: x['metrics']['accuracy'])
    print(f"\n{'='*70}")
    print(f"✓ Best C: {best['C']} (Accuracy: {best['metrics']['accuracy']:.4f})")
    print(f"{'='*70}\n")
    
    return best


def tune_svm(X_train, y_train, X_dev, y_dev, row_ids_dev):
    """
    Manual grid search for SVM C parameter.
    
    Tests C in [0.01, 0.1, 1.0, 10.0, 100.0]
    ⚠️  WARNING: Very time-consuming (2-3 hours total for all C values)
    
    Returns:
        dict: Best model, C value, and metrics
    """
    
    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    results = []
    
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER TUNING: SVM")
    print(f"{'='*70}")
    print(f"⚠️  WARNING: This will take ~30-45 minutes per C value\n")
    print(f"{'C':<10} {'Accuracy':<15} {'Macro F1':<15} {'EM':<15} {'Time (sec)':<15}")
    print(f"{'-'*70}")
    
    for c in c_values:
        model, train_time = train_svm(X_train, y_train, C=c, verbose=False)
        
        y_pred = model.predict(X_dev)
        y_proba = model.predict_proba(X_dev)[:, 1]
        
        metrics = evaluate_model(y_dev, y_pred, y_proba, row_ids_dev, 
                                 model_name=f"SVM (C={c})", verbose=False)
        
        print(f"{c:<10} {metrics['accuracy']:<15.4f} {metrics['macro_f1']:<15.4f} "
              f"{metrics['exact_match']:<15.4f} {train_time:<15.2f}")
        
        results.append({
            'C': c,
            'model': model,
            'metrics': metrics,
            'train_time': train_time
        })
    
    # Find best by accuracy
    best = max(results, key=lambda x: x['metrics']['accuracy'])
    print(f"\n{'='*70}")
    print(f"✓ Best C: {best['C']} (Accuracy: {best['metrics']['accuracy']:.4f})")
    print(f"{'='*70}\n")
    
    return best


# ============================================================================
# 6. MODEL SAVING
# ============================================================================

def save_model_and_metrics(model, scaler, metrics, model_name, model_type='lr'):
    """
    Save trained model, scaler, and metrics.
    
    Args:
        model: Trained model object
        scaler: Fitted StandardScaler
        metrics: Dictionary of evaluation metrics
        model_name: Name for saved files (e.g., 'logistic_regression')
        model_type: 'lr' or 'svm' (for naming conventions)
    """
    
    # Save model
    model_path = f"{MODEL_DIR}/{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"✓ Saved model: {model_path}")
    
    # Save scaler (only once, shared across models)
    scaler_path = f"{MODEL_DIR}/scaler_lexical.pkl"
    if not Path(scaler_path).exists():
        joblib.dump(scaler, scaler_path)
        print(f"✓ Saved scaler: {scaler_path}")
    
    # Save metrics
    metrics_path = f"{MODEL_DIR}/{model_name}_metrics.json"
    metrics_with_metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'model_type': model_type,
        'metrics': metrics
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_with_metadata, f, indent=2)
    print(f"✓ Saved metrics: {metrics_path}\n")


# ============================================================================
# 7. RESULTS COMPARISON
# ============================================================================

def create_comparison_table(results_dict):
    """
    Create and display comparison table of all trained models.
    
    Args:
        results_dict: Dictionary mapping model names to their metrics
    """
    
    df_data = {}
    for model_name, metrics in results_dict.items():
        df_data[model_name] = {
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Macro F1': f"{metrics['macro_f1']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'Exact Match': f"{metrics['exact_match']:.4f}",
        }
    
    df = pd.DataFrame(df_data).T
    
    print(f"\n{'='*80}")
    print(f"PHASE 3: MODEL COMPARISON")
    print(f"{'='*80}")
    print(df.to_string())
    print(f"{'='*80}\n")
    
    return df


# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

def main(args):
    """
    Main training and evaluation pipeline.
    """
    
    # Load data
    data = load_data(verbose=True)
    X_train_ohe = data['X_train_ohe']
    X_dev_ohe = data['X_dev_ohe']
    X_test_ohe = data['X_test_ohe']
    X_train_lex = data['X_train_lex']
    X_dev_lex = data['X_dev_lex']
    X_test_lex = data['X_test_lex']
    y_train = data['y_train']
    y_dev = data['y_dev']
    y_test = data['y_test']
    row_ids_train = data['row_ids_train']
    row_ids_dev = data['row_ids_dev']
    row_ids_test = data['row_ids_test']
    
    # Prepare features
    print(f"\n{'='*70}")
    print(f"PREPARING FEATURES")
    print(f"{'='*70}\n")
    
    X_train_combined, scaler = prepare_features(
        X_train_ohe, X_train_lex, 
        is_training=True, 
        verbose=True
    )
    X_dev_combined, _ = prepare_features(
        X_dev_ohe, X_dev_lex, 
        is_training=False, 
        scaler=scaler,
        verbose=True
    )
    X_test_combined, _ = prepare_features(
        X_test_ohe, X_test_lex, 
        is_training=False, 
        scaler=scaler,
        verbose=False
    )
    
    results = {}
    
    # Train Logistic Regression
    if args.model in ['lr', 'all']:
        if args.tune:
            best_lr = tune_logistic_regression(X_train_combined, y_train, 
                                               X_dev_combined, y_dev, row_ids_dev)
            lr_model = best_lr['model']
            lr_c = best_lr['C']
        else:
            lr_model, _ = train_logistic_regression(X_train_combined, y_train, 
                                                    C=1.0, verbose=True)
            lr_c = 1.0
        
        # Evaluate on dev
        y_dev_pred_lr = lr_model.predict(X_dev_combined)
        y_dev_proba_lr = lr_model.predict_proba(X_dev_combined)[:, 1]
        lr_metrics_dev = evaluate_model(y_dev, y_dev_pred_lr, y_dev_proba_lr, 
                                        row_ids_dev, "Logistic Regression (Dev)", verbose=True)
        
        # Evaluate on test
        y_test_pred_lr = lr_model.predict(X_test_combined)
        y_test_proba_lr = lr_model.predict_proba(X_test_combined)[:, 1]
        lr_metrics_test = evaluate_model(y_test, y_test_pred_lr, y_test_proba_lr, 
                                         row_ids_test, "Logistic Regression (Test)", verbose=True)
        
        # Save
        save_model_and_metrics(lr_model, scaler, lr_metrics_dev, 
                               'logistic_regression', model_type='lr')
        
        results['Logistic Regression'] = lr_metrics_dev
    
    # Train SVM
    if args.model in ['svm', 'all']:
        if args.tune:
            best_svm = tune_svm(X_train_combined, y_train, 
                               X_dev_combined, y_dev, row_ids_dev)
            svm_model = best_svm['model']
            svm_c = best_svm['C']
        else:
            svm_model, _ = train_svm(X_train_combined, y_train, 
                                     C=1.0, verbose=True)
            svm_c = 1.0
        
        # Evaluate on dev
        y_dev_pred_svm = svm_model.predict(X_dev_combined)
        y_dev_proba_svm = svm_model.predict_proba(X_dev_combined)[:, 1]
        svm_metrics_dev = evaluate_model(y_dev, y_dev_pred_svm, y_dev_proba_svm, 
                                         row_ids_dev, "SVM (Dev)", verbose=True)
        
        # Evaluate on test
        y_test_pred_svm = svm_model.predict(X_test_combined)
        y_test_proba_svm = svm_model.predict_proba(X_test_combined)[:, 1]
        svm_metrics_test = evaluate_model(y_test, y_test_pred_svm, y_test_proba_svm, 
                                          row_ids_test, "SVM (Test)", verbose=True)
        
        # Save
        save_model_and_metrics(svm_model, scaler, svm_metrics_dev, 
                               'svm_calibrated', model_type='svm')
        
        results['SVM'] = svm_metrics_dev
    
    # Comparison table
    if len(results) > 1:
        create_comparison_table(results)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description='Phase 3: Model A Traditional ML Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick Logistic Regression (2-3 minutes)
  python src/model_a_train.py --model lr
  
  # SVM with calibration (30-45 minutes)
  python src/model_a_train.py --model svm
  
  # Both models
  python src/model_a_train.py --model all
  
  # LR with hyperparameter tuning (test C values)
  python src/model_a_train.py --model lr --tune
  
  # Both with tuning (⚠️ WARNING: 2-3 hours)
  python src/model_a_train.py --model all --tune
        """
    )
    
    parser.add_argument(
        '--model',
        choices=['lr', 'svm', 'all'],
        default='lr',
        help='Which model(s) to train (default: lr)'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Run hyperparameter tuning (grid search over C values)'
    )
    
    parser.add_argument(
        '--data-dir',
        default=DATA_DIR,
        help=f'Path to processed data directory (default: {DATA_DIR})'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    print(f"\n{'='*80}")
    print(f"PHASE 3: TRADITIONAL ML MODELS FOR ANSWER VERIFICATION")
    print(f"{'='*80}")
    
    main(args)
    
    print(f"\n{'='*80}")
    print(f"✓ PHASE 3 COMPLETE")
    print(f"Trained models saved to: {MODEL_DIR}/")
    print(f"{'='*80}\n")
