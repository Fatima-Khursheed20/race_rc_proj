#!/usr/bin/env python3
"""
Confusion Matrix for Model B Hint Generation

This script evaluates the hint generation model on the dev set
and prints the confusion matrix.
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

from preprocessing import load_raw_splits
from hint_generator import load_hint_model, compute_sentence_features_batch
from model_b_train import build_hint_training_examples

def main():
    """Compute confusion matrix for hint generation model."""
    
    print("Loading model...")
    ranker, vectorizer = load_hint_model(Path('models/model_b/hint_generator'))
    
    print("Loading dev data...")
    splits = load_raw_splits(Path('data/raw'))
    dev_df = splits['dev']
    
    print("Building test examples...")
    X_test, y_test = build_hint_training_examples(dev_df, vectorizer)
    
    print(f"Test set: {X_test.shape[0]} examples, {np.sum(y_test)} positive")
    
    print("Predicting...")
    y_pred = ranker.predict(X_test)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

if __name__ == '__main__':
    main()