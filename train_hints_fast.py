"""Fast training script for hint generator on a subset for validation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from preprocessing import load_raw_splits, build_vectorizer
from hint_generator import (
    split_into_sentences,
    compute_sentence_features_batch,
    label_gold_hint_sentence,
    save_hint_model,
)

def train_hint_model_fast():
    """Train hint model on a subset of data for validation."""
    print("Loading data...")
    raw_splits = load_raw_splits(Path('data/raw'))
    train_df = raw_splits['train'].copy().iloc[:1000]  # Use 1000 samples for speed
    dev_df = raw_splits['dev'].copy().iloc[:500]
    test_df = raw_splits['test'].copy().iloc[:500]
    
    # Prepare
    for col in ['article', 'question', 'A', 'B', 'C', 'D']:
        train_df[col] = train_df[col].fillna("").astype(str)
        dev_df[col] = dev_df[col].fillna("").astype(str)
        test_df[col] = test_df[col].fillna("").astype(str)
    
    print("Building vectorizer...")
    vocab_texts = list(train_df['article'].astype(str)) + list(train_df['question'].astype(str))
    vectorizer = build_vectorizer(max_features=5000, min_df=2)
    vectorizer.fit(vocab_texts)
    
    print("Building training examples...")
    features = []
    labels = []
    for idx, (_, row) in enumerate(train_df.iterrows()):
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(train_df)} rows")
        
        article = str(row['article'])
        question = str(row['question'])
        correct_answer = str(row[row['answer']])
        
        sentences = split_into_sentences(article)
        if len(sentences) < 1:
            continue
        
        sent_features = compute_sentence_features_batch(
            sentences, article, question, correct_answer, vectorizer
        )
        features.extend([sent_features[i] for i in range(len(sent_features))])
        
        gold_idx = label_gold_hint_sentence(sentences, article, correct_answer)
        labels.extend([int(i == gold_idx) for i in range(len(sentences))])
    
    X_train = np.vstack(features)
    y_train = np.array(labels, dtype=int)
    
    print(f"\nTraining set: {X_train.shape[0]} examples, {y_train.sum()} positive")
    
    print("Training LogisticRegression...")
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    ranker = LogisticRegression(
        class_weight='balanced',
        solver='liblinear',
        random_state=42,
        max_iter=1000,
        C=1.0,
    )
    ranker.fit(X_train_sub, y_train_sub)
    
    val_pred = ranker.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, zero_division=0)
    print(f"Validation F1: {val_f1:.4f}")
    print(classification_report(y_val, val_pred, digits=4, zero_division=0))
    
    print("Saving model...")
    save_hint_model(ranker, vectorizer, Path('models/model_b/hint_generator'))
    print("✓ Model saved to models/model_b/hint_generator/")
    
    # Quick evaluation on dev
    print("\nEvaluating on dev set...")
    precisions_at_1 = []
    for idx, (_, row) in enumerate(dev_df.iterrows()):
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(dev_df)} rows")
        
        article = str(row['article'])
        question = str(row['question'])
        correct_answer = str(row[row['answer']])
        
        sentences = split_into_sentences(article)
        if len(sentences) < 1:
            continue
        
        gold_idx = label_gold_hint_sentence(sentences, article, correct_answer)
        X = compute_sentence_features_batch(
            sentences, article, question, correct_answer, vectorizer
        )
        y_pred_probs = ranker.predict_proba(X)[:, 1]
        top_1_idx = np.argmax(y_pred_probs)
        p_at_1 = float(top_1_idx == gold_idx)
        precisions_at_1.append(p_at_1)
    
    p1_mean = float(np.mean(precisions_at_1)) if precisions_at_1 else 0.0
    print(f"\nDev Precision@1: {p1_mean:.4f}")
    print("\n✓ Training complete!")

if __name__ == '__main__':
    train_hint_model_fast()
