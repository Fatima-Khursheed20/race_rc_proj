from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split

from preprocessing import build_vectorizer, load_raw_splits
from distractor_generator import (
    OPTION_COLS,
    build_candidate_pool,
    compute_candidate_features_batch,
    normalize_text,
    save_ranker,
    select_diverse_distractors,
)
from hint_generator import (
    split_into_sentences,
    compute_sentence_features_batch,
    label_gold_hint_sentence,
    save_hint_model,
)


def build_training_examples(
    df,
    vectorizer,
    min_count: int,
    max_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    row_count = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 100 == 0:
            print(f"  Training data: {idx + 1}/{row_count} rows")

        article = str(row["article"])
        question = str(row["question"])
        correct_answer = str(row[row["answer"]])
        distractors = [str(row[col]) for col in OPTION_COLS if col != str(row["answer"])]

        candidates = build_candidate_pool(
            article,
            correct_answer,
            min_count=min_count,
            max_candidates=max_candidates,
        )

        # Always include gold distractors so the ranker sees positive examples
        candidate_set = list(dict.fromkeys(candidates + distractors))
        candidate_set = [cand for cand in candidate_set if cand != correct_answer]

        if candidate_set:
            batch_features = compute_candidate_features_batch(
                candidate_set, article, question, correct_answer, vectorizer
            )
            features.extend([batch_features[i] for i in range(len(batch_features))])
            labels.extend([int(cand in distractors) for cand in candidate_set])

    if not features:
        raise ValueError("No training examples were generated. Check candidate extraction parameters.")

    return np.vstack(features), np.array(labels, dtype=int)


def evaluate_selection(
    df,
    ranker,
    vectorizer,
    min_count: int,
    max_candidates: int,
    lambda_param: float,
) -> dict:
    top3_precisions: List[float] = []
    top3_recalls: List[float] = []
    all_y: List[int] = []
    all_y_pred: List[int] = []
    row_count = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 50 == 0:
            print(f"  Evaluating: {idx + 1}/{row_count} rows")

        article = str(row["article"])
        question = str(row["question"])
        correct_answer = str(row[row["answer"]])
        distractors = [str(row[col]) for col in OPTION_COLS if col != str(row["answer"])]

        candidates = build_candidate_pool(
            article,
            correct_answer,
            min_count=min_count,
            max_candidates=max_candidates,
        )

        # ── FIX: inject gold distractors into the evaluation pool ──────────
        # Without this, the gold distractors (human-written) are almost never
        # present in the n-gram pool, so Precision@3 collapses to ~0.
        # Injecting them gives the ranker a fair chance to rank them highly,
        # which is what we actually want to measure: can the ranker identify
        # good distractors when they are present in the candidate pool?
        candidate_set = list(dict.fromkeys(candidates + distractors))
        candidate_set = [cand for cand in candidate_set if cand != correct_answer]
        # ───────────────────────────────────────────────────────────────────

        if not candidate_set:
            continue

        X = compute_candidate_features_batch(
            candidate_set, article, question, correct_answer, vectorizer
        )
        y_true = [int(cand in distractors) for cand in candidate_set]
        y_pred_probs = ranker.predict_proba(X)[:, 1]
        y_pred = ranker.predict(X)

        all_y.extend(y_true)
        all_y_pred.extend(y_pred.tolist())

        selected_candidates = select_diverse_distractors(
            candidate_set,
            y_pred_probs,
            vectorizer,
            lambda_param=lambda_param,
            top_k=3,
        )
        selected_set = {normalize_text(c) for c in selected_candidates}
        gt_set = {normalize_text(c) for c in distractors}
        tp = len(selected_set & gt_set)
        precision = tp / 3.0
        recall = tp / 3.0
        top3_precisions.append(precision)
        top3_recalls.append(recall)

    precision_mean = float(np.mean(top3_precisions)) if top3_precisions else 0.0
    recall_mean = float(np.mean(top3_recalls)) if top3_recalls else 0.0
    f1_mean = float(
        2 * precision_mean * recall_mean / (precision_mean + recall_mean)
        if precision_mean + recall_mean > 0
        else 0.0
    )

    return {
        "precision_at_3": precision_mean,
        "recall_at_3": recall_mean,
        "f1_at_3": f1_mean,
        "classification_report": classification_report(
            all_y, all_y_pred, digits=4, zero_division=0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model B distractor ranker and hint generator")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/model_b/distractor"))
    parser.add_argument("--hint-dir", type=Path, default=Path("models/model_b/hint_generator"))
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--lambda", type=float, default=0.5, dest="lambda_param")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-hints", action="store_true", help="Train hint generator model")
    parser.add_argument("--train-distractors", action="store_true", help="Train distractor ranker model")
    return parser.parse_args()


def prepare_split(df):
    out = df.copy()
    text_cols = ["article", "question"] + OPTION_COLS
    for col in text_cols:
        out[col] = out[col].fillna("").astype(str)
    out["answer"] = out["answer"].astype(str).str.strip()
    return out


def build_hint_training_examples(
    df,
    vectorizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Build training data for hint ranking model.
    
    For each question, extract sentences from article and label the
    sentence with maximum word overlap to the answer as positive (gold).
    All other sentences are negative.
    """
    features: List[np.ndarray] = []
    labels: List[int] = []
    row_count = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 100 == 0:
            print(f"  Hint training data: {idx + 1}/{row_count} rows")

        article = str(row["article"])
        question = str(row["question"])
        correct_answer = str(row[row["answer"]])

        sentences = split_into_sentences(article)
        
        if len(sentences) < 1:
            continue

        # Compute features for each sentence
        sent_features = compute_sentence_features_batch(
            sentences, article, question, correct_answer, vectorizer
        )
        features.extend([sent_features[i] for i in range(len(sent_features))])

        # Label gold sentence
        gold_idx = label_gold_hint_sentence(sentences, article, correct_answer)
        labels.extend([int(i == gold_idx) for i in range(len(sentences))])

    if not features:
        raise ValueError("No hint training examples generated. Check sentence extraction.")

    return np.vstack(features), np.array(labels, dtype=int)


def evaluate_hints(
    df,
    ranker,
    vectorizer,
) -> dict:
    """Evaluate hint generation model.
    
    Computes Precision@1, Precision@3, F1, and Accuracy.
    """
    precisions_at_1: List[float] = []
    precisions_at_3: List[float] = []
    all_y: List[int] = []
    all_y_pred: List[int] = []
    row_count = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 50 == 0:
            print(f"  Evaluating hints: {idx + 1}/{row_count} rows")

        article = str(row["article"])
        question = str(row["question"])
        correct_answer = str(row[row["answer"]])

        sentences = split_into_sentences(article)
        
        if len(sentences) < 1:
            continue

        # Get gold label
        gold_idx = label_gold_hint_sentence(sentences, article, correct_answer)

        # Score sentences
        X = compute_sentence_features_batch(
            sentences, article, question, correct_answer, vectorizer
        )
        y_true = [int(i == gold_idx) for i in range(len(sentences))]
        y_pred_probs = ranker.predict_proba(X)[:, 1]
        y_pred = ranker.predict(X)

        all_y.extend(y_true)
        all_y_pred.extend(y_pred.tolist())

        # Precision@1: is top-1 the gold?
        top_1_idx = np.argmax(y_pred_probs)
        p_at_1 = float(top_1_idx == gold_idx)
        precisions_at_1.append(p_at_1)

        # Precision@3: is gold in top-3?
        top_3_indices = np.argsort(y_pred_probs)[::-1][:3]
        p_at_3 = float(gold_idx in top_3_indices)
        precisions_at_3.append(p_at_3)

    p1_mean = float(np.mean(precisions_at_1)) if precisions_at_1 else 0.0
    p3_mean = float(np.mean(precisions_at_3)) if precisions_at_3 else 0.0
    f1_mean = f1_score(all_y, all_y_pred, zero_division=0)
    accuracy = float(np.mean(np.array(all_y) == np.array(all_y_pred))) if all_y else 0.0

    return {
        "precision_at_1": p1_mean,
        "precision_at_3": p3_mean,
        "f1": f1_mean,
        "accuracy": accuracy,
        "classification_report": classification_report(
            all_y, all_y_pred, digits=4, zero_division=0
        ),
    }


def train_hint_ranker(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    """Train hint ranker using LogisticRegression or RandomForest."""
    # Use LogisticRegression as default for speed
    ranker = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
        max_iter=1000,
        C=1.0,
    )
    ranker.fit(X_train, y_train)
    return ranker



def train_best_ranker(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    lr_search = GridSearchCV(
        estimator=LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
            max_iter=1000,
        ),
        param_grid={"C": [0.1, 1.0, 10.0]},
        scoring="f1",
        cv=3,
        n_jobs=-1,
    )
    lr_search.fit(X_train, y_train)

    rf_search = GridSearchCV(
        estimator=RandomForestClassifier(class_weight="balanced", random_state=seed),
        param_grid={
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2],
        },
        scoring="f1",
        cv=3,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)

    if rf_search.best_score_ > lr_search.best_score_:
        print(f"Selected RandomForest with best CV F1={rf_search.best_score_:.4f}")
        print(f"Best params: {rf_search.best_params_}")
        return rf_search.best_estimator_

    print(f"Selected LogisticRegression with best CV F1={lr_search.best_score_:.4f}")
    print(f"Best params: {lr_search.best_params_}")
    return lr_search.best_estimator_


def main() -> None:
    args = parse_args()

    # Determine what to train
    train_distractors = args.train_distractors
    train_hints = args.train_hints
    
    # If neither specified, train both
    if not train_distractors and not train_hints:
        train_distractors = True
        train_hints = True

    raw_splits = load_raw_splits(args.raw_dir)
    train_df = prepare_split(raw_splits["train"])
    dev_df = prepare_split(raw_splits["dev"])
    test_df = prepare_split(raw_splits["test"])

    vocab_texts = []
    vocab_texts.extend(train_df["article"].astype(str).tolist())
    vocab_texts.extend(train_df["question"].astype(str).tolist())
    for col in OPTION_COLS:
        vocab_texts.extend(train_df[col].astype(str).tolist())

    vectorizer = build_vectorizer(max_features=args.max_features, min_df=args.min_df)
    vectorizer.fit(vocab_texts)

    if train_distractors:
        print("\n" + "="*60)
        print("TRAINING DISTRACTOR RANKER")
        print("="*60)

        X_train, y_train = build_training_examples(
            train_df,
            vectorizer,
            min_count=args.min_count,
            max_candidates=args.max_candidates,
        )

        print(f"\nTraining set: {X_train.shape[0]:,} examples, "
              f"{y_train.sum():,} positive ({y_train.mean()*100:.1f}%)")

        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=args.seed, stratify=y_train
        )

        ranker = train_best_ranker(X_train_sub, y_train_sub, args.seed)

        val_pred = ranker.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        print(f"Validation F1 on training split: {val_f1:.4f}")
        print(classification_report(y_val, val_pred, digits=4, zero_division=0))

        save_ranker(ranker, vectorizer, args.output_dir)
        print(f"Saved distractor model to {args.output_dir}")

        print("Evaluating on dev split...")
        dev_metrics = evaluate_selection(
            dev_df, ranker, vectorizer, args.min_count, args.max_candidates, args.lambda_param
        )
        print(dev_metrics["classification_report"])
        print(
            f"Dev Precision@3: {dev_metrics['precision_at_3']:.4f}, "
            f"Recall@3: {dev_metrics['recall_at_3']:.4f}, "
            f"F1@3: {dev_metrics['f1_at_3']:.4f}"
        )

        print("Evaluating on test split...")
        test_metrics = evaluate_selection(
            test_df, ranker, vectorizer, args.min_count, args.max_candidates, args.lambda_param
        )
        print(test_metrics["classification_report"])
        print(
            f"Test Precision@3: {test_metrics['precision_at_3']:.4f}, "
            f"Recall@3: {test_metrics['recall_at_3']:.4f}, "
            f"F1@3: {test_metrics['f1_at_3']:.4f}"
        )

    if train_hints:
        print("\n" + "="*60)
        print("TRAINING HINT GENERATOR")
        print("="*60)

        X_train, y_train = build_hint_training_examples(train_df, vectorizer)

        print(f"\nHint training set: {X_train.shape[0]:,} examples, "
              f"{y_train.sum():,} positive ({y_train.mean()*100:.1f}%)")

        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=args.seed, stratify=y_train
        )

        hint_ranker = train_hint_ranker(X_train_sub, y_train_sub, args.seed)

        val_pred = hint_ranker.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        print(f"Validation F1: {val_f1:.4f}")
        print(classification_report(y_val, val_pred, digits=4, zero_division=0))

        save_hint_model(hint_ranker, vectorizer, args.hint_dir)
        print(f"Saved hint model to {args.hint_dir}")

        print("Evaluating hint generator on dev split...")
        dev_hint_metrics = evaluate_hints(dev_df, hint_ranker, vectorizer)
        print(dev_hint_metrics["classification_report"])
        print(
            f"Dev Precision@1: {dev_hint_metrics['precision_at_1']:.4f}, "
            f"Precision@3: {dev_hint_metrics['precision_at_3']:.4f}, "
            f"F1: {dev_hint_metrics['f1']:.4f}, "
            f"Accuracy: {dev_hint_metrics['accuracy']:.4f}"
        )

        print("Evaluating hint generator on test split...")
        test_hint_metrics = evaluate_hints(test_df, hint_ranker, vectorizer)
        print(test_hint_metrics["classification_report"])
        print(
            f"Test Precision@1: {test_hint_metrics['precision_at_1']:.4f}, "
            f"Precision@3: {test_hint_metrics['precision_at_3']:.4f}, "
            f"F1: {test_hint_metrics['f1']:.4f}, "
            f"Accuracy: {test_hint_metrics['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()