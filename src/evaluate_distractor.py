"""
Standalone evaluation script for the saved distractor ranker (Model B).
Loads the already-trained model and prints full metrics without retraining.

Usage:
    python src/evaluate_distractor.py --raw-dir data/raw --model-dir models/model_b/distractor
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt

from preprocessing import load_raw_splits
from distractor_generator import (
    OPTION_COLS,
    build_candidate_pool,
    compute_candidate_features_batch,
    load_ranker,
    normalize_text,
    select_diverse_distractors,
)


def prepare_split(df):
    out = df.copy()
    text_cols = ["article", "question"] + OPTION_COLS
    for col in text_cols:
        out[col] = out[col].fillna("").astype(str)
    out["answer"] = out["answer"].astype(str).str.strip()
    return out


def evaluate_split(
    df,
    ranker,
    vectorizer,
    split_name: str,
    min_count: int,
    max_candidates: int,
    lambda_param: float,
    save_plots: bool = True,
) -> dict:
    top3_precisions: List[float] = []
    top3_recalls: List[float] = []
    all_y: List[int] = []
    all_y_pred: List[int] = []
    all_y_prob: List[float] = []
    row_count = len(df)

    print(f"\n{'='*60}")
    print(f"  Evaluating: {split_name} split ({row_count:,} rows)")
    print(f"{'='*60}")

    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{row_count} rows done...")

        article = str(row["article"])
        question = str(row["question"])
        correct_answer = str(row[row["answer"]])
        distractors = [str(row[col]) for col in OPTION_COLS if col != str(row["answer"])]

        candidates = build_candidate_pool(
            article, correct_answer,
            min_count=min_count,
            max_candidates=max_candidates,
        )

        # Inject gold distractors so ranker has a fair chance to rank them
        candidate_set = list(dict.fromkeys(candidates + distractors))
        candidate_set = [c for c in candidate_set if c != correct_answer]

        if not candidate_set:
            continue

        X = compute_candidate_features_batch(
            candidate_set, article, question, correct_answer, vectorizer
        )
        y_true = [int(c in distractors) for c in candidate_set]
        y_pred_probs = ranker.predict_proba(X)[:, 1]
        y_pred = ranker.predict(X)

        all_y.extend(y_true)
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_pred_probs.tolist())

        selected = select_diverse_distractors(
            candidate_set, y_pred_probs, vectorizer,
            lambda_param=lambda_param, top_k=3,
        )
        selected_set = {normalize_text(c) for c in selected}
        gt_set = {normalize_text(c) for c in distractors}
        tp = len(selected_set & gt_set)
        top3_precisions.append(tp / 3.0)
        top3_recalls.append(tp / 3.0)

    # ── Binary classification metrics (ranker quality) ──────────────────
    all_y = np.array(all_y)
    all_y_pred = np.array(all_y_pred)

    accuracy  = accuracy_score(all_y, all_y_pred)
    precision = precision_score(all_y, all_y_pred, zero_division=0)
    recall    = recall_score(all_y, all_y_pred, zero_division=0)
    f1        = f1_score(all_y, all_y_pred, zero_division=0)
    macro_f1  = f1_score(all_y, all_y_pred, average="macro", zero_division=0)

    # ── Selection metrics (end-to-end quality) ───────────────────────────
    p3 = float(np.mean(top3_precisions)) if top3_precisions else 0.0
    r3 = float(np.mean(top3_recalls))    if top3_recalls    else 0.0
    f3 = 2*p3*r3/(p3+r3) if (p3+r3) > 0 else 0.0

    # ── Print results ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS — {split_name.upper()} SPLIT")
    print(f"{'='*60}")
    print(f"\n--- Ranker Binary Classification ---")
    print(f"  Accuracy:          {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision (pos):   {precision:.4f}")
    print(f"  Recall    (pos):   {recall:.4f}")
    print(f"  F1 (binary):       {f1:.4f}")
    print(f"  Macro F1:          {macro_f1:.4f}")
    print(f"\n  Full classification report:")
    print(classification_report(all_y, all_y_pred, digits=4, zero_division=0))

    print(f"--- End-to-End Selection (Precision/Recall/F1 @3) ---")
    print(f"  Precision@3:  {p3:.4f}  ({p3*100:.2f}%)")
    print(f"  Recall@3:     {r3:.4f}  ({r3*100:.2f}%)")
    print(f"  F1@3:         {f3:.4f}  ({f3*100:.2f}%)")
    print(f"{'='*60}\n")

    # ── Confusion matrix plot ────────────────────────────────────────────
    if save_plots:
        cm = confusion_matrix(all_y, all_y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not distractor", "Distractor"],
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {split_name} split")
        plot_path = f"distractor_confusion_{split_name.lower()}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Confusion matrix saved to: {plot_path}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "precision_at_3": p3,
        "recall_at_3": r3,
        "f1_at_3": f3,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved distractor ranker")
    parser.add_argument("--raw-dir",      type=Path, default=Path("data/raw"))
    parser.add_argument("--model-dir",    type=Path, default=Path("models/model_b/distractor"))
    parser.add_argument("--min-count",    type=int,  default=1)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--lambda",       type=float, default=0.5, dest="lambda_param")
    parser.add_argument("--splits",       nargs="+", default=["dev", "test"],
                        choices=["train", "dev", "test"],
                        help="Which splits to evaluate (default: dev test)")
    parser.add_argument("--no-plots",     action="store_true",
                        help="Skip saving confusion matrix plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model from: {args.model_dir}")
    ranker, vectorizer = load_ranker(args.model_dir)
    print(f"  Model type: {type(ranker).__name__}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")

    raw_splits = load_raw_splits(args.raw_dir)
    all_results = {}

    for split_name in args.splits:
        df = prepare_split(raw_splits[split_name])
        results = evaluate_split(
            df, ranker, vectorizer,
            split_name=split_name,
            min_count=args.min_count,
            max_candidates=args.max_candidates,
            lambda_param=args.lambda_param,
            save_plots=not args.no_plots,
        )
        all_results[split_name] = results

    # ── Summary table ────────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  SUMMARY TABLE")
        print(f"{'='*60}")
        header = f"{'Metric':<22}" + "".join(f"{s.upper():>12}" for s in all_results)
        print(header)
        print("-" * len(header))
        metrics_display = [
            ("Accuracy",     "accuracy"),
            ("Precision",    "precision"),
            ("Recall",       "recall"),
            ("F1 (binary)",  "f1"),
            ("Macro F1",     "macro_f1"),
            ("Precision@3",  "precision_at_3"),
            ("Recall@3",     "recall_at_3"),
            ("F1@3",         "f1_at_3"),
        ]
        for label, key in metrics_display:
            row = f"{label:<22}"
            for split_name in all_results:
                row += f"{all_results[split_name][key]:>12.4f}"
            print(row)
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
