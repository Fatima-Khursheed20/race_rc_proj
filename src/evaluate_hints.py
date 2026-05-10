"""
Standalone evaluation script for the saved hint generation model (Model B).

Usage:
    python src/evaluate_hints.py --raw-dir data/raw --model-dir models/model_b/hint_generator
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path.cwd() / 'src'))

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from hint_generator import load_hint_model, compute_sentence_features_batch, label_gold_hint_sentence, split_into_sentences
from preprocessing import load_raw_splits

try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    import importlib.util
    _HAS_NLTK = importlib.util.find_spec('nltk') is not None
except Exception:
    _HAS_NLTK = False

try:
    import importlib.util
    _HAS_ROUGE_SCORE = importlib.util.find_spec('rouge_score') is not None
except Exception:
    _HAS_ROUGE_SCORE = False

def _tokenize_text(text: str) -> List[str]:
    return [token for token in text.strip().split() if token]

if _HAS_NLTK:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import single_meteor_score

if _HAS_ROUGE_SCORE:
    from rouge_score import rouge_scorer


def evaluate_split(df, ranker, vectorizer, split_name: str, save_plots: bool = True) -> dict:
    precisions_at_1: List[float] = []
    precisions_at_3: List[float] = []
    all_y: List[int] = []
    all_y_pred: List[int] = []
    gold_sentences: List[str] = []
    pred_sentences: List[str] = []

    row_count = len(df)
    print(f"\nEvaluating hint extraction on {split_name} split ({row_count:,} rows)")

    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{row_count} rows processed")

        article = str(row["article"])
        question = str(row["question"])
        correct_answer = str(row[row["answer"]])

        sentences = split_into_sentences(article)
        if not sentences:
            continue

        gold_idx = label_gold_hint_sentence(sentences, article, correct_answer)
        if gold_idx < 0 or gold_idx >= len(sentences):
            continue

        X = compute_sentence_features_batch(sentences, article, question, correct_answer, vectorizer)
        y_pred_probs = ranker.predict_proba(X)[:, 1]
        y_pred = ranker.predict(X)

        all_y.extend([int(i == gold_idx) for i in range(len(sentences))])
        all_y_pred.extend(y_pred.tolist())

        top_1_idx = int(np.argmax(y_pred_probs))
        top_3_idx = np.argsort(y_pred_probs)[::-1][:3]

        precisions_at_1.append(float(top_1_idx == gold_idx))
        precisions_at_3.append(float(gold_idx in top_3_idx))

        gold_sentences.append(sentences[gold_idx])
        pred_sentences.append(sentences[top_1_idx])

    p1_mean = float(np.mean(precisions_at_1)) if precisions_at_1 else 0.0
    p3_mean = float(np.mean(precisions_at_3)) if precisions_at_3 else 0.0
    f1_mean = f1_score(all_y, all_y_pred, zero_division=0)
    accuracy = float(np.mean(np.array(all_y) == np.array(all_y_pred))) if all_y else 0.0

    print(f"\n--- Hint Evaluation Results ({split_name}) ---")
    print(f"Precision@1: {p1_mean:.4f} ({p1_mean*100:.2f}%)")
    print(f"Precision@3: {p3_mean:.4f} ({p3_mean*100:.2f}%)")
    print(f"Sentence-level F1: {f1_mean:.4f}")
    print(f"Sentence-level Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_y, all_y_pred, digits=4, zero_division=0))

    if _HAS_MATPLOTLIB:
        cm = confusion_matrix(all_y, all_y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not gold hint", "Gold hint"])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Hint Confusion Matrix — {split_name}")
        plot_path = f"hint_confusion_{split_name.lower()}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved confusion matrix plot to: {plot_path}")
    else:
        print("matplotlib not installed; skipping confusion matrix plot.")

    bleu_score = None
    rouge_scores = None
    meteor_score = None

    if _HAS_NLTK:
        weights = (0.5, 0.5)
        smoothing = SmoothingFunction().method1
        bleu_values = []
        meteor_values = []
        for gold, pred in zip(gold_sentences, pred_sentences):
            if not gold or not pred:
                continue
            gold_tokens = _tokenize_text(gold)
            pred_tokens = _tokenize_text(pred)
            if not gold_tokens or not pred_tokens:
                continue
            try:
                bleu_values.append(
                    sentence_bleu([gold_tokens], pred_tokens, weights=weights, smoothing_function=smoothing)
                )
            except Exception:
                pass
            try:
                meteor_values.append(
                    single_meteor_score(gold_tokens, pred_tokens)
                )
            except Exception:
                pass
        bleu_score = float(np.mean(bleu_values)) if bleu_values else 0.0
        meteor_score = float(np.mean(meteor_values)) if meteor_values else 0.0

    if _HAS_ROUGE_SCORE:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge1 = []
        rouge2 = []
        rougel = []
        for gold, pred in zip(gold_sentences, pred_sentences):
            if not gold or not pred:
                continue
            scores = scorer.score(gold, pred)
            rouge1.append(scores["rouge1"].fmeasure)
            rouge2.append(scores["rouge2"].fmeasure)
            rougel.append(scores["rougeL"].fmeasure)
        rouge_scores = {
            "rouge1": float(np.mean(rouge1)) if rouge1 else 0.0,
            "rouge2": float(np.mean(rouge2)) if rouge2 else 0.0,
            "rougeL": float(np.mean(rougel)) if rougel else 0.0,
        }

    if bleu_score is not None:
        print(f"BLEU (average sentence): {bleu_score:.4f}")
    else:
        print("nltk not installed; BLEU/METEOR unavailable.")

    if rouge_scores is not None:
        print("ROUGE scores:")
        print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
        print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
        print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")
    else:
        print("rouge-score not installed; ROUGE unavailable.")

    if meteor_score is not None:
        print(f"METEOR: {meteor_score:.4f}")
    else:
        print("nltk not installed; METEOR unavailable.")

    return {
        "precision_at_1": p1_mean,
        "precision_at_3": p3_mean,
        "f1": f1_mean,
        "accuracy": accuracy,
        "bleu": bleu_score,
        "meteor": meteor_score,
        "rouge": rouge_scores,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved hint generation model")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/model_b/hint_generator"))
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["train", "dev", "test"], help="Which splits to evaluate")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving confusion matrix plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading hint model from: {args.model_dir}")
    ranker, vectorizer = load_hint_model(args.model_dir)

    raw_splits = load_raw_splits(args.raw_dir)
    for split_name in args.splits:
        df = raw_splits[split_name]
        evaluate_split(df, ranker, vectorizer, split_name=split_name, save_plots=not args.no_plots)


if __name__ == "__main__":
    main()
