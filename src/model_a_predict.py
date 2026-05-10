"""Answer-verification scores for Model A (OHE + lexical + cosine, Phase 2 / model_a_train)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV

from preprocessing import compute_cosine_similarity_features, compute_lexical_features

EnsembleMode = Literal["lr", "soft_ls"]


def load_model_a_bundle(model_dir: Path | str):
    model_dir = Path(model_dir)
    clf = joblib.load(model_dir / "logistic_regression.pkl")
    vectorizer = joblib.load(model_dir / "ohe_vectorizer.pkl")
    scaler = joblib.load(model_dir / "scaler_lexical.pkl")
    return clf, vectorizer, scaler


def load_svm_if_available(model_dir: Path | str) -> CalibratedClassifierCV | None:
    p = Path(model_dir) / "svm_calibrated.pkl"
    if not p.exists():
        return None
    return joblib.load(p)


def _build_combined_matrix(
    article_trunc: str,
    question_s: str,
    opts: List[str],
    *,
    vectorizer,
    scaler,
) -> sparse.csr_matrix:
    combined_texts: List[str] = []
    lexical_rows: List[dict] = []
    for opt in opts:
        combined_texts.append(f"{article_trunc} [SEP] {question_s} [SEP] {opt}")
        lexical_rows.append(compute_lexical_features(article_trunc, question_s, opt))

    lex_df = pd.DataFrame(lexical_rows)
    x_lex_base = lex_df.to_numpy(dtype=np.float32)

    x_ohe = vectorizer.transform(combined_texts)
    row_ids = np.zeros(len(opts), dtype=np.int32)
    x_cos = compute_cosine_similarity_features(x_ohe, row_ids)
    x_lex_full = np.hstack([x_lex_base, x_cos])

    x_lex_scaled = scaler.transform(x_lex_full)
    x_lex_sparse = sparse.csr_matrix(x_lex_scaled)
    return sparse.hstack([x_ohe, x_lex_sparse])


def predict_option_verification_proba(
    article: str,
    question: str,
    options: Sequence[str],
    *,
    vectorizer,
    scaler,
    clf,
    svm_clf: CalibratedClassifierCV | None = None,
    ensemble_mode: EnsembleMode = "lr",
    max_article_words: int = 500,
) -> Tuple[np.ndarray, str]:
    opts = [str(o) for o in options]
    if len(opts) != 4:
        raise ValueError("Model A expects exactly 4 options")

    article_trunc = " ".join(str(article).split()[:max_article_words])
    question_s = str(question)

    x_combined = _build_combined_matrix(
        article_trunc, question_s, opts, vectorizer=vectorizer, scaler=scaler
    )

    lr_p = clf.predict_proba(x_combined)[:, 1]

    if ensemble_mode == "soft_ls" and svm_clf is not None:
        try:
            svm_p = svm_clf.predict_proba(x_combined)[:, 1]
        except ValueError:
            return lr_p, "logistic_regression_svm_mismatch"
        return (lr_p + svm_p) / 2.0, "soft_ensemble_lr_svm"

    return lr_p, "logistic_regression"


def predict_option_verification_proba_legacy(
    article: str,
    question: str,
    options: Sequence[str],
    *,
    vectorizer,
    scaler,
    clf,
    max_article_words: int = 500,
) -> np.ndarray:
    """Backward-compatible: probabilities only (LR)."""
    p, _ = predict_option_verification_proba(
        article,
        question,
        options,
        vectorizer=vectorizer,
        scaler=scaler,
        clf=clf,
        svm_clf=None,
        ensemble_mode="lr",
        max_article_words=max_article_words,
    )
    return p
