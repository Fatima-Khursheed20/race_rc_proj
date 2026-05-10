"""End-to-end quiz generation for the Streamlit UI.

Integrates Model A (answer verification probabilities), Model B distractor ranker,
and Model B hint ranker when artifacts exist under ``models/``.

Two modes:

1. **race_row** (optional dict from RACE CSV): use stored question / four options /
   correct-letter; regenerate hints with the hint model when available.

2. **Generated** (default): extract a supporting sentence as the keyed correct answer;
   optional **Phase 3 question generator** (``qg_ranker.pkl`` + WH templates); distractors from
   Model B ranker + fill; verification via Model A **LR** or **soft LR+SVM** ensemble (OHE pipeline).

   **Phase 4** (unsupervised / semi-supervised) metrics are surfaced from ``phase4/phase4_results.json``
   when present — clustering experiments are not used for live answer scoring in this UI.

Run Streamlit from the project root::

    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# Allow ``from preprocessing`` / ``from distractor_generator`` when loaded as ``src.inference``.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np

from distractor_generator import generate_distractors, load_ranker as load_distractor_ranker
from hint_generator import generate_hints, load_hint_model, split_into_sentences
from model_a_predict import (
    load_model_a_bundle,
    load_svm_if_available,
    predict_option_verification_proba,
)
from question_generator import generate_question_ranked, load_qg_ranker

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_A_DIR = ROOT / "models" / "model_a" / "traditional"
DEFAULT_DISTRACTOR_DIR = ROOT / "models" / "model_b" / "distractor"
DEFAULT_HINT_DIR = ROOT / "models" / "model_b" / "hint_generator"

# Fallback stems if ``qg_ranker.pkl`` is missing or disabled.
QUESTION_STEMS = [
    "According to the passage, which of the following statements is best supported by the text?",
    "Based only on what the passage states, which choice is clearly supported?",
    "Which conclusion can reasonably be inferred from details in the passage?",
    "The author implies which of the following through the passage?",
    "Which option best summarizes one specific factual point made in the text?",
    "Which statement would you include in bullet notes summarizing evidence from this passage?",
    "If asked to cite one claim the passage establishes, which option fits best?",
    "Which choice restates something the passage communicates, rather than guessing beyond it?",
    "Which sentence could be defended using direct support from this passage?",
    "What does the passage most directly tell the reader?",
]


def pick_question_stem(*, stem_rotate_key: Optional[int] = None) -> str:
    """Rotate through stems deterministically when ``stem_rotate_key`` increases (recommended for Streamlit UX)."""
    n = len(QUESTION_STEMS)
    if stem_rotate_key is None:
        return QUESTION_STEMS[random.randrange(n)]
    return QUESTION_STEMS[int(stem_rotate_key) % n]


def load_phase4_summary(model_a_dir: Path) -> Optional[dict]:
    """Return Phase 4 JSON summary if ``phase4/phase4_results.json`` exists."""
    path = model_a_dir / "phase4" / "phase4_results.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _artifact_status(model_a_dir: Path, distractor_dir: Path, hint_dir: Path) -> Dict[str, bool]:
    return {
        "model_a": (model_a_dir / "logistic_regression.pkl").exists()
        and (model_a_dir / "ohe_vectorizer.pkl").exists()
        and (model_a_dir / "scaler_lexical.pkl").exists(),
        "distractor_ranker": (distractor_dir / "distractor_ranker.pkl").exists()
        and (distractor_dir / "distractor_vectorizer.pkl").exists(),
        "hint_ranker": (hint_dir / "hint_ranker.pkl").exists(),
    }


def _artifact_status_extended(model_a_dir: Path, distractor_dir: Path, hint_dir: Path) -> Dict[str, Any]:
    base = _artifact_status(model_a_dir, distractor_dir, hint_dir)
    base["qg_ranker"] = (model_a_dir / "qg_ranker.pkl").exists()
    base["svm_calibrated"] = (model_a_dir / "svm_calibrated.pkl").exists()
    base["stacking_meta_lr"] = (model_a_dir / "ensemble_stacking_meta_lr.pkl").exists()
    base["phase4_results"] = (model_a_dir / "phase4" / "phase4_results.json").exists()
    return base


def models_available() -> Dict[str, Any]:
    """Lightweight check for the UI sidebar without loading picklers."""
    flags = _artifact_status_extended(
        DEFAULT_MODEL_A_DIR, DEFAULT_DISTRACTOR_DIR, DEFAULT_HINT_DIR
    )
    return {
        "paths": {
            "model_a": str(DEFAULT_MODEL_A_DIR),
            "distractor": str(DEFAULT_DISTRACTOR_DIR),
            "hints": str(DEFAULT_HINT_DIR),
            "phase4_json": str(DEFAULT_MODEL_A_DIR / "phase4" / "phase4_results.json"),
        },
        "loaded": flags,
    }


def normalize_candidate(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def pick_extractive_answer_sentence(article: str) -> str:
    sentences = split_into_sentences(str(article).strip())
    if not sentences:
        chunk = str(article).strip()
        return chunk[: min(280, len(chunk))] or "Insufficient passage text."

    def score_sentence(sent: str) -> float:
        words = re.findall(r"\b\w+\b", sent.lower())
        if len(words) < 6:
            return -1e9
        return float(len(words))

    sentences.sort(key=score_sentence, reverse=True)
    pool = sentences[: max(5, len(sentences) // 10 or 5)]
    return random.choice(pool) if pool else sentences[0]


def _other_sentences_for_distractors(article: str, correct_answer: str) -> List[str]:
    """Other passage sentences (plausible wrong statements) when the ranker returns too little."""
    correct_n = normalize_candidate(correct_answer)
    sents = split_into_sentences(str(article).strip())
    out: List[str] = []
    for s in sents:
        s = s.strip()
        if len(re.findall(r"\b\w+\b", s)) < 5:
            continue
        if normalize_candidate(s) == correct_n:
            continue
        out.append(s)
    random.shuffle(out)
    return out


def _phrase_fallbacks(article: str, correct_answer: str, seen: set[str], need: int) -> List[str]:
    """Short extractive phrases if we still lack three distractors."""
    words = re.findall(r"\b\w+\b", str(article))
    if len(words) < 12:
        return []
    chunks: List[str] = []
    step = max(6, len(words) // 12)
    for i in range(0, len(words) - 8, step):
        chunk = " ".join(words[i : i + 10])
        if len(chunk) < 30:
            continue
        cn = normalize_candidate(chunk)
        if cn in seen or cn == normalize_candidate(correct_answer):
            continue
        chunks.append(chunk[:200] + ("…" if len(chunk) > 200 else ""))
        seen.add(cn)
        if len(chunks) >= need:
            break
    return chunks


def merge_distractor_sources(
    article: str,
    correct_answer: str,
    ranker_distractors: List[str],
    need: int = 3,
) -> List[str]:
    """Prefer ranker output; fill with other sentences from the article; avoid generic placeholders."""
    seen: set[str] = {normalize_candidate(correct_answer)}
    out: List[str] = []

    for d in ranker_distractors:
        d = (d or "").strip()
        if not d:
            continue
        nd = normalize_candidate(d)
        if nd in seen:
            continue
        out.append(d)
        seen.add(nd)
        if len(out) >= need:
            return out[:need]

    for sent in _other_sentences_for_distractors(article, correct_answer):
        nd = normalize_candidate(sent)
        if nd in seen:
            continue
        out.append(sent)
        seen.add(nd)
        if len(out) >= need:
            return out[:need]

    for frag in _phrase_fallbacks(article, correct_answer, seen, need - len(out)):
        out.append(frag)
        if len(out) >= need:
            return out[:need]

    # Very short articles: splice non-overlapping raw slices so options stay distinct from the gold span.
    raw = str(article).strip()
    if raw and len(out) < need:
        slices = []
        slice_len = min(140, max(40, len(raw) // 5))
        for start in range(0, len(raw), slice_len):
            slug = raw[start : start + slice_len].strip()
            slug = " ".join(slug.split())
            if len(slug) < 25:
                continue
            dn = normalize_candidate(slug)
            if dn not in seen and dn != normalize_candidate(correct_answer):
                slices.append(slug[:220] + ("…" if len(slug) > 220 else ""))
                seen.add(dn)
            if len(slices) >= need:
                break
        out.extend(slices[: need - len(out)])

    suffix = 0
    while len(out) < need:
        suffix += 1
        filler = f"[Review the passage: supporting detail variation {suffix}.]"
        if normalize_candidate(filler) not in seen:
            seen.add(normalize_candidate(filler))
            out.append(filler)
    return out[:need]


def _shuffle_options(correct: str, distractors: List[str]) -> tuple[List[str], int]:
    opts = [correct] + distractors[:3]
    order = list(range(4))
    random.shuffle(order)
    shuffled = [opts[i] for i in order]
    gold_new = order.index(0)
    return shuffled, gold_new


def explain_from_probs(
    options: List[str], probs: np.ndarray, *, verification_backend: str = ""
) -> str:
    lines = []
    letters = ["A", "B", "C", "D"]
    header = "Model A verification scores (probability each option matches passage + question)."
    if verification_backend:
        header += f"\n**Backend:** `{verification_backend}` (OHE + lexical + cosine features from Phase 2)."
    for letter, opt, p in sorted(zip(letters, options, probs), key=lambda t: -t[2]):
        short = opt if len(opt) <= 120 else opt[:117] + "..."
        lines.append(f"- **{letter}**: `{p:.3f}` — {short}")
    return header + "\n" + "\n".join(lines)


def run_pipeline(
    article_text: str,
    race_row: Optional[Mapping[str, Any]] = None,
    *,
    model_a_dir: Optional[Path] = None,
    distractor_dir: Optional[Path] = None,
    hint_dir: Optional[Path] = None,
    distractor_lambda: float = 0.5,
    max_article_words: int = 500,
    quiz_stem_rotate_key: Optional[int] = None,
    use_question_generator: bool = True,
    verification_ensemble: str = "soft_ls",
) -> dict:
    article_text = (article_text or "").strip()
    if not article_text:
        return empty_result()

    model_a_dir = Path(model_a_dir or DEFAULT_MODEL_A_DIR)
    distractor_dir = Path(distractor_dir or DEFAULT_DISTRACTOR_DIR)
    hint_dir = Path(hint_dir or DEFAULT_HINT_DIR)

    flags: Dict[str, Any] = _artifact_status_extended(model_a_dir, distractor_dir, hint_dir)

    clf = vectorizer = scaler = None
    svm_clf = None
    if flags["model_a"]:
        try:
            clf, vectorizer, scaler = load_model_a_bundle(model_a_dir)
        except Exception:
            clf = vectorizer = scaler = None
            flags["model_a"] = False
    if flags.get("svm_calibrated"):
        try:
            svm_clf = load_svm_if_available(model_a_dir)
        except Exception:
            svm_clf = None
            flags["svm_calibrated"] = False

    qg_ranker = None
    if flags.get("qg_ranker"):
        try:
            qg_ranker = load_qg_ranker(model_a_dir)
        except Exception:
            qg_ranker = None
            flags["qg_ranker"] = False

    distractor_bundle = None
    if flags["distractor_ranker"]:
        try:
            distractor_bundle = load_distractor_ranker(distractor_dir)
        except Exception:
            distractor_bundle = None
            flags["distractor_ranker"] = False

    hint_bundle = None
    if flags["hint_ranker"]:
        try:
            hint_bundle = load_hint_model(hint_dir)
        except Exception:
            hint_bundle = None
            flags["hint_ranker"] = False

    phase4_summary = load_phase4_summary(model_a_dir)
    stacking_note: Optional[str] = None
    if flags.get("stacking_meta_lr"):
        stacking_note = (
            "`ensemble_stacking_meta_lr.pkl` was trained in Phase 3 Colab (meta features: LR+SVM+XGB on TF-IDF). "
            "This UI scores answers with the **OHE + lexical** pipeline; use **soft LR+SVM** here for a comparable ensemble."
        )

    ens_mode: str = "lr"
    if verification_ensemble == "soft_ls" and svm_clf is not None:
        ens_mode = "soft_ls"
    elif verification_ensemble == "soft_ls" and svm_clf is None:
        ens_mode = "lr"

    hints_default = [
        "Skim the opening for the topic.",
        "Re-read the middle paragraphs for supporting detail.",
        "Look for wording that aligns with key terms from the stem.",
    ]

    def run_verification(question: str, options: List[str]) -> tuple:
        if clf is None or vectorizer is None or scaler is None:
            return None, "", ""
        try:
            probs, backend = predict_option_verification_proba(
                article_text,
                question,
                options,
                vectorizer=vectorizer,
                scaler=scaler,
                clf=clf,
                svm_clf=svm_clf,
                ensemble_mode=ens_mode,
                max_article_words=max_article_words,
            )
            expl = explain_from_probs(options, probs, verification_backend=backend)
            return probs, expl, backend
        except Exception:
            return None, "Model A scores unavailable for this quiz.", ""

    if race_row is not None:
        question = str(race_row.get("question") or "").strip() or pick_question_stem(
            stem_rotate_key=quiz_stem_rotate_key
        )
        options = [
            str(race_row.get("A") or ""),
            str(race_row.get("B") or ""),
            str(race_row.get("C") or ""),
            str(race_row.get("D") or ""),
        ]
        ans_letter = str(race_row.get("answer") or "A").strip().upper()
        ci = {"A": 0, "B": 1, "C": 2, "D": 3}.get(ans_letter, 0)
        correct_text = options[ci]

        if hint_bundle:
            h_ranker, h_vec = hint_bundle
            hints_map = generate_hints(
                article_text,
                question,
                correct_text,
                h_ranker,
                h_vec,
            )
            hints = [hints_map["hint_1"], hints_map["hint_2"], hints_map["hint_3"]]
        else:
            hints = hints_default

        option_probs, explanation, v_backend = run_verification(question, options)
        if isinstance(option_probs, np.ndarray):
            op_list: Optional[List[float]] = option_probs.tolist()
        else:
            op_list = None

        return {
            "question": question,
            "options": options,
            "correct_index": ci,
            "hints": hints,
            "explanation": explanation or hints_default[0],
            "quiz_mode": "race_row",
            "models": flags,
            "option_probs": op_list,
            "predicted_correct_index": int(np.argmax(option_probs)) if option_probs is not None else None,
            "question_source": "race_dataset",
            "question_anchor": "",
            "verification_backend": v_backend,
            "phase4_summary": phase4_summary,
            "stacking_note": stacking_note,
        }

    correct_answer = pick_extractive_answer_sentence(article_text)
    question_anchor = ""
    question_source = "template_stem"
    if use_question_generator and qg_ranker is not None:
        question, question_anchor = generate_question_ranked(
            article_text, correct_answer, qg_ranker
        )
        question_source = "ml_ranked_templates"
    else:
        question = pick_question_stem(stem_rotate_key=quiz_stem_rotate_key)

    distractor_list: List[str] = []
    if distractor_bundle:
        ranker, d_vec = distractor_bundle
        try:
            distractor_list = generate_distractors(
                article_text,
                question,
                correct_answer,
                ranker,
                d_vec,
                lambda_param=distractor_lambda,
            )
        except Exception:
            distractor_list = []

    merged = merge_distractor_sources(article_text, correct_answer, distractor_list, need=3)
    options, correct_index = _shuffle_options(correct_answer, merged)

    if hint_bundle:
        h_ranker, h_vec = hint_bundle
        hints_map = generate_hints(
            article_text,
            question,
            options[correct_index],
            h_ranker,
            h_vec,
        )
        hints = [hints_map["hint_1"], hints_map["hint_2"], hints_map["hint_3"]]
    else:
        hints = hints_default

    option_probs, explanation, v_backend = run_verification(question, options)
    if isinstance(option_probs, np.ndarray):
        op_list = option_probs.tolist()
    else:
        op_list = None

    return {
        "question": question,
        "options": options,
        "correct_index": correct_index,
        "hints": hints,
        "explanation": explanation
        or "Generated quiz: ranker distractors + extractive key; verify with the passage.",
        "quiz_mode": "generated",
        "models": flags,
        "option_probs": op_list,
        "predicted_correct_index": int(np.argmax(option_probs)) if option_probs is not None else None,
        "question_source": question_source,
        "question_anchor": question_anchor,
        "verification_backend": v_backend,
        "phase4_summary": phase4_summary,
        "stacking_note": stacking_note,
    }


def empty_result() -> dict:
    z = _artifact_status_extended(DEFAULT_MODEL_A_DIR, DEFAULT_DISTRACTOR_DIR, DEFAULT_HINT_DIR)
    for k in list(z.keys()):
        z[k] = False
    return {
        "question": "No article provided",
        "options": ["—", "—", "—", "—"],
        "correct_index": 0,
        "hints": ["", "", ""],
        "explanation": "",
        "quiz_mode": "empty",
        "models": z,
        "option_probs": None,
        "predicted_correct_index": None,
        "question_source": "",
        "question_anchor": "",
        "verification_backend": "",
        "phase4_summary": None,
        "stacking_note": None,
    }
