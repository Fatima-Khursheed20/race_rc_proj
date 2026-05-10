"""Template + ML-ranker question generation (Phase 3 notebook parity).

Loads ``qg_ranker.pkl`` (RandomForest) trained on ranker_features.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from hint_generator import split_into_sentences, tokenize

COMMON_CAPS = {
    "The",
    "In",
    "It",
    "At",
    "On",
    "By",
    "An",
    "A",
    "This",
    "That",
    "These",
    "Those",
    "He",
    "She",
    "They",
    "We",
    "I",
    "His",
    "Her",
    "Their",
    "Its",
    "Our",
    "Was",
    "Were",
    "Has",
    "Have",
    "Had",
    "Is",
    "Are",
    "Am",
    "But",
    "And",
    "Or",
    "So",
    "Yet",
    "For",
    "Born",
}

# Keep anchors quiz-sized so WH-templates are not applied to an entire passage.
MAX_ANCHOR_WORDS = 42
MAX_ANCHOR_CHARS = 320
MIN_SENT_WORDS = 5
# "Who" + rest-of-sentence is only grammatical for short, person-focused clauses.
MAX_WORDS_FOR_WHO_TEMPLATE = 28


def detect_name(tokens: List[str], idx: int) -> bool:
    w = tokens[idx]
    return (
        len(w) > 1
        and w[0].isupper()
        and idx != 0
        and w not in COMMON_CAPS
        and w.isalpha()
    )


def _normalize_punctuation(text: str) -> str:
    """Insert space after commas stuck to following word (e.g. '1893,Bessie')."""
    t = re.sub(r"\s+", " ", str(text).strip())
    t = re.sub(r",(\S)", r", \1", t)
    # Period / ! / ? directly followed by a letter (no space) breaks sentence splitters.
    t = re.sub(r"([.!?])([A-Za-z\"'])", r"\1 \2", t)
    return t


def _split_oversized_segment(segment: str) -> List[str]:
    """Break long run-on clauses on common discourse boundaries."""
    s = segment.strip()
    if len(s) <= MAX_ANCHOR_CHARS:
        return [s] if len(tokenize(s)) >= MIN_SENT_WORDS else []

    parts = re.split(
        r"\s+(?:Then|But|So|Sadly|Soon|At the age of)\s+",
        s,
        flags=re.IGNORECASE,
    )
    chunks = [p.strip() for p in parts if len(tokenize(p.strip())) >= MIN_SENT_WORDS]
    return chunks if chunks else ([s[:MAX_ANCHOR_CHARS]] if s else [])


def segment_article_for_qg(article: str) -> List[str]:
    """Robust sentence list: . ! ? boundaries, comma+caps, and long-clause splits."""
    text = _normalize_punctuation(article)
    raw = split_into_sentences(text)
    sentences: List[str] = []
    for sent in raw:
        if len(sent) > MAX_ANCHOR_CHARS or len(tokenize(sent)) > MAX_ANCHOR_WORDS:
            sentences.extend(_split_oversized_segment(sent))
        else:
            sentences.append(sent)

    if len(sentences) <= 1 and len(text) > MAX_ANCHOR_CHARS:
        parts = re.split(r"(?<=[.!?])(?:\s+|$)|,\s+(?=[A-Z])", text)
        extra = [p.strip() for p in parts if len(tokenize(p.strip())) >= MIN_SENT_WORDS]
        if len(extra) > 1:
            sentences = extra

    seen = set()
    out: List[str] = []
    for s in sentences:
        s = s.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        s = _cap_anchor(s)
        if len(tokenize(s)) >= MIN_SENT_WORDS:
            out.append(s)
    return out


def _cap_anchor(sentence: str) -> str:
    toks = sentence.split()
    if len(toks) <= MAX_ANCHOR_WORDS and len(sentence) <= MAX_ANCHOR_CHARS:
        return sentence
    clipped = toks[:MAX_ANCHOR_WORDS]
    return " ".join(clipped) + ("…" if len(toks) > MAX_ANCHOR_WORDS else "")


def generate_candidates_from_sentence(sentence: str) -> List[str]:
    """Apply WH templates; avoid ungrammatical 'Who' + full biography."""
    sentence = _cap_anchor(sentence.strip())
    tokens = sentence.split()
    candidates: List[str] = []

    if len(tokens) < MIN_SENT_WORDS:
        return []

    head = " ".join(tokens[:12])

    for t in tokens:
        if re.fullmatch(r"\d{4}", t):
            clip = " ".join(tokens[: min(len(tokens), 22)])
            rest = clip.replace(t, "___", 1)
            candidates.append(f"When {rest}?")
            break

    for i, t in enumerate(tokens):
        if t.lower() in ("in", "at") and i + 1 < len(tokens):
            tail = " ".join(tokens[i + 1 : min(len(tokens), i + 18)])
            if tail:
                candidates.append(f"Where {tail}?")
            break

    for i, t in enumerate(tokens):
        if t.lower() in ("because", "since") and i + 1 < len(tokens):
            tail = " ".join(tokens[i + 1 : min(len(tokens), i + 20)])
            if tail:
                candidates.append(f"Why {tail}?")
            break

    if len(tokens) <= MAX_WORDS_FOR_WHO_TEMPLATE:
        for i, _ in enumerate(tokens[:22]):
            if detect_name(tokens, i):
                name = tokens[i]
                rest = " ".join(tokens[i + 1 : min(len(tokens), i + 15)])
                if rest:
                    candidates.append(f"What does the passage say {name} did regarding: {rest}?")
                break

    short = " ".join(tokens[:8])
    candidates.append(f'According to the passage, what is stated about "{short}"?')
    if not any(c.startswith("When ") for c in candidates):
        candidates.append(f"What information does the passage give about: {head}?")

    # De-duplicate preserving order
    out: List[str] = []
    seen: set[str] = set()
    for c in candidates:
        key = c.lower().strip()
        if key not in seen and len(c) < 500:
            seen.add(key)
            out.append(c)
    return out


def select_top_sentences(
    article: str, answer_text: str, top_n: int = 3
) -> Tuple[List[Tuple[int, int, str]], int]:
    """Return top_n sentences as (overlap_score, sentence_index_in_list, sentence)."""
    sentences = segment_article_for_qg(article)
    if not sentences:
        chunk = _normalize_punctuation(article)[:MAX_ANCHOR_CHARS]
        if len(tokenize(chunk)) >= MIN_SENT_WORDS:
            sentences = [chunk]

    answer_words = set(str(answer_text).lower().split())
    scored: List[Tuple[int, int, str]] = [
        (len(set(s.lower().split()) & answer_words), i, s) for i, s in enumerate(sentences)
    ]
    scored.sort(reverse=True)
    return scored[:top_n], len(sentences)


def ranker_features(question: str, anchor: str, anchor_pos: int, article_len: int) -> List[float]:
    q_tok = question.split()
    a_tok = anchor.split()
    wh_words = {"who", "what", "where", "when", "why", "how", "which"}
    overlap = len({t.lower() for t in q_tok} & {t.lower() for t in a_tok})
    has_wh = int(any(t.lower() in wh_words for t in q_tok))
    return [
        float(len(q_tok)),
        float(has_wh),
        float(overlap),
        float(anchor_pos / max(article_len, 1)),
        float(len(a_tok)),
    ]


def load_qg_ranker(model_dir: Path | str) -> RandomForestClassifier:
    return joblib.load(Path(model_dir) / "qg_ranker.pkl")


def generate_question_ranked(
    article: str,
    answer_text: str,
    ranker: Optional[RandomForestClassifier] = None,
    top_n: int = 3,
) -> Tuple[str, str]:
    """Return (best_question, anchor_sentence). Anchor may be empty if fallback."""
    article_s = str(article).strip()
    answer_s = str(answer_text).strip()
    if not article_s:
        return "What is the main idea of the passage?", ""

    top_sents, art_len = select_top_sentences(article_s, answer_s, top_n=top_n)
    if not top_sents or ranker is None:
        if top_sents:
            _, pos, anchor = top_sents[0]
            cands = generate_candidates_from_sentence(anchor)
            return (cands[0], anchor) if cands else ("What is the main idea of the passage?", anchor)
        return "What is the main idea of the passage?", ""

    all_candidates: List[Tuple[float, str, str]] = []
    for _, pos, anchor in top_sents:
        for cand in generate_candidates_from_sentence(anchor):
            if len(cand) > 450 or cand.count("?") > 1:
                continue
            feats = ranker_features(cand, anchor, pos, art_len)
            score = float(ranker.predict_proba(np.array([feats], dtype=np.float32))[0, 1])
            all_candidates.append((score, cand, anchor))

    if not all_candidates:
        return "What is the main idea of the passage?", ""

    all_candidates.sort(reverse=True)
    return all_candidates[0][1], all_candidates[0][2]
