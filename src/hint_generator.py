"""Hint generator module for Model B.

This module generates graduated hints for reading comprehension questions
using extractive methods. It ranks sentences from the article by relevance
to the question and correct answer, then selects three hints with increasing
specificity.

Run from project root:
    python src/hint_generator.py \
        --article "..." \
        --question "..." \
        --correct-answer "..." \
        --model-dir models/model_b/hint_generator
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from preprocessing import clean_text

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "as", "that", "this", "these", "those",
}


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return clean_text(str(text))


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    return [token for token in re.findall(r"\b\w+\b", text.lower()) if token]


def split_into_sentences(text: str) -> List[str]:
    """Split article text into sentences.
    
    Uses simple regex split on sentence delimiters, then filters and cleans.
    """
    # Split on sentence delimiters
    sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
    
    # Clean and filter
    cleaned = []
    for sent in sentences:
        sent = sent.strip()
        # Keep only sentences with at least 5 words
        if len(tokenize(sent)) >= 5:
            cleaned.append(sent)
    
    return cleaned


def get_word_overlap_tokens(tokens: List[str]) -> List[str]:
    """Filter tokens for overlap computation (remove stopwords)."""
    return [t for t in tokens if t not in STOPWORDS]


def compute_word_overlap_ratio(text1: str, text2: str) -> float:
    """Compute ratio of words from text2 that appear in text1."""
    tokens1 = set(get_word_overlap_tokens(tokenize(text1)))
    tokens2 = get_word_overlap_tokens(tokenize(text2))
    
    if not tokens2:
        return 0.0
    
    overlap = len([t for t in tokens2 if t in tokens1])
    return overlap / len(tokens2)


def contains_wh_cue(sentence: str, question: str) -> int:
    """Check if sentence contains WH-cue relevant to the question.
    
    If question starts with who/what/why/where/when/how, check if
    the sentence contains relevant keywords.
    """
    q_lower = question.lower().split()
    if not q_lower:
        return 0
    
    first_word = q_lower[0]
    wh_keywords = {
        'who': ['person', 'people', 'man', 'woman', 'boy', 'girl', 'character'],
        'what': ['thing', 'object', 'event', 'reason', 'idea'],
        'why': ['because', 'reason', 'caused', 'caused', 'due'],
        'where': ['place', 'location', 'at', 'in', 'near'],
        'when': ['time', 'date', 'before', 'after', 'during'],
        'how': ['way', 'method', 'process', 'through'],
    }
    
    s_lower = sentence.lower()
    for wh in ['who', 'what', 'why', 'where', 'when', 'how']:
        if first_word.startswith(wh) and wh in wh_keywords:
            if any(kw in s_lower for kw in wh_keywords[wh]):
                return 1
    
    return 0


def compute_keyword_density(sentence: str, question: str, correct_answer: str) -> float:
    """Fraction of non-stopwords in sentence that appear in question or answer."""
    s_tokens = get_word_overlap_tokens(tokenize(sentence))
    qa_tokens = set(get_word_overlap_tokens(tokenize(question + " " + correct_answer)))
    
    if not s_tokens:
        return 0.0
    
    density = len([t for t in s_tokens if t in qa_tokens]) / len(s_tokens)
    return density


def find_answer_sentence_idx(article: str, correct_answer: str, sentences: List[str]) -> int:
    """Find the sentence that contains the correct answer.
    
    Returns -1 if not found.
    """
    answer_tokens = set(get_word_overlap_tokens(tokenize(correct_answer)))
    
    for idx, sent in enumerate(sentences):
        sent_tokens = set(get_word_overlap_tokens(tokenize(sent)))
        if answer_tokens & sent_tokens:
            return idx
    
    return -1


def compute_distance_to_answer(sentence_idx: int, answer_sent_idx: int, total_sentences: int) -> float:
    """Compute normalized distance from sentence to the answer sentence.
    
    Normalized by total sentences.
    """
    if answer_sent_idx < 0:
        return 0.0
    
    distance = abs(sentence_idx - answer_sent_idx)
    return 1.0 - (distance / max(total_sentences, 1))


def compute_sentence_features(
    sentence: str,
    article: str,
    question: str,
    correct_answer: str,
    vectorizer: CountVectorizer | TfidfVectorizer,
    sentences: List[str],
    sentence_idx: int,
) -> np.ndarray:
    """Compute feature vector for a single sentence.
    
    Features:
    1. word_overlap_with_question
    2. word_overlap_with_answer
    3. sentence_position_ratio
    4. sentence_length
    5. contains_named_entity_proxy (capitalized tokens)
    6. contains_question_wh_word
    7. keyword_density
    8. distance_to_answer_sentence
    
    Returns: feature vector of shape (8,)
    """
    total_sentences = len(sentences)
    answer_sent_idx = find_answer_sentence_idx(article, correct_answer, sentences)
    
    # Feature 1: word overlap with question
    overlap_q = compute_word_overlap_ratio(sentence, question)
    
    # Feature 2: word overlap with answer
    overlap_a = compute_word_overlap_ratio(sentence, correct_answer)
    
    # Feature 3: sentence position ratio
    position_ratio = sentence_idx / max(total_sentences - 1, 1)
    
    # Feature 4: sentence length (normalized by median)
    sent_len = len(tokenize(sentence))
    # Assume median sentence length ~15 words
    norm_len = min(sent_len / 15.0, 3.0)
    
    # Feature 5: contains named entity proxy (capitalized tokens)
    capitalized_count = len([t for t in re.findall(r'\b\w+\b', sentence) if t[0].isupper() and t[0] not in '.!?'])
    ne_proxy = min(capitalized_count / max(len(tokenize(sentence)), 1), 1.0)
    
    # Feature 6: contains question WH word
    wh_cue = float(contains_wh_cue(sentence, question))
    
    # Feature 7: keyword density
    kw_density = compute_keyword_density(sentence, question, correct_answer)
    
    # Feature 8: distance to answer sentence
    dist_to_answer = compute_distance_to_answer(sentence_idx, answer_sent_idx, total_sentences)
    
    features = np.array([
        overlap_q,
        overlap_a,
        position_ratio,
        norm_len,
        ne_proxy,
        wh_cue,
        kw_density,
        dist_to_answer,
    ], dtype=np.float32)
    
    return features


def compute_sentence_features_batch(
    sentences: List[str],
    article: str,
    question: str,
    correct_answer: str,
    vectorizer: CountVectorizer | TfidfVectorizer,
) -> np.ndarray:
    """Compute features for all sentences.
    
    Returns: feature matrix of shape (len(sentences), 8)
    """
    features = []
    for idx, sent in enumerate(sentences):
        feat = compute_sentence_features(
            sent, article, question, correct_answer, vectorizer, sentences, idx
        )
        features.append(feat)
    
    return np.vstack(features) if features else np.zeros((0, 8), dtype=np.float32)


def label_gold_hint_sentence(
    sentences: List[str],
    article: str,
    correct_answer: str,
) -> int:
    """Label the gold hint sentence using word overlap proxy.
    
    The sentence with maximum word overlap with the correct answer
    is labeled as gold (positive). If tie, choose the sentence closest
    to the correct answer in the text.
    
    Returns: index of gold sentence, or 0 if no candidate found.
    """
    answer_tokens = set(get_word_overlap_tokens(tokenize(correct_answer)))
    
    max_overlap = -1
    best_idx = 0
    
    for idx, sent in enumerate(sentences):
        sent_tokens = set(get_word_overlap_tokens(tokenize(sent)))
        overlap = len(answer_tokens & sent_tokens)
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_idx = idx
    
    return best_idx if max_overlap > 0 else 0


def save_hint_model(model: LogisticRegression, vectorizer, model_dir: Path) -> None:
    """Save trained hint ranker and vectorizer."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_dir / "hint_ranker.pkl")
    if vectorizer is not None:
        joblib.dump(vectorizer, model_dir / "hint_vectorizer.pkl")


def load_hint_model(model_dir: Path) -> Tuple[LogisticRegression, CountVectorizer | TfidfVectorizer | None]:
    """Load trained hint ranker and vectorizer."""
    model_dir = Path(model_dir)
    
    ranker = joblib.load(model_dir / "hint_ranker.pkl")
    
    vectorizer = None
    if (model_dir / "hint_vectorizer.pkl").exists():
        vectorizer = joblib.load(model_dir / "hint_vectorizer.pkl")
    
    return ranker, vectorizer


def generate_hints(
    article: str,
    question: str,
    correct_answer: str,
    ranker: LogisticRegression,
    vectorizer: CountVectorizer | TfidfVectorizer | None = None,
) -> Dict[str, str]:
    """Generate three hints for a question.
    
    Args:
        article: article text
        question: question text
        correct_answer: the correct answer
        ranker: trained LogisticRegression model
        vectorizer: feature vectorizer (can be None)
    
    Returns:
        Dictionary with keys 'hint_1', 'hint_2', 'hint_3'
    """
    # Split into sentences
    sentences = split_into_sentences(article)
    
    if len(sentences) == 0:
        return {
            "hint_1": "No hints available.",
            "hint_2": "No hints available.",
            "hint_3": "No hints available.",
        }
    
    # If fewer than 3 sentences, pad with the best sentence
    if len(sentences) < 3:
        sentences = sentences + [sentences[0]] * (3 - len(sentences))
    
    # Compute features
    X = compute_sentence_features_batch(
        sentences, article, question, correct_answer, vectorizer
    )
    
    # Score sentences
    scores = ranker.predict_proba(X)[:, 1]  # probability of positive class
    
    # Get top 3 indices
    top_3_indices = np.argsort(scores)[::-1][:3]
    top_3_indices = sorted(top_3_indices)[::-1]  # Reverse: highest score first
    
    # Assign hints: Hint 3 (strongest) = top score, Hint 2 = 2nd, Hint 1 = 3rd
    hint_3_idx = top_3_indices[0]
    hint_2_idx = top_3_indices[1] if len(top_3_indices) > 1 else top_3_indices[0]
    hint_1_idx = top_3_indices[2] if len(top_3_indices) > 2 else top_3_indices[0]
    
    # Extract hint sentences
    hint_1 = sentences[hint_1_idx].strip()
    hint_2 = sentences[hint_2_idx].strip()
    hint_3 = sentences[hint_3_idx].strip()
    
    # Guardrails: avoid if too short or directly contains answer
    answer_words = set(get_word_overlap_tokens(tokenize(correct_answer)))
    
    for hints_idx in range(3):
        if hints_idx == 0:
            hint = hint_1
            idx = hint_1_idx
        elif hints_idx == 1:
            hint = hint_2
            idx = hint_2_idx
        else:
            hint = hint_3
            idx = hint_3_idx
        
        # Check if too short
        if len(tokenize(hint)) < 5:
            # Find next best sentence
            sorted_indices = np.argsort(scores)[::-1]
            for new_idx in sorted_indices:
                if new_idx not in [hint_1_idx, hint_2_idx, hint_3_idx]:
                    if len(tokenize(sentences[new_idx])) >= 5:
                        if hints_idx == 0:
                            hint_1_idx = new_idx
                            hint_1 = sentences[new_idx].strip()
                        elif hints_idx == 1:
                            hint_2_idx = new_idx
                            hint_2 = sentences[new_idx].strip()
                        else:
                            hint_3_idx = new_idx
                            hint_3 = sentences[new_idx].strip()
                        break
        
        # Check if directly states the answer
        hint_words = set(get_word_overlap_tokens(tokenize(hint)))
        if len(answer_words & hint_words) == len(answer_words) and len(answer_words) > 0:
            # Try to find an alternative
            sorted_indices = np.argsort(scores)[::-1]
            found = False
            for new_idx in sorted_indices:
                if new_idx not in [hint_1_idx, hint_2_idx, hint_3_idx]:
                    new_hint_words = set(get_word_overlap_tokens(tokenize(sentences[new_idx])))
                    if len(answer_words & new_hint_words) < len(answer_words):
                        if hints_idx == 0:
                            hint_1_idx = new_idx
                            hint_1 = sentences[new_idx].strip()
                        elif hints_idx == 1:
                            hint_2_idx = new_idx
                            hint_2 = sentences[new_idx].strip()
                        else:
                            hint_3_idx = new_idx
                            hint_3 = sentences[new_idx].strip()
                        found = True
                        break
            if not found:
                # Keep the current hint anyway
                pass
    
    return {
        "hint_1": hint_1,
        "hint_2": hint_2,
        "hint_3": hint_3,
    }


def main():
    """Command-line interface for hint generation."""
    parser = argparse.ArgumentParser(description="Generate hints for a reading comprehension question.")
    parser.add_argument("--article", type=str, required=True, help="Article text")
    parser.add_argument("--question", type=str, required=True, help="Question text")
    parser.add_argument("--correct-answer", type=str, required=True, help="Correct answer")
    parser.add_argument("--model-dir", type=str, default="models/model_b/hint_generator", help="Model directory")
    
    args = parser.parse_args()
    
    ranker, vectorizer = load_hint_model(args.model_dir)
    
    hints = generate_hints(
        args.article,
        args.question,
        args.correct_answer,
        ranker,
        vectorizer,
    )
    
    print("Generated Hints:")
    print(f"  Hint 1: {hints['hint_1']}")
    print(f"  Hint 2: {hints['hint_2']}")
    print(f"  Hint 3: {hints['hint_3']}")


if __name__ == "__main__":
    main()
