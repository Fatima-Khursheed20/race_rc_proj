from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text, build_vectorizer

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "as", "that", "this", "these", "those",
}
OPTION_COLS = ["A", "B", "C", "D"]


def normalize_text(text: str) -> str:
    return clean_text(str(text))


def tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"\b\w+\b", text.lower()) if token]


def extract_ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def count_phrase_in_text(phrase: str, text: str) -> int:
    pattern = re.escape(phrase)
    return len(re.findall(rf"\b{pattern}\b", text, flags=re.IGNORECASE))


def build_candidate_pool(
    article: str,
    correct_answer: str,
    min_count: int = 2,
    max_candidates: int = 100,
) -> List[str]:
    article_raw = str(article)
    article_clean = normalize_text(article_raw)
    answer_clean = normalize_text(correct_answer)
    article_tokens = re.findall(r"\b\w+\b", article_raw)

    candidates = []
    for n in (1, 2, 3):
        for ngram in extract_ngrams(article_tokens, n):
            ngram_clean = ngram.strip()
            ngram_norm = normalize_text(ngram_clean)
            if not ngram_clean or ngram_norm == answer_clean:
                continue
            if n == 1 and ngram_norm in STOPWORDS:
                continue
            if count_phrase_in_text(ngram_clean, article_raw) < min_count:
                continue
            if len(ngram_clean) < 2:
                continue
            answer_length = max(len(answer_clean), 1)
            if len(ngram_norm) > 1.5 * answer_length or len(ngram_norm) < 0.5 * answer_length:
                continue
            candidates.append(ngram_clean)

    candidates = list(dict.fromkeys(candidates))

    # Add actual answer options and correct answer variants to keep positive labels trainable.
    if answer_clean and answer_clean not in candidates:
        candidates.append(answer_clean)

    if len(candidates) > max_candidates:
        candidates = sorted(candidates, key=lambda cand: -count_phrase_in_text(cand, article_clean))[:max_candidates]

    return candidates


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def cleanup_tokens(text: str) -> List[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


def compute_candidate_features(
    candidate: str,
    article: str,
    question: str,
    correct_answer: str,
    vectorizer: CountVectorizer,
) -> np.ndarray:
    raw_candidate = str(candidate).strip()
    candidate = normalize_text(candidate)
    article = normalize_text(article)
    question = normalize_text(question)
    correct_answer = normalize_text(correct_answer)

    candidate_vec = vectorizer.transform([candidate])
    answer_vec = vectorizer.transform([correct_answer])
    question_vec = vectorizer.transform([question])

    sim_to_answer = float(cosine_similarity(candidate_vec, answer_vec)[0, 0])
    sim_to_question = float(cosine_similarity(candidate_vec, question_vec)[0, 0])

    candidate_chars = len(candidate)
    answer_chars = max(len(correct_answer), 1)
    candidate_words = candidate.split()
    answer_words = correct_answer.split()
    question_words = question.split()

    passage_frequency = float(count_phrase_in_text(candidate, article))
    position = article.find(candidate)
    position_ratio = float(position / max(len(article), 1)) if position >= 0 else 0.0

    overlap_with_answer = safe_ratio(
        len(set(candidate_words) & set(answer_words)), max(len(answer_words), 1)
    )
    overlap_with_question = safe_ratio(
        len(set(candidate_words) & set(question_words)), max(len(question_words), 1)
    )
    starts_with_same_word = int(candidate_words[0] == answer_words[0]) if candidate_words and answer_words else 0
    is_proper_noun_candidate = int(raw_candidate[:1].isupper()) if raw_candidate else 0

    return np.array(
        [
            sim_to_answer,
            sim_to_question,
            safe_ratio(candidate_chars, answer_chars),
            safe_ratio(len(candidate_words), max(len(answer_words), 1)),
            passage_frequency,
            position_ratio,
            overlap_with_answer,
            overlap_with_question,
            starts_with_same_word,
            is_proper_noun_candidate,
        ],
        dtype=float,
    )


def compute_candidate_features_batch(
    candidates: List[str],
    article: str,
    question: str,
    correct_answer: str,
    vectorizer: CountVectorizer,
) -> np.ndarray:
    article_clean = normalize_text(article)
    question_clean = normalize_text(question)
    correct_answer_clean = normalize_text(correct_answer)
    
    answer_words = correct_answer_clean.split()
    question_words = question_clean.split()
    
    answer_vec = vectorizer.transform([correct_answer_clean])
    question_vec = vectorizer.transform([question_clean])
    
    features_list = []
    
    for candidate in candidates:
        raw_candidate = str(candidate).strip()
        candidate_clean = normalize_text(candidate)
        candidate_vec = vectorizer.transform([candidate_clean])
        
        sim_to_answer = float(cosine_similarity(candidate_vec, answer_vec)[0, 0])
        sim_to_question = float(cosine_similarity(candidate_vec, question_vec)[0, 0])
        
        candidate_chars = len(candidate_clean)
        answer_chars = max(len(correct_answer_clean), 1)
        candidate_words = candidate_clean.split()
        
        passage_frequency = float(count_phrase_in_text(candidate_clean, article_clean))
        position = article_clean.find(candidate_clean)
        position_ratio = float(position / max(len(article_clean), 1)) if position >= 0 else 0.0
        
        overlap_with_answer = safe_ratio(
            len(set(candidate_words) & set(answer_words)), max(len(answer_words), 1)
        )
        overlap_with_question = safe_ratio(
            len(set(candidate_words) & set(question_words)), max(len(question_words), 1)
        )
        starts_with_same_word = int(candidate_words[0] == answer_words[0]) if candidate_words and answer_words else 0
        is_proper_noun_candidate = int(raw_candidate[:1].isupper()) if raw_candidate else 0
        
        features_list.append(
            np.array(
                [
                    sim_to_answer,
                    sim_to_question,
                    safe_ratio(candidate_chars, answer_chars),
                    safe_ratio(len(candidate_words), max(len(answer_words), 1)),
                    passage_frequency,
                    position_ratio,
                    overlap_with_answer,
                    overlap_with_question,
                    starts_with_same_word,
                    is_proper_noun_candidate,
                ],
                dtype=float,
            )
        )
    
    return np.vstack(features_list) if features_list else np.array([]).reshape(0, 10)


def phrase_cosine_similarity(a: str, b: str, vectorizer: CountVectorizer) -> float:
    a_vec = vectorizer.transform([a])
    b_vec = vectorizer.transform([b])
    return float(cosine_similarity(a_vec, b_vec)[0, 0])


def select_diverse_distractors(
    candidates: List[str],
    probabilities: np.ndarray,
    vectorizer: CountVectorizer,
    lambda_param: float = 0.5,
    top_k: int = 3,
) -> List[str]:
    if not candidates:
        return []
    selected: List[str] = []
    probs = probabilities.copy()
    remaining = candidates.copy()

    while len(selected) < top_k and remaining:
        if not selected:
            best_idx = int(np.argmax(probs))
        else:
            diversity_scores = []
            for idx, candidate in enumerate(remaining):
                similarities = [phrase_cosine_similarity(candidate, chosen, vectorizer) for chosen in selected]
                max_sim = max(similarities) if similarities else 0.0
                diversity_scores.append(probs[idx] - lambda_param * max_sim)
            best_idx = int(np.argmax(diversity_scores))

        selected.append(remaining.pop(best_idx))
        probs = np.delete(probs, best_idx)

    return selected


def generate_distractors(
    article: str,
    question: str,
    correct_answer: str,
    ranker,
    vectorizer: CountVectorizer,
    min_count: int = 2,
    max_candidates: int = 100,
    lambda_param: float = 0.5,
) -> List[str]:
    candidates = build_candidate_pool(article, correct_answer, min_count=min_count, max_candidates=max_candidates)
    candidates = [cand for cand in candidates if cand != normalize_text(correct_answer)]
    if not candidates:
        return []

    features = np.vstack(
        [compute_candidate_features(cand, article, question, correct_answer, vectorizer) for cand in candidates]
    )
    probabilities = ranker.predict_proba(features)[:, 1]
    top_indices = np.argsort(probabilities)[::-1]
    sorted_candidates = [candidates[i] for i in top_indices]
    sorted_probs = probabilities[top_indices]

    return select_diverse_distractors(sorted_candidates, sorted_probs, vectorizer, lambda_param=lambda_param, top_k=3)


def load_ranker(model_dir: Path):
    model_path = model_dir / "distractor_ranker.pkl"
    vectorizer_path = model_dir / "distractor_vectorizer.pkl"
    ranker = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return ranker, vectorizer


def save_ranker(ranker, vectorizer: CountVectorizer, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ranker, output_dir / "distractor_ranker.pkl")
    joblib.dump(vectorizer, output_dir / "distractor_vectorizer.pkl")


def candidate_label(candidates: List[str], distractors: List[str]) -> List[int]:
    distractor_set = {normalize_text(d) for d in distractors}
    return [int(normalize_text(candidate) in distractor_set) for candidate in candidates]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate distractors using a trained model")
    parser.add_argument("--article", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--correct-answer", type=str, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/model_b/distractor"))
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--lambda", type=float, default=0.5, dest="lambda_param")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ranker, vectorizer = load_ranker(args.model_dir)
    distractors = generate_distractors(
        article=args.article,
        question=args.question,
        correct_answer=args.correct_answer,
        ranker=ranker,
        vectorizer=vectorizer,
        min_count=args.min_count,
        max_candidates=args.max_candidates,
        lambda_param=args.lambda_param,
    )
    print("Generated distractors:")
    for idx, distractor in enumerate(distractors, start=1):
        print(f"{idx}. {distractor}")
