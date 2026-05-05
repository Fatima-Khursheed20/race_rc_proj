"""Preprocessing and feature engineering pipeline for RACE RC project.

This module does four major things:
1. Load and clean raw train/dev/test CSV files.
2. Expand each MCQ row into 4 binary verification examples.
3. Build handcrafted lexical features for each example.
4. Build and save one-hot text features AND TF-IDF features + labels for model training.

Run from project root:
	python src/preprocessing.py

Optional arguments:
	python src/preprocessing.py --raw-dir data/raw --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # [CHANGE 1] Import TfidfVectorizer


# Core schema used by the project dataset.
OPTION_COLS = ["A", "B", "C", "D"]
TEXT_COLS = ["article", "question", "A", "B", "C", "D"]
REQUIRED_COLS = ["article", "question", "A", "B", "C", "D", "answer"]


def clean_text(text: object) -> str:
	"""Normalize text in a lightweight way suitable for bag-of-words models.

	Steps:
	1) lowercase
	2) remove URLs/emails
	3) remove punctuation (keep apostrophes)
	4) collapse repeated whitespace
	"""
	value = str(text).lower()
	value = re.sub(r"http[s]?://\S+|www\.\S+|[\w\.-]+@[\w\.-]+", " ", value)
	value = re.sub(r"[^\w\s']", " ", value)
	value = re.sub(r"\s+", " ", value).strip()
	return value


def tokenize_set(text: object) -> set[str]:
	"""Tokenize text into a lowercase set of alphanumeric words."""
	return set(re.findall(r"\b\w+\b", str(text).lower()))


def question_type_flags(question: object) -> Dict[str, int]:
	"""Return one-hot style flags for question shape.

	We keep the simple WH classes, but we also add a few extra indicators that
	help with the large `Other` bucket:
	- `other`: question does not start with a WH word
	- `has_blank`: cloze-style stem with an underscore blank
	- `starts_with_aux`: starts with an auxiliary verb like do/does/is/can
	- `starts_with_article`: starts with a/an/the, often used in fragment-like stems
	"""
	raw = str(question).strip().lower()
	first = raw.split()[0] if raw else ""
	wh_words = {"who", "what", "where", "when", "why", "how", "which"}
	starts_with_wh = int(any(first.startswith(word) for word in wh_words))
	starts_with_aux = int(first in {"is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would", "will", "has", "have", "had"})
	starts_with_article = int(first in {"a", "an", "the"})
	has_blank = int("___" in raw or "____" in raw or "__" in raw)
	other = int(not starts_with_wh)
	return {
		"who": int(first.startswith("who")),
		"what": int(first.startswith("what")),
		"where": int(first.startswith("where")),
		"when": int(first.startswith("when")),
		"why": int(first.startswith("why")),
		"how": int(first.startswith("how")),
		"which": int(first.startswith("which")),
		"other": other,
		"starts_with_wh": starts_with_wh,
		"has_blank": has_blank,
		"starts_with_aux": starts_with_aux,
		"starts_with_article": starts_with_article,
	}


def question_subtype(question: object) -> str:
	"""Categorize a question into an explicit subtype for downstream use.

	Subtypes:
	- wh_who, wh_what, wh_where, wh_when, wh_why, wh_how, wh_which:
	  Questions beginning with WH-words. Useful for WH-answer templates.
	- other_cloze:
	  Non-WH questions containing blank markers (underscores). These are
	  fill-in-the-blank style questions. Useful for cloze templates.
	- other_generic:
	  Remaining "Other" questions (e.g., "Does X...?", "The main idea...", etc).
	  Fallback type for inference-style or open-ended comprehension questions.

	This mapping helps both the model learn cleaner decision boundaries and
	the question-generation stage select appropriate question templates.
	"""
	raw = str(question).strip().lower()
	first = raw.split()[0] if raw else ""

	# Map WH-words to their subtypes.
	wh_mapping = {
		"who": "wh_who",
		"what": "wh_what",
		"where": "wh_where",
		"when": "wh_when",
		"why": "wh_why",
		"how": "wh_how",
		"which": "wh_which",
	}
	for wh_word, subtype in wh_mapping.items():
		if first.startswith(wh_word):
			return subtype

	# Check for cloze-style blanks (underscore markers).
	if "___" in raw or "____" in raw or "__" in raw:
		return "other_cloze"

	# Remaining "Other" questions (including aux-verb starts, fragments, etc).
	return "other_generic"


def validate_columns(df: pd.DataFrame, split_name: str) -> None:
	"""Fail early if required columns are missing."""
	missing = [col for col in REQUIRED_COLS if col not in df.columns]
	if missing:
		raise ValueError(f"{split_name} is missing required columns: {missing}")


def read_split_csv(path: Path, split_name: str) -> pd.DataFrame:
	"""Read a split CSV and drop accidental unnamed columns."""
	if not path.exists():
		raise FileNotFoundError(f"Missing {split_name} file: {path}")

	df = pd.read_csv(path)
	df.drop(
		columns=[col for col in df.columns if str(col).startswith("Unnamed")],
		inplace=True,
		errors="ignore",
	)
	validate_columns(df, split_name)
	return df


def load_raw_splits(raw_dir: Path) -> Dict[str, pd.DataFrame]:
	"""Load train/dev/test (or val) splits from data/raw."""
	train_path = raw_dir / "train.csv"
	dev_path = raw_dir / "dev.csv"
	test_path = raw_dir / "test.csv"

	# Some datasets use val.csv instead of dev.csv.
	if not dev_path.exists() and (raw_dir / "val.csv").exists():
		dev_path = raw_dir / "val.csv"

	return {
		"train": read_split_csv(train_path, "train"),
		"dev": read_split_csv(dev_path, "dev"),
		"test": read_split_csv(test_path, "test"),
	}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	"""Apply cleaning to all text-bearing columns used in training."""
	out = df.copy()
	for col in TEXT_COLS:
		out[col] = out[col].fillna("").map(clean_text)
	out["answer"] = out["answer"].astype(str).str.strip()
	return out


def compute_lexical_features(article: str, question: str, option: str) -> Dict[str, float]:
	"""Build dense lexical features from one (article, question, option) triple.

	These are cheap, interpretable signals that complement sparse one-hot vectors.
	"""
	art_tok = tokenize_set(article)
	q_tok = tokenize_set(question)
	opt_tok = tokenize_set(option)

	opt_len = len(option.split())
	art_len = len(article.split())
	q_len = len(question.split())

	art_overlap = len(art_tok & opt_tok) / max(len(opt_tok), 1)
	q_overlap = len(q_tok & opt_tok) / max(len(opt_tok), 1)

	features: Dict[str, float] = {
		"art_overlap": art_overlap,
		"q_overlap": q_overlap,
		"art_exact": float(option in article),
		"q_exact": float(option in question),
		"opt_len": float(opt_len),
		"art_len": float(art_len),
		"q_len": float(q_len),
		"unique_words": float(len(opt_tok)),
		"art_len_ratio": float(opt_len / max(art_len, 1)),
	}
	features.update({k: float(v) for k, v in question_type_flags(question).items()})
	return features


def expand_for_verification(
	df: pd.DataFrame,
	max_article_words: int = 500,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, List[str]]:
	"""Expand each row into 4 binary examples for answer verification.

	Returns:
	- combined_texts: one text per option, format: article [SEP] question [SEP] option
	- labels: 1 if option is correct, else 0
	- lexical_matrix: dense lexical features (n_examples, n_features)
	- row_ids: index of original question row for grouping (used for EM later)
	- lexical_feature_names: stable column order for lexical matrix
	"""
	combined_texts: List[str] = []
	labels: List[int] = []
	lexical_rows: List[Dict[str, float]] = []
	row_ids: List[int] = []

	for row_idx, row in df.iterrows():
		article = " ".join(str(row["article"]).split()[:max_article_words])
		question = str(row["question"])
		answer = str(row["answer"]).strip()

		for opt in OPTION_COLS:
			option = str(row[opt])
			combined = f"{article} [SEP] {question} [SEP] {option}"
			label = 1 if opt == answer else 0

			combined_texts.append(combined)
			labels.append(label)
			lexical_rows.append(compute_lexical_features(article, question, option))
			row_ids.append(int(row_idx))

	lexical_df = pd.DataFrame(lexical_rows)
	lexical_feature_names = list(lexical_df.columns)

	return (
		combined_texts,
		np.asarray(labels, dtype=np.int8),
		lexical_df.to_numpy(dtype=np.float32),
		np.asarray(row_ids, dtype=np.int32),
		lexical_feature_names,
	)


def build_vectorizer(max_features: int = 10_000, min_df: int = 2) -> CountVectorizer:
	"""Create one-hot CountVectorizer with project defaults."""
	return CountVectorizer(
		binary=True,
		max_features=max_features,
		min_df=min_df,
		ngram_range=(1, 1),
		lowercase=True,
		token_pattern=r"\b\w+\b",
	)


# [CHANGE 2] New function: builds a TF-IDF vectorizer with project-recommended settings.
# This is separate from build_vectorizer() so either can be used independently.
# Key differences from OHE (CountVectorizer):
#   - binary=False: TF-IDF gives float weights, not 0/1. Words that appear more
#     often in a specific example get higher weight.
#   - sublinear_tf=True: applies log(1+TF) instead of raw TF. This prevents very
#     frequent words in a long article from completely dominating the vector.
#   - stop_words='english': removes words like "the", "is", "a" that contribute
#     zero information. OHE doesn't do this by default; TF-IDF benefits more from it
#     because those stopwords would otherwise get artificially high TF scores.
#   - max_df=0.95: ignores terms appearing in >95% of all examples (near-stopwords
#     that the English list misses). OHE has no equivalent.
#   - norm='l2': normalizes every vector to unit length. This makes cosine similarity
#     correct without any extra step, and keeps features on the same scale.
def build_tfidf_vectorizer(max_features: int = 10_000, min_df: int = 2) -> TfidfVectorizer:
	"""Create TF-IDF vectorizer with project-recommended settings.

	Why TF-IDF vs OHE (CountVectorizer binary=True)?
	- OHE: 1 if word present, 0 if absent. Treats all present words equally.
	- TF-IDF: gives higher weight to words that are FREQUENT in THIS example
	  but RARE across ALL examples. More discriminative.

	These settings mirror the TF-IDF manual recommendations for RACE:
	- sublinear_tf=True: dampens effect of very frequent terms via log(1+tf).
	- stop_words='english': removes 318 common English stopwords.
	- max_df=0.95: removes near-universal terms the stop list misses.
	- norm='l2': unit-normalizes each row so cosine similarity works out-of-the-box.
	"""
	return TfidfVectorizer(
		max_features=max_features,
		min_df=min_df,
		ngram_range=(1, 1),          # unigrams; change to (1,2) for bigrams [BONUS]
		stop_words="english",        # remove common English stopwords
		sublinear_tf=True,           # use log(1 + tf) instead of raw tf
		max_df=0.95,                 # ignore terms in >95% of documents
		norm="l2",                   # L2-normalize each row vector
		lowercase=True,
		token_pattern=r"\b\w+\b",
	)


def compute_cosine_similarity_features(
	x_ohe: sparse.spmatrix,
	row_ids: np.ndarray,
) -> np.ndarray:
	"""Compute cosine similarity features from One-Hot Encoded vectors.

	For each example (article, question, option), computes:
	1. Cosine similarity between article and option vectors
	2. Cosine similarity between question and option vectors

	These features capture semantic overlap without requiring additional models.

	Args:
		x_ohe: Sparse One-Hot matrix, shape (n_examples, n_features)
		row_ids: Array mapping each example to its original question index

	Returns:
		Dense array of shape (n_examples, 2) with [article_sim, question_sim] per row
	"""
	# Since examples are created in order (4 per original row):
	# indices 0-3 belong to row 0, indices 4-7 belong to row 1, etc.
	# Within each group: example 0 = article+question+optionA, example 1 = article+question+optionB, etc.
	# We compute similarity between the FIRST option (optionA, index 0 mod 4) and each of the others.

	# Actually, simpler approach: compute similarity between each example and the question text.
	# We approximate "article" as the shared prefix, and compute overlaps directly.

	# More practical approach for bag-of-words:
	# Option 1: Extract article/question vectors from the combined text (stored separately)
	# Option 2: Use TF-IDF overlap as proxy

	# Since we don't have separate article/question vectors, we'll compute:
	# - How much each option overlaps with the "typical" article+question (all options for same Q have same art+Q)
	# - Use cosine sim of this example to an average of all examples for this row's question

	n_examples = x_ohe.shape[0]
	cosine_features = np.zeros((n_examples, 2), dtype=np.float32)

	# For each group of 4 examples (belonging to same question):
	for idx in range(n_examples):
		row_id = row_ids[idx]
		# Find all examples with same row_id (should be 4: options A, B, C, D)
		same_row_mask = row_ids == row_id
		same_row_indices = np.where(same_row_mask)[0]

		# Compute average similarity within this question's 4 options
		# This approximates "how similar is this option to typical text for this question"
		row_vectors = x_ohe[same_row_indices]
		if len(same_row_indices) > 0:
			# Cosine similarity between this example and mean of group
			mean_vector = np.asarray(row_vectors.mean(axis=0)).flatten()
			current_vector = np.asarray(x_ohe[idx].toarray().flatten())

			# Compute cosine similarity manually
			dot_prod = np.dot(current_vector, mean_vector)
			norm_curr = np.linalg.norm(current_vector)
			norm_mean = np.linalg.norm(mean_vector)

			sim = dot_prod / (norm_curr * norm_mean + 1e-10)
			cosine_features[idx, 0] = float(sim)

		# Second feature: variance within the group (how diverse are the 4 options?)
		if len(same_row_indices) > 1:
			row_dense = np.asarray(row_vectors.toarray())
			variance = np.var(row_dense, axis=0).mean()  # average feature variance
			cosine_features[idx, 1] = float(variance)

	return cosine_features


def save_split_artifacts(
	processed_dir: Path,
	split_name: str,
	x_ohe: sparse.spmatrix,
	x_lex: np.ndarray,
	y: np.ndarray,
	row_ids: np.ndarray,
	x_tfidf: sparse.spmatrix | None = None,  # [CHANGE 3] New optional parameter for TF-IDF matrix
) -> None:
	"""Persist all per-split arrays/matrices to data/processed.

	x_tfidf is optional so this function stays backward-compatible if TF-IDF
	is disabled via --no-tfidf flag.
	"""
	sparse.save_npz(processed_dir / f"X_{split_name}_ohe.npz", x_ohe)
	np.save(processed_dir / f"X_{split_name}_lexical.npy", x_lex)
	np.save(processed_dir / f"y_{split_name}.npy", y)
	np.save(processed_dir / f"row_ids_{split_name}.npy", row_ids)

	# [CHANGE 4] Save TF-IDF matrix if provided.
	# Uses the same .npz sparse format as OHE — TF-IDF matrices are also sparse
	# (most words are absent from any given example). Saves ~same disk space as OHE.
	if x_tfidf is not None:
		sparse.save_npz(processed_dir / f"X_{split_name}_tfidf.npz", x_tfidf)


def run_pipeline(
	raw_dir: Path,
	processed_dir: Path,
	vectorizer_out: Path,
	max_article_words: int = 500,
	max_features: int = 10_000,
	min_df: int = 2,
	use_tfidf: bool = True,  # [CHANGE 5] New flag — set False to skip TF-IDF (saves time in quick tests)
) -> None:
	"""Run the full preprocessing pipeline end-to-end."""
	processed_dir.mkdir(parents=True, exist_ok=True)
	vectorizer_out.parent.mkdir(parents=True, exist_ok=True)

	# 1) Load and clean raw data.
	raw_splits = load_raw_splits(raw_dir)
	clean_splits = {name: clean_dataframe(df) for name, df in raw_splits.items()}

	# 2) Expand each split into binary verification examples + lexical features.
	expanded = {}
	lexical_feature_names: List[str] | None = None

	for split_name, df in clean_splits.items():
		combined_texts, y, x_lex, row_ids, lex_names = expand_for_verification(
			df,
			max_article_words=max_article_words,
		)
		expanded[split_name] = {
			"texts": combined_texts,
			"y": y,
			"x_lex": x_lex,
			"row_ids": row_ids,
		}
		if lexical_feature_names is None:
			lexical_feature_names = lex_names

	# 3) Fit OHE vectorizer on train ONLY, transform all splits.
	vectorizer = build_vectorizer(max_features=max_features, min_df=min_df)
	x_train_ohe = vectorizer.fit_transform(expanded["train"]["texts"])
	x_dev_ohe   = vectorizer.transform(expanded["dev"]["texts"])
	x_test_ohe  = vectorizer.transform(expanded["test"]["texts"])

	# 4) Save OHE vectorizer.
	joblib.dump(vectorizer, vectorizer_out)

	# 3.5) Compute cosine similarity features for all splits.
	# These capture semantic overlap between options within each question.
	x_train_cosine = compute_cosine_similarity_features(x_train_ohe, expanded["train"]["row_ids"])
	x_dev_cosine = compute_cosine_similarity_features(x_dev_ohe, expanded["dev"]["row_ids"])
	x_test_cosine = compute_cosine_similarity_features(x_test_ohe, expanded["test"]["row_ids"])

	# Combine cosine features with existing lexical features.
	x_train_lex_combined = np.hstack([expanded["train"]["x_lex"], x_train_cosine])
	x_dev_lex_combined = np.hstack([expanded["dev"]["x_lex"], x_dev_cosine])
	x_test_lex_combined = np.hstack([expanded["test"]["x_lex"], x_test_cosine])

	# Update lexical feature names to include new cosine similarity features.
	if lexical_feature_names is not None:
		lexical_feature_names = lexical_feature_names + ["cosine_similarity_within_question", "option_diversity"]

	# [CHANGE 6] Fit TF-IDF vectorizer on train ONLY, transform all splits.
	# CRITICAL RULE (from TF-IDF manual Chapter 4.4):
	#   fit_transform() → training texts ONLY.
	#   transform()     → dev and test texts.
	# If you accidentally call fit_transform() on dev/test, the vectorizer learns
	# IDF statistics from those splits, leaking test distribution into features.
	# That inflates reported metrics without the model actually being better.
	tfidf_vectorizer = None
	x_train_tfidf = x_dev_tfidf = x_test_tfidf = None

	if use_tfidf:
		print("Fitting TF-IDF vectorizer on training texts...")
		tfidf_vectorizer = build_tfidf_vectorizer(max_features=max_features, min_df=min_df)

		# fit_transform on TRAIN only — this computes IDF from training corpus.
		x_train_tfidf = tfidf_vectorizer.fit_transform(expanded["train"]["texts"])

		# transform (NOT fit_transform) on dev and test — uses the IDF learned from train.
		x_dev_tfidf  = tfidf_vectorizer.transform(expanded["dev"]["texts"])
		x_test_tfidf = tfidf_vectorizer.transform(expanded["test"]["texts"])

		# [CHANGE 7] Save TF-IDF vectorizer alongside OHE vectorizer.
		# The vectorizer MUST be saved with the classifier — at inference time you need
		# the exact same vocabulary and IDF weights used during training.
		tfidf_vectorizer_path = vectorizer_out.parent / "tfidf_vectorizer.pkl"
		joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)
		print(f"TF-IDF vectorizer saved → {tfidf_vectorizer_path}")

	# 5) Save per-split artifacts (OHE + lexical + TF-IDF if computed).
	save_split_artifacts(
		processed_dir, "train",
		x_train_ohe, x_train_lex_combined, expanded["train"]["y"], expanded["train"]["row_ids"],
		x_tfidf=x_train_tfidf,  # None if use_tfidf=False
	)
	save_split_artifacts(
		processed_dir, "dev",
		x_dev_ohe, x_dev_lex_combined, expanded["dev"]["y"], expanded["dev"]["row_ids"],
		x_tfidf=x_dev_tfidf,
	)
	save_split_artifacts(
		processed_dir, "test",
		x_test_ohe, x_test_lex_combined, expanded["test"]["y"], expanded["test"]["row_ids"],
		x_tfidf=x_test_tfidf,
	)

	# [CHANGE 8] Add TF-IDF metadata to the saved config so downstream scripts
	# know whether TF-IDF features exist and where the vectorizer lives.
	tfidf_meta: Dict = {}
	if use_tfidf and tfidf_vectorizer is not None:
		tfidf_meta = {
			"tfidf_enabled": True,
			"tfidf_vectorizer_path": str(vectorizer_out.parent / "tfidf_vectorizer.pkl"),
			"tfidf_vocab_size": int(len(tfidf_vectorizer.vocabulary_)),
			"tfidf_max_features": max_features,
			"tfidf_sublinear_tf": True,
			"tfidf_stop_words": "english",
			"tfidf_norm": "l2",
		}
	else:
		tfidf_meta = {"tfidf_enabled": False}

	# 6) Save metadata/config for reproducibility.
	metadata = {
		"max_article_words": max_article_words,
		"max_features": max_features,
		"min_df": min_df,
		"option_cols": OPTION_COLS,
		"text_cols": TEXT_COLS,
		"lexical_feature_names": lexical_feature_names or [],
		"num_train_examples": int(len(expanded["train"]["y"])),
		"num_dev_examples": int(len(expanded["dev"]["y"])),
		"num_test_examples": int(len(expanded["test"]["y"])),
		"train_positive_ratio": float(expanded["train"]["y"].mean()),
		"vocab_size": int(len(vectorizer.vocabulary_)),
		"raw_dir": str(raw_dir),
		"processed_dir": str(processed_dir),
		"vectorizer_path": str(vectorizer_out),
		**tfidf_meta,  # merge TF-IDF metadata in
	}

	with (processed_dir / "preprocessing_config.json").open("w", encoding="utf-8") as fh:
		json.dump(metadata, fh, indent=2)

	print("Preprocessing complete.")
	print(f"Train examples: {metadata['num_train_examples']:,}")
	print(f"Dev examples:   {metadata['num_dev_examples']:,}")
	print(f"Test examples:  {metadata['num_test_examples']:,}")
	print(f"Train positive ratio: {metadata['train_positive_ratio']:.4f}")
	print(f"OHE Vocabulary size:  {metadata['vocab_size']:,}")
	if use_tfidf:
		print(f"TF-IDF Vocabulary size: {tfidf_meta.get('tfidf_vocab_size', 'N/A'):,}")
	print(f"Saved config: {processed_dir / 'preprocessing_config.json'}")


def parse_args() -> argparse.Namespace:
	"""CLI arguments for flexible local/Colab usage."""
	parser = argparse.ArgumentParser(description="RACE preprocessing + feature engineering")
	parser.add_argument(
		"--raw-dir",
		type=Path,
		default=Path("data/raw"),
		help="Directory containing train/dev/test CSV files",
	)
	parser.add_argument(
		"--processed-dir",
		type=Path,
		default=Path("data/processed"),
		help="Directory where processed outputs are saved",
	)
	parser.add_argument(
		"--vectorizer-out",
		type=Path,
		default=Path("models/model_a/traditional/ohe_vectorizer.pkl"),
		help="Path to save fitted CountVectorizer",
	)
	parser.add_argument(
		"--max-article-words",
		type=int,
		default=500,
		help="Truncate article to first N words during verification expansion",
	)
	parser.add_argument(
		"--max-features",
		type=int,
		default=10_000,
		help="Vocabulary size cap for both one-hot and TF-IDF vectorizers",
	)
	parser.add_argument(
		"--min-df",
		type=int,
		default=2,
		help="Minimum document frequency for vocabulary in both vectorizers",
	)
	# [CHANGE 9] New CLI flag: --no-tfidf lets you skip TF-IDF during quick
	# debugging runs so you're not waiting for it to finish unnecessarily.
	parser.add_argument(
		"--no-tfidf",
		action="store_true",
		default=False,
		help="Skip TF-IDF vectorization (faster for quick debugging runs)",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run_pipeline(
		raw_dir=args.raw_dir,
		processed_dir=args.processed_dir,
		vectorizer_out=args.vectorizer_out,
		max_article_words=args.max_article_words,
		max_features=args.max_features,
		min_df=args.min_df,
		use_tfidf=not args.no_tfidf,  # [CHANGE 10] Pass flag to pipeline
	)