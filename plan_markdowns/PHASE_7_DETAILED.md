# Phase 7: Model B Hint Generator — Detailed Implementation Guide

**Marks at stake:** 10/100  
**Time estimate:** 1-2 days  
**Deliverable:** Trained hint generation pipeline, evaluation metrics, sample hints in `notebooks/experiments.ipynb`

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Task Definition](#2-task-definition)
3. [Hint Generation Strategy](#3-hint-generation-strategy)
4. [Data Preparation](#4-data-preparation)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Training](#6-model-training)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [File Structure & Outputs](#9-file-structure--outputs)
10. [Troubleshooting & Notes](#10-troubleshooting--notes)

---

## 1. Overview & Architecture

This phase builds on Model B distractor generation. After the system has a question and the correct answer, it must also produce three graduated hints.

### What this module does

```
Input: article text + question text + correct answer
         |
    split into sentences
         |
    score each sentence for relevance
         |
    choose three sentences with increasing specificity
         |
Output: hint_1, hint_2, hint_3
```

### Why this phase matters

Hints add pedagogical value and help users arrive at the answer without directly revealing it. Good hints are:
- progressively stronger
- anchored to the passage
- not obvious spoilers

---

## 2. Task Definition

### 2.1 Hint requirements

- **Hint 1:** broad, general, not too revealing
- **Hint 2:** more specific, points toward the correct paragraph or idea
- **Hint 3:** near-explicit, strongly indicates the answer without stating it directly

### 2.2 Output format

Return exactly 3 hints per question:

- `hint_1` (high-level)
- `hint_2` (mid-level)
- `hint_3` (strongest)

Example:
- Hint 1: "This question is about the main reason given in the second paragraph."
- Hint 2: "Look at the sentence describing why the boy stayed behind."
- Hint 3: "The answer is in the sentence mentioning the character's motivation."

---

## 3. Hint Generation Strategy

### 3.1 Extractive hint generation (recommended)

Use sentences already present in the article. This is reliable and avoids the complexity of text generation.

### 3.2 Hint ranking logic

1. Split article into sentences
2. Score each sentence for relevance to the question and correct answer
3. Sort sentences by score
4. Select three sentences and assign them so the least revealing sentence becomes Hint 1 and the most revealing becomes Hint 3

### 3.3 Why extractive works best here

- preserves grammatical correctness
- avoids hallucination
- ensures every hint is grounded in the passage
- easier to evaluate automatically

---

## 4. Data Preparation

### 4.1 Training data

Use the same raw RACE splits from Phase 2 and Phase 6. For each question row, you need:

- `article`
- `question`
- `correct_answer`
- all answer options (A, B, C, D)

### 4.2 Sentence extraction

1. Clean the article text using `clean_text()` from `src/preprocessing.py`
2. Split into sentences using a simple sentence tokenizer:
   - split on `.`, `?`, `!`
   - keep punctuation attached to the sentence
   - discard very short fragments

### 4.3 Gold hint sentence labeling

Because we do not have human-labeled hints, use a proxy label:
- choose the sentence with the highest word overlap with the correct answer
- if multiple sentences tie, choose the one closest to the answer in the text

This proxy gives you a target for training the hint scorer.

---

## 5. Feature Engineering

For each sentence, compute a feature vector that captures relevance, position, and answer connection.

### 5.1 Sentence-level features

1. `word_overlap_with_question`
   - ratio of question words found in sentence
2. `word_overlap_with_answer`
   - ratio of correct answer words found in sentence
3. `sentence_position_ratio`
   - sentence index / total sentences
4. `sentence_length`
   - number of words in the sentence
5. `contains_named_entity_proxy`
   - 1 if the sentence contains any capitalized token other than sentence-start
6. `contains_question_wh_word`
   - 1 if question begins with who/what/why/where/when/how and sentence contains the relevant cue
7. `keyword_density`
   - fraction of non-stopwords in sentence that also appear in the question or correct answer
8. `distance_to_answer_sentence`
   - if the correct answer appears literally in some sentence, distance from the candidate sentence to that sentence

### 5.2 Optional semantic feature

- `question_sentence_sim`
  - cosine similarity between sentence and question using a CountVectorizer or TF-IDF vectorizer

### 5.3 Feature construction notes

- use `normalize_text()` from `src/distractor_generator.py` or `clean_text()`
- remove simple stopwords for overlap counts
- keep all values numeric
- use the same vectorizer trained in Phase 6 if you want feature consistency

---

## 6. Model Training

### 6.1 Model choice

Use `LogisticRegression(class_weight='balanced')`.

Why:
- fast to train
- robust for binary sentence scoring
- fits your Phase 7 time budget

Alternative: `RandomForestClassifier(n_estimators=50)` for a stronger model if time remains.

### 6.2 Training objective

Treat hint sentence selection as binary classification:
- positive class = gold hint sentence
- negative class = all other sentences in the article

### 6.3 Training pipeline

1. Load raw train/dev/test CSVs
2. Clean text and split articles into sentences
3. Extract sentence features for each candidate sentence
4. Label the gold sentence as 1, others as 0
5. Train on training rows
6. Validate on dev rows
7. Test on test rows

### 6.4 Example training flow

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = build_sentence_training_data(train_df, vectorizer)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)
```

### 6.5 Save artifacts

Save:
- `models/model_b/hint_generator/hint_ranker.pkl`
- `models/model_b/hint_generator/hint_vectorizer.pkl` (if using a vectorizer)

---

## 7. Inference Pipeline

### 7.1 Generate hints for a new question

1. Clean the article and question
2. Split article into sentences
3. Compute sentence features
4. Score each sentence with the trained hint model
5. Sort sentences by score descending
6. Assign hints:
   - Hint 3 = top scored sentence
   - Hint 2 = second scored sentence
   - Hint 1 = third scored sentence

### 7.2 Guardrails

- If fewer than 3 candidate sentences exist, repeat the best sentence to return exactly 3 hints
- Avoid returning the correct answer as a literal hint sentence; instead use a higher-level sentence if necessary
- If the top sentences are too short (< 5 words), skip them for the next best sentence

### 7.3 Output format

Return a dictionary or JSON object:

```json
{
  "hint_1": "...",
  "hint_2": "...",
  "hint_3": "..."
}
```

### 7.4 Example inference command

```bash
python src/hint_generator.py \
  --article "..." \
  --question "..." \
  --correct-answer "..." \
  --model-dir models/model_b/hint_generator
```

---

## 8. Evaluation Metrics

### 8.1 Automatic evaluation

Use these metrics:
- `Precision@1`: whether the top predicted hint sentence matches the gold sentence proxy
- `Precision@3`: whether the gold sentence proxy appears in the top 3 predictions
- `F1` for binary hint sentence classification
- `Accuracy` of the sentence scorer

### 8.2 Human evaluation

Required for full marks. Create a simple spreadsheet or form with 20 examples:
- passage
- question
- correct answer
- hint 1, hint 2, hint 3

Ask evaluators whether hints:
- help understand the passage
- do not directly reveal the answer
- are ordered correctly from general to specific

Report the average score and standard deviation.

### 8.3 Baseline comparison

Compare against a naive baseline:
- Hint 3 = sentence with maximum word overlap with the question
- Hint 2 = second-best sentence
- Hint 1 = third-best sentence

Your trained model should outperform this baseline in Precision@3.

---

## 9. File Structure & Outputs

```
models/model_b/hint_generator/
├── hint_ranker.pkl
└── hint_vectorizer.pkl   # optional if you use sentence embeddings or TF-IDF

src/
├── hint_generator.py
└── model_b_train.py      # optionally extended to train both distractor and hint models

notebooks/experiments.ipynb
├── Hint generator section
│   ├── feature distribution plots
│   ├── model training results
│   ├── Precision@1 / Precision@3 tables
│   └── selected sample hints
```

---

## 10. Troubleshooting & Notes

### 10.1 Common issues

- **Sentence splitting too aggressive:** tune the tokenizer to keep meaningful sentences intact
- **Gold label noise:** answer-overlap proxy is imperfect; inspect 20 examples manually
- **Too many generic hints:** add question-specific cue features
- **Top sentence is literally the answer:** prefer the second or third sentence if the top one directly repeats the answer

### 10.2 Implementation tips

- Use the same cleaning utilities from `src/preprocessing.py`
- Reuse `CountVectorizer` or `TfidfVectorizer` from Model B if possible
- Keep the training script deterministic with `random_state=42`
- Save both model and vectorizer so inference is identical to training

### 10.3 Next step after Phase 7

After Phase 7, integrate the hint generator into the UI pipeline and ensure Model B returns:
- 3 distractors
- 3 graduated hints
- a correct answer label

This gives you a complete Model B experience for the final application.
