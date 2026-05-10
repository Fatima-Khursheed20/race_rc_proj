# Phase 6: Model B Distractor Generator — Detailed Implementation Guide

**Marks at stake:** 15/100  
**Time estimate:** 2-3 days (candidate extraction + feature engineering + training + evaluation)  
**Deliverable:** Trained distractor ranker, evaluation metrics, human evaluation form in `notebooks/experiments.ipynb`

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Data Preparation](#2-data-preparation)
3. [Candidate Phrase Extraction](#3-candidate-phrase-extraction)
4. [Feature Engineering for Candidates](#4-feature-engineering-for-candidates)
5. [Training the ML Ranker](#5-training-the-ml-ranker)
6. [Inference: Selecting Top-3 Distractors with Diversity](#6-inference-selecting-top-3-distractors-with-diversity)
7. [Alternative Approaches](#7-alternative-approaches)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [File Structure & Outputs](#9-file-structure--outputs)
10. [Troubleshooting & Performance](#10-troubleshooting--performance)
11. [Code Template & Integration](#11-code-template--integration)

---

## 1. Overview & Architecture

### 1.1 Task Definition

**Model B Sub-task 1: Distractor Generation**

```
Input:  article text + question text + correct answer text
         |
         v
[Candidate Extraction] → pool of 20-100 phrases from article
         |
         v
[Feature Computation] → 10 numerical features per candidate
         |
         v
[ML Ranker] → scores each candidate (0-1 probability)
         |
         v
[Diversity Selection] → top-3 diverse distractors
         |
Output: distractor_A, distractor_B, distractor_C
```

**Why this matters:** In educational assessments, distractors (wrong answer choices) must be plausible enough to test understanding without being obviously incorrect. Poor distractors make questions too easy; good ones reveal misconceptions.

### 1.2 Pipeline Components

1. **Candidate Extraction:** Extract plausible phrases from the article that could serve as answer options.
2. **Feature Engineering:** Compute similarity, frequency, and consistency features for each candidate.
3. **ML Ranking:** Train a binary classifier to score how "distractor-like" each candidate is.
4. **Diversity Enforcement:** Select top-3 candidates that are both high-scoring and dissimilar to each other.

### 1.3 Success Criteria

- **Plausibility:** Distractors should have moderate semantic similarity to the correct answer.
- **Incorrectness:** Must not match or contradict the article's facts.
- **Diversity:** The three distractors should not be paraphrases of each other.
- **Consistency:** All options (correct + distractors) should be grammatically similar.

---

## 2. Data Preparation

### 2.1 Input Data Format

Use the processed data from Phase 2. For each training example:
- `article`: cleaned article text
- `question`: cleaned question text  
- `correct_answer`: the correct option text (e.g., "New York City")
- `distractor_options`: list of the 3 actual RACE distractors for this question

### 2.2 Training Labels Creation

For supervised learning, you need positive and negative examples:

- **Positive examples (label=1):** The 3 actual RACE distractors for each question
- **Negative examples (label=0):** All other extracted candidates from the article that are not the correct answer

This creates a binary classification dataset where the model learns what makes a good distractor.

---

## 3. Candidate Phrase Extraction

### 3.1 Primary Method: N-gram Extraction

**Goal:** Generate 20-100 candidate phrases per article.

**Steps:**
1. Clean the article text (reuse `clean_text` from preprocessing.py)
2. Tokenize into words: `words = article.split()`
3. Extract n-grams:
   - Unigrams: single words
   - Bigrams: consecutive 2-word pairs
   - Trigrams: consecutive 3-word pairs
4. Filter candidates:
   - Remove exact matches to the correct answer
   - For unigrams: remove stopwords (use a basic list: ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
   - Keep only candidates that appear ≥2 times in the article
   - Length filter: keep candidates where `0.5 * len(correct_answer) ≤ len(candidate) ≤ 1.5 * len(correct_answer)`

**Expected output:** List of strings, e.g., ['New York', 'United States', 'large city', 'metropolitan area']

### 3.2 Alternative Method: Frequency-Based Selection

1. Count word frequencies in the article
2. Take top-30 most frequent content words
3. Filter out words that appear in the correct answer
4. Use these as unigram candidates

**When to use:** If n-gram extraction gives too many/noisy candidates.

### 3.3 Implementation Notes

- Store candidates as a list of strings
- Ensure candidates are unique (use set then convert back to list)
- Handle edge cases: very short articles, very long correct answers

---

## 4. Feature Engineering for Candidates

For each candidate phrase, compute exactly 10 numerical features:

### 4.1 Text Similarity Features

1. **ohe_cosine_sim_to_answer**: Cosine similarity between candidate's one-hot vector and correct answer's one-hot vector
   - Use the CountVectorizer from preprocessing.py
   - Transform both texts, compute cosine similarity

2. **ohe_cosine_sim_to_question**: Same as above, but similarity to question text

### 4.2 Length Consistency Features

3. **char_length_ratio**: `len(candidate) / len(correct_answer)`
4. **word_length_ratio**: `len(candidate.split()) / len(correct_answer.split())`

### 4.3 Frequency and Position Features

5. **passage_frequency**: Count how many times candidate appears in article
6. **position_in_article**: Fraction of article text before first occurrence of candidate (0.0 to 1.0)

### 4.4 Overlap Features

7. **word_overlap_with_answer**: `(len(set(candidate_words) & set(answer_words))) / len(answer_words)`
8. **word_overlap_with_question**: `(len(set(candidate_words) & set(question_words))) / len(question_words)`

### 4.5 Consistency Features

9. **starts_with_same_word**: 1 if candidate.split()[0] == correct_answer.split()[0], else 0
10. **is_proper_noun_candidate**: 1 if candidate[0].isupper(), else 0

### 4.6 Implementation Tips

- Use numpy arrays for features
- Handle division by zero (e.g., if correct_answer is empty)
- Normalize features if needed (though tree-based models like RandomForest don't require it)

---

## 5. Training the ML Ranker

### 5.1 Data Preparation for Training

**Loop over training data:**
```python
all_features = []
all_labels = []

for each training example:
    candidates = extract_candidates(article, correct_answer)
    features = [compute_features(candidate, article, question, correct_answer) for candidate in candidates]
    
    # Positive labels: actual RACE distractors
    for distractor in distractor_options:
        if distractor in candidates:
            all_features.append(compute_features(distractor, ...))
            all_labels.append(1)
    
    # Negative labels: other candidates
    for candidate in candidates:
        if candidate not in distractor_options and candidate != correct_answer:
            all_features.append(compute_features(candidate, ...))
            all_labels.append(0)
```

### 5.2 Model Selection

**Primary Model: LogisticRegression**
```python
from sklearn.linear_model import LogisticRegression
ranker = LogisticRegression(C=1.0, class_weight='balanced', random_state=42)
```

**Alternative Model: RandomForestClassifier**
```python
from sklearn.ensemble import RandomForestClassifier
ranker = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
```

### 5.3 Training Process

1. Convert features and labels to numpy arrays
2. Split into train/validation (80/20)
3. Fit the model: `ranker.fit(X_train, y_train)`
4. Evaluate on validation set
5. Save the trained model: `joblib.dump(ranker, 'models/model_b/hint_generator/distractor_ranker.pkl')`

### 5.4 Hyperparameter Tuning

- For LogisticRegression: try C in [0.1, 1.0, 10.0]
- For RandomForest: try n_estimators in [50, 100, 200]
- Use GridSearchCV with 3-fold CV
- Metric: F1 score (since classes are imbalanced)

---

## 6. Inference: Selecting Top-3 Distractors with Diversity

### 6.1 Basic Ranking

1. Extract candidates from article
2. Compute features for each candidate
3. Get predicted probabilities: `ranker.predict_proba(X)[:, 1]`
4. Sort candidates by probability (descending)

### 6.2 Diversity Enforcement (MMR Algorithm)

**Simplified Maximal Marginal Relevance:**

```python
def select_diverse_distractors(candidates, probs, lambda_param=0.5):
    selected = []
    
    while len(selected) < 3 and candidates:
        if not selected:
            # First selection: highest probability
            best_idx = np.argmax(probs)
        else:
            # Subsequent: balance relevance and diversity
            diversity_scores = []
            for i, candidate in enumerate(candidates):
                relevance = probs[i]
                max_sim = max(compute_similarity(candidate, selected_candidate) 
                            for selected_candidate in selected)
                diversity = relevance - lambda_param * max_sim
                diversity_scores.append(diversity)
            best_idx = np.argmax(diversity_scores)
        
        selected.append(candidates[best_idx])
        candidates.pop(best_idx)
        probs = np.delete(probs, best_idx)
    
    return selected
```

**Similarity function:** Use cosine similarity of one-hot vectors.

**Lambda parameter:** Try values 0.3 to 0.7. Higher values prioritize diversity over relevance.

### 6.3 Edge Cases

- If fewer than 3 candidates available, return what's available
- If all candidates are too similar, relax the lambda parameter
- Ensure selected distractors are not the correct answer

---

## 7. Alternative Approaches

### 7.1 Word2Vec Nearest Neighbors [BONUS]

1. Install gensim: `pip install gensim`
2. Download model: 
   ```python
   import gensim.downloader as api
   model = api.load("word2vec-google-news-300")
   ```
3. For correct answer, extract key word (longest non-stopword)
4. Find similar words: `model.most_similar(key_word, topn=20)`
5. Filter: remove words appearing in article, remove exact matches
6. Use top-3 as distractors

**Limitations:** Works for single words, not phrases.

### 7.2 Co-occurrence Matrix Approach

1. Build word co-occurrence matrix from training articles
2. For each word pair within 3-word windows, increment count
3. Use cosine similarity on this matrix to find similar words
4. Select top dissimilar words as distractors

---

## 8. Evaluation Metrics

### 8.1 Automatic Metrics

**Precision@3:** Fraction of selected distractors that are actual RACE distractors
**Recall@3:** Fraction of actual RACE distractors recovered in top-3
**F1@3:** Harmonic mean of precision and recall

**Implementation:**
```python
def evaluate_distractors(predictions, ground_truth_distractors):
    # predictions: list of 3 selected distractors
    # ground_truth_distractors: list of 3 actual distractors
    
    true_positives = len(set(predictions) & set(ground_truth_distractors))
    precision = true_positives / 3
    recall = true_positives / 3  # since we select exactly 3
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1
```

**Distractor Ranker Accuracy:** Binary classification accuracy for the ranker model.

### 8.2 Human Evaluation [Required]

Create a form with 20 examples:
- Show: passage, question, correct answer, your 3 distractors
- Rating scale: 1-5 (1=obviously wrong, 5=highly plausible)
- Collect responses from 3-5 evaluators
- Report: mean score ± standard deviation

### 8.3 Baseline Comparison

Compare against random selection and frequency-based selection.

---

## 9. File Structure & Outputs

```
models/model_b/hint_generator/
├── distractor_ranker.pkl          # Trained ranker model
└── [Optional] distractor_vectorizer.pkl  # If using custom vectorizer

notebooks/experiments.ipynb
├── Section: Distractor Generation
│   ├── Feature distribution plots
│   ├── Ranker training curves
│   ├── Precision/Recall/F1 table
│   └── Confusion matrix for ranker

src/
├── model_b_train.py               # Training script
└── distractor_generator.py        # Inference script

reports/
└── distractor_evaluation.xlsx     # Human evaluation results
```

---

## 10. Troubleshooting & Performance

### 10.1 Common Issues

- **Too few candidates:** Relax frequency threshold or use shorter n-grams
- **Poor diversity:** Adjust lambda parameter in MMR
- **Low precision:** Add more features or try RandomForest
- **Overfitting:** Use regularization, reduce features, increase training data

### 10.2 Performance Targets

- F1@3 > 0.3 (reasonable baseline)
- Human evaluation mean > 3.0
- Ranker accuracy > 0.7

### 10.3 Debugging Tips

- Visualize feature distributions for positive vs negative examples
- Check similarity distributions
- Manually inspect top-ranked candidates for problematic questions

---

## 11. Code Template & Integration

### 11.1 Training Script (model_b_train.py)

```python
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from preprocessing import load_raw_splits, clean_dataframe

# Load data
raw_splits = load_raw_splits(Path("data/raw"))
clean_splits = {name: clean_dataframe(df) for name, df in raw_splits.items()}

# Extract candidates and compute features
# [Implementation of candidate extraction and feature computation]

# Create training data
X, y = create_training_data(clean_splits['train'])

# Train model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
ranker = LogisticRegression(C=1.0, class_weight='balanced')
ranker.fit(X_train, y_train)

# Evaluate
val_predictions = ranker.predict(X_val)
# [Compute metrics]

# Save
joblib.dump(ranker, 'models/model_b/hint_generator/distractor_ranker.pkl')
```

### 11.2 Inference Script (distractor_generator.py)

```python
import joblib
import numpy as np

def generate_distractors(article, question, correct_answer, ranker):
    candidates = extract_candidates(article, correct_answer)
    features = [compute_features(cand, article, question, correct_answer) for cand in candidates]
    
    if not features:
        return ["No candidates found"] * 3
    
    X = np.array(features)
    probs = ranker.predict_proba(X)[:, 1]
    
    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    sorted_candidates = [candidates[i] for i in sorted_indices]
    sorted_probs = probs[sorted_indices]
    
    # Select diverse top-3
    distractors = select_diverse_distractors(sorted_candidates, sorted_probs)
    
    return distractors[:3]  # Ensure exactly 3

# Usage
ranker = joblib.load('models/model_b/hint_generator/distractor_ranker.pkl')
distractors = generate_distractors(article, question, correct_answer, ranker)
```

### 11.3 Integration with UI

In `ui/app.py`, add endpoint:
```python
@app.route('/generate-distractors', methods=['POST'])
def generate_distractors_endpoint():
    data = request.json
    distractors = generate_distractors(data['article'], data['question'], data['correct_answer'])
    return jsonify({'distractors': distractors})
```

---

**Next Steps:** After implementing distractors, move to Phase 7 (Hint Generation) or Phase 8 (UI). Test thoroughly before proceeding.</content>
<parameter name="filePath">d:\My Data\Sem 6\AI Lab\proj dump\race_rc_proj\plan_markdowns\PHASE_6_DETAILED.md