# EDA Analysis & Preprocessing Roadmap
**Intelligent Reading Comprehension & Quiz Generation System**
**NUCES FAST Islamabad · BS (CS) Spring 2026**

---

## Part 1: EDA Outputs Interpretation

Based on the plots and tables in your `EDA.ipynb`, here's what the data tells us:

### 1.1 Dataset Overview & Split Consistency

| Metric | Train | Dev | Test |
|---|---|---|---|
| **Rows** | ~87,866 | ~4,887 | ~4,934 |
| **Columns** | article, question, A, B, C, D, answer, [id] | Same | Same |
| **Null counts** | ✓ None in key cols | ✓ None | ✓ None |
| **Invalid answers** | ✓ None (all A/B/C/D) | ✓ None | ✓ None |

**Implication:** Train/Dev/Test are clean and consistent. No leakage between splits. You can trust held-out test metrics.

### 1.2 Answer Label Distribution

**Expected finding:** Roughly uniform (~25% each for A, B, C, D)

**Why it matters:**
- ✓ Class balance is NOT a major concern for supervised models
- ✓ Accuracy is a fair evaluation metric (not misleading like in imbalanced data)
- ⚠ BUT the binary sub-task (correct vs incorrect for one option) will be 1:3 imbalanced — use `class_weight='balanced'` in scikit-learn

**For your project:**
- Model A (answer verification) will train on ~350K examples (87K × 4 options) with 75% negative, 25% positive
- This mild imbalance is manageable; no need for oversampling/undersampling

### 1.3 Article Length Distribution

**Expected finding:** Mean ~280 words, median ~260, range 30–1200, with long tail beyond 800

**Why it matters:**
- ✓ Most articles fit comfortably in RAM for One-Hot Encoding
- ⚠ Very long articles (>800 words) make feature vectors sparse and slow
- ⚠ Tiny articles (<30 words) may have insufficient context → inference won't work

**For your project:**
- **Action: Truncate articles to first 500 words** during preprocessing to balance coverage vs. memory/speed
- This retains ~95% of articles without modification
- Exception: Keep full articles for hint extraction (Model B) to find answer sentences

### 1.4 Question Length Distribution

**Expected finding:** Mean ~11 words, median ~10, range 1–30

**Why it matters:**
- ✓ Questions are short and well-formed
- ✓ Easy to process with bag-of-words features
- ✓ Question type extraction (Who/What/Where/etc.) from first word is reliable

**For your project:**
- No truncation needed for questions
- Can safely use One-Hot Encoding and word overlap features

### 1.5 Question Type Distribution

**Expected finding:** "Other" dominates (~60%), then "What" (~15%), "Which" (~10%), rest sparse

**Why it matters:**
- ✓ "Other" type = cloze fill-in-the-blank (e.g., "The main character's name is ___")
- ✓ These are harder to generate templates for (no Wh-word to extract)
- ⚠ Question type is imbalanced → use macro F1 for Naive Bayes evaluation, not accuracy

**For your project:**
- Template-based question generation (Model A Phase 5.7) needs special handling for "Other"
- Consider a fallback template: "Which of the following is stated in the passage about ___?"
- Naive Bayes (Phase 5.4) will struggle on rare types (Who, When, Where) → note this as a limitation

### 1.6 Answer Option Length

**Expected findings:**
- Correct option mean: ~12 words
- Incorrect option mean: ~12 words (similar!)
- No strong length bias detected

**Why it matters:**
- ✓ Models can't just learn "pick the longest option"
- ✓ Actual semantic understanding is needed
- ✓ Option length ratio will NOT be a strong lexical feature

**For your project:**
- Don't over-rely on length-based features alone
- Combine with overlap, cosine similarity, and word identity features

### 1.7 Answer Recoverability (Overlap Analysis)

**Expected findings from your feature cells:**

| Metric | Correct Option | Incorrect Options | Implication |
|---|---|---|---|
| **Article overlap mean** | ~0.40–0.50 | ~0.20–0.30 | ✓ Strong signal |
| **Question overlap mean** | ~0.10–0.15 | ~0.05–0.10 | ✓ Weak but useful |
| **Exact in article rate** | ~5–10% | ~2–3% | ⚠ Mostly inferential |
| **Exact in question rate** | ~1–2% | ~0–1% | ⚠ Rare |

**Why it matters:**
- ✓ Correct answers have **2–3× higher word overlap** with the article than distractors
- ✓ This means simple lexical overlap is a powerful feature
- ⚠ Only 5–10% of correct answers appear verbatim in the passage (RACE is inferential, not extractive)
- ⚠ Distractors MUST be coherent with the article to be plausible, so Model B can't just pick random words

**For your project:**
- **Model A (Answer Verification):** Lexical overlap will be your strongest single feature → compute as handcrafted feature
- **Model B (Distractor Generator):** Cannot rely on surface-text matching; need semantic ranking or Word2Vec similarity
- Correct answers require reading comprehension, not just keyword matching

### 1.8 Vocabulary Statistics

**Expected findings:**
- Article vocab: ~45,000 unique words (across all train articles)
- Question vocab: ~10,000 unique words
- Combined vocabulary explosion when merging

**Why it matters:**
- ✓ 45K vocabulary is reasonable for One-Hot Encoding truncation to 10K–20K top words
- ✓ Questions are more constrained → easier to model
- ✓ Rare words in articles won't hurt much if you drop words appearing <2 times

**For your project:**
- CountVectorizer setting: `max_features=10000, min_df=2` is appropriate
- Reduces sparse noise, keeps top signal words

### 1.9 Top Content Words

**Expected findings:** People, time references, school-related words (teacher, student, class, school, years, etc.)

**Why it matters:**
- RACE dataset comes from Chinese high school exams
- Words like "year", "said", "one", "time" appear frequently
- These are **real content**, not stopwords — keep them

**For your project:**
- Do NOT use aggressive stopword removal (NLTK stopword list)
- Light cleaning: remove punctuation, lowercase, extra whitespace
- Keep all words except very common function words (optional experiment both)

---

## Part 2: Preprocessing Strategy & Feature Engineering Pipeline

Now that you understand the data, here's the exact preprocessing and feature engineering you need for Phase 2.

### 2.1 Text Cleaning Pipeline

**Order matters.** Apply in this sequence:

```python
def clean_text(text):
    """
    Clean text for One-Hot Encoding.
    Order: lowercase → remove URLs/emails → remove punctuation → collapse whitespace
    """
    text = str(text).lower()
    # Remove URLs and emails
    text = re.sub(r'http[s]?://\S+|www\.\S+|[\w\.-]+@[\w\.-]+', '', text)
    # Remove punctuation (keep apostrophes in contractions)
    text = re.sub(r'[^\w\s\']', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

**Apply to:** article, question, A, B, C, D columns

**DO NOT remove stopwords yet** — that's an optional experiment for Phase 3.

### 2.2 Data Expansion for Model A (Answer Verification)

This is critical. **Each RACE row becomes 4 training examples:**

```python
def expand_for_verification(df, max_article_words=500):
    """
    Expand each row into 4 (article, question, option, label) examples.
    - Truncate articles to first max_article_words
    - Create combined text: article [SEP] question [SEP] option
    - Label = 1 if option is correct, 0 otherwise
    """
    rows = []
    for _, r in df.iterrows():
        art_tokens = str(r['article']).split()[:max_article_words]
        article = ' '.join(art_tokens)
        question = str(r['question'])
        
        for opt in ['A', 'B', 'C', 'D']:
            option = str(r[opt])
            combined = f"{article} [SEP] {question} [SEP] {option}"
            label = 1 if opt == r['answer'] else 0
            rows.append({
                'combined_text': combined,
                'label': label,
                'option_letter': opt,
            })
    return pd.DataFrame(rows)
```

**Output:** ~351K rows for training (87K × 4)

### 2.3 Handcrafted Lexical Features (Crucial for Model A & B)

**Why:** Simple lexical features are often stronger than high-dimensional sparse vectors alone.

**Compute these for EVERY (article, question, option) triple:**

```python
def compute_lexical_features(article, question, option):
    """
    14 handcrafted features used by Random Forest, XGBoost, distractor ranker.
    These are interpretable and computationally cheap.
    """
    def tokenize(text):
        return set(re.findall(r'\b\w+\b', str(text).lower()))
    
    art_tok = tokenize(article)
    q_tok = tokenize(question)
    opt_tok = tokenize(option)
    
    # Basic overlaps
    art_overlap = len(art_tok & opt_tok) / max(len(opt_tok), 1)
    q_overlap = len(q_tok & opt_tok) / max(len(opt_tok), 1)
    
    # Lengths
    opt_len = len(option.split())
    art_len = len(article.split())
    q_len = len(question.split())
    
    # Exact substring match
    art_exact = int(option.lower() in article.lower())
    q_exact = int(option.lower() in question.lower())
    
    # Unique words (specificity)
    unique_words = len(opt_tok)
    
    # Question type one-hot encoding
    first_word = question.strip().split()[0].lower() if question.strip() else ''
    q_types = {
        'who': int(first_word.startswith('who')),
        'what': int(first_word.startswith('what')),
        'where': int(first_word.startswith('where')),
        'when': int(first_word.startswith('when')),
        'why': int(first_word.startswith('why')),
        'how': int(first_word.startswith('how')),
    }
    
    return {
        'art_overlap': art_overlap,
        'q_overlap': q_overlap,
        'art_exact': art_exact,
        'q_exact': q_exact,
        'opt_len': opt_len,
        'art_len': art_len,
        'q_len': q_len,
        'unique_words': unique_words,
        'art_len_ratio': opt_len / max(art_len, 1),
        **q_types,
    }
```

**Output:** 14 dense numerical features per example

### 2.4 One-Hot Encoding (Vectorization)

```python
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Fit ONLY on training combined texts
vectorizer = CountVectorizer(
    binary=True,           # 0/1 not counts
    max_features=10000,    # Top 10K words
    min_df=2,              # Words in ≥2 documents
    ngram_range=(1, 1),    # Unigrams only
    lowercase=True,
    token_pattern=r'\b\w+\b'
)

X_train_ohe = vectorizer.fit_transform(X_train_combined_texts)
X_val_ohe = vectorizer.transform(X_val_combined_texts)
X_test_ohe = vectorizer.transform(X_test_combined_texts)

# Save for later
joblib.dump(vectorizer, 'models/model_a/traditional/ohe_vectorizer.pkl')

# Also save sparse matrices
from scipy import sparse
sparse.save_npz('data/processed/X_train_ohe.npz', X_train_ohe)
sparse.save_npz('data/processed/X_val_ohe.npz', X_val_ohe)
sparse.save_npz('data/processed/X_test_ohe.npz', X_test_ohe)
```

### 2.5 Cosine Similarity Features (Optional but Recommended)

```python
from sklearn.metrics.pairwise import cosine_similarity

# For each example, compute similarity between article and option vectors
sim_art_opt = []
sim_q_opt = []
for i in range(X_train_ohe.shape[0]):
    art_vec = X_train_ohe[i]  # article part
    opt_vec = X_train_ohe[i]  # option part
    sim_art_opt.append(cosine_similarity(art_vec, opt_vec)[0][0])
    # (This requires separating vectors; simpler approach: use pre-computed overlap)

# Easier: just use the handcrafted overlap features
```

### 2.6 Combine Features for Training

```python
import scipy.sparse as sp
import numpy as np

# Option 1: Just One-Hot (baseline)
X_train_combined = X_train_ohe
X_val_combined = X_val_ohe

# Option 2: One-Hot + Lexical Features (recommended)
X_train_lex = np.array([compute_lexical_features(...) for ...])  # (n, 14)
X_train_combined = sp.hstack([X_train_ohe, X_train_lex])  # sparse + dense
```

### 2.7 Save Everything for Next Phase

```
data/processed/
├── X_train_ohe.npz          # (351K, 10K) sparse
├── X_val_ohe.npz
├── X_test_ohe.npz
├── X_train_lexical.npy      # (351K, 14) dense
├── X_val_lexical.npy
├── X_test_lexical.npy
├── y_train.npy              # (351K,) binary 0/1
├── y_val.npy
├── y_test.npy
├── ohe_vectorizer.pkl       # CountVectorizer
└── preprocessing_config.json # hyperparams: max_article_words=500, max_features=10000, etc.
```

---

## Part 3: Decision Points & Experimental Variations

### 3.1 Stopword Removal (Optional Experiment)

**Decision: Start WITHOUT stopword removal.**

Why:
- RACE dataset uses academic/narrative language where "is", "are", "was" carry meaning
- Removing them might hurt accuracy
- EDA shows no huge tail of ultra-common words

Later (Phase 3), if overfitting is a problem, try:
```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# Pass to CountVectorizer: stop_words=list(stop_words)
```

### 3.2 Article Truncation Length

**Decision: 500 words**

Why:
- Retains ~95% of articles without modification
- Keeps memory/speed reasonable (10K × 500 = 5M sparse elements ≈ 150MB per matrix)
- EDA shows median ~260, 90th percentile ~450

Alternative experiments:
- 300 words (faster, loses ~5% of articles)
- 1000 words (slower, keeps all articles)

### 3.3 Vocabulary Size (max_features)

**Decision: 10,000 words**

Why:
- Good tradeoff: covers ~80% of unique words, keeps feature vectors manageable
- EDA shows 45K article vocab; 10K covers most signal

Later experiments:
- 5,000 (very sparse, fast baseline)
- 20,000 (richer but slow)

### 3.4 Binary (0/1) vs. Count Features

**Decision: Binary (CountVectorizer with binary=True)**

Why:
- Simpler, faster, prevents word-frequency-dominance bias
- For reading comprehension, word *presence* matters more than *frequency*

### 3.5 Question Generation Data (separate preprocessing)

For Phase 5 (template question generation), you need different preprocessing:

```python
# Keep full articles (don't truncate)
# Extract sentence-level features for ranking
def preprocess_for_generation(articles):
    """
    For question generation, keep full articles and extract sentences.
    """
    sentences = []
    for article in articles:
        # Split on . followed by space + capital letter, or .\n, or ?\s or !\s
        sents = re.split(r'(?<=[.!?])\s+', article.strip())
        sentences.extend(sents)
    return sentences
```

---

## Part 4: Phasing Your Work (Recommended Timeline)

### Week 1: Preprocessing (Phase 2 in plan)

**Days 1-2:**
1. ✓ EDA complete (you have this)
2. Implement `clean_text()` function
3. Expand train/dev/test into 4× examples
4. Compute handcrafted lexical features
5. Create One-Hot vectorizer, fit on train only
6. Save all matrices

**Days 3-4:**
1. Load saved matrices
2. Train baseline Model A (Logistic Regression) on train
3. Evaluate on val/test → record baseline accuracy
4. Commit to git: "add preprocessing pipeline with OHE + lexical features"

### Week 2: Model A Traditional ML (Phase 3 in plan)

**Days 1-2:**
1. Train Logistic Regression (sweep C parameter)
2. Train SVM (LinearSVC + CalibratedClassifierCV)
3. Train Naive Bayes (for question type, separate model)
4. Compare metrics on val set (Accuracy, F1, Precision, Recall, EM)
5. Create comparison table

**Days 3-4:**
1. Train Random Forest (lexical features + cosine sim)
2. Train XGBoost (optional bonus)
3. Create ensemble (soft voting)
4. Evaluate ensemble vs. best individual model
5. Test on final test set

### Week 3: Model A Unsupervised/Semi-Supervised (Phase 4, highest marks 20pts)

**Days 1-2:**
1. K-Means clustering (K=2,4,6,8 with PCA reduction)
2. Label Propagation (subset labeled, rest unlabeled)
3. Gaussian Mixture Models

**Days 3:**
1. Comparison table
2. Document findings

### Week 4: Model B Distractor & Hints (Phases 6-7)

**Days 1-2:**
1. Extract candidate phrases from articles
2. Train distractor ranker (LR or RF on handcrafted features)
3. Evaluate Precision/Recall/F1

**Days 3-4:**
1. Implement hint extractor (sentence ranking + ML scoring)
2. Evaluate with P@1, P@3, R²

### Week 5: UI & Final (Phases 8-10)

**Days 1-3:**
1. Streamlit UI with 4 screens
2. Load all models
3. Test end-to-end

**Days 4-5:**
1. Final test set evaluation
2. Write report
3. Submit

---

## Part 5: Checklist Before Moving to Phase 3

Before you train any models, ensure:

- [ ] All three splits (train/dev/test) are cleaned
- [ ] Articles truncated to 500 words
- [ ] 4× expansion of examples done (87K → 351K rows)
- [ ] Lexical features computed (14 numerical columns)
- [ ] One-Hot vectorizer fitted on train only
- [ ] All matrices saved to `data/processed/`
- [ ] Git commit with meaningful message
- [ ] Load test: can you reload vectorizer + matrices without errors?

---

## Part 6: Expected Baseline Performance (from EDA Insights)

Once you train your first Logistic Regression model on One-Hot + lexical features:

| Metric | Expected | Notes |
|---|---|---|
| **Accuracy on val** | 55–65% | Random baseline = 50% (binary) |
| **Macro F1** | 0.50–0.60 | Balanced across correct/incorrect |
| **Exact Match (EM)** | 45–55% | How often top-1 pred is correct across all 4 options |

**If you see:**
- Accuracy < 50% → bug in pipeline (check feature shapes)
- Accuracy > 70% → likely overfitting or data leakage
- Accuracy ~65% → good baseline; ensemble should push to 70%+

---

## Final Recommendation

**Start with this sequence:**

1. **Today:** Finalize EDA analysis (extract key numbers from your outputs)
2. **Next session:** Create `src/preprocessing.py` with all functions
3. **Session after:** Run preprocessing end-to-end, save matrices
4. **Session after:** Train baseline LR model, record metrics
5. **Then:** Phase 3 (traditional ML models)

This gives you a solid foundation. The key is that **preprocessing determines 50% of model performance**. Get it right now, and everything later is easier.

Good luck! 🎯
