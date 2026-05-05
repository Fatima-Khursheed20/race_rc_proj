# Intelligent Reading Comprehension & Quiz Generation System
## Complete Execution Plan — BS (CS) Spring 2026 | NUCES FAST Islamabad

---

> **How to use this document:** Read it top to bottom once before writing a single line of code. Every phase builds on the previous one. Each section tells you *what* to do, *why* you're doing it, *what the output looks like*, and *what can go wrong*. Optional/bonus tasks are marked with `[BONUS]`.

---

## Table of Contents

1. [Project at a Glance](#1-project-at-a-glance)
2. [Environment Setup](#2-environment-setup)
3. [Phase 1 — Data Exploration (EDA)](#3-phase-1--data-exploration-eda)
4. [Phase 2 — Preprocessing & Feature Engineering](#4-phase-2--preprocessing--feature-engineering)
5. [Phase 3 — Model A: Traditional ML (Supervised)](#5-phase-3--model-a-traditional-ml-supervised)
6. [Phase 4 — Model A: Unsupervised & Semi-Supervised](#6-phase-4--model-a-unsupervised--semi-supervised)
7. [Phase 5 — Model A: Ensemble Strategy](#7-phase-5--model-a-ensemble-strategy)
8. [Phase 6 — Model B: Distractor Generator](#8-phase-6--model-b-distractor-generator)
9. [Phase 7 — Model B: Hint Generator](#9-phase-7--model-b-hint-generator)
10. [Phase 8 — User Interface (All 4 Screens)](#10-phase-8--user-interface-all-4-screens)
11. [Phase 9 — Evaluation, Tuning & Testing](#11-phase-9--evaluation-tuning--testing)
12. [Phase 10 — Final Report](#12-phase-10--final-report)
13. [Phase 11 — Submission Checklist](#13-phase-11--submission-checklist)
14. [Concepts Glossary](#14-concepts-glossary)
15. [Common Mistakes to Avoid](#15-common-mistakes-to-avoid)

---

## 1. Project at a Glance

### What the system does (end-to-end flow)

```
User pastes a reading passage
        |
        v
[Preprocessing Module]
  - clean text
  - convert to numbers (One-Hot Encoding)
        |
   +----+----+
   |         |
   v         v
[Model A]  [Model B]
Q&A Gen/   Distractor
Verifier   + Hint Gen
   |         |
   +----+----+
        |
        v
[UI — 4 screens]
  Screen 1: Paste article, press Submit
  Screen 2: See question, pick answer A/B/C/D, press Check
  Screen 3: Reveal hints one by one
  Screen 4: Developer analytics dashboard
```

### Two AI Pipelines

| Pipeline | Input | Output | Models Used |
|---|---|---|---|
| Model A | Article text | Generated question + correct answer label | Logistic Regression, SVM, Naive Bayes, Random Forest, XGBoost, K-Means, GMM, Label Propagation |
| Model B | Article + Question + Correct Answer | 3 distractor options + 3 graduated hints | LR, Random Forest, Word2Vec, sentence-transformers cosine ranking |

### Grading breakdown (know where the marks are)

| Component | Marks | What earns full marks |
|---|---|---|
| EDA & Preprocessing | 10 | Visualizations, clean code, documented pipeline |
| Model A Traditional ML | 15 | ≥2 models, feature engineering, comparison table |
| Model A Unsupervised/Semi-Supervised | 20 | At least 1 approach, evaluated with correct metrics |
| Model A Ensemble | 5 | Shows improvement over individual models |
| Model B Distractor Gen | 15 | Plausible distractors, P/R/F1/Confusion Matrix |
| Model B Hint Gen | 10 | Graduated hints that guide without revealing |
| User Interface | 15 | All 4 screens, smooth UX, error handling |
| Final Report | 5 | Clear methodology, results, discussion, limitations |
| Code Quality | 5 | Readable, documented, meaningful commits |
| **TOTAL** | **100** | |

---

## 2. Environment Setup

### 2.1 Folder Structure

Create this exact folder structure before writing any code. Having a clean structure from day one prevents chaos later.

```
race_rc_project/
├── data/
│   ├── raw/                  ← put train.csv, val.csv, test.csv here
│   └── processed/            ← save cleaned/encoded data here
├── models/
│   ├── model_a/
│   │   └── traditional/      ← save .pkl files for LR, SVM, etc.
│   └── model_b/
│       └── traditional/      ← save .pkl files for distractor ranker, hint scorer
├── src/
│   ├── preprocessing.py      ← all cleaning + encoding functions
│   ├── model_a_train.py      ← training script for Model A
│   ├── model_b_train.py      ← training script for Model B
│   ├── inference.py          ← unified inference: given article → question + answer + distractors + hints
│   └── evaluate.py           ← all metric computation
├── ui/
│   ├── app.py                ← main Streamlit entry point
│   └── components/           ← reusable UI pieces (optional to modularize)
├── notebooks/
│   ├── EDA.ipynb             ← Phase 1 work
│   └── experiments.ipynb     ← Phase 3–7 experiments
├── tests/
│   └── test_inference.py     ← unit tests
├── requirements.txt
├── README.md
└── report/
    └── final_report.pdf
```

### 2.2 Python Version

Use Python 3.9 or 3.10. Avoid 3.12+ because some libraries (especially older gensim) have compatibility issues.

### 2.3 Install all dependencies

Create `requirements.txt` with these contents and pin exact versions:

```
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
xgboost==2.0.3
gensim==4.3.2
sentence-transformers==2.3.1
streamlit==1.29.0
joblib==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
jupyter==1.0.0
scipy==1.11.4
nltk==3.8.1
```

Install with: `pip install -r requirements.txt`

### 2.4 Download the RACE Dataset

- Go to Kaggle and search "RACE Reading Comprehension Dataset"
- Download the ZIP, extract it
- You should get three files: `train.csv`, `val.csv`, `test.csv`
- Place all three inside `data/raw/`
- Expected sizes: train ~87,866 rows, val ~4,887 rows, test ~4,934 rows
- Each row has columns: `id, article, question, A, B, C, D, answer`

### 2.5 Set up Git

Initialize a git repo immediately. Commit after every meaningful step. Your commit history is worth 5 marks.

```
git init
git add .
git commit -m "initial project structure"
```

Good commit message examples:
- `"add preprocessing pipeline with OHE vectorizer"`
- `"train logistic regression baseline — val accuracy 58%"`
- `"fix label encoding bug in answer verification"`

---

## 3. Phase 1 — Data Exploration (EDA)

**Deliverable:** `notebooks/EDA.ipynb`
**Marks at stake:** 10/100
**Time estimate:** Days 1–2

### 3.1 What EDA is and why you do it

EDA (Exploratory Data Analysis) means looking at your data carefully before training any model. You want to understand: how big is it? is it balanced? what do the texts look like? what patterns exist? This directly informs every decision you make in preprocessing and modeling.

### 3.2 Loading the data

Load all three splits using pandas. Print the shape (rows × columns) and the column names. Print the first 5 rows using `.head()`. Check for null/missing values using `.isnull().sum()`.

**What to look for:**
- Are there any rows with missing articles or questions? (There shouldn't be, but verify)
- Does the `answer` column contain only A, B, C, D?

### 3.3 Visualizations you MUST include

**Plot 1 — Answer Label Distribution (Bar Chart)**

Count how often each label (A, B, C, D) appears in the training set. Plot as a bar chart with counts labeled on top of each bar.

*Why this matters:* If A appears 40% of the time and D only 20%, your model might just learn to predict A always. This is called class imbalance and affects which evaluation metrics you trust.

*What you should find:* RACE is roughly balanced — each answer appears about 25% of the time.

**Plot 2 — Article Length Distribution (Histogram)**

Compute the number of words in each article (split by spaces). Plot as a histogram with ~30 bins.

*Why this matters:* Very long articles will make One-Hot Encoding very sparse and slow. You may decide to truncate articles to the first N words.

*What you should find:* Most articles are 200–400 words. Some are very long (800+).

**Plot 3 — Question Length Distribution (Histogram)**

Same as above but for the question column.

**Plot 4 — Question Type Distribution (Bar Chart)**

Categorize each question by its first word: Who, What, Where, When, Why, How, Which, Other. Count and plot.

*Why this matters:* Naive Bayes will be trained to classify question types. You need to know if these classes are balanced.

**Plot 5 — Answer Option Length Distribution (Box Plot)**

For all four options (A, B, C, D), plot a box plot of word count. Compare correct answers vs. incorrect answers.

*Why this matters:* If correct answers tend to be longer, your model might cheat by learning length instead of semantics.

**Plot 6 — Word Cloud of Most Common Article Words** `[BONUS]`

Use the `wordcloud` library to generate a word cloud of the most frequent words across all articles. This gives a qualitative feel for the dataset's vocabulary.

**Plot 7 — Passage-level Statistics Table**

Create a summary table with:
- Mean, median, min, max, std for article word count
- Mean, median, min, max, std for question word count
- Mean, median, min, max, std for answer option word count
- Total vocabulary size (unique words across all articles in train set)

### 3.4 Insights to write in EDA notebook

After each plot, write a markdown cell (not code) explaining what you observe and what it means for your modeling choices. For example: "The answer distribution is roughly uniform, so accuracy is a fair metric and class imbalance is not a major concern."

---

## 4. Phase 2 — Preprocessing & Feature Engineering

**Deliverable:** `src/preprocessing.py` with importable functions
**Time estimate:** Days 3–4

### 4.1 What preprocessing means

Machine learning models cannot read text. They can only process numbers. Preprocessing is the pipeline that converts raw text → clean text → numerical feature vectors that models can use.

### 4.2 Text Cleaning

Write a function `clean_text(text)` that does the following steps in order:

1. **Lowercase** — convert all characters to lowercase so "The" and "the" are treated the same
2. **Remove punctuation** — strip characters like `.`, `,`, `!`, `?`, `"`, `'`. Use Python's `re` module with a pattern like `[^\w\s]`
3. **Remove extra whitespace** — collapse multiple spaces into one using `re.sub(r'\s+', ' ', text).strip()`
4. **Optional — remove stopwords** — stopwords are very common words (the, is, a, an, of) that carry little meaning. You can use NLTK's stopword list. However, removing stopwords can hurt model performance if the model needs grammatical context, so test both ways.

Apply `clean_text` to the `article`, `question`, and all four option columns (A, B, C, D).

### 4.3 Creating Training Examples for Model A (Answer Verification)

This is the most important preprocessing step to understand.

**The task:** Given (article, question, option), predict whether this option is correct (label=1) or wrong (label=0).

**How to create training examples from RACE rows:**

Each RACE row gives you one article, one question, four options, and one correct answer label (e.g., "B"). You expand each row into FOUR training examples:

- Example 1: input = article + " [SEP] " + question + " [SEP] " + option_A, label = 1 if answer=="A" else 0
- Example 2: input = article + " [SEP] " + question + " [SEP] " + option_B, label = 1 if answer=="B" else 0
- Example 3: input = article + " [SEP] " + question + " [SEP] " + option_C, label = 1 if answer=="C" else 0
- Example 4: input = article + " [SEP] " + question + " [SEP] " + option_D, label = 1 if answer=="D" else 0

The `[SEP]` token is just a separator string — it helps the model know where the article ends and the question begins.

This turns 87,866 rows into ~351,464 training examples (4× expansion). Label distribution will be 25% positive (1) and 75% negative (0) — this is mild class imbalance.

**Important practical note:** The full expanded dataset with One-Hot Encoding will be very large in memory. Consider subsetting to the first 20,000–30,000 rows of train.csv for initial experiments, then scaling up once your pipeline works.

### 4.4 One-Hot Encoding (Primary Feature Method)

**What One-Hot Encoding is:**
You build a vocabulary of the most common N words across your entire training corpus. Each training example (a concatenated string) becomes a vector of length N where each position is 1 if that vocabulary word appears in the string, and 0 if it does not.

For example, if vocabulary = ["cat", "dog", "runs", "fast"] and your text is "the cat runs":
→ vector = [1, 0, 1, 0] (cat present, dog absent, runs present, fast absent)

This is called binary bag-of-words. Scikit-learn's `CountVectorizer` with `binary=True` does exactly this.

**Implementation steps:**

Step 1 — Create the vectorizer:
Use `CountVectorizer` from scikit-learn with these settings:
- `binary=True` — gives 1/0 instead of raw counts
- `max_features=10000` — keep only top 10,000 most frequent words (start here, try 5000 and 20000 too)
- `min_df=2` — ignore words appearing in fewer than 2 documents (removes typos/noise)
- `ngram_range=(1,1)` — use single words only (unigrams). `[BONUS]` try (1,2) for bigrams

Step 2 — Fit on training data only:
Call `vectorizer.fit(X_train_texts)`. Never fit on val or test data — that would be data leakage (your model would cheat by knowing validation vocabulary).

Step 3 — Transform all splits:
Call `vectorizer.transform(X_train)`, `vectorizer.transform(X_val)`, `vectorizer.transform(X_test)`.
The result is a sparse matrix (most values are 0, only stores non-zero entries for efficiency).

Step 4 — Save the fitted vectorizer:
Use `joblib.dump(vectorizer, 'models/model_a/traditional/ohe_vectorizer.pkl')` so you can reload it during inference without refitting.

### 4.5 Handcrafted Lexical Features

In addition to One-Hot Encoding, compute these numerical features for each training example. These will be especially useful for Random Forest and XGBoost.

Create a function `compute_lexical_features(article, question, option, correct_answer)` that returns a list of numbers:

1. **word_overlap_article_option** — count of words that appear in BOTH the article and the option, divided by option length. High overlap = option is likely extractable from article.

2. **word_overlap_question_option** — count of words in BOTH the question and the option, divided by option length.

3. **option_length** — number of words in the option.

4. **article_length** — number of words in the article.

5. **question_length** — number of words in the question.

6. **option_position_in_article** — does the option text appear as a substring in the article? 1 if yes, 0 if no.

7. **char_length_ratio** — len(option) / (len(article) + 1). Longer articles might have longer correct answers.

8. **unique_words_in_option** — count of unique words in the option (measures specificity).

9. **question_type_who** — 1 if question starts with "who", 0 otherwise.
10. **question_type_what** — 1 if starts with "what".
11. **question_type_where** — 1 if starts with "where".
12. **question_type_when** — 1 if starts with "when".
13. **question_type_why** — 1 if starts with "why".
14. **question_type_how** — 1 if starts with "how".

These 14 features, when stacked horizontally with the One-Hot vectors using `scipy.sparse.hstack`, give your models richer information.

### 4.6 TF-IDF Vectorization `[BONUS/OPTIONAL]`

TF-IDF stands for Term Frequency–Inverse Document Frequency. It's like One-Hot but smarter: instead of 0/1, each word gets a score based on how often it appears in this document (TF) divided by how common it is across all documents (IDF). Rare but present words get high scores; common words like "the" get low scores.

Use scikit-learn's `TfidfVectorizer` with the same `max_features` setting. Fit on training only. Compare performance against One-Hot Encoding in your results table — the document says TF-IDF is optional but will earn you bonus marks if included with a comparison.

### 4.7 Cosine Similarity Feature

Cosine similarity measures how "close" two vectors are. If two texts are very similar, their One-Hot vectors point in similar directions in high-dimensional space.

For each training example, compute:
- Cosine similarity between the One-Hot vector of the article and the One-Hot vector of the option
- Cosine similarity between the One-Hot vector of the question and the One-Hot vector of the option

Use `sklearn.metrics.pairwise.cosine_similarity`. Add these two numbers as additional features.

### 4.8 Saving Processed Data

After creating all feature matrices, save them to `data/processed/` using `joblib.dump` or numpy's `scipy.sparse.save_npz` (for sparse matrices). This way, if you need to retrain a model you don't re-run preprocessing from scratch (which can take 10–20 minutes on the full dataset).

Save:
- `X_train_ohe.npz` — sparse matrix, One-Hot features
- `X_val_ohe.npz`
- `X_test_ohe.npz`
- `y_train.npy`, `y_val.npy`, `y_test.npy` — label arrays (0 or 1)
- `X_train_lexical.npy` — dense matrix of handcrafted features
- `ohe_vectorizer.pkl` — fitted vectorizer

---

## 5. Phase 3 — Model A: Traditional ML (Supervised)

**Deliverable:** Trained models saved as .pkl files. Metrics reported in experiments.ipynb.
**Marks at stake:** 15/100
**Time estimate:** Days 4–5

### 5.1 Overview of the task

Model A has two sub-tasks:

**Sub-task 1 — Answer Verification:**
Given (article + question + option) as a feature vector, predict: is this option the correct answer? (Binary classification: 1=correct, 0=wrong)

During inference, you run this for all four options and pick the one with the highest predicted probability as the system's answer.

**Sub-task 2 — Question Generation:**
Given an article, produce a plausible question. This uses a template-based rule system combined with an ML ranker (covered in section 5.6).

### 5.2 Model 1 — Logistic Regression

**What it is:** A linear model that learns a weight for every feature. At prediction time, it multiplies feature values by their learned weights and passes the sum through a sigmoid function to get a probability between 0 and 1.

**Why it works for text:** With 10,000 One-Hot features, Logistic Regression learns which words in the combined article+question+option text are predictive of being the correct answer.

**Implementation steps:**

1. Load `X_train_ohe` and `y_train` from disk
2. Optionally concatenate lexical features: `scipy.sparse.hstack([X_train_ohe, X_train_lexical])`
3. Initialize: `LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', class_weight='balanced')`
   - `C` controls regularization strength (smaller C = stronger regularization = simpler model)
   - `class_weight='balanced'` handles the 1:3 class imbalance automatically
   - `max_iter=1000` gives the optimizer enough iterations to converge
4. Fit on training data
5. Predict on validation data
6. Compute and print: Accuracy, Macro F1, Precision, Recall, Confusion Matrix
7. Save model with joblib

**Hyperparameter tuning for LR:**
Try values of C: [0.01, 0.1, 1.0, 10.0]. For each, record val accuracy. This is a manual grid search. `[BONUS]` Use `GridSearchCV` to automate this with 5-fold cross-validation.

### 5.3 Model 2 — Support Vector Machine (SVM)

**What it is:** SVM finds the hyperplane that maximally separates two classes in feature space. For text classification, `LinearSVC` is the standard choice because full kernel SVMs are too slow on high-dimensional sparse data.

**Why it works:** SVM is excellent at finding decision boundaries in high-dimensional sparse spaces (exactly what One-Hot text data is).

**Implementation steps:**

1. Use `LinearSVC(C=1.0, max_iter=2000, class_weight='balanced')`
2. Because LinearSVC doesn't natively output probabilities (needed for ensemble), wrap it: `CalibratedClassifierCV(LinearSVC(...), cv=5)` — this adds probability calibration
3. Fit, evaluate, save

**Features for SVM:** Use One-Hot Encoding + cosine similarity features (section 4.7 features stacked in)

**Hyperparameter tuning for SVM:** Try C in [0.01, 0.1, 1, 10]

### 5.4 Model 3 — Naive Bayes (for Question Type Classification)

**What it is:** A probabilistic classifier based on Bayes' theorem. "Naive" because it assumes all features are independent of each other (which isn't true, but works surprisingly well in practice for text).

**Why it's used here:** For classifying question TYPE (Who/What/Where/When/Why/How), Naive Bayes is fast and effective. This is a separate, simpler classification task.

**Implementation steps:**

1. Create a new dataset from training data: X = question text, y = question type label (extracted from the first word of each question)
2. Labels: 0=Who, 1=What, 2=Where, 3=When, 4=Why, 5=How, 6=Other
3. Vectorize questions with CountVectorizer (bag-of-words, small vocabulary ~2000)
4. Use `MultinomialNB(alpha=1.0)` — `alpha` is the Laplace smoothing parameter
5. Fit, evaluate with accuracy and per-class F1
6. Save model

**Why question type matters:** In the UI, knowing the question type can help generate better template questions and display appropriate hints.

### 5.5 Model 4 — Random Forest (for Difficulty Estimation) `[BONUS for full marks]`

**What it is:** An ensemble of many Decision Trees. Each tree is trained on a random subset of training examples and a random subset of features. The trees vote and the majority decision wins.

**Why it's used here:** For predicting how difficult a question is (easy/medium/hard). You create difficulty labels from RACE metadata or heuristics (e.g., questions requiring reasoning across multiple sentences = hard).

**Creating difficulty labels (since RACE doesn't directly provide them):**

Define a simple heuristic:
- Easy: the correct answer appears verbatim as a substring in the article
- Medium: the correct answer shares 3+ words with a sentence in the article
- Hard: the correct answer requires inferring beyond what's explicitly stated

**Implementation steps:**

1. Apply heuristic to label each training example as 0 (easy), 1 (medium), or 2 (hard)
2. Features: handcrafted lexical features from section 4.5 (article length, question length, word overlap, etc.) — these are more interpretable for a tree-based model than sparse One-Hot vectors
3. Use `RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)`
4. Fit, evaluate, save
5. Display the feature importance ranking (which features matter most)

### 5.6 Model 5 — XGBoost (for Answer Verification) `[BONUS]`

**What it is:** Extreme Gradient Boosting. Builds trees sequentially where each new tree corrects the errors of the previous trees. Generally achieves higher accuracy than Random Forest.

**Implementation steps:**

1. Use `XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, use_label_encoder=False, eval_metric='logloss')`
2. Input features: semantic similarity (cosine sim from One-Hot) + lexical features (NOT the full 10,000-dim OHE vector — XGBoost is slow on very high-dimensional sparse data)
3. Fit with early stopping on val set to prevent overfitting
4. Evaluate, save

### 5.7 Template-Based Question Generation

This is the "generation" part of Model A. Given an article, produce a question.

**Step 1 — Sentence Selection (ML-assisted):**

Split the article into sentences by splitting on periods (or use a simple heuristic: split on ". " where the next character is uppercase).

For each sentence, compute its One-Hot vector. Compute cosine similarity between the sentence vector and the correct answer vector. The sentence with the highest similarity is the "anchor sentence" — this is where the answer lives.

**Step 2 — Apply Wh-word Templates:**

Define template rules:

Template rule examples:
- If the sentence contains a person's name (capitalized word not at start): generate "Who + [rest of sentence without the name] + ?"
- If the sentence contains a year (4-digit number): generate "When + [rest of sentence without year] + ?"
- If the sentence contains a location (word following "in" or "at"): generate "Where + [rest of sentence] + ?"
- If the sentence contains a reason (word following "because" or "since"): generate "Why + [rest of sentence] + ?"
- Default/fallback: "What is the main idea of the passage?"

Person name detection without NLP tools: a word is a "name" if it is capitalized, not the first word of the sentence, and not a common word (you can maintain a small list of 200 common non-name capitalized words like "The", "In", "It", etc.).

**Step 3 — ML Ranker:**

After generating multiple candidate questions (one per template applied to the top-3 sentences), rank them using a trained SVM or Random Forest.

Features for the ranker:
- Number of words in the generated question (fluency proxy)
- Does the question contain a Wh-word? (1/0)
- Word overlap between question and anchor sentence
- Position of anchor sentence in article (earlier = usually more important)
- Length of anchor sentence

Label each candidate question as good (1) or bad (0) by comparing against the actual RACE question (good = high word overlap with actual question).

Train this ranker on training split. At inference time, generate candidates and pick the top-ranked one.

### 5.8 Evaluation for Model A Traditional ML

Compute and display the following in a clear comparison table in your notebook:

| Model | Val Accuracy | Val Macro F1 | Val Precision | Val Recall | Exact Match |
|---|---|---|---|---|---|
| Logistic Regression (OHE) | ? | ? | ? | ? | ? |
| SVM (OHE + cosine) | ? | ? | ? | ? | ? |
| XGBoost (lexical + cosine) | ? | ? | ? | ? | ? |

**Exact Match (EM) for verification:**
EM is computed differently from binary accuracy. For each original RACE question, run your verifier on all four options, pick the option with highest probability, and check if it matches the gold answer label. EM = fraction of questions where the top-ranked option is the correct one.

**Confusion Matrix:**
Plot using `ConfusionMatrixDisplay` from sklearn. Should show True Positive (correctly predicted correct answers), True Negative (correctly predicted wrong answers), False Positive (predicted correct but actually wrong), False Negative (predicted wrong but actually correct).

---

## 6. Phase 4 — Model A: Unsupervised & Semi-Supervised

**Deliverable:** Experiments in `notebooks/experiments.ipynb`. Comparison table.
**Marks at stake:** 20/100 — this is the highest single component!
**Time estimate:** Week 2, Days 1–2

### 6.1 Why this component is worth 20 marks

The project specifically requires you to explore what can be learned WITHOUT labels. This tests your understanding of the difference between supervised and unsupervised learning, and demonstrates that you can think beyond just "train a classifier."

### 6.2 K-Means Clustering

**What it is:** K-Means partitions your data into K groups (clusters) such that items within a cluster are as similar as possible. It starts with K random "centroids" (cluster centers), assigns each point to the nearest centroid, recomputes centroids as the mean of assigned points, and repeats until stable.

**What you're clustering:**
- Question-answer pairs represented as One-Hot feature vectors
- Each data point = (question One-Hot vector) concatenated with (correct answer One-Hot vector)
- Use a reduced-dimension version: apply PCA to reduce 10,000 dims to 50–100 dims first (otherwise K-Means is very slow and inaccurate in high dimensions)

**PCA (Principal Component Analysis):** Reduces dimensionality by finding the directions of maximum variance. You lose some information but keep the most important structure.

**Implementation steps:**

1. Take a sample of 5,000–10,000 training examples (K-Means is slow on full dataset)
2. Apply PCA: `PCA(n_components=50)` — reduce to 50 dimensions. Fit on training sample only.
3. Run K-Means: start with K=4 (matching the 4 answer positions), then try K=2,4,6,8
4. For each K, compute: Inertia (within-cluster sum of squares — lower is better), Silhouette Score (higher is better, range -1 to 1)
5. Plot the "elbow curve": x-axis = K, y-axis = inertia. The "elbow" point suggests the optimal K.

**Evaluating cluster quality:**

Silhouette Score: measures how well-separated clusters are. Score close to 1 = excellent separation, close to 0 = overlapping clusters, negative = wrong cluster assignment.

Clustering Purity: for each cluster, find the most common true label (A, B, C, or D). Purity = fraction of points that match the dominant label. Compute per-cluster and overall.

**What to report:**
- Silhouette score for K=4
- Purity score for K=4
- Whether clusters correspond to answer position (A/B/C/D) or something else (e.g., question type)
- A 2D visualization: apply PCA to 2 dimensions for plotting, color-code by cluster

### 6.3 Label Propagation (Semi-Supervised)

**What it is:** You have a small labeled dataset and a large unlabeled dataset. Label Propagation builds a graph where nodes are data points and edges represent similarity. Known labels "spread" across edges to unlabeled nodes, like dye diffusing through water.

**Why this is relevant:** In real educational settings, you might only have labeled data for 1,000 questions but have access to 90,000 unlabeled articles. This simulates that scenario.

**Implementation steps:**

1. Take your full feature matrix `X_train_ohe`
2. Create a partial label array `y_partial`:
   - Keep labels for the first 2,000 examples
   - Set all other labels to -1 (meaning "unlabeled" in scikit-learn's semi-supervised API)
3. Apply PCA first (100 components) to reduce dimensionality — Label Propagation on 10,000-dim data is intractable
4. Initialize: `LabelPropagation(kernel='knn', n_neighbors=7, max_iter=1000)`
   - `kernel='knn'` means connect each point to its K nearest neighbors
   - `n_neighbors=7` — try 5, 7, 10 and compare
5. Fit on the partial label array
6. Predict on validation set and compute F1

**What to report:**
- F1 score with 2,000 labeled examples vs. supervised F1 with 2,000 examples vs. supervised F1 with all examples
- This shows whether semi-supervised learning helps when labels are scarce

**Alternative: Label Spreading** `[BONUS]`

`LabelSpreading` is similar but softer — it allows the algorithm to change even labeled points slightly, making it more robust to label noise. Try it with `alpha=0.2` (soft clamping) and compare to Label Propagation.

### 6.4 Gaussian Mixture Models (GMM)

**What it is:** Like K-Means but probabilistic. Instead of hard cluster assignments, every data point gets a probability of belonging to each cluster. Internally models each cluster as a Gaussian (bell-curve) distribution.

**Why this is useful:** For question-answer patterns, soft memberships make sense — a question might be "50% like a Why question, 50% like a What question."

**Implementation steps:**

1. Use PCA-reduced features (same 50-component version as K-Means)
2. Initialize: `GaussianMixture(n_components=4, covariance_type='full', random_state=42, max_iter=200)`
   - `n_components=4` — same as K-Means experiment for comparability
   - `covariance_type='full'` means each cluster has its own full covariance matrix (most expressive but slowest). Also try 'diag' and 'spherical'.
3. Fit on training sample
4. Get soft probabilities: `gmm.predict_proba(X)` — shape (N, 4)
5. Get hard assignments: `gmm.predict(X)` — shape (N,)
6. Compute BIC (Bayesian Information Criterion) and AIC (Akaike Information Criterion) for different numbers of components. Lower is better. Plot these to find optimal K.

**What to report:**
- Cluster purity for GMM (same metric as K-Means)
- BIC/AIC curves
- Comparison: does GMM find different/better clusters than K-Means?

### 6.5 Comparison Table (Required for Full Marks)

Create a table in your notebook comparing all approaches:

| Approach | Metric | Score | Notes |
|---|---|---|---|
| Supervised LR (all labels) | Macro F1 | ? | Ceiling — best possible |
| Supervised LR (2k labels) | Macro F1 | ? | Weak supervision baseline |
| Label Propagation (2k labels) | Macro F1 | ? | Semi-supervised |
| K-Means (K=4) | Purity | ? | Unsupervised |
| GMM (K=4) | Purity | ? | Unsupervised |

---

## 7. Phase 5 — Model A: Ensemble Strategy

**Deliverable:** Final Model A results on test set with ensemble beating individual models.
**Marks at stake:** 5/100
**Time estimate:** Week 2, Day 3

### 7.1 Why ensembling works

Individual models make different kinds of mistakes. A Logistic Regression might fail on passages with unusual vocabulary. An SVM might struggle with long articles. By combining predictions, the ensemble is more robust — when one model is wrong, others can correct it.

### 7.2 Soft Voting Ensemble

**What it is:** Each model outputs a probability for each class. You average the probabilities and pick the class with the highest average.

**Implementation steps:**

1. Collect probability outputs from LR, SVM (CalibratedClassifierCV), and Naive Bayes on validation set
2. Average: `avg_probs = (lr_probs + svm_probs + nb_probs) / 3`
3. Final prediction: `np.argmax(avg_probs, axis=1)`
4. Evaluate on validation set — compare against best individual model

**VotingClassifier shortcut:** scikit-learn has a built-in `VotingClassifier` with `voting='soft'` that handles this automatically.

### 7.3 Hard Voting Ensemble

**What it is:** Each model makes a class prediction (not probability). The class that gets the most votes wins.

Use `VotingClassifier` with `voting='hard'`. Include LR + SVM + NB.

### 7.4 Stacking Ensemble `[BONUS — extra thoroughness]`

**What it is:** A two-level approach. Level-1 models (LR, SVM, NB) each produce predictions. A Level-2 "meta-classifier" (usually a simple LR) is trained on the Level-1 predictions as its input features.

**Implementation:**

1. Use `StackingClassifier` from scikit-learn
2. Base estimators: [('lr', LR_model), ('svm', SVM_model), ('nb', NB_model)]
3. Final estimator: `LogisticRegression()`
4. Use 5-fold cross-validation internally (`cv=5`) to prevent overfitting

**Important:** The stacking meta-classifier must be trained on held-out predictions (cross-validated), not the training set predictions — otherwise it overfits. `StackingClassifier` handles this automatically.

### 7.5 Final Model A Reporting

Report on the test set:
- Best individual model: name, Accuracy, Macro F1, EM
- Best ensemble: type, Accuracy, Macro F1, EM
- Delta (improvement): by how much did ensembling help?

---

## 8. Phase 6 — Model B: Distractor Generator

**Deliverable:** Working distractor pipeline producing 3 plausible wrong answers per question.
**Marks at stake:** 15/100
**Time estimate:** Week 2, Days 3–4

### 8.1 What makes a good distractor

A good distractor must be:
1. **Plausible** — looks like a possible correct answer to someone who hasn't read the passage
2. **Incorrect** — definitively wrong according to the passage
3. **Diverse** — the three distractors should not be slight paraphrases of each other
4. **Grammatically consistent** — all options should be the same grammatical form (all nouns, all sentences, etc.)

Getting all four constraints right simultaneously is what makes this hard.

### 8.2 Pipeline Overview

```
Input: article, question, correct_answer
         |
Step 1: Extract candidate phrases from article
         |
Step 2: Compute features for each candidate
         |
Step 3: ML Ranker scores candidates
         |
Step 4: Select top-3 non-answer candidates with diversity check
         |
Output: distractor_1, distractor_2, distractor_3
```

### 8.3 Step 1 — Candidate Phrase Extraction

**Goal:** Produce a pool of possible distractor phrases from the article.

**Method 1 — n-gram extraction (primary, no NLP tools needed):**

Split the article into words (after cleaning). Extract all unigrams (single words), bigrams (2-word pairs), and trigrams (3-word groups). Filter by:
- Remove candidates that match the correct answer (exact string match)
- Remove stopwords from unigrams (keep only content words)
- Keep candidates with frequency ≥ 2 in the article (single appearances are likely noise)
- Keep candidates whose length (in characters) is within 50% of the correct answer's length (grammatical consistency proxy)

**Method 2 — Frequency-based selection (alternative):**

Identify the top-30 most frequent content words in the article. These are high-frequency candidates. Among them, find words that do NOT appear in the correct answer string. These become candidates.

You should end up with 20–100 candidate phrases per article.

### 8.4 Step 2 — Feature Engineering for Candidates

For each candidate phrase, compute these features:

1. **ohe_cosine_sim_to_answer** — cosine similarity between candidate's One-Hot vector and correct answer's One-Hot vector. A good distractor should have moderate similarity (not too similar = obviously wrong, not too different = obviously unrelated).

2. **ohe_cosine_sim_to_question** — cosine similarity to the question. Higher = more related to what's being asked.

3. **char_length_ratio** — len(candidate) / len(correct_answer). Good distractors have similar length to correct answers.

4. **word_length_ratio** — same but in words.

5. **passage_frequency** — how many times the candidate appears in the article. Too frequent = might be the actual answer. Too rare = might be noise.

6. **position_in_article** — what fraction of the article has passed when this candidate first appears (0.0–1.0). Earlier mentions tend to be more important.

7. **word_overlap_with_answer** — count of words shared with correct answer / answer length. Should be low for distractors.

8. **word_overlap_with_question** — count of words shared with question / question length.

9. **starts_with_same_word** — 1 if candidate starts with same word as correct answer. Ensures grammatical consistency.

10. **is_proper_noun_candidate** — 1 if first letter is uppercase (simple heuristic for names/places).

### 8.5 Step 3 — Train the ML Ranker

**Creating training labels:**

For each RACE training row, you already know the 3 actual distractors (options A, B, C, D minus the correct answer). Label these as 1 (good distractor). Label all other extracted candidates from the article as 0.

This gives you a binary classification problem: given candidate features, predict whether this candidate was used as a distractor.

**Models to train:**

Primary: `LogisticRegression(C=1.0, class_weight='balanced')`
- Input: 10 numerical features from step 8.4
- Output: probability that candidate is a good distractor

Alternative: `RandomForestClassifier(n_estimators=100)`
- Same features, often better because feature interactions matter here

**Training steps:**
1. Loop over all training rows, extract candidates, label them
2. Compute features for all candidates
3. Combine into one big feature matrix and label array
4. Train/evaluate on 80/20 split (within the training data)
5. Save trained ranker

### 8.6 Step 4 — Selecting Top-3 with Diversity

After ranking all candidates by predicted probability, naively taking top-3 might give three nearly identical candidates (e.g., three consecutive bigrams from the same sentence).

**Diversity enforcement:**

Maximal Marginal Relevance (MMR) approach (simplified):
1. Select the top-ranked candidate
2. For each remaining candidate, compute a diversity score: `diversity = predicted_prob - λ * max_sim_to_already_selected`
   - `λ` is a balance parameter (try 0.3–0.7)
   - `max_sim_to_already_selected` = highest cosine similarity between this candidate and any already-selected distractor
3. Pick the candidate with highest diversity score
4. Repeat until 3 distractors selected

This ensures the three distractors are both plausible AND different from each other.

### 8.7 Word2Vec Nearest Neighbour Approach `[BONUS — Alternative Method]`

**What it is:** Use a pre-trained Word2Vec model (trained on Google News, 3 million words, 300 dimensions). Find words closest to the correct answer word in the embedding space. These are semantically similar words — plausible but possibly different meaning.

**Implementation steps:**

1. Load pre-trained Word2Vec: `api.load("word2vec-google-news-300")` via gensim. Note: this is a ~1.6GB download — do it once and cache it.
2. For the correct answer, split into individual words, get the most "important" word (longest, or most specific)
3. Find top-20 nearest neighbors: `model.most_similar("answer_word", topn=20)`
4. Filter neighbors: remove words that appear in the article (they're extractable = not plausible distractors), remove words that are too similar to the correct answer (exact match), remove stopwords
5. Take the top-3 filtered neighbors as distractors

**Limitation to note in report:** Word2Vec works at the word level, not phrase level. For multi-word correct answers (e.g., "New York City"), you'll need to use the most distinctive word (e.g., "York") or try averaging word vectors.

**One-Hot Vocabulary + Co-occurrence approach (alternative to Word2Vec):**

Build a co-occurrence matrix: for every word pair (w1, w2) that appear within a window of 3 words in the same sentence, increment count[w1][w2]. This captures which words appear in similar contexts. Use cosine similarity on this matrix to find similar words as distractor candidates.

### 8.8 Evaluation for Model B Distractor Generation

**Automatic metrics:**

Precision: Of the top-3 ranked candidates your system selects, what fraction are actual RACE distractors? (Measured on test set where you know the true distractors)

Recall: Of the 3 actual RACE distractors, what fraction did your system recover?

F1: Harmonic mean of Precision and Recall.

Distractor Ranker Accuracy: For a given question, is at least one of your top-3 candidates a true RACE distractor?

Confusion Matrix: For the binary ranker (good distractor / not good distractor), plot the confusion matrix.

**Human evaluation form** `[Required for submission]`:

Create a Google Form or simple spreadsheet with 20 test examples. For each example, show the passage, question, correct answer, and your 3 generated distractors. Ask evaluators (classmates, instructor) to rate each distractor on a 1–5 Likert scale:
- 1 = obviously wrong / unrelated
- 3 = somewhat plausible
- 5 = highly plausible, could easily fool someone

Average these scores and report mean ± std.

---

## 9. Phase 7 — Model B: Hint Generator

**Deliverable:** Working hint generator producing 3 graduated hints per question.
**Marks at stake:** 10/100
**Time estimate:** Week 2, Day 4

### 9.1 What graduated hints mean

Hint 1: Most general — points the reader toward the topic area without giving the answer or even clearly indicating which part of the text to look at. E.g., "This question is about what happens early in the passage."

Hint 2: More specific — indicates which part of the passage is relevant, without giving away the answer. E.g., "Look at the second paragraph, which describes..."

Hint 3: Near-explicit — almost gives the answer. E.g., "The sentence starting with 'The main reason...' is the key to answering this question."

### 9.2 Extractive Hint Generation

**What extractive means:** Instead of generating new text, you extract existing sentences from the passage and present them as hints. This is simpler and more reliable than generating new text.

**Step 1 — Score each sentence:**

For each sentence in the article, compute a relevance score to the question. Two methods:

Method A — Bag-of-words overlap (classical ML, no neural network):
- Clean both sentence and question
- Compute word overlap: |words(sentence) ∩ words(question)| / |words(question)|
- This gives a simple relevance score

Method B — Sentence embedding cosine similarity `[BONUS — higher quality]`:
- Use `SentenceTransformer('all-MiniLM-L6-v2')` to encode each sentence and the question into 384-dimensional dense vectors
- Compute cosine similarity between each sentence vector and the question vector
- Higher similarity = more relevant sentence

**Step 2 — Rank sentences:**

Sort sentences by their relevance score (descending). The most relevant sentence is closest to the answer; the least relevant is the most general.

**Step 3 — Assign hints in REVERSE order:**

Hint 1 = the 3rd or 4th most relevant sentence (general, doesn't give it away)
Hint 2 = the 2nd most relevant sentence (more specific)
Hint 3 = the most relevant sentence (near-explicit)

This reverse assignment is key: you don't want Hint 1 to immediately reveal the answer.

### 9.3 ML-Scored Hint Extractor

**Going beyond simple word overlap:** Train a Logistic Regression to score sentences more accurately.

**Creating training labels:**

For each RACE training row, the "gold hint sentence" is the sentence in the article that has the highest word overlap with the correct answer (this is our proxy for "the sentence that contains the answer").

Label this sentence as 1 (good hint). Label all other sentences as 0.

**Features per sentence:**

1. keyword_overlap_with_question — fraction of question words appearing in sentence
2. keyword_overlap_with_answer — fraction of answer words in sentence
3. sentence_position_normalized — sentence index / total sentences (0.0 = first, 1.0 = last)
4. sentence_length — number of words
5. contains_named_entity_proxy — 1 if sentence contains a capitalized word (simple proxy)
6. contains_question_wh_word — 1 if sentence contains a word that matches the question type (e.g., if question asks "why", does sentence contain "because"?)
7. tf_score — average TF score of sentence words (optional enhancement)

**Training:**

1. For each training article, split into sentences, create one row per sentence
2. Label top-overlap sentence as 1, rest as 0
3. Compute features for all sentences
4. Train `LogisticRegression(class_weight='balanced')`
5. Save model

**Inference:**

Given a new article and question, split into sentences, compute features, run through trained LR, get probability scores, sort, assign as Hint 3 (highest) → Hint 2 → Hint 1 (lowest of top-3).

### 9.4 Hint Quality Evaluation

**R² Score (for regression variant) `[BONUS]`:**

Instead of binary labeling (0/1), create continuous relevance labels for each sentence: the cosine similarity between the sentence's word vector and the correct answer's word vector (a float between 0 and 1). Train a Ridge Regression instead of Logistic Regression to predict these continuous scores. Evaluate with R² (coefficient of determination): how well do predicted scores correlate with true scores?

**Precision@K:**

For each test article, your hint extractor produces a ranked list of sentences. Precision@K = fraction of the top-K ranked sentences that match the gold hint sentence (or are in the top-K true relevant sentences). Report Precision@1, Precision@3.

**Human evaluation:**

Same human evaluation form as distractors. Show 20 examples with all 3 hints. Ask evaluators: "Does this hint help without giving away the answer?" on a 1–5 scale.

---

## 10. Phase 8 — User Interface (All 4 Screens)

**Deliverable:** `ui/app.py` — a fully functional Streamlit application.
**Marks at stake:** 15/100
**Time estimate:** Week 2, Day 4

### 10.1 Why Streamlit

Streamlit is a Python library that turns a Python script into a web app. You write Python functions and add Streamlit widgets (buttons, text areas, dropdowns). Streamlit re-runs your script top-to-bottom every time the user interacts with anything. State between interactions is managed with `st.session_state`.

### 10.2 Overall App Structure

Use Streamlit's sidebar navigation to switch between screens. Define four pages:

- Sidebar radio button: "Article Input | Quiz | Hints | Dashboard"
- Use `st.session_state` to persist the article, generated question, answer, distractors, and hints across page transitions

### 10.3 Screen 1 — Article Input

**Purpose:** User enters or loads a passage and triggers both model pipelines.

**Elements required:**

1. App title and brief description at the top

2. A large text area: "Paste your reading passage here"

3. A "Load Random RACE Example" button — when clicked, load a random row from `test.csv`, populate the text area with its article, and also store the gold question/answer for comparison. This is critical for quick testing.

4. A "Submit" button — when clicked:
   - Show a spinner/loading indicator: `with st.spinner("Generating question and distractors...")`
   - Call `inference.run_pipeline(article)` from your inference module
   - Store results in `st.session_state`
   - Show a success message: "✅ Question generated! Switch to Quiz tab."

5. Error handling:
   - If text area is empty when Submit is clicked: `st.error("Please enter a passage before submitting.")`
   - If article is too short (< 50 words): `st.warning("Article seems very short. Results may be low quality.")`
   - If model loading fails: `st.error("Model files not found. Please run training scripts first.")`

6. Show a word count indicator below the text area (updates as user types, using `st.empty()`)

### 10.4 Screen 2 — Question & Answer Quiz View

**Purpose:** User answers the generated quiz question and receives feedback.

**Elements required:**

1. Display the generated question in a styled box (use `st.info()` or a markdown header)

2. Show four options as radio buttons. IMPORTANT: shuffle the order of correct answer and distractors randomly every time, so the correct answer isn't always in the same position. Use `random.shuffle()`.

3. A "Check Answer" button.

4. When Check is clicked:
   - Run Model A verifier on the selected option
   - If correct: `st.success("✅ Correct! [explanation from passage]")` — show the hint 3 sentence as explanation
   - If wrong: `st.error("❌ Incorrect. Try using hints.")` — show which option was correct with a brief explanation

5. Option to compare against the original RACE question (if loaded from dataset): a checkbox "Show RACE original question for comparison"

6. Track: number of attempts, was a hint used before answering? (Store in session_state for analytics)

### 10.5 Screen 3 — Hint Panel

**Purpose:** Progressive hints that guide without revealing the answer.

**Elements required:**

1. Three collapsible expanders (use `st.expander`):
   - "Hint 1 — General Clue" (starts closed)
   - "Hint 2 — More Specific" (starts closed)
   - "Hint 3 — Near Explicit" (starts closed)

2. Each expander, when opened, reveals the hint sentence and also highlights where in the article it comes from (show the sentence with bold formatting or a yellow background using markdown `**text**`).

3. A "Reveal Answer" button — should only appear AFTER all three expanders have been opened. Implement with session_state tracking: `if st.session_state.get('hints_revealed', 0) >= 3: st.button("Reveal Answer")`

4. When "Reveal Answer" is clicked: show the correct answer with a highlighted box and a brief explanation.

### 10.6 Screen 4 — Developer / Analytics Dashboard

**Purpose:** Show model performance metrics and session logs.

**Elements required:**

1. Model A Performance Section:
   - Display metrics (Accuracy, F1, Precision, Recall) as `st.metric()` widgets — these render as large highlighted numbers
   - Display confusion matrix as a matplotlib figure: `st.pyplot(fig)`
   - Source these from pre-computed results saved during training (load a JSON file with metrics)

2. Model B Performance Section:
   - Distractor ranker: Precision, Recall, F1, Accuracy as metric widgets
   - Hint extractor: Precision@3, R² score

3. Session Log Table:
   - Track each inference made in this session: timestamp, article length, generated question, selected answer, correct/wrong, hints used
   - Display as `st.dataframe()` with sortable columns
   - "Export to CSV" button: `st.download_button("Download session log", data=csv_string, file_name="session_log.csv")`

4. Inference Latency:
   - Track how long each Model A and Model B call takes (use Python's `time.time()`)
   - Display as a line chart: `st.line_chart()` showing latency over time

### 10.7 UX Requirements (required for full marks)

- **Loading indicators** on every model call — never let the UI freeze silently
- **Friendly error messages** for all failure states — no raw Python exceptions visible to user
- **Colour contrast** — avoid light grey text on white background. Use Streamlit's built-in `st.success`, `st.error`, `st.warning`, `st.info` for semantic colouring
- **Font size** — don't override defaults; Streamlit's defaults are accessible
- **Keyboard navigation** — Streamlit handles this automatically for standard widgets
- **Model transparency note** — add a small footnote on Screens 2 and 3: "⚠️ This question was AI-generated. Results may contain errors."
- **Usable without manual** — add a `st.sidebar.markdown("## How to use")` with 4 bullet instructions

### 10.8 The Inference Module (`src/inference.py`)

This is the glue code that wires Model A and Model B together. It should expose a single function: `run_pipeline(article_text) → dict`

The returned dict contains:
```python
{
  "question": "What did Sarah decide to do?",
  "correct_answer": "Buy groceries",
  "distractors": ["Meet a friend", "Exercise", "Go home"],
  "all_options_shuffled": ["Go home", "Buy groceries", "Meet a friend", "Exercise"],
  "correct_option_label": "B",  # after shuffling
  "hints": ["Hint 1 text", "Hint 2 text", "Hint 3 text"],
  "model_a_latency_ms": 47,
  "model_b_latency_ms": 31
}
```

**Model loading:** Load models ONCE at startup using `@st.cache_resource` decorator (Streamlit's caching mechanism). This prevents reloading the model on every user interaction.

---

## 11. Phase 9 — Evaluation, Tuning & Testing

**Deliverable:** Full evaluation on test set. Unit tests.
**Time estimate:** Week 2, Day 5

### 11.1 Full Evaluation on Test Set

Run EVERY model on the held-out test set (which you haven't touched until now). Record all metrics in `src/evaluate.py`.

For Model A:
- Verification Accuracy, Macro F1, Precision, Recall, Exact Match
- Confusion Matrix
- Per-question-type performance (breakdown by Who/What/Where/When/Why) — do some question types perform better?

For Model B Distractors:
- Precision, Recall, F1 for distractor ranker
- Distractor Accuracy (at least 1 of top-3 matches true distractor)
- Confusion Matrix for ranker

For Model B Hints:
- Precision@1, Precision@3
- R² score (if regression hint scorer implemented)

For Ensemble:
- Delta improvement over best individual model

### 11.2 Hyperparameter Sweep `[BONUS — extra thoroughness]`

Run a systematic hyperparameter search using GridSearchCV or RandomizedSearchCV.

For Logistic Regression: sweep C ∈ {0.001, 0.01, 0.1, 1, 10, 100} and solver ∈ {lbfgs, liblinear}

For SVM: sweep C ∈ {0.01, 0.1, 1, 10}

For Random Forest: sweep n_estimators ∈ {50, 100, 200} and max_depth ∈ {5, 10, None}

For vectorizer: sweep max_features ∈ {5000, 10000, 20000} and ngram_range ∈ {(1,1), (1,2)}

Use 3-fold cross-validation (not 5-fold, to save time) on the training split only.

Report: best hyperparameters found and the improvement over defaults.

### 11.3 Unit Tests (`tests/test_inference.py`)

Write at least 5 unit tests:

1. Test that `clean_text` correctly lowercases and removes punctuation
2. Test that the vectorizer transforms a known string to the expected shape
3. Test that `run_pipeline` returns a dict with all required keys
4. Test that distractors do not include the correct answer
5. Test that all 3 hints are non-empty strings
6. `[BONUS]` Test that inference completes within 10 seconds on a standard-length article

Run tests with `python -m pytest tests/` before final submission.

### 11.4 Performance Constraints Check

The project document requires:
- Inference for one article + question must complete in under 10 seconds
- Models must be trainable on RTX 3060 12GB or Google Colab T4

Profile your inference code: use Python's `time.time()` to measure how long each step takes. If it exceeds 10 seconds, identify the bottleneck (usually vectorizer transform on long articles) and optimize: truncate articles to first 500 words, reduce vocabulary size.

---

## 12. Phase 10 — Final Report

**Deliverable:** `report/final_report.pdf`
**Marks at stake:** 5/100 but required for submission

### 12.1 Report Structure (follow exactly)

The report must have exactly these 11 sections:

**1. Abstract (max 200 words)**
Summarize the entire project: problem, dataset, methods, key results, and conclusion. Write this LAST even though it appears first.

**2. Introduction & Motivation**
Explain why automated quiz generation matters (education technology context), what gaps exist, and what your system does. End with a paragraph previewing the rest of the report.

**3. Related Work (cite ≥ 5 papers)**

Must include all of these from the provided references:
- Lai et al. 2017 (RACE dataset)
- Du et al. 2017 (question generation)
- Zhao et al. 2018 (paragraph-level question generation)
- Guo et al. 2016 (distractor generation)
- Devlin et al. 2019 (BERT)

For each paper: 2–3 sentences on what they did, one sentence on how your work relates to or differs from theirs.

**4. Dataset Analysis**
Present your EDA findings with plots. Discuss class balance, article length distribution, question type distribution. Discuss the bias consideration: RACE comes from Chinese school exams — discuss how this might limit generalization to other demographics or text styles.

**5. Model A: Design, Training, Results**
Explain each model you trained, the features used, training procedure, and results table. Include confusion matrices. Explain why some models performed better than others.

**6. Model B: Design, Training, Results**
Same structure as Section 5 but for distractor generation and hint extraction. Include human evaluation results.

**7. User Interface Description**
Include screenshots of all 4 screens with numbered callouts explaining each element. Justify your choice of Streamlit (or other framework).

**8. Evaluation & Discussion**
Synthesize all results. Which component works best? Where does the system fail? How does it compare to naive baselines?

**9. Limitations & Future Work**
Be honest about what doesn't work well. Common limitations to discuss:
- One-Hot Encoding loses word order information — future work could use contextual embeddings
- Template-based question generation produces rigid/unnatural questions
- The distractor quality depends heavily on article vocabulary
- RACE dataset bias toward Chinese school exam topics

**10. Conclusion**
2 paragraphs: what you built, what you learned.

**11. References**
Use a consistent citation format (APA or IEEE). Include all papers cited in the report.

---

## 13. Phase 11 — Submission Checklist

Go through this list before submitting:

### Code & Repository
- [ ] GitHub repository shared with instructor
- [ ] Meaningful commit history (≥15 commits, meaningful messages)
- [ ] `requirements.txt` with pinned versions
- [ ] `README.md` with: project description, setup instructions, how to run training, how to launch UI
- [ ] All `.py` files have docstrings and inline comments
- [ ] No hardcoded file paths — use `os.path.join` or `pathlib.Path`
- [ ] No API keys or credentials committed to git

### Data & Models
- [ ] `data/raw/` contains train.csv, val.csv, test.csv (or instructions to download)
- [ ] `data/processed/` contains saved feature matrices
- [ ] All trained model `.pkl` files present in `models/`
- [ ] Vectorizer `.pkl` file present

### Notebooks
- [ ] `notebooks/EDA.ipynb` — runs top-to-bottom without errors, all plots rendered
- [ ] `notebooks/experiments.ipynb` — training experiments with results tables

### Application
- [ ] `ui/app.py` launches with `streamlit run ui/app.py`
- [ ] All 4 screens functional
- [ ] "Load Random RACE Example" button works
- [ ] Error handling for empty input
- [ ] Loading indicators present
- [ ] Analytics dashboard shows real metrics (not placeholder zeros)
- [ ] CSV export works

### Evaluation
- [ ] Full test set evaluation run
- [ ] All metrics reported (Accuracy, Macro F1, Precision, Recall, EM, Confusion Matrix)
- [ ] Human evaluation form completed with ≥10 evaluator responses
- [ ] Unit tests pass (`pytest tests/`)

### Report & Demo
- [ ] `report/final_report.pdf` — all 11 sections present, ≥5 references cited
- [ ] All plots readable (not cut off, axes labeled)
- [ ] Screenshots of all 4 UI screens in report
- [ ] 10-minute demo video recorded OR prepared for live presentation
- [ ] Ethical considerations section present in report

---

## 14. Concepts Glossary

Use this section as a quick reference when you encounter an unfamiliar term.

**Accuracy** — fraction of predictions that are correct. Simple but misleading when classes are imbalanced.

**Bag-of-Words** — represent text as counts (or 0/1) of word occurrences, ignoring word order.

**BIC/AIC** — metrics for selecting the number of components in GMM. Lower = better model. BIC penalizes complexity more than AIC.

**Binary Classification** — predicting one of two classes (0 or 1, correct/wrong).

**Class Imbalance** — when one class appears much more often than others. In Model A, 75% of examples are label=0 (wrong answer) and 25% are label=1 (correct answer). Use `class_weight='balanced'` to compensate.

**Confusion Matrix** — a 2×2 (for binary) or N×N table showing predictions vs. actual labels. Rows = actual class, Columns = predicted class. Diagonal = correct predictions.

**Cosine Similarity** — measure of angle between two vectors. 1 = same direction (identical texts), 0 = perpendicular (no overlap), -1 = opposite.

**Cross-Validation** — split training data into K folds, train on K-1 folds, evaluate on 1 fold, repeat K times. Gives more reliable performance estimate than single train/val split.

**Data Leakage** — accidentally including information from the test set in training, giving unrealistically good results. Never fit your vectorizer on val/test data.

**Dense Matrix** — a matrix where most values are non-zero. Handcrafted feature vectors are dense.

**EDA** — Exploratory Data Analysis. Looking at your data before modeling.

**Elbow Curve** — plot of K-Means inertia vs. K. The "elbow" point is where adding more clusters gives diminishing returns.

**Ensemble** — combining multiple models to get better predictions than any single model.

**Exact Match (EM)** — for multi-class prediction: 1 if the top-ranked option matches the gold answer label exactly, 0 otherwise.

**F1 Score** — harmonic mean of Precision and Recall. Good when you care about both. Macro F1 = average F1 across all classes (treats all classes equally regardless of size).

**Feature Engineering** — creating informative numerical representations from raw data.

**GMM (Gaussian Mixture Model)** — probabilistic clustering model. Assigns soft probabilities to cluster membership.

**Gradient Boosting (XGBoost)** — builds trees sequentially, each correcting previous errors.

**Hard Voting** — each model votes for a class, majority wins.

**Hyperparameter** — a setting you choose before training (e.g., C in SVM, n_estimators in RF). Not learned from data.

**Hyperparameter Tuning** — systematically trying different hyperparameter values to find the best ones.

**Inertia** — in K-Means, total sum of squared distances from each point to its cluster center. Lower = tighter clusters.

**joblib** — Python library for saving/loading Python objects (especially scikit-learn models) to disk.

**K-Means** — partitions data into K clusters by minimizing within-cluster variance.

**Label Propagation** — semi-supervised algorithm that spreads known labels to unlabeled data points via a similarity graph.

**Lexical Features** — features derived from text properties like word counts, overlap scores, and length ratios. Interpretable but less powerful than embeddings.

**Logistic Regression** — linear model for classification that outputs probabilities.

**Macro F1** — average F1 computed per class, then averaged. Treats all classes equally.

**max_features** — in CountVectorizer, the maximum vocabulary size (keep top N most frequent words).

**min_df** — in CountVectorizer, minimum document frequency. Words appearing in fewer than min_df documents are ignored.

**MMR (Maximal Marginal Relevance)** — algorithm to select items that are both relevant AND diverse.

**Multiclass Classification** — predicting one of N classes (N > 2). Question type classification is multiclass (Who/What/Where/When/Why/How/Other).

**Naive Bayes** — probabilistic classifier assuming feature independence. Fast and effective for text.

**n-gram** — a sequence of N consecutive words. "New York" is a bigram (n=2).

**One-Hot Encoding** — represent text as a binary vector where 1 = word present, 0 = word absent.

**Overfitting** — model performs well on training data but poorly on new data. Happens when model is too complex.

**PCA (Principal Component Analysis)** — reduces dimensionality by projecting onto directions of maximum variance.

**Pickle (.pkl)** — Python's native object serialization format. Used by joblib to save models.

**Precision** — of all things your model predicted as positive, what fraction were actually positive?

**R² Score** — coefficient of determination. How well does your model explain the variance in the true labels? 1.0 = perfect, 0.0 = no better than predicting the mean.

**Random Forest** — ensemble of decision trees, each trained on random data/feature subsets. Votes for final prediction.

**Recall** — of all actual positive things, what fraction did your model find?

**Regularization** — technique to prevent overfitting by penalizing model complexity. In LR/SVM, controlled by the C parameter (smaller C = stronger regularization).

**RMSE** — Root Mean Squared Error. For regression tasks. Lower is better.

**Semi-Supervised Learning** — uses both labeled and unlabeled data. Useful when labeling is expensive.

**sentence-transformers** — library for computing dense semantic embeddings of sentences. Used for hint ranking.

**Silhouette Score** — measures cluster quality. Range -1 to 1. Higher = better separation.

**Soft Voting** — each model outputs probabilities, which are averaged to get final prediction.

**Sparse Matrix** — a matrix where most values are 0. Scikit-learn returns sparse matrices from CountVectorizer/TfidfVectorizer to save memory.

**Stacking** — ensemble where a meta-classifier is trained on the predictions of base classifiers.

**Stopwords** — common words (the, is, a, of) that carry little semantic meaning.

**SVM (Support Vector Machine)** — classifier that finds the maximum-margin hyperplane between classes. LinearSVC is fast for high-dimensional text.

**TF-IDF** — Term Frequency-Inverse Document Frequency. Weights words by importance, not just presence.

**Underfitting** — model is too simple to capture the data's patterns. Low training AND validation accuracy.

**Unsupervised Learning** — finding patterns in data without labels.

**Word2Vec** — neural method that represents words as dense vectors where similar words are geometrically close.

---

## 15. Common Mistakes to Avoid

**Mistake 1 — Fitting vectorizer on all data:**
Only call `vectorizer.fit()` on training data. Then `vectorizer.transform()` on val and test separately. Fitting on val/test is data leakage.

**Mistake 2 — Not handling class imbalance:**
Your Model A labels are 25% positive, 75% negative. Without `class_weight='balanced'`, models tend to just predict the majority class. Always set this parameter.

**Mistake 3 — Evaluating on training data:**
Always evaluate on the validation set, never on training data. Training accuracy is meaningless — any model can memorize training data.

**Mistake 4 — Running out of memory:**
The full RACE dataset with OHE encoding can be gigabytes. Start experiments with a subset (10,000–20,000 rows). Only scale to full dataset once the pipeline works end-to-end.

**Mistake 5 — Not saving models:**
If you don't save trained models with joblib, you have to retrain from scratch every time you restart Python. Always save immediately after training.

**Mistake 6 — Distractors containing the correct answer:**
Always filter out any candidate that matches the correct answer string before returning distractors.

**Mistake 7 — Not shuffling answer options in UI:**
If the correct answer is always option A, users will notice. Shuffle options randomly at display time.

**Mistake 8 — Hard-coded file paths:**
Use `os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'train.csv')` instead of `'C:/Users/yourname/project/data/raw/train.csv'`. Hard-coded paths break on other machines.

**Mistake 9 — Committing model files to git:**
Large .pkl files (especially Word2Vec) can be hundreds of MB. Add `models/*.pkl` and `data/raw/*.csv` to `.gitignore`. Instead, add instructions in README for how to regenerate them.

**Mistake 10 — Leaving the test set contaminated:**
Run your final evaluation on test set ONCE, at the very end. If you tune hyperparameters based on test set performance, your reported results are inflated.

**Mistake 11 — Not testing the full pipeline end-to-end early:**
Write a minimal version of inference.py (even with dummy models) early in Week 1. This lets you build the UI early and catch integration bugs before the final days.

---

*End of Execution Plan — Race RC Project 2026 | NUCES FAST Islamabad*
