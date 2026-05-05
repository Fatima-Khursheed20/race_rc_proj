# Phase 4: Model A Unsupervised & Semi-Supervised — Detailed Implementation Guide

**Marks at stake:** 20/100 — **HIGHEST SINGLE COMPONENT**  
**Time estimate:** 3-4 days (K-Means, Label Propagation, GMM experiments)  
**Deliverable:** Clustering results, semi-supervised comparison table in `notebooks/experiments.ipynb`

---

## Table of Contents

1. [Why This Phase Worth 20 Marks](#1-why-this-phase-worth-20-marks)
2. [Unsupervised vs Semi-Supervised Overview](#2-unsupervised-vs-semi-supervised-overview)
3. [Dimensionality Reduction with PCA](#3-dimensionality-reduction-with-pca)
4. [K-Means Clustering](#4-k-means-clustering)
5. [Label Propagation (Semi-Supervised)](#5-label-propagation-semi-supervised)
6. [Gaussian Mixture Models](#6-gaussian-mixture-models)
7. [Evaluation Metrics & Clustering Quality](#7-evaluation-metrics--clustering-quality)
8. [Comparison & Results Table](#8-comparison--results-table)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Troubleshooting & Performance](#10-troubleshooting--performance)

---

## 1. Why This Phase Worth 20 Marks

### 1.1 Project Intent

This phase tests your understanding of **learning WITHOUT labels**. In real-world scenarios:

- You have millions of unlabeled articles (cheap to collect)
- You have only hundreds of labeled questions (expensive to annotate)
- Can you extract structure from unlabeled data to improve predictions?

This is **not busywork**. It's teaching you:
1. When to use unsupervised learning (unlabeled data abundance)
2. When to use semi-supervised learning (mixed labeled + unlabeled)
3. How unsupervised insights inform supervised models
4. Evaluation metrics beyond accuracy (silhouette score, purity, information-theoretic measures)

### 1.2 Grading Breakdown

```
Phase 4: Unsupervised & Semi-Supervised = 20/100 marks

  ├─ K-Means clustering: 5 marks
  │   (Silhouette score, purity, elbow analysis, interpretability)
  │
  ├─ Label Propagation: 5 marks
  │   (Semi-supervised performance vs supervised baselines)
  │
  ├─ Gaussian Mixture Models: 5 marks
  │   (AIC/BIC analysis, soft probabilities, comparison to K-Means)
  │
  └─ Overall comparison & insights: 5 marks
      (Quality of analysis, documentation, visualization, conclusions)
```

**Missing any component** → loses 5 marks. All three (K-Means, Label Prop, GMM) **required** for full marks.

### 1.3 Integration with Phase 3

Phase 3 gave you: LR (62% accuracy), SVM (64% accuracy) on **supervised** answers.

Phase 4 asks: **Without labels, can unsupervised methods find the same structure?**

If K-Means discovers clusters that naturally correspond to answer positions (A/B/C/D) or question difficulty, that validates your features are discriminative. If not, it suggests you need better features or ensemble strategies.

---

## 2. Unsupervised vs Semi-Supervised Overview

### 2.1 Conceptual Difference

```
SUPERVISED (Phase 3):
  Input: (features, label)  for every example
  Task:  Learn decision boundary using all labels
  Example: LR trained on 281k labeled examples → 62% accuracy
  
UNSUPERVISED (Phase 4a - K-Means, GMM):
  Input: (features) only — NO labels
  Task:  Find natural groupings / clusters
  Example: K-Means on same 281k examples (ignoring labels) → Do clusters match A/B/C/D?
  Evaluation: Silhouette score, purity (if ground truth available)
  
SEMI-SUPERVISED (Phase 4b - Label Propagation):
  Input: (features, label) for 10% of examples + (features) for 90% unlabeled
  Task:  Learn from both labeled and unlabeled examples
  Example: LP trained on 28k labeled + 253k unlabeled → beats supervised with 28k labels alone
  Evaluation: F1 score (compare: LP with 28k labels vs supervised with 28k labels)
```

### 2.2 Why Unsupervised First?

```
Flow:
  1. Train supervised (LR/SVM) → baseline metrics
  2. Run unsupervised (K-Means) → discover structure without labels
  3. Analyze: Do clusters align with answer positions?
           Do clusters separate easy vs hard questions?
  4. Run semi-supervised (LP) → leverage unlabeled data to improve predictions
  5. Compare: Semi-supervised >> supervised with few labels?
```

---

## 3. Dimensionality Reduction with PCA

### 3.1 Why PCA?

**The problem:** Your features have 10,023 dimensions (10k OHE + 23 lexical).

**Why that's bad for unsupervised learning:**
- K-Means becomes inaccurate in high dimensions ("curse of dimensionality")
- Distance metrics break down (all points roughly equidistant from centroid)
- Training is slow: O(n × d × k × iterations) where d=10,023
- Visualization impossible (can't plot in 10k dimensions)

**PCA solution:** Reduce to 50-100 dimensions while preserving 85%+ variance.

### 3.2 PCA Implementation

```python
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def apply_pca(X_train, X_dev, X_test, n_components=50):
    """Apply PCA to reduce dimensionality."""
    
    # Initialize PCA
    pca = PCA(n_components=n_components, random_state=42)
    
    # Fit on training data only (prevent data leakage)
    X_train_pca = pca.fit_transform(X_train)
    X_dev_pca = pca.transform(X_dev)
    X_test_pca = pca.transform(X_test)
    
    # Analyze explained variance
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var_ratio = np.cumsum(explained_var_ratio)
    
    print(f"\n{'='*70}")
    print(f"PCA Dimensionality Reduction")
    print(f"{'='*70}")
    print(f"Original dimensions:        {X_train.shape[1]:>6}")
    print(f"Reduced dimensions:         {n_components:>6}")
    print(f"Explained variance ratio:   {explained_var_ratio.sum():.4f} ({explained_var_ratio.sum()*100:.2f}%)")
    print(f"Cumulative variance (top10):")
    for i in range(min(10, len(cumulative_var_ratio))):
        print(f"  PC{i+1}: {cumulative_var_ratio[i]:.4f} ({cumulative_var_ratio[i]*100:.2f}%)")
    print(f"{'='*70}\n")
    
    # Plot explained variance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, min(51, len(explained_var_ratio)+1)), 
            explained_var_ratio[:50])
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Individual Explained Variance by PC')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_var_ratio)+1), cumulative_var_ratio, 'b-', linewidth=2)
    plt.axhline(y=0.85, color='r', linestyle='--', label='85% threshold')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return X_train_pca, X_dev_pca, X_test_pca, pca

# Usage
X_train_pca, X_dev_pca, X_test_pca, pca = apply_pca(
    X_train_combined,  # Combined sparse OHE + lexical
    X_dev_combined,
    X_test_combined,
    n_components=50
)

print(f"✓ PCA applied: {X_train_pca.shape}")
# Output: (281168, 50)
```

### 3.3 PCA Interpretation

**What you'll see:**

```
PCA Dimensionality Reduction
======================================================================
Original dimensions:        10023
Reduced dimensions:           50
Explained variance ratio:   0.8542 (85.42%)
Cumulative variance (top10):
  PC1: 0.1523 (15.23%)    ← First component captures 15% of variance
  PC2: 0.2145 (21.45%)    ← First 2 PCs capture 21.45% total
  PC3: 0.2631 (26.31%)
  ...
  PC50: 0.8542 (85.42%)   ← All 50 PCs capture 85.42% variance
======================================================================
```

**Rule of thumb:** Use 50-100 components to capture 85-95% variance. For clustering, 50 is usually sufficient.

---

## 4. K-Means Clustering

### 4.1 What K-Means Does

```
Algorithm (simplified):
  1. Pick K random points as initial centroids (cluster centers)
  2. Assign each data point to nearest centroid (Euclidean distance)
  3. Recompute centroids as mean of assigned points
  4. Repeat steps 2-3 until stable (centroids don't move)

Goal: Minimize within-cluster sum of squares (WCSS)
      = sum of squared distances from each point to its centroid
```

**For your question data:**
- K=4 likely (matching 4 answer positions A/B/C/D)
- Also test K=2, 6, 8 (see if other structures emerge)

### 4.2 Elbow Method

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def find_optimal_k(X_pca, k_range=range(2, 11), plot=True):
    """
    Elbow method: find optimal K by analyzing inertia curve.
    Inertia = within-cluster sum of squares (lower is better)
    Elbow = point where inertia stops decreasing significantly
    """
    
    inertias = []
    silhouette_scores = []
    
    print(f"\n{'='*70}")
    print(f"K-Means: Finding Optimal K (Elbow Method)")
    print(f"{'='*70}\n")
    print(f"{'K':<5} {'Inertia':<15} {'Silhouette Score':<20}")
    print(f"{'-'*40}")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, 
                       random_state=42, n_jobs=-1, verbose=0)
        kmeans.fit(X_pca)
        
        inertia = kmeans.inertia_
        inertias.append(inertia)
        
        # Silhouette score (ranges -1 to 1, higher is better)
        from sklearn.metrics import silhouette_score
        sil_score = silhouette_score(X_pca, kmeans.labels_)
        silhouette_scores.append(sil_score)
        
        print(f"{k:<5} {inertia:<15.2f} {sil_score:<20.4f}")
    
    print(f"{'='*70}\n")
    
    # Plot
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Inertia curve (elbow)
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (K)')
        ax1.set_ylabel('Inertia (WCSS)')
        ax1.set_title('Elbow Method: Find Optimal K')
        ax1.grid(True, alpha=0.3)
        for i, (k, inertia) in enumerate(zip(k_range, inertias)):
            ax1.annotate(f'{inertia:.0f}', (k, inertia), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        # Silhouette score curve
        ax2.plot(k_range, silhouette_scores, 'rs-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (K)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs K')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        for i, (k, sil) in enumerate(zip(k_range, silhouette_scores)):
            ax2.annotate(f'{sil:.3f}', (k, sil), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.show()
    
    return inertias, silhouette_scores

# Usage
inertias, silhouette_scores = find_optimal_k(X_train_pca, k_range=range(2, 11))
```

**Sample output:**

```
K-Means: Finding Optimal K (Elbow Method)
======================================================================

K     Inertia         Silhouette Score
----------------------------------------
2     4521340.25      0.3421
3     3894521.12      0.3156
4     3456789.45      0.2987
5     3124567.89      0.2743
6     2987654.32      0.2456
7     2876543.21      0.2134
8     2765432.10      0.1923

======================================================================

Observation: Inertia decreases, but rate of decrease slows after K=4
             Silhouette score peaks at K=2 (0.3421)
             For 4 answer positions, K=4 is reasonable
```

### 4.3 Training K-Means

```python
def train_kmeans(X_train_pca, k=4, verbose=True):
    """Train K-Means with specified K."""
    
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',           # Smart initialization (better than random)
        n_init=10,                  # Try 10 random initializations, pick best
        max_iter=300,
        tol=1e-4,
        random_state=42,
        n_jobs=-1,
        verbose=1 if verbose else 0
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training K-Means (K={k})")
        print(f"{'='*70}")
        print(f"Features: {X_train_pca.shape[1]} (PCA-reduced)")
        print(f"Samples: {X_train_pca.shape[0]:,}")
        print(f"Clusters: {k}")
        print(f"{'='*70}\n")
    
    import time
    start_time = time.time()
    kmeans.fit(X_train_pca)
    training_time = time.time() - start_time
    
    if verbose:
        print(f"✓ Training completed in {training_time:.2f} seconds")
        print(f"  Inertia: {kmeans.inertia_:.2f}")
        print(f"  Cluster sizes: {np.bincount(kmeans.labels_)}")
    
    return kmeans, training_time

# Train K-Means with K=4 (answer positions)
kmeans_k4, time_k4 = train_kmeans(X_train_pca, k=4, verbose=True)
```

### 4.4 Clustering Quality Metrics

```python
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import matplotlib.pyplot as plt

def evaluate_kmeans(X_pca, kmeans, y_true=None, model_name="K-Means"):
    """Evaluate K-Means clustering quality."""
    
    labels_pred = kmeans.labels_
    
    print(f"\n{'='*70}")
    print(f"Clustering Quality Evaluation: {model_name}")
    print(f"{'='*70}\n")
    
    # 1. Silhouette Score (overall)
    sil_score = silhouette_score(X_pca, labels_pred)
    print(f"Silhouette Score (overall):  {sil_score:.4f}")
    print(f"  Range: [-1, 1]")
    print(f"  Interpretation:")
    print(f"    > 0.5   : Strong structure")
    print(f"    0.3-0.5 : Reasonable structure")
    print(f"    < 0.3   : Weak structure\n")
    
    # 2. Silhouette Score (per sample)
    sil_samples = silhouette_samples(X_pca, labels_pred)
    
    # 3. Cluster sizes
    unique_clusters, cluster_sizes = np.unique(labels_pred, return_counts=True)
    print(f"Cluster sizes:")
    for cluster_id, size in zip(unique_clusters, cluster_sizes):
        pct = (size / len(labels_pred)) * 100
        print(f"  Cluster {cluster_id}: {size:>7,} ({pct:>5.2f}%)")
    print()
    
    # 4. Purity (if true labels available)
    if y_true is not None:
        purity = compute_purity(y_true, labels_pred)
        print(f"Purity: {purity:.4f}")
        print(f"  (Fraction of points assigned to majority true label in their cluster)")
        print()
    
    # 5. Silhouette plot
    k = len(unique_clusters)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    y_lower = 10
    for i in range(k):
        cluster_sil_values = sil_samples[labels_pred == i]
        cluster_sil_values.sort()
        
        size_cluster_i = cluster_sil_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_sil_values,
                         facecolor=color, edgecolor=color, alpha=0.7,
                         label=f'Cluster {i}')
        y_lower = y_upper + 10
    
    ax.axvline(x=sil_score, color="red", linestyle="--", label=f"Mean: {sil_score:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster Label")
    ax.set_title(f"Silhouette Plot: {model_name}")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
    
    print(f"{'='*70}\n")
    
    return sil_score

# Usage
kmeans_sil_score = evaluate_kmeans(X_train_pca, kmeans_k4, y_true=y_train, 
                                    model_name="K-Means K=4")
```

### 4.5 Does K-Means Discover Answer Positions?

```python
def analyze_cluster_label_correspondence(y_true, y_pred, row_ids):
    """
    Check if clusters correspond to answer positions (A/B/C/D).
    Hypothesis: If features are discriminative, K-Means should separate
    correct (label=1) from incorrect (label=0) answers.
    """
    
    print(f"\n{'='*70}")
    print(f"Cluster ↔ Label Analysis")
    print(f"{'='*70}\n")
    
    # Reshape by original question
    n_questions = len(np.unique(row_ids))
    y_true_reshaped = y_true.reshape(n_questions, 4)
    y_pred_reshaped = y_pred.reshape(n_questions, 4)
    
    # For each question, check if the correct answer clusters differently
    correct_cluster_ids = []
    incorrect_cluster_ids = []
    
    for q_idx in range(n_questions):
        true_labels = y_true_reshaped[q_idx]  # Which option is correct (0,0,0,1)
        pred_clusters = y_pred_reshaped[q_idx]
        
        correct_pos = np.argmax(true_labels)  # Position of correct answer (0-3)
        correct_cluster = pred_clusters[correct_pos]
        
        for pos in range(4):
            if pos == correct_pos:
                correct_cluster_ids.append(correct_cluster)
            else:
                incorrect_cluster_ids.append(pred_clusters[pos])
    
    # Distribution
    print("Cluster distribution for CORRECT answers (should be biased to 1-2 clusters):")
    unique, counts = np.unique(correct_cluster_ids, return_counts=True)
    for cluster_id, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
        pct = (count / len(correct_cluster_ids)) * 100
        print(f"  Cluster {cluster_id}: {count:>6,} ({pct:>5.2f}%)")
    
    print("\nCluster distribution for INCORRECT answers:")
    unique, counts = np.unique(incorrect_cluster_ids, return_counts=True)
    for cluster_id, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
        pct = (count / len(incorrect_cluster_ids)) * 100
        print(f"  Cluster {cluster_id}: {count:>6,} ({pct:>5.2f}%)")
    
    # Chi-square test for independence
    from scipy.stats import chi2_contingency
    
    # Contingency table
    contingency = np.zeros((2, 4))  # 2 labels (correct/incorrect) × 4 clusters
    for cluster_id in range(4):
        contingency[1, cluster_id] = np.sum(np.array(correct_cluster_ids) == cluster_id)
        contingency[0, cluster_id] = np.sum(np.array(incorrect_cluster_ids) == cluster_id)
    
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-square test: χ² = {chi2:.2f}, p-value = {p_value:.2e}")
    print(f"Interpretation: p < 0.05 → clusters are significantly associated with labels")
    print(f"{'='*70}\n")

# Usage
analyze_cluster_label_correspondence(y_train, kmeans_k4.labels_, row_ids_train)
```

---

## 5. Label Propagation (Semi-Supervised)

### 5.1 The Problem Setting

```
Standard supervised learning (Phase 3):
  Train: 281,168 labeled examples (all have correct/incorrect label)
  Dev:   35,144 labeled examples
  
Real-world scenario (what LP addresses):
  Train: 28,117 labeled examples (10% of original 281k)
         253,051 unlabeled examples (90% of original 281k)
  Dev:   35,144 labeled examples
  
Question: Can we learn better from 28k labeled + 253k unlabeled
          than from just 28k labeled?
```

### 5.2 How Label Propagation Works

```
Intuition: Labels "spread" from labeled to unlabeled examples

1. Build similarity graph:
   - Each point is a node
   - Connect each point to its K nearest neighbors
   - Edge weights = similarity between neighbors

2. Initialize labels:
   - Labeled examples: known labels
   - Unlabeled examples: unknown (initialize probabilistically)

3. Propagation:
   - Update label probabilities:
     P(y=1 | unlabeled point) = weighted average of neighbors' labels
   - Repeat until convergence

Result: Unlabeled examples get predicted labels based on labeled neighbors
```

### 5.3 Implementation

```python
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import numpy as np

def create_partial_labels(y_train, labeled_fraction=0.1):
    """
    Create partially labeled dataset.
    Only 10% of training examples have labels, 90% are unlabeled (-1).
    """
    
    n_samples = len(y_train)
    n_labeled = int(n_samples * labeled_fraction)
    
    # Randomly select which examples to keep labels for
    labeled_indices = np.random.choice(n_samples, n_labeled, replace=False)
    
    # Create partial label array
    y_partial = np.full(n_samples, -1, dtype=np.int32)  # -1 = unlabeled
    y_partial[labeled_indices] = y_train[labeled_indices]
    
    print(f"\n{'='*70}")
    print(f"Creating Partially Labeled Dataset")
    print(f"{'='*70}")
    print(f"Total examples: {n_samples:,}")
    print(f"Labeled:       {n_labeled:,} ({labeled_fraction*100:.1f}%)")
    print(f"Unlabeled:     {n_samples - n_labeled:,} ({(1-labeled_fraction)*100:.1f}%)")
    print(f"Label distribution (labeled subset):")
    unique, counts = np.unique(y_train[labeled_indices], return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count:>6,} ({(count/n_labeled)*100:>5.2f}%)")
    print(f"{'='*70}\n")
    
    return y_partial, labeled_indices

def train_label_propagation(X_train_pca, y_partial, kernel='knn', n_neighbors=7):
    """Train Label Propagation on partially labeled data."""
    
    lp = LabelPropagation(
        kernel=kernel,
        n_neighbors=n_neighbors,
        max_iter=1000,
        tol=1e-3
    )
    
    print(f"\n{'='*70}")
    print(f"Training Label Propagation")
    print(f"{'='*70}")
    print(f"Kernel: {kernel}")
    print(f"Neighbors (K): {n_neighbors}")
    print(f"Features: {X_train_pca.shape[1]}")
    print(f"Samples: {X_train_pca.shape[0]:,}")
    print(f"{'='*70}\n")
    
    import time
    start_time = time.time()
    lp.fit(X_train_pca, y_partial)
    training_time = time.time() - start_time
    
    print(f"✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return lp, training_time

# Create partially labeled training data
y_partial, labeled_indices = create_partial_labels(y_train, labeled_fraction=0.1)

# Train Label Propagation
lp_model, lp_time = train_label_propagation(
    X_train_pca, y_partial, 
    kernel='knn', 
    n_neighbors=7
)

# Predict on full training set
y_train_pred_lp = lp_model.predict(X_train_pca)

# Predict on dev set (using full trained model)
y_dev_pred_lp = lp_model.predict(X_dev_pca)
```

### 5.4 Hyperparameter Tuning for LP

```python
def tune_label_propagation(X_train_pca, y_partial, X_dev_pca, y_dev):
    """
    Tune Label Propagation hyperparameters.
    Key parameter: n_neighbors (K in KNN graph)
    """
    
    from sklearn.metrics import f1_score
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"Tuning Label Propagation: n_neighbors (K)")
    print(f"{'='*70}\n")
    print(f"{'K':<5} {'F1 Score (train)':<20} {'F1 Score (dev)':<20} {'Time (sec)':<15}")
    print(f"{'-'*60}")
    
    for k_neighbors in [3, 5, 7, 10, 15, 20]:
        lp = LabelPropagation(kernel='knn', n_neighbors=k_neighbors, max_iter=1000)
        
        import time
        start = time.time()
        lp.fit(X_train_pca, y_partial)
        fit_time = time.time() - start
        
        # Predictions
        y_train_pred = lp.predict(X_train_pca)
        y_dev_pred = lp.predict(X_dev_pca)
        
        # F1 scores
        f1_train = f1_score(y_train, y_train_pred, average='macro')
        f1_dev = f1_score(y_dev, y_dev_pred, average='macro')
        
        results.append({
            'k_neighbors': k_neighbors,
            'f1_train': f1_train,
            'f1_dev': f1_dev,
            'time': fit_time,
            'model': lp
        })
        
        print(f"{k_neighbors:<5} {f1_train:<20.4f} {f1_dev:<20.4f} {fit_time:<15.2f}")
    
    # Find best by dev F1
    best_result = max(results, key=lambda x: x['f1_dev'])
    print(f"\n{'='*70}")
    print(f"Best K: {best_result['k_neighbors']} (Dev F1: {best_result['f1_dev']:.4f})")
    print(f"{'='*70}\n")
    
    return results, best_result

# Tune LP
lp_results, lp_best = tune_label_propagation(
    X_train_pca, y_partial, 
    X_dev_pca, y_dev
)
```

### 5.5 Semi-Supervised vs Supervised Comparison

```python
def compare_supervised_vs_semisupervised(y_dev, 
                                         y_dev_pred_supervised_small,
                                         y_dev_pred_lp):
    """
    Compare semi-supervised LP (trained on 10% labeled + 90% unlabeled)
    vs supervised on just 10% labeled data.
    """
    
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    
    print(f"\n{'='*70}")
    print(f"Semi-Supervised vs Supervised Comparison")
    print(f"{'='*70}")
    print(f"Supervised trained on: 10% labeled ({int(0.1*len(y_dev)):,} examples)")
    print(f"LP trained on: 10% labeled + 90% unlabeled ({len(y_dev):,} total)\n")
    
    metrics = {
        'Accuracy': accuracy_score,
        'Macro F1': lambda yt, yp: f1_score(yt, yp, average='macro'),
        'Precision': lambda yt, yp: precision_score(yt, yp),
        'Recall': lambda yt, yp: recall_score(yt, yp),
    }
    
    print(f"{'Metric':<20} {'Supervised (10%)':<25} {'Label Prop':<25} {'Improvement':<15}")
    print(f"{'-'*85}")
    
    for metric_name, metric_func in metrics.items():
        sup_score = metric_func(y_dev, y_dev_pred_supervised_small)
        lp_score = metric_func(y_dev, y_dev_pred_lp)
        improvement = ((lp_score - sup_score) / sup_score) * 100
        
        print(f"{metric_name:<20} {sup_score:<25.4f} {lp_score:<25.4f} {improvement:>+6.2f}%")
    
    print(f"{'='*70}\n")
    
    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    metrics_names = list(metrics.keys())
    supervised_scores = [metrics[m](y_dev, y_dev_pred_supervised_small) for m in metrics_names]
    lp_scores = [metrics[m](y_dev, y_dev_pred_lp) for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, supervised_scores, width, label='Supervised (10%)', alpha=0.8)
    bars2 = ax.bar(x + width/2, lp_scores, width, label='Label Propagation', alpha=0.8)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Semi-Supervised (LP) vs Supervised (10% labeled)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.4, 0.7])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
```

---

## 6. Gaussian Mixture Models

### 6.1 Why GMM vs K-Means?

```
K-Means: Hard assignment
  Each point → exactly 1 cluster
  Decision: point belongs to cluster 3 (probability 0 or 1)

GMM: Soft assignment
  Each point → probability distribution over clusters
  Decision: point is 40% cluster 1, 35% cluster 2, 25% cluster 3
  
Why soft better?:
  - Real data has ambiguous points (near decision boundary)
  - GMM captures this uncertainty explicitly
  - Can use soft probabilities for downstream tasks
```

### 6.2 GMM Implementation

```python
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np

def train_gmm(X_train_pca, n_components=4, covariance_type='full'):
    """Train Gaussian Mixture Model."""
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,  # 'full', 'tied', 'diag', 'spherical'
        max_iter=200,
        n_init=10,
        random_state=42,
        verbose=1
    )
    
    print(f"\n{'='*70}")
    print(f"Training Gaussian Mixture Model")
    print(f"{'='*70}")
    print(f"Components: {n_components}")
    print(f"Covariance type: {covariance_type}")
    print(f"Samples: {X_train_pca.shape[0]:,}")
    print(f"Features: {X_train_pca.shape[1]}")
    print(f"{'='*70}\n")
    
    import time
    start_time = time.time()
    gmm.fit(X_train_pca)
    training_time = time.time() - start_time
    
    print(f"✓ Training completed in {training_time:.2f} seconds\n")
    
    return gmm, training_time

def evaluate_gmm(gmm, X_pca, y_true=None, model_name="GMM"):
    """Evaluate GMM with AIC/BIC and clustering metrics."""
    
    # Get cluster assignments (hard)
    labels = gmm.predict(X_pca)
    
    # Get probabilities (soft)
    proba = gmm.predict_proba(X_pca)
    
    print(f"\n{'='*70}")
    print(f"GMM Evaluation: {model_name}")
    print(f"{'='*70}\n")
    
    # 1. AIC and BIC (lower is better)
    aic = gmm.aic(X_pca)
    bic = gmm.bic(X_pca)
    
    print(f"AIC (Akaike Information Criterion):  {aic:.2f}")
    print(f"  Balances model fit with complexity")
    print(f"  Lower = better fit AND simpler model\n")
    
    print(f"BIC (Bayesian Information Criterion): {bic:.2f}")
    print(f"  Penalizes complexity more than AIC")
    print(f"  Lower = better\n")
    
    # 2. Log-likelihood
    log_likelihood = gmm.score(X_pca)
    print(f"Log-likelihood per sample:          {log_likelihood:.4f}")
    print(f"  Higher = better fit\n")
    
    # 3. Silhouette score (same as K-Means)
    sil_score = silhouette_score(X_pca, labels)
    print(f"Silhouette Score (hard assignments): {sil_score:.4f}\n")
    
    # 4. Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Cluster sizes (hard assignments):")
    for cluster_id, count in zip(unique, counts):
        pct = (count / len(labels)) * 100
        print(f"  Cluster {cluster_id}: {count:>7,} ({pct:>5.2f}%)")
    print()
    
    # 5. Posterior entropy (measure of uncertainty)
    # Entropy close to log(K) = maximum uncertainty (uniform distribution)
    # Entropy close to 0 = maximum confidence (one cluster dominates)
    entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
    mean_entropy = entropy.mean()
    max_entropy = np.log(gmm.n_components)
    
    print(f"Posterior Entropy (uncertainty measure):")
    print(f"  Mean: {mean_entropy:.4f}")
    print(f"  Max possible (uniform): {max_entropy:.4f}")
    print(f"  Ratio: {mean_entropy/max_entropy:.4f}")
    print(f"  (Lower = more confident assignments)\n")
    
    # 6. Purity (if true labels available)
    if y_true is not None:
        purity = compute_purity(y_true, labels)
        print(f"Purity: {purity:.4f}\n")
    
    print(f"{'='*70}\n")
    
    return {
        'aic': aic,
        'bic': bic,
        'log_likelihood': log_likelihood,
        'silhouette': sil_score,
        'entropy': mean_entropy,
        'labels': labels,
        'proba': proba
    }

# Train GMM with K=4
gmm_k4, gmm_time = train_gmm(X_train_pca, n_components=4, covariance_type='full')

# Evaluate
gmm_results = evaluate_gmm(gmm_k4, X_train_pca, y_true=y_train, model_name="GMM K=4")
```

### 6.3 AIC/BIC Model Selection

```python
def model_selection_with_aic_bic(X_train_pca, k_range=range(2, 11)):
    """Use AIC/BIC to find optimal number of components."""
    
    aics = []
    bics = []
    
    print(f"\n{'='*70}")
    print(f"GMM Model Selection: AIC/BIC Analysis")
    print(f"{'='*70}\n")
    print(f"{'K':<5} {'AIC':<20} {'BIC':<20}")
    print(f"{'-'*45}")
    
    for k in k_range:
        gmm = GaussianMixture(n_components=k, n_init=10, random_state=42, verbose=0)
        gmm.fit(X_train_pca)
        
        aic = gmm.aic(X_train_pca)
        bic = gmm.bic(X_train_pca)
        
        aics.append(aic)
        bics.append(bic)
        
        print(f"{k:<5} {aic:<20.2f} {bic:<20.2f}")
    
    print(f"{'='*70}\n")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # AIC curve
    ax1.plot(k_range, aics, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Components (K)')
    ax1.set_ylabel('AIC')
    ax1.set_title('AIC vs K (lower is better)')
    ax1.grid(True, alpha=0.3)
    best_k_aic = k_range[np.argmin(aics)]
    ax1.axvline(x=best_k_aic, color='r', linestyle='--', alpha=0.5, label=f'Best K={best_k_aic}')
    ax1.legend()
    
    # BIC curve
    ax2.plot(k_range, bics, 'rs-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Components (K)')
    ax2.set_ylabel('BIC')
    ax2.set_title('BIC vs K (lower is better)')
    ax2.grid(True, alpha=0.3)
    best_k_bic = k_range[np.argmin(bics)]
    ax2.axvline(x=best_k_bic, color='g', linestyle='--', alpha=0.5, label=f'Best K={best_k_bic}')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return aics, bics

# Run model selection
aics, bics = model_selection_with_aic_bic(X_train_pca, k_range=range(2, 11))
```

---

## 7. Evaluation Metrics & Clustering Quality

### 7.1 Silhouette Score

```python
def silhouette_analysis(X_pca, labels, model_name="Model"):
    """
    Silhouette coefficient: measures how similar a point is to its own cluster
    vs other clusters. Range [-1, 1]:
      +1: point is well-matched to its cluster, far from other clusters
       0: point is on the border between clusters
      -1: point is assigned to wrong cluster
    """
    
    from sklearn.metrics import silhouette_samples, silhouette_score
    import matplotlib.cm as cm
    
    silhouette_avg = silhouette_score(X_pca, labels)
    sample_silhouette_values = silhouette_samples(X_pca, labels)
    
    print(f"Average Silhouette Score: {silhouette_avg:.4f}")
    
    # Interpretation
    if silhouette_avg > 0.5:
        print("  Strong cluster structure")
    elif silhouette_avg > 0.3:
        print("  Reasonable cluster structure")
    else:
        print("  Weak cluster structure")
```

### 7.2 Purity

```python
def compute_purity(y_true, y_pred):
    """
    Purity: For each cluster, find the dominant true label.
    Purity = (sum of max counts per cluster) / total points
    
    Measures: do clusters contain mostly one true class?
    Range: [0, 1], higher is better
    """
    
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import contingency_matrix
    
    # Contingency matrix: rows=true labels, cols=pred clusters
    contingency = contingency_matrix(y_true, y_pred)
    
    # Find best matching between clusters and labels
    row_ind, col_ind = linear_sum_assignment(-contingency)
    
    # Purity
    purity = contingency[row_ind, col_ind].sum() / len(y_true)
    
    return purity
```

### 7.3 Normalized Mutual Information (NMI)

```python
def compute_nmi(y_true, y_pred):
    """
    Normalized Mutual Information: measures agreement between
    true labels and predicted clusters.
    Range: [0, 1], higher is better
    1 = perfect clustering (clusters match true labels)
    0 = random clustering
    """
    
    from sklearn.metrics import normalized_mutual_info_score
    
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return nmi
```

---

## 8. Comparison & Results Table

### 8.1 Building the Master Comparison Table

```python
def create_phase4_comparison_table(results_dict):
    """
    Create comprehensive Phase 4 results table.
    
    Rows: K-Means K=4, K-Means K=2,6,8, Label Prop, GMM, etc.
    Cols: Silhouette, Purity, NMI, Inertia/AIC/BIC, Time
    """
    
    import pandas as pd
    
    comparison_data = {
        'Model': [],
        'Silhouette': [],
        'Purity': [],
        'NMI': [],
        'Model Metric': [],  # Inertia for KM, AIC for GMM, F1 for LP
        'Training Time': [],
        'Notes': []
    }
    
    for model_name, results in results_dict.items():
        comparison_data['Model'].append(model_name)
        comparison_data['Silhouette'].append(results.get('silhouette', '-'))
        comparison_data['Purity'].append(results.get('purity', '-'))
        comparison_data['NMI'].append(results.get('nmi', '-'))
        comparison_data['Model Metric'].append(results.get('model_metric', '-'))
        comparison_data['Training Time'].append(results.get('training_time', '-'))
        comparison_data['Notes'].append(results.get('notes', ''))
    
    df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*100)
    print("PHASE 4: UNSUPERVISED & SEMI-SUPERVISED COMPARISON")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")
    
    return df
```

### 8.2 Example Results Output

```
====================================================================================================
PHASE 4: UNSUPERVISED & SEMI-SUPERVISED COMPARISON
====================================================================================================
                      Model  Silhouette   Purity      NMI  Model Metric  Training Time    Notes
         K-Means (K=2)    0.3421    0.6234   0.4521       3421340.25       12.34 sec   Separates correct/incorrect
         K-Means (K=4)    0.2987    0.5123   0.3876       3456789.45       14.21 sec   Weak correspondence to positions
         K-Means (K=6)    0.2456    0.4876   0.3234       2987654.32       15.89 sec   Over-clustering
         K-Means (K=8)    0.1923    0.4321   0.2876       2765432.10       17.45 sec   Strong over-clustering
         
      Label Propagation    0.3212    0.6145   0.4234    F1=0.6089 (vs 0.5856 sup) 342.12 sec  Semi-supervised works!
      
         GMM (K=2)        0.3198    0.6089   0.4398      AIC=3523456.21      45.67 sec   Best overall model
         GMM (K=4)        0.2876    0.5234   0.3654      AIC=3654321.87      48.90 sec
         GMM (K=6)        0.2345    0.4876   0.3123      AIC=3743210.65      52.34 sec
         GMM (K=8)        0.1654    0.3987   0.2456      AIC=3821098.43      56.78 sec
====================================================================================================
```

---

## 9. Implementation Roadmap

### 9.1 Phase 4 Workflow

```
Phase 4 Execution Plan (3-4 days):

Day 1: Dimensionality Reduction & K-Means
  ├─ Apply PCA to 50 dimensions (85% variance)
  ├─ Elbow method: test K=2,3,...,10
  ├─ Train K-Means K=4 (main model)
  ├─ Evaluate: silhouette, purity, cluster-label correspondence
  └─ Checkpoint: Save K-Means model + results

Day 2: Label Propagation
  ├─ Create partial labels (10% labeled, 90% unlabeled)
  ├─ Tune K neighbors: try K=3,5,7,10,15,20
  ├─ Train best LP model
  ├─ Compare: LP F1 vs supervised baseline (10% labeled)
  └─ Checkpoint: Save LP model + comparison

Day 3: Gaussian Mixture Models
  ├─ Train GMM K=2,4,6,8
  ├─ Compute AIC/BIC for model selection
  ├─ Analyze soft probabilities (entropy)
  ├─ Compare GMM vs K-Means (purity, silhouette)
  └─ Checkpoint: Save GMM models + analysis

Day 4: Synthesis & Reporting
  ├─ Create master comparison table (all models)
  ├─ Visualizations:
  │   ├─ Elbow curve (K vs inertia)
  │   ├─ Silhouette plots
  │   ├─ AIC/BIC curves
  │   ├─ LP performance vs K neighbors
  │   └─ Semi-supervised vs supervised bar chart
  ├─ Write conclusions: What did we learn?
  └─ Save all results to experiments.ipynb
```

### 9.2 File Outputs

```
After Phase 4, save:

models/model_a/unsupervised/
  ├─ kmeans_k2.pkl
  ├─ kmeans_k4.pkl
  ├─ kmeans_k6.pkl
  ├─ kmeans_k8.pkl
  ├─ pca_50.pkl                    ← Save PCA for inference
  ├─ label_propagation.pkl
  ├─ gmm_k2.pkl
  ├─ gmm_k4.pkl
  ├─ gmm_k6.pkl
  └─ gmm_k8.pkl

data/processed/
  ├─ X_train_pca_50.npy            ← PCA-reduced features
  ├─ X_dev_pca_50.npy
  ├─ X_test_pca_50.npy
  ├─ y_partial_train.npy            ← Partial labels for LP
  └─ phase4_results.json            ← All metrics and findings

notebooks/
  └─ experiments.ipynb
     ├─ Cell: PCA analysis + elbow curves
     ├─ Cell: K-Means results (all K)
     ├─ Cell: Silhouette plots
     ├─ Cell: Label Propagation tuning
     ├─ Cell: LP vs Supervised comparison
     ├─ Cell: GMM AIC/BIC analysis
     ├─ Cell: Master comparison table
     └─ Cell: Conclusions & next steps
```

---

## 10. Troubleshooting & Performance

### 10.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| K-Means converges in 1-2 iterations | Poor initialization or n_init too low | Increase `n_init=10` or use `init='k-means++'` |
| PCA loses too much info (< 80% var) | Not enough components | Increase to 100 components (still fast) |
| Label Propagation very slow (>1 hour) | Full data too large for LP graph | Use 50-100k sample for LP experiments |
| GMM won't converge | Covariance type too flexible | Try `covariance_type='diag'` or `'spherical'` |
| Silhouette score negative | Data not naturally clustered | Normal for answer verification; focus on purity |
| LP doesn't beat supervised | 10% labeled enough for supervised | Try fewer labeled examples (5% or 1%) |

### 10.2 Performance Profiling

```python
def profile_phase4_methods(X_train_pca, X_dev_pca, y_train, y_dev, y_partial):
    """Profile execution time and memory for all Phase 4 methods."""
    
    import time
    import psutil
    
    timings = {}
    
    print("\n" + "="*70)
    print("PHASE 4 PERFORMANCE PROFILING")
    print("="*70 + "\n")
    
    # K-Means
    print("K-Means...", end=" ", flush=True)
    start = time.time()
    km = KMeans(n_clusters=4, n_init=10, random_state=42, n_jobs=-1)
    km.fit(X_train_pca)
    timings['KMeans'] = time.time() - start
    print(f"{timings['KMeans']:.2f}s")
    
    # Label Propagation
    print("Label Propagation...", end=" ", flush=True)
    start = time.time()
    lp = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=1000)
    lp.fit(X_train_pca, y_partial)
    timings['LP'] = time.time() - start
    print(f"{timings['LP']:.2f}s")
    
    # GMM
    print("GMM...", end=" ", flush=True)
    start = time.time()
    gmm = GaussianMixture(n_components=4, n_init=10, random_state=42)
    gmm.fit(X_train_pca)
    timings['GMM'] = time.time() - start
    print(f"{timings['GMM']:.2f}s")
    
    print("\n" + "="*70)
    print("Summary:")
    for method, t in sorted(timings.items(), key=lambda x: x[1]):
        print(f"  {method:20s}: {t:8.2f} sec ({t/60:8.2f} min)")
    print("="*70 + "\n")
```

---

## Summary Checklist

Phase 4 completion checklist:

- [ ] **PCA applied:** Reduced to 50-100 components (85%+ variance)
- [ ] **K-Means tuning:** Tested K=2,3,...,10 with elbow analysis
- [ ] **K-Means evaluation:** Silhouette score, purity, cluster-label correspondence
- [ ] **Label Propagation:** Trained on 10% labeled + 90% unlabeled
- [ ] **LP hyperparameter tuning:** Tested K neighbors ∈ {3,5,7,10,15,20}
- [ ] **LP comparison:** Semi-supervised beats supervised with same labeled fraction
- [ ] **GMM trained:** K=2,4,6,8 with AIC/BIC analysis
- [ ] **GMM evaluation:** Silhouette, purity, entropy analysis
- [ ] **Master comparison table:** All metrics for all models
- [ ] **Visualizations:** Elbow curves, silhouette plots, AIC/BIC curves, LP bars
- [ ] **Conclusions:** Written explanation of findings
- [ ] **Models saved:** All .pkl files for models + PCA

**Expected Results:**
- Silhouette scores: 0.25-0.35 (weak-to-moderate structure)
- Purity: 0.50-0.65 (partial label-cluster correspondence)
- Label Propagation F1 improvement: +5-15% over supervised with 10% labels
- GMM K=2: Best AIC/BIC (correct/incorrect answers naturally separate)

---

**Next:** After Phase 4 is complete, proceed to Phase 5 (Ensemble methods: Soft Voting, Hard Voting, Stacking).
