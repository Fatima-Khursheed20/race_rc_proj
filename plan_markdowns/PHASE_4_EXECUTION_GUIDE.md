# Phase 4 Execution Guide (Colab + Project Integration)

This guide explains exactly what to run for Phase 4 and how to integrate outputs back into your project after running on Colab.

## 1) What Was Implemented

Code file:
- src/model_a_phase4.py

Implemented tasks:
- Dimensionality reduction: TruncatedSVD (default) or PCA
- K-Means clustering (multiple K values)
- Label Propagation (semi-supervised)
- Gaussian Mixture Models (multiple component counts)
- JSON output with all metrics + compact summary

Saved artifacts:
- models/model_a/traditional/phase4/phase4_results.json
- models/model_a/traditional/phase4/svd_reducer_50.pkl (or PCA reducer)
- models/model_a/traditional/phase4/scaler_lexical_phase4.pkl

## 2) Colab Setup (What To Run Where)

Run these in a Colab code cell.

### Cell A: Mount Drive and change directory

```python
from google.colab import drive
import os

drive.mount('/content/drive')
PROJECT_PATH = '/content/drive/My Drive/Semester 6/AI/Project/race_rc_proj'
os.chdir(PROJECT_PATH)
print('cwd:', os.getcwd())
```

### Cell B: Install dependencies

```python
!pip install -q numpy pandas scipy scikit-learn joblib
```

### Cell C: Run full Phase 4

```python
!python src/model_a_phase4.py --task all --dim-method svd --n-components 50 --max-samples-cluster 80000 --max-samples-lp 20000 --labeled-fraction 0.1
```

Expected runtime:
- K-Means + GMM: moderate
- Label Propagation: can be slow; controlled by --max-samples-lp

### Cell D: Optional targeted runs

K-Means only:

```python
!python src/model_a_phase4.py --task kmeans --k-values 2 3 4 5 6 7 8 --dim-method svd --n-components 50
```

Label Propagation only:

```python
!python src/model_a_phase4.py --task lp --max-samples-lp 20000 --labeled-fraction 0.1 --lp-neighbors 15 --dim-method svd --n-components 50
```

GMM only:

```python
!python src/model_a_phase4.py --task gmm --gmm-components 2 4 6 8 --dim-method svd --n-components 50
```

## 3) Recommended Defaults

Use these first:
- --dim-method svd
- --n-components 50
- --max-samples-cluster 80000
- --max-samples-lp 20000
- --labeled-fraction 0.1

Why:
- SVD works directly with sparse features and is memory-safe.
- Label Propagation scales poorly, so sample size control is important.

## 4) How To Integrate Back Into Your Project After Colab

After training on Colab, your outputs are already in Google Drive (inside your project folder). To integrate locally:

1. Ensure local project has these files after sync/download:
   - models/model_a/traditional/phase4/phase4_results.json
   - models/model_a/traditional/phase4/svd_reducer_50.pkl (or pca_reducer_50.pkl)
   - models/model_a/traditional/phase4/scaler_lexical_phase4.pkl

2. Pull/copy those artifacts into your local repo folder at the same paths.

3. Add Phase 4 reporting in your notebook:
   - Open notebooks/experiments.ipynb
   - Load phase4_results.json and render comparison table/plots

4. Add report references:
   - Include K-Means best K and silhouette/purity
   - Include LP gain over supervised-on-small-labels baseline
   - Include GMM best component count by BIC and corresponding clustering quality

## 5) Minimal Integration Snippet (Notebook)

Paste into a notebook cell:

```python
import json
from pathlib import Path

p = Path('models/model_a/traditional/phase4/phase4_results.json')
with p.open('r', encoding='utf-8') as f:
    phase4 = json.load(f)

print('Summary:')
print(json.dumps(phase4.get('summary', {}), indent=2))
```

## 6) Suggested Git Add/Commit After Colab

Run locally after syncing artifacts:

```bash
git add src/model_a_phase4.py
git add plan_markdowns/PHASE_4_EXECUTION_GUIDE.md
git add models/model_a/traditional/phase4/phase4_results.json
git add models/model_a/traditional/phase4/*.pkl
git commit -m "phase 4: add kmeans, label propagation, gmm pipeline and artifacts"
```

## 7) Troubleshooting

If Label Propagation is too slow or runs out of memory:
- Lower --max-samples-lp to 12000 or 8000
- Lower --lp-neighbors to 10

If clustering is slow:
- Lower --max-samples-cluster to 50000

If explained variance is too low:
- Increase --n-components from 50 to 80 or 100
