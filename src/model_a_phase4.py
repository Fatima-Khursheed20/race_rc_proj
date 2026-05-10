"""
Phase 4: Model A Unsupervised + Semi-Supervised Experiments

Implements:
- Dimensionality reduction (TruncatedSVD or PCA)
- K-Means clustering
- Label Propagation (semi-supervised)
- Gaussian Mixture Models

Usage examples:
  python src/model_a_phase4.py --task all
  python src/model_a_phase4.py --task kmeans --k-values 2 3 4 5 6
  python src/model_a_phase4.py --task lp --max-samples-lp 20000 --labeled-fraction 0.1
  python src/model_a_phase4.py --task gmm --gmm-components 2 4 6 8

Notes:
- For large sparse features, TruncatedSVD is safer than dense PCA.
- Label Propagation is O(n^2); use controlled sample size.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation


DATA_DIR = "data/processed"
MODEL_DIR = "models/model_a/traditional"
PHASE4_DIR = f"{MODEL_DIR}/phase4"
RANDOM_STATE = 42

Path(PHASE4_DIR).mkdir(parents=True, exist_ok=True)


def load_data(data_dir=DATA_DIR, verbose=True):
    """Load preprocessed features, labels, and row IDs from Phase 2."""

    try:
        X_train_ohe = sparse.load_npz(f"{data_dir}/X_train_ohe.npz")
        X_dev_ohe = sparse.load_npz(f"{data_dir}/X_dev_ohe.npz")
        X_test_ohe = sparse.load_npz(f"{data_dir}/X_test_ohe.npz")

        X_train_lex = np.load(f"{data_dir}/X_train_lexical.npy")
        X_dev_lex = np.load(f"{data_dir}/X_dev_lexical.npy")
        X_test_lex = np.load(f"{data_dir}/X_test_lexical.npy")

        y_train = np.load(f"{data_dir}/y_train.npy")
        y_dev = np.load(f"{data_dir}/y_dev.npy")
        y_test = np.load(f"{data_dir}/y_test.npy")

        row_ids_train = np.load(f"{data_dir}/row_ids_train.npy")
        row_ids_dev = np.load(f"{data_dir}/row_ids_dev.npy")
        row_ids_test = np.load(f"{data_dir}/row_ids_test.npy")

    except FileNotFoundError as exc:
        print("ERROR: Missing processed files. Run Phase 2 preprocessing first.")
        print(str(exc))
        sys.exit(1)

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 4 DATA LOADED")
        print("=" * 70)
        print(f"Train OHE: {X_train_ohe.shape} | Lex: {X_train_lex.shape} | y: {y_train.shape}")
        print(f"Dev   OHE: {X_dev_ohe.shape} | Lex: {X_dev_lex.shape} | y: {y_dev.shape}")
        print(f"Test  OHE: {X_test_ohe.shape} | Lex: {X_test_lex.shape} | y: {y_test.shape}")
        print("=" * 70 + "\n")

    return {
        "X_train_ohe": X_train_ohe,
        "X_dev_ohe": X_dev_ohe,
        "X_test_ohe": X_test_ohe,
        "X_train_lex": X_train_lex,
        "X_dev_lex": X_dev_lex,
        "X_test_lex": X_test_lex,
        "y_train": y_train,
        "y_dev": y_dev,
        "y_test": y_test,
        "row_ids_train": row_ids_train,
        "row_ids_dev": row_ids_dev,
        "row_ids_test": row_ids_test,
    }


def prepare_features(X_ohe, X_lex, is_training=False, scaler=None):
    """Scale lexical features and concatenate with sparse OHE features."""

    if is_training:
        scaler = StandardScaler()
        X_lex_scaled = scaler.fit_transform(X_lex)
    else:
        if scaler is None:
            raise ValueError("Scaler is required for transform-only mode")
        X_lex_scaled = scaler.transform(X_lex)

    X_lex_sparse = sparse.csr_matrix(X_lex_scaled)
    X_combined = sparse.hstack([X_ohe, X_lex_sparse]).tocsr()
    return X_combined, scaler


def sample_rows(X, y, row_ids, max_samples, seed=RANDOM_STATE):
    """Uniform random sampling helper for computationally expensive tasks."""

    n_total = X.shape[0]
    if max_samples is None or max_samples >= n_total:
        idx = np.arange(n_total)
        return X, y, row_ids, idx

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_total, size=max_samples, replace=False))

    if sparse.issparse(X):
        X_s = X[idx]
    else:
        X_s = X[idx, :]

    y_s = y[idx]
    row_ids_s = row_ids[idx]
    return X_s, y_s, row_ids_s, idx


def reduce_dimensions(X_train, X_dev, X_test, method="svd", n_components=50):
    """Reduce dimensionality for unsupervised and semi-supervised methods."""

    print("\n" + "=" * 70)
    print(f"DIMENSIONALITY REDUCTION ({method.upper()})")
    print("=" * 70)
    print(f"Input dimensions: {X_train.shape[1]:,}")
    print(f"Target dimensions: {n_components}")

    start = time.time()

    if method == "pca":
        # PCA requires dense input. Use only when memory allows.
        X_train_dense = X_train.toarray() if sparse.issparse(X_train) else X_train
        X_dev_dense = X_dev.toarray() if sparse.issparse(X_dev) else X_dev
        X_test_dense = X_test.toarray() if sparse.issparse(X_test) else X_test

        reducer = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_train_red = reducer.fit_transform(X_train_dense)
        X_dev_red = reducer.transform(X_dev_dense)
        X_test_red = reducer.transform(X_test_dense)

    else:
        reducer = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        X_train_red = reducer.fit_transform(X_train)
        X_dev_red = reducer.transform(X_dev)
        X_test_red = reducer.transform(X_test)

    elapsed = time.time() - start
    explained = float(np.sum(reducer.explained_variance_ratio_))

    print(f"Explained variance ratio: {explained:.4f} ({explained * 100:.2f}%)")
    print(f"Output shape (train): {X_train_red.shape}")
    print(f"Reduction time: {elapsed:.2f}s ({elapsed / 60:.2f} min)")
    print("=" * 70 + "\n")

    return X_train_red, X_dev_red, X_test_red, reducer, explained, elapsed


def purity_score(y_true, y_pred_cluster):
    """Cluster purity metric using majority class count per cluster."""

    cm = contingency_matrix(y_true, y_pred_cluster)
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)


def evaluate_clustering(y_true, cluster_labels, X_emb):
    """Evaluate clustering quality with multiple metrics."""

    metrics = {
        "silhouette": float(silhouette_score(X_emb, cluster_labels)),
        "purity": float(purity_score(y_true, cluster_labels)),
        "nmi": float(normalized_mutual_info_score(y_true, cluster_labels)),
        "ari": float(adjusted_rand_score(y_true, cluster_labels)),
    }
    return metrics


def run_kmeans(X_train_red, y_train, k_values):
    """Run K-Means over multiple K and return metric summary."""

    print("\n" + "=" * 70)
    print("K-MEANS EXPERIMENT")
    print("=" * 70)

    results = {}
    for k in k_values:
        start = time.time()
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=RANDOM_STATE,
        )
        cluster_labels = km.fit_predict(X_train_red)
        elapsed = time.time() - start

        metrics = evaluate_clustering(y_train, cluster_labels, X_train_red)
        metrics["inertia"] = float(km.inertia_)
        metrics["fit_time_sec"] = float(elapsed)

        results[str(k)] = metrics

        print(
            f"K={k:<2} | silhouette={metrics['silhouette']:.4f} | "
            f"purity={metrics['purity']:.4f} | nmi={metrics['nmi']:.4f} | "
            f"inertia={metrics['inertia']:.2f}"
        )

    best_k = max(results.items(), key=lambda kv: kv[1]["silhouette"])[0]
    print(f"Best K by silhouette: {best_k}")
    print("=" * 70 + "\n")

    return {"results_by_k": results, "best_k_by_silhouette": int(best_k)}


def run_label_propagation(
    X_train_red,
    y_train,
    X_dev_red,
    y_dev,
    labeled_fraction=0.1,
    n_neighbors=15,
):
    """Run Label Propagation with partial labels and compare to supervised baseline."""

    print("\n" + "=" * 70)
    print("LABEL PROPAGATION EXPERIMENT")
    print("=" * 70)

    n_samples = X_train_red.shape[0]
    n_labeled = max(1, int(n_samples * labeled_fraction))

    rng = np.random.default_rng(RANDOM_STATE)
    labeled_idx = np.sort(rng.choice(n_samples, size=n_labeled, replace=False))

    y_semi = np.full_like(y_train, fill_value=-1)
    y_semi[labeled_idx] = y_train[labeled_idx]

    lp = LabelPropagation(kernel="knn", n_neighbors=n_neighbors, max_iter=1000)

    start = time.time()
    lp.fit(X_train_red, y_semi)
    lp_time = time.time() - start

    y_dev_lp = lp.predict(X_dev_red)
    lp_acc = accuracy_score(y_dev, y_dev_lp)
    lp_f1 = f1_score(y_dev, y_dev_lp, average="macro")

    # Supervised baseline with same labeled subset only.
    sup = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    sup.fit(X_train_red[labeled_idx], y_train[labeled_idx])
    y_dev_sup = sup.predict(X_dev_red)
    sup_acc = accuracy_score(y_dev, y_dev_sup)
    sup_f1 = f1_score(y_dev, y_dev_sup, average="macro")

    gain_acc = float(lp_acc - sup_acc)
    gain_f1 = float(lp_f1 - sup_f1)

    print(f"Samples used: {n_samples:,}")
    print(f"Labeled fraction: {labeled_fraction:.2f} ({n_labeled:,} labeled)")
    print(f"LP dev accuracy: {lp_acc:.4f} | macro_f1: {lp_f1:.4f}")
    print(f"SUP dev accuracy: {sup_acc:.4f} | macro_f1: {sup_f1:.4f}")
    print(f"Gain (LP - SUP): acc={gain_acc:+.4f}, f1={gain_f1:+.4f}")
    print(f"LP fit time: {lp_time:.2f}s ({lp_time / 60:.2f} min)")
    print("=" * 70 + "\n")

    return {
        "labeled_fraction": float(labeled_fraction),
        "n_neighbors": int(n_neighbors),
        "n_samples": int(n_samples),
        "n_labeled": int(n_labeled),
        "lp_dev_accuracy": float(lp_acc),
        "lp_dev_macro_f1": float(lp_f1),
        "supervised_dev_accuracy": float(sup_acc),
        "supervised_dev_macro_f1": float(sup_f1),
        "gain_accuracy": gain_acc,
        "gain_macro_f1": gain_f1,
        "fit_time_sec": float(lp_time),
    }


def run_gmm(X_train_red, y_train, components_list):
    """Run Gaussian Mixture Models with AIC/BIC and clustering metrics."""

    print("\n" + "=" * 70)
    print("GAUSSIAN MIXTURE MODEL EXPERIMENT")
    print("=" * 70)

    results = {}
    for n_comp in components_list:
        start = time.time()
        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type="full",
            random_state=RANDOM_STATE,
            reg_covar=1e-6,
        )
        gmm.fit(X_train_red)

        labels = gmm.predict(X_train_red)
        elapsed = time.time() - start

        metrics = evaluate_clustering(y_train, labels, X_train_red)
        metrics["aic"] = float(gmm.aic(X_train_red))
        metrics["bic"] = float(gmm.bic(X_train_red))
        metrics["fit_time_sec"] = float(elapsed)

        probs = gmm.predict_proba(X_train_red)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1).mean()
        metrics["avg_posterior_entropy"] = float(entropy)

        results[str(n_comp)] = metrics

        print(
            f"C={n_comp:<2} | silhouette={metrics['silhouette']:.4f} | "
            f"purity={metrics['purity']:.4f} | bic={metrics['bic']:.2f} | "
            f"entropy={metrics['avg_posterior_entropy']:.4f}"
        )

    best_bic = min(results.items(), key=lambda kv: kv[1]["bic"])[0]
    print(f"Best components by BIC: {best_bic}")
    print("=" * 70 + "\n")

    return {"results_by_components": results, "best_components_by_bic": int(best_bic)}


def save_json(payload, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def create_summary_table(phase4_results):
    """Create compact table-like dict for quick inspection."""

    summary = {}

    if "kmeans" in phase4_results:
        best_k = str(phase4_results["kmeans"]["best_k_by_silhouette"])
        best_metrics = phase4_results["kmeans"]["results_by_k"][best_k]
        summary["kmeans_best"] = {
            "k": int(best_k),
            "silhouette": best_metrics["silhouette"],
            "purity": best_metrics["purity"],
            "nmi": best_metrics["nmi"],
        }

    if "label_propagation" in phase4_results:
        lp = phase4_results["label_propagation"]
        summary["label_propagation"] = {
            "labeled_fraction": lp["labeled_fraction"],
            "lp_dev_accuracy": lp["lp_dev_accuracy"],
            "supervised_dev_accuracy": lp["supervised_dev_accuracy"],
            "gain_accuracy": lp["gain_accuracy"],
        }

    if "gmm" in phase4_results:
        best_c = str(phase4_results["gmm"]["best_components_by_bic"])
        best_metrics = phase4_results["gmm"]["results_by_components"][best_c]
        summary["gmm_best"] = {
            "components": int(best_c),
            "silhouette": best_metrics["silhouette"],
            "purity": best_metrics["purity"],
            "bic": best_metrics["bic"],
        }

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: Unsupervised + Semi-Supervised")

    parser.add_argument("--task", choices=["kmeans", "lp", "gmm", "all"], default="all")
    parser.add_argument("--data-dir", default=DATA_DIR)

    parser.add_argument("--dim-method", choices=["svd", "pca"], default="svd")
    parser.add_argument("--n-components", type=int, default=50)

    parser.add_argument("--max-samples-cluster", type=int, default=80000)
    parser.add_argument("--max-samples-lp", type=int, default=20000)

    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--gmm-components", type=int, nargs="+", default=[2, 4, 6, 8])

    parser.add_argument("--labeled-fraction", type=float, default=0.1)
    parser.add_argument("--lp-neighbors", type=int, default=15)

    return parser.parse_args()


def main():
    args = parse_args()

    data = load_data(data_dir=args.data_dir, verbose=True)

    X_train, scaler = prepare_features(data["X_train_ohe"], data["X_train_lex"], is_training=True)
    X_dev, _ = prepare_features(data["X_dev_ohe"], data["X_dev_lex"], is_training=False, scaler=scaler)
    X_test, _ = prepare_features(data["X_test_ohe"], data["X_test_lex"], is_training=False, scaler=scaler)

    X_train_red, X_dev_red, X_test_red, reducer, explained, reduction_time = reduce_dimensions(
        X_train,
        X_dev,
        X_test,
        method=args.dim_method,
        n_components=args.n_components,
    )

    phase4_results = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "reduction": {
            "method": args.dim_method,
            "n_components": args.n_components,
            "explained_variance_ratio": explained,
            "fit_time_sec": reduction_time,
        },
    }

    if args.task in ("kmeans", "all"):
        X_km, y_km, _, _ = sample_rows(
            X_train_red,
            data["y_train"],
            data["row_ids_train"],
            max_samples=args.max_samples_cluster,
        )
        phase4_results["kmeans"] = run_kmeans(X_km, y_km, args.k_values)

    if args.task in ("lp", "all"):
        X_lp_train, y_lp_train, _, _ = sample_rows(
            X_train_red,
            data["y_train"],
            data["row_ids_train"],
            max_samples=args.max_samples_lp,
        )
        # Use same subset size for dev if needed to control runtime.
        X_lp_dev, y_lp_dev, _, _ = sample_rows(
            X_dev_red,
            data["y_dev"],
            data["row_ids_dev"],
            max_samples=min(args.max_samples_lp, X_dev_red.shape[0]),
        )

        phase4_results["label_propagation"] = run_label_propagation(
            X_lp_train,
            y_lp_train,
            X_lp_dev,
            y_lp_dev,
            labeled_fraction=args.labeled_fraction,
            n_neighbors=args.lp_neighbors,
        )

    if args.task in ("gmm", "all"):
        X_gmm, y_gmm, _, _ = sample_rows(
            X_train_red,
            data["y_train"],
            data["row_ids_train"],
            max_samples=args.max_samples_cluster,
        )
        phase4_results["gmm"] = run_gmm(X_gmm, y_gmm, args.gmm_components)

    phase4_results["summary"] = create_summary_table(phase4_results)

    reducer_path = f"{PHASE4_DIR}/{args.dim_method}_reducer_{args.n_components}.pkl"
    scaler_path = f"{PHASE4_DIR}/scaler_lexical_phase4.pkl"
    results_path = f"{PHASE4_DIR}/phase4_results.json"

    joblib.dump(reducer, reducer_path)
    joblib.dump(scaler, scaler_path)
    save_json(phase4_results, results_path)

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)
    print(f"Saved reducer: {reducer_path}")
    print(f"Saved scaler:  {scaler_path}")
    print(f"Saved results: {results_path}")

    print("\nSummary:")
    print(json.dumps(phase4_results["summary"], indent=2))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
