"""
examples/cancer/run.py
-----------------------
Trains a DecisionTree (our implementation) AND sklearn's DecisionTreeClassifier
on the Breast Cancer Wisconsin dataset, then prints a side-by-side comparison.

Run from the project root:

    python -m examples.cancer.run
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for saving files
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier as SklearnDT

from decision_tree.tree import DecisionTree
from decision_tree.utils import train_test_split
from examples.cancer.dataset import CLASS_NAMES, load_cancer_dataset


class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = filepath.open("w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, data: str) -> int:
        self._stdout.write(data)
        return self._file.write(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ── Shared hyperparameters (same for both models) ───────────────────────────
MAX_DEPTH = 5
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 3
RANDOM_STATE = 42


def _entropy_per_feature(X: np.ndarray, feature_names: list[str]) -> None:
    """Print Shannon entropy (bits, 10-bin histogram) for each feature."""
    print("\nFeature entropy on training data (bits, 10-bin histogram):")
    print(f"  {'Feature':<36}  {'Entropy':>7}")
    print("  " + "-" * 46)
    for i, name in enumerate(feature_names):
        col = X[:, i]
        counts, _ = np.histogram(col, bins=10)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        feat_entropy = float(-np.sum(probs * np.log2(probs)))
        print(f"  {name:<36}  {feat_entropy:>7.4f}")


def main() -> None:
    # ── 1. Load data ─────────────────────────────────────────────────────────
    X, y, feature_names = load_cancer_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("\n" + "=" * 58)
    print("   Breast Cancer Wisconsin — Decision Tree Comparison")
    print("=" * 58)
    print(f"  Total samples   : {len(X)} (569 original)")
    print(f"  Features        : {X.shape[1]}")
    print(f"  Train / Test    : {len(X_train)} / {len(X_test)}")

    # ── Class distribution ───────────────────────────────────────────────────
    print("\nClass distribution:")
    print(f"  {'Class':<12}  {'Label':>5}  {'Train':>6}  {'Test':>6}")
    print("  " + "-" * 32)
    for label, name in enumerate(CLASS_NAMES):
        n_train = int(np.sum(y_train == label))
        n_test  = int(np.sum(y_test  == label))
        print(f"  {name:<12}  {label:>5}  {n_train:>6}  {n_test:>6}")

    # ── Class legend ─────────────────────────────────────────────────────────
    print("\nClass legend:")
    for label, name in enumerate(CLASS_NAMES):
        print(f"  class {label} → {name}")

    # ── Entropy per feature ──────────────────────────────────────────────────
    _entropy_per_feature(X_train, feature_names)

    # ── 2. Train — our DecisionTree ──────────────────────────────────────────
    our_tree = DecisionTree(
        criterion="gini",
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
    )
    t0 = time.perf_counter()
    our_tree.fit(X_train, y_train)
    our_train_time = time.perf_counter() - t0

    our_train_acc = our_tree.score(X_train, y_train)
    our_test_acc  = our_tree.score(X_test,  y_test)

    our_preds = our_tree.predict(X_test)

    # ── 3. Train — sklearn DecisionTreeClassifier ────────────────────────────
    sk_tree = SklearnDT(
        criterion="gini",
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )
    t0 = time.perf_counter()
    sk_tree.fit(X_train, y_train)
    sk_train_time = time.perf_counter() - t0

    sk_train_acc = sk_tree.score(X_train, y_train)
    sk_test_acc  = sk_tree.score(X_test,  y_test)

    sk_preds = sk_tree.predict(X_test)

    # ── 4. Side-by-side comparison ───────────────────────────────────────────
    print("\n" + "=" * 58)
    print("   Results Comparison")
    print("=" * 58)
    print(f"  {'Metric':<22}  {'Our DecisionTree':>16}  {'sklearn':>16}")
    print("  " + "-" * 56)
    print(f"  {'Train accuracy':<22}  {our_train_acc:>15.2%}  {sk_train_acc:>15.2%}")
    print(f"  {'Test  accuracy':<22}  {our_test_acc:>15.2%}  {sk_test_acc:>15.2%}")
    print(f"  {'Tree depth':<22}  {_tree_depth(our_tree):>16}  {sk_tree.get_depth():>16}")
    print(f"  {'Leaf nodes':<22}  {_count_leaves(our_tree):>16}  {sk_tree.get_n_leaves():>16}")
    print(f"  {'Train time (ms)':<22}  {our_train_time*1000:>15.2f}  {sk_train_time*1000:>15.2f}")

    # ── 5. Prediction agreement ──────────────────────────────────────────────
    agree = int(np.sum(our_preds == sk_preds))
    print(f"\nPrediction agreement on test set: {agree}/{len(X_test)} "
          f"({agree/len(X_test):.1%})")

    # ── 6. Per-class accuracy ────────────────────────────────────────────────
    print("\nPer-class accuracy on test set:")
    print(f"  {'Class':<12}  {'Our model':>10}  {'sklearn':>10}  {'Samples':>8}")
    print("  " + "-" * 44)
    for label, name in enumerate(CLASS_NAMES):
        mask = y_test == label
        if mask.sum() == 0:
            continue
        our_acc_c = float(np.mean(our_preds[mask] == y_test[mask]))
        sk_acc_c  = float(np.mean(sk_preds[mask]  == y_test[mask]))
        print(f"  {name:<12}  {our_acc_c:>10.2%}  {sk_acc_c:>10.2%}  {mask.sum():>8}")

    # ── 7. Feature importances (top 10) ─────────────────────────────────────
    assert our_tree.feature_importances_ is not None
    sk_imp = sk_tree.feature_importances_
    our_imp = our_tree.feature_importances_

    # rank by sklearn importance (most commonly used reference)
    top10_idx = np.argsort(sk_imp)[::-1][:10]

    print("\nTop-10 feature importances:")
    print(f"  {'Feature':<36}  {'Our model':>10}  {'sklearn':>10}")
    print("  " + "-" * 58)
    for i in top10_idx:
        print(f"  {feature_names[i]:<36}  {our_imp[i]:>10.4f}  {sk_imp[i]:>10.4f}")

    # ── 8. Test records (first 10) ───────────────────────────────────────────
    print("\nTest records (first 10):")
    for i in range(min(10, len(X_test))):
        true_name = CLASS_NAMES[int(y_test[i])]
        our_name  = CLASS_NAMES[int(our_preds[i])]
        sk_name   = CLASS_NAMES[int(sk_preds[i])]
        our_ok = "✓" if our_preds[i] == y_test[i] else "✗"
        sk_ok  = "✓" if sk_preds[i]  == y_test[i] else "✗"
        agree_mark = "=" if our_preds[i] == sk_preds[i] else "≠"
        print(
            f"  [{i:>2}]  True: {true_name:<10}  "
            f"Ours: {our_name:<10} {our_ok}  "
            f"sklearn: {sk_name:<10} {sk_ok}  "
            f"[{agree_mark}]"
        )

    # ── 9. Plot: mean radius vs probability ──────────────────────────────────
    _plot_feature_vs_probability(
        X_test, y_test, our_preds, sk_tree, feature_names,
        save_path=Path("results/cancer_plot.png"),
    )
    print()


def _plot_feature_vs_probability(
    X_test: np.ndarray,
    y_test: np.ndarray,
    our_preds: np.ndarray,
    sk_tree: SklearnDT,
    feature_names: list[str],
    save_path: Path,
) -> None:
    """Scatter plot: mean radius (x) vs predicted probability (y).

    * sklearn uses predict_proba → smooth probability of Benign (class 1).
    * Our model gives a hard 0/1 decision — plotted as horizontal bands
      with vertical jitter so overlapping points are visible.
    """
    feat_idx   = feature_names.index("mean radius")
    feat_vals  = X_test[:, feat_idx]

    # sklearn probabilities for class 1 (Benign)
    sk_proba = sk_tree.predict_proba(X_test)[:, 1]
    # our model: hard 0/1 (cast to float for the y-axis)
    our_proba = our_preds.astype(float)

    colors = ["#e74c3c" if c == 0 else "#2ecc71" for c in y_test]  # red=Malignant, green=Benign

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.02, 0.02, size=len(feat_vals))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle("Mean Radius vs Predicted Probability — Breast Cancer", fontsize=13)

    # ── Left: sklearn predict_proba ──────────────────────────────────────────
    ax = axes[0]
    sort_idx = np.argsort(feat_vals)
    ax.scatter(feat_vals, sk_proba, c=colors, alpha=0.7, edgecolors="k", linewidths=0.3, s=40)
    ax.plot(feat_vals[sort_idx], sk_proba[sort_idx], color="steelblue", alpha=0.3, lw=1)
    ax.axhline(0.5, color="grey", linestyle="--", lw=1, label="Decision boundary (p=0.5)")
    ax.set_xlabel("mean radius")
    ax.set_ylabel("P(Benign)")
    ax.set_title("sklearn DecisionTreeClassifier\n(predict_proba)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    # ── Right: our model hard prediction ────────────────────────────────────
    ax = axes[1]
    ax.scatter(feat_vals, our_proba + jitter, c=colors, alpha=0.7,
               edgecolors="k", linewidths=0.3, s=40)
    ax.axhline(0.5, color="grey", linestyle="--", lw=1, label="Decision boundary")
    ax.set_xlabel("mean radius")
    ax.set_ylabel("Predicted class (0=Malignant, 1=Benign)")
    ax.set_title("Our DecisionTree\n(hard 0/1 prediction)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0  Malignant", "1  Benign"])
    ax.set_ylim(-0.2, 1.2)
    ax.legend(fontsize=8)

    # ── Shared legend for true class colours ────────────────────────────────
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=8, label="True: Malignant"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
               markersize=8, label="True: Benign"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04), fontsize=9)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {save_path}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tree_depth(tree: DecisionTree) -> int:
    """Compute the maximum depth of a fitted DecisionTree."""
    def _depth(node) -> int:
        if node is None or node.is_leaf:
            return 0
        return 1 + max(_depth(node.left), _depth(node.right))
    return _depth(tree.root)


def _count_leaves(tree: DecisionTree) -> int:
    """Count leaf nodes in a fitted DecisionTree."""
    def _count(node) -> int:
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return _count(node.left) + _count(node.right)
    return _count(tree.root)


if __name__ == "__main__":
    results_path = Path("results/cancer_results.txt")
    tee = _Tee(results_path)
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = tee._stdout
        tee.close()
    print(f"Results saved to {results_path}")
