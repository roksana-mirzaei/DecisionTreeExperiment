"""
examples/sensor_failure/run.py
-------------------------------
End-to-end pipeline for the robotics sensor-failure classification example.

Demonstrates:
  1. Generating the synthetic sensor dataset via ``make_sensor_dataset``.
  2. Splitting into 80 % train / 20 % test using ``train_test_split``.
  3. Fitting a ``DecisionTree`` (no scikit-learn in the core logic).
  4. Reporting train and test accuracy.
  5. Displaying feature importances as a simple bar chart in the terminal.
  6. Pretty-printing the full tree structure with ASCII indentation.
  7. Running a single-sample prediction as a sanity check.

Run from the project root directory:

    python -m examples.sensor_failure.run
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for saving files
import matplotlib.pyplot as plt
import numpy as np

from decision_tree.tree import DecisionTree
from decision_tree.utils import print_tree, train_test_split
from examples.sensor_failure.dataset import (
    CLASS_NAMES,
    FEATURE_NAMES,
    make_sensor_dataset,
)


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


def main() -> None:
    """Train, evaluate, and display a decision tree for sensor-failure data."""

    # ── 1. Generate data ────────────────────────────────────────────────────
    X, y = make_sensor_dataset(n_samples=600, random_state=0)

    # ── 2. Train / test split (80 / 20) ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # ── 3. Fit ───────────────────────────────────────────────────────────────
    clf = DecisionTree(
        criterion="gini",
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=3,
    )
    clf.fit(X_train, y_train)

    # ── 4. Accuracy ──────────────────────────────────────────────────────────
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print("\n" + "=" * 46)
    print("   Sensor Failure — Decision Tree Report")
    print("=" * 46)
    print(f"  Train samples  : {len(X_train)}")
    print(f"  Test  samples  : {len(X_test)}")
    print(f"  Train accuracy : {train_acc:.2%}")
    print(f"  Test  accuracy : {test_acc:.2%}")

    # ── Class distribution ───────────────────────────────────────────────────
    print("\nClass distribution:")
    print(f"  {'Class':<22}  {'Train':>6}  {'Test':>6}")
    print("  " + "-" * 36)
    for label, name in enumerate(CLASS_NAMES):
        n_train = int(np.sum(y_train == label))
        n_test  = int(np.sum(y_test  == label))
        print(f"  {name:<22}  {n_train:>6}  {n_test:>6}")

    # ── 4b. Entropy per feature (binned) ─────────────────────────────────────
    # Each continuous feature is discretised into 10 equal-width bins;
    # Shannon entropy (bits) is then computed over those bin counts.
    # Higher entropy = more spread-out / uncertain feature values.
    print("\nFeature entropy on training data (bits, 10-bin histogram):")
    print(f"  {'Feature':<18}  {'Entropy (bits)':>14}")
    print("  " + "-" * 34)
    for i, name in enumerate(FEATURE_NAMES):
        col = X_train[:, i]
        counts, _ = np.histogram(col, bins=10)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        feat_entropy = float(-np.sum(probs * np.log2(probs)))
        print(f"  {name:<18}  {feat_entropy:>14.4f}")

    # ── 5. Feature importances (ASCII bar chart) ─────────────────────────────
    assert clf.feature_importances_ is not None
    print("\nFeature importances:")
    for name, imp in zip(FEATURE_NAMES, clf.feature_importances_):
        bar_length = int(imp * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"  {name:<18}  {imp:.4f}  {bar}")

    # ── 6. Tree structure ────────────────────────────────────────────────────
    print("\nTree structure:")
    assert clf.root is not None
    print_tree(clf.root, feature_names=FEATURE_NAMES, class_names=CLASS_NAMES)

    # ── Class legend ─────────────────────────────────────────────────────────
    print("\nClass legend:")
    for label, name in enumerate(CLASS_NAMES):
        print(f"  class {label} → {name}")

    # ── 7. Single-sample prediction ──────────────────────────────────────────
    # Construct a sample that should look like a mechanical failure:
    # high vibration (~210 Hz), normal temperature (~44 °C), stable voltage.
    test_sample = np.array([[210.0, 44.0, 4.9, 1.3, 26.0]])
    pred_label = int(clf.predict(test_sample)[0])
    print(f"\nSingle-sample prediction:")
    print(f"  Features : vibration=210 Hz, temp=44 °C, voltage=4.9 V, "
          f"current=1.3 A, latency=26 ms")
    print(f"  Predicted: {CLASS_NAMES[pred_label]} (class {pred_label})")

    # ── 8. Test-set records (first 10) ───────────────────────────────────────
    print("\nTest records (first 10):")
    y_pred_test = clf.predict(X_test)
    for i in range(min(10, len(X_test))):
        v = X_test[i]
        true_name = CLASS_NAMES[int(y_test[i])]
        pred_name = CLASS_NAMES[int(y_pred_test[i])]
        ok = "✓" if y_test[i] == y_pred_test[i] else "✗"
        print(
            f"  [{i:>2}]  vib={v[0]:>7.2f} Hz "
            f"temp={v[1]:>6.2f} °C "
            f"volt={v[2]:>5.2f} V "
            f"curr={v[3]:>5.2f} A "
            f"resp={v[4]:>6.2f} ms"
        )
        print(f"        True: {true_name:<20}  Pred: {pred_name:<20}  {ok}\n")

    # ── 9. Plot: vibration_hz vs predicted class ──────────────────────────────
    _plot_feature_vs_prediction(
        X_test, y_test, clf,
        feature_name="vibration_hz",
        feature_idx=FEATURE_NAMES.index("vibration_hz"),
        save_path=Path("results/sensor_failure_plot.png"),
    )


def _plot_feature_vs_prediction(
    X_test: np.ndarray,
    y_test: np.ndarray,
    clf: DecisionTree,
    feature_name: str,
    feature_idx: int,
    save_path: Path,
) -> None:
    """Scatter plot: top feature value (x) vs predicted class (y).

    Each dot is a test sample coloured by its TRUE class; the y-position
    shows what class the tree PREDICTED.  Misclassified samples appear as
    a dot whose colour does not match its row's expected position.
    """
    feat_vals = X_test[:, feature_idx]
    preds = clf.predict(X_test)

    # colour = true class
    palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    colors = [palette[int(c)] for c in y_test]

    rng = np.random.default_rng(1)
    jitter = rng.uniform(-0.25, 0.25, size=len(feat_vals))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(feat_vals, preds + jitter, c=colors, alpha=0.75,
               edgecolors="k", linewidths=0.3, s=45)

    # draw horizontal lines at each class value
    for label, name in enumerate(CLASS_NAMES):
        ax.axhline(label, color="grey", linestyle="--", lw=0.7, alpha=0.6)
        ax.text(feat_vals.max() * 1.01, label, name, va="center", fontsize=8)

    ax.set_xlabel(feature_name)
    ax.set_ylabel("Predicted class")
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels([f"{i}" for i in range(len(CLASS_NAMES))])
    ax.set_title(f"{feature_name} vs Predicted Class — Sensor Failure\n"
                 "(dot colour = true class; y-position = predicted class)")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=palette[i], edgecolor="k", label=f"{i}: {name}")
        for i, name in enumerate(CLASS_NAMES)
    ]
    ax.legend(handles=legend_handles, title="True class",
              loc="upper left", fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    results_path = Path("results/sensor_failure_results.txt")
    tee = _Tee(results_path)
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = tee._stdout
        tee.close()
    print(f"Results saved to {results_path}")
