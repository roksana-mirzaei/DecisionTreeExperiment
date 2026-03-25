"""
tests/test_splitter.py
-----------------------
Unit tests for the impurity metrics and split-search logic
(decision_tree/splitter.py).

Covers:
  * ``gini_impurity`` — pure nodes, balanced binary, multi-class, empty.
  * ``entropy``        — pure nodes, balanced binary, empty.
  * ``information_gain`` — perfect splits, no-improvement splits, bad criterion.
  * ``best_split``     — identifies the correct feature, returns positive gain,
                         handles ties and constant features gracefully.
"""

from __future__ import annotations

import numpy as np
import pytest

from decision_tree.splitter import (
    best_split,
    entropy,
    gini_impurity,
    information_gain,
)


# ---------------------------------------------------------------------------
# gini_impurity
# ---------------------------------------------------------------------------


def test_gini_pure_node_is_zero() -> None:
    """A perfectly pure node must have Gini = 0."""
    assert gini_impurity(np.array([1, 1, 1, 1])) == pytest.approx(0.0)


def test_gini_balanced_binary_is_half() -> None:
    """A 50/50 binary split must produce Gini = 0.5."""
    assert gini_impurity(np.array([0, 0, 1, 1])) == pytest.approx(0.5)


def test_gini_empty_array_is_zero() -> None:
    assert gini_impurity(np.array([], dtype=int)) == pytest.approx(0.0)


def test_gini_multiclass_is_less_than_one() -> None:
    """Gini must always be strictly less than 1.0."""
    y = np.array([0, 1, 2, 3] * 5)
    assert 0.0 < gini_impurity(y) < 1.0


def test_gini_single_sample_is_zero() -> None:
    assert gini_impurity(np.array([7])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------


def test_entropy_pure_node_is_zero() -> None:
    """A pure node must have entropy = 0 bits."""
    assert entropy(np.array([0, 0, 0])) == pytest.approx(0.0)


def test_entropy_balanced_binary_is_one_bit() -> None:
    """A balanced binary distribution must have entropy = 1.0 bit."""
    assert entropy(np.array([0, 1, 0, 1])) == pytest.approx(1.0)


def test_entropy_empty_array_is_zero() -> None:
    assert entropy(np.array([], dtype=int)) == pytest.approx(0.0)


def test_entropy_is_non_negative() -> None:
    y = np.array([0, 0, 1, 2, 2, 2])
    assert entropy(y) >= 0.0


# ---------------------------------------------------------------------------
# information_gain
# ---------------------------------------------------------------------------


def test_ig_perfect_split_equals_parent_impurity() -> None:
    """Splitting into two pure children must yield IG = parent impurity."""
    y_parent = np.array([0, 0, 1, 1])
    y_left = np.array([0, 0])
    y_right = np.array([1, 1])
    ig = information_gain(y_parent, y_left, y_right, "gini")
    # Parent is 0.5; after perfect split children are 0.0 each → IG = 0.5
    assert ig == pytest.approx(0.5)


def test_ig_no_improvement_is_approximately_zero() -> None:
    """A split where each child has the same class distribution as the parent yields IG ≈ 0."""
    # Parent: [0, 0, 1, 1] — gini = 0.5
    # Left:   [0, 1]       — gini = 0.5  (same distribution)
    # Right:  [0, 1]       — gini = 0.5  (same distribution)
    # Weighted children = 0.5 → IG = 0.5 - 0.5 = 0.0
    y_parent = np.array([0, 0, 1, 1])
    y_left = np.array([0, 1])
    y_right = np.array([0, 1])
    ig = information_gain(y_parent, y_left, y_right, "gini")
    assert ig == pytest.approx(0.0, abs=1e-9)


def test_ig_entropy_criterion_works() -> None:
    y_parent = np.array([0, 0, 1, 1])
    y_left = np.array([0, 0])
    y_right = np.array([1, 1])
    ig = information_gain(y_parent, y_left, y_right, "entropy")
    assert ig == pytest.approx(1.0)


def test_ig_unknown_criterion_raises() -> None:
    with pytest.raises(ValueError, match="Unknown criterion"):
        information_gain(np.array([0, 1]), np.array([0]), np.array([1]), "mse")


def test_ig_empty_parent_is_zero() -> None:
    ig = information_gain(np.array([], dtype=int), np.array([0]), np.array([1]))
    assert ig == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# best_split
# ---------------------------------------------------------------------------


def test_best_split_finds_correct_feature() -> None:
    """Feature 0 is a perfect separator; best_split must choose it."""
    X = np.array([[1.0, 5.0], [2.0, 4.0], [8.0, 5.0], [9.0, 4.0]])
    y = np.array([0, 0, 1, 1])
    feat, thresh, gain = best_split(X, y, "gini")
    assert feat == 0
    assert gain > 0.0


def test_best_split_returns_positive_gain_for_separable_data() -> None:
    X = np.array([[1.0], [2.0], [10.0], [11.0]])
    y = np.array([0, 0, 1, 1])
    _, _, gain = best_split(X, y)
    assert gain > 0.0


def test_best_split_constant_feature_returns_zero_gain() -> None:
    """When all feature values are identical, no split improves purity."""
    X = np.array([[5.0], [5.0], [5.0], [5.0]])
    y = np.array([0, 0, 1, 1])
    _, _, gain = best_split(X, y)
    assert gain == pytest.approx(0.0)


def test_best_split_pure_labels_returns_zero_gain() -> None:
    """A pure node cannot be improved by any split."""
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1, 1, 1])
    _, _, gain = best_split(X, y)
    assert gain == pytest.approx(0.0)


def test_best_split_threshold_is_midpoint() -> None:
    """The returned threshold must be the midpoint between two adjacent values."""
    X = np.array([[1.0], [3.0], [7.0], [9.0]])
    y = np.array([0, 0, 1, 1])
    _, threshold, _ = best_split(X, y)
    # The optimal split is between 3 and 7 → midpoint = 5.0
    assert threshold == pytest.approx(5.0)


def test_best_split_n_features_subsampling() -> None:
    """Passing n_features=1 must still return a valid (feature, threshold, gain) tuple."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 5))
    y = (X[:, 0] > 0).astype(int)      # feature 0 is the ground-truth separator
    feat, thresh, gain = best_split(X, y, "gini", n_features=1)
    assert isinstance(feat, int)
    assert isinstance(thresh, float)
    assert gain >= 0.0
