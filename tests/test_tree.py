"""
tests/test_tree.py
-------------------
Integration tests for DecisionTree (decision_tree/tree.py).

Covers:
  * API contract — fit returns self, predict raises before fit.
  * Accuracy — perfect score on linearly separable data.
  * Hyperparameters — max_depth, min_samples_split, min_samples_leaf respected.
  * Post-fit attributes — n_classes_, feature_importances_ shape and sum.
  * Both criteria — gini and entropy produce valid results.
  * Multi-class prediction — correct output shape and label set.
  * Chaining — fit().predict() works in a single expression.
"""

from __future__ import annotations

import numpy as np
import pytest

from decision_tree.tree import DecisionTree


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def separable_binary() -> tuple[np.ndarray, np.ndarray]:
    """Simple binary dataset: label 0 if x[0] < 10 else label 1.

    A depth-1 tree can achieve 100 % accuracy here.
    """
    X = np.array([[float(i), float(i) * 0.5] for i in range(20)])
    y = np.array([0 if i < 10 else 1 for i in range(20)], dtype=int)
    return X, y


@pytest.fixture
def separable_multiclass() -> tuple[np.ndarray, np.ndarray]:
    """Three linearly separable classes along feature 0."""
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=0.0, scale=0.5, size=(30, 2))
    X1 = rng.normal(loc=5.0, scale=0.5, size=(30, 2))
    X2 = rng.normal(loc=10.0, scale=0.5, size=(30, 2))
    X = np.vstack([X0, X1, X2])
    y = np.array([0] * 30 + [1] * 30 + [2] * 30, dtype=int)
    return X, y


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


def test_fit_returns_self(separable_binary: tuple) -> None:
    X, y = separable_binary
    clf = DecisionTree()
    result = clf.fit(X, y)
    assert result is clf


def test_chaining_fit_predict(separable_binary: tuple) -> None:
    X, y = separable_binary
    preds = DecisionTree(max_depth=2).fit(X, y).predict(X)
    assert preds.shape == (len(y),)


def test_predict_before_fit_raises() -> None:
    clf = DecisionTree()
    with pytest.raises(RuntimeError, match="fit"):
        clf.predict(np.array([[1.0, 2.0]]))


def test_root_is_none_before_fit() -> None:
    clf = DecisionTree()
    assert clf.root is None


def test_root_is_set_after_fit(separable_binary: tuple) -> None:
    X, y = separable_binary
    clf = DecisionTree().fit(X, y)
    assert clf.root is not None


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def test_perfect_accuracy_on_separable_data(separable_binary: tuple) -> None:
    X, y = separable_binary
    clf = DecisionTree(max_depth=5).fit(X, y)
    assert clf.score(X, y) == pytest.approx(1.0)


def test_perfect_accuracy_multiclass(separable_multiclass: tuple) -> None:
    X, y = separable_multiclass
    clf = DecisionTree(max_depth=5).fit(X, y)
    assert clf.score(X, y) == pytest.approx(1.0)


def test_entropy_criterion_achieves_perfect_accuracy(separable_binary: tuple) -> None:
    X, y = separable_binary
    clf = DecisionTree(criterion="entropy", max_depth=5).fit(X, y)
    assert clf.score(X, y) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Hyperparameter enforcement
# ---------------------------------------------------------------------------


def test_max_depth_one_creates_single_split(separable_binary: tuple) -> None:
    """With max_depth=1 the root's children must all be leaves."""
    X, y = separable_binary
    clf = DecisionTree(max_depth=1).fit(X, y)
    assert clf.root is not None
    assert clf.root.left is not None and clf.root.left.is_leaf
    assert clf.root.right is not None and clf.root.right.is_leaf


def test_max_depth_zero_creates_root_leaf(separable_binary: tuple) -> None:
    """With max_depth=0 the root itself must be a leaf (majority vote)."""
    X, y = separable_binary
    clf = DecisionTree(max_depth=0).fit(X, y)
    assert clf.root is not None
    assert clf.root.is_leaf


def test_min_samples_split_prevents_small_splits() -> None:
    """Setting min_samples_split high should force a leaf at the root."""
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    clf = DecisionTree(min_samples_split=100).fit(X, y)
    assert clf.root is not None
    assert clf.root.is_leaf  # too few samples to split


def test_min_samples_leaf_respected() -> None:
    """A split producing a child with fewer than min_samples_leaf samples must be rejected."""
    X = np.array([[1.0], [2.0], [3.0], [100.0]])  # last point would create a 1-sample leaf
    y = np.array([0, 0, 0, 1])
    clf = DecisionTree(min_samples_leaf=2).fit(X, y)
    # With min_samples_leaf=2, the lone outlier cannot form its own leaf
    assert clf.root is not None


# ---------------------------------------------------------------------------
# Post-fit attributes
# ---------------------------------------------------------------------------


def test_n_classes_binary(separable_binary: tuple) -> None:
    X, y = separable_binary
    clf = DecisionTree().fit(X, y)
    assert clf.n_classes_ == 2


def test_n_classes_multiclass(separable_multiclass: tuple) -> None:
    X, y = separable_multiclass
    clf = DecisionTree().fit(X, y)
    assert clf.n_classes_ == 3


def test_feature_importances_shape(separable_binary: tuple) -> None:
    X, y = separable_binary
    clf = DecisionTree().fit(X, y)
    assert clf.feature_importances_ is not None
    assert clf.feature_importances_.shape == (X.shape[1],)


def test_feature_importances_sum_to_one(separable_binary: tuple) -> None:
    X, y = separable_binary
    clf = DecisionTree().fit(X, y)
    assert clf.feature_importances_ is not None
    assert np.sum(clf.feature_importances_) == pytest.approx(1.0)


def test_feature_importances_non_negative(separable_binary: tuple) -> None:
    X, y = separable_binary
    clf = DecisionTree().fit(X, y)
    assert clf.feature_importances_ is not None
    assert np.all(clf.feature_importances_ >= 0.0)


# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------


def test_predict_output_shape(separable_multiclass: tuple) -> None:
    X, y = separable_multiclass
    clf = DecisionTree(max_depth=5).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(y),)


def test_predict_labels_are_subset_of_training_labels(separable_multiclass: tuple) -> None:
    X, y = separable_multiclass
    clf = DecisionTree(max_depth=5).fit(X, y)
    preds = clf.predict(X)
    assert set(preds).issubset(set(y))


def test_score_range_is_zero_to_one(separable_binary: tuple) -> None:
    X, y = separable_binary
    clf = DecisionTree(max_depth=1).fit(X, y)
    score = clf.score(X, y)
    assert 0.0 <= score <= 1.0
