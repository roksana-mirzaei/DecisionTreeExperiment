"""
decision_tree/splitter.py -> it has one job - find the place to cut the data at each node.
--------------------------
Pure functions for computing impurity metrics and finding optimal splits.

Supports both classification and regression tasks via three criteria:

  Classification:
    * ``gini_impurity``      — Gini impurity  (default, fast).
    * ``entropy``            — Shannon entropy (information gain).

  Regression:
    * ``variance_reduction`` — Reduction in target variance (MSE-based).

  Shared:
    * ``information_gain``   — Weighted impurity/variance decrease for any criterion.
    * ``best_split``         — Scans all (feature, threshold) pairs to find the one
                               that maximises the chosen criterion's score.

All functions are stateless and side-effect-free, making them straightforward
to unit-test and reason about independently of the tree structure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray



# ---------------------------------------------------------------------------
# Impurity criteria — classification
# ---------------------------------------------------------------------------


def gini_impurity(y: NDArray[np.int_]) -> float:
    """Compute the Gini impurity of a label array.

    .. math::

        G = 1 - \\sum_{k=1}^{K} p_k^2

    A value of 0 means the node is perfectly pure (all samples share one label).
    For a balanced binary node the maximum is 0.5.

    Parameters
    ----------
    y : ndarray of int, shape (n_samples,)
        Integer class labels for the samples at this node.

    Returns
    -------
    float
        Gini impurity in the range [0, 1 - 1/K].
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return float(1.0 - np.sum(probs**2))


# ---------------------------------------------------------------------------


def entropy(y: NDArray[np.int_]) -> float:
    """Compute the Shannon entropy of a label array.

    .. math::

        H = -\\sum_{k=1}^{K} p_k \\log_2 p_k

    A value of 0 means the node is perfectly pure.
    Maximum is :math:`\\log_2 K` for a uniform distribution over K classes.

    Parameters
    ----------
    y : ndarray of int, shape (n_samples,)
        Integer class labels for the samples at this node.

    Returns
    -------
    float
        Entropy in bits (range [0, log2(n_classes)]).
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    # np.where avoids log(0) without silencing the NumPy warning globally
    log_probs = np.where(probs > 0.0, np.log2(probs), 0.0)
    return float(-np.sum(probs * log_probs))


# ---------------------------------------------------------------------------
# Impurity criterion — regression
# ---------------------------------------------------------------------------


def variance_reduction(y: NDArray[np.float64]) -> float:
    """Compute the variance of a continuous target array.

    Used as the impurity measure for **regression** trees.  A node with
    variance 0 is perfectly pure (all targets identical).  The split that
    reduces variance the most is chosen — equivalent to minimising MSE.

    .. math::

        V = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\bar{y})^2

    Parameters
    ----------
    y : ndarray of float, shape (n_samples,)
        Continuous target values at this node.

    Returns
    -------
    float
        Variance of *y*.  Returns 0.0 for empty or single-element arrays.
    """
    if len(y) < 2:
        return 0.0
    return float(np.var(y))


# ---------------------------------------------------------------------------
# Shared — works for classification (gini / entropy) and regression (variance)
# ---------------------------------------------------------------------------


def information_gain(
    y_parent: NDArray[np.float64],
    y_left: NDArray[np.float64],
    y_right: NDArray[np.float64],
    criterion: str = "gini",
) -> float:
    """Compute the impurity decrease (information gain) produced by a binary split.

    Works for **classification** (gini / entropy) and **regression** (variance).

    .. math::

        IG = H(\\text{parent})
             - \\frac{|L|}{|P|}\\,H(L)
             - \\frac{|R|}{|P|}\\,H(R)

    where *H* is :func:`gini_impurity`, :func:`entropy`, or
    :func:`variance_reduction` depending on ``criterion``.

    Parameters
    ----------
    y_parent : ndarray
        Targets/labels of all samples *before* the split.
    y_left : ndarray
        Targets/labels routed to the left child (feature ≤ threshold).
    y_right : ndarray
        Targets/labels routed to the right child (feature > threshold).
    criterion : {'gini', 'entropy', 'variance'}
        Impurity measure to use.
        Use ``'gini'`` or ``'entropy'`` for classification;
        use ``'variance'`` for regression.

    Returns
    -------
    float
        Non-negative impurity decrease.  Higher values indicate better splits.

    Raises
    ------
    ValueError
        If ``criterion`` is not one of ``'gini'``, ``'entropy'``, ``'variance'``.
    """
    if criterion == "gini":
        impurity_fn = gini_impurity  # type: ignore[assignment]
    elif criterion == "entropy":
        impurity_fn = entropy  # type: ignore[assignment]
    elif criterion == "variance":
        impurity_fn = variance_reduction  # type: ignore[assignment]
    else:
        raise ValueError(
            f"Unknown criterion '{criterion}'. "
            "Expected 'gini' or 'entropy' (classification) "
            "or 'variance' (regression)."
        )

    n = len(y_parent)
    if n == 0:
        return 0.0

    parent_impurity = impurity_fn(y_parent)
    left_weight = len(y_left) / n
    right_weight = len(y_right) / n
    weighted_children = (
        left_weight * impurity_fn(y_left) + right_weight * impurity_fn(y_right)
    )
    return parent_impurity - weighted_children



def best_split(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    criterion: str = "gini",
    n_features: int | None = None,
) -> tuple[int, float, float]:
    """Find the (feature, threshold) pair that maximises impurity decrease.

    Works for both **classification** (``criterion='gini'`` or ``'entropy'``)
    and **regression** (``criterion='variance'``).

    Considers all unique midpoint thresholds derived from the sorted unique
    values of each feature.  Candidate thresholds are the midpoints between
    consecutive distinct values — this is the standard CART approach.

    Parameters
    ----------
    X : ndarray of float, shape (n_samples, n_features)
        Feature matrix for the samples at the current node.
    y : ndarray of float or int, shape (n_samples,)
        Target values (integer class labels for classification,
        continuous values for regression).
    criterion : {'gini', 'entropy', 'variance'}
        Impurity criterion forwarded to :func:`information_gain`.
        Use ``'gini'`` or ``'entropy'`` for classification;
        use ``'variance'`` for regression.
    n_features : int | None
        Number of features to randomly sample for consideration at this split.
        ``None`` (default) considers all features (standard decision tree).
        Set to ``int(sqrt(total_features))`` for Random Forest behaviour.

    Returns
    -------
    best_feature : int
        Column index of the best splitting feature.
    best_threshold : float
        Threshold value that maximises the gain.
    best_gain : float
        Information gain achieved by this split.
        Returns ``(0, 0.0, 0.0)`` when no beneficial split is found.
    """
    _, total_features = X.shape
    rng = np.random.default_rng()  # unseeded for randomness; pass seed explicitly if needed

    feature_indices: NDArray[np.intp] = (
        rng.choice(total_features, size=n_features, replace=False)
        if n_features is not None
        else np.arange(total_features, dtype=np.intp)
    )

    best_gain = -np.inf
    best_feature = 0
    best_threshold = 0.0

    for feat_idx in feature_indices:
        column = X[:, feat_idx]
        sorted_unique = np.unique(column)

        if len(sorted_unique) < 2:
            # All values identical — no useful split possible for this feature
            continue

        # Candidate thresholds: midpoints between consecutive unique values
        thresholds = (sorted_unique[:-1] + sorted_unique[1:]) / 2.0

        for threshold in thresholds:
            left_mask = column <= threshold
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            gain = information_gain(y, y[left_mask], y[right_mask], criterion)
            if gain > best_gain:
                best_gain = gain
                best_feature = int(feat_idx)
                best_threshold = float(threshold)

    return best_feature, best_threshold, max(float(best_gain), 0.0)
