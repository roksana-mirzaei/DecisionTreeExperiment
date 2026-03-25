"""
decision_tree/utils.py
-----------------------
Lightweight utility functions shared across the decision_tree package.

Contains:
  * ``most_common_label`` — majority-vote class label from a label array.
  * ``train_test_split``  — reproducible random split into train/test sets.
  * ``print_tree``        — pretty-print a fitted tree to stdout (for debugging
                            and live demos during a mentorship session).

Intentionally thin: all heavy numerical work lives in splitter.py and tree.py.
Only NumPy is used — no additional dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .node import Node


def most_common_label(y: NDArray[np.int_]) -> int:
    """Return the most frequent class label in *y*.

    In the case of a tie, the numerically smallest label wins — ensuring
    deterministic behaviour without requiring an additional random seed.

    Parameters
    ----------
    y : ndarray of int, shape (n_samples,)
        Array of integer class labels.

    Returns
    -------
    int
        The majority class label.

    Examples
    --------
    >>> import numpy as np
    >>> most_common_label(np.array([0, 1, 1, 2, 1]))
    1
    >>> most_common_label(np.array([0, 0, 1, 1]))  # tie → smallest label
    0
    """
    values, counts = np.unique(y, return_counts=True)
    return int(values[np.argmax(counts)])


def train_test_split(
    X: NDArray[np.float64],
    y: NDArray[np.int_],
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int_],
    NDArray[np.int_],
]:
    """Randomly split arrays into train and test subsets.

    A lightweight replacement for ``sklearn.model_selection.train_test_split``
    that keeps this project free of external ML-library dependencies.

    Parameters
    ----------
    X : ndarray of float, shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of int, shape (n_samples,)
        Class labels.
    test_size : float
        Proportion of the dataset to include in the test split (default 0.2).
        Must be in (0, 1).
    random_state : int | None
        Seed for the NumPy random number generator.  Pass an integer for
        reproducible splits (e.g. in examples and tests).

    Returns
    -------
    X_train, X_test, y_train, y_test : four ndarrays
        Training and test subsets, in that order.
    """
    rng = np.random.default_rng(random_state)
    n = len(y)
    indices = rng.permutation(n)
    split = int(n * (1.0 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def print_tree(
    node: Node,
    feature_names: list[str] | None = None,
    class_names: list[str] | None = None,
    indent: str = "",
) -> None:
    """Pretty-print a fitted decision tree to stdout.

    Useful for quick inspection during development and live demos.
    The output mirrors the tree structure as an indented ASCII diagram.

    Parameters
    ----------
    node : Node
        Root of the (sub-)tree to print.  Typically ``clf.root``.
    feature_names : list[str] | None
        Human-readable names for each feature column.  Falls back to
        ``"feature_<i>"`` when ``None``.
    class_names : list[str] | None
        Human-readable names for each class label.  When provided, leaf
        nodes print ``class <i> (ClassName)`` instead of just the number.
    indent : str
        Indentation prefix — used internally for recursive calls.

    Examples
    --------
    >>> print_tree(clf.root, feature_names=["vibration_hz", "temperature_c"])
    [vibration_hz <= 125.0000]  (n=480, impurity=0.7500)
    ├── True:
    │   [temperature_c <= 60.0000]  (n=240, impurity=0.5000)
    │   ├── True:
    │   │   └── [LEAF] Predict class 0 (Normal)  (n=120)
    ...
    """
    if node.is_leaf:
        label = int(node.value)  # type: ignore[arg-type]
        name = f" ({class_names[label]})" if class_names else ""
        print(f"{indent}└── [LEAF] Predict class {label}{name}  (n={node.n_samples})")
        return

    fname = (
        feature_names[node.feature_index]  # type: ignore[index]
        if feature_names
        else f"feature_{node.feature_index}"
    )
    print(
        f"{indent}[{fname} <= {node.threshold:.4f}]"
        f"  (n={node.n_samples}, impurity={node.impurity:.4f})"
    )
    if node.left:
        print(f"{indent}├── True:")
        print_tree(node.left, feature_names, class_names, indent + "│   ")
    if node.right:
        print(f"{indent}└── False:")
        print_tree(node.right, feature_names, class_names, indent + "    ")
