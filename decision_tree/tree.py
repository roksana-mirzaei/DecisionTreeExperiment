"""
decision_tree/tree.py
---------------------
DecisionTree — a single estimator that handles both classification and regression.

Task is selected via ``criterion``:
  * ``'gini'`` or ``'entropy'`` → classification  (leaf stores majority class label)
  * ``'variance'``             → regression       (leaf stores mean target value)

Trains using the CART algorithm by recursively calling
``decision_tree.splitter.best_split`` and building a tree of ``Node`` objects.
Predicts by traversing the grown tree from root to a leaf.

Designed for *clarity over raw speed*: the recursive Python implementation
makes each algorithmic step explicit and easy to follow during a mentorship
session.  NumPy is used for array operations; no scikit-learn in the core logic.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .node import Node
from .splitter import best_split, entropy, gini_impurity, variance_reduction
from .utils import most_common_label


class DecisionTree:
    """Binary decision tree for classification **and** regression.

    The task is determined by ``criterion``:

    * **Classification** — ``criterion='gini'`` (default) or ``'entropy'``.
      Leaf nodes store the majority class label.  Use :meth:`predict` to get
      integer class labels and :meth:`score` for accuracy.

    * **Regression** — ``criterion='variance'``.
      Leaf nodes store the mean target value.  Use :meth:`predict` to get
      continuous predictions and :meth:`score` for R² score.

    Parameters
    ----------
    criterion : {'gini', 'entropy', 'variance'}
        Impurity function used to evaluate split quality.
        ``'gini'`` / ``'entropy'`` → classification.
        ``'variance'`` → regression (minimises MSE).
    max_depth : int | None
        Maximum depth of the tree.  ``None`` allows unlimited growth.
    min_samples_split : int
        Minimum samples a node must have to be considered for splitting.
    min_samples_leaf : int
        Minimum samples required in *each* child after a split.
    n_features : int | None
        Features randomly sampled per split.  ``None`` uses all features.
        Set to ``int(sqrt(p))`` for Random-Forest-style subsampling.

    Attributes
    ----------
    root : Node | None
        Root node of the fitted tree.  ``None`` before :meth:`fit`.
    n_classes_ : int
        Number of unique class labels (classification only; 0 for regression).
    feature_importances_ : ndarray of float, shape (n_features,)
        Relative importance of each feature (sums to 1.0).

    Examples
    --------
    Classification:

    >>> import numpy as np
    >>> from decision_tree import DecisionTree
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = DecisionTree(criterion='gini', max_depth=2).fit(X, y)
    >>> clf.predict(np.array([[2, 3], [6, 7]]))
    array([0, 1])

    Regression:

    >>> y_cont = np.array([1.2, 1.5, 8.1, 8.4])
    >>> reg = DecisionTree(criterion='variance', max_depth=2).fit(X, y_cont)
    >>> reg.predict(np.array([[2, 3], [6, 7]]))
    array([1.35, 8.25])
    """

    def __init__(
        self,
        criterion: Literal["gini", "entropy", "variance"] = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_features: int | None = None,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features

        # Set after fit()
        self.root: Node | None = None
        self.n_classes_: int = 0
        self.feature_importances_: NDArray[np.float64] | None = None
        self._n_total_features: int = 0


    @property
    def _is_regression(self) -> bool:
        """Return True when the tree is in regression mode (criterion='variance')."""
        return self.criterion == "variance"

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> DecisionTree:
        """Grow the decision tree from training data.

        Parameters
        ----------
        X : ndarray of float, shape (n_samples, n_features)
            Feature matrix.  Converted to float64 internally.
        y : ndarray of int or float, shape (n_samples,)
            Integer class labels for classification, continuous values for regression.

        Returns
        -------
        self
            Fitted estimator — enables method chaining (``clf.fit(X, y).predict(X_test)``).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # n_classes_ is only meaningful for classification
        self.n_classes_ = 0 if self._is_regression else int(np.unique(y).size)
        self._n_total_features = X.shape[1]
        self.feature_importances_ = np.zeros(self._n_total_features, dtype=np.float64)

        self.root = self._grow_tree(X, y, depth=0)

        # Normalise importances so they sum to 1.0
        total = float(self.feature_importances_.sum())
        if total > 0.0:
            self.feature_importances_ /= total

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict targets for each sample in *X*.

        Parameters
        ----------
        X : ndarray of float, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray, shape (n_samples,)
            Integer class labels for classification, float values for regression.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self.root is None:
            raise RuntimeError("This DecisionTree is not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        predictions = np.array([self._traverse(x, self.root) for x in X], dtype=np.float64)
        # Return int array for classification so callers get clean label arrays
        if not self._is_regression:
            return predictions.astype(np.int_)  # type: ignore[return-value]
        return predictions

    def score(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> float:
        """Return a goodness-of-fit score on ``(X, y)``.

        * **Classification** — returns mean accuracy (fraction correct), range [0, 1].
        * **Regression**     — returns R² score (1.0 = perfect fit, 0.0 = predicts mean).

        Parameters
        ----------
        X : ndarray of float, shape (n_samples, n_features)
        y : ndarray of int or float, shape (n_samples,)
            True targets.

        Returns
        -------
        float
            Accuracy for classification; R² for regression.
        """
        y = np.asarray(y, dtype=np.float64)
        preds = np.asarray(self.predict(X), dtype=np.float64)
        if self._is_regression:
            # R² = 1 - SS_res / SS_tot
            ss_res = float(np.sum((y - preds) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
        return float(np.mean(preds == y))


    def _leaf_value(self, y: NDArray[np.float64]) -> float:
        """Compute the value stored at a leaf node.

        * Classification → majority class label (integer cast to float).
        * Regression     → mean of target values.
        """
        if self._is_regression:
            return float(np.mean(y))
        return float(most_common_label(y.astype(np.int_)))

    def _current_impurity(self, y: NDArray[np.float64]) -> float:
        """Compute impurity / variance at a node using the configured criterion."""
        if self.criterion == "gini":
            return gini_impurity(y.astype(np.int_))
        if self.criterion == "entropy":
            return entropy(y.astype(np.int_))
        return variance_reduction(y)  # 'variance'

    def _grow_tree(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        depth: int,
    ) -> Node:
        """Recursively build the tree and return the root of the sub-tree.

        Stops growing (creates a leaf) when any stopping criterion is met:
        * ``depth == max_depth``
        * All targets in *y* are identical (pure node).
        * Fewer than ``min_samples_split`` samples remain.
        * No split produces positive impurity decrease.
        * Either child would have fewer than ``min_samples_leaf`` samples.

        Parameters
        ----------
        X : ndarray of float, shape (n_samples, n_features)
            Feature matrix for the samples reaching this node.
        y : ndarray of float, shape (n_samples,)
            Targets (class labels for classification, continuous for regression).
        depth : int
            Current depth (root call uses 0).

        Returns
        -------
        Node
            Root of the (sub-)tree grown from the given data.
        """
        n_samples = len(y)
        current_imp = self._current_impurity(y)

        # ---- Stopping conditions ----
        hit_max_depth = self.max_depth is not None and depth >= self.max_depth
        too_few = n_samples < self.min_samples_split
        is_pure = int(np.unique(y).size) == 1

        if hit_max_depth or too_few or is_pure:
            return Node(
                value=self._leaf_value(y),
                impurity=current_imp,
                n_samples=n_samples,
                depth=depth,
            )

        # ---- Find best split ----
        feat_idx, threshold, gain = best_split(X, y, self.criterion, self.n_features)

        if gain == 0.0:
            # No beneficial split; collapse to leaf
            return Node(
                value=self._leaf_value(y),
                impurity=current_imp,
                n_samples=n_samples,
                depth=depth,
            )

        # ---- Partition ----
        left_mask = X[:, feat_idx] <= threshold
        right_mask = ~left_mask

        # Enforce min_samples_leaf on both sides
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return Node(
                value=self._leaf_value(y),
                impurity=current_imp,
                n_samples=n_samples,
                depth=depth,
            )

        # Accumulate weighted impurity decrease for feature importance
        self.feature_importances_[feat_idx] += (  # type: ignore[index]
            (n_samples / (n_samples or 1)) * gain
        )

        return Node(
            feature_index=feat_idx,
            threshold=threshold,
            left=self._grow_tree(X[left_mask], y[left_mask], depth + 1),
            right=self._grow_tree(X[right_mask], y[right_mask], depth + 1),
            impurity=current_imp,
            n_samples=n_samples,
            depth=depth,
        )

    # Private — prediction

    def _traverse(self, x: NDArray[np.float64], node: Node) -> float:
        """Recursively walk the tree for a single sample.

        Parameters
        ----------
        x : ndarray of float, shape (n_features,)
            A single feature vector.
        node : Node
            Current node (initial call uses ``self.root``).

        Returns
        -------
        float
            Leaf value: majority class label (classification) or
            mean target value (regression).
        """
        if node.is_leaf:
            return float(node.value)  # type: ignore[arg-type]
        assert node.left is not None and node.right is not None  # always true for internal nodes
        if x[node.feature_index] <= node.threshold:  # type: ignore[index]
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def __repr__(self) -> str:  # pragma: no cover
        task = "regression" if self._is_regression else "classification"
        return (
            f"DecisionTree("
            f"criterion='{self.criterion}', task='{task}', "
            f"max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"min_samples_leaf={self.min_samples_leaf})"
        )
