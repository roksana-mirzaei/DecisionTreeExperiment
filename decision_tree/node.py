"""
decision_tree/node.py
---------------------
Defines the Node dataclass — the atomic building block of a decision tree.

Each node is either:
  * An **internal node**: holds a split rule (``feature_index``, ``threshold``)
    and two children (``left``, ``right``).
  * A **leaf node**: holds a predicted class label (``value``) and no children.

Keeping Node as a plain dataclass (no methods beyond properties) makes the
tree logic in tree.py easy to read and test in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Node:
    """A single node in a binary decision tree.

    Internal nodes store a split rule (``feature_index`` and ``threshold``).
    Leaf nodes store the predicted class label in ``value``.

    Attributes
    ----------
    feature_index : int | None
        Index of the feature column used for splitting.
        ``None`` for leaf nodes.
    threshold : float | None
        Numeric split boundary: samples with ``X[feature_index] <= threshold``
        go left; all others go right.  ``None`` for leaf nodes.
    left : Node | None
        Left child (samples satisfying the split condition).
    right : Node | None
        Right child (samples *not* satisfying the split condition).
    value : int | float | None
        Predicted class label (classification) or mean target value (regression)
        stored at a leaf node.  ``None`` for internal nodes.
    impurity : float
        Gini impurity or entropy at this node, computed from the training
        samples that reached it.  Used for feature importance calculations.
    n_samples : int
        Number of training samples that reached this node during fitting.
    depth : int
        Depth of this node in the tree (root = 0).
    """

    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional[Node] = None
    right: Optional[Node] = None
    value: Optional[float] = None
    impurity: float = 0.0
    n_samples: int = 0
    depth: int = 0



    @property
    def is_leaf(self) -> bool:
        """Return ``True`` when this node holds a class prediction (leaf)."""
        return self.value is not None

    def __repr__(self) -> str:  # pragma: no cover
        if self.is_leaf:
            return (
                f"Node(leaf, value={self.value}, "
                f"n_samples={self.n_samples}, depth={self.depth})"
            )
        return (
            # split is an internal node
            f"Node(split, feature={self.feature_index}, "
            f"threshold={self.threshold:.4f}, "
            f"n_samples={self.n_samples}, depth={self.depth})"
        )
