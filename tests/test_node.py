"""
tests/test_node.py
-------------------
Unit tests for the Node dataclass (decision_tree/node.py).

Covers:
  * ``is_leaf`` property behaviour for leaf vs. internal nodes.
  * Default field values.
  * ``__repr__`` output for both node types.
"""

from __future__ import annotations

import pytest

from decision_tree.node import Node


# ---------------------------------------------------------------------------
# is_leaf property
# ---------------------------------------------------------------------------


def test_leaf_node_is_leaf() -> None:
    """A node with ``value`` set must be identified as a leaf."""
    leaf = Node(value=1, n_samples=10)
    assert leaf.is_leaf is True


def test_internal_node_is_not_leaf() -> None:
    """A node with ``feature_index`` and ``threshold`` must not be a leaf."""
    node = Node(feature_index=0, threshold=5.0, n_samples=20)
    assert node.is_leaf is False


def test_default_node_is_not_leaf() -> None:
    """A freshly created Node with no arguments should not be a leaf."""
    node = Node()
    assert node.is_leaf is False


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_default_depth_is_zero() -> None:
    node = Node()
    assert node.depth == 0


def test_default_impurity_is_zero() -> None:
    node = Node()
    assert node.impurity == 0.0


def test_default_n_samples_is_zero() -> None:
    node = Node()
    assert node.n_samples == 0


def test_default_feature_and_threshold_are_none() -> None:
    node = Node()
    assert node.feature_index is None
    assert node.threshold is None


# ---------------------------------------------------------------------------
# Children linkage
# ---------------------------------------------------------------------------


def test_children_are_none_by_default() -> None:
    node = Node(feature_index=2, threshold=3.14)
    assert node.left is None
    assert node.right is None


def test_child_assignment() -> None:
    """Assigning child Nodes should update the parent's left/right fields."""
    parent = Node(feature_index=0, threshold=1.0, n_samples=10)
    left_child = Node(value=0, n_samples=5, depth=1)
    right_child = Node(value=1, n_samples=5, depth=1)

    parent.left = left_child
    parent.right = right_child

    assert parent.left is left_child
    assert parent.right is right_child
    assert parent.left.is_leaf is True
    assert parent.right.is_leaf is True


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_leaf_contains_leaf_keyword() -> None:
    leaf = Node(value=2, n_samples=5, depth=3)
    assert "leaf" in repr(leaf).lower()


def test_repr_leaf_contains_value() -> None:
    leaf = Node(value=2, n_samples=5, depth=3)
    assert "2" in repr(leaf)


def test_repr_internal_contains_split_keyword() -> None:
    node = Node(feature_index=1, threshold=3.14, n_samples=50, depth=1)
    assert "split" in repr(node).lower()


def test_repr_internal_contains_threshold() -> None:
    node = Node(feature_index=1, threshold=3.14, n_samples=50, depth=1)
    # repr uses 4 decimal places
    assert "3.1400" in repr(node)
