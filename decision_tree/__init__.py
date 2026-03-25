"""
decision_tree/__init__.py
--------------------------
Public API for the decision_tree package.

Re-exports the two symbols most commonly needed by callers, so users can write:

    from decision_tree import DecisionTree, Node

All implementation details live in the submodules (node, splitter, tree, utils).
"""

from decision_tree.node import Node
from decision_tree.tree import DecisionTree

__all__ = ["DecisionTree", "Node"]
__version__ = "0.1.0"
