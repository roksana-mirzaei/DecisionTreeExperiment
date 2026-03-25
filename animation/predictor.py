"""
animation/predictor.py
-----------------------
Manim scene that animates how a trained decision tree classifies a *single*
unseen sample: a highlighted marker travels from root to leaf, pausing at
each node to show the split condition being evaluated against the sample's
feature values.

Requirements
------------
Manim Community Edition >= 0.18:
    pip install "decision-tree-edu[animation]"

Run with:
    manim -pql animation/predictor.py PredictScene   # low quality, preview
    manim -pqh animation/predictor.py PredictScene   # high quality
"""

from __future__ import annotations

# Guard the import so the rest of the project stays importable without Manim.
try:
    from manim import (
        GREEN,
        RED,
        WHITE,
        YELLOW,
        DOWN,
        LEFT,
        RIGHT,
        UP,
        Arrow,
        FadeIn,
        FadeOut,
        Flash,
        Indicate,
        Scene,
        SurroundingRectangle,
        Text,
        VGroup,
        Write,
    )

    _MANIM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MANIM_AVAILABLE = False
    Scene = object  # type: ignore[assignment,misc]

from animation.tree_builder import CLASS_COLOURS, NodeMobject


class PredictScene(Scene):  # type: ignore[misc]
    """Manim scene: animate a step-by-step prediction traversal.

    For each node visited along the root-to-leaf path:
    1. Pulse the active node with ``Indicate``.
    2. Display the split condition and the sample's feature value.
    3. Flash ``TRUE`` (green) or ``FALSE`` (red) next to the condition.
    4. Move an ``Arrow`` marker along the chosen edge to the next node.
    5. At the leaf, flash the predicted class label prominently.

    Intended usage
    --------------
    Set ``self.clf`` to a fitted ``DecisionTree`` and
    ``self.sample`` to a 1-D NumPy array of feature values before Manim
    calls ``construct``.  Typically done via a wrapper script or Manim's
    ``--kwargs`` CLI argument.
    """

    def __init__(self, **kwargs: object) -> None:
        # TODO: Accept clf, sample, and optional feature / class names:
        #
        #   self.clf           = kwargs.pop("clf", None)
        #   self.sample        = kwargs.pop("sample", None)
        #   self.feature_names = kwargs.pop("feature_names", None)
        #   self.class_names   = kwargs.pop("class_names", None)
        super().__init__(**kwargs)
        # node-id → NodeMobject mapping populated by _render_tree
        self._node_map: dict[int, NodeMobject] = {}

    def construct(self) -> None:
        """Entry point — orchestrates the full prediction animation.

        Planned stages:
        1. ``_render_tree``      — draw the static tree diagram.
        2. ``_show_sample``      — display the sample's feature vector on screen.
        3. ``_traverse_animate`` — walk root-to-leaf with decision animations.
        4. ``_show_result``      — reveal the predicted class.
        """
        # TODO: Replace ``raise`` with stage calls once helpers are implemented:
        #
        #   self._render_tree(self.clf.root, position=UP * 3)
        #   self._show_sample(self.sample, self.feature_names or [])
        #   self._traverse_animate(self.clf.root, self.sample)
        #   prediction = self.clf.predict(self.sample.reshape(1, -1))[0]
        #   class_name = (self.class_names or {})[prediction] or str(prediction)
        #   self._show_result(prediction, class_name)
        raise NotImplementedError(
            "PredictScene.construct() is a stub — implement stage helpers first."
        )

    # ------------------------------------------------------------------
    # Stage helpers — implement these one by one
    # ------------------------------------------------------------------

    def _render_tree(
        self,
        node: object,
        position: object,
        h_spread: float = 3.5,
    ) -> None:
        """Build and display a static tree diagram; populate ``self._node_map``.

        Re-uses the same recursive BFS layout as ``BuildTreeScene._build_tree_diagram``,
        but without step-by-step animation — the full tree appears at once so
        the viewer can focus on the traversal path.

        Parameters
        ----------
        node : Node
            Root of the tree (typically ``clf.root``).
        position : np.ndarray (3,)
            Scene coordinate for the root ``NodeMobject``.
        h_spread : float
            Initial horizontal spread between children (halved per level).

        Implementation sketch
        ---------------------
        label = f"x[{node.feature_index}] ≤ {node.threshold:.2f}"
                if not node.is_leaf else f"Class {node.value}"
        mob = NodeMobject(label)
        mob.move_to(position)
        self._node_map[id(node)] = mob
        self.add(mob)                     # add without animation for static render
        if not node.is_leaf:
            # draw arrows + recurse for left and right children
            ...
        """
        # TODO: Implement static tree layout.
        raise NotImplementedError

    def _show_sample(self, sample: object, feature_names: list[str]) -> None:
        """Display the incoming sample's feature vector in a corner table.

        Parameters
        ----------
        sample : ndarray of float, shape (n_features,)
            Feature values of the sample being classified.
        feature_names : list[str]
            Human-readable names for each feature column.

        Implementation sketch
        ---------------------
        lines = [Text(f"{name}: {val:.2f}", font_size=18)
                 for name, val in zip(feature_names, sample)]
        table = VGroup(*lines).arrange(DOWN, aligned_edge=LEFT)
        table.to_corner(DL)
        self.play(FadeIn(table))
        """
        # TODO: Implement feature-vector display.
        raise NotImplementedError

    def _traverse_animate(
        self,
        node: object,
        sample: object,
    ) -> None:
        """Recursively walk the tree, animating each decision step.

        At each internal node:
        1. ``Indicate`` the current ``NodeMobject`` with a pulsing outline.
        2. Show the split condition text: e.g. ``"vibration_hz (210) ≤ 125?"``
        3. Flash a GREEN ``TRUE`` or RED ``FALSE`` label.
        4. Move a small ``Arrow`` marker along the correct child edge.
        5. Recurse into the chosen child node.

        Parameters
        ----------
        node : Node
            Current node (initial call with ``clf.root``).
        sample : ndarray of float, shape (n_features,)
            Feature values of the sample being classified.

        Implementation sketch
        ---------------------
        mob = self._node_map[id(node)]
        self.play(Indicate(mob, scale_factor=1.3))
        if node.is_leaf: return

        val = sample[node.feature_index]
        goes_left = val <= node.threshold
        condition_text = Text(
            f"x[{node.feature_index}]={val:.2f} ≤ {node.threshold:.2f}?",
            font_size=20,
        ).next_to(mob, DOWN * 0.4)
        self.play(Write(condition_text))

        result = Text("TRUE" if goes_left else "FALSE",
                      color=GREEN if goes_left else RED, font_size=22)
        result.next_to(condition_text, DOWN * 0.4)
        self.play(FadeIn(result))
        self.play(FadeOut(condition_text), FadeOut(result))

        next_node = node.left if goes_left else node.right
        self._traverse_animate(next_node, sample)
        """
        # TODO: Implement animated traversal.
        raise NotImplementedError

    def _show_result(self, predicted_class: int, class_name: str) -> None:
        """Display the final prediction with a celebratory animation.

        Parameters
        ----------
        predicted_class : int
            Integer class label returned by the tree.
        class_name : str
            Human-readable name for the class (e.g. ``"Mechanical failure"``).

        Implementation sketch
        ---------------------
        result = Text(f"Predicted: {class_name}", font_size=36,
                      color=CLASS_COLOURS.get(predicted_class, WHITE))
        result.to_edge(DOWN)
        self.play(Write(result))
        self.play(Flash(result, color=CLASS_COLOURS.get(predicted_class, WHITE),
                        line_length=0.3, num_lines=12))
        self.wait(2)
        """
        # TODO: Implement result-reveal animation.
        raise NotImplementedError
