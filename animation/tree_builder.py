"""
animation/tree_builder.py
--------------------------
Manim scene that animates the step-by-step *construction* of a decision tree.

Workflow (planned):
  1. Show the raw dataset as a 2-D scatter plot (first two features).
  2. Animate each axis-aligned split as a coloured vertical/horizontal line
     appearing in the order the CART algorithm discovered them (DFS).
  3. Transition to a tree-diagram view, growing nodes and edges in BFS order.
  4. Highlight leaf nodes with their predicted-class colour.

Requirements
------------
Manim Community Edition >= 0.18:
    pip install "decision-tree-edu[animation]"

On macOS also install system libraries:
    brew install cairo pango ffmpeg

Run with:
    manim -pql animation/tree_builder.py BuildTreeScene   # low quality, preview
    manim -pqh animation/tree_builder.py BuildTreeScene   # high quality
"""

from __future__ import annotations

# Manim is an optional dependency; guard the import so the rest of the project
# stays importable in environments without Manim installed.
try:
    from manim import (
        BLUE,
        DEGREES,
        DOWN,
        GREEN,
        LEFT,
        ORANGE,
        RED,
        RIGHT,
        UP,
        WHITE,
        YELLOW,
        AnimationGroup,
        Arrow,
        Axes,
        Create,
        Dot,
        FadeIn,
        FadeOut,
        Indicate,
        Line,
        Rectangle,
        Scene,
        Text,
        VGroup,
        Write,
    )

    _MANIM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MANIM_AVAILABLE = False
    Scene = object  # type: ignore[assignment,misc]


# Colour palette — maps integer class label → Manim colour constant
CLASS_COLOURS: dict[int, object] = {
    0: GREEN,   # Normal
    1: RED,     # Overheating
    2: ORANGE,  # Mechanical failure
    3: YELLOW,  # Electrical fault
}

# Human-readable class names matching CLASS_COLOURS keys
CLASS_NAMES: dict[int, str] = {
    0: "Normal",
    1: "Overheating",
    2: "Mech. failure",
    3: "Elec. fault",
}

# Depth-level colours for split lines drawn over the scatter plot
DEPTH_COLOURS: list[object] = [BLUE, GREEN, ORANGE, RED, YELLOW]


class NodeMobject(VGroup):  # type: ignore[misc]
    """A Manim ``VGroup`` that represents a single decision-tree node visually.

    Renders as a rounded rectangle containing either a split condition
    (internal node) or a predicted class label (leaf node).

    Parameters
    ----------
    label : str
        Text to display inside the node box (e.g. ``"vibration ≤ 125 Hz"``).
    color : Manim colour
        Fill colour for the rectangle background.
    width : float
        Width of the rectangle in Manim scene units (default 2.8).
    height : float
        Height of the rectangle in Manim scene units (default 0.8).
    font_size : int
        Font size for the label text (default 22).
    """

    def __init__(
        self,
        label: str,
        color: object = BLUE,  # type: ignore[assignment]
        width: float = 2.8,
        height: float = 0.8,
        font_size: int = 22,
    ) -> None:
        super().__init__()
        box = Rectangle(
            width=width,
            height=height,
            fill_color=color,  # type: ignore[arg-type]
            fill_opacity=0.85,
            stroke_color=WHITE,
            stroke_width=2,
        )
        text = Text(label, font_size=font_size, color=WHITE)
        text.move_to(box.get_center())
        self.add(box, text)


class BuildTreeScene(Scene):  # type: ignore[misc]
    """Manim scene: animate a decision tree being built split by split.

    Intended usage
    --------------
    Run this scene after setting ``self.clf`` to a fitted
    ``DecisionTree`` and ``self.X`` / ``self.y`` to the training data.
    A convenience wrapper script in ``examples/sensor_failure/run.py`` can
    supply these via a ``config`` dict or command-line ``--kwargs``.

    Animation stages
    ----------------
    1. ``_show_dataset``       — scatter plot of training data in 2-D feature space.
    2. ``_animate_splits``     — axis-aligned split lines appear DFS, colour-coded by depth.
    3. ``_build_tree_diagram`` — BFS layout of node + edge mobjects grows from root.
    4. ``_highlight_leaves``   — leaf nodes flash their class colour.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._node_mobjects: dict[int, NodeMobject] = {}
        self._leaf_mobjects: list[NodeMobject] = []

    def construct(self) -> None:
        """Entry point called by Manim — orchestrates the full animation."""
        import sys
        import os
        import numpy as np

        # Ensure the project root is importable when Manim is invoked directly
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)

        from examples.sensor_failure.dataset import FEATURE_NAMES, make_sensor_dataset
        from decision_tree.tree import DecisionTree

        X, y = make_sensor_dataset(n_samples=400, random_state=42)
        clf = DecisionTree(criterion="gini", max_depth=4, min_samples_split=10, min_samples_leaf=3)
        clf.fit(X, y)

        # Stage 1 — scatter plot
        axes = self._show_dataset(X, y, list(FEATURE_NAMES))
        self.wait(0.5)

        # Stage 2 — split lines DFS
        self._animate_splits(
            clf.root,
            axes,
            x_min=float(X[:, 0].min()),
            x_max=float(X[:, 0].max()),
            y_min=float(X[:, 1].min()),
            y_max=float(X[:, 1].max()),
        )
        self.wait(1)

        # Clear and transition to tree diagram
        self.play(FadeOut(*self.mobjects))  # type: ignore[arg-type]
        self.wait(0.3)

        # Stage 3 — tree diagram
        self._build_tree_diagram(clf.root, position=UP * 3.2)
        self.wait(0.5)

        # Stage 4 — highlight leaves
        self._highlight_leaves()
        self.wait(2)

    # ------------------------------------------------------------------
    # Stage helpers — implement these one by one
    # ------------------------------------------------------------------

    def _show_dataset(
        self,
        X: object,
        y: object,
        feature_names: list[str] | None = None,
    ) -> object:
        """Render a 2-D scatter plot of the training dataset.

        Parameters
        ----------
        X : ndarray of float, shape (n_samples, n_features)
            Feature matrix (only the first two columns are plotted).
        y : ndarray of int, shape (n_samples,)
            Class labels used to colour each dot via ``CLASS_COLOURS``.
        feature_names : list[str] | None
            Used for axis labels; falls back to ``"feature_0"`` / ``"feature_1"``.

        Returns
        -------
        Axes
            The Manim ``Axes`` object (needed by ``_animate_splits`` to map
            data coordinates to scene coordinates).

        Implementation sketch
        ---------------------
        axes = Axes(x_range=[...], y_range=[...], ...)
        self.play(Create(axes))
        for xi, label in zip(X, y):
            dot = Dot(axes.c2p(xi[0], xi[1]), color=CLASS_COLOURS[label])
            self.play(FadeIn(dot, run_time=0.02))
        """
        import numpy as np

        x0 = X[:, 0]  # type: ignore[index]
        x1 = X[:, 1]  # type: ignore[index]
        xn = (feature_names[0] if feature_names else "feature_0")
        yn = (feature_names[1] if feature_names else "feature_1")

        def _nice_range(arr: object) -> list[float]:
            lo = float(np.min(arr))  # type: ignore[arg-type]
            hi = float(np.max(arr))  # type: ignore[arg-type]
            pad = (hi - lo) * 0.12
            step = round((hi - lo) / 5, 1) or 1.0
            return [lo - pad, hi + pad, step]

        axes = Axes(
            x_range=_nice_range(x0),
            y_range=_nice_range(x1),
            x_length=7,
            y_length=5,
            axis_config={"color": WHITE, "include_tip": False},
        )
        x_label = Text(xn, font_size=20, color=WHITE).next_to(axes, DOWN, buff=0.25)
        y_label = (
            Text(yn, font_size=20, color=WHITE)
            .rotate(90 * DEGREES)
            .next_to(axes, LEFT, buff=0.3)
        )
        self.play(Create(axes), Write(x_label), Write(y_label))

        dots = VGroup(
            *[
                Dot(axes.c2p(float(xi[0]), float(xi[1])), color=CLASS_COLOURS.get(int(lbl), WHITE), radius=0.06)  # type: ignore[arg-type]
                for xi, lbl in zip(X, y)  # type: ignore[call-overload]
            ]
        )
        self.play(FadeIn(dots, run_time=1.5))
        return axes

    def _animate_splits(
        self,
        node: object,
        axes: object,
        depth: int = 0,
        x_min: float = -10.0,
        x_max: float = 10.0,
        y_min: float = -10.0,
        y_max: float = 10.0,
    ) -> None:
        """Recursively draw axis-aligned split lines for each internal node.

        Parameters
        ----------
        node : Node
            Current node (start with ``clf.root`` for the full tree).
        axes : Axes
            Manim Axes object returned by ``_show_dataset``.
        depth : int
            Current recursion depth (controls line colour from ``DEPTH_COLOURS``).
        x_min, x_max, y_min, y_max : float
            Current bounding box — narrows with each recursive call so lines
            only extend to the region actually partitioned by this node.

        Implementation sketch
        ---------------------
        if node.is_leaf: return
        color = DEPTH_COLOURS[depth % len(DEPTH_COLOURS)]
        if node.feature_index == 0:  # vertical split
            x_scene = axes.c2p(node.threshold, 0)[0]
            line = Line(axes.c2p(node.threshold, y_min),
                        axes.c2p(node.threshold, y_max), color=color)
            self.play(Create(line))
            self._animate_splits(node.left, axes, depth+1, x_min, node.threshold, y_min, y_max)
            self._animate_splits(node.right, axes, depth+1, node.threshold, x_max, y_min, y_max)
        else:  # horizontal split (feature_index == 1)
            ...
        """
        if node is None or node.is_leaf:  # type: ignore[union-attr]
            return
        color = DEPTH_COLOURS[depth % len(DEPTH_COLOURS)]
        fi = node.feature_index  # type: ignore[union-attr]
        thr = node.threshold  # type: ignore[union-attr]
        if fi == 0:  # vertical split line
            line = Line(
                axes.c2p(thr, y_min),  # type: ignore[union-attr]
                axes.c2p(thr, y_max),  # type: ignore[union-attr]
                color=color,
                stroke_width=2,
            )
            self.play(Create(line), run_time=0.4)
            self._animate_splits(node.left, axes, depth + 1, x_min, thr, y_min, y_max)  # type: ignore[union-attr]
            self._animate_splits(node.right, axes, depth + 1, thr, x_max, y_min, y_max)  # type: ignore[union-attr]
        elif fi == 1:  # horizontal split line
            line = Line(
                axes.c2p(x_min, thr),  # type: ignore[union-attr]
                axes.c2p(x_max, thr),  # type: ignore[union-attr]
                color=color,
                stroke_width=2,
            )
            self.play(Create(line), run_time=0.4)
            self._animate_splits(node.left, axes, depth + 1, x_min, x_max, y_min, thr)  # type: ignore[union-attr]
            self._animate_splits(node.right, axes, depth + 1, x_min, x_max, thr, y_max)  # type: ignore[union-attr]
        else:
            # Split on a feature not plotted — recurse into both children without drawing
            self._animate_splits(node.left, axes, depth + 1, x_min, x_max, y_min, y_max)  # type: ignore[union-attr]
            self._animate_splits(node.right, axes, depth + 1, x_min, x_max, y_min, y_max)  # type: ignore[union-attr]

    def _build_tree_diagram(
        self,
        node: object,
        position: object,
        h_spread: float = 3.5,
    ) -> None:
        """Place ``NodeMobject`` instances and connecting ``Arrow`` edges.

        Uses a simple recursive layout:
        * The current node is placed at ``position``.
        * The left child is placed at ``position + LEFT * h_spread + DOWN * 1.5``.
        * The right child is placed at ``position + RIGHT * h_spread + DOWN * 1.5``.
        * ``h_spread`` is halved at each level to prevent overlapping.

        Parameters
        ----------
        node : Node
            Current node (start with ``clf.root``).
        position : np.ndarray (3,)
            Scene coordinates for the centre of this node's ``NodeMobject``.
        h_spread : float
            Horizontal spread between left and right children (halved each level).

        Implementation sketch
        ---------------------
        label = f"x[{node.feature_index}] ≤ {node.threshold:.2f}" if not node.is_leaf
                else f"Class {node.value}"
        color = BLUE if not node.is_leaf else CLASS_COLOURS[node.value]
        mob = NodeMobject(label, color=color)
        mob.move_to(position)
        self._node_mobjects[id(node)] = mob
        self.play(FadeIn(mob))

        if not node.is_leaf:
            left_pos = position + LEFT * h_spread + DOWN * 1.5
            right_pos = position + RIGHT * h_spread + DOWN * 1.5
            # Draw arrows ...
            self._build_tree_diagram(node.left, left_pos, h_spread / 2)
            self._build_tree_diagram(node.right, right_pos, h_spread / 2)
        else:
            self._leaf_mobjects.append(mob)
        """
        if node is None:  # type: ignore[union-attr]
            return
        is_leaf: bool = node.is_leaf  # type: ignore[union-attr]
        if is_leaf:
            cls_idx = int(node.value)  # type: ignore[union-attr]
            cls_name = CLASS_NAMES.get(cls_idx, "")
            label = f"Class {cls_idx}\n{cls_name}"
            color = CLASS_COLOURS.get(int(node.value), BLUE)  # type: ignore[union-attr]
        else:
            label = f"x[{node.feature_index}] ≤ {node.threshold:.1f}"  # type: ignore[union-attr]
            color = BLUE
        mob = NodeMobject(label, color=color)
        mob.move_to(position)  # type: ignore[arg-type]
        self._node_mobjects[id(node)] = mob
        self.play(FadeIn(mob), run_time=0.35)

        if not is_leaf:
            left_pos = position + LEFT * h_spread + DOWN * 1.5  # type: ignore[operator]
            right_pos = position + RIGHT * h_spread + DOWN * 1.5  # type: ignore[operator]
            # Draw arrows from this node's bottom to children's top
            def _arrow(start_mob: NodeMobject, end_pos: object) -> Arrow:
                return Arrow(
                    start_mob.get_bottom(),
                    end_pos + UP * 0.4,  # type: ignore[operator]
                    buff=0,
                    color=WHITE,
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.15,
                )
            left_arrow = _arrow(mob, left_pos)
            right_arrow = _arrow(mob, right_pos)
            self.play(Create(left_arrow), Create(right_arrow), run_time=0.3)
            self._build_tree_diagram(node.left, left_pos, h_spread / 2)  # type: ignore[union-attr]
            self._build_tree_diagram(node.right, right_pos, h_spread / 2)  # type: ignore[union-attr]
        else:
            self._leaf_mobjects.append(mob)

    def _highlight_leaves(self) -> None:
        """Animate a colour flash on all collected leaf ``NodeMobject`` instances.

        Called after ``_build_tree_diagram`` populates ``self._leaf_mobjects``.

        Implementation sketch
        ---------------------
        animations = [Indicate(mob, color=WHITE, scale_factor=1.4)
                      for mob in self._leaf_mobjects]
        self.play(AnimationGroup(*animations, lag_ratio=0.15))
        """
        if not self._leaf_mobjects:
            return
        animations = [
            Indicate(mob, color=WHITE, scale_factor=1.3)  # type: ignore[arg-type]
            for mob in self._leaf_mobjects
        ]
        self.play(AnimationGroup(*animations, lag_ratio=0.2))
