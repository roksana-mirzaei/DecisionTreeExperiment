# Decision Tree from Scratch

> **Author's note**
>
> This is a personal experiment to understand how decision trees work from the ground up — implementing the algorithm myself and comparing the results to scikit-learn's version.
>
> I used [GitHub Copilot](https://github.com/features/copilot) to help with boilerplate and speed up the process.
>
> Animations are built with the [Manim Community](https://docs.manim.community/) library.

---

An educational Python project for a hands-on mentorship session on Decision Trees.
The project has three layers of learning:

1. **Core algorithm** — a `DecisionTree` built from scratch (pure Python + NumPy, no scikit-learn for the logic) following CART (Classification and Regression Trees) principles.
2. **Manim animations** — step-by-step visual scenes showing how the tree is *built* (recursive splits) and how it *predicts* (path from root to leaf).
3. **Real-world example** — classifying industrial sensor failures in a robotics context (overheating, mechanical faults, electrical faults, normal operation).

---

## Project structure

```
DecisionTree/
├── decision_tree/              # Core algorithm — no ML-library deps
│   ├── __init__.py             # Public re-exports
│   ├── node.py                 # Node dataclass (internal + leaf)
│   ├── splitter.py             # Gini / entropy / information-gain / best_split
│   ├── tree.py                 # DecisionTree (fit / predict / score)
│   └── utils.py                # Helpers: most_common_label, train_test_split, print_tree
│
├── examples/
│   ├── sensor_failure/         # Robotics sensor-failure classification
│   │   ├── dataset.py          # Synthetic multi-variate Gaussian dataset generator
│   │   └── run.py              # End-to-end train → evaluate → visualise pipeline
│   └── cancer/                 # Breast Cancer Wisconsin — our DT vs sklearn comparison
│       ├── dataset.py          # Loads sklearn breast_cancer dataset
│       └── run.py              # Side-by-side comparison: our DT vs sklearn DT
│
├── animation/                  # Manim scenes (optional dependency)
│   ├── tree_builder.py         # Scene: animate the tree being built split-by-split
│   └── predictor.py            # Scene: animate a single prediction traversal
│
├── tests/                      # pytest test suite
│   ├── test_node.py
│   ├── test_splitter.py
│   └── test_tree.py
│
├── pyproject.toml              # Build metadata + tool config (ruff, mypy, pytest)
├── requirements.txt            # Pinned dependencies for direct pip installs
└── README.md
```

---

## Setup & running

### Prerequisites

- Python **3.11+**
- [pip](https://pip.pypa.io/)

### 1 — Create a virtual environment

```bash
cd DecisionTree
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2 — Install dependencies

```bash
pip install numpy pytest pytest-cov ruff mypy
pip install -e . --no-build-isolation
```

### 3 — Run the sensor-failure example

```bash
python -m examples.sensor_failure.run
```

Full output is saved to `results/sensor_failure_results.txt` and the scatter plot to `results/sensor_failure_plot.png`.

### 4 — Run the cancer vs sklearn comparison

Requires scikit-learn (`pip install scikit-learn`).

```bash
python -m examples.cancer.run
```

This trains the **same hyperparameters** on the Breast Cancer Wisconsin dataset
(569 samples, 30 features, binary: Malignant / Benign) using both our
`DecisionTree` and `sklearn.tree.DecisionTreeClassifier`.

Full output is saved to `results/cancer_results.txt` and the probability plot to `results/cancer_plot.png`.

### 5 — Run the tests

```bash
pytest                          # run all 53 tests
pytest -v                       # verbose output per test
pytest --cov=decision_tree      # with coverage report
```

### 5 — Install animation dependencies (optional)

The Manim scenes require [Manim Community Edition](https://docs.manim.community/):

```bash
pip install manim
# macOS also needs: brew install cairo pango ffmpeg
```

### 6 — Run the Manim animations (once animation stubs are implemented)

```bash
manim -pql animation/tree_builder.py BuildTreeScene   # low quality, preview
manim -pqh animation/tree_builder.py BuildTreeScene   # high quality

manim -pql animation/predictor.py PredictScene
```

| Flag | Meaning |
|------|---------|
| `-p` | Open rendered video automatically |
| `-ql` | Low quality — fast, for development |
| `-qh` | High quality — for presentations |

---

## How it works

### CART algorithm (brief)

1. **Splitting criterion** — at each node, find the `(feature, threshold)` pair that maximises *information gain* relative to either Gini impurity or Shannon entropy.
2. **Recursive growth** — partition the dataset and recurse on each half until a stopping condition is met (max depth, minimum samples, or a pure node).
3. **Prediction** — traverse from root to a leaf following the split conditions; return the majority class stored in the leaf.
4. **Feature importance** — accumulated as the impurity-weighted gain at each split, then normalised to sum to 1.

### Manim animation design

- `BuildTreeScene` — plots the raw dataset, then replays splits as coloured axis-aligned lines, and transitions to a growing tree diagram.
- `PredictScene` — renders the full tree statically, highlights the active node at each step, and moves an arrow along the chosen edge until the leaf is reached.

---

## Key concepts covered in this session

| Concept | Where |
|---------|-------|
| CART algorithm | `decision_tree/tree.py` |
| Gini impurity / entropy | `decision_tree/splitter.py` |
| Recursive tree construction | `DecisionTree._grow_tree` |
| Feature importance | `DecisionTree.feature_importances_` |
| Dataclass design | `decision_tree/node.py` |
| Manim scene structure | `animation/` |
| Composition over inheritance | `NodeMobject` uses `VGroup` composition |
| Type hints + PEP 8 | Throughout the codebase |
