# Decision Tree from Scratch

> **Author's note** — This is a personal experiment to understand decision trees by implementing the algorithm from scratch and comparing it against scikit-learn. Built with help from [GitHub Copilot](https://github.com/features/copilot). Animations use [Manim Community](https://docs.manim.community/).

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

## Run

```bash
# Sensor failure example
python -m examples.sensor_failure.run
# → results/sensor_failure_results.txt, results/sensor_failure_plot.png

# Cancer vs sklearn comparison
python -m examples.cancer.run
# → results/cancer_results.txt, results/cancer_plot.png

# Tests
pytest

# Animation (requires: brew install cairo pango ffmpeg)
manim -pql animation/tree_builder.py BuildTreeScene
manim -pqh animation/tree_builder.py BuildTreeScene   # high quality
```

## Structure

```
decision_tree/    # Core CART algorithm — pure Python + NumPy
examples/         # sensor_failure/ and cancer/ examples
animation/        # Manim scenes (tree_builder.py, predictor.py)
tests/            # pytest suite
results/          # Auto-generated outputs (.txt + .png)
```

## License

MIT

