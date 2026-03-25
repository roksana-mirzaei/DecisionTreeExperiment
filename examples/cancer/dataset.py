"""
examples/cancer/dataset.py
---------------------------
Loads the sklearn Breast Cancer Wisconsin dataset and exposes it in a
format consistent with the rest of this project.

Dataset facts
-------------
* 569 samples, 30 numeric features (cell-nucleus measurements from FNA biopsy)
* Binary classification:
    0 — Malignant  (cancer present)
    1 — Benign     (no cancer)
* Source: sklearn.datasets.load_breast_cancer
  (original: UCI ML Repository, Wolberg et al. 1995)

Features (30 total)
-------------------
Computed from a digitised image of a fine-needle aspirate of a breast mass.
They describe characteristics of the cell nuclei present in the image:

  mean radius, mean texture, mean perimeter, mean area, mean smoothness,
  mean compactness, mean concavity, mean concave points, mean symmetry,
  mean fractal dimension, ... plus standard error and worst (largest) values
  for each of the 10 properties above.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer


CLASS_NAMES: list[str] = ["Malignant", "Benign"]


def load_cancer_dataset() -> tuple[NDArray[np.float64], NDArray[np.int_], list[str]]:
    """Load the Breast Cancer Wisconsin dataset from sklearn.

    Returns
    -------
    X : ndarray of float, shape (569, 30)
        Feature matrix — 30 cell-nucleus measurements.
    y : ndarray of int, shape (569,)
        Labels: 0 = Malignant, 1 = Benign.
    feature_names : list[str]
        Names for each of the 30 columns.
    """
    data = load_breast_cancer()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.int_)
    feature_names = list(data.feature_names)
    return X, y, feature_names
