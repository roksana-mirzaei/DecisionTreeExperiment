"""
examples/sensor_failure/dataset.py
------------------------------------
Generates a synthetic, labelled sensor-failure dataset for a robotic actuator.

Scenario
--------
An industrial robot arm is monitored by five onboard sensors.  We want to
classify the robot's current operating state from a single snapshot of readings:

  Feature           Unit    Description
  ─────────────────────────────────────────────────────────────────────────
  vibration_hz      Hz      Mechanical vibration frequency of the actuator
  temperature_c     °C      Temperature near the motor coil
  voltage_v         V       DC supply voltage at the sensor node
  current_a         A       Current draw of the sensor / actuator
  response_ms       ms      Round-trip communication latency

Target classes (integer labels):
  0 — Normal operation   Clean readings within spec
  1 — Overheating        Temperature spike; slightly elevated current
  2 — Mechanical failure High-frequency vibration; normal temperature
  3 — Electrical fault   Voltage drop; high current; high latency

Each class is drawn from a distinct multivariate Gaussian so a shallow
decision tree can achieve high accuracy, making the core algorithm easy
to validate and interpret during a mentorship session.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Metadata — used by run.py and the animation scenes
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "vibration_hz",
    "temperature_c",
    "voltage_v",
    "current_a",
    "response_ms",
]

CLASS_NAMES: list[str] = [
    "Normal",
    "Overheating",
    "Mechanical failure",
    "Electrical fault",
]


def make_sensor_dataset(
    n_samples: int = 500,
    random_state: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Generate a synthetic labelled sensor-failure dataset.

    Parameters
    ----------
    n_samples : int
        Total number of samples.  Split equally across the four classes.
        Must be a multiple of 4; excess samples are silently dropped.
    random_state : int
        NumPy random seed used for reproducibility.

    Returns
    -------
    X : ndarray of float, shape (n_samples, 5)
        Feature matrix with columns in the order given by :data:`FEATURE_NAMES`.
    y : ndarray of int, shape (n_samples,)
        Integer class labels in ``{0, 1, 2, 3}``.

    Notes
    -----
    The Gaussian means and covariances are hand-tuned so that the classes are
    well-separated along different feature axes:
    * Class 2 is separated by ``vibration_hz``.
    * Class 1 is separated by ``temperature_c``.
    * Class 3 is separated by ``voltage_v``, ``current_a``, and ``response_ms``.
    This mirrors a realistic scenario where different fault modes manifest in
    different sensor channels.
    """
    rng = np.random.default_rng(random_state)
    n_per_class = n_samples // 4

    # ------------------------------------------------------------------
    # Class 0 — Normal operation
    # vibration: low, temperature: comfortable, voltage: stable ~5 V
    # ------------------------------------------------------------------
    X0 = rng.multivariate_normal(
        mean=[50.0, 35.0, 5.0, 1.2, 10.0],
        cov=np.diag([25.0, 9.0, 0.04, 0.01, 4.0]),
        size=n_per_class,
    )

    # ------------------------------------------------------------------
    # Class 1 — Overheating
    # temperature spikes above 80 °C; slightly elevated current draw
    # ------------------------------------------------------------------
    X1 = rng.multivariate_normal(
        mean=[55.0, 85.0, 4.8, 1.8, 15.0],
        cov=np.diag([30.0, 16.0, 0.05, 0.04, 9.0]),
        size=n_per_class,
    )

    # ------------------------------------------------------------------
    # Class 2 — Mechanical failure
    # high-frequency vibration from bearing wear; temperature normal
    # ------------------------------------------------------------------
    X2 = rng.multivariate_normal(
        mean=[200.0, 45.0, 4.9, 1.3, 25.0],
        cov=np.diag([400.0, 16.0, 0.04, 0.01, 16.0]),
        size=n_per_class,
    )

    # ------------------------------------------------------------------
    # Class 3 — Electrical fault
    # voltage drop, overcurrent, high communication latency
    # ------------------------------------------------------------------
    X3 = rng.multivariate_normal(
        mean=[60.0, 40.0, 3.2, 3.5, 60.0],
        cov=np.diag([25.0, 9.0, 0.09, 0.25, 100.0]),
        size=n_per_class,
    )

    X = np.vstack([X0, X1, X2, X3]).astype(np.float64)
    y = np.concatenate(
        [
            np.full(n_per_class, 0, dtype=np.int_),
            np.full(n_per_class, 1, dtype=np.int_),
            np.full(n_per_class, 2, dtype=np.int_),
            np.full(n_per_class, 3, dtype=np.int_),
        ]
    )

    # Shuffle so classes are interleaved (avoids order-dependent bugs)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]
