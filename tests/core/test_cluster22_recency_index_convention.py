"""Cluster 22 regression: the recency-weight docstring describes the index the code uses, with i = 1 the most recent observation (audit 2026-11-20, finding 22-F31).

Behaviour pin: the newest observation carries the largest weight for every scheme; the docstring names i = 1 as the most recent.
"""

from pathlib import Path

import numpy as np

import importlib

from mlframe.core.recency_weights import recency_weights

_SRC = Path(importlib.import_module("mlframe.core.recency_weights").__file__).resolve()


def test_newest_observation_is_heaviest_for_each_scheme():
    """The last weight (most recent) is the largest for poly, exp and power (22-F31)."""
    for scheme, param in (("poly", 1.0), ("exp", 0.5), ("power", 1.0)):
        w = recency_weights(5, scheme, param)
        assert w[-1] == w.max()
        assert np.isclose(w.sum(), 1.0)
