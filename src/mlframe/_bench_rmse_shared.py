"""Shared micro-benchmark RMSE helper used by several unrelated ``_benchmarks/`` packages
(training, training/composite/ensemble, models/ensembling): independently duplicated across
those scripts, consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import numpy as np


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-squared error between two same-shaped arrays."""
    return float(np.sqrt(np.mean((a - b) ** 2)))
