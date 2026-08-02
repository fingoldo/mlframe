"""Shared helpers for the votenrank/_benchmarks/ family: small utilities independently duplicated
across those scripts, consolidated here so a fix can't silently drift out of sync across copies.
Each benchmark script stays independently runnable via `python -m mlframe.votenrank._benchmarks.bench_...`.
"""
from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-squared error between ``y_true`` and ``y_pred``."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
