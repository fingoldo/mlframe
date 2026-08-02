"""Shared helper for the ``training/core`` lag-router pair (``_ood_lag_router.py`` /
``_volatility_lag_router.py``): independently duplicated across those modules, consolidated
here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import numpy as np


def rmse_min50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE over the finite-valued rows of both arrays; returns NaN when fewer than 50 finite pairs remain, since an RMSE from a tiny sample is not a reliable routing signal."""
    f = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(f.sum()) < 50:
        return float("nan")
    d = y_true[f] - y_pred[f]
    return float(np.sqrt(np.mean(d * d)))
