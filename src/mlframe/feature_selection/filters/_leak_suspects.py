"""Post-fit diagnostic: a selected raw column that alone almost determines the target is reported as a possible leak.

A pure-statistics filter cannot tell a leakage proxy of the target from a genuinely very strong feature, so nothing is dropped; the suspects are recorded in ``leak_suspects_`` and named in a warning.
A binary target is judged by the single-column AUC of the column (rank-sum form), a continuous one by the absolute Spearman correlation; multiclass targets are not judged.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SUSPECT_SCORE = 0.995  # single-column AUC (binary) or |Spearman| (continuous) at or above which a column is reported
_MIN_ROWS = 100


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks of ``values`` (ties share the mean rank)."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)  # Intentional float64: rank sums of up to n^2 / 2
    ranks[order] = np.arange(1, len(values) + 1)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    return (sums / counts)[inverse]


def leak_score(column: np.ndarray, y: np.ndarray) -> float:
    """How close ``column`` alone is to determining ``y``: ``max(AUC, 1 - AUC)`` for a two-class ``y``, ``|Spearman|`` for a continuous one, 0.0 when it cannot be judged."""
    mask = np.isfinite(column)
    column, y = column[mask], y[mask]
    if len(column) < _MIN_ROWS or np.ptp(column) == 0:
        return 0.0
    classes = np.unique(y)
    ranks = _rank(column)
    if len(classes) == 2:
        pos = y == classes[1]
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            return 0.0
        auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(max(auc, 1.0 - auc))
    if len(classes) <= 10:
        return 0.0
    yr = _rank(y.astype(np.float64))  # Intentional float64: rank correlation input
    if np.ptp(yr) == 0:
        return 0.0
    return float(abs(np.corrcoef(ranks, yr)[0, 1]))


def record_leak_suspects(estimator: Any, X: Any, y: Any) -> list:
    """Score the selected raw numeric columns of the fitted ``estimator`` against ``y``, store the suspects in ``estimator.leak_suspects_`` and warn once naming them; returns the suspects."""
    suspects: list = []
    try:
        names = [str(c) for c in estimator.feature_names_in_]
        selected = [names[int(i)] for i in (estimator.support_ if estimator.support_ is not None else []) if 0 <= int(i) < len(names)]
        y_arr = np.asarray(y).ravel()
        for name in selected:
            col = X[name]
            values = np.asarray(col.to_numpy() if hasattr(col, "to_numpy") else col)
            if values.dtype.kind not in "fiub":
                continue
            score = leak_score(values.astype(np.float64), y_arr)  # Intentional float64: rank-based, ties detected on the exact values
            if score >= SUSPECT_SCORE:
                suspects.append((name, round(score, 4)))
    except Exception as e:  # nosec B110 - a diagnostic must never break a fit
        logger.debug("leak suspect check skipped: %s", e)
    estimator.leak_suspects_ = suspects
    if suspects:
        shown = ", ".join(f"{n} ({s})" for n, s in suspects[:5])
        warnings.warn(
            f"MRMR: selected column(s) almost determine the target on their own: {shown}. A statistical filter cannot tell a leak from a very strong feature, so they are kept; check them for leakage.",
            UserWarning,
            stacklevel=3,
        )
    return suspects
