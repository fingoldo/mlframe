"""Benjamini-Hochberg (1995) step-up rejection, shared by the feature-selection modules that control FDR across many tests (audit 2026-11-20, 24-F60).

A leaf module with numpy only, so a selector can import it without pulling in the rest of the package.
"""

from __future__ import annotations

import numpy as np


def benjamini_hochberg_reject(p_values, alpha: float, dtype=np.float64) -> np.ndarray:
    """Boolean mask, ``True`` where the null is rejected at FDR ``alpha`` across all tested hypotheses.

    Sorts the p-values, finds the largest rank whose p-value is at most ``rank / m * alpha``, and rejects every hypothesis at or below that rank.
    ``dtype`` is the float type the p-values are compared in.
    """
    p = np.asarray(p_values, dtype=dtype)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * (np.arange(1, m + 1) / m)
    below = ranked <= thresholds
    reject = np.zeros(m, dtype=bool)
    if below.any():
        k = np.nonzero(below)[0].max()
        reject[order[: k + 1]] = True
    return reject


__all__ = ["benjamini_hochberg_reject"]
