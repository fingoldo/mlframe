"""Small helpers shared by the scan-based feature-engineering families (offset product, row statistics, out-of-fold warp)."""

from __future__ import annotations

import numpy as np


def rank_scaled(y: np.ndarray) -> np.ndarray:
    """Average ranks of ``y`` scaled into ``(0, 1]``, a bounded target whose bin means are robust to heavy tails."""
    from scipy.stats import rankdata

    return np.asarray(rankdata(y, method="average") / float(len(y)))


def scan_index(n: int, scan_rows: int) -> np.ndarray:
    """Evenly spaced row indices of the search sample (all rows when ``n`` is small)."""
    return np.arange(n) if n <= scan_rows else np.linspace(0, n - 1, int(scan_rows)).astype(np.int64)
