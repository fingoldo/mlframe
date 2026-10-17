"""Pre-check of the feature-base target mix: does ``T = y - beta * u`` still hold structure that a column or a column pair can explain?

The second pass of the mix fits a whole child estimator on ``T``. When the base already explains the target, ``T`` is noise and that fit finds nothing (about 40% of a strict-GPU xor3 fit). Two
family-wise 5% screens on ``T`` decide whether it is worth running, the same calibration the residual rescue uses:

* marginal: the held-out gain of a free-form one-column fit (``main_effect_bins`` levels fitted on the even rows, applied to the odd ones) over the constant, as a t statistic, Bonferroni over the
  columns; a weak term of one column under a dominant term shows here;
* pairs: ``pair_interaction_screen`` on ``T`` (interaction left after the additive model).

Either one significant sends ``T`` to the child; neither skips it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._fe_gain_stats import quantile_codes
from ._pair_residual_screen import FAMILY_ALPHA, MAX_COLS, N_BINS, main_effect_bins, pair_interaction_screen
from ._y_encoding import FEW_CLASSES_MAX

_MIN_ROWS = 800


def target_has_structure(X: "pd.DataFrame", target: np.ndarray, *, scan_rows: int = 50_000) -> bool:
    """True when a column or a column pair of ``X`` explains part of ``target`` beyond chance (family-wise 5% each); True as well when the check cannot be made (too few rows)."""
    from scipy.stats import norm

    n = len(X)
    if n < _MIN_ROWS:
        return True
    rows = np.arange(n) if n <= scan_rows else np.linspace(0, n - 1, int(scan_rows)).astype(np.int64)
    t = np.asarray(target, dtype=np.float64).ravel()[rows]
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    even = np.arange(len(rows)) % 2 == 0
    base = float(t[even].mean())
    bins = main_effect_bins(int(even.sum()))
    stats = []
    for c in cols:
        x = np.asarray(X[c].to_numpy(), dtype=np.float64)[rows]
        finite = np.isfinite(x)
        if finite.sum() < _MIN_ROWS or np.unique(x[finite]).size <= FEW_CLASSES_MAX:
            continue
        q = quantile_codes(x, bins)
        cnt = np.bincount(q[even], minlength=bins)
        mean = np.where(cnt > 0, np.bincount(q[even], weights=t[even], minlength=bins) / np.maximum(cnt, 1), base)
        odd = ~even
        d = (t[odd] - base) ** 2 - (t[odd] - mean[q[odd]]) ** 2
        sd = float(d.std())
        stats.append(float(d.mean()) / (sd / np.sqrt(len(d))) if sd > 0.0 else 0.0)
    if stats and max(stats) > float(norm.isf(FAMILY_ALPHA / len(stats))):
        return True
    return bool(pair_interaction_screen(X, t if len(rows) == n else _expand(t, rows, n), cols, scan_rows=scan_rows, max_cols=MAX_COLS))


def _expand(t: np.ndarray, rows: np.ndarray, n: int) -> np.ndarray:
    """``t`` placed back at ``rows`` of an ``n``-row vector (the pair screen subsamples again by the same rule)."""
    out = np.zeros(n, dtype=np.float64)
    out[rows] = t
    return out


__all__ = ["target_has_structure", "N_BINS"]
