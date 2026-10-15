"""Floor-parity features with a data-driven period: ``floor(x / period) mod 2`` on a single numeric column.

A target of the form ``floor(k * x) mod 2`` (a square wave in one continuous column) carries no linear signal in the raw
column and is not fitted by the smooth or sinusoidal bases. The period is not known in advance, so it is found by scanning a
geometric grid for the period whose parity feature has the highest mutual information with the target. Each candidate column
is kept only when that MI beats the raw column's MI by a clear margin, and the features are replayed from recipes that hold
the column name, the period and the op, so ``transform`` reproduces them exactly without the target.
"""

from __future__ import annotations

import logging
from typing import Optional
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numba import njit, prange

from ._periodic_fe import apply_modular, build_modular_recipe, engineered_name_modular

logger = logging.getLogger(__name__)

__all__ = ["scan_parity_period", "hybrid_parity_fe_with_recipes"]

_GRID_POINTS = 120
_BINS = 10


# bench-attempt-rejected (2026-10-14): cupy grid scan (per-period bincount, MI on device, data resident) vs this njit kernel on GTX 1050 Ti,
# n=1e5: 0.030 s vs 0.026 s; n=1e6: 0.257 s vs 0.227 s. The 8-core prange kernel stays the only backend for this scan.
@njit(cache=True, parallel=True)
def _parity_mi_grid_njit(v: np.ndarray, y_codes: np.ndarray, n_classes: int, periods: np.ndarray) -> np.ndarray:
    """MI (nats) of ``floor(v / p) mod 2`` with the discrete target for every candidate period ``p``, periods scanned in parallel.

    Counts the 2 x n_classes contingency table per period straight from the codes, so the scan costs O(n) per period with no
    per-call allocation; the class totals do not depend on the period and are counted once. The period is applied as a multiply
    by its reciprocal, which can move a value that sits exactly on a bin edge into the neighbouring bin; the selected feature is
    unchanged because the grid is geometric and the MI is flat across that edge.
    """
    n = v.shape[0]
    out = np.zeros(periods.shape[0], dtype=np.float64)
    cls_tot = np.zeros(n_classes, dtype=np.int64)
    for i in range(n):
        cls_tot[y_codes[i]] += 1
    for g in prange(periods.shape[0]):
        inv = 1.0 / periods[g]
        counts = np.zeros((2, n_classes), dtype=np.int64)
        for i in range(n):
            c = int(np.floor(v[i] * inv)) & 1
            counts[c, y_codes[i]] += 1
        mi = 0.0
        for c in range(2):
            row = counts[c, :].sum()
            if row == 0:
                continue
            for k in range(n_classes):
                nij = counts[c, k]
                if nij == 0:
                    continue
                mi += (nij / n) * np.log((nij * n) / (row * cls_tot[k]))
        out[g] = mi
    return out


@njit(cache=True)
def _mi_codes_njit(x_codes: np.ndarray, n_x: int, y_codes: np.ndarray, n_y: int) -> float:
    """Plug-in MI (nats) of two integer-coded vectors from one contingency pass; equals ``sklearn.metrics.mutual_info_score``."""
    n = x_codes.shape[0]
    joint = np.zeros((n_x, n_y), dtype=np.int64)
    for i in range(n):
        joint[x_codes[i], y_codes[i]] += 1
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for a in range(n_x):
        if px[a] == 0:
            continue
        for b in range(n_y):
            nab = joint[a, b]
            if nab == 0:
                continue
            mi += (nab / n) * np.log((nab * n) / (px[a] * py[b]))
    return mi


def _parity_grid(v: np.ndarray) -> np.ndarray:
    """Geometric grid of candidate periods spanning the column's central range, from a small fraction to twice its span."""
    lo, hi = np.nanquantile(v, [0.01, 0.99])
    span = float(hi - lo)
    if not np.isfinite(span) or span <= 0.0:
        return np.empty(0)
    return np.geomspace(span / 200.0, span * 2.0, _GRID_POINTS)


def _raw_mi(v: np.ndarray, y: np.ndarray) -> float:
    """MI of the raw column with the target, after quantile binning into ``_BINS`` bins."""
    edges = np.unique(np.nanquantile(v, np.linspace(0.0, 1.0, _BINS + 1)))
    codes = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, max(len(edges) - 2, 0)).astype(np.int64)
    _, y_codes = np.unique(y, return_inverse=True)
    return float(_mi_codes_njit(codes, max(len(edges) - 1, 1), y_codes.astype(np.int64).ravel(), int(y_codes.max()) + 1))


def scan_parity_period(v: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """``(best_period, best_mi)`` of the parity feature ``floor(v / period) mod 2`` against the discrete target ``y``.

    Returns ``(nan, 0.0)`` when the column has no usable span, so the caller can skip it.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    finite = np.isfinite(v)
    if not finite.all():
        v, y = v[finite], y[finite]
    grid = _parity_grid(v)
    if grid.size == 0:
        return float("nan"), 0.0
    _, y_codes = np.unique(y, return_inverse=True)
    mis = _parity_mi_grid_njit(np.ascontiguousarray(v, dtype=np.float64), y_codes.astype(np.int64).ravel(), int(y_codes.max()) + 1, grid)
    g = int(np.argmax(mis))
    return float(grid[g]), max(float(mis[g]), 0.0)


def hybrid_parity_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    top_k: int = 4,
    min_uplift: float = 0.05,
) -> tuple[pd.DataFrame, list[str], list, pd.DataFrame]:
    """Append the best parity feature of each numeric column whose MI beats the raw column by ``min_uplift``, at most ``top_k``.

    Returns ``(X_augmented, appended_names, recipes, scores)``. ``scores`` lists every scanned column with its raw MI, the
    best period and the parity MI, so the decision is inspectable.
    """
    y_arr = np.asarray(y).ravel()
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    rows = []
    for c in cols:
        v = np.asarray(X[c].to_numpy(), dtype=np.float64)
        ok = np.isfinite(v)
        if ok.sum() < 20:
            continue
        raw = _raw_mi(v[ok], y_arr[ok])
        period, pmi = scan_parity_period(v[ok], y_arr[ok])
        if not np.isfinite(period):
            continue
        rows.append({"col": c, "raw_mi": raw, "period": period, "parity_mi": pmi, "uplift": pmi - raw})
    scores = pd.DataFrame(rows, columns=["col", "raw_mi", "period", "parity_mi", "uplift"])
    keep = scores[scores["uplift"] >= float(min_uplift)].sort_values("parity_mi", ascending=False).head(int(top_k))

    X_aug = X.copy()
    appended, recipes = [], []
    for _, r in keep.iterrows():
        name = engineered_name_modular(str(r["col"]), float(r["period"]), "parity")
        if name in X_aug.columns:
            continue
        X_aug[name] = apply_modular(np.asarray(X[r["col"]].to_numpy(), dtype=np.float64), float(r["period"]), "parity")
        appended.append(name)
        recipes.append(build_modular_recipe(name=name, src_name=str(r["col"]), period=float(r["period"]), op="parity"))
    return X_aug, appended, recipes, scores
