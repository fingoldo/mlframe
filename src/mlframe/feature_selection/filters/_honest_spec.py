"""Numba specification of the honest supervised null for the plain 'mdlp' binning strategy.

Built once per fit from the same frame and target the production categorisation uses. Columns with 1 < distinct values <= 32 are binned
midpoint-wise without the target (their observed codes are exact under permutation); the rest are supervised and are re-binned by the njit
chain on every permuted target. The target is encoded with the production ``_encode_mdlp_target`` (permutation-invariant), and the
target-free collapse fallbacks are precomputed per column.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

_LOW_CARD_CAP = 32  # compute_col_edges default (kwargs 'low_card_cap')
_FALLBACK_NBINS = 5  # collapsed_fallback_nbins default


def _pad_rows(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack ragged 1-D rows into a zero-padded 2-D matrix; returns the matrix and each row's length."""
    lengths = np.array([r.size for r in rows], dtype=np.int64)
    width = max(1, int(lengths.max()) if lengths.size else 1)
    # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
    out = np.zeros((len(rows), width), dtype=np.float64)
    for j, r in enumerate(rows):
        out[j, : r.size] = r
    return out, lengths


class _FrameColumns:
    """The spec's ``raw`` matrix of a float32 frame without NaN, read from the frame instead of copied out of it: ``raw[:, j]`` is column ``j`` of the frame (zeros for a column that is not numeric),
    the same values the dense matrix holds. The dense copy was the largest array of the spec (4 bytes per cell, 400 MB at a million rows by 100 columns) and duplicated the frame the fit holds anyway;
    the spec lives on the fit's closure, which is dropped after the fit and never pickled."""

    dtype = np.dtype(np.float32)

    def __init__(self, X: pd.DataFrame, cols: Sequence[str], n: int):
        self._X = X
        self._cols = [c if (c in X.columns and pd.api.types.is_numeric_dtype(X[c])) else None for c in cols]
        self.shape = (int(n), len(self._cols))

    def __getitem__(self, key: tuple) -> np.ndarray:
        """Column ``key[1]`` for ``key[0] == slice(None)``."""
        rows, j = key
        if rows != slice(None):
            raise IndexError("_FrameColumns serves whole columns only")
        name = self._cols[int(j)]
        if name is None:
            return np.zeros(self.shape[0], dtype=np.float32)
        return np.asarray(self._X[name].to_numpy(dtype=np.float32, copy=False))


def build_honest_spec(X: pd.DataFrame, cols: Sequence[str], y_raw: np.ndarray, max_depth: int, method: str = "mdlp") -> dict | None:
    """Spec dict for ``_mdlp_honest_njit.honest_null_columns`` over the first ``len(cols)`` factor columns, or None when a column is not
    usable (non-numeric or non-finite): the caller then keeps the categorise-per-permutation path."""
    from ._adaptive_nbins import _edges_from_quantiles
    from ._supervised_collapse_refine import _fallback_edges
    from .supervised_binning import _encode_mdlp_target

    n = len(y_raw)
    m = len(cols)
    nan_cols = np.zeros((n, m), dtype=np.bool_)
    # float32 frames keep a float32 matrix: widening a float32 value to float64 is exact, so the cut search reads the same numbers (a column with NaN is median-filled in double
    # precision, so any NaN keeps the double matrix); the (n, m) matrix is the largest array of the spec, 4 bytes per cell instead of 8
    _numeric = [c for c in cols if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    _f32_exact = bool(_numeric) and all(X[c].dtype == np.float32 for c in _numeric) and not any(bool(X[c].isna().any()) for c in _numeric)
    raw: Any = _FrameColumns(X, cols, n) if _f32_exact else np.zeros((n, m), dtype=np.float64)  # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
    is_cont = np.zeros(m, dtype=np.bool_)
    q_fb_rows: list[np.ndarray] = []
    q_rf_rows: list[np.ndarray] = []
    numeric_js: list[int] = []
    for j, c in enumerate(cols):
        if c not in X.columns:
            # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
            q_fb_rows.append(np.empty(0, dtype=np.float64))
            # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
            q_rf_rows.append(np.empty(0, dtype=np.float64))
            continue
        if not pd.api.types.is_numeric_dtype(X[c]):
            # Categorical (non-numeric) columns are factorised without the target, so their codes are invariant under permutation of y:
            # they stay fixed (observed codes), exactly like the low-cardinality numeric columns.
            # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
            q_fb_rows.append(np.empty(0, dtype=np.float64))
            # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
            q_rf_rows.append(np.empty(0, dtype=np.float64))
            continue
        # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
        v = np.asarray(X[c].to_numpy(), dtype=np.float64)
        if np.isinf(v).any():
            return None
        nan_here = np.isnan(v)
        if nan_here.all():
            return None
        if nan_here.any():
            # The production categorisation fills NaN with the column median before binning (``_handle_missing``): the cut search,
            # the fallback edges and the unique-value count all see the filled column. The NaN rows keep their own code afterwards.
            v = np.where(nan_here, float(np.nanmedian(v)), v)
            nan_cols[:, j] = nan_here
        if not _f32_exact:
            raw[:, j] = v
        q_fb_rows.append(np.empty(0, dtype=np.float64))  # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
        q_rf_rows.append(np.empty(0, dtype=np.float64))  # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
        numeric_js.append(j)
    # The distinct-value count and the two quantile edge sets of a column read only that column, so the columns run on the edge-jobs pool; results are written by column index.
    from concurrent.futures import ThreadPoolExecutor

    from ._adaptive_nbins import _resolve_edge_jobs

    def _edges_of(j: int) -> "tuple[bool, np.ndarray, np.ndarray]":
        """``(is continuous, ranked-quantile edges, fallback edges)`` of the filled column ``j``."""
        # One sort serves the distinct count and both quantile edge sets (all three are order-invariant, and the percentile selection and the unique pass are cheap on sorted input).
        v = np.sort(raw[:, j]).astype(np.float64, copy=False)  # Intentional float64: the edges are computed in double precision from the (exactly widened) sorted column
        if 1 + int(np.count_nonzero(v[1:] != v[:-1])) <= _LOW_CARD_CAP:
            return False, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)  # Intentional float64: MDLP cut search at the production binning width
        # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
        return True, np.asarray(_edges_from_quantiles(v, _FALLBACK_NBINS), dtype=np.float64), np.asarray(_fallback_edges(v, "quantile", _FALLBACK_NBINS), dtype=np.float64)

    jobs = _resolve_edge_jobs(None, len(numeric_js))
    if jobs > 1 and len(numeric_js) > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            edge_sets = list(pool.map(_edges_of, numeric_js))
    else:
        edge_sets = [_edges_of(j) for j in numeric_js]
    for j, (cont, fb, rf) in zip(numeric_js, edge_sets):
        is_cont[j] = cont
        q_fb_rows[j] = fb
        q_rf_rows[j] = rf
    if method == "oos":
        # Held-out-validated MDLP: the cut search reads the raw (compact) class labels directly.
        _, enc = np.unique(np.asarray(y_raw), return_inverse=True)
        enc = enc.reshape(-1)
        depth_eff = int(max_depth)
    else:
        enc = _encode_mdlp_target(np.asarray(y_raw), 64, False, 16, int(max_depth))
    if method == "oos":
        labels = np.asarray(enc)
    elif isinstance(enc, tuple):
        labels, depth_eff = np.asarray(enc[0]), int(enc[1])
    else:
        labels, depth_eff = np.asarray(enc), int(max_depth)
    labels = np.ascontiguousarray(labels.ravel().astype(np.int64))
    q_fb, q_fb_len = _pad_rows(q_fb_rows)
    q_rf, q_rf_len = _pad_rows(q_rf_rows)
    return {
        "method": method,
        "raw": raw if _f32_exact else np.ascontiguousarray(raw),
        "nan_mask": nan_cols if nan_cols.any() else None,
        "has_nan": bool(nan_cols.any()),
        "is_cont": is_cont,
        "q_fb": q_fb,
        "q_fb_len": q_fb_len,
        "q_rf": q_rf,
        "q_rf_len": q_rf_len,
        "y_enc": labels,
        "n_classes_total": int(labels.max()) + 1,
        "max_depth": depth_eff,
        "min_split_size": 5,
        "val_min_split_size": 5,
        "oos_tolerance": 0.3,
        "holdout_frac": 0.3,
        "alpha": 0.05,
        "n_permutations": 15,
        "seed": 0,
    }


__all__ = ["build_honest_spec"]
