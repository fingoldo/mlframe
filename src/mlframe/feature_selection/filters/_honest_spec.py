"""Numba specification of the honest supervised null for the plain 'mdlp' binning strategy.

Built once per fit from the same frame and target the production categorisation uses. Columns with 1 < distinct values <= 32 are binned
midpoint-wise without the target (their observed codes are exact under permutation); the rest are supervised and are re-binned by the njit
chain on every permuted target. The target is encoded with the production ``_encode_mdlp_target`` (permutation-invariant), and the
target-free collapse fallbacks are precomputed per column.
"""

from __future__ import annotations

from mlframe.feature_selection.filters._fe_frame_ops import fe_column_is_numeric, fe_column_values
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from mlframe.feature_selection.filters._fe_family_timing import fe_timed

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
    """The spec's ``raw`` matrix read from the frame instead of copied out of it: ``raw[:, j]`` is column ``j`` of the frame in the matrix dtype (zeros for a column that is not numeric), or the
    median-filled column held in ``filled`` when the frame's column has NaN. The dense copy was the largest array of the spec (4 to 8 bytes per cell, 400 to 800 MB at a million rows by
    100 columns) and duplicated the frame the fit holds anyway; the spec lives on the fit's closure, which is dropped after the fit and never pickled."""

    def __init__(self, X: pd.DataFrame, cols: Sequence[str], n: int, dtype: Any, filled: "dict[int, np.ndarray]"):
        self._X = X
        self._cols = [c if (c in X.columns and pd.api.types.is_numeric_dtype(X[c])) else None for c in cols]
        self._filled = filled
        self.dtype = np.dtype(dtype)
        self.shape = (int(n), len(self._cols))

    def __eq__(self, other: object) -> bool:
        """Equal when both read the same columns in the same dtype and shape (NaN equal): two specs built from the same frame compare equal like two dense matrices would."""
        if not isinstance(other, _FrameColumns):
            return NotImplemented
        if self.shape != other.shape or self.dtype != other.dtype:
            return False
        return all(np.array_equal(self[:, j], other[:, j], equal_nan=True) for j in range(self.shape[1]))

    __hash__ = None  # type: ignore[assignment]  # a view of a mutable frame: compared by value, never a dict key

    def __getitem__(self, key: tuple) -> np.ndarray:
        """Column ``key[1]`` for ``key[0] == slice(None)``."""
        rows, j = key
        if rows != slice(None):
            raise IndexError("_FrameColumns serves whole columns only")
        return self.column(int(j), self.dtype)

    def column(self, j: int, dtype: Any) -> np.ndarray:
        """Column ``j`` in ``dtype``: a float32 column of the frame is read as it is when ``dtype`` is float32 (no widening copy), a median-filled one is converted from its double-precision fill."""
        if j in self._filled:
            return self._filled[j].astype(dtype, copy=False)
        name = self._cols[j]
        if name is None:
            return np.zeros(self.shape[0], dtype=dtype)
        return np.asarray(fe_column_values(self._X, name), dtype=dtype)


def _exact_in_float32(dtype: Any) -> bool:
    """Whether every value of a column of ``dtype`` is the same number in float32: float32 itself, the lazy engineered float32 column, booleans and the integers of at most 16 bits (the target the frame carries is one)."""
    from ._lazy_engineered import LazyEngineeredDtype

    if isinstance(dtype, LazyEngineeredDtype):
        return True
    return isinstance(dtype, np.dtype) and (dtype == np.float32 or dtype == np.bool_ or (dtype.kind in "iu" and dtype.itemsize <= 2))


def _float32_exact_flags(X: pd.DataFrame, cols: Sequence[str], nan_by_col: "dict[int, np.ndarray]") -> np.ndarray:
    """Per column of ``cols``: whether the spec serves it in float32 without changing a value. A column the frame does not have or that is not numeric is served as zeros (exact); a numeric one is exact when
    its dtype is float32 in value and it has no NaN to fill."""
    flags = np.ones(len(cols), dtype=np.bool_)
    for j, c in enumerate(cols):
        if c not in X.columns or not pd.api.types.is_numeric_dtype(X[c]):
            continue
        flags[j] = _exact_in_float32(X[c].dtype) and j not in nan_by_col
    return flags


def _median_filled_column(col: pd.Series) -> "tuple[np.ndarray, np.ndarray] | None":
    """``(values, NaN mask)`` of a numeric column with NaN replaced by the column median, or ``None`` when it holds an infinity or only NaN.

    The production categorisation fills NaN with the column median before binning (``_handle_missing``): the cut search, the fallback edges and the unique-value count all see the filled
    column. The NaN rows keep their own code afterwards."""
    v = np.asarray(col.to_numpy(), dtype=np.float64)  # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
    if np.isinf(v).any():
        return None
    nan_here = np.isnan(v)
    if nan_here.all():
        return None
    if nan_here.any():
        v = np.where(nan_here, float(np.nanmedian(v)), v)
    return v, nan_here


@fe_timed("stage:honest_spec")
def build_honest_spec(X: pd.DataFrame, cols: Sequence[str], y_raw: np.ndarray, max_depth: int, method: str = "mdlp") -> dict | None:
    """Spec dict for ``_mdlp_honest_njit.honest_null_columns`` over the first ``len(cols)`` factor columns, or None when a column is not
    usable (non-numeric or non-finite): the caller then keeps the categorise-per-permutation path."""
    from ._adaptive_nbins import _edges_from_quantiles
    from ._supervised_collapse_refine import _fallback_edges
    from .supervised_binning import _encode_mdlp_target

    n = len(y_raw)
    m = len(cols)
    nan_by_col: dict[int, np.ndarray] = {}  # the NaN masks of the columns that have NaN; the (n, m) bool matrix exists only when one does
    # float32 frames keep a float32 matrix: widening a float32 value to float64 is exact, so the cut search reads the same numbers (a column with NaN is median-filled in double
    # precision, so any NaN keeps the double matrix); the (n, m) matrix is the largest array of the spec, 4 bytes per cell instead of 8
    # The dtype test alone decides here: the target and the lazy engineered columns the frame carries are float32 in value although their dtype is not float32, and a test that read ``dtype == float32`` turned
    # the matrix to float64 in every fit that has one of them (the sort of the honest null then ran on 8-byte keys). The NaN test comes with the column pass below.
    _numeric = [c for c in cols if fe_column_is_numeric(X, c)]
    _f32_dtypes = bool(_numeric) and all(_exact_in_float32(X[c].dtype) for c in _numeric)
    filled: dict[int, np.ndarray] = {}  # the median-filled columns of the frame's columns that have NaN
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
        prepared = _median_filled_column(X[c])
        if prepared is None:
            return None
        v, nan_here = prepared
        if nan_here.any():
            nan_by_col[j] = nan_here
            filled[j] = v
        q_fb_rows.append(np.empty(0, dtype=np.float64))  # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
        q_rf_rows.append(np.empty(0, dtype=np.float64))  # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
        numeric_js.append(j)
    # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning); float32 only when every column is float32 in value and none has NaN
    raw: Any = _FrameColumns(X, cols, n, np.float32 if (_f32_dtypes and not nan_by_col) else np.float64, filled)
    # Per column: float32 in value (its dtype says so and it has no NaN to fill). A block of the honest null whose continuous columns are all such is read, sorted and binned in float32 although the matrix
    # is double: one median-filled column or one column computed in double makes the matrix double, and the other thousand are float32 widened.
    f32_exact = _float32_exact_flags(X, cols, nan_by_col)
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
    nan_mask: "np.ndarray | None" = None
    if nan_by_col:
        nan_mask = np.zeros((n, m), dtype=np.bool_)
        for j, col_mask in nan_by_col.items():
            nan_mask[:, j] = col_mask
    return {
        "method": method,
        "raw": raw,
        "raw_f32_exact": f32_exact,
        "nan_mask": nan_mask,
        "has_nan": nan_mask is not None,
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
