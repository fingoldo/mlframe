"""Fit-scoped memoization for cross-family raw-column MI baselines + dense correlation-matrix builds.

Several independent opt-in orthogonal-FE scorer layers (total-correlation, routing, adaptive-degree,
cluster-basis, diff-basis, adaptive-arity) each independently compute a full raw-column MI(x; y) batch
and/or a dense NaN-mean-filled Pearson |corr| matrix over the SAME numeric-column universe whenever the
caller leaves ``cols=None`` (the common default) -- a "kitchen sink" MRMR config stacking 2+ of these
layers otherwise repeats the identical O(n*p) MI batch / O(p^2*n) corrcoef pass once per layer within ONE
``MRMR.fit()`` call.

Mirrors ``dedup_collinear_memo_scope()`` in ``_orth_dedup.py`` / ``heavy_tail_memo_scope()`` in
``hermite_fe._hermite_robust``: a ``threading.local`` cache, OFF by default (``None``), enabled for the
scope of one ``MRMR.fit()`` via :func:`orth_scoring_memo_scope`. Every helper here is a pure passthrough
(identical result, same underlying call) when no scope is active, so behavior outside an active scope is
byte-for-byte unchanged.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Optional, Sequence, cast

import numpy as np
import pandas as pd

_SCORING_MEMO = threading.local()


@contextlib.contextmanager
def orth_scoring_memo_scope():
    """Enable the fit-scoped raw-MI-baseline / corr-matrix memo for one ``MRMR.fit()`` call, then clear
    it. Nesting-safe: an inner scope reuses the outer cache; the outer owner clears it."""
    if getattr(_SCORING_MEMO, "cache", None) is not None:
        yield  # reuse outer scope; outer owner clears
        return
    _SCORING_MEMO.cache = {}
    try:
        yield
    finally:
        _SCORING_MEMO.cache = None


def _col_hash(X: pd.DataFrame, c: str) -> tuple:
    """Content-hash key for one column, float64-normalized so two callers casting to different dtypes
    for OTHER purposes still key identically here (the actual computation below still uses each caller's
    own ``dtype``, folded into the outer cache key separately)."""
    from .._fe_resident_operands import _content_hash

    arr = np.ascontiguousarray(np.asarray(X[c].to_numpy(), dtype=np.float64))
    return (arr.shape, _content_hash(arr))


def cached_raw_mi_baseline(
    cols: Sequence[str], raw_mat: np.ndarray, y_int: np.ndarray, *, nbins: int,
) -> dict[str, float]:
    """Return ``{col: MI(col; y_int)}`` for the caller's OWN already-built ``(n, len(cols))`` matrix
    ``raw_mat`` (columns aligned 1:1 with ``cols``), computed via ONE ``_mi_classif_batch`` call.

    Deliberately takes the PREPARED matrix rather than ``(X, cols, dtype)``: the raw-MI-baseline callers
    across the orth-FE family do not all pre-process identically (some pass a raw ``_crit_np_dtype()``-cast
    slice straight through, letting ``_mi_classif_batch`` handle NaN internally; others pre-fill NaN with
    the column mean before casting) -- reusing each caller's own already-built matrix keeps that
    convention exactly intact; the memo only adds a cache layer in front of the identical
    ``_mi_classif_batch`` call, it never changes what gets computed.

    Memoized PER COLUMN within an active :func:`orth_scoring_memo_scope` keyed by ``(column content hash
    + dtype + shape, y_int content hash, nbins)`` -- a caller requesting a subset of already-cached columns
    (same content, same dtype, same NaN treatment) is a pure cache hit; any new columns are batched
    together in ONE ``_mi_classif_batch`` call (not one call per miss). Outside an active scope this is
    unconditionally a full fresh batch call -- byte-for-byte identical to calling
    ``_mi_classif_batch(raw_mat, y_int, nbins=nbins)`` directly.
    """
    from ._orth_mi_backends import _mi_classif_batch

    cols = list(cols)
    if not cols:
        return {}
    _memo = getattr(_SCORING_MEMO, "cache", None)
    if _memo is None:
        mi = _mi_classif_batch(raw_mat, y_int, nbins=nbins)
        return {c: float(v) for c, v in zip(cols, mi.tolist())}
    from .._fe_resident_operands import _content_hash

    y_hash = _content_hash(np.ascontiguousarray(np.asarray(y_int, dtype=np.int64)))
    result: dict[str, float] = {}
    missing_idx: list[int] = []
    missing_cols: list[str] = []
    keys: dict[str, tuple] = {}
    for j, c in enumerate(cols):
        col_arr = np.ascontiguousarray(raw_mat[:, j])
        key = ("raw_mi_col", c, col_arr.shape, str(col_arr.dtype), _content_hash(col_arr), y_hash, int(nbins))
        keys[c] = key
        hit = _memo.get(key)
        if hit is None:
            missing_idx.append(j)
            missing_cols.append(c)
        else:
            result[c] = hit
    if missing_idx:
        sub = np.ascontiguousarray(raw_mat[:, missing_idx])
        mi = _mi_classif_batch(sub, y_int, nbins=nbins)
        for c, v in zip(missing_cols, mi.tolist()):
            fv = float(v)
            result[c] = fv
            _memo[keys[c]] = fv
    return result


def cached_dense_finite_corr_matrix(
    X: pd.DataFrame, cols: Optional[Sequence[str]], *, dtype,
) -> tuple[list[str], np.ndarray]:
    """Dense NaN-mean-filled Pearson ``|corr|`` matrix over ``cols`` (``None`` = all numeric columns of
    ``X``), dropping constant / all-NaN columns -- the shared build both
    ``_orthogonal_cluster_basis_fe._discover_clusters`` and
    ``_orthogonal_diff_basis_fe.detect_correlated_pairs`` need (identical NaN-fill + corrcoef recipe).

    Memoized (whole-list key, mirroring ``_orth_dedup._dedup_cache_key``) within an active
    :func:`orth_scoring_memo_scope`; a pure fresh build outside an active scope -- byte-for-byte identical
    to the inline build either caller used to do itself.

    Returns ``(dense_names, abs_corr)`` where ``abs_corr`` is the ``len(dense_names) x len(dense_names)``
    absolute-Pearson matrix (an empty ``(0, 0)`` array when fewer than 2 dense columns survive filtering).
    """
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cols = [c for c in cols if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]

    def _build() -> tuple[list[str], np.ndarray]:
        """Compute the dense-column names and their absolute-Pearson correlation matrix."""
        dense_arrays: list[np.ndarray] = []
        dense_names: list[str] = []
        for c in cols:
            arr = np.asarray(X[c].to_numpy(), dtype=dtype)
            finite = np.isfinite(arr)
            if not finite.any():
                continue
            if not finite.all():
                arr = np.where(finite, arr, float(np.nanmean(arr[finite])))
            if float(arr.std()) <= 1e-12:
                continue
            dense_arrays.append(arr)
            dense_names.append(c)
        if len(dense_names) < 2:
            return dense_names, np.zeros((0, 0), dtype=np.float64)
        mat = np.vstack(dense_arrays)
        corr = np.corrcoef(mat)
        if corr.ndim == 0:
            return dense_names, np.zeros((0, 0), dtype=np.float64)
        return dense_names, np.abs(corr)

    _memo = getattr(_SCORING_MEMO, "cache", None)
    if _memo is None or not cols:
        return _build()
    key = ("dense_corr", tuple(_col_hash(X, c) for c in cols), str(dtype))
    hit = _memo.get(key)
    if hit is not None:
        return cast("tuple[list[str], np.ndarray]", hit)
    result = _build()
    _memo[key] = result
    return result
