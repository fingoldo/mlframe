"""Pre-screen of the raw columns for the FE families that may run on a smaller column pool (``audits/2026-11-09/fe_candidate_pool_plan.md``).

The FE cascades run on the raw frame BEFORE the screening, so no relevance set exists when they pick their columns. This is the cheap stand-in: each numeric raw column is
quantile-coded on the fit's shared row subsample, its plug-in mutual information with the target codes is read off one joint count, and the G-test of that MI
(``2 m MI`` is chi-square distributed with ``(levels - 1) (classes - 1)`` degrees of freedom when the column is independent of the target) gives a p-value. The pool is the
columns with a p-value below ``p_loose`` (a bound looser than the screening's own FDR cut, so a weak marginal signal that a transform could lift is kept) plus the ``top_k`` columns
by MI, plus every column that has non-finite values on the subsample (they are not judged here).

Only families whose signal shows in the marginal binned MI may use it: a U or window shape does (binned MI sees a non-monotone dependence), an XOR, a fast oscillation, a ratio
partner or a group key does not, and those families keep all columns. ``MRMR(fe_shape_pool="relevant")`` applies the pool to the unary shape families; the default ``"all"`` changes nothing.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

DEFAULT_P_LOOSE = 0.2
DEFAULT_TOP_K = 5
DEFAULT_BINS = 10


def marginal_mi_pvalues(codes: np.ndarray, levels: np.ndarray, y_codes: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """``(mi, p_value)`` per row of ``codes`` (columns, m) against ``y_codes`` (m,): the plug-in MI in nats and the chi-square p-value of its G statistic."""
    from .._analytic_mi_null import _chi2_sf

    n_cols, m = codes.shape
    _, y_dense = np.unique(y_codes, return_inverse=True)
    ky = int(y_dense.max()) + 1
    width = int(codes.max()) + 1
    joint = np.bincount((np.arange(n_cols)[:, None] * width + codes.astype(np.int64)).ravel() * ky + np.tile(y_dense, n_cols), minlength=n_cols * width * ky).reshape(n_cols, width, ky).astype(np.float64)  # Intentional float64: counts and their logarithms in the G statistic
    px = joint.sum(axis=2, keepdims=True)
    py = joint.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = joint * (np.log(joint) + np.log(float(m)) - np.log(px) - np.log(py))
    mi = np.nansum(np.where(joint > 0, term, 0.0), axis=(1, 2)) / m
    df = np.maximum(1, (np.asarray(levels, dtype=np.float64) - 1.0) * (ky - 1))
    p = np.array([1.0 if d <= 0 or ky < 2 else float(_chi2_sf(max(0.0, 2.0 * m * v), float(d))) for v, d in zip(mi, df)])
    return mi, p


def raw_column_pool(
    X: Any,
    y_codes: np.ndarray,
    *,
    subsample_idx: Optional[np.ndarray] = None,
    columns: Optional[Sequence[str]] = None,
    n_bins: int = DEFAULT_BINS,
    p_loose: float = DEFAULT_P_LOOSE,
    top_k: int = DEFAULT_TOP_K,
) -> "list[str]":
    """The names of the numeric columns of ``X`` (among ``columns`` when given) that pass the marginal pre-screen, in frame order; every numeric column when the rows are too few to judge."""
    from .._numba_utils import quantile_codes_block
    from .._quantile_codes_gpu import quantile_codes_block_auto

    names = [c for c in (columns if columns is not None else X.columns) if c in X.columns and np.issubdtype(np.asarray(X[c].iloc[:1]).dtype, np.number)]
    y = np.asarray(y_codes).ravel()
    idx = np.arange(len(y)) if subsample_idx is None else np.asarray(subsample_idx)
    if len(names) == 0 or idx.size < 200:
        return names
    block = np.ascontiguousarray(np.column_stack([np.asarray(X[c].to_numpy(), dtype=np.float32)[idx] for c in names]).T)
    finite = np.isfinite(block).all(axis=1)
    ys = y[idx]
    keep = np.flatnonzero(finite)
    pool = {names[j] for j in np.flatnonzero(~finite)}
    if keep.size:
        v = np.ascontiguousarray(block[keep])
        codes = np.empty(v.shape, dtype=np.int16)
        nx = np.empty(v.shape[0], dtype=np.int64)
        quantile_codes_block_auto(v, int(n_bins), codes, nx, host_kernel=quantile_codes_block)
        mi, p = marginal_mi_pvalues(codes, nx, ys)
        pool |= {names[keep[j]] for j in np.flatnonzero(p < p_loose)}
        pool |= {names[keep[j]] for j in np.argsort(-mi)[: int(top_k)]}
    return [c for c in names if c in pool]


__all__ = ["marginal_mi_pvalues", "raw_column_pool"]


def restrict_to_shape_pool(estimator: Any, X: Any, y: Any, cols: "Optional[Sequence[str]]") -> "Optional[list[str]]":
    """``cols`` (``None`` = every numeric raw column) narrowed to the pre-screen pool when ``estimator.fe_shape_pool == "relevant"``, else returned unchanged.

    Called by the unary shape families (hinge, wavelet) only; the pre-screen runs on the fit's shared row subsample."""
    if str(getattr(estimator, "fe_shape_pool", "all")) != "relevant":
        return None if cols is None else list(cols)
    from .._y_encoding import encode_y_for_classif_mi

    pool = raw_column_pool(X, encode_y_for_classif_mi(np.asarray(y)), subsample_idx=getattr(estimator, "_fe_shared_subsample_idx", None), columns=cols)
    return pool


__all__ += ["restrict_to_shape_pool"]
