"""Constants and array helpers shared by ``error_analysis`` and its carved-out sibling modules.

These live here rather than in either module so both import ONE definition. A second copy of the overlay bin
count or the drift z-quantile would drift silently, and the two modules would then disagree about the same
chart's binning -- the exact failure the renderer clusters' duplicated constants produced.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Histogram resolution for per-feature / target overlays; above ~60 the density curves turn into noisy combs
# at the row counts this subsystem sees.
DEFAULT_OVERLAY_BINS: int = 40
# Two-sided 95% normal quantile, used to size the target-drift bar from each split's own sampling error.
_DRIFT_Z: float = 1.96


def _as_float_1d(a: np.ndarray) -> np.ndarray:
    """Coerce ``a`` to a flat float64 array."""
    return np.asarray(a, dtype=np.float64).ravel()


__all__ = ["DEFAULT_OVERLAY_BINS", "_DRIFT_Z", "_as_float_1d"]


def _row_count(X: Any) -> int:
    """Row count of ``X`` (frame ``len`` / ndarray first axis) without materialising it."""
    if _is_frame(X):
        return len(X)
    arr = np.asarray(X)
    return arr.shape[0] if arr.ndim >= 1 else 0


def _is_frame(X: Any) -> bool:
    """True when ``X`` is a pandas / polars frame (has ``columns`` and is indexable) rather than an ndarray."""
    return hasattr(X, "columns") and hasattr(X, "__getitem__") and not isinstance(X, np.ndarray)


def _resolve_feature_names(X: Any, feature_names: Optional[Sequence[str]]) -> List[str]:
    """Feature names WITHOUT densifying the matrix -- for callers that only need a handful of columns at a few rows.

    Mirrors :func:`_resolve_feature_matrix`'s naming, but skips the whole-frame ``column_stack``. Frames expose their
    column labels directly; an ndarray gets positional ``f{i}`` names. Lets ``worst_k_table`` rank importances and pick
    label columns without building the full dense matrix it would immediately discard all but a K x top_fi slice of.
    """
    if _is_frame(X):
        cols = list(X.columns)
        return list(feature_names) if feature_names is not None else [str(c) for c in cols]
    arr = np.asarray(X)
    ncols = 1 if arr.ndim == 1 else arr.shape[1]
    return list(feature_names) if feature_names is not None else [f"f{i}" for i in range(ncols)]


def _pull_columns_at_rows(X: Any, col_indices: Sequence[int], row_idx: np.ndarray) -> Dict[int, np.ndarray]:
    """Densify ONLY ``col_indices`` at ``row_idx`` -- bit-identical to ``_resolve_feature_matrix(X)[row_idx][:, j]``.

    Returns ``{col_index -> float64 values at row_idx}``. Non-numeric frame columns are label-encoded over the FULL
    column (``np.unique`` codes) exactly as :func:`_resolve_feature_matrix` does, then indexed -- so the codes match
    the full-matrix path. Avoids building+discarding the whole dense matrix when only a few columns at a few rows
    are needed.
    """
    out: Dict[int, np.ndarray] = {}
    if _is_frame(X):
        cols = list(X.columns)
        for j in col_indices:
            col = X[cols[j]]
            arr = col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col)
            if arr.dtype.kind in "OUS" or arr.dtype.kind == "b":
                # An object column holding non-scalar elements (e.g. a list-valued embedding column) can't be
                # stringified by ``astype(str)`` -- numpy raises "setting an array element with a sequence"
                # trying to broadcast the list into a fixed-width string array. Mirror _resolve_feature_matrix's
                # embedding-column handling: substitute NaN rather than crash (a single embedding vector isn't
                # a meaningful scalar table cell anyway).
                if arr.dtype.kind == "O" and any(isinstance(v, (list, tuple, np.ndarray)) for v in arr):
                    out[j] = np.full(len(row_idx), np.nan, dtype=np.float64)
                    continue
                _, codes = np.unique(arr.astype(str), return_inverse=True)
                out[j] = codes.astype(np.float64)[row_idx]
            else:
                out[j] = arr.astype(np.float64)[row_idx]
        return out
    mat = np.asarray(X, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    for j in col_indices:
        out[j] = mat[row_idx, j]
    return out

# Over/under tail fraction for error-bias tagging (Evidently's signature 5% tails).
DEFAULT_TAIL_FRACTION: float = 0.05
