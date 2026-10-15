"""Input-frame extraction helpers for the composite estimator, kept in a leaf module.

Leaf module: the estimator, its frame and predict siblings, and the package facade all read these, and this module imports nothing from the estimator package.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd
from .._composite_utils import is_polars_df as _is_polars_df

# Post-inverse y-clip bounds: extend the ``[Q001(y_train), Q999(y_train)]`` envelope by a fraction of its span on each side (see ``_y_train_clip_bounds``).
# Values outside this extended envelope are unphysical for the training distribution and almost certainly the result of ``exp`` / division blow-up.
# Low side extends by ``(1 - _Y_CLIP_LOW_FRAC)`` spans, high side by ``(_Y_CLIP_HIGH_FRAC - 1)`` spans, so the high tail is given far more room (heavy-tail asymmetry).
_Y_CLIP_LOW_FRAC: float = 0.1
_Y_CLIP_HIGH_FRAC: float = 10.0


def _y_train_clip_bounds(y_train: np.ndarray) -> tuple[float, float]:
    """Compute post-inverse y-clip bounds from train target.

    Bounds are the Q001 / Q999 envelope extended by a fraction of its span (``low = q_low - 0.9*span``, ``high = q_high + 9*span``) to keep predictions
    inside a physically plausible range even when the inverse transform produces extreme values for out-of-distribution rows. The asymmetric extension
    (0.9 spans below, 9 spans above) reflects the typical heavy-tail asymmetry of regression targets in finance / volume / rate domains; a symmetric clip
    would bite legitimate upper-tail predictions on log-normally distributed targets.
    """
    finite = y_train[np.isfinite(y_train)]
    if finite.size == 0:
        return float("-inf"), float("inf")
    # Single np.quantile([q_low, q_high]) does one sort instead of two -> ~2x; bit-identical to two separate calls.
    q_low, q_high = (float(v) for v in np.quantile(finite, (0.001, 0.999)))
    # Constant / near-constant train target gives zero span (degenerate envelope); fall back to a small wiggle around the median below.
    span = q_high - q_low
    if span <= 0:
        # Constant target on train; allow +/- 10% wiggle.
        med = float(np.median(finite))
        return med - 0.1 * abs(med) - 1e-6, med + 0.1 * abs(med) + 1e-6
    low = q_low - (1.0 - _Y_CLIP_LOW_FRAC) * span
    high = q_high + (_Y_CLIP_HIGH_FRAC - 1.0) * span
    return low, high


def _extract_base(X: Any, base_column: str) -> np.ndarray:
    """Pull base values from X (pandas / polars DataFrame).

    Raises ``KeyError`` with a helpful message if the column is missing
    -- this most commonly bites callers who configured MRMR / RFECV
    that dropped the base column before reaching the wrapper. The
    message points at the fix (``forced_keep_columns`` in the feature
    selection config).
    """
    if base_column not in getattr(X, "columns", ()):
        # A synthetic interaction base is recomputed from its parents, so a fresh frame needs no extra column.
        from .._synthetic_bases import synthetic_column

        syn = synthetic_column(X, base_column)
        if syn is not None:
            return syn
    # Polars
    if _is_polars_df(X):
        if base_column not in X.columns:
            raise KeyError(
                f"CompositeTargetEstimator: base column '{base_column}' missing from X. "
                "If feature selection (MRMR/RFECV) is dropping it, add base_column "
                "to forced_keep_columns in the feature selection config."
            )
        # Polars Series.to_numpy() already returns an ndarray; the prior
        # np.asarray wrapper allocated a redundant view. astype(copy=False)
        # avoids a second copy when the dtype already matches.
        return np.asarray(X.get_column(base_column).to_numpy().astype(np.float64, copy=False))
    if isinstance(X, pd.DataFrame):
        if base_column not in X.columns:
            raise KeyError(
                f"CompositeTargetEstimator: base column '{base_column}' missing from X. "
                "Columns: " + ", ".join(map(str, X.columns[:8])) + ("..." if len(X.columns) > 8 else "")
            )
        return np.asarray(X[base_column].to_numpy(dtype=np.float64))
    raise TypeError(f"CompositeTargetEstimator: unsupported X type {type(X).__name__}; " "pass a pandas or polars DataFrame.")


def _extract_groups(X: Any, group_column: str) -> np.ndarray:
    """Pull group labels from X without numeric casting.

    Unlike :func:`_extract_base`, this preserves the dtype (string /
    integer / categorical) so per-row group lookups work in the
    grouped transform. Returns a 1-D ndarray.
    """
    if _is_polars_df(X):
        if group_column not in X.columns:
            raise KeyError(f"CompositeTargetEstimator: group column '{group_column}' " f"missing from X.")
        # Polars Series.to_numpy() already returns an ndarray; the prior
        # np.asarray wrapper allocated a redundant view.
        return np.asarray(X.get_column(group_column).to_numpy())
    if isinstance(X, pd.DataFrame):
        if group_column not in X.columns:
            raise KeyError(f"CompositeTargetEstimator: group column '{group_column}' " f"missing from X.")
        return np.asarray(X[group_column].to_numpy())
    raise TypeError(f"CompositeTargetEstimator: unsupported X type {type(X).__name__}; " "pass pandas / polars DataFrame.")


def _extract_base_matrix(X: Any, base_columns: Sequence[str]) -> np.ndarray:
    """Multi-column variant of :func:`_extract_base`.

    Returns a 2-D ``(n, K)`` ndarray where K = ``len(base_columns)``.
    For K=1 this is identical to ``_extract_base(...).reshape(-1, 1)``.
    The K-column case is used by ``linear_residual_multi`` and any
    future multi-base transform.

    Same missing-column / unsupported-type semantics as
    :func:`_extract_base`. Errors include the offending column name so
    feature-selection drops are easy to diagnose.
    """
    if len(base_columns) == 0:
        raise ValueError("CompositeTargetEstimator: base_columns is empty; multi-base " "transforms require at least one base column.")
    # Single-select fast path: a polars ``.select(cols).to_numpy()`` materialises all K cols in one Arrow
    # buffer (~3-5x faster than the per-column-then-column_stack loop on K>=4 cols, validated on a
    # 1M-row x 8-col synthetic in bench_extract_base_matrix.py). Same for pandas: ``loc[:, cols].to_numpy()``
    # avoids per-column dispatch.
    cols_list = list(base_columns)
    if any(c not in getattr(X, "columns", ()) for c in cols_list):
        # One or more synthetic interaction bases: resolve column by column (each missing one from its parents).
        return np.column_stack([_extract_base(X, c) for c in cols_list])
    if _is_polars_df(X):
        missing = [c for c in cols_list if c not in X.columns]
        if missing:
            raise KeyError(
                f"CompositeTargetEstimator: base columns {missing!r} missing from X. "
                "If feature selection (MRMR/RFECV) is dropping them, add the base_columns "
                "to forced_keep_columns in the feature selection config."
            )
        arr = X.select(cols_list).to_numpy()
        return np.asarray(arr.astype(np.float64, copy=False) if arr.dtype != np.float64 else arr)
    if isinstance(X, pd.DataFrame):
        missing = [c for c in cols_list if c not in X.columns]
        if missing:
            raise KeyError(
                f"CompositeTargetEstimator: base columns {missing!r} missing from X. "
                "Columns: " + ", ".join(map(str, X.columns[:8])) + ("..." if len(X.columns) > 8 else "")
            )
        return np.asarray(X.loc[:, cols_list].to_numpy(dtype=np.float64, copy=False))
    # Fallback: route per-column for unknown X type (preserves prior behaviour).
    cols = [_extract_base(X, c) for c in cols_list]
    return np.column_stack(cols)
