"""CompositeTargetEstimator + helpers: sklearn-compatible wrapper that hides the transform-and-invert loop from downstream callers. Split out of composite.py to keep wrapper surface independent of CompositeTargetDiscovery; composite.py re-exports every symbol below at its bottom for full back-compat."""


from __future__ import annotations

import logging
import warnings
from collections import deque
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import NotFittedError

try:
    import polars as pl  # type: ignore
    _HAS_POLARS = True
except ImportError:  # pragma: no cover - polars optional dep
    pl = None  # type: ignore
    _HAS_POLARS = False


def _is_polars_df(x: Any) -> bool:
    """ENS-P2-6: prefer explicit isinstance check over duck-typing on
    ``hasattr(x, "to_pandas")`` (which mis-detects any object exposing that
    method - mocks, custom wrappers, sklearn pipeline stubs)."""
    return _HAS_POLARS and isinstance(x, pl.DataFrame)


from .composite.transforms import (
    DomainViolationError,
    Transform,
    UnknownTransformError,
    _TRANSFORMS_REGISTRY,
    get_transform,
)

logger = logging.getLogger(__name__)


# Bounds for the post-inverse y-clip, expressed as multipliers on the
# ``[Q001(y_train), Q999(y_train)]`` envelope. Values outside this
# extended envelope are unphysical for the training distribution and
# almost certainly the result of ``exp`` / division blow-up.
_Y_CLIP_LOW_FRAC: float = 0.1
_Y_CLIP_HIGH_FRAC: float = 10.0


# ----------------------------------------------------------------------

def _y_train_clip_bounds(y_train: np.ndarray) -> tuple[float, float]:
    """Compute post-inverse y-clip bounds from train target.

    Bounds are extended Q001 / Q999 envelope multiplied by safety
    factors to keep predictions inside a physically plausible range
    even when the inverse transform produces extreme values for
    out-of-distribution rows. The asymmetric multipliers (0.1 lower,
    10x upper) reflect the typical heavy-tail asymmetry of regression
    targets in finance / volume / rate domains; symmetric clip would
    bite legitimate upper-tail predictions on log-normally distributed
    targets.
    """
    finite = y_train[np.isfinite(y_train)]
    if finite.size == 0:
        return float("-inf"), float("inf")
    q_low = float(np.quantile(finite, 0.001))
    q_high = float(np.quantile(finite, 0.999))
    # Edge case: q_low or q_high is 0 -> multiplier collapses bound;
    # fall back to absolute envelope around 0.
    span = q_high - q_low
    if span <= 0:
        # Constant target on train; allow +/- 10% wiggle.
        med = float(np.median(finite))
        return med - 0.1 * abs(med) - 1e-6, med + 0.1 * abs(med) + 1e-6
    low = q_low - (1.0 - _Y_CLIP_LOW_FRAC) * span
    high = q_high + (_Y_CLIP_HIGH_FRAC - 1.0) * span
    return low, high


from .utils import coerce_to_1d_numpy as _to_1d_numpy  # noqa: E402,F401


def _extract_base(X: Any, base_column: str) -> np.ndarray:
    """Pull base values from X (pandas / polars / structured ndarray).

    Raises ``KeyError`` with a helpful message if the column is missing
    -- this most commonly bites callers who configured MRMR / RFECV
    that dropped the base column before reaching the wrapper. The
    message points at the fix (``forced_keep_columns`` in the feature
    selection config).
    """
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
        return X.get_column(base_column).to_numpy().astype(np.float64, copy=False)
    if isinstance(X, pd.DataFrame):
        if base_column not in X.columns:
            raise KeyError(
                f"CompositeTargetEstimator: base column '{base_column}' missing from X. "
                "Columns: " + ", ".join(map(str, X.columns[:8])) + ("..." if len(X.columns) > 8 else "")
            )
        return X[base_column].to_numpy(dtype=np.float64)
    raise TypeError(
        f"CompositeTargetEstimator: unsupported X type {type(X).__name__}; "
        "pass pandas / polars DataFrame or a structured ndarray with named columns."
    )


def _extract_groups(X: Any, group_column: str) -> np.ndarray:
    """Pull group labels from X without numeric casting.

    Unlike :func:`_extract_base`, this preserves the dtype (string /
    integer / categorical) so per-row group lookups work in the
    grouped transform. Returns a 1-D ndarray.
    """
    if _is_polars_df(X):
        if group_column not in X.columns:
            raise KeyError(
                f"CompositeTargetEstimator: group column '{group_column}' "
                f"missing from X."
            )
        # Polars Series.to_numpy() already returns an ndarray; the prior
        # np.asarray wrapper allocated a redundant view.
        return X.get_column(group_column).to_numpy()
    if isinstance(X, pd.DataFrame):
        if group_column not in X.columns:
            raise KeyError(
                f"CompositeTargetEstimator: group column '{group_column}' "
                f"missing from X."
            )
        return X[group_column].to_numpy()
    raise TypeError(
        f"CompositeTargetEstimator: unsupported X type {type(X).__name__}; "
        "pass pandas / polars DataFrame."
    )


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
        raise ValueError(
            "CompositeTargetEstimator: base_columns is empty; multi-base "
            "transforms require at least one base column."
        )
    # Single-select fast path: a polars ``.select(cols).to_numpy()`` materialises all K cols in one Arrow
    # buffer (~3-5x faster than the per-column-then-column_stack loop on K>=4 cols, validated on a
    # 1M-row x 8-col synthetic in bench_extract_base_matrix.py). Same for pandas: ``loc[:, cols].to_numpy()``
    # avoids per-column dispatch.
    cols_list = list(base_columns)
    if _is_polars_df(X):
        missing = [c for c in cols_list if c not in X.columns]
        if missing:
            raise KeyError(
                f"CompositeTargetEstimator: base columns {missing!r} missing from X. "
                "If feature selection (MRMR/RFECV) is dropping them, add the base_columns "
                "to forced_keep_columns in the feature selection config."
            )
        arr = X.select(cols_list).to_numpy()
        return arr.astype(np.float64, copy=False) if arr.dtype != np.float64 else arr
    if isinstance(X, pd.DataFrame):
        missing = [c for c in cols_list if c not in X.columns]
        if missing:
            raise KeyError(
                f"CompositeTargetEstimator: base columns {missing!r} missing from X. "
                "Columns: " + ", ".join(map(str, X.columns[:8])) + ("..." if len(X.columns) > 8 else "")
            )
        return X.loc[:, cols_list].to_numpy(dtype=np.float64, copy=False)
    # Fallback: route per-column for unknown X type (preserves prior behaviour for ndarray-with-names etc.).
    cols = [_extract_base(X, c) for c in cols_list]
    return np.column_stack(cols)


# Wave 102 (2026-05-21): CompositeTargetEstimator class (~945 lines) moved
# to sibling file _composite_target_estimator.py to drop this file below
# the 1k-line monolith threshold. Re-exported below so existing callers
# (`from mlframe.training.composite_estimator import CompositeTargetEstimator`)
# keep working. The sibling imports the top-level helpers (defined at the
# TOP of this file) from the partial-module state -- safe because this
# re-export at the BOTTOM triggers the sibling load AFTER those helpers
# have already been bound.
from ._composite_target_estimator import CompositeTargetEstimator  # noqa: F401, E402

# ----------------------------------------------------------------------
# Per-quantile ensemble blending
# ----------------------------------------------------------------------

def predict_quantile_ensemble(
    members: Sequence[Any],
    X: Any,
    quantiles: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Blend multi-quantile predictions from ``members`` PER-QUANTILE COLUMN.

    Each member ``m`` must expose ``predict_quantile(X, alpha)`` returning a scalar-alpha
    1-D vector OR a multi-alpha 2-D ``(n_samples, n_quantiles)`` matrix. The blend is
    elementwise (per row x per quantile column) -- it preserves the quantile dimension
    rather than collapsing to a single ``.predict()``-style scalar per row.

    Why this matters: arithmetic-mean of the POINT predictions of K quantile heads
    discards the predictive interval entirely (each member would have answered a
    different question if asked for q10 vs q90). The correct blend stacks the q10
    columns from every member, the q50 columns from every member, the q90 columns
    from every member, and averages each STACK independently.

    Parameters
    ----------
    members : sequence of fitted estimators
        Each must expose ``predict_quantile(X, alpha=float)``. When the member's
        ``predict_quantile`` natively supports a multi-quantile array (CatBoost
        MultiQuantile, XGBoost ``quantile_alpha=[...]``), it is called once per
        ``alpha`` for back-compat with the scalar-alpha contract on
        ``CompositeTargetEstimator.predict_quantile``.
    X : feature frame
        Passed through to each member unchanged.
    quantiles : sequence of float
        Quantile levels to blend at. Must be sorted ascending; values in (0, 1).
    weights : sequence of float, optional
        Per-member non-negative weights. Renormalised to sum to 1. ``None`` means
        equal weights (arithmetic mean across members per column).

    Returns
    -------
    np.ndarray, shape (n_samples, n_quantiles)
        Blended per-quantile predictions in original y-scale.

    Raises
    ------
    ValueError
        On empty member list, mismatched per-member output shapes, mismatched
        quantile sets across members, weights length mismatch, non-sorted /
        out-of-range quantiles, or weights summing to zero.
    """
    if not members:
        raise ValueError("predict_quantile_ensemble: members is empty")

    quantiles_arr = np.asarray(quantiles, dtype=np.float64)
    if quantiles_arr.ndim != 1 or quantiles_arr.size == 0:
        raise ValueError(
            f"predict_quantile_ensemble: quantiles must be a non-empty 1-D sequence, got shape {quantiles_arr.shape!r}"
        )
    if np.any((quantiles_arr <= 0.0) | (quantiles_arr >= 1.0)):
        raise ValueError(
            f"predict_quantile_ensemble: quantiles must be strictly between 0 and 1; got {quantiles_arr.tolist()!r}"
        )
    if not np.all(np.diff(quantiles_arr) > 0):
        raise ValueError(
            f"predict_quantile_ensemble: quantiles must be sorted ascending and unique; got {quantiles_arr.tolist()!r}"
        )

    if weights is None:
        w = np.full(len(members), 1.0 / len(members), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(members),):
            raise ValueError(
                f"predict_quantile_ensemble: weights length {w.shape!r} != n_members {len(members)}"
            )
        if np.any(w < 0):
            raise ValueError(
                f"predict_quantile_ensemble: weights must be non-negative; got {w.tolist()!r}"
            )
        w_sum = float(w.sum())
        if w_sum <= 0:
            raise ValueError("predict_quantile_ensemble: weights sum to zero")
        w = w / w_sum

    # Materialise every member's (n_samples, n_quantiles) matrix. Different quantile sets
    # MUST raise -- silently truncating the union would hand back a confidence interval
    # at the wrong nominal coverage.
    per_member: list[np.ndarray] = []
    expected_shape: Optional[Tuple[int, int]] = None
    quantiles_list = [float(v) for v in quantiles_arr]
    for idx, m in enumerate(members):
        if not hasattr(m, "predict_quantile"):
            raise ValueError(
                f"predict_quantile_ensemble: member {idx} ({type(m).__name__!r}) lacks predict_quantile(X, alpha); "
                "wrap each member in CompositeTargetEstimator or another quantile-aware estimator."
            )
        # Single batched call when the member's ``predict_quantile`` accepts a sequence
        # (e.g. CatBoost MultiQuantile, CompositeTargetEstimator). We attempt one batched
        # call and fall back to per-alpha scalar calls only on TypeError / ValueError /
        # mismatched-shape -- LightGBM and sklearn QuantileRegressor are scalar-only.
        member_mat: Optional[np.ndarray] = None
        try:
            batched = m.predict_quantile(X, quantiles_list)
            batched_arr = np.asarray(batched, dtype=np.float64)
            if batched_arr.ndim == 2 and batched_arr.shape[1] == len(quantiles_list):
                member_mat = batched_arr
        except (TypeError, ValueError):
            member_mat = None
        if member_mat is None:
            cols: list[np.ndarray] = []
            for alpha_v in quantiles_arr:
                try:
                    pred = m.predict_quantile(X, float(alpha_v))
                except TypeError:
                    pred = m.predict_quantile(X, alpha=float(alpha_v))
                pred_arr = np.asarray(pred, dtype=np.float64)
                if pred_arr.ndim == 2:
                    if pred_arr.shape[1] != 1:
                        raise ValueError(
                            f"predict_quantile_ensemble: member {idx} returned shape {pred_arr.shape!r} for scalar alpha={alpha_v}; "
                            "expected (n_samples,) or (n_samples, 1)."
                        )
                    pred_arr = pred_arr[:, 0]
                elif pred_arr.ndim != 1:
                    raise ValueError(
                        f"predict_quantile_ensemble: member {idx} predict_quantile returned ndim={pred_arr.ndim} (expected 1 or 2)"
                    )
                cols.append(pred_arr)
            member_mat = np.column_stack(cols)
        if expected_shape is None:
            expected_shape = member_mat.shape
        elif member_mat.shape != expected_shape:
            raise ValueError(
                f"predict_quantile_ensemble: member {idx} produced shape {member_mat.shape!r}, "
                f"member 0 produced {expected_shape!r}. Per-quantile blend requires identical "
                "(n_samples, n_quantiles) across members -- different quantile sets are ambiguous "
                "(union vs intersection vs interpolated re-evaluation)."
            )
        per_member.append(member_mat)

    # Stack along a new member axis -> (M, N, K), then weighted-mean over M -> (N, K).
    stacked = np.stack(per_member, axis=0)
    # Broadcast weights along (N, K) without materialising a full (M, N, K) weight tensor.
    blended = np.tensordot(w, stacked, axes=(0, 0))
    return blended
