"""CompositeTargetEstimator + helpers: sklearn-compatible wrapper that hides the transform-and-invert loop from downstream callers. Split out of composite.py to keep wrapper surface independent of CompositeTargetDiscovery; composite.py re-exports every symbol below at its bottom for full back-compat."""

from __future__ import annotations

import logging

# ``deque`` is re-exported here as the module-level binding the bound ``update()``
# rolling-buffer method resolves (pinned by test_m1_update_uses_module_level_deque
# -- it must NOT be a lazy local reimport).
from collections import deque
from typing import (
    Any,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from .._composite_utils import is_polars_df as _is_polars_df


from ..transforms import (
    DomainViolationError,
    Transform,
    UnknownTransformError,
    get_transform,
)

logger = logging.getLogger(__name__)


from ._input_extract import (  # moved to the leaf module; re-exported here
    _Y_CLIP_HIGH_FRAC,
    _Y_CLIP_LOW_FRAC,
    _extract_base,
    _extract_base_matrix,
    _extract_groups,
    _y_train_clip_bounds,
)

# ----------------------------------------------------------------------


from ...utils import coerce_to_1d_numpy as _to_1d_numpy

# CompositeTargetEstimator class (~945 lines) moved
# to sibling file _composite_target_estimator.py to drop this file below
# the 1k-line monolith threshold. Re-exported below so existing callers
# (`from mlframe.training.composite.estimator import CompositeTargetEstimator`)
# keep working. The sibling imports the top-level helpers (defined at the
# TOP of this file) from the partial-module state -- safe because this
# re-export at the BOTTOM triggers the sibling load AFTER those helpers
# have already been bound.
from ._estimator import CompositeTargetEstimator

# ----------------------------------------------------------------------
# Per-quantile ensemble blending
# ----------------------------------------------------------------------


def _normalised_member_weights(weights, n_members):
    """Per-member blend weights: uniform when ``weights`` is None, else validated, non-negative and renormalised to sum to 1."""
    if weights is None:
        w = np.full(n_members, 1.0 / n_members, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n_members,):
            raise ValueError(f"predict_quantile_ensemble: weights length {w.shape!r} != n_members {n_members}")
        if np.any(w < 0):
            raise ValueError(f"predict_quantile_ensemble: weights must be non-negative; got {w.tolist()!r}")
        w_sum = float(w.sum())
        if w_sum <= 0:
            raise ValueError("predict_quantile_ensemble: weights sum to zero")
        w = w / w_sum
    return w

def _member_quantile_matrix(m, idx, X, quantiles_arr, quantiles_list):
    """One member's ``(n_samples, n_quantiles)`` prediction matrix: a single batched call when supported, else one scalar-alpha call per quantile."""
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
    except Exception as _probe_err:  # best-effort: the per-alpha path computes the same quantiles
        logger.debug(
            "predict_quantile_ensemble: batched probe for member %d (%s) failed (%r); falling back to per-alpha calls.",
            idx,
            type(m).__name__,
            _probe_err,
        )
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
                raise ValueError(f"predict_quantile_ensemble: member {idx} predict_quantile returned ndim={pred_arr.ndim} (expected 1 or 2)")
            cols.append(pred_arr)
        member_mat = np.column_stack(cols)
    return member_mat

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
        raise ValueError(f"predict_quantile_ensemble: quantiles must be a non-empty 1-D sequence, got shape {quantiles_arr.shape!r}")
    if np.any((quantiles_arr <= 0.0) | (quantiles_arr >= 1.0)):
        raise ValueError(f"predict_quantile_ensemble: quantiles must be strictly between 0 and 1; got {quantiles_arr.tolist()!r}")
    if not np.all(np.diff(quantiles_arr) > 0):
        raise ValueError(f"predict_quantile_ensemble: quantiles must be sorted ascending and unique; got {quantiles_arr.tolist()!r}")

    w = _normalised_member_weights(weights, len(members))

    # Materialise every member's (n_samples, n_quantiles) matrix. Different quantile sets
    # MUST raise -- silently truncating the union would hand back a confidence interval
    # at the wrong nominal coverage.
    per_member: list[np.ndarray] = []
    expected_shape: Optional[Tuple[int, ...]] = None
    quantiles_list = [float(v) for v in quantiles_arr]
    for idx, m in enumerate(members):
        member_mat = _member_quantile_matrix(m, idx, X, quantiles_arr, quantiles_list)
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
