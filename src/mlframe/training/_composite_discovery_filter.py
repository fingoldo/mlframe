"""Feature filter for :class:`CompositeTargetDiscovery`.

Carved out of ``composite_discovery`` via method-rebinding to keep the parent
facade under the LOC budget. Bound onto the class at the parent module's bottom.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from .composite_screening import (
    _extract_column_array,
    _is_numeric_column,
    _safe_abs_corr_all,
)

logger = logging.getLogger(__name__)


def _filter_features(
    self,
    df: Any,
    feature_cols: Sequence[str],
    y_train: np.ndarray,
    train_idx: np.ndarray,
) -> list[str]:
    """Drop columns that are non-numeric, near-constant on train, match a
    forbidden name pattern, or correlate suspiciously highly with y on
    train (likely derived-from-y leakage).

    Drops are recorded on ``self._filter_drops`` (list of dicts with name +
    reason + value) so :meth:`fit` can surface them in the report and so
    callers can audit false positives -- the corr filter in particular is
    prone to misfiring on legitimate autoregressive lag features such as
    a ``y_prev`` column.
    """
    # First pass: cheap-fail filters (name patterns, type, finite count,
    # near-constant). Build a list of survivors + their train-row arrays so the
    # corr check can be vectorised across all survivors in ONE matrix op
    # (~2.2x faster vs per-column ``_safe_corr`` loop on 200 cols x 80K rows).
    drops: list[dict[str, Any]] = []
    corr_drops: list[tuple[str, float]] = []
    candidates: list[str] = []
    candidate_arrays: list[np.ndarray] = []
    for col in feature_cols:
        if col == self._target_col:
            continue
        if any(p.search(col) for p in self._patterns_compiled):
            drops.append({"name": col, "reason": "forbidden_pattern"})
            continue
        if not _is_numeric_column(df, col):
            drops.append({"name": col, "reason": "non_numeric"})
            continue
        arr = _extract_column_array(df, col)[train_idx]
        finite_mask = np.isfinite(arr)
        if finite_mask.sum() < 50:
            drops.append({
                "name": col, "reason": "insufficient_finite_rows",
                "n_finite": int(finite_mask.sum()),
            })
            continue
        ptp = float(np.ptp(arr[finite_mask]))
        if ptp <= self.config.constant_base_eps:
            drops.append({
                "name": col, "reason": "constant_or_near_constant",
                "ptp": ptp,
            })
            continue
        candidates.append(col)
        candidate_arrays.append(arr)

    # Vectorised corr filter on survivors. Replaces the per-column
    # ``abs(_safe_corr(arr, y_train))`` loop. NaN rows in the survivor matrix
    # are imputed with column-mean before the corr-vs-y dot product, which is
    # a small approximation versus per-column NaN masking but only matters for
    # columns with sparse NaN -- and those have already passed the
    # ``finite_mask.sum() < 50`` gate above with at least 50 finite rows.
    # Acceptable trade-off for the ~600ms saving on 200-feature filter calls.
    kept: list[str] = []
    if candidates:
        X_train = np.column_stack(candidate_arrays)
        # Free the per-column ndarrays the moment they land in the stacked matrix:
        # candidate_arrays holds (n_features) views/copies that double the peak
        # footprint until we let them go (~8 GB on a 4M-row x 500-col float32 frame).
        candidate_arrays.clear()
        # nanmean over (N, F) requires no temp; nb the prior np.where(isfinite, X, nan)
        # built a SECOND full-frame copy purely to silence non-finite cells, redundant.
        col_means = np.nanmean(X_train, axis=0)
        non_finite_mask = ~np.isfinite(X_train)
        if non_finite_mask.any():
            # X_train is a freshly-allocated buffer owned by this function; mutating
            # in-place is safe (the .copy() removed here cost another full-frame
            # allocation -- ~8 GB transient on the 4M-row prod frame).
            X_train[non_finite_mask] = np.broadcast_to(
                col_means, X_train.shape,
            )[non_finite_mask]
        abs_corrs = _safe_abs_corr_all(y_train, X_train)
        threshold = float(self.config.forbidden_base_corr_threshold)
        for col, corr_val in zip(candidates, abs_corrs.tolist()):
            if corr_val >= threshold:
                drops.append({
                    "name": col, "reason": "forbidden_base_corr_threshold",
                    "corr": float(corr_val), "threshold": threshold,
                })
                corr_drops.append((col, float(corr_val)))
            else:
                kept.append(col)
    self._filter_drops = drops
    # Loud warning for corr-threshold drops: this is the filter most likely to
    # misfire on legitimate strong predictors (autoregressive lags,
    # near-deterministic features). Make it visible at INFO so users can spot a false positive.
    if corr_drops:
        corr_drops.sort(key=lambda t: -t[1])
        preview = ", ".join(f"{n}=|corr|{c:.6f}" for n, c in corr_drops[:5])
        logger.info(
            "[CompositeTargetDiscovery] corr-threshold filter dropped "
            "%d feature(s) (threshold=%.6f): %s%s. If a legitimate "
            "lag/strong predictor was dropped, raise "
            "forbidden_base_corr_threshold or pass it via "
            "base_candidates=[...] explicitly.",
            len(corr_drops),
            self.config.forbidden_base_corr_threshold,
            preview,
            "" if len(corr_drops) <= 5 else f" (+{len(corr_drops) - 5} more)",
        )
    return kept
