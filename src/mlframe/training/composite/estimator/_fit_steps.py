"""Step helpers of ``CompositeTargetEstimator.fit``: input validation, sample-weight resolution, inner fit, clip bounds and fitted-state stamping."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import clone

from ...utils import coerce_to_1d_numpy as _to_1d_numpy
from ._estimator_helpers import _estimator_fit_accepts_sample_weight
from ._input_extract import _y_train_clip_bounds
from ._inner_frame import frame_for_inner

logger = logging.getLogger("mlframe.training.composite.estimator._estimator")


def validate_fit_inputs(est: Any, X: Any, y: Any) -> None:
    """Reject a missing base estimator / target and an X-y length mismatch before any transform work."""
    if est.base_estimator is None:
        raise ValueError("CompositeTargetEstimator: base_estimator must not be None.")

    # sklearn's `check_requires_y_none` looks for this wording specifically; without it a supervised
    # estimator handed y=None surfaced as "coerce_to_numpy: input is None", which names an internal helper
    # rather than the contract the caller broke.
    if y is None:
        raise ValueError("CompositeTargetEstimator requires y to be passed, but the target y is None.")

    # Validate the X/y pairing BEFORE any transform or base-column work. The length check further down only
    # runs on the requires_base path and compares y against the extracted base column, so a unary
    # y-transform accepted a y of the wrong length outright, and a non-finite y was never rejected at all:
    # it flowed into the transform, and `diff` on an inf produced an inf target the inner estimator then
    # fitted, so the failure surfaced as a nonsense model rather than an error. sklearn's
    # `check_regressors_train` and `check_supervised_y_no_nan` both pin exactly these two contracts.
    _y_check = _to_1d_numpy(y).astype(np.float64)
    _n_x = getattr(X, "shape", (None,))[0] if hasattr(X, "shape") else (len(X) if hasattr(X, "__len__") else None)
    if _n_x is not None and _n_x != len(_y_check):
        raise ValueError(f"CompositeTargetEstimator.fit: X has {_n_x} rows but y has {len(_y_check)} -- " f"caller passed misaligned inputs.")
    # NO non-finite-y guard here, deliberately. sklearn's `check_supervised_y_no_nan` wants one, and an
    # earlier version of this validation added it -- wrongly. The recurrent transforms (EWMA residual,
    # frac-diff, seasonal residual) produce NaN warm-up rows in the TARGET by construction, and the fit
    # below carry-forward-fills them (`_carry_forward_fill(y_arr, np.isfinite(y_arr))`). Rejecting a
    # non-finite y therefore refuses a supported capability, and the check is pinned as an expected sklearn
    # failure with that reason rather than satisfied by breaking the feature.


def resolve_fit_sample_weight(est: Any, sample_weight: np.ndarray | None, valid: np.ndarray, base_train: np.ndarray) -> np.ndarray | None:
    """Subset caller weights to the valid rows, or derive 1/|base| variance-stabilising weights for ratio/logratio."""
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)[valid]
    elif getattr(est, "auto_variance_stabilise", False) and est.transform_name in ("ratio", "logratio"):
        # Variance-stabilising weights for ratio/logratio: under a
        # multiplicative DGP residual variance scales with |base|, so
        # weight ~ 1/|base| (floored at the 5th percentile to avoid blow-up
        # on near-zero base) flattens it. ``base_train`` is already filtered
        # to valid rows, so re-applying ``valid`` here would IndexError.
        base_valid = base_train.astype(np.float64)
        abs_base = np.abs(base_valid)
        floor_q = (
            float(
                np.quantile(
                    abs_base[abs_base > 0],
                    0.05,
                )
            )
            if (abs_base > 0).any()
            else 1.0
        )
        abs_base_clipped = np.maximum(abs_base, floor_q)
        w = 1.0 / abs_base_clipped
        # Normalise to mean 1 so loss scale matches unweighted.
        w *= w.size / w.sum()
        sample_weight = w
    return sample_weight


def fit_inner_estimator(est: Any, X_valid: Any, t_train: np.ndarray, sample_weight: np.ndarray | None, fit_kwargs: dict[str, Any]) -> tuple[Any, Any]:
    """Clone and fit the inner estimator on the T-scale target; returns ``(fitted_estimator, X_valid_as_seen_by_inner)``."""
    # Fit the inner estimator. Clone first so the unfitted
    # prototype passed in stays untouched and sklearn.clone() of
    # the wrapper produces a fresh inner.
    estimator = clone(est.base_estimator)
    X_valid = frame_for_inner(estimator, X_valid)  # a polars frame only for an inner that reads polars
    # Monotonic-constraint passthrough. ``X_valid`` is the exact frame the inner trains on (group_column already dropped), so its width is the post-drop feature count the constraint must match. Validated here (not in __init__) because the dropped-column set is only known once transform / group_column resolve at fit. See the class docstring for the T-scale semantics.
    if est.monotone_constraints is not None:
        est._apply_monotone_constraints(estimator, est._count_feature_columns(X_valid))
    if sample_weight is not None:
        # Signature-gate sample_weight rather than an ``except TypeError``
        # retry. The retry mis-attributed a TypeError raised deep inside a
        # weight-AWARE inner fit (e.g. a downstream dtype/shape bug) to "no
        # sample_weight support", silently dropping the weighting on a
        # re-fit. ``has_fit_parameter`` (sklearn) is the canonical check and
        # also resolves metadata-routed / delegated fits.
        if _estimator_fit_accepts_sample_weight(estimator):
            estimator.fit(X_valid, t_train, sample_weight=sample_weight, **fit_kwargs)
        else:
            logger.info(
                "[CompositeTargetEstimator] inner estimator '%s' does not accept " "sample_weight; ignoring.",
                type(est.base_estimator).__name__,
            )
            estimator.fit(X_valid, t_train, **fit_kwargs)
    else:
        estimator.fit(X_valid, t_train, **fit_kwargs)
    return estimator, X_valid


def compute_clip_bounds(y_train: np.ndarray, t_train: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return ``(y_clip_low, y_clip_high, y_train_median, t_clip_low, t_clip_high)`` for the fitted-params dict."""
    # Stash post-inverse y-clip bounds + train median for fallback. Use
    # np.nanmedian so a NaN-bearing y_train doesn't silently make
    # y_train_median == NaN -- the fallback constant is used at predict time
    # when the transform is unsafe, and a NaN there would poison every row
    # that triggered the fallback.
    y_clip_low, y_clip_high = _y_train_clip_bounds(y_train)
    y_train_median = float(np.nanmedian(y_train))
    if not np.isfinite(y_train_median):
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "composite_estimator: y_train has no finite values; " "y_train_median fallback set to 0.0. Predict-time " "fallback will return 0 instead of NaN.",
        )
        y_train_median = 0.0

    # T-scale clip bounds. Inner predict() can blow out far past the T-train
    # envelope on heavy-tail residual targets (prod: un-calibrated XGB Huber
    # produced T_hat in [-50,+340] for T_train in [-50,+50]; the y-clip only
    # catches the part outside [y_min,y_max], not the extrapolated middle).
    # T-clip BEFORE inverse uses MAD-scaled bounds (median(T) +/- 10*MAD), so
    # in-distribution mass is unaffected while gross blow-up is bounded.
    t_finite = t_train[np.isfinite(t_train)]
    if t_finite.size >= 10:
        t_med = float(np.median(t_finite))
        t_mad = float(np.median(np.abs(t_finite - t_med)))
        if t_mad > 0:
            t_clip_low = t_med - 10.0 * t_mad
            t_clip_high = t_med + 10.0 * t_mad
            t_observed_min = float(t_finite.min())
            t_observed_max = float(t_finite.max())
            t_clip_low = min(t_clip_low, t_observed_min)
            t_clip_high = max(t_clip_high, t_observed_max)
        else:
            t_clip_low, t_clip_high = float("-inf"), float("inf")
    else:
        t_clip_low, t_clip_high = float("-inf"), float("inf")
    return y_clip_low, y_clip_high, y_train_median, t_clip_low, t_clip_high


def stamp_feature_counts_and_stats(est: Any, X: Any, estimator: Any, transform: Any) -> None:
    """Record ``feature_names_in_`` / the wrapper-seen feature count and initialise the live predict counters."""
    # Best-effort feature names (pandas / polars). Narrowed except: an ndarray X legitimately lacks ``.columns`` (AttributeError); anything else (e.g. a polars frame whose column iteration raises) is anomalous and should surface, not silently swallow under bare Exception.
    try:
        est.feature_names_in_ = list(X.columns)
    except AttributeError:
        pass
    # Stamp the feature count the WRAPPER saw at the X boundary so the
    # class-level ``n_features_in_`` property reports a value consistent
    # with ``feature_names_in_``. For grouped transforms the inner is fit on
    # F-1 columns (group_column dropped), so delegating to ``inner.n_features_in_``
    # would under-count by one and break the sklearn
    # ``n_features_in_ == len(feature_names_in_)`` invariant. Prefer the
    # column count when available; else fall back to the inner's count.
    _names = getattr(est, "feature_names_in_", None)
    if _names is not None:
        est._n_features_in_wrapper = len(_names)
    else:
        _inner_n = getattr(estimator, "n_features_in_", None)
        if _inner_n is not None:
            # ndarray X with grouped transform: inner saw F-1 (group col was
            # dropped); add it back so the count matches what the wrapper
            # ingested.
            _extra = 1 if (transform.requires_groups and est.group_column) else 0
            est._n_features_in_wrapper = int(_inner_n) + _extra
    # Live counters initialised lazily by predict.
    est.runtime_stats_ = {
        "predict_calls": 0,
        "predict_rows_total": 0,
        "domain_violation_rows": 0,
        "y_clip_low_hits": 0,
        "y_clip_high_hits": 0,
        "t_clip_low_hits": 0,
        "t_clip_high_hits": 0,
    }
