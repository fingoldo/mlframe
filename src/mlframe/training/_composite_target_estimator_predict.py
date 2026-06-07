"""Predict family for ``CompositeTargetEstimator``: ``_predict_unclipped``, ``predict_pre_clip``, ``predict``, ``predict_quantile``.

Carved out of ``_composite_target_estimator.py`` to keep the parent below the 1k-line monolith threshold. Functions here become bound methods on ``CompositeTargetEstimator`` at the parent's bottom via direct class-attribute assignment. Behavioural identity is preserved bit-for-bit.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from .composite_estimator import _extract_groups
from .composite.transforms import get_transform

logger = logging.getLogger(__name__)


def _predict_unclipped(self, X: Any) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Internal: compute the pre-clip y-scale prediction plus the row count and the params dict.

    Pulled out of ``predict`` so callers that need the raw (un-clipped) y-hat for diagnostics (e.g. honest pre-clip train
    RMSE - which the clip cannot improve on rows that are in-envelope by construction) can get it without re-running
    the inverse twice. ``predict`` is implemented as a thin clip-applying wrapper around this.
    """
    if not hasattr(self, "estimator_"):
        raise RuntimeError(
            "CompositeTargetEstimator.predict called before fit."
        )
    transform = get_transform(self.transform_name)
    params = self.fitted_params_

    # Determine the row count for the prediction. For unary
    # transforms (``requires_base=False``) we cannot rely on a base
    # column for the size; ask the inner estimator's prediction
    # length implicitly by deferring extraction.
    if transform.requires_base:
        base_columns = self._resolve_base_columns()
        base_arr = self._extract_base_for_transform(X, base_columns)
    else:
        # Placeholder zeros sized to predict-time inputs. We let the
        # inner predict (below) drive the row count, and re-size the
        # placeholder after t_hat is known. Cheap (zeros allocate
        # lazily on virtually-mmapped memory).
        base_arr = None
    # Domain check at predict: y is unknown, so we ask the transform
    # to gate on base-side conditions only (e.g. base > 0 for
    # logratio, |base| > 0 for ratio). The ``y=None`` sentinel is
    # part of the domain_check contract -- transforms whose forward
    # is purely y-side (none here) would still need a y; the four
    # core transforms all degrade cleanly. For unary transforms
    # (no base) we delegate to the transform's own domain_check on
    # ``y=None, base=None`` which the registry adapter handles by
    # returning all-True for the t_hat row count below.
    if transform.requires_base:
        domain_ok = transform.domain_check(None, base_arr)
    else:
        domain_ok = None  # sized after t_hat is computed

    # Grouped-transform support at predict.
    # Extract per-row group labels and thread them as kwargs to the
    # inverse call. Unseen groups (not present at fit) fall back to
    # global alpha/beta inside the transform's inverse impl, so no
    # caller-side handling is needed here.
    inverse_kwargs: dict[str, Any] = {}
    if transform.requires_groups:
        if not self.group_column:
            raise ValueError(
                f"CompositeTargetEstimator.predict: transform "
                f"'{self.transform_name}' requires groups but "
                f"``group_column`` is not configured."
            )
        inverse_kwargs["groups"] = _extract_groups(X, self.group_column)

    # Pass X through to the inner unchanged. NEVER materialise the
    # frame here -- on a 100 GB polars frame a silent conversion
    # blows the host out of memory. Caller is responsible for
    # ensuring the frame type is acceptable to the inner estimator
    # (mlframe strategies handle this at the suite level).
    # For grouped transforms: strip group_column from X before
    # predict so the inner doesn't see the (typically string)
    # plumbing column -- same logic as fit().
    X_for_inner = (
        self._drop_columns(X, [self.group_column])
        if transform.requires_groups and self.group_column
        else X
    )
    t_hat = np.asarray(
        self.estimator_.predict(X_for_inner), dtype=np.float64,
    ).reshape(-1)

    # T-scale clip BEFORE inverse. Heavy-tail composite targets
    # (observed in prod on XGB) can blow predictions
    # 30x outside the T-train envelope; the post-inverse y-clip
    # only catches what falls outside the y-train range, missing
    # the wildly-extrapolated middle. T-clip here bounds the
    # blow-up at its source while leaving in-distribution T
    # predictions untouched. NaN-only batches and missing bounds
    # (legacy fitted_params without t_clip_*) are no-ops.
    t_clip_low = params.get("t_clip_low", float("-inf"))
    t_clip_high = params.get("t_clip_high", float("+inf"))
    if (np.isfinite(t_clip_low) or np.isfinite(t_clip_high)):
        t_low_hits = int(np.sum(t_hat < t_clip_low))
        t_high_hits = int(np.sum(t_hat > t_clip_high))
        if t_low_hits or t_high_hits:
            t_hat = np.clip(t_hat, t_clip_low, t_clip_high)
            logger.warning(
                "[CompositeTargetEstimator] T-clip applied transform='%s' "
                "base='%s': %d row(s) below %.4g, %d row(s) above %.4g. "
                "Inner predict produced T-scale outliers (likely Huber-slope "
                "miscalibration or extreme-tail blow-up).",
                self.transform_name, self.base_column,
                t_low_hits, t_clip_low, t_high_hits, t_clip_high,
            )

    # Unary transforms have no base column at predict-time; size the
    # placeholder + domain mask to match t_hat. Inverse ignores the
    # base arg (registry adapter; see composite_transforms.py).
    if not transform.requires_base:
        base_arr = np.zeros_like(t_hat)
        domain_ok = np.ones_like(t_hat, dtype=bool)

    # Apply inverse only on valid rows; fill the rest with fallback.
    if domain_ok.all():
        y_hat = transform.inverse(t_hat, base_arr, params, **inverse_kwargs)
    else:
        y_hat = np.full_like(t_hat, fill_value=np.nan, dtype=np.float64)
        # Inverse on valid rows only; placeholder base for invalid
        # rows is irrelevant since we overwrite immediately.
        base_safe = np.where(domain_ok, base_arr, 1.0)
        y_hat_valid = transform.inverse(
            t_hat, base_safe, params, **inverse_kwargs,
        )
        y_hat[domain_ok] = y_hat_valid[domain_ok]
        # Fallback for invalid rows.
        if self.fallback_predict == "y_train_median":
            y_hat[~domain_ok] = params["y_train_median"]
        elif self.fallback_predict == "nan":
            pass  # already NaN
        else:
            raise ValueError(
                f"CompositeTargetEstimator: unknown fallback_predict "
                f"'{self.fallback_predict}'; choose 'y_train_median' or 'nan'."
            )

    n_violation = int((~domain_ok).sum())
    n_rows = int(t_hat.size)
    meta = {"params": params, "n_violation": n_violation, "n_rows": n_rows}
    return y_hat, n_rows, meta


def predict_pre_clip(self, X: Any) -> np.ndarray:
    """Return the inverse-of-transform y-prediction WITHOUT the train-envelope clip applied.

    The post-hoc clip ``[y_clip_low, y_clip_high]`` is a no-op on train rows by construction (they ARE the envelope) so any
    train metric computed on the clipped prediction is identical to the un-clipped one. The clip only does work on val /
    test rows that drift outside the train range. Computing pre- AND post-clip RMSE separately exposes the clip's actual
    contribution per split instead of folding the no-op train case into a falsely "improved" headline number.
    """
    y_hat_unclipped, _, _ = self._predict_unclipped(X)
    return y_hat_unclipped


def predict(self, X: Any) -> np.ndarray:
    y_hat, n, meta = self._predict_unclipped(X)
    params = meta["params"]
    n_violation = meta["n_violation"]
    t_hat_size = meta["n_rows"]

    # Post-inverse y-clip. Prediction outside the train envelope is
    # almost always exp() / division blow-up; clip and count for
    # observability.
    low = params["y_clip_low"]
    high = params["y_clip_high"]
    low_hits = int(np.sum(y_hat < low))
    high_hits = int(np.sum(y_hat > high))
    if low_hits or high_hits:
        y_hat = np.clip(y_hat, low, high)

    # Update runtime_stats_ cumulatively.
    n = t_hat_size
    rs = self.runtime_stats_
    rs["predict_calls"] += 1
    rs["predict_rows_total"] += n
    rs["domain_violation_rows"] += n_violation
    rs["y_clip_low_hits"] += low_hits
    rs["y_clip_high_hits"] += high_hits

    # Optional metrics callback (Prometheus / StatsD / DataDog hook).
    # Failures are logged at DEBUG and swallowed -- monitoring must
    # never break inference.
    cb = getattr(self, "runtime_stats_callback", None)
    if cb is not None:
        try:
            cb({
                "transform_name": self.transform_name,
                "base_column": self.base_column,
                "batch_n": n,
                "batch_domain_violation_rows": n_violation,
                "batch_y_clip_low_hits": low_hits,
                "batch_y_clip_high_hits": high_hits,
                "cumulative_predict_calls": rs["predict_calls"],
                "cumulative_predict_rows_total": rs["predict_rows_total"],
                "cumulative_domain_violation_rows": rs["domain_violation_rows"],
                "cumulative_y_clip_low_hits": rs["y_clip_low_hits"],
                "cumulative_y_clip_high_hits": rs["y_clip_high_hits"],
            })
        except Exception as cb_err:
            logger.debug(
                "[CompositeTargetEstimator] runtime_stats_callback failed: %s",
                cb_err,
            )
    return y_hat


def predict_quantile(
    self, X: Any, alpha: float | Sequence[float] = 0.5,
) -> np.ndarray:
    """y-scale quantile prediction by inverting the inner's
    T-scale quantile.

    Requires the inner estimator to expose ``predict_quantile(X,
    alpha)`` -- e.g. CatBoost ``MultiQuantile``, LightGBM
    ``quantile_alpha``, sklearn ``QuantileRegressor``. The wrapper
    calls inner -> ``T_q`` then applies the transform's inverse.

    Accepts either a scalar ``alpha`` (returns ``(n_samples,)``) or
    an array-like of K quantile levels (returns ``(n_samples, K)``);
    the 2-D path preserves the quantile dimension so per-quantile
    ensemble blending (``predict_quantile_ensemble``) can operate
    column-wise instead of collapsing to a single point estimate.

    **Quantile preservation under inverse**:

    | transform        | inverse                  | preserves quantiles?               |
    |------------------|--------------------------|------------------------------------|
    | ``diff``           | ``T + base``               | always (monotonic in T)            |
    | ``linear_residual``| ``T + alpha*base + beta``  | always                             |
    | ``logratio``       | ``base * exp(softcap(T))`` | yes (base > 0 already required)    |
    | ``ratio``          | ``T * base``               | flips when ``base < 0``; raises    |

    For ``ratio`` with mixed-sign base raises ``NotImplementedError``
    rather than silently swap the high / low quantiles.
    """
    if not hasattr(self, "estimator_"):
        raise RuntimeError(
            "CompositeTargetEstimator.predict_quantile called before fit."
        )
    inner = self.estimator_
    if not hasattr(inner, "predict_quantile"):
        raise NotImplementedError(
            f"inner estimator {type(inner).__name__!r} does not expose "
            "predict_quantile(X, alpha); wrap a quantile regressor "
            "(CatBoost MultiQuantile / LightGBM quantile_alpha / "
            "sklearn QuantileRegressor) instead."
        )

    transform = get_transform(self.transform_name)
    params = self.fitted_params_
    base_columns = self._resolve_base_columns()
    base_arr = self._extract_base_for_transform(X, base_columns)

    # Sign-flip guard for ratio: T < 0 and base < 0 produces
    # positive y; high T-quantile would swap to low y-quantile.
    if self.transform_name == "ratio":
        if np.any(base_arr < 0):
            raise NotImplementedError(
                "predict_quantile is not supported for transform 'ratio' "
                "when base contains negative values: y = T * base flips "
                "the quantile ordering on negative-base rows. Use "
                "predict() for point predictions or switch transform."
            )

    # logratio domain check: base must be > 0 (already required at
    # fit time, but predict-time inputs may differ).
    if self.transform_name == "logratio":
        if np.any(base_arr <= 0):
            raise NotImplementedError(
                "predict_quantile for transform 'logratio' requires "
                "base > 0 on every row. Filter the input or use "
                "predict() with the wrapper's domain fallback."
            )

    alpha_is_scalar = np.isscalar(alpha)
    try:
        t_raw = np.asarray(
            inner.predict_quantile(X, alpha), dtype=np.float64,
        )
    except TypeError:
        # Some libs name the kwarg differently (e.g. quantile=, q=).
        t_raw = np.asarray(
            inner.predict_quantile(X, alpha=alpha), dtype=np.float64,
        )

    # Preserve quantile dimensionality: scalar alpha -> 1-D (n_samples,);
    # array alpha -> 2-D (n_samples, K). Flattening unconditionally would
    # collapse a (n_samples, K) multi-quantile head into one mean-like
    # vector and destroy the predictive interval downstream blenders need.
    low = params.get("y_clip_low", float("-inf"))
    high = params.get("y_clip_high", float("inf"))
    if alpha_is_scalar:
        t_q = t_raw.reshape(-1)
        y_q = transform.inverse(t_q, base_arr, params)
        return np.clip(y_q, low, high)

    if t_raw.ndim == 1:
        # Inner emits per-alpha 1-D and was called with a vector -- broadcast
        # against the single base column we already extracted (which is 1-D).
        y_q = transform.inverse(t_raw, base_arr, params)
        return np.clip(y_q, low, high).reshape(-1, 1)
    if t_raw.ndim != 2:
        raise ValueError(
            f"CompositeTargetEstimator.predict_quantile: inner returned ndim={t_raw.ndim}; expected 1 or 2."
        )
    # Per-column inverse: base_arr is (n_samples,), so reshape to broadcast.
    cols: list[np.ndarray] = []
    for k in range(t_raw.shape[1]):
        y_col = transform.inverse(t_raw[:, k], base_arr, params)
        cols.append(np.clip(y_col, low, high))
    return np.column_stack(cols)
