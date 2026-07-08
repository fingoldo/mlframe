"""Predict family for ``CompositeTargetEstimator``: ``_predict_unclipped``, ``predict_pre_clip``, ``predict``, ``predict_quantile``.

Functions here become bound methods on ``CompositeTargetEstimator`` at the parent's bottom via direct class-attribute assignment.

The base-side domain mask, the T-scale clip, the domain-aware inverse-with-fallback, and the runtime-stats / callback recording are factored into module-level helpers (``_compute_base_domain_ok`` / ``_apply_t_clip`` / ``_inverse_with_fallback`` / ``_record_runtime_stats``) so the point-predict and quantile-predict paths share one implementation and gate NaN / out-of-domain bases identically.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from sklearn.exceptions import NotFittedError

from . import _extract_groups
from . import _soft_shrink as _soft_shrink
from ..transforms import get_transform

logger = logging.getLogger(__name__)


def _compute_base_domain_ok(transform, base_arr: np.ndarray, params: dict[str, Any]):
    """Predict-time base-side domain mask for a ``requires_base`` transform.

    Mirrors the gate ``_predict_unclipped`` builds: the params-free
    ``domain_check`` (with ``y=None`` at predict, base-side conditions only)
    refined by the optional fitted-params-aware ``domain_check_fitted`` hook so
    rows whose learned denominator / offset lands in an eps-clamped band route
    to the fallback instead of a distorted inverse.

    Returns a 1-D boolean ndarray; absent ``domain_check_fitted`` (the 30+
    params-free transforms) leaves the gate bit-identical to the plain check.
    """
    domain_ok = np.asarray(transform.domain_check(None, base_arr))
    _dcf = getattr(transform, "domain_check_fitted", None)
    if _dcf is not None:
        _domain_ok_fitted = np.asarray(_dcf(None, base_arr, params), dtype=bool)
        if _domain_ok_fitted.shape == domain_ok.shape:
            domain_ok = domain_ok & _domain_ok_fitted
    return domain_ok


def _apply_t_clip(self, t_hat: np.ndarray, params: dict[str, Any]):
    """Clip ``t_hat`` to the fitted T-train envelope BEFORE the inverse.

    Heavy-tail composite targets (observed in prod on XGB) can blow predictions
    far outside the T-train envelope; the post-inverse y-clip only catches what
    falls outside the y-train range, missing the wildly-extrapolated middle.
    T-clip here bounds the blow-up at its source while leaving in-distribution T
    untouched. NaN-only batches and missing bounds (legacy fitted_params without
    ``t_clip_*``) are no-ops.

    Returns ``(t_hat_maybe_clipped, t_low_hits, t_high_hits)``. The hit counts
    are surfaced to ``runtime_stats_`` / the callback by callers so the clip is
    observable without scraping logs.
    """
    t_clip_low = params.get("t_clip_low", float("-inf"))
    t_clip_high = params.get("t_clip_high", float("+inf"))
    t_low_hits = t_high_hits = 0
    if np.isfinite(t_clip_low) or np.isfinite(t_clip_high):
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
    return t_hat, t_low_hits, t_high_hits


def _finite_median_fallback(params: dict[str, Any]) -> float:
    """Resolve the ``y_train_median`` fallback constant to a FINITE value.

    The main ``fit()`` path coerces a non-finite ``y_train_median`` to ``0.0``,
    but the ``from_fitted_inner`` / discovery path can stash ``float('nan')``
    when ``y_train`` had no finite values (degenerate domain). Under
    ``fallback_predict='y_train_median'`` that NaN would be the fill value for
    every domain-violating / non-finite-inverse row, silently leaking NaN
    through and breaking the all-finite predict contract. Coerce to ``0.0`` here
    so the fallback is always finite regardless of which fit path built params.
    """
    med = params.get("y_train_median", 0.0)
    return float(med) if np.isfinite(med) else 0.0


def _inverse_with_fallback(
    self, transform, t_hat: np.ndarray, base_arr: np.ndarray,
    domain_ok: np.ndarray, params: dict[str, Any], inverse_kwargs: dict[str, Any],
) -> np.ndarray:
    """Invert ``t_hat`` on domain-valid rows; route the rest to the fallback.

    Domain-violating rows get ``y_train_median`` (or NaN under
    ``fallback_predict='nan'``); a transform inverse that produces NaN/inf even
    on a domain-valid row (Yeo-Johnson saturation, logratio exp-overflow) is
    routed to the same fallback so a single poisoned row never leaks through.
    Shared by ``_predict_unclipped`` and ``predict_quantile`` so both paths gate
    NaN/out-of-domain bases identically.
    """
    if domain_ok.all():
        y_hat = np.asarray(
            transform.inverse(t_hat, base_arr, params, **inverse_kwargs),
            dtype=np.float64,
        ).reshape(-1)
    else:
        y_hat = np.full_like(t_hat, fill_value=np.nan, dtype=np.float64)
        # Inverse on valid rows only; placeholder base for invalid rows is
        # irrelevant since we overwrite immediately. domain_ok is (n,);
        # base_arr is (n,) single-base or (n,K) multi-base, so broadcast the
        # row mask along the column axis (a flat np.where would raise a
        # (n,) vs (n,K) broadcast ValueError for K>=2).
        mask = domain_ok if base_arr.ndim == 1 else domain_ok[:, None]
        base_safe = np.where(mask, base_arr, 1.0)
        y_hat_valid = np.asarray(
            transform.inverse(t_hat, base_safe, params, **inverse_kwargs),
            dtype=np.float64,
        ).reshape(-1)
        y_hat[domain_ok] = y_hat_valid[domain_ok]
        if self.fallback_predict == "y_train_median":
            y_hat[~domain_ok] = _finite_median_fallback(params)
        elif self.fallback_predict == "nan":
            pass  # already NaN
        else:
            raise ValueError(f"CompositeTargetEstimator: unknown fallback_predict " f"'{self.fallback_predict}'; choose 'y_train_median' or 'nan'.")

    # General non-finite guard: a transform inverse can produce NaN/inf even on
    # a domain-valid row. np.clip cannot repair NaN, so a single poisoned row
    # would otherwise leak straight through. Route to the same fallback.
    nonfinite = ~np.isfinite(y_hat)
    if nonfinite.any():
        if self.fallback_predict == "y_train_median":
            y_hat[nonfinite] = _finite_median_fallback(params)
        # 'nan' fallback: leave as-is (caller opted into NaN sentinels).
        logger.warning(
            "[CompositeTargetEstimator] transform='%s' base='%s': %d row(s) "
            "inverted to non-finite y; routed to fallback_predict='%s'.",
            self.transform_name, self.base_column,
            int(nonfinite.sum()), self.fallback_predict,
        )
    return y_hat


# Runtime-stats keys that accumulate across predict calls. Listed once so
# legacy fitted instances (built before a key existed) can be back-filled to 0
# on first use rather than KeyError-ing on the ``+=`` update.
_RUNTIME_STAT_KEYS = (
    "predict_calls", "predict_rows_total", "domain_violation_rows",
    "y_clip_low_hits", "y_clip_high_hits", "t_clip_low_hits", "t_clip_high_hits",
)


def _record_runtime_stats(
    self, n: int, n_violation: int, low_hits: int, high_hits: int,
    t_low_hits: int, t_high_hits: int,
) -> None:
    """Accumulate per-batch counters into ``runtime_stats_`` and fire the callback.

    Shared by ``predict`` and ``predict_quantile`` so both surface the same
    observability payload -- including the T-clip hit counts that were
    previously logged at WARNING every batch then discarded. Missing keys (a
    pre-T-clip-stats fitted instance unpickled after this change) are seeded to
    0 so the ``+=`` never KeyErrors. The callback's failures are logged at DEBUG
    and swallowed -- monitoring must never break inference.
    """
    rs = self.runtime_stats_
    for _k in _RUNTIME_STAT_KEYS:
        rs.setdefault(_k, 0)
    rs["predict_calls"] += 1
    rs["predict_rows_total"] += n
    rs["domain_violation_rows"] += n_violation
    rs["y_clip_low_hits"] += low_hits
    rs["y_clip_high_hits"] += high_hits
    rs["t_clip_low_hits"] += t_low_hits
    rs["t_clip_high_hits"] += t_high_hits

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
                "batch_t_clip_low_hits": t_low_hits,
                "batch_t_clip_high_hits": t_high_hits,
                "cumulative_predict_calls": rs["predict_calls"],
                "cumulative_predict_rows_total": rs["predict_rows_total"],
                "cumulative_domain_violation_rows": rs["domain_violation_rows"],
                "cumulative_y_clip_low_hits": rs["y_clip_low_hits"],
                "cumulative_y_clip_high_hits": rs["y_clip_high_hits"],
                "cumulative_t_clip_low_hits": rs["t_clip_low_hits"],
                "cumulative_t_clip_high_hits": rs["t_clip_high_hits"],
            })
        except Exception as cb_err:
            logger.debug(
                "[CompositeTargetEstimator] runtime_stats_callback failed: %s",
                cb_err,
            )


def _predict_unclipped(self, X: Any) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Internal: compute the pre-clip y-scale prediction plus the row count and the params dict.

    Pulled out of ``predict`` so callers that need the raw (un-clipped) y-hat for diagnostics (e.g. honest pre-clip train
    RMSE - which the clip cannot improve on rows that are in-envelope by construction) can get it without re-running
    the inverse twice. ``predict`` is implemented as a thin clip-applying wrapper around this.
    """
    if not hasattr(self, "estimator_"):
        raise NotFittedError("CompositeTargetEstimator.predict called before fit.")
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
    # Base-side domain mask. The params-free ``domain_check`` cannot see learned
    # params, so ``_compute_base_domain_ok`` refines it with the
    # fitted-params-aware hook where present (centered_ratio's eps-floored
    # denominator) and is bit-identical for the 30+ params-free transforms.
    if transform.requires_base:
        domain_ok = _compute_base_domain_ok(transform, base_arr, params)
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
                f"CompositeTargetEstimator.predict: transform " f"'{self.transform_name}' requires groups but " f"``group_column`` is not configured."
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
    X_for_inner = self._drop_columns(X, [self.group_column]) if transform.requires_groups and self.group_column else X
    t_hat = np.asarray(
        self.estimator_.predict(X_for_inner), dtype=np.float64,
    ).reshape(-1)

    # T-scale clip BEFORE inverse (shared with predict_quantile). The hit
    # counts are returned so predict() can surface them in runtime_stats_ /
    # the callback instead of discarding them to a per-batch WARNING.
    t_hat, t_low_hits, t_high_hits = _apply_t_clip(self, t_hat, params)

    # Unary transforms have no base column at predict-time; size the
    # placeholder + domain mask to match t_hat. Inverse ignores the
    # base arg (registry adapter; see composite_transforms.py).
    if not transform.requires_base:
        base_arr = np.zeros_like(t_hat)
        domain_ok = np.ones_like(t_hat, dtype=bool)

    # Soft base-shrink (default ON): a base value OUTSIDE the fit-time calibration range is smoothly
    # shrunk toward the boundary so the base-additive inverse degrades gracefully instead of exploding on
    # unseen-group tails; in-range rows are byte-identical. Disabled/inapplicable -> base_eff IS base_arr.
    base_eff, shrunk_mask, deep_ood = _soft_shrink.compute(self, transform, base_arr, params)

    # Apply inverse only on valid rows; fill the rest with the fallback, with
    # the non-finite guard. Shared with predict_quantile so both paths gate
    # NaN/out-of-domain bases identically.
    y_hat = _inverse_with_fallback(
        self, transform, t_hat, base_eff, domain_ok, params, inverse_kwargs,
    )
    # Smart fallback for DEEPLY out-of-distribution rows: the causal lag when present, else the wrapper
    # fallback. Runs on the RAW (un-shrunk) base so a lag-as-base failsafe uses the true observed value.
    if deep_ood is not None:
        _soft_shrink.apply_smart_fallback(self, y_hat, deep_ood, base_arr, domain_ok, X, params)
    _soft_shrink.record_info(self, shrunk_mask, deep_ood, int(t_hat.size))

    n_violation = int((~domain_ok).sum())
    n_rows = int(t_hat.size)
    meta = {
        "params": params, "n_violation": n_violation, "n_rows": n_rows,
        "t_low_hits": t_low_hits, "t_high_hits": t_high_hits,
    }
    return y_hat, n_rows, meta


def predict_pre_clip(self, X: Any) -> np.ndarray:
    """Return the inverse-of-transform y-prediction WITHOUT the train-envelope clip applied.

    The post-hoc clip ``[y_clip_low, y_clip_high]`` is a no-op on train rows by construction (they ARE the envelope) so any
    train metric computed on the clipped prediction is identical to the un-clipped one. The clip only does work on val /
    test rows that drift outside the train range. Computing pre- AND post-clip RMSE separately exposes the clip's actual
    contribution per split instead of folding the no-op train case into a falsely "improved" headline number.
    """
    y_hat_unclipped, _, _ = self._predict_unclipped(X)
    return np.asarray(y_hat_unclipped)


def predict(self, X: Any) -> np.ndarray:
    """Predict on the original target scale (bound as ``CompositeTargetEstimator.predict``).

    Runs the inner estimator, inverts the fitted target transform (with domain-aware fallback for
    out-of-domain / NaN bases), then clips predictions to the fitted train envelope, counting
    violations for observability. Returns the original-scale ``y_hat``.
    """
    y_hat, n, meta = self._predict_unclipped(X)
    params = meta["params"]
    n_violation = meta["n_violation"]
    n = meta["n_rows"]

    # Post-inverse y-clip. Prediction outside the train envelope is
    # almost always exp() / division blow-up; clip and count for
    # observability.
    low = params["y_clip_low"]
    high = params["y_clip_high"]
    low_hits = int(np.sum(y_hat < low))
    high_hits = int(np.sum(y_hat > high))
    if low_hits or high_hits:
        y_hat = np.clip(y_hat, low, high)

    # Accumulate counters + fire the callback. The T-clip hit counts (computed
    # inside _predict_unclipped) flow through here so they are observable in
    # runtime_stats_ / the callback rather than only in a per-batch WARNING.
    _record_runtime_stats(
        self, n, n_violation, low_hits, high_hits,
        meta["t_low_hits"], meta["t_high_hits"],
    )
    return np.asarray(y_hat)


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

    Domain / clip parity with ``predict``: rows whose base fails the base-side
    domain check (NaN / out-of-domain) route to the ``fallback_predict`` value
    instead of returning a silent NaN quantile, the inner T-quantile is T-clipped
    to the fitted envelope before the inverse, and the per-batch counters
    (domain violations, T-clip hits) accumulate into ``runtime_stats_`` and fire
    the ``runtime_stats_callback``.
    """
    if not hasattr(self, "estimator_"):
        raise NotFittedError("CompositeTargetEstimator.predict_quantile called before fit.")
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
    # Unary transforms (requires_base=False) have no base column; extracting
    # one raises 'base_columns is empty'. Mirror _predict_unclipped: defer to
    # a zeros placeholder sized to t_raw, which the unary inverse ignores.
    if transform.requires_base:
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

        # reciprocal_residual: y = 1/(T + 1/base) is DECREASING in T, so the
        # inverse swaps the quantile order (the alpha-quantile of T maps to the
        # (1-alpha)-quantile of y). Raise rather than silently return inverted
        # intervals.
        if self.transform_name == "reciprocal_residual":
            raise NotImplementedError(
                "predict_quantile is not supported for transform "
                "'reciprocal_residual': y = 1/(T + 1/base) is monotone "
                "DECREASING in T, which swaps the quantile ordering. Use "
                "predict() for point predictions or switch transform."
            )
    else:
        base_arr = None

    # Grouped-transform parity with predict(). The grouped inverse needs per-row
    # group labels, and the inner must NOT see the (string) group_column.
    inverse_kwargs: dict[str, Any] = {}
    X_for_inner = X
    if transform.requires_groups:
        if not self.group_column:
            raise ValueError(
                f"CompositeTargetEstimator.predict_quantile: transform " f"'{self.transform_name}' requires groups but group_column is " f"not configured."
            )
        inverse_kwargs["groups"] = _extract_groups(X, self.group_column)
        X_for_inner = self._drop_columns(X, [self.group_column])

    alpha_is_scalar = np.isscalar(alpha)
    try:
        t_raw = np.asarray(
            inner.predict_quantile(X_for_inner, alpha), dtype=np.float64,
        )
    except TypeError:
        # Some libs name the kwarg differently (e.g. quantile=, q=).
        t_raw = np.asarray(
            inner.predict_quantile(X_for_inner, alpha=alpha), dtype=np.float64,
        )

    # Base-side domain mask (parity with predict). A NaN / out-of-domain base
    # otherwise yields a silent NaN quantile; instead route those rows through
    # the same fallback predict() uses. Unary transforms have no base, so all
    # rows are in-domain and the placeholder base is ignored by the inverse.
    n_rows = int(t_raw.shape[0])
    if base_arr is None:
        base_arr = np.zeros(n_rows, dtype=np.float64)
        domain_ok = np.ones(n_rows, dtype=bool)
    elif transform.requires_base:
        domain_ok = _compute_base_domain_ok(transform, base_arr, params)
    else:
        domain_ok = np.ones(n_rows, dtype=bool)

    # Preserve quantile dimensionality: scalar alpha -> 1-D (n_samples,);
    # array alpha -> 2-D (n_samples, K). Flattening unconditionally would
    # collapse a (n_samples, K) multi-quantile head into one mean-like
    # vector and destroy the predictive interval downstream blenders need.
    low = params.get("y_clip_low", float("-inf"))
    high = params.get("y_clip_high", float("inf"))

    def _invert_one(t_col: np.ndarray) -> np.ndarray:
        """T-clip (parity with predict) + domain-aware inverse + y-clip for one
        quantile level. Accumulates the T-clip hits into the closure totals."""
        nonlocal _t_low_total, _t_high_total
        t_clipped, _tl, _th = _apply_t_clip(self, t_col.reshape(-1), params)
        _t_low_total += _tl
        _t_high_total += _th
        y_col = _inverse_with_fallback(
            self, transform, t_clipped, base_arr, domain_ok, params, inverse_kwargs,
        )
        return np.clip(y_col, low, high)

    _t_low_total = _t_high_total = 0
    n_violation = int((~domain_ok).sum())

    if alpha_is_scalar:
        y_q = _invert_one(t_raw)
        _record_runtime_stats(
            self, n_rows, n_violation, 0, 0, _t_low_total, _t_high_total,
        )
        return y_q

    if t_raw.ndim == 1:
        # Inner emits per-alpha 1-D and was called with a vector -- broadcast
        # against the single base column we already extracted (which is 1-D).
        y_q = _invert_one(t_raw).reshape(-1, 1)
        _record_runtime_stats(
            self, n_rows, n_violation, 0, 0, _t_low_total, _t_high_total,
        )
        return y_q
    if t_raw.ndim != 2:
        raise ValueError(f"CompositeTargetEstimator.predict_quantile: inner returned ndim={t_raw.ndim}; expected 1 or 2.")
    # Per-column inverse: base_arr is (n_samples,), so reshape to broadcast.
    cols = [_invert_one(t_raw[:, k]) for k in range(t_raw.shape[1])]
    _record_runtime_stats(
        self, n_rows, n_violation, 0, 0, _t_low_total, _t_high_total,
    )
    return np.column_stack(cols)
