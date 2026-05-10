"""Composite target transforms, estimator wrapper, and discovery.

Building blocks for composite-target discovery. This module ships:

1. The transform registry (forward / inverse / fit / domain check).
2. ``CompositeTargetEstimator`` -- sklearn-compatible wrapper that
   hides the transform-and-invert loop from downstream callers.
3. ``CompositeTargetDiscovery`` -- auto-finds the best (base, transform)
   pairs by MI gain over the raw target, with strict train-only
   fitting and forbidden-base filtering.

Concept. A composite target is a transform ``T = f(y, base)`` such
that the model learns ``T`` from features ``X`` (typically excluding
the dominant feature ``base``), and a wrapper applies ``f^{-1}`` at
predict time to recover ``y`` in the original scale. The structural
example: ``y = TVT`` and ``base = TVT_prev``, where the autoregressive
lag is captured natively by the transform and the model is forced to
explain the remaining residual.

Public surface
--------------
- :class:`Transform` -- frozen dataclass, one entry per transform.
- :data:`_TRANSFORMS_REGISTRY` / :func:`get_transform` /
  :func:`list_transforms` -- registry lookup.
- :class:`CompositeTargetEstimator` -- sklearn-compatible wrapper that
  fits an inner regressor on ``T`` and inverts at predict.
- :exc:`DomainViolationError`, :exc:`UnknownTransformError`.

Design choices
--------------
- Transforms are looked up by **name** at fit/predict time, never
  stored as per-instance callables. This keeps :func:`sklearn.clone`
  semantics honest, makes pickle work with the standard library
  (no closure traps -> no PII leakage via captured DataFrames), and
  lets the wrapper survive process boundaries (joblib, Optuna).
- Transforms are **frozen**: ``forward``, ``inverse``, ``fit``,
  ``domain_check`` are pure module-level functions registered in
  :data:`_TRANSFORMS_REGISTRY` at import time. Adding a new transform =
  one dataclass entry + one parametrized test row.
- Fitted parameters (``alpha``, ``beta``, MAD floor, post-inverse
  y-clip bounds) are computed **only on training rows passed to
  ``fit``**. The wrapper never re-fits at predict time; downstream
  composite-target discovery is responsible for using the same
  ``train_idx`` discipline at the screening step.
- Numerical safety: MAD-soft-cap with floor (against degenerate
  ``T_train`` collapsing to a constant), post-inverse y-clip to the
  ``[Q001/10, Q999*10]`` bounds of ``y_train`` (against ``exp(...)``
  blow-up in ``logratio``), and ``np.isfinite`` guards on incoming
  ``base`` values at predict (against adversarial ``+inf`` injection).

Out of scope for this module
----------------------------
- Discovery (auto-find ``base`` and best transform): future PR.
- Cross-target ensembling: future PR.
- ``base_margin`` / classification residuals: regression only here.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from timeit import default_timer as timer
from typing import (
    Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple, Union,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

logger = logging.getLogger(__name__)


# Soft-cap MAD floor: when MAD(T_train) is below
# ``_MAD_FLOOR_FRAC * std(y_train)``, we substitute the latter to keep
# the soft-cap bound numerically meaningful even if the transform
# produced a degenerate (near-constant) T on train. Without this,
# logratio's MAD-cap collapses to zero on degenerate train and every
# prediction inverts to ``base * exp(0) = base`` silently.
_MAD_FLOOR_FRAC: float = 1e-3

# Bounds for the post-inverse y-clip, expressed as multipliers on the
# ``[Q001(y_train), Q999(y_train)]`` envelope. Values outside this
# extended envelope are unphysical for the training distribution and
# almost certainly the result of ``exp`` / division blow-up.
_Y_CLIP_LOW_FRAC: float = 0.1
_Y_CLIP_HIGH_FRAC: float = 10.0

# Multiplier for MAD-soft-cap on T_hat (logratio in particular).
_MAD_SOFT_CAP_K: float = 10.0


class UnknownTransformError(KeyError):
    """Raised when a transform name is not in :data:`_TRANSFORMS_REGISTRY`."""


class DomainViolationError(ValueError):
    """Raised at fit time when the input domain is incompatible with the
    chosen transform (e.g. ``logratio`` requested but ``y`` contains
    non-positive values).

    At predict time we do NOT raise -- per-row violations are handled
    via fall-back values + counters logged on
    ``CompositeTargetEstimator.runtime_stats_``.
    """


# ----------------------------------------------------------------------
# Transform registry
# ----------------------------------------------------------------------

# Tags used to filter the registry into presets.
TAG_CORE: str = "core"           # diff / ratio / logratio / linear_residual
TAG_EXTENDED: str = "extended"   # placeholder; future presets may add more
TAG_REGRESSION: str = "regression"


@dataclass(frozen=True)
class Transform:
    """One row of the transform registry.

    The four functions form a contract:

    - ``fit(y_train, base_train)`` -> ``dict`` of transform-specific
      fitted parameters (e.g. ``{"alpha": float, "beta": float}``).
      Pure: must NOT mutate inputs and must NOT close over external
      state. The dict is JSON-serialisable.
    - ``forward(y, base, params)`` -> ``T``: applies the transform.
    - ``inverse(T_hat, base, params)`` -> ``y_hat``: applies the
      inverse. Wrapper additionally clips the output to the y-bounds
      stored alongside ``params``.
    - ``domain_check(y, base)`` -> boolean mask of valid rows. Wrapper
      uses this at fit-time to drop invalid rows BEFORE calling
      ``fit`` / ``forward``, and at predict-time to flag rows where
      the inverse cannot be applied cleanly (those rows fall back to
      ``y_train_median``).
    """

    name: str
    forward: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], np.ndarray]
    inverse: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], np.ndarray]
    fit: Callable[[np.ndarray, np.ndarray], Dict[str, Any]]
    domain_check: Callable[[np.ndarray, np.ndarray], np.ndarray]
    description: str
    tags: FrozenSet[str] = field(default_factory=frozenset)


# ----------------------------------------------------------------------
# diff: T = y - base. Always defined, no params, no domain restrictions.
# ----------------------------------------------------------------------

def _diff_fit(y: np.ndarray, base: np.ndarray) -> Dict[str, Any]:
    return {}


def _diff_forward(y: np.ndarray, base: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    return y - base


def _diff_inverse(t_hat: np.ndarray, base: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    return t_hat + base


def _diff_domain(y: Optional[np.ndarray], base: np.ndarray) -> np.ndarray:
    base_ok = np.isfinite(base)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)


# ----------------------------------------------------------------------
# ratio: T = y / base. Requires |base| >= eps.
# ----------------------------------------------------------------------

def _ratio_fit(y: np.ndarray, base: np.ndarray) -> Dict[str, Any]:
    # eps relative to the typical scale of base on train -- small enough
    # not to bias the transform but large enough to keep division
    # numerically clean. Stored in params so predict time uses the
    # SAME eps (no train/test drift).
    scale = float(np.median(np.abs(base[np.isfinite(base) & (base != 0)])))
    eps = max(scale * 1e-6, 1e-12) if scale > 0 else 1e-12
    return {"eps": eps}


def _ratio_forward(y: np.ndarray, base: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    eps = float(params["eps"])
    safe_base = np.where(np.abs(base) < eps, np.sign(base + 1e-300) * eps, base)
    return y / safe_base


def _ratio_inverse(t_hat: np.ndarray, base: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    return t_hat * base


def _ratio_domain(y: Optional[np.ndarray], base: np.ndarray) -> np.ndarray:
    base_ok = np.isfinite(base) & (np.abs(base) > 0)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)


# ----------------------------------------------------------------------
# logratio: T = log(y) - log(base). Requires y, base > 0.
# Inverse uses MAD-soft-cap on T_hat to prevent exp() blow-up.
# ----------------------------------------------------------------------

def _logratio_fit(y: np.ndarray, base: np.ndarray) -> Dict[str, Any]:
    # T_train computed in the valid domain (caller has already filtered).
    t_train = np.log(y) - np.log(base)
    median_t = float(np.median(t_train))
    mad_train = float(np.median(np.abs(t_train - median_t)))
    # Floor against degenerate T_train (constant on train) -- otherwise
    # MAD = 0 collapses every prediction to ``base * exp(median_t)``,
    # which still ranks as "naive baseline" at predict time but at
    # least does not distort in-distribution predictions.
    std_y = float(np.std(y))
    mad_floor = _MAD_FLOOR_FRAC * std_y if std_y > 0 else 1e-6
    mad_eff = max(mad_train, mad_floor)
    return {
        "median_t": median_t,
        "mad_train": mad_train,
        "mad_eff": mad_eff,
        "soft_cap_k": _MAD_SOFT_CAP_K,
    }


def _logratio_forward(y: np.ndarray, base: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    return np.log(y) - np.log(base)


def _logratio_inverse(t_hat: np.ndarray, base: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    median_t = float(params["median_t"])
    mad = float(params["mad_eff"])
    k = float(params["soft_cap_k"])
    # Soft-cap is centred on median(T_train), NOT on zero -- otherwise
    # any T distribution offset from zero (the typical case for
    # logratio when y and base have similar scale) gets clobbered by
    # the cap and inverse predictions collapse to ``base``.
    cap = k * mad
    t_capped = np.clip(t_hat, median_t - cap, median_t + cap)
    return base * np.exp(t_capped)


def _logratio_domain(y: Optional[np.ndarray], base: np.ndarray) -> np.ndarray:
    base_ok = np.isfinite(base) & (base > 0)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y) & (y > 0)


# ----------------------------------------------------------------------
# linear_residual: T = y - alpha*base - beta. OLS on train.
# ----------------------------------------------------------------------

def _linear_residual_fit(
    y: np.ndarray, base: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """OLS fit with optional sample weights.

    Weighted least squares is implemented in closed form via
    ``np.linalg.lstsq`` on the row-scaled system
    ``sqrt(w) * X * beta = sqrt(w) * y`` (standard reformulation).
    Weights are normalised to sum to ``len(y)`` so the fit's
    numerical scale matches the unweighted case (avoids tiny
    coefficients on small w values).
    """
    n = len(y)
    if n < 2:
        return {"alpha": 0.0, "beta": float(np.mean(y)) if n > 0 else 0.0}
    X = np.column_stack([base.astype(np.float64), np.ones(n, dtype=np.float64)])
    y_f = y.astype(np.float64)

    if sample_weight is None:
        coef, *_ = np.linalg.lstsq(X, y_f, rcond=None)
    else:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.size != n:
            raise ValueError(
                f"_linear_residual_fit: sample_weight length {w.size} != y length {n}"
            )
        # Drop non-positive weights silently; warn if all zero.
        finite = np.isfinite(w) & (w > 0)
        if not finite.any():
            return {"alpha": 0.0, "beta": float(np.mean(y_f))}
        # Normalise weights to mean 1 so the system has the same
        # numerical scale as the unweighted version. lstsq handles
        # rank-deficient cases.
        w_norm = w[finite]
        w_norm = w_norm * (n / w_norm.sum())
        sw = np.sqrt(w_norm)
        X_w = X[finite] * sw[:, None]
        y_w = y_f[finite] * sw
        coef, *_ = np.linalg.lstsq(X_w, y_w, rcond=None)
    alpha = float(coef[0])
    beta = float(coef[1])
    return {"alpha": alpha, "beta": beta}


def _linear_residual_forward(
    y: np.ndarray, base: np.ndarray, params: Dict[str, Any],
) -> np.ndarray:
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    return y - alpha * base - beta


def _linear_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: Dict[str, Any],
) -> np.ndarray:
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    return t_hat + alpha * base + beta


def _linear_residual_domain(y: Optional[np.ndarray], base: np.ndarray) -> np.ndarray:
    base_ok = np.isfinite(base)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)


# ----------------------------------------------------------------------
# Registry and lookup
# ----------------------------------------------------------------------

_TRANSFORMS_REGISTRY: Dict[str, Transform] = {
    "diff": Transform(
        name="diff",
        forward=_diff_forward,
        inverse=_diff_inverse,
        fit=_diff_fit,
        domain_check=_diff_domain,
        description="T = y - base. Inverse y_hat = T_hat + base. No fitted parameters.",
        tags=frozenset({TAG_CORE, TAG_REGRESSION}),
    ),
    "ratio": Transform(
        name="ratio",
        forward=_ratio_forward,
        inverse=_ratio_inverse,
        fit=_ratio_fit,
        domain_check=_ratio_domain,
        description=(
            "T = y / base. Inverse y_hat = T_hat * base. Requires |base| > 0; "
            "fitted eps stored from train scale."
        ),
        tags=frozenset({TAG_CORE, TAG_REGRESSION}),
    ),
    "logratio": Transform(
        name="logratio",
        forward=_logratio_forward,
        inverse=_logratio_inverse,
        fit=_logratio_fit,
        domain_check=_logratio_domain,
        description=(
            "T = log(y) - log(base). Inverse y_hat = base * exp(softcap(T_hat)). "
            "Requires y, base > 0."
        ),
        tags=frozenset({TAG_CORE, TAG_REGRESSION}),
    ),
    "linear_residual": Transform(
        name="linear_residual",
        forward=_linear_residual_forward,
        inverse=_linear_residual_inverse,
        fit=_linear_residual_fit,
        domain_check=_linear_residual_domain,
        description=(
            "T = y - alpha*base - beta with (alpha, beta) fitted via OLS on train. "
            "Inverse y_hat = T_hat + alpha*base + beta."
        ),
        tags=frozenset({TAG_CORE, TAG_REGRESSION}),
    ),
}


def get_transform(name: str) -> Transform:
    """Lookup helper. Raises :exc:`UnknownTransformError` for typos."""
    try:
        return _TRANSFORMS_REGISTRY[name]
    except KeyError:
        raise UnknownTransformError(
            f"Unknown transform '{name}'. Registered: {sorted(_TRANSFORMS_REGISTRY)}"
        )


def list_transforms(*, tags: Optional[FrozenSet[str]] = None) -> List[str]:
    """Return registered transform names, optionally filtered by tag
    intersection (any-of: a transform passes if it has at least one of
    the requested tags)."""
    if tags is None:
        return sorted(_TRANSFORMS_REGISTRY)
    return sorted(
        name for name, t in _TRANSFORMS_REGISTRY.items() if t.tags & tags
    )


# ----------------------------------------------------------------------
# CompositeTargetEstimator wrapper
# ----------------------------------------------------------------------

def _y_train_clip_bounds(y_train: np.ndarray) -> Tuple[float, float]:
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


def _to_1d_numpy(arr: Any) -> np.ndarray:
    if hasattr(arr, "to_numpy"):
        out = arr.to_numpy()
    elif hasattr(arr, "values"):
        out = arr.values
    else:
        out = np.asarray(arr)
    return np.asarray(out).reshape(-1)


def _extract_base(X: Any, base_column: str) -> np.ndarray:
    """Pull base values from X (pandas / polars / structured ndarray).

    Raises ``KeyError`` with a helpful message if the column is missing
    -- this most commonly bites callers who configured MRMR / RFECV
    that dropped the base column before reaching the wrapper. The
    message points at the fix (``forced_keep_columns`` in the feature
    selection config).
    """
    # Polars
    if hasattr(X, "to_pandas") and not isinstance(X, pd.DataFrame):
        if base_column not in X.columns:
            raise KeyError(
                f"CompositeTargetEstimator: base column '{base_column}' missing from X. "
                "If feature selection (MRMR/RFECV) is dropping it, add base_column "
                "to forced_keep_columns in the feature selection config."
            )
        return np.asarray(X.get_column(base_column).to_numpy()).astype(np.float64)
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


class CompositeTargetEstimator(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper that fits an inner regressor on a
    transformed target and inverts at predict time.

    Parameters
    ----------
    base_estimator
        Any sklearn-compatible regressor with ``fit(X, y)`` /
        ``predict(X)``. The wrapper clones it at ``fit`` time so the
        unfitted prototype passed in stays clean. ``feature_importances_``,
        ``columns``, ``get_booster()``, and other common attributes are
        delegated through the wrapper transparently.
    transform_name
        One of :func:`list_transforms`. Determines the forward / inverse
        applied to the target.
    base_column
        Name of the column in ``X`` carrying the base feature. Required:
        the wrapper extracts ``base`` from this column at both ``fit``
        and ``predict`` time. If the inner feature-selection step drops
        the column, ``predict`` will raise ``KeyError``.
    fallback_predict
        Strategy for rows where the inverse transform cannot be applied
        (e.g. ``base[row] = inf`` for ``ratio``, or ``base[row] <= 0``
        for ``logratio``). Default ``"y_train_median"`` substitutes the
        median of the training target; ``"nan"`` returns NaN so callers
        can decide.
    drop_invalid_rows
        If True (default), rows that fail ``transform.domain_check`` at
        fit time are dropped before fitting the inner estimator. If
        False, ``fit`` raises :exc:`DomainViolationError` instead.

    Attributes set at fit
    ---------------------
    fitted_params_
        Dict carrying transform-specific fitted parameters (e.g.
        ``alpha``, ``beta`` for linear_residual) plus the post-inverse
        y-clip bounds (``y_clip_low``, ``y_clip_high``) and
        ``y_train_median`` for fallback predictions.
    estimator_
        The fitted clone of ``base_estimator``.
    feature_names_in_
        Column names seen at fit (best effort: pandas / polars).
    runtime_stats_
        Live counters: ``domain_violation_rate`` (fraction of rows at
        predict where ``base`` was non-finite or out of domain),
        ``y_clip_hit_rate`` (fraction of predictions clipped to the
        y-bounds). Populated lazily on first ``predict`` and updated
        cumulatively across calls.
    """

    def __init__(
        self,
        base_estimator: Any = None,
        transform_name: str = "diff",
        base_column: str = "",
        fallback_predict: str = "y_train_median",
        drop_invalid_rows: bool = True,
        runtime_stats_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        auto_variance_stabilise: bool = False,
    ) -> None:
        self.base_estimator = base_estimator
        self.transform_name = transform_name
        self.base_column = base_column
        self.fallback_predict = fallback_predict
        self.drop_invalid_rows = drop_invalid_rows
        # R10b improvement #8: when transform is ratio/logratio and
        # caller doesn't provide explicit sample_weight, auto-compute
        # variance-stabilising weights ~ 1/|base| (capped) to flatten
        # heteroscedasticity that ratio/logratio targets exhibit
        # (residuals on T-scale scale with |base| under multiplicative
        # DGP; LightGBM minimising MSE on T over-fits to the high-
        # variance regime). Default off -- opt-in because it changes
        # the loss; recommended on heavy-tail targets where logratio
        # was already chosen.
        self.auto_variance_stabilise = auto_variance_stabilise
        # Optional callback fired at the end of every ``predict`` call
        # with a snapshot of the per-batch counters. Use to hook into
        # Prometheus / StatsD / DataDog without coupling the wrapper
        # to a metrics library. The callback receives a dict with
        # keys ``batch_n``, ``batch_domain_violation_rows``,
        # ``batch_y_clip_low_hits``, ``batch_y_clip_high_hits``,
        # plus the cumulative-since-fit counters under
        # ``cumulative_*``. Errors raised by the callback are logged
        # at DEBUG and swallowed -- monitoring failures must never
        # poison the predict path.
        self.runtime_stats_callback = runtime_stats_callback

    # ------------------------------------------------------------------
    # Alternate constructor: post-hoc wrapping
    # ------------------------------------------------------------------

    @classmethod
    def from_fitted_inner(
        cls,
        fitted_inner: Any,
        transform_name: str,
        base_column: str,
        transform_fitted_params: Dict[str, Any],
        y_train: np.ndarray,
        fallback_predict: str = "y_train_median",
    ) -> "CompositeTargetEstimator":
        """Build a wrapper around an ALREADY-FITTED inner model.

        This is the path used by the suite-level integration: the
        per-target training loop trains models on the T-scale composite
        target unaware of the wrapper, then post-hoc wrapping converts
        each fitted model to a y-scale predictor.

        Bypasses the wrapper's own ``fit`` because the inner is
        already fitted and re-fitting would either re-train (waste)
        or fail (no train data at this point in the suite). We
        populate the same ``estimator_`` / ``fitted_params_`` /
        ``runtime_stats_`` state that ``fit`` would have set, so
        ``predict`` is contract-identical to fit-then-predict.

        Parameters
        ----------
        fitted_inner
            The already-fitted base estimator. Caller is responsible
            for ensuring it was trained on the T-scale composite target
            (i.e. ``transform.forward(y, base, transform_fitted_params)``).
        transform_name, base_column
            See :meth:`__init__`.
        transform_fitted_params
            Transform-specific fitted parameters (``alpha`` / ``beta``
            for linear_residual, ``mad_eff`` / ``median_t`` for
            logratio, ``eps`` for ratio). Comes from
            :class:`CompositeSpec.fitted_params`.
        y_train
            Training-row ``y`` values used by the discovery pass. The
            wrapper computes ``y_train_median``, ``y_clip_low``, and
            ``y_clip_high`` from this so the predict-time fallbacks
            and post-inverse y-clip work without needing the train
            data later.
        fallback_predict
            See :meth:`__init__`.
        """
        instance = cls(
            base_estimator=fitted_inner,
            transform_name=transform_name,
            base_column=base_column,
            fallback_predict=fallback_predict,
            drop_invalid_rows=True,
        )
        # Validate we can lookup the transform up-front so a typo
        # surfaces here, not on first predict.
        get_transform(transform_name)

        y_train = np.asarray(y_train).reshape(-1).astype(np.float64)
        finite = np.isfinite(y_train)
        if finite.size == 0 or not finite.any():
            y_train_median = float("nan")
            y_clip_low, y_clip_high = float("-inf"), float("inf")
        else:
            y_train_median = float(np.median(y_train[finite]))
            y_clip_low, y_clip_high = _y_train_clip_bounds(y_train[finite])

        instance.estimator_ = fitted_inner
        instance.fitted_params_ = {
            **dict(transform_fitted_params),
            "y_clip_low": y_clip_low,
            "y_clip_high": y_clip_high,
            "y_train_median": y_train_median,
        }
        instance.runtime_stats_ = {
            "predict_calls": 0,
            "predict_rows_total": 0,
            "domain_violation_rows": 0,
            "y_clip_low_hits": 0,
            "y_clip_high_hits": 0,
        }
        return instance

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[np.ndarray] = None,
        **fit_kwargs: Any,
    ) -> "CompositeTargetEstimator":
        if self.base_estimator is None:
            raise ValueError("CompositeTargetEstimator: base_estimator must not be None.")
        if not self.base_column:
            raise ValueError("CompositeTargetEstimator: base_column must be a non-empty string.")
        transform = get_transform(self.transform_name)

        y_arr = _to_1d_numpy(y).astype(np.float64)
        base_arr = _extract_base(X, self.base_column)
        if len(y_arr) != len(base_arr):
            raise ValueError(
                f"CompositeTargetEstimator.fit: y has {len(y_arr)} rows but X "
                f"has {len(base_arr)} -- caller passed misaligned inputs."
            )

        # Apply domain check before fitting transform params: invalid
        # rows must NOT bias the OLS / MAD computation.
        valid = transform.domain_check(y_arr, base_arr)
        n_invalid = int((~valid).sum())
        if n_invalid > 0:
            if not self.drop_invalid_rows:
                raise DomainViolationError(
                    f"CompositeTargetEstimator.fit: transform '{self.transform_name}' "
                    f"requires domain conditions; {n_invalid} of {len(y_arr)} rows violate. "
                    "Set drop_invalid_rows=True to drop them automatically."
                )
            logger.info(
                "[CompositeTargetEstimator] dropping %d/%d (%.2f%%) rows "
                "violating domain of transform '%s'.",
                n_invalid, len(y_arr), 100.0 * n_invalid / len(y_arr), self.transform_name,
            )

        y_train = y_arr[valid]
        base_train = base_arr[valid]

        if y_train.size == 0:
            raise DomainViolationError(
                f"CompositeTargetEstimator.fit: 0 rows pass domain_check for "
                f"transform '{self.transform_name}'. Check inputs."
            )

        # Fit transform-specific params on the valid train rows.
        # Pass sample_weight through to weight-aware transforms
        # (currently only ``linear_residual``); other transforms
        # ignore the kwarg.
        sample_weight_train = (
            np.asarray(sample_weight)[valid] if sample_weight is not None else None
        )
        try:
            transform_params = transform.fit(
                y_train, base_train, sample_weight=sample_weight_train,
            )
        except TypeError:
            # Transform.fit doesn't accept sample_weight (most don't).
            transform_params = transform.fit(y_train, base_train)

        # Compute T on the valid rows.
        t_train = transform.forward(y_train, base_train, transform_params)

        # Sanity: T must be finite or the inner estimator will choke.
        if not np.all(np.isfinite(t_train)):
            n_bad = int((~np.isfinite(t_train)).sum())
            raise DomainViolationError(
                f"CompositeTargetEstimator.fit: transform '{self.transform_name}' produced "
                f"{n_bad} non-finite T values on train. Likely numerical issue (extreme y / base)."
            )

        # Subset X to valid rows. Branch on type so we don't pull
        # pandas APIs on polars frames. Wrapper passes through whatever
        # frame the caller provided to the inner estimator; if the
        # inner doesn't accept that frame type (e.g. LightGBM 4.5 +
        # sklearn 1.6 on polars), the caller is responsible for the
        # conversion BEFORE reaching the wrapper. mlframe strategies
        # already do this at the suite level. We must not silently
        # materialise large polars frames -- that defeats the whole
        # zero-copy Arrow path on multi-GB datasets.
        X_valid = self._subset_rows(X, valid)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)[valid]
        elif (getattr(self, "auto_variance_stabilise", False)
                and self.transform_name in ("ratio", "logratio")):
            # R10b #8: auto-compute variance-stabilising weights.
            # For multiplicative DGP, residual variance scales with
            # |base|. Weight ~ 1/|base| (capped at the 5th percentile
            # to avoid blow-up on near-zero base) flattens it.
            base_valid = base_train[valid].astype(np.float64)
            abs_base = np.abs(base_valid)
            floor_q = float(np.quantile(
                abs_base[abs_base > 0], 0.05,
            )) if (abs_base > 0).any() else 1.0
            abs_base_clipped = np.maximum(abs_base, floor_q)
            w = 1.0 / abs_base_clipped
            # Normalise to mean 1 so loss scale matches unweighted.
            w *= w.size / w.sum()
            sample_weight = w

        # Fit the inner estimator. Clone first so the unfitted
        # prototype passed in stays untouched and sklearn.clone() of
        # the wrapper produces a fresh inner.
        estimator = clone(self.base_estimator)
        if sample_weight is not None:
            try:
                estimator.fit(X_valid, t_train, sample_weight=sample_weight, **fit_kwargs)
            except TypeError:
                # Inner doesn't accept sample_weight; fall back.
                logger.info(
                    "[CompositeTargetEstimator] inner estimator '%s' does not accept "
                    "sample_weight; ignoring.",
                    type(self.base_estimator).__name__,
                )
                estimator.fit(X_valid, t_train, **fit_kwargs)
        else:
            estimator.fit(X_valid, t_train, **fit_kwargs)

        # Stash post-inverse y-clip bounds + train median for fallback.
        y_clip_low, y_clip_high = _y_train_clip_bounds(y_train)
        y_train_median = float(np.median(y_train))

        self.estimator_ = estimator
        self.fitted_params_ = {
            **transform_params,
            "y_clip_low": y_clip_low,
            "y_clip_high": y_clip_high,
            "y_train_median": y_train_median,
            "n_train_valid": int(y_train.size),
            "n_train_invalid": n_invalid,
        }
        # Best-effort feature names (pandas / polars).
        try:
            self.feature_names_in_ = list(X.columns)
        except Exception:
            pass
        # Live counters initialised lazily by predict.
        self.runtime_stats_ = {
            "predict_calls": 0,
            "predict_rows_total": 0,
            "domain_violation_rows": 0,
            "y_clip_low_hits": 0,
            "y_clip_high_hits": 0,
        }
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError(
                "CompositeTargetEstimator.predict called before fit."
            )
        transform = get_transform(self.transform_name)
        params = self.fitted_params_

        base_arr = _extract_base(X, self.base_column)
        # Domain check at predict: y is unknown, so we ask the transform
        # to gate on base-side conditions only (e.g. base > 0 for
        # logratio, |base| > 0 for ratio). The ``y=None`` sentinel is
        # part of the domain_check contract -- transforms whose forward
        # is purely y-side (none here) would still need a y; the four
        # core transforms all degrade cleanly.
        domain_ok = transform.domain_check(None, base_arr)

        # Pass X through to the inner unchanged. NEVER materialise the
        # frame here -- on a 100 GB polars frame a silent conversion
        # blows the host out of memory. Caller is responsible for
        # ensuring the frame type is acceptable to the inner estimator
        # (mlframe strategies handle this at the suite level).
        t_hat = np.asarray(self.estimator_.predict(X), dtype=np.float64).reshape(-1)

        # Apply inverse only on valid rows; fill the rest with fallback.
        if domain_ok.all():
            y_hat = transform.inverse(t_hat, base_arr, params)
        else:
            y_hat = np.full_like(t_hat, fill_value=np.nan, dtype=np.float64)
            # Inverse on valid rows only; placeholder base for invalid
            # rows is irrelevant since we overwrite immediately.
            base_safe = np.where(domain_ok, base_arr, 1.0)
            y_hat_valid = transform.inverse(t_hat, base_safe, params)
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
        n = int(t_hat.size)
        n_violation = int((~domain_ok).sum())
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

    # ------------------------------------------------------------------
    # Quantile predictions (ConfidenceAnalysisConfig integration)
    # ------------------------------------------------------------------

    def predict_quantile(
        self, X: Any, alpha: float = 0.5,
    ) -> np.ndarray:
        """y-scale quantile prediction by inverting the inner's
        T-scale quantile.

        Requires the inner estimator to expose ``predict_quantile(X,
        alpha)`` -- e.g. CatBoost ``MultiQuantile``, LightGBM
        ``quantile_alpha``, sklearn ``QuantileRegressor``. The wrapper
        calls inner -> ``T_q`` then applies the transform's inverse.

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
        base_arr = _extract_base(X, self.base_column)

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

        try:
            t_q = np.asarray(
                inner.predict_quantile(X, alpha), dtype=np.float64,
            ).reshape(-1)
        except TypeError:
            # Some libs name the kwarg differently (e.g. quantile=, q=).
            t_q = np.asarray(
                inner.predict_quantile(X, alpha=alpha), dtype=np.float64,
            ).reshape(-1)
        y_q = transform.inverse(t_q, base_arr, params)
        # Reuse the same y-clip bounds applied in predict() to defend
        # against extreme tail T-quantiles producing physically
        # implausible y.
        low = params.get("y_clip_low", float("-inf"))
        high = params.get("y_clip_high", float("inf"))
        return np.clip(y_q, low, high)

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        return getattr(getattr(self, "estimator_", None), "feature_importances_", None)

    @property
    def coef_(self) -> Optional[np.ndarray]:
        return getattr(getattr(self, "estimator_", None), "coef_", None)

    @property
    def intercept_(self) -> Optional[float]:
        return getattr(getattr(self, "estimator_", None), "intercept_", None)

    def get_booster(self):
        """XGBoost shim."""
        est = getattr(self, "estimator_", None)
        if est is None:
            raise RuntimeError("get_booster called before fit.")
        return est.get_booster()

    @property
    def booster_(self):
        """LightGBM shim."""
        return getattr(getattr(self, "estimator_", None), "booster_", None)

    @property
    def n_features_in_(self) -> Optional[int]:
        return getattr(getattr(self, "estimator_", None), "n_features_in_", None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _subset_rows(X: Any, mask: np.ndarray) -> Any:
        """Row-subset X, preserving the dataframe flavour. Polars / pandas
        / ndarray supported. Raises TypeError otherwise."""
        if hasattr(X, "to_pandas") and not isinstance(X, pd.DataFrame):
            # Polars: lazy-import to keep the module light when polars
            # isn't installed in the environment.
            import polars as pl
            return X.filter(pl.Series(mask))
        if isinstance(X, pd.DataFrame):
            return X.loc[mask].reset_index(drop=True)
        if isinstance(X, np.ndarray):
            return X[mask]
        raise TypeError(
            f"CompositeTargetEstimator: unsupported X type {type(X).__name__} for row subsetting."
        )


# ----------------------------------------------------------------------
# CompositeProvenance + report helpers
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CompositeProvenance:
    """Production-grade metadata for one composite-target spec.

    Carries everything a downstream consumer needs to (a) understand
    *why* this composite was selected, (b) reproduce the inverse at
    serving time without consulting source code, and (c) audit the
    selection trail months later when the original DS has moved on
    and stakeholders ask "what does this number mean".

    Why this exists. Without provenance, ``TVT__linear_residual__TVT_prev``
    is an opaque key. With provenance, the same key reads as
    "predicts residual after subtracting fitted alpha=0.952 of the
    previous-period TVT_prev value (R^2_train = 0.91), selected
    because removing the linear contribution exposed a residual MI
    of 0.165 against the remaining features".

    Convert to dict via :meth:`to_dict` (JSON-serialisable) or to a
    stakeholder-ready paragraph via :meth:`to_audit_trail`.
    """

    # Identity
    composite_id: str
    discovery_timestamp: str  # ISO 8601, no datetime obj to keep dict-pickle clean
    discovery_random_state: int

    # Origin
    target_col: str
    transform_name: str
    base_column: str

    # Human-readable formula
    forward_formula_human: str
    inverse_formula_human: str
    stakeholder_description: str

    # Fitted parameters (reproducible inversion)
    fitted_params: Dict[str, Any]

    # Justification numbers
    mi_y: float
    mi_t: float
    mi_gain: float
    valid_domain_frac: float
    n_train_rows: int

    # Optional: weight in cross-target ensemble (filled at integration time).
    ensemble_weight: Optional[float] = None
    ensemble_strategy: Optional[str] = None

    @classmethod
    def from_spec(
        cls,
        spec: "CompositeSpec",
        random_state: int,
        *,
        ensemble_weight: Optional[float] = None,
        ensemble_strategy: Optional[str] = None,
    ) -> "CompositeProvenance":
        """Construct provenance from a discovered :class:`CompositeSpec`.

        Pulls human-readable formula text from the registered transform
        and the spec's fitted parameters, plus a deterministic
        ``composite_id`` (sha256 prefix) so the same spec recurring in
        future runs is recognisable.
        """
        from datetime import datetime, timezone
        import hashlib
        import json

        # Stable id derived from (target, transform, base, fitted_params).
        canonical = json.dumps(
            {
                "target_col": spec.target_col,
                "transform_name": spec.transform_name,
                "base_column": spec.base_column,
                "fitted_params": spec.fitted_params,
            },
            sort_keys=True, default=str,
        )
        composite_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

        forward, inverse, description = _format_transform_formulas(
            transform_name=spec.transform_name,
            base_column=spec.base_column,
            target_col=spec.target_col,
            fitted_params=spec.fitted_params,
        )

        return cls(
            composite_id=composite_id,
            discovery_timestamp=datetime.now(timezone.utc).isoformat(),
            discovery_random_state=random_state,
            target_col=spec.target_col,
            transform_name=spec.transform_name,
            base_column=spec.base_column,
            forward_formula_human=forward,
            inverse_formula_human=inverse,
            stakeholder_description=description,
            fitted_params=dict(spec.fitted_params),
            mi_y=spec.mi_y,
            mi_t=spec.mi_t,
            mi_gain=spec.mi_gain,
            valid_domain_frac=spec.valid_domain_frac,
            n_train_rows=spec.n_train_rows,
            ensemble_weight=ensemble_weight,
            ensemble_strategy=ensemble_strategy,
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable plain dict (for ``metadata`` storage)."""
        return {
            "composite_id": self.composite_id,
            "discovery_timestamp": self.discovery_timestamp,
            "discovery_random_state": self.discovery_random_state,
            "target_col": self.target_col,
            "transform_name": self.transform_name,
            "base_column": self.base_column,
            "forward_formula_human": self.forward_formula_human,
            "inverse_formula_human": self.inverse_formula_human,
            "stakeholder_description": self.stakeholder_description,
            "fitted_params": dict(self.fitted_params),
            "mi_y": float(self.mi_y),
            "mi_t": float(self.mi_t),
            "mi_gain": float(self.mi_gain),
            "valid_domain_frac": float(self.valid_domain_frac),
            "n_train_rows": int(self.n_train_rows),
            "ensemble_weight": (None if self.ensemble_weight is None
                                else float(self.ensemble_weight)),
            "ensemble_strategy": self.ensemble_strategy,
        }

    def to_audit_trail(self) -> str:
        """Single-paragraph human-readable summary suitable for a Slack
        message or a code-review comment. Quotes the exact numbers
        that justified inclusion so the reader can cross-check."""
        ensemble_clause = ""
        if self.ensemble_weight is not None and self.ensemble_strategy is not None:
            ensemble_clause = (
                f" In the cross-target {self.ensemble_strategy} ensemble it "
                f"received weight {self.ensemble_weight:.3f}."
            )
        return (
            f"Composite '{self.target_col}__{self.transform_name}__{self.base_column}' "
            f"(id={self.composite_id}) was discovered using "
            f"random_state={self.discovery_random_state} on "
            f"{self.n_train_rows} train rows ({self.valid_domain_frac:.1%} of valid domain). "
            f"It was selected because MI(T, X\\base)={self.mi_t:.4f} vs "
            f"MI(y, X\\base)={self.mi_y:.4f} (gain={self.mi_gain:+.4f}), "
            f"meaning the transform '{self.stakeholder_description}' exposed "
            f"residual structure the remaining features can predict more easily. "
            f"Forward: {self.forward_formula_human}. "
            f"Inverse: {self.inverse_formula_human}.{ensemble_clause}"
        )


# Friendly transform-name-to-paragraph table.
_TRANSFORM_DESCRIPTIONS: Dict[str, str] = {
    "diff": ("predicts the residual after subtracting the base feature "
             "from the target"),
    "ratio": ("predicts the multiplicative factor relating target to "
              "base feature"),
    "logratio": ("predicts the log-ratio of target to base feature, "
                 "stabilising heavy-tail distributions"),
    "linear_residual": ("predicts the residual after subtracting a "
                        "fitted linear contribution of the base feature"),
}


def _format_transform_formulas(
    transform_name: str, base_column: str, target_col: str,
    fitted_params: Dict[str, Any],
) -> Tuple[str, str, str]:
    """Return (forward_human, inverse_human, description) strings.

    Strings interpolate fitted parameter values where applicable. Used
    by :class:`CompositeProvenance` to render audit-friendly formula
    descriptions without forcing the caller to know the registry.
    """
    desc = _TRANSFORMS_REGISTRY.get(transform_name)
    description = _TRANSFORM_DESCRIPTIONS.get(transform_name, "")
    if transform_name == "diff":
        return (
            f"T = {target_col} - {base_column}",
            f"y_hat = T_hat + {base_column}",
            description,
        )
    if transform_name == "ratio":
        eps = fitted_params.get("eps", 1e-12)
        return (
            f"T = {target_col} / {base_column}  (with |{base_column}| >= {eps:.3g})",
            f"y_hat = T_hat * {base_column}",
            description,
        )
    if transform_name == "logratio":
        median_t = fitted_params.get("median_t", 0.0)
        mad_eff = fitted_params.get("mad_eff", 0.0)
        return (
            f"T = log({target_col}) - log({base_column})  (requires {target_col}, {base_column} > 0)",
            f"y_hat = {base_column} * exp(softcap(T_hat, {median_t:.4g} +/- 10*{mad_eff:.4g}))",
            description,
        )
    if transform_name == "linear_residual":
        alpha = fitted_params.get("alpha", 0.0)
        beta = fitted_params.get("beta", 0.0)
        return (
            f"T = {target_col} - {alpha:.4g} * {base_column} - ({beta:.4g})",
            f"y_hat = T_hat + {alpha:.4g} * {base_column} + ({beta:.4g})",
            description,
        )
    # Unknown / future transform: fall back to a generic description.
    return (
        f"T = forward({target_col}, {base_column}) [{transform_name}]",
        f"y_hat = inverse(T_hat, {base_column}) [{transform_name}]",
        description or f"transform '{transform_name}'",
    )


def report_to_markdown(
    *,
    target_col: str,
    specs: Sequence["CompositeSpec"],
    failures: Sequence[Dict[str, Any]] = (),
    ensemble_metadata: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> str:
    """Render a stakeholder-ready Markdown report for one target's
    composite-target discovery output.

    Sections:

    1. Summary line: target name, count of kept specs, count of rejected.
    2. Discovered specs table with mi_y / mi_t / mi_gain / valid_frac.
    3. Per-spec audit paragraph (one per spec).
    4. Rejected candidates table with reason.
    5. Ensemble metadata if provided.

    All user-controlled strings (column names, target names) are NOT
    HTML-escaped in this version because Markdown is plain text by
    default; if the caller renders to HTML elsewhere they should
    escape there.
    """
    lines: List[str] = []
    lines.append(f"# Composite-target discovery report: `{target_col}`")
    lines.append("")
    lines.append(
        f"**{len(specs)}** discovered spec(s); **{len(failures)}** rejected candidate(s)."
    )
    lines.append("")

    if specs:
        lines.append("## Discovered specs")
        lines.append("")
        lines.append("| name | base | transform | mi_y | mi_t | mi_gain | valid_frac | n_train |")
        lines.append("|------|------|-----------|------|------|---------|-----------|---------|")
        for spec in specs:
            lines.append(
                f"| `{spec.name}` | `{spec.base_column}` | `{spec.transform_name}` | "
                f"{spec.mi_y:.4f} | {spec.mi_t:.4f} | {spec.mi_gain:+.4f} | "
                f"{spec.valid_domain_frac:.1%} | {spec.n_train_rows} |"
            )
        lines.append("")
        lines.append("## Per-spec audit")
        lines.append("")
        for spec in specs:
            ensemble_w = None
            ensemble_strat = None
            if ensemble_metadata:
                # Look up this spec's weight if it appears in the
                # ensemble's component list.
                for nm, w in zip(
                    ensemble_metadata.get("component_names", []),
                    ensemble_metadata.get("weights", []),
                ):
                    if nm.startswith(spec.name + "#"):
                        ensemble_w = float(w)
                        ensemble_strat = ensemble_metadata.get("strategy")
                        break
            prov = CompositeProvenance.from_spec(
                spec=spec, random_state=random_state,
                ensemble_weight=ensemble_w,
                ensemble_strategy=ensemble_strat,
            )
            lines.append(f"### `{spec.name}`")
            lines.append("")
            lines.append(prov.to_audit_trail())
            lines.append("")

    if failures:
        lines.append("## Rejected candidates")
        lines.append("")
        lines.append("| base | transform | reason |")
        lines.append("|------|-----------|--------|")
        for f in failures:
            base = f.get("base_column", "?")
            transform = f.get("transform_name", "?")
            reason = f.get("reason", "")
            lines.append(f"| `{base}` | `{transform}` | {reason} |")
        lines.append("")

    if ensemble_metadata:
        lines.append("## Cross-target ensemble")
        lines.append("")
        lines.append(f"Strategy: **{ensemble_metadata.get('strategy', '?')}**")
        lines.append("")
        lines.append("| component | weight |")
        lines.append("|-----------|-------:|")
        for nm, w in zip(
            ensemble_metadata.get("component_names", []),
            ensemble_metadata.get("weights", []),
        ):
            lines.append(f"| `{nm}` | {w:.4f} |")
        lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# CompositeCrossTargetEnsemble
# ----------------------------------------------------------------------


def derive_seeds(random_state: int, components: Sequence[str]) -> Dict[str, int]:
    """Derive deterministic per-component seeds from a master seed.

    Uses sha256 truncation to keep the values stable across Python /
    numpy versions (no dependence on hash() salt randomisation). The
    returned dict maps each component name to a 32-bit unsigned int.

    Why this exists. Discovery has several internal sources of
    randomness (MI sampling, tiny-model CV split, OOF holdout split,
    bootstrap CI). Threading the same ``random_state`` through every
    one of them creates correlation: if the master seed produces an
    "easy" MI sample it tends to also produce an "easy" CV split.
    Sub-seeds break the correlation while keeping reproducibility:
    same master seed -> same sub-seeds -> same downstream randomness.
    """
    import hashlib
    import struct
    out: Dict[str, int] = {}
    for c in components:
        h = hashlib.sha256(f"{random_state}::{c}".encode("utf-8")).digest()
        out[c] = struct.unpack("<I", h[:4])[0]
    return out


def detect_gpu_in_use(mlframe_models: Sequence[str]) -> List[str]:
    """Return list of model families that may be using GPU.

    Best-effort detection: imports each library only if it appears in
    ``mlframe_models`` and probes for GPU availability via the
    library's standard health-check API. Returns the subset that has
    GPU detected. Returns empty list when no GPU library is in use.

    Used by the suite to emit a one-shot warning when composite mode
    is combined with GPU training: GPU non-determinism is amplified
    by K composite-model fits and can surface as ensemble weight
    drift across runs even when ``random_state`` is fixed.
    """
    detected: List[str] = []
    families = {str(m).lower() for m in mlframe_models}
    if any(f in families for f in ("lgb", "lightgbm")):
        try:
            import lightgbm as lgb  # noqa: F401
            # LightGBM doesn't have a portable "is GPU available"
            # check; we infer from the user's stated intent only.
            # Conservative: skip the warning if we can't tell.
        except ImportError:
            pass
    if any(f in families for f in ("xgb", "xgboost")):
        try:
            import xgboost as xgb
            try:
                # XGBoost build info is the canonical "GPU available?"
                # signal post-2.x.
                bi = xgb.build_info()
                if isinstance(bi, dict) and bi.get("USE_CUDA", False):
                    detected.append("xgboost")
            except Exception:
                pass
        except ImportError:
            pass
    if any(f in families for f in ("cb", "catboost")):
        try:
            from catboost.utils import get_gpu_device_count
            if get_gpu_device_count() > 0:
                detected.append("catboost")
        except Exception:
            pass
    return detected


def env_signature() -> Dict[str, Optional[str]]:
    """Snapshot of library versions relevant to composite-target
    discovery + serialisation. Stored on metadata so a pickle saved
    today can be reload-validated tomorrow against version drift.

    Returns ``None`` for any library not installed.
    """
    sig: Dict[str, Optional[str]] = {}
    for libname in ("numpy", "pandas", "polars", "sklearn", "lightgbm",
                    "xgboost", "catboost", "scipy", "dill"):
        try:
            mod = __import__(libname)
            sig[libname] = getattr(mod, "__version__", None)
        except Exception:
            sig[libname] = None
    return sig


def compute_oof_holdout_predictions(
    component_models: List[Any],
    component_names: List[str],
    component_specs: List[Optional[Dict[str, Any]]],
    train_X: Any,
    y_train_full: np.ndarray,
    base_train_full_per_spec: Dict[str, np.ndarray],
    holdout_frac: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute honest holdout predictions for each component.

    Approach: take a single random ``holdout_frac`` slice of train,
    re-fit a clone of each component's inner on the remaining
    (1-holdout_frac) rows, and predict on the held-out slice. For
    wrapped composite-target components we re-apply the spec's
    transform on the same stack_train slice to get T values, train
    the inner clone on (X_stack_train, T_stack_train), then wrap
    using ``CompositeTargetEstimator.from_fitted_inner`` and predict
    in y-scale on stack_holdout. For raw-target components the inner
    clone is fit directly on (X_stack_train, y_stack_train).

    Single-split (not K-fold) keeps the additional compute bounded
    at ``len(components)`` re-fits. Returns:

    - ``holdout_preds_matrix``: (n_holdout, K) y-scale predictions.
    - ``y_holdout``: (n_holdout,) original-scale targets.
    - ``surviving_names``: subset of ``component_names`` whose
      re-fit succeeded (any failures are dropped from the matrix
      so callers can re-align weight vectors).
    """
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split

    n_train = len(y_train_full)
    if n_train < 50 or holdout_frac <= 0 or holdout_frac >= 1:
        return np.zeros((0, len(component_models))), np.zeros(0), []

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n_train)
    n_holdout = max(int(round(n_train * holdout_frac)), 1)
    holdout_idx = np.sort(perm[:n_holdout])
    train_idx = np.sort(perm[n_holdout:])

    # Subset X. Branch on type so we don't pull pandas APIs on
    # polars frames.
    if hasattr(train_X, "to_pandas") and not isinstance(train_X, pd.DataFrame):
        import polars as pl
        train_mask = np.zeros(n_train, dtype=bool)
        train_mask[train_idx] = True
        X_stack = train_X.filter(pl.Series(train_mask))
        X_holdout = train_X.filter(pl.Series(~train_mask))
    elif isinstance(train_X, pd.DataFrame):
        X_stack = train_X.iloc[train_idx].reset_index(drop=True)
        X_holdout = train_X.iloc[holdout_idx].reset_index(drop=True)
    else:
        X_stack = train_X[train_idx]
        X_holdout = train_X[holdout_idx]

    y_stack = y_train_full[train_idx].astype(np.float64)
    y_holdout = y_train_full[holdout_idx].astype(np.float64)

    holdout_cols: List[np.ndarray] = []
    surviving_names: List[str] = []
    for model, name, spec in zip(component_models, component_names, component_specs):
        try:
            if isinstance(model, CompositeTargetEstimator):
                # Composite-target wrapper. Re-fit the inner on
                # stack_train T values, then re-wrap and predict.
                if spec is None:
                    raise ValueError("composite component with no spec")
                base_full = base_train_full_per_spec.get(spec["base_column"])
                if base_full is None:
                    raise ValueError(
                        f"missing base column '{spec['base_column']}' for OOF"
                    )
                base_stack = base_full[train_idx]
                transform = get_transform(spec["transform_name"])
                valid = transform.domain_check(y_stack, base_stack)
                # Drop invalid rows from stack_train; the inner will
                # train only on rows where T is finite.
                if valid.sum() < 10:
                    raise ValueError("too few valid rows after domain filter")
                t_stack = transform.forward(
                    y_stack[valid], base_stack[valid], spec["fitted_params"],
                )
                inner_clone = clone(model.estimator_)
                if hasattr(X_stack, "iloc"):
                    X_stack_valid = X_stack.iloc[valid].reset_index(drop=True)
                elif hasattr(X_stack, "filter") and not isinstance(X_stack, np.ndarray):
                    import polars as pl
                    X_stack_valid = X_stack.filter(pl.Series(valid))
                else:
                    X_stack_valid = X_stack[valid]
                inner_clone.fit(X_stack_valid, t_stack)
                wrapped = CompositeTargetEstimator.from_fitted_inner(
                    fitted_inner=inner_clone,
                    transform_name=spec["transform_name"],
                    base_column=spec["base_column"],
                    transform_fitted_params=spec["fitted_params"],
                    y_train=y_stack[valid],
                )
                preds = wrapped.predict(X_holdout)
            else:
                # Raw-target component. Re-fit the inner on
                # (X_stack, y_stack) and predict on X_holdout.
                inner_clone = clone(model)
                inner_clone.fit(X_stack, y_stack)
                preds = inner_clone.predict(X_holdout)
            preds = np.asarray(preds).reshape(-1).astype(np.float64)
            if not np.all(np.isfinite(preds)):
                # NaN preds on holdout -- exclude from ensemble.
                raise ValueError("non-finite holdout predictions")
            holdout_cols.append(preds)
            surviving_names.append(name)
        except Exception as exc:
            logger.warning(
                "[CompositeCrossTargetEnsemble] OOF refit failed for component "
                "'%s': %s. Excluded from ensemble weights.", name, exc,
            )
            continue

    if not holdout_cols:
        return np.zeros((n_holdout, 0)), y_holdout, []
    return np.column_stack(holdout_cols), y_holdout, surviving_names


class CompositeCrossTargetEnsemble:
    """Weighted-average ensemble of K composite-target predictors plus
    optionally the raw-target predictor.

    All input models MUST already produce y-scale predictions (i.e. be
    :class:`CompositeTargetEstimator` wrappers OR a raw regressor on
    the original target). The ensemble does not invert anything --
    it just averages.

    The ensemble class itself is strategy-neutral: weights are
    pre-computed by :meth:`from_train_metrics` (the recommended path)
    or :meth:`from_uniform_weights` (mean baseline) and frozen on the
    instance. ``predict`` is one matrix-vector product.

    Validation gate
    ---------------
    :meth:`from_train_metrics` runs a built-in gate: it compares the
    ensemble's train-set RMSE against the best single component's
    train-set RMSE. If the ensemble is worse, it returns the best
    single component instead and logs a warning. The check is
    biased optimistic (uses train data) but still catches the most
    common failure mode -- a high-variance candidate with a stretched
    weight that drags the ensemble below the strongest component.
    """

    def __init__(
        self,
        component_models: List[Any],
        component_names: List[str],
        weights: np.ndarray,
        strategy: str,
        notes: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(component_models) == 0:
            raise ValueError("CompositeCrossTargetEnsemble: empty component list.")
        if len(component_models) != len(component_names) or len(component_models) != len(weights):
            raise ValueError(
                "CompositeCrossTargetEnsemble: component_models, component_names, "
                "and weights must be same length; got "
                f"{len(component_models)} / {len(component_names)} / {len(weights)}."
            )
        weights = np.asarray(weights, dtype=np.float64)
        wsum = float(weights.sum())
        if wsum <= 0 or not math.isfinite(wsum):
            raise ValueError(
                f"CompositeCrossTargetEnsemble: weights must sum to positive finite "
                f"value; got sum={wsum}."
            )
        self.component_models = list(component_models)
        self.component_names = list(component_names)
        self.weights = weights / wsum  # always normalised
        self.strategy = strategy
        self.notes = dict(notes or {})

    # ------------------------------------------------------------------
    # Constructors / factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_uniform_weights(
        cls,
        component_models: List[Any],
        component_names: List[str],
    ) -> "CompositeCrossTargetEnsemble":
        """Equal-weight average: ``w_k = 1/K`` for all components."""
        n = len(component_models)
        return cls(
            component_models=component_models,
            component_names=component_names,
            weights=np.full(n, 1.0 / n) if n > 0 else np.array([]),
            strategy="mean",
        )

    @classmethod
    def from_linear_stack(
        cls,
        component_models: List[Any],
        component_names: List[str],
        component_predictions: np.ndarray,  # (n_train, K) y-scale predictions
        y_train: np.ndarray,
        ridge_alpha: float = 1.0,
    ) -> "CompositeCrossTargetEnsemble":
        """Linear stacking via Ridge regression.

        Fits a Ridge model ``y_train ~ X @ w + b`` where ``X`` is the
        per-component prediction matrix on train. The resulting
        weights are the stack coefficients; intercept is folded into
        the bias by absorbing it as an extra ``+b/n`` per component
        (good enough when Ridge converges).

        ``ridge_alpha`` is fixed (no internal CV) -- callers wanting
        alpha tuning should ridge-CV externally and pass the chosen
        alpha. Higher alpha -> more regularisation -> closer to mean.

        Returns negative weights when a component is anti-correlated
        with the target -- this is fine, the ensemble may still work.
        ``predict`` re-normalises only the magnitudes, so a negative
        weight means the component's prediction is subtracted.
        """
        from sklearn.linear_model import Ridge
        n = len(component_models)
        if n == 0:
            raise ValueError("from_linear_stack: empty component list.")
        component_predictions = np.asarray(component_predictions, dtype=np.float64)
        if component_predictions.shape[1] != n:
            raise ValueError(
                f"from_linear_stack: prediction matrix has {component_predictions.shape[1]} "
                f"columns, expected {n} (one per component)."
            )
        y = np.asarray(y_train, dtype=np.float64).reshape(-1)
        if len(y) != component_predictions.shape[0]:
            raise ValueError(
                f"from_linear_stack: y_train length {len(y)} != prediction "
                f"matrix rows {component_predictions.shape[0]}."
            )
        # Drop rows with non-finite y or predictions.
        finite = np.isfinite(y) & np.all(np.isfinite(component_predictions), axis=1)
        if finite.sum() < n + 2:
            logger.warning(
                "[CompositeCrossTargetEnsemble] linear_stack: only %d finite rows for "
                "%d components; falling back to oof_weighted-style mean.",
                int(finite.sum()), n,
            )
            return cls.from_uniform_weights(component_models, component_names)

        ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
        ridge.fit(component_predictions[finite], y[finite])
        raw_weights = np.asarray(ridge.coef_, dtype=np.float64)
        # Sanity: if all weights are zero or non-finite, fall back.
        if not np.any(raw_weights) or not np.all(np.isfinite(raw_weights)):
            logger.warning(
                "[CompositeCrossTargetEnsemble] linear_stack: degenerate weights; "
                "falling back to mean."
            )
            return cls.from_uniform_weights(component_models, component_names)
        # The constructor normalises by sum. For linear_stack this is
        # NOT semantically right (negative weights, intercept), so we
        # bypass the normalisation by building manually.
        instance = cls(
            component_models=component_models,
            component_names=component_names,
            weights=np.abs(raw_weights) + 1e-12,  # placeholder for constructor
            strategy="linear_stack",
        )
        # Inject the actual (un-normalised) weights + intercept.
        instance.weights = raw_weights
        instance.notes = {
            "ridge_alpha": ridge_alpha,
            "intercept": float(ridge.intercept_),
            "raw_weights": raw_weights.tolist(),
            "n_train_rows": int(finite.sum()),
        }
        instance._linear_stack_intercept = float(ridge.intercept_)
        return instance

    @classmethod
    def from_nnls_stack(
        cls,
        component_models: List[Any],
        component_names: List[str],
        component_predictions: np.ndarray,
        y_train: np.ndarray,
    ) -> "CompositeCrossTargetEnsemble":
        """Non-negative least squares stacking.

        Fits ``y = X @ w`` subject to ``w >= 0`` via
        ``scipy.optimize.nnls``. Weights are then normalised to sum
        to 1, which keeps the predict path identical to mean /
        oof_weighted (no separate intercept handling). This is the
        recommended stack when component predictions are all already
        in y-scale (no negative weight makes physical sense), and is
        less prone to overfitting than ridge stacking on small data.
        """
        from scipy.optimize import nnls
        n = len(component_models)
        if n == 0:
            raise ValueError("from_nnls_stack: empty component list.")
        component_predictions = np.asarray(component_predictions, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).reshape(-1)
        finite = np.isfinite(y) & np.all(np.isfinite(component_predictions), axis=1)
        if finite.sum() < n + 2:
            logger.warning(
                "[CompositeCrossTargetEnsemble] nnls_stack: only %d finite rows for "
                "%d components; falling back to mean.",
                int(finite.sum()), n,
            )
            return cls.from_uniform_weights(component_models, component_names)

        try:
            w, _residual = nnls(component_predictions[finite], y[finite])
        except RuntimeError as exc:
            logger.warning(
                "[CompositeCrossTargetEnsemble] nnls_stack: solver failed (%s); "
                "falling back to mean.", exc,
            )
            return cls.from_uniform_weights(component_models, component_names)

        if w.sum() <= 0 or not np.all(np.isfinite(w)):
            logger.warning(
                "[CompositeCrossTargetEnsemble] nnls_stack: zero or non-finite "
                "weights; falling back to mean."
            )
            return cls.from_uniform_weights(component_models, component_names)

        return cls(
            component_models=component_models,
            component_names=component_names,
            weights=w,
            strategy="nnls_stack",
            notes={
                "raw_weights_pre_normalise": w.tolist(),
                "n_train_rows": int(finite.sum()),
            },
        )

    @classmethod
    def from_train_metrics(
        cls,
        component_models: List[Any],
        component_names: List[str],
        component_train_rmse: Sequence[float],
        baseline_train_rmse: Optional[float] = None,
    ) -> Union["CompositeCrossTargetEnsemble", Any]:
        """Build an ensemble weighted by *gain over a naive baseline*.

        The gain-over-naive convention defends against the trivial
        "raw model with TVT_prev as a feature" beating the naive
        ``predict y = base`` baseline by a tiny margin: that model's
        absolute RMSE is small (good), but its *gain* over the naive
        baseline is also small, so it gets a sensible weight rather
        than dominating the ensemble simply because its RMSE is
        numerically smaller than a good but harder-target composite.

        ``baseline_train_rmse`` is the RMSE of the naive predictor
        ``y_hat = base`` on train (or any sensible benchmark; pass
        the noisiest reasonable predictor's RMSE). If None, the
        median of ``component_train_rmse`` is used as a self-
        normalising fallback.

        If every component's RMSE is worse than the baseline, the
        method returns the SINGLE best-RMSE component instead of the
        ensemble (validation gate). Log line announces the fallback.
        """
        n = len(component_models)
        if n == 0:
            raise ValueError("from_train_metrics: empty component list.")
        rmses = np.asarray(component_train_rmse, dtype=np.float64)
        if len(rmses) != n:
            raise ValueError(
                f"from_train_metrics: rmse list len {len(rmses)} != n_components {n}."
            )
        if not np.all(np.isfinite(rmses)):
            raise ValueError("from_train_metrics: rmses contain non-finite values.")

        if baseline_train_rmse is None:
            baseline = float(np.median(rmses))
        else:
            baseline = float(baseline_train_rmse)
            if not math.isfinite(baseline):
                baseline = float(np.median(rmses))

        gains = np.maximum(0.0, baseline - rmses)
        if gains.sum() <= 0:
            # No component beats baseline. Return the single best by
            # RMSE; ensemble would be no improvement.
            best_idx = int(np.argmin(rmses))
            logger.warning(
                "[CompositeCrossTargetEnsemble] no component beats the baseline "
                "RMSE=%.4g; falling back to single best component '%s' (RMSE=%.4g).",
                baseline, component_names[best_idx], rmses[best_idx],
            )
            return component_models[best_idx]

        # The "no component beats baseline" gate fires above; the
        # only remaining decision is to build the ensemble.
        # We deliberately do NOT add an independence-bound RMSE gate
        # here: composite-target predictions correlate (same train
        # data, shared base feature), so the independence formula
        # overestimates ensemble RMSE and the gate would fire on
        # legitimate ensembles. The true validation gate -- "ensemble
        # OOF-RMSE > best single OOF-RMSE" -- requires real CV-OOF
        # predictions per component, which the per-target loop does
        # not currently expose. A future PR may add OOF storage; for
        # now the user is expected to evaluate the ensemble on a
        # held-out test set themselves.
        weights = gains / gains.sum()
        best_single_idx = int(np.argmin(rmses))
        best_single_rmse = float(rmses[best_single_idx])
        return cls(
            component_models=component_models,
            component_names=component_names,
            weights=weights,
            strategy="oof_weighted",
            notes={
                "baseline_train_rmse": baseline,
                "component_train_rmses": rmses.tolist(),
                "best_single_rmse": best_single_rmse,
                "best_single_name": component_names[best_single_idx],
                "gate_fallback": False,
            },
        )

    # ------------------------------------------------------------------
    # sklearn-ish API
    # ------------------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        """Weighted combination of per-component predictions.

        For ``mean`` / ``oof_weighted`` / ``nnls_stack`` strategies
        weights are non-negative and sum to 1; the result is a
        weighted average. For ``linear_stack`` strategy weights may be
        negative, do not sum to 1, and an intercept is added -- the
        result is the Ridge stack's prediction
        ``y_hat = X @ w + intercept``.
        """
        if not self.component_models:
            raise RuntimeError("CompositeCrossTargetEnsemble: no components.")
        per_component = []
        for model, name in zip(self.component_models, self.component_names):
            try:
                # Fold the dtype cast into the asarray call so we don't
                # allocate twice on the predict hot path. ``copy=False``
                # is the asarray default; the dtype kwarg lets us skip
                # a separate ``.astype()`` round-trip.
                pred = np.asarray(model.predict(X), dtype=np.float64).reshape(-1)
            except Exception as exc:
                logger.warning(
                    "[CompositeCrossTargetEnsemble] component '%s' predict failed: "
                    "%s. Excluding from this batch's ensemble (re-normalising).",
                    name, exc,
                )
                pred = None
            per_component.append(pred)

        ok = [(p, w) for p, w in zip(per_component, self.weights) if p is not None]
        if not ok:
            raise RuntimeError(
                "CompositeCrossTargetEnsemble.predict: all components failed."
            )
        preds_matrix = np.column_stack([p for p, _ in ok])
        weights = np.array([w for _, w in ok], dtype=np.float64)

        if self.strategy == "linear_stack":
            # Ridge stack: predictions = X @ w + intercept. Do NOT
            # renormalise weights. If a component dropped out, drop
            # its weight contribution -- the rest of the linear
            # combination is still valid (just with one fewer term).
            intercept = float(getattr(self, "_linear_stack_intercept", 0.0))
            return (preds_matrix * weights[None, :]).sum(axis=1) + intercept

        # Convex strategies (mean / oof_weighted / nnls_stack):
        # re-normalise weights across surviving components.
        if weights.sum() <= 0:
            # All surviving weights collapsed to zero -- fall back to
            # mean across surviving components.
            weights = np.full_like(weights, 1.0 / len(weights))
        else:
            weights = weights / weights.sum()
        return (preds_matrix * weights[None, :]).sum(axis=1)

    def export_metadata(self) -> Dict[str, Any]:
        """Plain-dict snapshot for ``metadata`` storage."""
        return {
            "strategy": self.strategy,
            "component_names": list(self.component_names),
            "weights": self.weights.tolist(),
            "notes": dict(self.notes),
        }

    def cap_inference_components(
        self, max_components: int,
    ) -> "CompositeCrossTargetEnsemble":
        """Return a NEW ensemble holding only the top-N components by
        absolute weight.

        Use case: production online prediction with a latency budget
        that can't afford running K=8 wrappers per row. Trims to the
        largest-weighted components and re-normalises (or preserves
        the linear-stack semantics by keeping the matching subset of
        weights + intercept). Returns a new ensemble; the original
        is unchanged.

        ``max_components <= 0`` or ``>= len(components)`` -> returns
        a copy of self unchanged (no trimming).
        """
        if max_components <= 0 or max_components >= len(self.component_models):
            return CompositeCrossTargetEnsemble(
                component_models=list(self.component_models),
                component_names=list(self.component_names),
                weights=np.asarray(self.weights, dtype=np.float64),
                strategy=self.strategy,
                notes=dict(self.notes),
            )
        # Pick top-N by |weight|.
        order = np.argsort(-np.abs(np.asarray(self.weights, dtype=np.float64)))
        keep = sorted(order[:max_components].tolist())
        new = CompositeCrossTargetEnsemble(
            component_models=[self.component_models[i] for i in keep],
            component_names=[self.component_names[i] for i in keep],
            weights=np.asarray([self.weights[i] for i in keep], dtype=np.float64),
            strategy=self.strategy,
            notes={**self.notes, "capped_to_top_n": int(max_components),
                   "dropped_components": [
                       self.component_names[i]
                       for i in range(len(self.component_models))
                       if i not in keep
                   ]},
        )
        # Linear stack: preserve intercept too.
        if self.strategy == "linear_stack" and hasattr(self, "_linear_stack_intercept"):
            new._linear_stack_intercept = self._linear_stack_intercept
        return new


# ----------------------------------------------------------------------
# CompositeSpec + CompositeTargetDiscovery
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CompositeSpec:
    """Frozen description of one discovered composite target.

    ``fitted_params`` is the dict returned by the corresponding
    :class:`Transform`'s ``fit`` (same shape consumed by ``forward`` /
    ``inverse``). Stored so :meth:`CompositeTargetDiscovery.iter_transform`
    can apply the transform to the full frame at integration time, and
    so downstream code can rebuild a :class:`CompositeTargetEstimator`
    with the exact same params used during discovery.

    ``mi_gain`` is the difference in MI between (T, X-without-base) and
    (y, X) -- positive means the transform makes the residual MORE
    predictable from the remaining features (the goal). Negative is
    possible (transform destroyed signal); discovery filters on
    ``eps_mi_gain``.

    ``valid_domain_frac`` is the share of train rows that pass
    ``transform.domain_check``. Discovery filters on
    ``min_valid_domain_frac`` so a transform that only works for ~half
    the rows isn't promoted.
    """

    name: str  # f"{target_col}__{transform_name}__{base_column}"
    target_col: str
    transform_name: str
    base_column: str
    fitted_params: Dict[str, Any]
    mi_gain: float
    mi_y: float
    mi_t: float
    valid_domain_frac: float
    n_train_rows: int


def _extract_column_array(df: Any, col: str) -> np.ndarray:
    """Pull a single column out as a 1-D float64 ndarray. Polars / pandas
    only -- never materialise a whole-frame conversion."""
    if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame):
        return np.asarray(df.get_column(col).to_numpy()).astype(np.float64)
    if isinstance(df, pd.DataFrame):
        return df[col].to_numpy(dtype=np.float64)
    raise TypeError(
        f"CompositeTargetDiscovery: unsupported df type {type(df).__name__}"
    )


def _is_numeric_column(df: Any, col: str) -> bool:
    """True if ``col`` is numeric in ``df``. Falls back to False on
    error -- discovery skips non-numeric base candidates rather than
    risking a cast bomb on object-dtype columns."""
    try:
        if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame):
            import polars as pl  # lazy
            dtype = df.schema[col]
            # ``dtype.is_numeric()`` covers Float*, Int*, UInt* (added in
            # polars 0.19); fall back to a hard-coded set on older
            # versions that ship the dtypes but not the helper.
            try:
                return bool(dtype.is_numeric())
            except AttributeError:
                return dtype in {
                    pl.Float32, pl.Float64,
                    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                }
        if isinstance(df, pd.DataFrame):
            return pd.api.types.is_numeric_dtype(df[col])
    except Exception:
        return False
    return False


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that returns 0.0 (not NaN) on degenerate
    inputs (constant array, all-NaN). Used in the forbidden-base
    near-derived filter where NaN would falsely pass the threshold.

    Implemented via explicit centred-vector dot product instead of
    ``np.corrcoef`` to avoid the 2x2 matrix construction overhead
    (~1.17x faster on 80K rows; benchmarked 2026-05-10).
    """
    finite = np.isfinite(a) & np.isfinite(b)
    n = int(finite.sum())
    if n < 3:
        return 0.0
    a_f = a[finite]
    b_f = b[finite]
    a_dev = a_f - a_f.mean()
    b_dev = b_f - b_f.mean()
    var_a = float(np.dot(a_dev, a_dev))
    var_b = float(np.dot(b_dev, b_dev))
    if var_a < 1e-24 or var_b < 1e-24:
        return 0.0
    return float(np.dot(a_dev, b_dev) / np.sqrt(var_a * var_b))


def _safe_abs_corr_all(
    y: np.ndarray, X: np.ndarray,
) -> np.ndarray:
    """Vectorised ``|corr(y, X[:, j])|`` for all j in one pass.

    Single matrix op gives 2.2x over the per-column loop on dense
    inputs (200 features x 80K rows: 558ms vs 1220ms). Returns 0.0
    for columns whose centred dot is below numerical tolerance.

    NaN handling differs from ``_safe_corr``: rows where ``y`` is
    non-finite are masked GLOBALLY (not per-column). For composite-
    target use this is acceptable because callers gate columns on
    sufficient finite count BEFORE passing them in (see
    ``_filter_features``); per-column NaN masking is reserved for
    the scalar ``_safe_corr`` path.
    """
    y_finite = np.isfinite(y)
    if y_finite.sum() < 3:
        return np.zeros(X.shape[1])
    y_f = y[y_finite]
    X_f = X[y_finite]
    n = y_f.size
    y_dev = y_f - y_f.mean()
    var_y = float(np.dot(y_dev, y_dev))
    if var_y < 1e-24:
        return np.zeros(X.shape[1])
    X_means = X_f.mean(axis=0)
    X_dev = X_f - X_means
    var_X = (X_dev * X_dev).sum(axis=0)
    out = np.zeros(X.shape[1])
    safe = var_X >= 1e-24
    if safe.any():
        cov = X_dev[:, safe].T @ y_dev
        denom = np.sqrt(var_y * var_X[safe])
        out[safe] = np.abs(cov / denom)
    return out


def _residualise(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS-residualise ``y`` against ``x``. Returns ``y - alpha*x - beta``.
    If x is constant, returns ``y - mean(y)``."""
    finite = np.isfinite(y) & np.isfinite(x)
    if finite.sum() < 3 or np.std(x[finite]) < 1e-12:
        out = y.astype(np.float64).copy()
        out -= float(np.mean(out[finite])) if finite.any() else 0.0
        return out
    X = np.column_stack([x[finite].astype(np.float64), np.ones(int(finite.sum()))])
    coef, *_ = np.linalg.lstsq(X, y[finite].astype(np.float64), rcond=None)
    alpha = float(coef[0])
    beta = float(coef[1])
    out = y.astype(np.float64) - alpha * x - beta
    return out


def _mi_pair_bin(
    x: np.ndarray, y: np.ndarray, *, nbins: int,
) -> float:
    """Discrete MI between two 1-D continuous arrays via quantile binning.

    Discretises both axes into ``nbins`` quantile bins (so each bin
    holds ~equal mass), then computes
    ``MI = sum_ij p(i, j) * log(p(i, j) / (p_x(i) * p_y(j)))``
    using the joint frequency table. Equivalent to the bin-based MI
    estimator widely used in feature-selection libraries.

    Tradeoffs vs the kNN Kraskov estimator (sklearn default):

    - **5-10x faster** on n>1000: O(n + nbins^2) vs O(n*log(n))
      kd-tree queries.
    - **Biased low on heavy-tail distributions** because the equal-mass
      bins concentrate rare-tail values into one bin, hiding
      structure.
    - **Less sensitive to small sample size**: the kNN estimator
      becomes unstable below n=50; bin-based stays usable down to
      ~5*nbins rows.

    Implementation notes (engineering-honest, after benchmarking):

    Several optimisation attempts were tried and rejected:

    - **numba JIT of the full pipeline** (commit history: tried with
      both partial-JIT and full-JIT kernels). On n=1000 the JIT
      gives a 2.6x speedup, but on n>=10000 it is *slower* than
      numpy because numpy's sort / searchsorted / bincount are
      SIMD-vectorised C, and numba's JIT'd Python loops cannot
      beat them. Plus a one-shot ~5 s compile cost on first call.
      Production callers always pass mi_sample_n>=20K rows, so
      numpy wins where it matters. Removed.
    - **np.partition instead of np.quantile** for cut edges. The
      single-position partition is 1.5x faster than np.quantile,
      but the multi-position np.partition (one call selecting all
      nbins-1 positions) becomes O(n * nbins) and ends up *slower*
      on n>=100K than np.quantile's optimised sort-based path.
      Reverted to np.quantile.

    Verdict: the numpy implementation here is at the
    speed-of-vectorised-C floor for this algorithm. Further wins
    require dropping to a different algorithm entirely (e.g. a
    streaming hash-bin estimator that avoids the O(n log n) sort
    altogether).
    """
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 5 * nbins:
        return 0.0
    x_f = x[finite]
    y_f = y[finite]
    qs = np.linspace(0, 1, nbins + 1)[1:-1]
    x_edges = np.quantile(x_f, qs)
    y_edges = np.quantile(y_f, qs)
    x_idx = np.searchsorted(x_edges, x_f, side="right").astype(np.int64)
    y_idx = np.searchsorted(y_edges, y_f, side="right").astype(np.int64)
    np.clip(x_idx, 0, nbins - 1, out=x_idx)
    np.clip(y_idx, 0, nbins - 1, out=y_idx)
    combo = x_idx * nbins + y_idx
    joint_counts = np.bincount(combo, minlength=nbins * nbins).reshape(nbins, nbins)
    n_total = float(joint_counts.sum())
    if n_total <= 0:
        return 0.0
    pxy = joint_counts.astype(np.float64) / n_total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    log_terms = np.zeros_like(pxy)
    log_terms[nz] = np.log(pxy[nz] / (px * py)[nz])
    mi = float((pxy * log_terms).sum())
    return max(0.0, mi)


def _mi_to_target(
    feature_matrix: np.ndarray,
    target: np.ndarray,
    *,
    n_neighbors: int,
    random_state: int,
    estimator: str = "knn",
    nbins: int = 16,
    aggregation: str = "mean",
) -> float:
    """Aggregated MI of each feature column with ``target``.

    Two estimators:

    - ``"knn"`` (default): sklearn's Kraskov kNN estimator. Higher
      accuracy on heavy-tail distributions; slow on n>10k.
    - ``"bin"``: quantile-bin estimator (``_mi_pair_bin`` per column).
      5-10x faster; biased low on heavy-tail.

    Aggregation across feature columns:

    - ``"mean"`` (default since 2026-05-10, R10b stat #1):
      ``sum_j MI(x_j, target) / n_features``. Invariant to feature
      count -- the metric stays comparable when the feature set
      shrinks (because the base column was excluded). Statistician's
      review flagged the legacy ``"sum"`` aggregation as biased:
      sum-of-marginal-MI overcounts shared information when X is
      correlated, AND the over-count differs between numerator
      ``MI(T, X_no_base)`` and denominator ``MI(y, X_no_base)``
      because changing the target reweights how features overlap.
      Mean is the simplest fix that removes the dimension confound.
    - ``"sum"``: legacy behaviour. Set explicitly for backward-
      compatibility on existing benchmarks.
    """
    finite = np.isfinite(target) & np.all(np.isfinite(feature_matrix), axis=1)
    if finite.sum() < 50:
        return 0.0
    target_f = target[finite]
    fm_f = feature_matrix[finite]
    if fm_f.shape[1] == 0:
        return 0.0
    if estimator == "bin":
        per_feat = np.array([
            _mi_pair_bin(fm_f[:, j], target_f, nbins=nbins)
            for j in range(fm_f.shape[1])
        ])
    else:
        from sklearn.feature_selection import mutual_info_regression
        per_feat = mutual_info_regression(
            fm_f, target_f, n_neighbors=n_neighbors, random_state=random_state,
        )
    if aggregation == "sum":
        return float(np.sum(per_feat))
    # Default: mean (statistician #1).
    return float(np.mean(per_feat))


def _build_tiny_model(family: str, *, n_estimators: int, num_leaves: int,
                      learning_rate: float, random_state: int,
                      deterministic: bool = False) -> Any:
    """Lazy-build a tiny regressor for the requested family. Lazy
    imports keep the discovery module light when those libraries
    aren't installed.

    When ``deterministic=True``, inject the well-known per-family
    determinism flags so run-to-run results are bit-exact at a
    5-10% per-fit cost. See ``deterministic_screening_models`` config
    field for the rationale.

    LightGBM determinism set:
    - ``deterministic=True``: forces deterministic histograms +
      bin-construction + tree-learner.
    - ``force_row_wise=True``: row-wise histogram aggregation is
      deterministic; the column-wise default is faster but uses
      atomic adds whose order varies.
    - ``force_col_wise=False``: explicitly OFF; otherwise it overrides
      ``force_row_wise``.

    XGBoost determinism set:
    - ``tree_method="hist"``: explicit hist; the auto-pick may flip
      to ``"approx"`` with non-deterministic atomic ops.
    - ``predictor="auto"``: keep -- predict path is already deterministic.
    - XGB doesn't expose a single ``deterministic`` switch the way
      LGB does; ``hist`` is the deterministic path.

    CatBoost determinism set:
    - ``boosting_type="Plain"``: the ``Ordered`` default is faster
      but uses random ordering which differs run-to-run; ``Plain``
      is deterministic.

    Linear (Ridge) is already deterministic by construction.
    """
    family_lower = family.lower()
    if family_lower in ("lgb", "lightgbm"):
        import lightgbm as lgb
        kwargs = dict(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1, verbose=-1, force_col_wise=True,
        )
        if deterministic:
            # ``force_col_wise`` + ``force_row_wise`` are mutually
            # exclusive in LightGBM; flip the pair when going
            # deterministic.
            kwargs["force_col_wise"] = False
            kwargs["force_row_wise"] = True
            kwargs["deterministic"] = True
        return lgb.LGBMRegressor(**kwargs)
    if family_lower in ("xgb", "xgboost"):
        import xgboost as xgb
        kwargs = dict(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1, verbosity=0,
        )
        if deterministic:
            kwargs["tree_method"] = "hist"
        return xgb.XGBRegressor(**kwargs)
    if family_lower in ("cb", "catboost"):
        import catboost as cb
        kwargs = dict(
            iterations=n_estimators,
            depth=4,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=False,
        )
        if deterministic:
            kwargs["boosting_type"] = "Plain"
        return cb.CatBoostRegressor(**kwargs)
    if family_lower in ("linear", "ridge"):
        from sklearn.linear_model import Ridge
        # Ridge is deterministic by construction; the flag is a no-op
        # here but accepting the kwarg keeps the call signature
        # uniform across families.
        return Ridge(alpha=1.0, random_state=random_state)
    raise ValueError(
        f"_build_tiny_model: unknown family '{family}'. "
        "Supported: lightgbm, xgboost, catboost, linear / ridge."
    )


def _tiny_cv_rmse_raw_y(
    y_train: np.ndarray,
    x_train_matrix: np.ndarray,
    *,
    family: str,
    n_estimators: int,
    num_leaves: int,
    learning_rate: float,
    cv_folds: int,
    random_state: int,
    n_jobs: int = 1,
    deterministic: bool = False,
    return_per_bin: bool = False,
    n_bins: int = 5,
    bin_var: Optional[np.ndarray] = None,
):
    """CV-RMSE of a tiny model trained DIRECTLY on raw y (no transform).

    Used as the raw-y baseline against which composite-target tiny CV-RMSEs
    are compared in :meth:`CompositeTargetDiscovery._tiny_model_rerank`.
    Composite specs that fail to beat this baseline are rejected -- the
    primary safeguard that catches "wrong base" cases where MI-gain
    passes barely but the resulting target is harder for the model to
    predict than y itself (e.g. subtracting a spatial coordinate that has
    global trend with y but no structural residual signal).

    Same fit / fold / parallelism contract as :func:`_tiny_cv_rmse_y_scale`
    so the comparison is apples-to-apples.
    """
    from sklearn.model_selection import KFold
    n = len(y_train)
    if n < cv_folds * 10:
        return float("nan")
    y_clean = y_train.astype(np.float64)
    if not np.all(np.isfinite(y_clean)):
        finite_mask = np.isfinite(y_clean)
        y_clean = y_clean[finite_mask]
        x_clean = x_train_matrix[finite_mask]
    else:
        x_clean = x_train_matrix
    if len(y_clean) < cv_folds * 10:
        return float("nan")

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # bin_var aligns to the masked y_clean / x_clean. If caller
    # passed it, mask it the same way.
    if bin_var is not None and len(bin_var) == len(y_train):
        if not np.all(np.isfinite(y_train)):
            bin_var_clean = bin_var[np.isfinite(y_train)]
        else:
            bin_var_clean = bin_var
    else:
        bin_var_clean = None

    def _one_fold(
        train_fold: np.ndarray, val_fold: np.ndarray,
    ) -> Tuple[float, Optional[np.ndarray]]:
        try:
            model = _build_tiny_model(
                family,
                n_estimators=n_estimators,
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                random_state=random_state,
                deterministic=deterministic,
            )
            if n_jobs > 1 and hasattr(model, "set_params"):
                try:
                    model.set_params(n_jobs=1)
                except Exception:
                    pass
            model.fit(x_clean[train_fold], y_clean[train_fold])
            y_hat = np.asarray(model.predict(x_clean[val_fold])).reshape(-1)
            diff = y_hat.astype(np.float64) - y_clean[val_fold]
            finite = np.isfinite(diff)
            if finite.sum() == 0:
                return float("nan"), None
            rmse = float(np.sqrt(np.mean(diff[finite] * diff[finite])))
            per_bin = None
            if return_per_bin and bin_var_clean is not None:
                per_bin = _per_bin_rmse(
                    y_clean[val_fold], y_hat,
                    bin_var_clean[val_fold], n_bins=n_bins,
                )
            return rmse, per_bin
        except Exception:
            return float("nan"), None

    splits = list(kf.split(x_clean))
    if n_jobs > 1 and len(splits) > 1:
        try:
            from joblib import Parallel, delayed
            fold_results = Parallel(
                n_jobs=min(n_jobs, len(splits)),
                backend="threading",
            )(delayed(_one_fold)(tr, va) for tr, va in splits)
        except ImportError:
            fold_results = [_one_fold(tr, va) for tr, va in splits]
    else:
        fold_results = [_one_fold(tr, va) for tr, va in splits]
    fold_rmses = [r for r, _ in fold_results if math.isfinite(r)]
    if not fold_rmses:
        if return_per_bin:
            return float("nan"), np.full(n_bins, float("nan"))
        return float("nan")
    mean_rmse = float(np.mean(fold_rmses))
    if not return_per_bin:
        return mean_rmse
    per_bin_arrays = [pb for _, pb in fold_results if pb is not None]
    if not per_bin_arrays:
        return mean_rmse, np.full(n_bins, float("nan"))
    per_bin_stack = np.stack(per_bin_arrays, axis=0)
    with np.errstate(invalid="ignore"):
        per_bin_mean = np.nanmean(per_bin_stack, axis=0)
    return mean_rmse, per_bin_mean


def _tiny_cv_rmse_y_scale_multiseed(
    *args,
    n_seed_repeats: int = 1,
    base_random_state: int = 0,
    return_per_seed: bool = False,
    **kwargs,
):
    """Multi-seed wrapper around :func:`_tiny_cv_rmse_y_scale`.

    R10b improvement #10: with cv_folds=3, a single CV split has
    high variance. Repeat the K-fold split with different random
    seeds and return the MEDIAN of the per-seed mean RMSEs (instead
    of a single point estimate). When ``return_per_bin=True``, also
    returns the per-bin median across seeds.

    R10b statistician #4: when ``return_per_seed=True``, also returns
    the array of per-seed mean RMSEs so callers can run a paired
    Wilcoxon test against a reference (raw-y baseline) array.

    ``n_seed_repeats=1`` is the legacy single-seed path -- exact
    same numerical result as calling the underlying function once.
    """
    if n_seed_repeats <= 1:
        kwargs["random_state"] = base_random_state
        result = _tiny_cv_rmse_y_scale(*args, **kwargs)
        if return_per_seed:
            mean = result[0] if isinstance(result, tuple) else result
            per_seed_arr = np.array(
                [mean] if math.isfinite(mean) else [],
                dtype=np.float64,
            )
            if isinstance(result, tuple):
                return result + (per_seed_arr,)
            return result, per_seed_arr
        return result
    seed_results = []
    seed_per_bins = []
    return_pb = kwargs.get("return_per_bin", False)
    for s_idx in range(n_seed_repeats):
        kwargs["random_state"] = base_random_state + s_idx * 7919
        result = _tiny_cv_rmse_y_scale(*args, **kwargs)
        if return_pb and isinstance(result, tuple):
            mean_rmse, per_bin = result
            if math.isfinite(mean_rmse):
                seed_results.append(mean_rmse)
                seed_per_bins.append(per_bin)
        else:
            if math.isfinite(result):
                seed_results.append(result)
    seed_arr = np.array(seed_results, dtype=np.float64)
    if not seed_results:
        if return_pb:
            res = float("nan"), np.full(kwargs.get("n_bins", 5), float("nan"))
            return res + (seed_arr,) if return_per_seed else res
        return (float("nan"), seed_arr) if return_per_seed else float("nan")
    median_rmse = float(np.median(seed_results))
    if return_pb:
        if seed_per_bins:
            stack = np.stack(seed_per_bins, axis=0)
            with np.errstate(invalid="ignore"):
                median_per_bin = np.nanmedian(stack, axis=0)
        else:
            median_per_bin = np.full(kwargs.get("n_bins", 5), float("nan"))
        if return_per_seed:
            return median_rmse, median_per_bin, seed_arr
        return median_rmse, median_per_bin
    if return_per_seed:
        return median_rmse, seed_arr
    return median_rmse


def _tiny_cv_rmse_raw_y_multiseed(
    *args,
    n_seed_repeats: int = 1,
    base_random_state: int = 0,
    return_per_seed: bool = False,
    **kwargs,
):
    """Multi-seed wrapper around :func:`_tiny_cv_rmse_raw_y`. See
    :func:`_tiny_cv_rmse_y_scale_multiseed` for the rationale."""
    if n_seed_repeats <= 1:
        kwargs["random_state"] = base_random_state
        result = _tiny_cv_rmse_raw_y(*args, **kwargs)
        if return_per_seed:
            mean = result[0] if isinstance(result, tuple) else result
            per_seed_arr = np.array(
                [mean] if math.isfinite(mean) else [],
                dtype=np.float64,
            )
            if isinstance(result, tuple):
                return result + (per_seed_arr,)
            return result, per_seed_arr
        return result
    seed_results = []
    seed_per_bins = []
    return_pb = kwargs.get("return_per_bin", False)
    for s_idx in range(n_seed_repeats):
        kwargs["random_state"] = base_random_state + s_idx * 7919
        result = _tiny_cv_rmse_raw_y(*args, **kwargs)
        if return_pb and isinstance(result, tuple):
            mean_rmse, per_bin = result
            if math.isfinite(mean_rmse):
                seed_results.append(mean_rmse)
                seed_per_bins.append(per_bin)
        else:
            if math.isfinite(result):
                seed_results.append(result)
    seed_arr = np.array(seed_results, dtype=np.float64)
    if not seed_results:
        if return_pb:
            res = float("nan"), np.full(kwargs.get("n_bins", 5), float("nan"))
            return res + (seed_arr,) if return_per_seed else res
        return (float("nan"), seed_arr) if return_per_seed else float("nan")
    median_rmse = float(np.median(seed_results))
    if return_pb:
        if seed_per_bins:
            stack = np.stack(seed_per_bins, axis=0)
            with np.errstate(invalid="ignore"):
                median_per_bin = np.nanmedian(stack, axis=0)
        else:
            median_per_bin = np.full(kwargs.get("n_bins", 5), float("nan"))
        if return_per_seed:
            return median_rmse, median_per_bin, seed_arr
        return median_rmse, median_per_bin
    if return_per_seed:
        return median_rmse, seed_arr
    return median_rmse


def _per_bin_rmse(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    bin_var: np.ndarray,
    n_bins: int = 5,
) -> np.ndarray:
    """RMSE within each quantile-bin of ``bin_var``. Returns
    array of shape ``(n_bins,)``; bins with too few rows return NaN.

    Used by the regime-aware gate to detect specs that beat raw on
    average but underperform within a particular slice of the data
    (e.g. logratio is correct on multiplicative-regime rows but
    actively wrong on additive-regime rows; mean RMSE hides this).
    """
    finite = np.isfinite(y_true) & np.isfinite(y_hat) & np.isfinite(bin_var)
    if finite.sum() < n_bins * 5:
        return np.full(n_bins, float("nan"))
    y_t = y_true[finite]
    y_p = y_hat[finite]
    bv = bin_var[finite]
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(bv, qs)
    bin_idx = np.searchsorted(edges, bv, side="right")
    np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)
    out = np.full(n_bins, float("nan"))
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 5:
            continue
        diff = y_p[mask] - y_t[mask]
        out[b] = float(np.sqrt(np.mean(diff * diff)))
    return out


def _tiny_cv_rmse_y_scale(
    y_train: np.ndarray,
    base_train: np.ndarray,
    transform: "Transform",
    fitted_params: Dict[str, Any],
    x_train_matrix: np.ndarray,
    *,
    family: str,
    n_estimators: int,
    num_leaves: int,
    learning_rate: float,
    cv_folds: int,
    random_state: int,
    n_jobs: int = 1,
    deterministic: bool = False,
    return_per_bin: bool = False,
    n_bins: int = 5,
):
    """Compute CV-RMSE of a tiny model on the y-scale (after inverse).

    1. Apply ``transform.forward`` to (y_train, base_train) -> T.
    2. K-fold split on the train rows.
    3. For each fold: fit tiny model on T_train_fold, predict T_hat
       on the held-out fold, apply transform.inverse to recover
       y_hat in the original scale, score against y_held.
    4. Return mean across folds.

    Folds run in parallel when ``n_jobs > 1`` via joblib. Each fold
    fit gets ``n_jobs_per_fit = max(1, total_cpus // n_jobs)`` cores
    so the inner LightGBM doesn't oversubscribe. NaN if anything
    degenerates so callers can deprioritise.
    """
    from sklearn.model_selection import KFold
    n = len(y_train)
    if n < cv_folds * 10:
        return float("nan")
    valid = transform.domain_check(y_train, base_train)
    if valid.sum() < cv_folds * 10:
        return float("nan")
    y_clean = y_train[valid].astype(np.float64)
    base_clean = base_train[valid].astype(np.float64)
    x_clean = x_train_matrix[valid]
    t_clean = transform.forward(y_clean, base_clean, fitted_params)
    if not np.all(np.isfinite(t_clean)):
        return float("nan")

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def _one_fold(
        train_fold: np.ndarray, val_fold: np.ndarray,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Return (fold_rmse, per_bin_rmse_or_None)."""
        try:
            model = _build_tiny_model(
                family,
                n_estimators=n_estimators,
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                random_state=random_state,
                deterministic=deterministic,
            )
            # When folds run in parallel, cap LightGBM's intra-fit
            # threads to avoid CPU oversubscription.
            if n_jobs > 1 and hasattr(model, "set_params"):
                try:
                    model.set_params(n_jobs=1)
                except Exception:
                    pass
            model.fit(x_clean[train_fold], t_clean[train_fold])
            t_hat = np.asarray(model.predict(x_clean[val_fold])).reshape(-1)
            y_hat = transform.inverse(
                t_hat, base_clean[val_fold], fitted_params,
            )
            # R10b improvement #4: wrapper-aware clipping. The
            # production CompositeTargetEstimator.predict applies
            # the same y-clip on inverse output to keep predictions
            # inside a physically plausible range. Mirror that here
            # so screening RMSE matches deployed RMSE (otherwise
            # heavy-tail transforms like logratio look better in
            # screening than they actually deliver).
            y_clip_low, y_clip_high = _y_train_clip_bounds(
                y_clean[train_fold]
            )
            y_hat = np.clip(
                y_hat.astype(np.float64), y_clip_low, y_clip_high,
            )
            # Domain-violation fallback: rows where the transform's
            # domain_check fails on val use y_train_median (matches
            # wrapper.predict). The wrapper computes domain_check on
            # (y, base) but at inference y is unknown -- so the
            # wrapper fallback uses y=None handling. Here on val we
            # know y_clean[val_fold]; emulate the wrapper logic by
            # fall-backing rows where y_hat is non-finite OR where
            # the inverse pushed beyond the clip.
            non_finite = ~np.isfinite(y_hat)
            if non_finite.any():
                y_train_median = float(np.median(
                    y_clean[train_fold][np.isfinite(y_clean[train_fold])]
                )) if np.isfinite(y_clean[train_fold]).any() else 0.0
                y_hat[non_finite] = y_train_median
            diff = y_hat - y_clean[val_fold]
            finite = np.isfinite(diff)
            if finite.sum() == 0:
                return float("nan"), None
            rmse = float(np.sqrt(np.mean(diff[finite] * diff[finite])))
            per_bin = None
            if return_per_bin:
                per_bin = _per_bin_rmse(
                    y_clean[val_fold], y_hat,
                    base_clean[val_fold], n_bins=n_bins,
                )
            return rmse, per_bin
        except Exception:
            return float("nan"), None

    splits = list(kf.split(x_clean))
    if n_jobs > 1 and len(splits) > 1:
        try:
            from joblib import Parallel, delayed
            fold_results = Parallel(
                n_jobs=min(n_jobs, len(splits)),
                backend="threading",
            )(delayed(_one_fold)(tr, va) for tr, va in splits)
        except ImportError:
            fold_results = [_one_fold(tr, va) for tr, va in splits]
    else:
        fold_results = [_one_fold(tr, va) for tr, va in splits]

    fold_rmses = [r for r, _ in fold_results if math.isfinite(r)]
    if not fold_rmses:
        if return_per_bin:
            return float("nan"), np.full(n_bins, float("nan"))
        return float("nan")
    mean_rmse = float(np.mean(fold_rmses))
    if not return_per_bin:
        return mean_rmse
    # Aggregate per-bin: mean across folds (NaN-skipping).
    per_bin_arrays = [pb for _, pb in fold_results if pb is not None]
    if not per_bin_arrays:
        return mean_rmse, np.full(n_bins, float("nan"))
    per_bin_stack = np.stack(per_bin_arrays, axis=0)
    with np.errstate(invalid="ignore"):
        per_bin_mean = np.nanmean(per_bin_stack, axis=0)
    return mean_rmse, per_bin_mean


def _sample_indices(
    n: int, sample_n: Optional[int], random_state: int,
    *,
    strategy: str = "random",
    y: Optional[np.ndarray] = None,
    n_strata: int = 10,
) -> np.ndarray:
    """Return a sorted array of row indices to use for MI screening.

    Two strategies:

    - ``"random"`` (default): uniform random sample of ``sample_n``
      rows from ``n``. Cheap, unbiased on average, but high-variance
      on heavy-tail targets (the rare-tail rows that carry most of
      the signal can be over- or under-represented in any one draw).

    - ``"stratified_quantile"``: bin ``y`` into ``n_strata`` quantile
      bins, then sample ``sample_n / n_strata`` rows from each bin.
      Tail rows get oversampled relative to natural frequency. Use
      when ``y`` is heavy-tail (financial returns, fraud scores,
      queue lengths) -- gives stable MI rankings across runs because
      each bin contributes a guaranteed number of rows.

    Sorted so the (mostly-temporal) row order is preserved -- avoids
    biasing the MI estimate on temporal data.
    """
    if sample_n is None or n <= sample_n:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    if strategy == "random" or y is None or n_strata < 2:
        idx = rng.choice(n, size=sample_n, replace=False)
        idx.sort()
        return idx
    if strategy != "stratified_quantile":
        raise ValueError(
            f"_sample_indices: unknown strategy '{strategy}'. "
            "Choose from 'random' or 'stratified_quantile'."
        )

    # Stratified quantile sampling. Bin y into n_strata quantile
    # bins, sample ceil(sample_n / n_strata) from each.
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if y_arr.size != n:
        # Caller passed mismatched y; fall back to random.
        idx = rng.choice(n, size=sample_n, replace=False)
        idx.sort()
        return idx
    finite_mask = np.isfinite(y_arr)
    if finite_mask.sum() < n_strata * 2:
        # Too few finite y; can't stratify, fall back to random.
        idx = rng.choice(n, size=sample_n, replace=False)
        idx.sort()
        return idx
    # Compute quantile cuts on finite y.
    qs = np.linspace(0, 1, n_strata + 1)[1:-1]
    cuts = np.quantile(y_arr[finite_mask], qs)
    # Assign each finite row to a stratum [0, n_strata-1]; non-finite
    # rows get a separate stratum at the end so they aren't dropped
    # silently.
    stratum = np.searchsorted(cuts, y_arr, side="right")
    np.clip(stratum, 0, n_strata - 1, out=stratum)
    stratum[~finite_mask] = n_strata  # extra "non-finite" bin

    per_stratum = max(1, sample_n // n_strata)
    picked: List[np.ndarray] = []
    for s in range(n_strata + 1):
        bin_rows = np.where(stratum == s)[0]
        if bin_rows.size == 0:
            continue
        take = min(bin_rows.size, per_stratum)
        if take == bin_rows.size:
            picked.append(bin_rows)
        else:
            chosen = rng.choice(bin_rows, size=take, replace=False)
            picked.append(chosen)
    out = np.concatenate(picked) if picked else np.arange(min(n, sample_n))
    out.sort()
    return out[:sample_n]


class CompositeTargetDiscovery:
    """Auto-find the best (base, transform) pairs for a regression target.

    Workflow
    --------
    1. Resolve base candidates (auto via residualised-MI ranking, OR
       user-supplied list). Apply the forbidden-pattern + corr + ptp
       filters to drop columns that are leakage-prone (target encoding,
       derived-from-y, near-constant).
    2. For each (base, transform) pair, fit transform-specific params
       on **train_idx only**, compute T on the train sample, and score
       MI(T, X \\ {base}) against MI(y, X).
    3. Filter by ``min_valid_domain_frac`` and ``eps_mi_gain``, sort
       by MI gain descending, keep top ``top_k_after_mi``.

    Leakage discipline (CRITICAL)
    -----------------------------
    Every fitted parameter (alpha/beta for linear_residual, MAD for
    logratio, eps for ratio, MI bin edges for screening, y-clip
    quantiles, etc.) is computed from rows in ``train_idx`` ONLY. Test
    and validation rows are NEVER touched at fit. The unit test
    ``test_alpha_train_only_changes_with_train_idx`` proves this:
    fitting on two different ``train_idx`` slices of the same df
    yields different alpha, while fitting on the same train_idx
    yields identical alpha. If you ever change the implementation to
    add a "use full df for X" shortcut, that test will fail.
    """

    def __init__(self, config: Any) -> None:
        if isinstance(config, dict):
            from .configs import CompositeTargetDiscoveryConfig
            config = CompositeTargetDiscoveryConfig(**config)
        self.config = config
        self._patterns_compiled: List[re.Pattern] = [
            re.compile(p) for p in config.forbidden_base_patterns
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: Any,
        target_col: str,
        feature_cols: Sequence[str],
        train_idx: np.ndarray,
        val_idx: Optional[np.ndarray] = None,
        test_idx: Optional[np.ndarray] = None,
    ) -> "CompositeTargetDiscovery":
        """Discover composite-target specs.

        Parameters
        ----------
        df
            Pandas or polars frame containing ``target_col`` and
            ``feature_cols`` as columns.
        target_col
            Column name of the regression target.
        feature_cols
            Candidate feature columns. Base candidates are drawn from
            this set when ``config.base_candidates="auto"``.
        train_idx
            Row indices to use for fitting transform params and
            scoring. **Required** -- no implicit "use full df" shortcut.
        val_idx, test_idx
            Stored on the instance for later integrity checks; never
            touched during fit.
        """
        if not self.config.enabled:
            self.specs_: List[CompositeSpec] = []
            self.report_: List[Dict[str, Any]] = []
            self.train_idx_ = np.asarray(train_idx)
            self._df_ref = df
            self._target_col = target_col
            return self

        train_idx = np.asarray(train_idx)
        # Stash the identifiers BEFORE the early-return paths so
        # _filter_features (which reads ``self._target_col``) and
        # iter_transform (which reads ``self._df_ref``) work even on
        # the no-spec degenerate cases.
        self._target_col = target_col
        self._df_ref = df
        self.train_idx_ = train_idx

        if train_idx.size < 50:
            logger.warning(
                "[CompositeTargetDiscovery] train_idx has only %d rows; "
                "MI estimates unreliable. Discovery yields no specs.",
                train_idx.size,
            )
            self.specs_ = []
            self.report_ = []
            return self

        t0 = timer()

        # Pull target on train rows. We never touch val/test.
        y_full = _extract_column_array(df, target_col)
        y_train = y_full[train_idx]

        # R10b stat #8: auto-boost mi_n_strata on heavy-tail y. The
        # default 10 strata produces unstable MI estimates when the
        # tail dominates the signal -- one or two tail rows per bin.
        # Detect via skew or kurtosis on train; if either is high,
        # bump n_strata to ``mi_n_strata_heavy_tail`` (default 30).
        # User-configured ``mi_n_strata`` remains the floor; we only
        # bump when the auto-detected boost is HIGHER.
        y_finite_for_check = y_train[np.isfinite(y_train)]
        if y_finite_for_check.size >= 100:
            y_std = float(y_finite_for_check.std())
            if y_std > 1e-12:
                z_centered = (y_finite_for_check - y_finite_for_check.mean()) / y_std
                skew = float(np.mean(z_centered ** 3))
                kurt = float(np.mean(z_centered ** 4) - 3.0)
                if abs(skew) > 2.0 or kurt > 5.0:
                    boost = int(getattr(
                        self.config, "mi_n_strata_heavy_tail", 30,
                    ))
                    cur_n_strata = int(getattr(
                        self.config, "mi_n_strata", 10,
                    ))
                    if boost > cur_n_strata:
                        # Mutate config in-place ONLY if we own a
                        # copy (avoid leaking into callers' shared
                        # config). model_copy is safe here because
                        # discovery already gets a per-target config
                        # clone in core.py when hint is enabled.
                        try:
                            new_cfg = self.config.model_copy(
                                update={"mi_n_strata": boost}
                            )
                            self.config = new_cfg
                            logger.info(
                                "[CompositeTargetDiscovery] heavy-tail y "
                                "detected (skew=%.2f, kurt=%.2f); boosted "
                                "mi_n_strata %d -> %d.",
                                skew, kurt, cur_n_strata, boost,
                            )
                        except Exception:
                            pass  # leave at user-configured value.

        # Filter feature_cols by name patterns AND constancy on train.
        usable_features = self._filter_features(df, feature_cols, y_train, train_idx)

        # Resolve base candidates.
        base_candidates = self._resolve_base_candidates(
            df, target_col, usable_features, y_train, train_idx,
        )
        if not base_candidates:
            logger.warning(
                "[CompositeTargetDiscovery] no usable base candidates after "
                "forbidden-pattern / corr / ptp / numeric filters. "
                "Discovery yields no specs."
            )
            self.specs_ = []
            self.report_ = []
            return self

        # Down-sample for MI screening. Stratified-quantile when
        # configured -- guarantees per-bin coverage on heavy-tail y.
        y_train_for_strat = y_full[train_idx]
        sample_idx = _sample_indices(
            train_idx.size, self.config.mi_sample_n, self.config.random_state,
            strategy=getattr(self.config, "mi_sample_strategy", "random"),
            y=y_train_for_strat,
            n_strata=getattr(self.config, "mi_n_strata", 10),
        )
        train_idx_screen = train_idx[sample_idx]
        y_screen = y_full[train_idx_screen]

        # mi_y baseline is computed PER-BASE because the X-without-base
        # feature set differs per candidate. Comparing MI(T, X_no_base)
        # against MI(y, X) (full X) confounds two effects: target
        # transformation AND removal of the dominant feature. We want
        # only the first effect, so both halves use the same feature
        # set: X without the base column.

        # Score each (base, transform).
        candidates: List[Dict[str, Any]] = []
        for base in base_candidates:
            base_train = _extract_column_array(df, base)[train_idx]
            base_screen = base_train[sample_idx]
            x_remaining = [c for c in usable_features if c != base]
            if not x_remaining:
                continue
            x_remaining_matrix = self._build_feature_matrix(
                df, x_remaining, train_idx_screen,
            )

            # MI(y, X_remaining) -- baseline for THIS base. The model
            # trained on raw y from X_remaining (base dropped from
            # features) sets the bar; a composite target only earns
            # its keep if MI(T, X_remaining) > this.
            mi_y_for_base = _mi_to_target(
                x_remaining_matrix, y_screen,
                n_neighbors=self.config.mi_n_neighbors,
                random_state=self.config.random_state,
                estimator=self.config.mi_estimator,
                nbins=self.config.mi_nbins,
                aggregation=getattr(self.config, "mi_aggregation", "mean"),
            )

            for transform_name in self.config.transforms:
                try:
                    transform = get_transform(transform_name)
                except UnknownTransformError as exc:
                    logger.warning("[CompositeTargetDiscovery] %s; skipping.", exc)
                    continue

                # Domain check on train, drop invalids, fit transform
                # params on the surviving rows only.
                valid = transform.domain_check(y_train, base_train)
                valid_frac = float(valid.mean()) if valid.size else 0.0
                if valid_frac < self.config.min_valid_domain_frac:
                    candidates.append(self._reject(
                        base, transform_name, mi_y_for_base, valid_frac,
                        reason=f"valid_domain_frac={valid_frac:.3f} "
                               f"< {self.config.min_valid_domain_frac:.3f}",
                    ))
                    continue
                if not valid.any():
                    continue

                fitted_params = transform.fit(y_train[valid], base_train[valid])
                # T on the screening sample (which is a subset of train).
                valid_screen = transform.domain_check(y_screen, base_screen)
                if valid_screen.sum() < 50:
                    candidates.append(self._reject(
                        base, transform_name, mi_y_for_base, valid_frac,
                        reason="too few rows in screening sample after domain filter",
                    ))
                    continue
                t_screen = transform.forward(
                    y_screen[valid_screen], base_screen[valid_screen], fitted_params,
                )

                # MI(T, X_remaining) on the same valid rows -- comparable
                # to mi_y_for_base computed on the same x_remaining.
                x_screen_valid = x_remaining_matrix[valid_screen]
                mi_t = _mi_to_target(
                    x_screen_valid, t_screen,
                    n_neighbors=self.config.mi_n_neighbors,
                    random_state=self.config.random_state,
                    estimator=self.config.mi_estimator,
                    nbins=self.config.mi_nbins,
                aggregation=getattr(self.config, "mi_aggregation", "mean"),
                )
                # When the screening sample shrunk after domain
                # filtering (logratio with negative rows in train),
                # the mi_y baseline for THIS base must also be
                # recomputed on the same valid_screen subset to keep
                # comparison fair.
                if valid_screen.sum() < y_screen.size:
                    mi_y_compare = _mi_to_target(
                        x_screen_valid, y_screen[valid_screen],
                        n_neighbors=self.config.mi_n_neighbors,
                        random_state=self.config.random_state,
                        estimator=self.config.mi_estimator,
                        nbins=self.config.mi_nbins,
                aggregation=getattr(self.config, "mi_aggregation", "mean"),
                    )
                else:
                    mi_y_compare = mi_y_for_base
                mi_gain = mi_t - mi_y_compare

                # R10b stat #8: bootstrap CI on mi_gain. The
                # point-estimate has a noise floor that scales with
                # screening-sample size and y-tail heaviness; the
                # absolute eps_mi_gain threshold misses this. Bootstrap
                # produces a 95% CI; the gate compares against the
                # LOWER CI bound (LCB), not the point estimate. Spec
                # is rejected if LCB <= eps_mi_gain.
                bootstrap_n = int(getattr(
                    self.config, "mi_gain_bootstrap_n", 0,
                ))
                mi_gain_lcb = mi_gain  # default: point estimate.
                if bootstrap_n > 0:
                    boot_rng = np.random.default_rng(
                        int(getattr(
                            self.config, "mi_gain_bootstrap_random_state", 12345,
                        ))
                    )
                    n_screen = int(valid_screen.sum())
                    boot_gains = np.empty(bootstrap_n)
                    for b in range(bootstrap_n):
                        idx_b = boot_rng.integers(0, n_screen, size=n_screen)
                        x_boot = x_screen_valid[idx_b]
                        t_boot = t_screen[idx_b]
                        y_boot = y_screen[valid_screen][idx_b]
                        try:
                            mi_t_b = _mi_to_target(
                                x_boot, t_boot,
                                n_neighbors=self.config.mi_n_neighbors,
                                random_state=self.config.random_state,
                                estimator=self.config.mi_estimator,
                                nbins=self.config.mi_nbins,
                                aggregation=getattr(
                                    self.config, "mi_aggregation", "mean",
                                ),
                            )
                            mi_y_b = _mi_to_target(
                                x_boot, y_boot,
                                n_neighbors=self.config.mi_n_neighbors,
                                random_state=self.config.random_state,
                                estimator=self.config.mi_estimator,
                                nbins=self.config.mi_nbins,
                                aggregation=getattr(
                                    self.config, "mi_aggregation", "mean",
                                ),
                            )
                            boot_gains[b] = mi_t_b - mi_y_b
                        except Exception:
                            boot_gains[b] = float("nan")
                    boot_finite = boot_gains[np.isfinite(boot_gains)]
                    if boot_finite.size >= bootstrap_n // 2:
                        mi_gain_lcb = float(np.percentile(boot_finite, 2.5))

                spec = CompositeSpec(
                    name=f"{target_col}__{transform_name}__{base}",
                    target_col=target_col,
                    transform_name=transform_name,
                    base_column=base,
                    fitted_params=dict(fitted_params),
                    mi_gain=mi_gain,
                    mi_y=mi_y_compare,
                    mi_t=mi_t,
                    valid_domain_frac=valid_frac,
                    n_train_rows=int(valid.sum()),
                )
                candidates.append({
                    "spec": spec,
                    "kept": False,  # set after filtering
                    "reason": "",
                    "mi_gain_lcb": float(mi_gain_lcb),
                })

        # Filter + sort.
        kept_specs: List[CompositeSpec] = []
        for entry in candidates:
            spec: Optional[CompositeSpec] = entry.get("spec")
            if spec is None:
                continue  # already a reject
            # R10b stat #8: gate compares LCB (lower CI bound), not
            # point estimate, when bootstrap is enabled. Falls back
            # to point estimate when LCB unavailable.
            mi_gain_for_gate = entry.get("mi_gain_lcb", spec.mi_gain)
            if mi_gain_for_gate <= self.config.eps_mi_gain:
                entry["reason"] = (
                    f"mi_gain={spec.mi_gain:.4f} <= eps={self.config.eps_mi_gain:.4f}"
                )
                continue
            kept_specs.append(spec)
            entry["kept"] = True

        kept_specs.sort(key=lambda s: -s.mi_gain)
        kept_specs = kept_specs[: self.config.top_k_after_mi]

        # R10b stat #6: rolling-origin alpha drift detection for
        # linear_residual specs. Fit alpha on first / second halves
        # of train; Chow-style z-score on the difference. Specs with
        # |z| > threshold either rejected or flagged depending on
        # config.
        if (getattr(self.config, "detect_linear_residual_alpha_drift", True)
                and any(s.transform_name == "linear_residual"
                        for s in kept_specs)):
            self._alpha_drift_flags: Dict[str, Dict[str, float]] = {}
            drift_threshold = float(getattr(
                self.config, "alpha_drift_z_threshold", 3.0,
            ))
            reject_on_drift = bool(getattr(
                self.config, "reject_on_alpha_drift", False,
            ))
            half = len(train_idx) // 2
            if half >= 50:
                drift_dropped: List[Tuple[str, float]] = []
                drift_kept: List[CompositeSpec] = []
                for s in kept_specs:
                    if s.transform_name != "linear_residual":
                        drift_kept.append(s)
                        continue
                    base_full = _extract_column_array(df, s.base_column)
                    y_full_arr = y_full
                    idx1 = train_idx[:half]
                    idx2 = train_idx[half:]
                    try:
                        params1 = _linear_residual_fit(
                            y_full_arr[idx1], base_full[idx1],
                        )
                        params2 = _linear_residual_fit(
                            y_full_arr[idx2], base_full[idx2],
                        )
                    except Exception:
                        drift_kept.append(s)
                        continue
                    a1 = float(params1.get("alpha", 0.0))
                    a2 = float(params2.get("alpha", 0.0))
                    # Pooled SE estimate via residual variance / std(base).
                    base_t = base_full[train_idx]
                    base_t_finite = base_t[np.isfinite(base_t)]
                    base_std = (
                        float(base_t_finite.std()) if base_t_finite.size > 1
                        else 1.0
                    )
                    y_t = y_full_arr[train_idx]
                    y_t_finite = y_t[np.isfinite(y_t)]
                    y_std = (
                        float(y_t_finite.std()) if y_t_finite.size > 1
                        else 1.0
                    )
                    # Approximate SE(alpha_hat) ~ y_std / (sqrt(n_half) *
                    # base_std). The factor of 2 from the half-sample.
                    if base_std < 1e-12 or half < 2:
                        drift_kept.append(s)
                        continue
                    se_alpha = y_std / (np.sqrt(half) * base_std)
                    z = abs(a1 - a2) / max(se_alpha, 1e-12)
                    self._alpha_drift_flags[s.name] = {
                        "alpha_first_half": a1,
                        "alpha_second_half": a2,
                        "z_score": float(z),
                    }
                    if z > drift_threshold:
                        if reject_on_drift:
                            drift_dropped.append((s.name, float(z)))
                            continue
                        else:
                            logger.warning(
                                "[CompositeTargetDiscovery] alpha drift "
                                "detected for spec=%s (alpha first-half="
                                "%.4f, second-half=%.4f, z=%.2f > %.2f). "
                                "Concept drift -- LR may underperform on "
                                "test. Set reject_on_alpha_drift=True to "
                                "drop automatically.",
                                s.name, a1, a2, z, drift_threshold,
                            )
                    drift_kept.append(s)
                if drift_dropped:
                    logger.info(
                        "[CompositeTargetDiscovery] alpha drift gate "
                        "dropped %d linear_residual spec(s): %s",
                        len(drift_dropped),
                        ", ".join(
                            f"{n}(z={z:.2f})"
                            for n, z in drift_dropped[:5]
                        ),
                    )
                kept_specs = drift_kept

        # R10b improvement #6: collapse redundant linear_residual ->
        # diff when alpha ~ 1 and beta ~ 0 (linear_residual has zero
        # information advantage over diff but carries 2 fitted params).
        # Drop linear_residual specs whose alpha is close to 1.0 on
        # the data scale IF a diff spec for the same base also kept.
        # Skipped if the config eps is 0 (feature disabled).
        alpha_eps = float(getattr(
            self.config, "collapse_linear_residual_alpha_eps", 0.05,
        ))
        if alpha_eps > 0 and len(kept_specs) > 1:
            diff_bases = {
                s.base_column for s in kept_specs
                if s.transform_name == "diff"
            }
            collapsed: List[CompositeSpec] = []
            collapsed_dropped: List[Tuple[str, float]] = []
            std_y = float(np.std(y_train[np.isfinite(y_train)])) or 1.0
            for s in kept_specs:
                if s.transform_name != "linear_residual" \
                        or s.base_column not in diff_bases:
                    collapsed.append(s)
                    continue
                alpha = float(s.fitted_params.get("alpha", float("nan")))
                beta = float(s.fitted_params.get("beta", 0.0))
                base_train = _extract_column_array(df, s.base_column)[train_idx]
                base_finite = np.isfinite(base_train)
                std_base = (
                    float(np.std(base_train[base_finite]))
                    if base_finite.any() else 1.0
                )
                if std_base < 1e-12:
                    collapsed.append(s)
                    continue
                # Scale-invariant alpha deviation: how much linear-residual
                # diverges from diff (which is alpha=1, beta=0) relative
                # to the data scale.
                alpha_dev = abs(alpha - 1.0) * std_base / std_y
                beta_dev = abs(beta) / std_y
                if alpha_dev < alpha_eps and beta_dev < alpha_eps:
                    collapsed_dropped.append(
                        (s.name, float(alpha_dev))
                    )
                    continue
                collapsed.append(s)
            if collapsed_dropped:
                preview = ", ".join(
                    f"{n}(alpha_dev={d:.4f})"
                    for n, d in collapsed_dropped[:3]
                )
                logger.info(
                    "[CompositeTargetDiscovery] collapsed %d "
                    "linear_residual spec(s) into diff (alpha~1): %s",
                    len(collapsed_dropped), preview,
                )
            kept_specs = collapsed

        # Phase B: tiny-model rerank. Re-rank the MI-survivors by
        # CV-RMSE on the y-scale (the actual prediction objective).
        # Skip when ``screening == "mi"`` -- callers who want only
        # MI ranking pay zero rerank cost.
        if (kept_specs and self.config.screening in ("tiny_model", "hybrid")
                and self.config.tiny_screening_models in ("single_lgbm",
                                                           "per_family")):
            kept_specs = self._tiny_model_rerank(
                kept_specs=kept_specs,
                df=df,
                target_col=target_col,
                usable_features=usable_features,
                train_idx=train_idx,
                y_full=y_full,
            )

        if not kept_specs:
            mode = self.config.fail_on_no_gain
            msg = (
                f"[CompositeTargetDiscovery] no candidate cleared mi_gain > "
                f"{self.config.eps_mi_gain} on target='{target_col}'."
            )
            if mode == "raise":
                raise RuntimeError(msg)
            logger.warning(msg + f" (fail_on_no_gain={mode!r})")

        elapsed = timer() - t0
        logger.info(
            "[CompositeTargetDiscovery] target='%s' discovered %d spec(s) "
            "from %d candidate(s) in %.2fs",
            target_col, len(kept_specs), len(candidates), elapsed,
        )

        # Bookkeeping. (target_col + df_ref + train_idx already stashed.)
        self.specs_ = kept_specs
        self.report_ = [self._entry_to_report(e) for e in candidates]
        self.val_idx_ = np.asarray(val_idx) if val_idx is not None else None
        self.test_idx_ = np.asarray(test_idx) if test_idx is not None else None
        self.elapsed_seconds_ = elapsed
        return self

    def iter_transform(self, df: Any) -> Iterator[Tuple[str, np.ndarray]]:
        """Yield ``(spec_name, T_values)`` per discovered spec, applied
        to ALL rows of ``df``. Streaming generator: we never
        materialise more than one T column at a time -- on a 4M-row
        frame with K=8 specs that saves ~250 MB peak.

        Rows that fail ``domain_check`` get ``NaN`` in T, so downstream
        target-aware filters drop them automatically when fitting the
        per-spec model. The wrapper at predict time uses its own
        ``y_train_median`` fallback for those rows.
        """
        if not getattr(self, "specs_", None):
            return
        target_col = self._target_col
        y_full = _extract_column_array(df, target_col)
        for spec in self.specs_:
            base_full = _extract_column_array(df, spec.base_column)
            transform = get_transform(spec.transform_name)
            valid = transform.domain_check(y_full, base_full)
            t = np.full(y_full.shape[0], np.nan, dtype=np.float64)
            if valid.any():
                t[valid] = transform.forward(
                    y_full[valid], base_full[valid], spec.fitted_params,
                )
            yield spec.name, t

    def export_specs(self) -> List[Dict[str, Any]]:
        """Plain-dict snapshot of discovered specs for ``metadata`` storage."""
        return [
            {
                "name": s.name,
                "target_col": s.target_col,
                "transform_name": s.transform_name,
                "base_column": s.base_column,
                "fitted_params": dict(s.fitted_params),
                "mi_gain": s.mi_gain,
                "mi_y": s.mi_y,
                "mi_t": s.mi_t,
                "valid_domain_frac": s.valid_domain_frac,
                "n_train_rows": s.n_train_rows,
            }
            for s in getattr(self, "specs_", [])
        ]

    def report(self) -> List[Dict[str, Any]]:
        """All evaluated candidates including rejected ones with reasons."""
        return list(getattr(self, "report_", []))

    @property
    def tiny_rerank_scores_(self) -> Dict[str, float]:
        """Per-spec tiny CV-RMSE on y-scale (after Phase B rerank).

        Empty when ``screening="mi"`` or rerank didn't run. Keyed by
        spec name. Useful for surfacing "why did this composite get
        kept / rejected" diagnostics.
        """
        return dict(getattr(self, "_tiny_rerank_scores", {}))

    @property
    def raw_y_baseline_rmse_(self) -> float:
        """Tiny CV-RMSE of a model trained directly on raw y on the
        same screening sample / folds / family used by Phase B rerank.

        ``nan`` when the raw-y baseline gate didn't run
        (``require_beats_raw_baseline=False``, screening="mi", or
        degenerate sample).
        """
        return float(getattr(self, "_raw_y_baseline_rmse", float("nan")))

    def filter_drops(self) -> List[Dict[str, Any]]:
        """Columns that were filtered out before MI ranking, with reason
        and the offending value (corr, ptp, n_finite). Useful for audit
        when discovery seems to "miss" an obvious base candidate -- the
        most common cause is a corr-threshold false positive on a
        legitimate autoregressive lag feature.
        """
        return list(getattr(self, "_filter_drops", []))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _filter_features(
        self,
        df: Any,
        feature_cols: Sequence[str],
        y_train: np.ndarray,
        train_idx: np.ndarray,
    ) -> List[str]:
        """Drop columns that are non-numeric, near-constant on train,
        match a forbidden name pattern, or correlate suspiciously
        highly with y on train (likely derived-from-y leakage).

        Drops are recorded on ``self._filter_drops`` (list of dicts
        with name + reason + value) so :meth:`fit` can surface them
        in the report and so callers can audit false positives -- the
        corr filter in particular is prone to misfiring on legitimate
        autoregressive lag features such as ``TVT_prev``.
        """
        # First pass: cheap-fail filters (name patterns, type, finite
        # count, near-constant). Build a list of survivors + their
        # train-row arrays so the corr check can be vectorised across
        # all survivors in ONE matrix op (2.2x faster vs per-column
        # ``_safe_corr`` loop on 200 cols x 80K rows; benchmarked
        # 2026-05-10).
        drops: List[Dict[str, Any]] = []
        corr_drops: List[Tuple[str, float]] = []
        candidates: List[str] = []
        candidate_arrays: List[np.ndarray] = []
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
        # ``abs(_safe_corr(arr, y_train))`` loop. NaN rows in the
        # survivor matrix are imputed with column-mean before the
        # corr-vs-y dot product, which is a small approximation
        # versus per-column NaN masking but only matters for columns
        # with sparse NaN -- and those have already passed the
        # ``finite_mask.sum() < 50`` gate above with at least 50
        # finite rows. Acceptable trade-off for the ~600ms saving on
        # 200-feature filter calls.
        kept: List[str] = []
        if candidates:
            X_train = np.column_stack(candidate_arrays)
            # Impute non-finite cells with per-column mean to keep
            # the vectorised dot product well-defined.
            col_means = np.nanmean(
                np.where(np.isfinite(X_train), X_train, np.nan),
                axis=0,
            )
            non_finite_mask = ~np.isfinite(X_train)
            if non_finite_mask.any():
                X_train = X_train.copy()
                # Per-column mean fill (broadcast).
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
        # Loud warning for corr-threshold drops: this is the filter
        # most likely to misfire on legitimate strong predictors
        # (autoregressive lags, near-deterministic features). Make it
        # visible at INFO so users can spot a false positive.
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

    def _resolve_base_candidates(
        self,
        df: Any,
        target_col: str,
        usable_features: Sequence[str],
        y_train: np.ndarray,
        train_idx: np.ndarray,
    ) -> List[str]:
        """Return the base candidates to evaluate.

        For ``base_candidates="auto"``, rank features by *structural*
        MI gain: for each feature ``x``, residualise ``y`` against
        ``x`` (remove the linear contribution) and score
        ``MI(residual, X \\ {x})``. Features whose autoregressive
        contribution genuinely *opens up* the rest of the feature
        space rank higher than features that just correlate with ``y``
        through a global trend.
        """
        config = self.config
        if isinstance(config.base_candidates, str) and config.base_candidates == "auto":
            return self._auto_base(df, usable_features, y_train, train_idx)
        # Explicit list. Keep only entries that survived feature filters.
        explicit = list(config.base_candidates)
        kept = [c for c in explicit if c in usable_features]
        if len(kept) != len(explicit):
            dropped = sorted(set(explicit) - set(kept))
            logger.warning(
                "[CompositeTargetDiscovery] explicit base_candidates dropped "
                "by filters (forbidden/constant/non-numeric/leak-corr): %s", dropped,
            )
        return kept

    def _auto_base(
        self,
        df: Any,
        usable_features: Sequence[str],
        y_train: np.ndarray,
        train_idx: np.ndarray,
    ) -> List[str]:
        """Rank candidates by per-feature MI with y on the screening
        sample, take the top-K.

        Why pairwise MI(y, x) and not the more elaborate "residualised
        gain" of round-2 critique R2.27: the residualised metric
        ranks candidates by how predictable ``y - alpha*x - beta``
        is from the remaining features. On a feature whose linear
        contribution is small, the residual still contains the
        dominant feature itself (we did not subtract it), so the
        remaining feature set predicts the residual perfectly --
        which inverts the ranking versus what we want. Pairwise
        MI(y, x) directly measures "how much information about y
        does this single feature carry" and surfaces ``TVT_prev`` at
        top-1 on the canonical autoregressive case.

        The forbidden-base + ptp + corr filters elsewhere already
        catch the pathologies the residualised metric was meant to
        guard against (target encoding, near-constant features,
        derived-from-y).
        """
        if not usable_features:
            # Every feature was filtered out (forbidden / non-numeric /
            # constant / corr-threshold). Don't ask sklearn to do MI on
            # a 0-column matrix -- it raises ValueError. Return empty
            # cleanly so discovery falls through to the no-spec path.
            logger.info(
                "[CompositeTargetDiscovery] auto-base: 0 usable features "
                "after filtering; no base candidates available."
            )
            return []

        # Hint-aware ranking: BaselineDiagnostics ablation already
        # measured each feature's predictive contribution directly
        # (drop feature -> RMSE delta). That signal beats pairwise
        # MI(y, x), which gets fooled by features with global trend
        # but no structural residual signal (spatial coords on
        # geographically-trended y is the canonical case). When a
        # hint is provided, prepend hint features (preserving order)
        # then fill remaining slots with MI-ranked features.
        usable_set = set(usable_features)
        hint_raw = list(getattr(self.config, "dominant_features_hint", None) or [])
        hint_kept: List[str] = []
        hint_dropped: List[str] = []
        for c in hint_raw:
            if c in usable_set and c not in hint_kept:
                hint_kept.append(c)
            else:
                hint_dropped.append(c)
        if hint_dropped:
            logger.info(
                "[CompositeTargetDiscovery] dominant_features_hint dropped "
                "%d entries (filtered or not in feature_cols): %s",
                len(hint_dropped), hint_dropped[:5],
            )
        top_k = self.config.auto_base_top_k
        # R10c bug #5 fix: adaptive hint cap. Previous fixed cap of
        # ``max(1, top_k // 2)`` was too aggressive when BD ablation
        # confidently identified the dominant base (e.g. delta% > 100%
        # for the top-1 feature on production TVT). Now: if the user
        # supplied a hint AND we have ablation strengths in metadata,
        # check the strength signal. Strong hint (top-1 delta_pct >
        # ``hint_strength_threshold_pct``, default 50%) -> use FULL
        # hint (no cap). Weak/absent strength info -> fall back to the
        # half-slot cap so MI-leaders still get evaluated.
        #
        # Rationale: BD ablation directly measures "drop feature -> RMSE
        # delta%" which is a high-quality signal. When it screams +501%
        # for TVT_prev (real production case), trust it; don't dilute
        # with MI-leaders that may be lower-quality features.
        strong_hint_threshold = float(getattr(
            self.config, "hint_strength_threshold_pct", 50.0,
        ))
        # Strength info is plumbed via the suite-level hint precompute
        # at core.py and stored on the discovery instance for this fit.
        # Absent = treat as unknown strength -> use half-slot cap.
        hint_strengths = getattr(self, "_hint_strengths_pct", None)
        is_strong_hint = (
            hint_strengths is not None
            and len(hint_strengths) > 0
            and max(hint_strengths[:len(hint_kept)]) >= strong_hint_threshold
        )
        if is_strong_hint:
            # Full hint -- no cap. Log so it's auditable.
            logger.info(
                "[CompositeTargetDiscovery] auto-base using FULL hint "
                "(%d candidates, max ablation delta%% = %.1f%% >= %.1f%% "
                "threshold; trusting BD over MI ranking).",
                len(hint_kept), max(hint_strengths[:len(hint_kept)]),
                strong_hint_threshold,
            )
            hint_cap = top_k  # effectively no cap
        else:
            hint_cap = max(1, top_k // 2)
            if len(hint_kept) > hint_cap:
                logger.info(
                    "[CompositeTargetDiscovery] auto-base capping hint "
                    "contribution to %d/%d slots (was %d hint candidates; "
                    "strength signal weak or absent) so MI-leaders also "
                    "get evaluated; full hint list preserved as feature "
                    "ordering source.",
                    hint_cap, top_k, len(hint_kept),
                )
                hint_kept = hint_kept[:hint_cap]

        sample_idx = _sample_indices(
            train_idx.size, self.config.mi_sample_n, self.config.random_state,
            strategy=getattr(self.config, "mi_sample_strategy", "random"),
            y=y_train,
            n_strata=getattr(self.config, "mi_n_strata", 10),
        )
        train_idx_screen = train_idx[sample_idx]
        y_screen = y_train[sample_idx]

        x_matrix = self._build_feature_matrix(df, usable_features, train_idx_screen)
        finite = np.isfinite(y_screen) & np.all(np.isfinite(x_matrix), axis=1)
        if finite.sum() < 50:
            logger.warning(
                "[CompositeTargetDiscovery] auto-base: only %d finite rows in "
                "screening sample; falling back to feature-list order.", int(finite.sum()),
            )
            return list(usable_features)[: self.config.auto_base_top_k]
        # Per-feature MI honours config.mi_estimator: bin-based when
        # the screening pipeline opted for the fast estimator.
        if self.config.mi_estimator == "bin":
            mi_per_feature = np.array([
                _mi_pair_bin(x_matrix[finite, j], y_screen[finite],
                             nbins=self.config.mi_nbins)
                for j in range(x_matrix.shape[1])
            ])
        else:
            from sklearn.feature_selection import mutual_info_regression
            mi_per_feature = mutual_info_regression(
                x_matrix[finite], y_screen[finite],
                n_neighbors=self.config.mi_n_neighbors,
                random_state=self.config.random_state,
            )
        # R10b improvement #7: structural detectors for time-index
        # and spatial-coordinate features. Cheap heuristics applied
        # to the screening matrix; flagged features are demoted
        # (large MI penalty) so they only win base selection when
        # genuinely high-MI relative to alternatives.
        demote_set: set = set()
        if getattr(self.config, "auto_base_demote_time_index", True) \
                and finite.sum() >= 50:
            # Spearman(rank(x), arange(n)) computed via the cleaner
            # equivalent: |corr(argsort(argsort(x)), arange(n))|.
            n_screen = int(finite.sum())
            row_idx = np.arange(n_screen, dtype=np.float64)
            # R10c bug #2 extension: hint features are IMMUNE from
            # the time-index demoter too. BD ablation already proved
            # they predict y; demoting silently is wrong.
            time_hint_protected = set(hint_kept) if hint_kept else set()
            for j, col_name in enumerate(usable_features):
                if col_name in time_hint_protected:
                    continue
                col_finite = x_matrix[finite, j]
                col_ranks = np.argsort(np.argsort(col_finite)).astype(np.float64)
                # Pearson on rank vs row-index = Spearman(x, time).
                spearman = abs(_safe_corr(col_ranks, row_idx))
                if spearman > 0.95:
                    demote_set.add(col_name)
            if demote_set:
                logger.info(
                    "[CompositeTargetDiscovery] auto-base detected %d "
                    "time-index-like feature(s) (rank ~ row order, "
                    "|Spearman| > 0.95): %s. Demoted in MI ranking.",
                    len(demote_set), sorted(demote_set)[:5],
                )
        if getattr(self.config, "auto_base_demote_spatial_coords", True) \
                and len(usable_features) >= 3 and finite.sum() >= 50:
            # R10c bug #1 fix: spatial-coord block detector tightened
            # after a production geological-data run demoted 17
            # features (entire feature set). Previously: ``>=2 cross-
            # correlations |corr|>0.5`` -- fires on any moderately-
            # correlated feature group.
            #
            # Tightened criteria for "spatial-coord block":
            #   1. Block size 3 <= K <= 6 (X/Y/Z triplet up to a
            #      5-coord positional spec; anything larger is a
            #      feature GROUP, not spatial coords).
            #   2. EVERY pair within the block has |corr| > 0.75
            #      (not 0.5 -- geological features routinely correlate
            #      at 0.5-0.7 from physics, not from being coords).
            #   3. Mean within-block |corr| > 0.80 (catches X/Y/Z
            #      typical corr range while rejecting lower-corr
            #      industrial feature groups).
            # All three must hold; otherwise the group is preserved.
            X_screen = x_matrix[finite]
            n_feats = X_screen.shape[1]
            corr_matrix = np.zeros((n_feats, n_feats))
            for j in range(n_feats):
                for k in range(j + 1, n_feats):
                    c = abs(_safe_corr(X_screen[:, j], X_screen[:, k]))
                    corr_matrix[j, k] = c
                    corr_matrix[k, j] = c
            spatial_demoted: List[str] = []
            # For each feature j, find its "tight neighbourhood":
            # features k where |corr(j, k)| > 0.75. If that
            # neighbourhood (including j) is size 3-6 AND has mean
            # within-pair corr > 0.80, demote ALL members.
            for j, col_name in enumerate(usable_features):
                tight_neighbours = np.where(corr_matrix[j] > 0.75)[0]
                if not (2 <= len(tight_neighbours) <= 5):
                    continue
                block_idx = np.r_[j, tight_neighbours]
                block_idx = np.unique(block_idx)
                if not (3 <= len(block_idx) <= 6):
                    continue
                # Mean within-block pairwise corr.
                sub = corr_matrix[np.ix_(block_idx, block_idx)]
                upper = sub[np.triu_indices_from(sub, k=1)]
                if upper.size == 0:
                    continue
                if float(upper.mean()) < 0.80:
                    continue
                # Also require EVERY pair > 0.75 (no weak edge in the
                # cluster).
                if float(upper.min()) < 0.75:
                    continue
                # Cluster qualifies -- demote every member EXCEPT
                # those on the hint list (BD ablation already proved
                # they predict y; demoting them silently is the same
                # production bug pattern as the dedup-vs-hint race).
                hint_protected = set(hint_kept) if hint_kept else set()
                for k in block_idx:
                    name_k = usable_features[k]
                    if name_k in hint_protected:
                        continue
                    if name_k not in demote_set:
                        demote_set.add(name_k)
                        spatial_demoted.append(name_k)
            if spatial_demoted:
                logger.info(
                    "[CompositeTargetDiscovery] auto-base detected "
                    "spatial-coord block of %d feature(s) (tight "
                    "cluster, |pair-corr| > 0.75, mean > 0.80, size "
                    "3-6): %s. Demoted in MI ranking.",
                    len(spatial_demoted),
                    sorted(spatial_demoted)[:8],
                )

        # R10b improvement #2: permutation-MI null filter. Catches
        # features whose MI(y, x) is non-trivial only because of a
        # shared monotonic component (time/spatial trend), not
        # structural information about y. Computes MI(y, shuffle(x))
        # with block shuffles to preserve marginal autocorrelation,
        # then requires MI(y, x) > mean_null + n_sigma * std_null.
        n_perms = int(getattr(self.config, "auto_base_null_perms", 0) or 0)
        if n_perms > 0:
            n_sigma = float(getattr(
                self.config, "auto_base_null_z_threshold", 3.0,
            ))
            block_len_cfg = getattr(
                self.config, "auto_base_null_block_length", "auto",
            )
            n_screen = int(finite.sum())
            if isinstance(block_len_cfg, str) and block_len_cfg == "auto":
                block_len = max(1, int(np.sqrt(n_screen)))
            else:
                try:
                    block_len = max(1, int(block_len_cfg))
                except (TypeError, ValueError):
                    block_len = max(1, int(np.sqrt(n_screen)))

            def _block_shuffle(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
                if block_len <= 1:
                    out = arr.copy()
                    rng.shuffle(out)
                    return out
                m = arr.size
                n_blocks = (m + block_len - 1) // block_len
                blocks = [arr[i * block_len:(i + 1) * block_len]
                          for i in range(n_blocks)]
                perm = rng.permutation(len(blocks))
                shuffled = np.concatenate([blocks[p] for p in perm])
                return shuffled[:m]

            rng_perm = np.random.default_rng(
                int(self.config.random_state) + 7919
            )
            y_finite = y_screen[finite]
            null_means = np.zeros(x_matrix.shape[1])
            null_stds = np.zeros(x_matrix.shape[1])
            for j in range(x_matrix.shape[1]):
                col = x_matrix[finite, j]
                null_mis = np.empty(n_perms)
                for p in range(n_perms):
                    shuffled = _block_shuffle(col, rng_perm)
                    if self.config.mi_estimator == "bin":
                        null_mis[p] = _mi_pair_bin(
                            shuffled, y_finite, nbins=self.config.mi_nbins,
                        )
                    else:
                        from sklearn.feature_selection import mutual_info_regression
                        null_mis[p] = float(mutual_info_regression(
                            shuffled.reshape(-1, 1), y_finite,
                            n_neighbors=self.config.mi_n_neighbors,
                            random_state=self.config.random_state,
                        )[0])
                null_means[j] = float(null_mis.mean())
                null_stds[j] = float(null_mis.std())
            null_threshold = null_means + n_sigma * np.maximum(
                null_stds, 1e-9,
            )
            passes_null = mi_per_feature > null_threshold
            null_dropped: List[Tuple[str, float, float]] = []
            for j, (mi_val, col_name) in enumerate(
                zip(mi_per_feature.tolist(), usable_features)
            ):
                if not passes_null[j]:
                    null_dropped.append((
                        col_name, float(mi_val), float(null_threshold[j]),
                    ))
            if null_dropped:
                preview = ", ".join(
                    f"{n}(mi={m:.4f}<=null+{n_sigma:.0f}sigma={t:.4f})"
                    for n, m, t in null_dropped[:5]
                )
                logger.info(
                    "[CompositeTargetDiscovery] permutation-MI null "
                    "dropped %d feature(s) (z<%.0f, block_len=%d, "
                    "perms=%d): %s",
                    len(null_dropped), n_sigma, block_len, n_perms,
                    preview,
                )
            # Mask out features that didn't pass the null.
            mi_for_ranking = np.where(passes_null, mi_per_feature, -np.inf)
        else:
            mi_for_ranking = mi_per_feature.copy()
        # R10b improvement #7: apply demotion to time-index / spatial-
        # coord candidates. Subtract a large penalty so they sort
        # below all non-demoted features but stay reachable as a
        # last resort.
        if demote_set:
            for j, col_name in enumerate(usable_features):
                if col_name in demote_set:
                    mi_for_ranking[j] -= 1e6
        ranked = sorted(
            zip(mi_for_ranking.tolist(), usable_features),
            key=lambda t: -t[0],
        )
        # Strip features whose MI was masked out (-inf) so the ranking
        # tail doesn't include null-failed candidates.
        ranked = [(m, c) for m, c in ranked if math.isfinite(m)]
        # Cross-base correlation dedup. Two highly-correlated bases
        # (typical: ``TVT_prev``, ``TVT_prev_lag2``, ``TVT_smooth_3``)
        # produce near-identical composites that waste Phase B compute
        # AND inflate ensemble correlation, hurting cross-target
        # diversity. After ranking, drop a candidate if its absolute
        # corr against any already-kept candidate exceeds
        # ``auto_base_dedup_corr_threshold``. Skipped candidates are
        # logged at INFO. Configurable via
        # ``CompositeTargetDiscoveryConfig.auto_base_dedup_corr_threshold``;
        # set to 1.0 to disable.
        dedup_threshold = float(getattr(
            self.config, "auto_base_dedup_corr_threshold", 0.95,
        ))
        if 0 < dedup_threshold < 1.0 and len(ranked) > 1:
            kept_ranked: List[Tuple[float, str]] = []
            kept_arrays: Dict[str, np.ndarray] = {}
            dedup_dropped: List[Tuple[str, str, float]] = []
            # R10c bug #2 fix: hint features are IMMUNE from dedup.
            # Otherwise on geological data with high feature
            # cross-correlation (e.g. Z ~ TVT_prev at |corr|=0.974),
            # the lower-MI hint candidate gets dropped against a
            # higher-MI non-hint one, then later re-injected by the
            # hint-merge step with a poisoned score from the demoter.
            # Hint features were chosen by the upstream BD ablation
            # specifically because they predict y; their relevance is
            # already established and shouldn't be filtered by
            # raw-feature redundancy.
            hint_set = set(hint_kept)
            for mi_score, col in ranked:
                col_arr = x_matrix[finite, usable_features.index(col)]
                drop_due_to: Optional[Tuple[str, float]] = None
                if col in hint_set:
                    # Hint features always pass dedup.
                    pass
                else:
                    for kept_col, kept_arr in kept_arrays.items():
                        pair_corr = abs(_safe_corr(col_arr, kept_arr))
                        if pair_corr >= dedup_threshold:
                            drop_due_to = (kept_col, float(pair_corr))
                            break
                if drop_due_to is None:
                    kept_ranked.append((mi_score, col))
                    kept_arrays[col] = col_arr
                else:
                    dedup_dropped.append(
                        (col, drop_due_to[0], drop_due_to[1])
                    )
            if dedup_dropped:
                preview = ", ".join(
                    f"{c}~={ref}(|corr|={corr:.3f})"
                    for c, ref, corr in dedup_dropped[:5]
                )
                logger.info(
                    "[CompositeTargetDiscovery] auto-base dedup dropped "
                    "%d candidate(s) at |corr|>=%.3f: %s",
                    len(dedup_dropped), dedup_threshold, preview,
                )
            ranked = kept_ranked
        # Combine hint (priority) + MI-ranked tail. Hint always wins
        # the leading slots; MI fills up to auto_base_top_k.
        if hint_kept:
            mi_tail: List[str] = []
            for _, c in ranked:
                if c in hint_kept:
                    continue
                mi_tail.append(c)
                if len(hint_kept) + len(mi_tail) >= top_k:
                    break
            top = hint_kept + mi_tail
            top = top[:top_k]
            mi_lookup = {c: mi for mi, c in ranked}
            scores = ", ".join(
                f"{c}={mi_lookup.get(c, float('nan')):.4f}{'(hint)' if c in hint_kept else ''}"
                for c in top
            )
            logger.info(
                "[CompositeTargetDiscovery] auto-base top-%d (%d hint, %d MI): %s",
                len(top), len(hint_kept), len(mi_tail), scores,
            )
            return top

        top = [c for _, c in ranked[: top_k]]
        if top:
            scores = ", ".join(
                f"{c}={mi:.4f}" for mi, c in ranked[: top_k]
            )
            logger.info(
                "[CompositeTargetDiscovery] auto-base top-%d by MI(y, x): %s",
                len(top), scores,
            )
        return top

    def _tiny_model_rerank(
        self,
        kept_specs: List["CompositeSpec"],
        df: Any,
        target_col: str,
        usable_features: Sequence[str],
        train_idx: np.ndarray,
        y_full: np.ndarray,
    ) -> List["CompositeSpec"]:
        """Phase B: re-rank MI-survivors by CV-RMSE on y-scale.

        For each surviving spec:
        1. Build the feature matrix (X-without-base) on a screening
           sample of train rows.
        2. Compute CV-RMSE per family in
           ``self.config.tiny_screening_families``.
        3. Aggregate per-spec score by ``tiny_consensus``:
           - "union": min CV-RMSE across families (best-case).
           - "borda": Borda-count rank aggregation.
        4. Re-sort, take top-``top_m_after_tiny``.
        """
        sample_n = min(self.config.tiny_model_sample_n, train_idx.size)
        # Phase B benefits from stratified sampling on heavy-tail y
        # for the same reason Phase A does -- tiny-model CV-RMSE on a
        # tail-empty sample mis-ranks transforms that only matter in
        # the tail.
        y_train_for_strat = y_full[train_idx]
        sample_idx = _sample_indices(
            train_idx.size, sample_n, self.config.random_state,
            strategy=getattr(self.config, "mi_sample_strategy", "random"),
            y=y_train_for_strat,
            n_strata=getattr(self.config, "mi_n_strata", 10),
        )
        train_idx_screen = train_idx[sample_idx]
        y_screen = y_full[train_idx_screen]

        if self.config.tiny_screening_models == "single_lgbm":
            families = ["lightgbm"]
        else:  # per_family
            families = [f for f in self.config.tiny_screening_families]
            if not families:
                families = ["lightgbm"]

        # Per-spec CV-RMSE per family. When K specs share a base
        # (the typical case: auto-base picks one TVT_prev-style
        # dominant feature, all K transforms operate on it), the
        # per-base ``x_remaining`` matrix and ``base_screen`` array
        # are recomputable from the same inputs. Cache them by base
        # to avoid K redundant builds (each ~50 ndarray copies on a
        # 200K-row sample).
        per_family_scores: Dict[str, List[float]] = {f: [] for f in families}
        _per_base_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for spec in kept_specs:
            cached = _per_base_cache.get(spec.base_column)
            if cached is None:
                base_screen = (
                    _extract_column_array(df, spec.base_column)[train_idx_screen]
                )
                x_remaining = [
                    c for c in usable_features if c != spec.base_column
                ]
                x_matrix = self._build_feature_matrix(
                    df, x_remaining, train_idx_screen,
                )
                _per_base_cache[spec.base_column] = (base_screen, x_matrix)
            else:
                base_screen, x_matrix = cached
            transform = get_transform(spec.transform_name)
            n_seed_repeats = max(1, int(getattr(
                self.config, "tiny_model_n_seed_repeats", 1,
            )))
            use_wilcoxon = bool(getattr(
                self.config, "use_wilcoxon_gate", False,
            ))
            for family in families:
                if use_wilcoxon:
                    result = _tiny_cv_rmse_y_scale_multiseed(
                        y_train=y_screen,
                        base_train=base_screen,
                        transform=transform,
                        fitted_params=spec.fitted_params,
                        x_train_matrix=x_matrix,
                        family=family,
                        n_estimators=self.config.tiny_model_n_estimators,
                        num_leaves=self.config.tiny_model_num_leaves,
                        learning_rate=self.config.tiny_model_learning_rate,
                        cv_folds=self.config.tiny_model_cv_folds,
                        n_jobs=getattr(self.config, "tiny_model_n_jobs", 1),
                        deterministic=getattr(
                            self.config, "deterministic_screening_models", False,
                        ),
                        n_seed_repeats=n_seed_repeats,
                        base_random_state=self.config.random_state,
                        return_per_seed=True,
                    )
                    rmse, per_seed = result[0], result[-1]
                    # Stash per-seed array on the discovery instance
                    # (keyed by spec.name + family) for the Wilcoxon
                    # gate to compare against raw-y per-seed.
                    self._wilcoxon_per_seed_composite = getattr(
                        self, "_wilcoxon_per_seed_composite", {}
                    )
                    self._wilcoxon_per_seed_composite[
                        (spec.name, family)
                    ] = per_seed
                else:
                    rmse = _tiny_cv_rmse_y_scale_multiseed(
                        y_train=y_screen,
                        base_train=base_screen,
                        transform=transform,
                        fitted_params=spec.fitted_params,
                        x_train_matrix=x_matrix,
                        family=family,
                        n_estimators=self.config.tiny_model_n_estimators,
                        num_leaves=self.config.tiny_model_num_leaves,
                        learning_rate=self.config.tiny_model_learning_rate,
                        cv_folds=self.config.tiny_model_cv_folds,
                        n_jobs=getattr(self.config, "tiny_model_n_jobs", 1),
                        deterministic=getattr(
                            self.config, "deterministic_screening_models", False,
                        ),
                        n_seed_repeats=n_seed_repeats,
                        base_random_state=self.config.random_state,
                    )
                per_family_scores[family].append(rmse)

        # Aggregate -> single score per spec.
        consensus = self.config.tiny_consensus
        agg_scores: List[float] = []
        for i, spec in enumerate(kept_specs):
            family_rmses = [per_family_scores[f][i] for f in families]
            finite = [r for r in family_rmses if math.isfinite(r)]
            if not finite:
                agg_scores.append(float("inf"))
                continue
            if consensus == "union":
                # Best (lowest) family RMSE. "Union" = "kept if any
                # family ranks it well".
                agg_scores.append(min(finite))
            elif consensus == "borda":
                # Borda needs ranks per family.
                # Build rank tables per family, sum ranks per spec
                # below at the after-loop step. For simplicity use
                # mean RMSE here as a Borda proxy on a per-spec
                # basis -- for a 2-3 family setup the Borda result
                # collapses to mean rank, equivalent to mean RMSE.
                agg_scores.append(float(np.mean(finite)))
            else:
                agg_scores.append(min(finite))

        # Persist tiny CV-RMSE keyed by spec name -- callers read it
        # via :attr:`CompositeTargetDiscovery.tiny_rerank_scores_`.
        # CompositeSpec is frozen; we keep the per-spec scoring on the
        # discovery instance instead of mutating the spec.
        self._tiny_rerank_scores: Dict[str, float] = {
            kept_specs[i].name: float(agg_scores[i])
            for i in range(len(kept_specs))
        }

        # R10b improvement #1: regime-aware gate. In addition to the
        # global mean RMSE, compute per-quintile-of-base RMSE for each
        # spec AND for the raw-y baseline (binned by the SAME variable
        # for apples-to-apples). A spec is rejected if it loses to
        # raw on any quintile by more than ``per_bin_tolerance`` --
        # catches "two-regime" failures where logratio is correct on
        # multiplicative rows but actively wrong on additive rows.
        per_bin_n_bins = int(getattr(self.config, "per_bin_n_bins", 0) or 0)
        per_bin_tol = float(getattr(
            self.config, "raw_baseline_per_bin_tolerance", 1.10,
        ))
        per_bin_enabled = (
            per_bin_n_bins > 0
            and getattr(self.config, "require_beats_raw_baseline", True)
        )
        # Per-spec per-bin RMSE: spec_name -> ndarray(n_bins,)
        spec_per_bin_rmse: Dict[str, np.ndarray] = {}
        if per_bin_enabled:
            # Recompute per-spec scores with return_per_bin=True. This
            # is a SECOND pass; cost is the per-bin breakdown only
            # (the LGBM fits already happened in the first pass --
            # per-bin reuses fold predictions). For simplicity we just
            # rerun the inner loop with the same params; future
            # optimization can cache predictions from the first pass.
            for i, spec in enumerate(kept_specs):
                cached = _per_base_cache.get(spec.base_column)
                if cached is None:
                    continue
                base_screen, x_remaining_matrix = cached
                transform = get_transform(spec.transform_name)
                # One-family per-bin (use first family for speed).
                family = families[0]
                result = _tiny_cv_rmse_y_scale(
                    y_train=y_screen, base_train=base_screen,
                    transform=transform, fitted_params=spec.fitted_params,
                    x_train_matrix=x_remaining_matrix,
                    family=family,
                    n_estimators=self.config.tiny_model_n_estimators,
                    num_leaves=self.config.tiny_model_num_leaves,
                    learning_rate=self.config.tiny_model_learning_rate,
                    cv_folds=self.config.tiny_model_cv_folds,
                    random_state=self.config.random_state,
                    n_jobs=getattr(self.config, "tiny_model_n_jobs", 1),
                    deterministic=getattr(
                        self.config, "deterministic_screening_models", False,
                    ),
                    return_per_bin=True, n_bins=per_bin_n_bins,
                )
                if isinstance(result, tuple):
                    _, per_bin = result
                    spec_per_bin_rmse[spec.name] = per_bin

        # Raw-y baseline gate. Train a tiny model directly on raw y
        # using the SAME folds / sample / family as the composite
        # rerank above, so the comparison is apples-to-apples. Reject
        # any composite whose tiny RMSE >= raw_baseline * tolerance.
        # Configured via ``require_beats_raw_baseline`` /
        # ``raw_baseline_tolerance``.
        raw_rmse_per_family: Dict[str, float] = {}
        raw_per_bin_per_base: Dict[str, np.ndarray] = {}
        raw_baseline: float = float("nan")
        gate_rejected_names: List[Tuple[str, float, float]] = []
        per_bin_rejected_names: List[Tuple[str, str, float, float]] = []
        if getattr(self.config, "require_beats_raw_baseline", True):
            # Build a feature matrix using ALL usable_features on the
            # screening sample (raw-y training has no special "base"
            # to drop, so include everything).
            x_full = self._build_feature_matrix(
                df, list(usable_features), train_idx_screen,
            )
            n_seed_repeats_raw = max(1, int(getattr(
                self.config, "tiny_model_n_seed_repeats", 1,
            )))
            use_wilcoxon = bool(getattr(
                self.config, "use_wilcoxon_gate", False,
            ))
            raw_per_seed_per_family: Dict[str, np.ndarray] = {}
            for family in families:
                if use_wilcoxon:
                    res = _tiny_cv_rmse_raw_y_multiseed(
                        y_train=y_screen,
                        x_train_matrix=x_full,
                        family=family,
                        n_estimators=self.config.tiny_model_n_estimators,
                        num_leaves=self.config.tiny_model_num_leaves,
                        learning_rate=self.config.tiny_model_learning_rate,
                        cv_folds=self.config.tiny_model_cv_folds,
                        n_jobs=getattr(self.config, "tiny_model_n_jobs", 1),
                        deterministic=getattr(
                            self.config, "deterministic_screening_models", False,
                        ),
                        n_seed_repeats=n_seed_repeats_raw,
                        base_random_state=self.config.random_state,
                        return_per_seed=True,
                    )
                    raw_rmse_per_family[family] = res[0]
                    raw_per_seed_per_family[family] = res[-1]
                else:
                    raw_rmse_per_family[family] = _tiny_cv_rmse_raw_y_multiseed(
                        y_train=y_screen,
                        x_train_matrix=x_full,
                        family=family,
                        n_estimators=self.config.tiny_model_n_estimators,
                        num_leaves=self.config.tiny_model_num_leaves,
                        learning_rate=self.config.tiny_model_learning_rate,
                        cv_folds=self.config.tiny_model_cv_folds,
                        n_jobs=getattr(self.config, "tiny_model_n_jobs", 1),
                        deterministic=getattr(
                            self.config, "deterministic_screening_models", False,
                        ),
                        n_seed_repeats=n_seed_repeats_raw,
                        base_random_state=self.config.random_state,
                    )
            # Per-base raw-y per-bin breakdown for the regime gate.
            # Cached by base column so multiple specs sharing a base
            # compute baselines once.
            if per_bin_enabled:
                for spec in kept_specs:
                    if spec.base_column in raw_per_bin_per_base:
                        continue
                    cached = _per_base_cache.get(spec.base_column)
                    if cached is None:
                        continue
                    base_screen, _ = cached
                    family = families[0]
                    raw_result = _tiny_cv_rmse_raw_y(
                        y_train=y_screen,
                        x_train_matrix=x_full,
                        family=family,
                        n_estimators=self.config.tiny_model_n_estimators,
                        num_leaves=self.config.tiny_model_num_leaves,
                        learning_rate=self.config.tiny_model_learning_rate,
                        cv_folds=self.config.tiny_model_cv_folds,
                        random_state=self.config.random_state,
                        n_jobs=getattr(self.config, "tiny_model_n_jobs", 1),
                        deterministic=getattr(
                            self.config, "deterministic_screening_models", False,
                        ),
                        return_per_bin=True, n_bins=per_bin_n_bins,
                        bin_var=base_screen,
                    )
                    if isinstance(raw_result, tuple):
                        _, raw_per_bin = raw_result
                        raw_per_bin_per_base[spec.base_column] = raw_per_bin
            finite_raw = [r for r in raw_rmse_per_family.values()
                          if math.isfinite(r)]
            if finite_raw:
                # Apples-to-apples with consensus aggregation above.
                if consensus == "union":
                    raw_baseline = min(finite_raw)
                else:
                    raw_baseline = float(np.mean(finite_raw))
            tol = float(getattr(self.config, "raw_baseline_tolerance", 1.02))
            threshold = (raw_baseline * tol
                         if math.isfinite(raw_baseline) else float("inf"))
            self._raw_y_baseline_rmse = (
                float(raw_baseline) if math.isfinite(raw_baseline)
                else float("nan")
            )
            if math.isfinite(raw_baseline):
                survivors = []
                gate_alpha = float(getattr(
                    self.config, "gate_alpha", 0.05,
                ))
                wilcoxon_rejected: List[Tuple[str, float]] = []
                for i, spec in enumerate(kept_specs):
                    score = agg_scores[i]
                    if math.isfinite(score) and score >= threshold:
                        gate_rejected_names.append(
                            (spec.name, score, threshold)
                        )
                        continue
                    # R10b stat #4: paired Wilcoxon signed-rank test
                    # on per-seed RMSE diffs (composite - raw). Reject
                    # spec unless the median diff is significantly
                    # negative at level gate_alpha.
                    if (use_wilcoxon
                            and hasattr(self, "_wilcoxon_per_seed_composite")):
                        family = families[0]
                        comp_per_seed = self._wilcoxon_per_seed_composite.get(
                            (spec.name, family)
                        )
                        raw_per_seed = raw_per_seed_per_family.get(family)
                        if (comp_per_seed is not None
                                and raw_per_seed is not None
                                and len(comp_per_seed) == len(raw_per_seed)
                                and len(comp_per_seed) >= 3):
                            try:
                                from scipy.stats import wilcoxon
                                diff = comp_per_seed - raw_per_seed
                                # One-sided: composite better (less RMSE)
                                # so we want diff < 0; alternative='less'.
                                stat_res = wilcoxon(
                                    diff, alternative="less",
                                    zero_method="wilcox",
                                )
                                p_value = float(stat_res.pvalue)
                                if p_value > gate_alpha:
                                    wilcoxon_rejected.append(
                                        (spec.name, p_value)
                                    )
                                    continue
                            except (ImportError, ValueError) as _wx_err:
                                # Scipy missing or all-zero diffs ->
                                # fall through to the threshold-only
                                # gate (no Wilcoxon rejection).
                                logger.debug(
                                    "[CompositeTargetDiscovery] "
                                    "Wilcoxon gate skipped for spec=%s: %s",
                                    spec.name, _wx_err,
                                )
                    # R10b improvement #1: per-bin gate. Composite
                    # passes the global mean test; now check that
                    # per-bin RMSE doesn't blow out vs the raw-y
                    # per-bin baseline on any quintile of base.
                    if (per_bin_enabled
                            and spec.name in spec_per_bin_rmse
                            and spec.base_column in raw_per_bin_per_base):
                        spec_pb = spec_per_bin_rmse[spec.name]
                        raw_pb = raw_per_bin_per_base[spec.base_column]
                        per_bin_threshold = raw_pb * per_bin_tol
                        # Element-wise compare, ignoring NaN bins.
                        worst_ratio = 0.0
                        worst_bin_idx = -1
                        for b in range(len(spec_pb)):
                            if (math.isfinite(spec_pb[b])
                                    and math.isfinite(raw_pb[b])
                                    and raw_pb[b] > 0):
                                ratio = spec_pb[b] / raw_pb[b]
                                if ratio > worst_ratio:
                                    worst_ratio = ratio
                                    worst_bin_idx = b
                        if worst_ratio >= per_bin_tol:
                            per_bin_rejected_names.append((
                                spec.name,
                                f"bin_{worst_bin_idx}",
                                float(worst_ratio),
                                float(per_bin_tol),
                            ))
                            continue
                    survivors.append((i, spec, score))
                if not survivors:
                    logger.warning(
                        "[CompositeTargetDiscovery] raw-y baseline gate "
                        "rejected ALL %d composite candidate(s) "
                        "(raw_baseline=%.4f, tolerance=%.2f). Examples: %s. "
                        "Falling back to raw target only -- discovery "
                        "yields no specs.",
                        len(gate_rejected_names),
                        raw_baseline, tol,
                        ", ".join(
                            f"{n}=RMSE{r:.4f}>{t:.4f}"
                            for n, r, t in gate_rejected_names[:3]
                        ),
                    )
                    return []
                if gate_rejected_names:
                    logger.info(
                        "[CompositeTargetDiscovery] raw-y baseline gate "
                        "rejected %d/%d composite(s) (raw_baseline=%.4f, "
                        "tolerance=%.2f): %s",
                        len(gate_rejected_names), len(kept_specs),
                        raw_baseline, tol,
                        ", ".join(
                            f"{n}(RMSE={r:.4f}>{t:.4f})"
                            for n, r, t in gate_rejected_names
                        ),
                    )
                if per_bin_rejected_names:
                    logger.info(
                        "[CompositeTargetDiscovery] regime-aware per-bin "
                        "gate rejected %d composite(s) (passed global "
                        "mean but blew out on a base quintile, "
                        "tolerance=%.2f): %s",
                        len(per_bin_rejected_names), per_bin_tol,
                        ", ".join(
                            f"{n}@{b}(ratio={r:.2f}>{t:.2f})"
                            for n, b, r, t in per_bin_rejected_names[:5]
                        ),
                    )
                if wilcoxon_rejected:
                    logger.info(
                        "[CompositeTargetDiscovery] Wilcoxon gate "
                        "rejected %d composite(s) (paired one-sided test "
                        "on per-seed RMSE diffs vs raw, alpha=%.3f): %s",
                        len(wilcoxon_rejected), gate_alpha,
                        ", ".join(
                            f"{n}(p={p:.3f})"
                            for n, p in wilcoxon_rejected[:5]
                        ),
                    )
                # Replace kept_specs/agg_scores with survivors only.
                kept_specs = [s for _, s, _ in survivors]
                agg_scores = [sc for _, _, sc in survivors]

        # Sort by aggregated score (ascending: lowest RMSE wins).
        order = np.argsort(agg_scores)
        reranked = [kept_specs[i] for i in order]
        # Trim to top-M.
        top_m = max(1, self.config.top_m_after_tiny)
        reranked = reranked[:top_m]
        # Logging: show the rerank effect.
        original_top = [s.name for s in kept_specs[: top_m]]
        new_top = [s.name for s in reranked]
        if original_top != new_top:
            logger.info(
                "[CompositeTargetDiscovery] tiny-model rerank changed top-%d. "
                "Before (by mi_gain): %s. After (by CV-RMSE on y-scale): %s.",
                top_m, original_top, new_top,
            )
        else:
            logger.info(
                "[CompositeTargetDiscovery] tiny-model rerank confirmed top-%d "
                "from MI ranking: %s.", top_m, new_top,
            )
        return reranked

    def _build_feature_matrix(
        self, df: Any, cols: Sequence[str], idx: np.ndarray,
    ) -> np.ndarray:
        """Materialise a 2-D ndarray of the requested columns at the
        requested rows. Used only for MI screening on the small
        sample slice -- never on the full frame."""
        if not cols:
            return np.zeros((idx.size, 0), dtype=np.float64)
        cols_arrays = [_extract_column_array(df, c)[idx] for c in cols]
        return np.column_stack(cols_arrays)

    def _reject(
        self, base: str, transform_name: str, mi_y: float, valid_frac: float,
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "spec": None,
            "kept": False,
            "rejected": True,
            "base": base,
            "transform_name": transform_name,
            "valid_domain_frac": valid_frac,
            "mi_y": mi_y,
            "reason": reason,
        }

    def _entry_to_report(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        spec = entry.get("spec")
        if spec is None:
            return {
                "name": f"__{entry['transform_name']}__{entry['base']}",
                "kept": False,
                "rejected": True,
                "reason": entry["reason"],
                "base_column": entry["base"],
                "transform_name": entry["transform_name"],
                "mi_gain": float("nan"),
                "valid_domain_frac": entry.get("valid_domain_frac", float("nan")),
            }
        return {
            "name": spec.name,
            "kept": entry.get("kept", False),
            "rejected": False,
            "reason": entry.get("reason", ""),
            "base_column": spec.base_column,
            "transform_name": spec.transform_name,
            "mi_gain": spec.mi_gain,
            "mi_y": spec.mi_y,
            "mi_t": spec.mi_t,
            "valid_domain_frac": spec.valid_domain_frac,
            "n_train_rows": spec.n_train_rows,
        }

