"""Composite-target transform registry: 11 transforms (diff, ratio, logratio, linear_residual + multi/grouped/quantile/monotonic/ewma/rolling_quantile/frac_diff extended set) + Transform dataclass + UnknownTransformError / DomainViolationError. Split out of composite.py to keep transform-implementation surface separate from the wrapper + discovery surface; composite.py re-exports every symbol below at its bottom for full back-compat."""


from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from types import MappingProxyType as _MappingProxyType
from typing import (
    Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple,
)

import numpy as np

try:
    import numba as _numba  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _numba = None  # type: ignore
    _HAS_NUMBA = False

logger = logging.getLogger(__name__)


# Module-level numba kernels (JIT compile on first call). Pure-Python fallback
# is the recursion in-line below when numba is not installed.
if _HAS_NUMBA:

    @_numba.njit(cache=True)
    def _ewma_kernel(base_f: np.ndarray, alpha: float, anchor: float) -> np.ndarray:
        n = base_f.size
        out = np.empty(n, dtype=np.float64)
        state = anchor
        for i in range(n):
            x = base_f[i]
            if np.isfinite(x):
                state = (1.0 - alpha) * state + alpha * x
            out[i] = state
        return out

    @_numba.njit(cache=True)
    def _frac_diff_inverse_kernel(
        t_f: np.ndarray, lags: int, weights: np.ndarray, anchor: float,
    ) -> np.ndarray:
        n = t_f.size
        out = np.empty(n, dtype=np.float64)
        inv_w0 = 1.0 / weights[0]
        for i in range(n):
            lag_sum = 0.0
            upper = min(i + 1, lags + 1)
            for k_idx in range(1, upper):
                lag_sum += weights[k_idx] * out[i - k_idx]
            for k_idx in range(upper, lags + 1):
                lag_sum += weights[k_idx] * anchor
            out[i] = (t_f[i] - lag_sum) * inv_w0
        return out
else:
    _ewma_kernel = None  # type: ignore
    _frac_diff_inverse_kernel = None  # type: ignore


# Soft-cap MAD floor: when MAD(T_train) is below
# ``_MAD_FLOOR_FRAC * std(y_train)``, we substitute the latter to keep
# the soft-cap bound numerically meaningful even if the transform
# produced a degenerate (near-constant) T on train. Without this,
# logratio's MAD-cap collapses to zero on degenerate train and every
# prediction inverts to ``base * exp(0) = base`` silently.
_MAD_FLOOR_FRAC: float = 1e-3

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
      state. Values may be ndarrays (median/quantile/monotonic/rank/
      smoothing_spline/quantile_normal): serialise via the spec serialiser, NOT raw json.dumps.
    - ``forward(y, base, params)`` -> ``T``: applies the transform.
    - ``inverse(T_hat, base, params)`` -> ``y_hat``: applies the
      inverse. Wrapper additionally clips the output to the y-bounds
      stored alongside ``params``.
    - ``domain_check(y, base)`` -> boolean mask of valid rows. Wrapper
      uses this at fit-time to drop invalid rows BEFORE calling
      ``fit`` / ``forward``, and at predict-time to flag rows where
      the inverse cannot be applied cleanly (those rows fall back to
      ``y_train_median``).
    - ``domain_check_fitted(y, base, params)`` -> boolean mask, OPTIONAL.
      The pre-fit ``domain_check`` cannot see fitted parameters, so any
      transform whose validity depends on a learned parameter (e.g.
      ``log_y``'s ``offset`` -- rows with ``y + offset <= 0`` produce NaN
      under ``log``; ``centered_ratio``'s shift ``c`` -- rows where
      ``base + c`` lands in the near-zero eps-floor band) cannot enforce
      its true domain through ``domain_check`` alone. When a transform
      sets this hook, callers re-evaluate the valid mask AFTER ``fit`` and
      drop the newly-invalid rows so they never reach ``forward`` (which
      would silently emit NaN T). The same ``(y, base)`` -> all-True /
      ``y=None`` predict-time contract as ``domain_check`` applies: with
      ``y=None`` only the base-side, params-derived conditions are gated.
      Transforms without params-dependent validity leave this ``None`` and
      callers fall back to the params-free ``domain_check`` result.
    """

    name: str
    forward: Callable[..., np.ndarray]
    inverse: Callable[..., np.ndarray]
    fit: Callable[..., dict[str, Any]]
    domain_check: Callable[[np.ndarray, np.ndarray], np.ndarray]
    description: str
    tags: frozenset[str] = field(default_factory=frozenset)
    # Optional fitted-params-aware domain refinement. See the contract note
    # above. ``None`` (default) means the transform's validity is fully
    # captured by the params-free ``domain_check``. When set, the signature
    # is ``(y, base, params) -> boolean mask`` with the same ``y=None``
    # predict-time sentinel handling as ``domain_check``.
    domain_check_fitted: Callable[..., np.ndarray] | None = None
    # Grouped-transform support. When True, the wrapper extracts a
    # ``groups`` array from a configured column and
    # passes it as a keyword argument to ``fit`` / ``forward`` /
    # ``inverse``. Used by ``linear_residual_grouped`` (per-segment alpha
    # with James-Stein shrinkage). Default False: legacy transforms
    # keep their 3-arg signatures and the wrapper never passes groups.
    requires_groups: bool = False
    # Unary y-transform support (Pack J): when False the wrapper skips
    # the base-column extraction step and feeds a placeholder ``None``
    # array as the ``base`` arg to fit / forward / inverse / domain_check.
    # Unary transforms (``cbrt_y``, ``log_y``, ``yeo_johnson_y``,
    # ``quantile_normal_y``) declare ``requires_base=False`` so callers
    # may instantiate ``CompositeTargetEstimator`` without configuring
    # a base column. Default True keeps the 11 legacy bivariate
    # transforms unchanged.
    requires_base: bool = True
    # Time-recurrent forward support. When True, ``forward`` reads each row's
    # value in the context of its NEIGHBOURS in the row sequence (an EWMA
    # carry-forward, a centred rolling window, a fractional-difference weight
    # tail). For those transforms the fit-time domain filter must NOT compact
    # the sequence before the forward: dropping a domain-violating row would
    # close the gap and shift every later (or windowed) row's recurrence state,
    # so T near a filtered gap would differ from the predict-time T computed on
    # the uncompacted frame. The wrapper therefore runs the forward over the
    # FULL (uncompacted) y / base sequence and masks to the valid rows AFTER,
    # exactly mirroring the predict path (which never compacts). Default False:
    # the 30+ pointwise transforms see their forward applied to the already-
    # filtered rows, which is bit-identical to the full-then-mask order for
    # them. Set True only for ``ewma_residual`` / ``rolling_quantile_ratio`` /
    # ``frac_diff``.
    recurrent: bool = False


# ----------------------------------------------------------------------
# Parent-resident threshold constants referenced as default-arg values
# by sibling-modules at their top-level (linear / nonlinear). They MUST
# stay in the parent because the siblings import them at module load
# time. The Pack-N additions (multi/grouped/robust gates) live here for
# the same reason.
# ----------------------------------------------------------------------

_LINRES_ROBUST_MAD_K: float = 3.0
"""Hard threshold: rows with |residual| > k * sigma_MAD are dropped before the second-pass OLS. k=3 keeps ~99.7% of inliers under Normality and drops outliers heavier than 3-sigma."""

_LINRES_ROBUST_MIN_KEEP_FRAC: float = 0.5
"""Safety: if the MAD-trim would drop more than (1 - this) of rows, fall back to plain OLS. Protects against pathological all-outlier targets where MAD is too small and trimming becomes destructive."""

# Condition-number gate above which joint OLS is considered unstable
# and the multi-base transform falls back to zero-alpha + intercept.
# 30 is the conventional threshold (Belsley/Kuh/Welsch); above that,
# multicollinearity inflates standard errors enough that the alpha
# estimates carry no useful information. Exposed as module-level so
# tests can monkey-patch without recompiling.
_MULTI_BASE_COND_NUMBER_MAX: float = 30.0

# Minimum rows per group below which the group falls back to the global
# fit. Belsley/Kuh/Welsch rule of thumb is ">= 10 x predictors" for stable
# OLS; for 1-base + intercept that's 20-30. We use 30 as a safe default;
# exposed as module-level so tests can monkey-patch.
_GROUPED_MIN_GROUP_SIZE: int = 30

_QUANTILE_RESIDUAL_DEFAULT_N_BINS: int = 10
_QUANTILE_RESIDUAL_DEFAULT_MIN_BIN_N: int = 50

_MONOTONIC_RESIDUAL_DEFAULT_N_KNOTS: int = 12
_MONOTONIC_RESIDUAL_DEFAULT_MIN_KNOT_N: int = 30

# Pack D 2026-05-18: degeneracy threshold for the fitted PCHIP g(base).
# When ``(knots_y.max() - knots_y.min()) / std(y_train) < this``, the
# spline is essentially a constant -- the transform T = y - g(base)
# collapses to ``T = y - const`` (identity up to a global shift) and
# downstream models on T produce SAME predictions as on raw y. 0.05
# = the spline must capture at least 5% of y's std to count as non-
# trivial; below that the discovery loop must reject the spec.
_MONOTONIC_DEGENERACY_RATIO: float = 0.05

_EWMA_RESIDUAL_DEFAULT_K: int = 7  # half-life-like span; alpha = 2 / (k + 1) ~= 0.25
_FRAC_DIFF_DEFAULT_D: float = 0.5  # Lopez de Prado standard order
_FRAC_DIFF_DEFAULT_LAGS: int = 30  # maximum weight tail used in the truncated series


# ----------------------------------------------------------------------
# Pack J: unary y-only transforms (no base column required).
# Re-export of unary raw helpers; the wrapping into the registry's
# (y, base, params) signature happens in _composite_transforms_registry.
# ----------------------------------------------------------------------
from .unary import (  # noqa: E402,F401
    cbrt_y_fit as _cbrt_y_fit_raw,
    cbrt_y_forward as _cbrt_y_forward_raw,
    cbrt_y_inverse as _cbrt_y_inverse_raw,
    cbrt_y_domain as _cbrt_y_domain_raw,
    log_y_fit as _log_y_fit_raw,
    log_y_forward as _log_y_forward_raw,
    log_y_inverse as _log_y_inverse_raw,
    log_y_domain as _log_y_domain_raw,
    yeo_johnson_y_fit as _yj_y_fit_raw,
    yeo_johnson_y_forward as _yj_y_forward_raw,
    yeo_johnson_y_inverse as _yj_y_inverse_raw,
    yeo_johnson_y_domain as _yj_y_domain_raw,
    quantile_normal_y_fit as _qn_y_fit_raw,
    quantile_normal_y_forward as _qn_y_forward_raw,
    quantile_normal_y_inverse as _qn_y_inverse_raw,
    quantile_normal_y_domain as _qn_y_domain_raw,
    chain_bivariate_then_unary_fit as _chain_fit_raw,
    chain_bivariate_then_unary_forward as _chain_forward_raw,
    chain_bivariate_then_unary_inverse as _chain_inverse_raw,
    chain_multi_stage_fit as _chain_multi_fit_raw,
    chain_multi_stage_forward as _chain_multi_forward_raw,
    chain_multi_stage_inverse as _chain_multi_inverse_raw,
)


# ----------------------------------------------------------------------
# Simple transforms (diff / additive_residual / median_residual /
# y_quantile_clip / ratio / rolling_quantile_ratio): carved into
# ``_composite_transforms_simple.py``. Imported here so the registry
# build below sees the public-name functions.
# ----------------------------------------------------------------------
from .simple import *  # noqa: E402,F401,F403


# ----------------------------------------------------------------------
# Sibling-module function bindings. The two transform clusters live in
# ``_composite_transforms_linear.py`` (linear-residual + logratio family)
# and ``_composite_transforms_nonlinear.py`` (monotonic / quantile / ewma
# / frac-diff / chain / etc.). Imported here -- BEFORE the module-level
# registry construction below but AFTER all parent-resident constants the
# siblings reference as function-signature defaults (_GROUPED_MIN_GROUP_SIZE,
# _QUANTILE_RESIDUAL_DEFAULT_*, etc.). Siblings themselves only lazy-import
# parent helpers from inside function bodies, so loading them here does not
# deadlock.
# ----------------------------------------------------------------------
from .linear import (  # noqa: E402,F401
    _linear_residual_domain, _linear_residual_fit, _linear_residual_forward,
    _linear_residual_grouped_domain, _linear_residual_grouped_fit,
    _linear_residual_grouped_forward, _linear_residual_grouped_inverse,
    _linear_residual_inverse, _linear_residual_multi_domain,
    _linear_residual_multi_fit, _linear_residual_multi_forward,
    _linear_residual_multi_inverse, _linear_residual_robust_fit,
    _logratio_domain, _logratio_fit, _logratio_forward, _logratio_inverse,
)
from .nonlinear import (  # noqa: E402,F401
    _ewma_compute, _ewma_compute_batched, _ewma_dispatch, _ewma_residual_domain,
    _ewma_residual_fit, _ewma_residual_forward, _ewma_residual_inverse,
    _frac_diff_domain, _frac_diff_fit, _frac_diff_forward, _frac_diff_inverse,
    _frac_diff_inverse_compute, _frac_diff_inverse_compute_batched,
    _frac_diff_inverse_dispatch, _frac_diff_weights,
    _james_stein_shrinkage_factor, _lookup_ewma_backend,
    _lookup_frac_diff_inv_backend, _make_chain_transform,
    _make_multi_chain_transform, _monotonic_residual_domain,
    _monotonic_residual_fit, _monotonic_residual_forward, _monotonic_residual_g,
    _monotonic_residual_inverse, _quantile_residual_assign_bins,
    _quantile_residual_domain, _quantile_residual_fit, _quantile_residual_forward,
    _quantile_residual_inverse, _rolling_median, _row_alpha_beta,
)


# ----------------------------------------------------------------------
# Registry + naming: imported AFTER all functional siblings load so the
# registry dict literal can resolve every per-transform function. The
# naming sibling consumes the registry; both re-exported here so callers
# keep using ``from mlframe.training.composite_transforms import ...``.
# ----------------------------------------------------------------------
from .registry import (  # noqa: E402,F401
    _TRANSFORMS_REGISTRY,
    _make_unary_registry_adapter,
    _cbrt_fit, _cbrt_forward, _cbrt_inverse, _cbrt_domain, _cbrt_domain_fitted,
    _log_fit_a, _log_forward_a, _log_inverse_a, _log_domain_a, _log_domain_fitted_a,
    _yj_fit_a, _yj_forward_a, _yj_inverse_a, _yj_domain_a, _yj_domain_fitted_a,
    _qn_fit_a, _qn_forward_a, _qn_inverse_a, _qn_domain_a, _qn_domain_fitted_a,
    _centered_ratio_domain_fitted,
)
from .naming import (  # noqa: E402,F401
    TRANSFORM_NAME_SHORT,
    _COMPOSITE_NAME_FRAGMENTS,
    compose_target_name,
    get_transform,
    is_composite_target_name,
    list_transforms,
)


# Read-only view exported to callers; the underlying ``_TRANSFORMS_REGISTRY`` is module-private and any extension layer must edit it explicitly. Prevents test / extension code from pop-ing a transform and silently corrupting subsequent suite runs in the same process.
TRANSFORMS_REGISTRY: "Mapping[str, Transform]" = _MappingProxyType(_TRANSFORMS_REGISTRY)
