"""Shared imports + per-transform fit/forward/inverse/domain functions + per-unary registry adapters
for the transforms registry, split out of ``registry.py`` so both dict halves
(``_registry_part1.py`` / ``_registry_part2.py``) can share them without a circular import.

Star-imported by both halves (see ``__all__`` below) -- an established re-export pattern in this
package (see ``RUF100`` note in CLAUDE.md re: star-import re-export markers).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# Parent's ``Transform`` dataclass + TAG_* sentinel constants live in
# ``composite_transforms.py``, which imports this sibling at its bottom
# AFTER defining Transform + the TAG constants. The static cycle is
# whitelisted in ``tests/test_meta/test_no_import_cycles.py`` (same
# pattern as ``_composite_transforms_linear`` / ``_nonlinear``).
from . import (
    TAG_CORE,
    TAG_EXTENDED,
    TAG_REGRESSION,
    Transform,
)
from .simple import (
    _additive_residual_domain,
    _additive_residual_fit,
    _additive_residual_forward,
    _additive_residual_inverse,
    _diff_domain,
    _diff_fit,
    _diff_forward,
    _diff_inverse,
    _median_residual_domain,
    _median_residual_fit,
    _median_residual_forward,
    _median_residual_inverse,
    _ratio_domain,
    _ratio_fit,
    _ratio_forward,
    _ratio_inverse,
    _rolling_quantile_ratio_domain,
    _rolling_quantile_ratio_fit,
    _rolling_quantile_ratio_centered_fit,
    _rolling_quantile_ratio_forward,
    _rolling_quantile_ratio_inverse,
    _y_quantile_clip_domain,
    _y_quantile_clip_fit,
    _y_quantile_clip_forward,
    _y_quantile_clip_inverse,
)
from .linear import (
    _linear_residual_domain,
    _linear_residual_fit,
    _linear_residual_forward,
    _linear_residual_grouped_domain,
    _linear_residual_grouped_fit,
    _linear_residual_grouped_forward,
    _linear_residual_grouped_inverse,
    _linear_residual_inverse,
    _linear_residual_multi_domain,
    _linear_residual_multi_fit,
    _linear_residual_multi_forward,
    _linear_residual_multi_inverse,
    _linear_residual_robust_fit,
    _theilsen_residual_fit,
    _logratio_domain,
    _logratio_fit,
    _logratio_forward,
    _logratio_inverse,
)
from .nonlinear import (
    _ewma_residual_domain,
    _ewma_residual_fit,
    _ewma_residual_forward,
    _ewma_residual_inverse,
    _frac_diff_domain,
    _frac_diff_fit,
    _frac_diff_forward,
    _frac_diff_inverse,
    _make_chain_transform,
    _make_multi_chain_transform,
    _monotonic_residual_domain,
    _monotonic_residual_fit,
    _monotonic_residual_forward,
    _monotonic_residual_inverse,
    _quantile_residual_domain,
    _quantile_residual_fit,
    _quantile_residual_forward,
    _quantile_residual_inverse,
)
from .extended import (
    _asinh_residual_domain,
    _asinh_residual_fit,
    _asinh_residual_forward,
    _asinh_residual_inverse,
    _centered_ratio_domain,
    _centered_ratio_fit,
    _centered_ratio_forward,
    _centered_ratio_inverse,
    _geometric_mean_residual_domain,
    _geometric_mean_residual_fit,
    _geometric_mean_residual_forward,
    _geometric_mean_residual_inverse,
    _pairwise_interaction_residual_domain,
    _pairwise_interaction_residual_fit,
    _pairwise_interaction_residual_forward,
    _pairwise_interaction_residual_inverse,
    _polynomial_residual_deg2_domain,
    _polynomial_residual_deg2_fit,
    _polynomial_residual_deg2_forward,
    _polynomial_residual_deg2_inverse,
    _rank_residual_domain,
    _rank_residual_fit,
    _rank_residual_forward,
    _rank_residual_inverse,
    _reciprocal_residual_domain,
    _reciprocal_residual_fit,
    _reciprocal_residual_forward,
    _reciprocal_residual_inverse,
    _smoothing_spline_residual_domain,
    _smoothing_spline_residual_fit,
    _smoothing_spline_residual_forward,
    _smoothing_spline_residual_inverse,
)
from .categorical import (
    _target_encoding_residual_domain,
    _target_encoding_residual_fit,
    _target_encoding_residual_forward,
    _target_encoding_residual_inverse,
)
from ._causal_anchor import (
    _causal_anchor_residual_domain,
    _causal_anchor_residual_fit,
    _causal_anchor_residual_forward,
    _causal_anchor_residual_inverse,
)
from ._second_diff import (
    _second_diff_domain,
    _second_diff_fit,
    _second_diff_forward,
    _second_diff_inverse,
)
from ._rank_ecdf import (
    _rank_ecdf_residual_domain,
    _rank_ecdf_residual_fit,
    _rank_ecdf_residual_forward,
    _rank_ecdf_residual_inverse,
)
from ._seasonal import (
    _seasonal_residual_domain,
    _seasonal_residual_fit,
    _seasonal_residual_forward,
    _seasonal_residual_inverse,
)
from ._volatility import (
    _volatility_normalized_residual_domain,
    _volatility_normalized_residual_fit,
    _volatility_normalized_residual_forward,
    _volatility_normalized_residual_inverse,
)
from ._multi_extra import (
    _asinh_residual_multi_domain,
    _asinh_residual_multi_fit,
    _asinh_residual_multi_forward,
    _asinh_residual_multi_inverse,
    _linear_residual_multi_robust_fit,
)
from ._nadaraya_watson import (
    _nadaraya_watson_residual_domain,
    _nadaraya_watson_residual_fit,
    _nadaraya_watson_residual_forward,
    _nadaraya_watson_residual_inverse,
)
from ._gaussian_copula import (
    _gaussian_copula_residual_domain,
    _gaussian_copula_residual_fit,
    _gaussian_copula_residual_forward,
    _gaussian_copula_residual_inverse,
)
from ._grouped_extra import (
    _ewma_residual_grouped_domain,
    _ewma_residual_grouped_fit,
    _ewma_residual_grouped_forward,
    _ewma_residual_grouped_inverse,
    _frac_diff_grouped_domain,
    _frac_diff_grouped_fit,
    _frac_diff_grouped_forward,
    _frac_diff_grouped_inverse,
    _monotonic_residual_grouped_domain,
    _monotonic_residual_grouped_fit,
    _monotonic_residual_grouped_forward,
    _monotonic_residual_grouped_inverse,
    _quantile_residual_grouped_domain,
    _quantile_residual_grouped_fit,
    _quantile_residual_grouped_forward,
    _quantile_residual_grouped_inverse,
    _rolling_quantile_ratio_grouped_domain,
    _rolling_quantile_ratio_grouped_fit,
    _rolling_quantile_ratio_grouped_forward,
    _rolling_quantile_ratio_grouped_inverse,
)
from .unary import (
    cbrt_y_domain as _cbrt_y_domain_raw,
    cbrt_y_fit as _cbrt_y_fit_raw,
    cbrt_y_forward as _cbrt_y_forward_raw,
    cbrt_y_inverse as _cbrt_y_inverse_raw,
    log_y_domain as _log_y_domain_raw,
    log_y_fit as _log_y_fit_raw,
    log_y_forward as _log_y_forward_raw,
    log_y_inverse as _log_y_inverse_raw,
    quantile_normal_y_domain as _qn_y_domain_raw,
    quantile_normal_y_fit as _qn_y_fit_raw,
    quantile_normal_y_forward as _qn_y_forward_raw,
    quantile_normal_y_inverse as _qn_y_inverse_raw,
    signed_power_y_domain as _sp_y_domain_raw,
    signed_power_y_fit as _sp_y_fit_raw,
    signed_power_y_forward as _sp_y_forward_raw,
    signed_power_y_inverse as _sp_y_inverse_raw,
    yeo_johnson_y_domain as _yj_y_domain_raw,
    yeo_johnson_y_fit as _yj_y_fit_raw,
    yeo_johnson_y_forward as _yj_y_forward_raw,
    yeo_johnson_y_inverse as _yj_y_inverse_raw,
    box_cox_y_domain as _bc_y_domain_raw,
    box_cox_y_fit as _bc_y_fit_raw,
    box_cox_y_forward as _bc_y_forward_raw,
    box_cox_y_inverse as _bc_y_inverse_raw,
)


def _make_unary_registry_adapter(
    fit_fn, forward_fn, inverse_fn, domain_fn, domain_fitted_fn=None,
) -> "tuple[Callable, Callable, Callable, Callable, Callable | None]":
    """Adapt a unary (y, params) signature to the registry's (y, base, params) signature by ignoring ``base``. Returns (fit_adapter, forward_adapter, inverse_adapter, domain_adapter[, domain_fitted_adapter]).

    ``domain_fitted_fn`` (optional, signature ``(y, params) -> mask``) wires
    the fitted-params-aware domain hook for unary transforms whose validity
    depends on a learned parameter (e.g. ``log_y``'s ``offset``: rows with
    ``y + offset <= 0`` are out of domain only once ``offset`` is known). When
    ``None`` the returned 5th element is ``None`` and the registry entry leaves
    ``domain_check_fitted`` unset (params-free ``domain_check`` is exact)."""

    def _fit(y, base):
        """Registry-shaped fit adapter: drop ``base``, delegate to the unary ``fit_fn(y)``."""
        return fit_fn(y)

    def _forward(y, base, params):
        """Registry-shaped forward adapter: drop ``base``, delegate to the unary ``forward_fn(y, params)``."""
        return forward_fn(y, params)

    def _inverse(t_hat, base, params):
        """Registry-shaped inverse adapter: drop ``base``, delegate to the unary ``inverse_fn(t_hat, params)``."""
        return inverse_fn(t_hat, params)

    def _domain(y, base):
        """Registry-shaped domain adapter: params-free unary domain at fit time, all-True at predict time (no base constraint for unary transforms)."""
        # The unary helper accepts (y) or (y, params); the registry
        # contract is domain_check(y, base) at fit-time and (None, base)
        # at predict-time. Predict-side call passes y=None so we cannot
        # apply the unary domain on y -- gate on finite base / always-True
        # for unary which has no base constraint at predict.
        if y is None:
            return np.ones(len(base) if hasattr(base, "__len__") else 1, dtype=bool)
        return domain_fn(y)

    if domain_fitted_fn is None:
        return _fit, _forward, _inverse, _domain, None

    def _domain_fitted(y, base, params):
        """Registry-shaped fitted-domain adapter: params-aware unary domain at fit time, all-True at predict time (see docstring below)."""
        # Fitted-domain for unary: no base constraint, so at predict time
        # (y is None) the per-row domain cannot be re-checked from base
        # alone (e.g. log_y's ``y + offset > 0`` needs y). Return all-True
        # for the predict-side row count, matching ``_domain``. At fit/
        # screening time y is present and we gate on the params-aware
        # unary domain (e.g. ``y + offset > 0``).
        if y is None:
            return np.ones(len(base) if hasattr(base, "__len__") else 1, dtype=bool)
        return domain_fitted_fn(y, params)

    return _fit, _forward, _inverse, _domain, _domain_fitted


# Pre-build per-unary adapters (cheap, done once at import). The 5th element
# is the fitted-params-aware domain adapter (``None`` for transforms whose
# params-free domain is exact).
#
# Explicit annotations below (rather than relying on inference through the tuple-unpack) are
# needed for mypy to resolve these names when imported by ``_registry_part1.py`` /
# ``_registry_part2.py`` -- cross-module inference of an unannotated multi-target tuple
# assignment is a known mypy limitation.
_cbrt_fit: Callable
_cbrt_forward: Callable
_cbrt_inverse: Callable
_cbrt_domain: Callable
_cbrt_domain_fitted: Callable | None
_log_fit_a: Callable
_log_forward_a: Callable
_log_inverse_a: Callable
_log_domain_a: Callable
_log_domain_fitted_a: Callable | None
_yj_fit_a: Callable
_yj_forward_a: Callable
_yj_inverse_a: Callable
_yj_domain_a: Callable
_yj_domain_fitted_a: Callable | None
_qn_fit_a: Callable
_qn_forward_a: Callable
_qn_inverse_a: Callable
_qn_domain_a: Callable
_qn_domain_fitted_a: Callable | None
_sp_fit_a: Callable
_sp_forward_a: Callable
_sp_inverse_a: Callable
_sp_domain_a: Callable
_sp_domain_fitted_a: Callable | None
_bc_fit_a: Callable
_bc_forward_a: Callable
_bc_inverse_a: Callable
_bc_domain_a: Callable
_bc_domain_fitted_a: Callable | None

_cbrt_fit, _cbrt_forward, _cbrt_inverse, _cbrt_domain, _cbrt_domain_fitted = _make_unary_registry_adapter(
    _cbrt_y_fit_raw, _cbrt_y_forward_raw, _cbrt_y_inverse_raw, _cbrt_y_domain_raw,
)
_log_fit_a, _log_forward_a, _log_inverse_a, _log_domain_a, _log_domain_fitted_a = _make_unary_registry_adapter(
    _log_y_fit_raw, _log_y_forward_raw, _log_y_inverse_raw,
    # log_y_domain is the 2-arg form (y, params); wrap to drop params at fit-time.
    lambda y: _log_y_domain_raw(y),
    # The pre-fit domain only checks isfinite(y); the TRUE log_y domain is
    # ``y + offset > 0``, knowable only after ``fit`` sets offset. Without
    # this hook, screening forwards log() over ``y <= -offset`` rows (silent
    # NaN T -> biased MI gain) and the wrapper later hard-raises
    # DomainViolationError on the same rows. Pass the params-aware raw form.
    domain_fitted_fn=_log_y_domain_raw,
)
_yj_fit_a, _yj_forward_a, _yj_inverse_a, _yj_domain_a, _yj_domain_fitted_a = _make_unary_registry_adapter(
    _yj_y_fit_raw, _yj_y_forward_raw, _yj_y_inverse_raw,
    lambda y: _yj_y_domain_raw(y),
)
_qn_fit_a, _qn_forward_a, _qn_inverse_a, _qn_domain_a, _qn_domain_fitted_a = _make_unary_registry_adapter(
    _qn_y_fit_raw, _qn_y_forward_raw, _qn_y_inverse_raw,
    lambda y: _qn_y_domain_raw(y),
)
_sp_fit_a, _sp_forward_a, _sp_inverse_a, _sp_domain_a, _sp_domain_fitted_a = _make_unary_registry_adapter(
    _sp_y_fit_raw, _sp_y_forward_raw, _sp_y_inverse_raw,
    lambda y: _sp_y_domain_raw(y),
)
_bc_fit_a, _bc_forward_a, _bc_inverse_a, _bc_domain_a, _bc_domain_fitted_a = _make_unary_registry_adapter(
    _bc_y_fit_raw, _bc_y_forward_raw, _bc_y_inverse_raw,
    lambda y: _bc_y_domain_raw(y),
)


def _centered_ratio_domain_fitted(y, base, params):
    """Fitted-domain for ``centered_ratio`` (T = y / (base + c)).

    The pre-fit ``_centered_ratio_domain`` only gates on finite y / base; the
    real per-row validity depends on the learned shift ``c`` and eps-floor:
    a row whose ``base + c`` lands inside the near-zero ``[-eps, eps]`` band
    has its denominator clamped to ``+/- eps`` in ``forward``/``inverse``, so
    T no longer reflects the true ratio and the round-trip is only approximate
    on that row. Those rows are excluded from screening + fit so the divisor
    clamp never silently distorts the MI estimate / fitted scale. Mirrors the
    ``domain_check`` ``y=None`` predict-time contract: with ``y`` unknown we
    still gate the base-side ``|base + c| >= eps`` condition (knowable from
    params), so the same rows are flagged at predict time.
    """
    base_arr = np.asarray(base, dtype=np.float64)
    if params is None:
        # No fitted params yet -> fall back to the params-free domain.
        return _centered_ratio_domain(y, base)
    c = float(params.get("c", 0.0))
    eps = float(params.get("eps", 0.0))
    shifted = base_arr + c
    base_ok = np.isfinite(base_arr) & (np.abs(shifted) >= eps)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64))


__all__ = [
    "TAG_CORE",
    "TAG_EXTENDED",
    "TAG_REGRESSION",
    "Transform",
    "_additive_residual_domain",
    "_additive_residual_fit",
    "_additive_residual_forward",
    "_additive_residual_inverse",
    "_asinh_residual_domain",
    "_asinh_residual_fit",
    "_asinh_residual_forward",
    "_asinh_residual_inverse",
    "_asinh_residual_multi_domain",
    "_asinh_residual_multi_fit",
    "_asinh_residual_multi_forward",
    "_asinh_residual_multi_inverse",
    "_bc_domain_a",
    "_bc_domain_fitted_a",
    "_bc_fit_a",
    "_bc_forward_a",
    "_bc_inverse_a",
    "_bc_y_domain_raw",
    "_bc_y_fit_raw",
    "_bc_y_forward_raw",
    "_bc_y_inverse_raw",
    "_causal_anchor_residual_domain",
    "_causal_anchor_residual_fit",
    "_causal_anchor_residual_forward",
    "_causal_anchor_residual_inverse",
    "_cbrt_domain",
    "_cbrt_domain_fitted",
    "_cbrt_fit",
    "_cbrt_forward",
    "_cbrt_inverse",
    "_cbrt_y_domain_raw",
    "_cbrt_y_fit_raw",
    "_cbrt_y_forward_raw",
    "_cbrt_y_inverse_raw",
    "_centered_ratio_domain",
    "_centered_ratio_domain_fitted",
    "_centered_ratio_fit",
    "_centered_ratio_forward",
    "_centered_ratio_inverse",
    "_diff_domain",
    "_diff_fit",
    "_diff_forward",
    "_diff_inverse",
    "_ewma_residual_domain",
    "_ewma_residual_fit",
    "_ewma_residual_forward",
    "_ewma_residual_grouped_domain",
    "_ewma_residual_grouped_fit",
    "_ewma_residual_grouped_forward",
    "_ewma_residual_grouped_inverse",
    "_ewma_residual_inverse",
    "_frac_diff_domain",
    "_frac_diff_fit",
    "_frac_diff_forward",
    "_frac_diff_grouped_domain",
    "_frac_diff_grouped_fit",
    "_frac_diff_grouped_forward",
    "_frac_diff_grouped_inverse",
    "_frac_diff_inverse",
    "_gaussian_copula_residual_domain",
    "_gaussian_copula_residual_fit",
    "_gaussian_copula_residual_forward",
    "_gaussian_copula_residual_inverse",
    "_geometric_mean_residual_domain",
    "_geometric_mean_residual_fit",
    "_geometric_mean_residual_forward",
    "_geometric_mean_residual_inverse",
    "_linear_residual_domain",
    "_linear_residual_fit",
    "_linear_residual_forward",
    "_linear_residual_grouped_domain",
    "_linear_residual_grouped_fit",
    "_linear_residual_grouped_forward",
    "_linear_residual_grouped_inverse",
    "_linear_residual_inverse",
    "_linear_residual_multi_domain",
    "_linear_residual_multi_fit",
    "_linear_residual_multi_forward",
    "_linear_residual_multi_inverse",
    "_linear_residual_multi_robust_fit",
    "_linear_residual_robust_fit",
    "_log_domain_a",
    "_log_domain_fitted_a",
    "_log_fit_a",
    "_log_forward_a",
    "_log_inverse_a",
    "_log_y_domain_raw",
    "_log_y_fit_raw",
    "_log_y_forward_raw",
    "_log_y_inverse_raw",
    "_logratio_domain",
    "_logratio_fit",
    "_logratio_forward",
    "_logratio_inverse",
    "_make_chain_transform",
    "_make_multi_chain_transform",
    "_make_unary_registry_adapter",
    "_median_residual_domain",
    "_median_residual_fit",
    "_median_residual_forward",
    "_median_residual_inverse",
    "_monotonic_residual_domain",
    "_monotonic_residual_fit",
    "_monotonic_residual_forward",
    "_monotonic_residual_grouped_domain",
    "_monotonic_residual_grouped_fit",
    "_monotonic_residual_grouped_forward",
    "_monotonic_residual_grouped_inverse",
    "_monotonic_residual_inverse",
    "_nadaraya_watson_residual_domain",
    "_nadaraya_watson_residual_fit",
    "_nadaraya_watson_residual_forward",
    "_nadaraya_watson_residual_inverse",
    "_pairwise_interaction_residual_domain",
    "_pairwise_interaction_residual_fit",
    "_pairwise_interaction_residual_forward",
    "_pairwise_interaction_residual_inverse",
    "_polynomial_residual_deg2_domain",
    "_polynomial_residual_deg2_fit",
    "_polynomial_residual_deg2_forward",
    "_polynomial_residual_deg2_inverse",
    "_qn_domain_a",
    "_qn_domain_fitted_a",
    "_qn_fit_a",
    "_qn_forward_a",
    "_qn_inverse_a",
    "_qn_y_domain_raw",
    "_qn_y_fit_raw",
    "_qn_y_forward_raw",
    "_qn_y_inverse_raw",
    "_quantile_residual_domain",
    "_quantile_residual_fit",
    "_quantile_residual_forward",
    "_quantile_residual_grouped_domain",
    "_quantile_residual_grouped_fit",
    "_quantile_residual_grouped_forward",
    "_quantile_residual_grouped_inverse",
    "_quantile_residual_inverse",
    "_rank_ecdf_residual_domain",
    "_rank_ecdf_residual_fit",
    "_rank_ecdf_residual_forward",
    "_rank_ecdf_residual_inverse",
    "_rank_residual_domain",
    "_rank_residual_fit",
    "_rank_residual_forward",
    "_rank_residual_inverse",
    "_ratio_domain",
    "_ratio_fit",
    "_ratio_forward",
    "_ratio_inverse",
    "_reciprocal_residual_domain",
    "_reciprocal_residual_fit",
    "_reciprocal_residual_forward",
    "_reciprocal_residual_inverse",
    "_rolling_quantile_ratio_centered_fit",
    "_rolling_quantile_ratio_domain",
    "_rolling_quantile_ratio_fit",
    "_rolling_quantile_ratio_forward",
    "_rolling_quantile_ratio_grouped_domain",
    "_rolling_quantile_ratio_grouped_fit",
    "_rolling_quantile_ratio_grouped_forward",
    "_rolling_quantile_ratio_grouped_inverse",
    "_rolling_quantile_ratio_inverse",
    "_seasonal_residual_domain",
    "_seasonal_residual_fit",
    "_seasonal_residual_forward",
    "_seasonal_residual_inverse",
    "_second_diff_domain",
    "_second_diff_fit",
    "_second_diff_forward",
    "_second_diff_inverse",
    "_smoothing_spline_residual_domain",
    "_smoothing_spline_residual_fit",
    "_smoothing_spline_residual_forward",
    "_smoothing_spline_residual_inverse",
    "_sp_domain_a",
    "_sp_domain_fitted_a",
    "_sp_fit_a",
    "_sp_forward_a",
    "_sp_inverse_a",
    "_sp_y_domain_raw",
    "_sp_y_fit_raw",
    "_sp_y_forward_raw",
    "_sp_y_inverse_raw",
    "_target_encoding_residual_domain",
    "_target_encoding_residual_fit",
    "_target_encoding_residual_forward",
    "_target_encoding_residual_inverse",
    "_theilsen_residual_fit",
    "_volatility_normalized_residual_domain",
    "_volatility_normalized_residual_fit",
    "_volatility_normalized_residual_forward",
    "_volatility_normalized_residual_inverse",
    "_y_quantile_clip_domain",
    "_y_quantile_clip_fit",
    "_y_quantile_clip_forward",
    "_y_quantile_clip_inverse",
    "_yj_domain_a",
    "_yj_domain_fitted_a",
    "_yj_fit_a",
    "_yj_forward_a",
    "_yj_inverse_a",
    "_yj_y_domain_raw",
    "_yj_y_fit_raw",
    "_yj_y_forward_raw",
    "_yj_y_inverse_raw",
]
