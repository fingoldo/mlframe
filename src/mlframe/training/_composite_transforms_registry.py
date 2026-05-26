"""Transforms registry carved out of
``mlframe.training.composite_transforms``.

Builds the ``_TRANSFORMS_REGISTRY`` dict mapping transform name -> :class:`Transform`. The four functional clusters (simple / linear / nonlinear / unary) define the underlying fit/forward/inverse/domain functions; this module wires them into the public registry.

Imported by the parent AFTER all four functional siblings load (init-order matters: the registry literal references every per-transform function by binding-resolution at module-import time).

Bound back into the parent's namespace via re-export at the parent's
module bottom so historical
``from mlframe.training.composite_transforms import _TRANSFORMS_REGISTRY`` resolves.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Parent's ``Transform`` dataclass + TAG_* sentinel constants live in
# ``composite_transforms.py``, which imports this sibling at its bottom
# AFTER defining Transform + the TAG constants. The static cycle is
# whitelisted in ``tests/test_meta/test_no_import_cycles.py`` (same
# pattern as ``_composite_transforms_linear`` / ``_nonlinear``).
from .composite_transforms import (
    TAG_CORE,
    TAG_EXTENDED,
    TAG_REGRESSION,
    Transform,
)
from ._composite_transforms_simple import (
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
    _rolling_quantile_ratio_forward,
    _rolling_quantile_ratio_inverse,
    _y_quantile_clip_domain,
    _y_quantile_clip_fit,
    _y_quantile_clip_forward,
    _y_quantile_clip_inverse,
)
from ._composite_transforms_linear import (
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
    _logratio_domain,
    _logratio_fit,
    _logratio_forward,
    _logratio_inverse,
)
from ._composite_transforms_nonlinear import (
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
from .composite_unary_transforms import (
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
    yeo_johnson_y_domain as _yj_y_domain_raw,
    yeo_johnson_y_fit as _yj_y_fit_raw,
    yeo_johnson_y_forward as _yj_y_forward_raw,
    yeo_johnson_y_inverse as _yj_y_inverse_raw,
)


def _make_unary_registry_adapter(
    fit_fn, forward_fn, inverse_fn, domain_fn,
):
    """Adapt a unary (y, params) signature to the registry's (y, base, params) signature by ignoring ``base``. Returns (fit_adapter, forward_adapter, inverse_adapter, domain_adapter)."""

    def _fit(y, base):  # noqa: ARG001
        return fit_fn(y)

    def _forward(y, base, params):  # noqa: ARG001
        return forward_fn(y, params)

    def _inverse(t_hat, base, params):  # noqa: ARG001
        return inverse_fn(t_hat, params)

    def _domain(y, base):  # noqa: ARG001
        # The unary helper accepts (y) or (y, params); the registry
        # contract is domain_check(y, base) at fit-time and (None, base)
        # at predict-time. Predict-side call passes y=None so we cannot
        # apply the unary domain on y -- gate on finite base / always-True
        # for unary which has no base constraint at predict.
        if y is None:
            return np.ones(len(base) if hasattr(base, "__len__") else 1, dtype=bool)
        return domain_fn(y)

    return _fit, _forward, _inverse, _domain


# Pre-build per-unary adapters (cheap, done once at import).
_cbrt_fit, _cbrt_forward, _cbrt_inverse, _cbrt_domain = _make_unary_registry_adapter(
    _cbrt_y_fit_raw, _cbrt_y_forward_raw, _cbrt_y_inverse_raw, _cbrt_y_domain_raw,
)
_log_fit_a, _log_forward_a, _log_inverse_a, _log_domain_a = _make_unary_registry_adapter(
    _log_y_fit_raw, _log_y_forward_raw, _log_y_inverse_raw,
    # log_y_domain is the 2-arg form (y, params); wrap to drop params at fit-time.
    lambda y: _log_y_domain_raw(y),
)
_yj_fit_a, _yj_forward_a, _yj_inverse_a, _yj_domain_a = _make_unary_registry_adapter(
    _yj_y_fit_raw, _yj_y_forward_raw, _yj_y_inverse_raw,
    lambda y: _yj_y_domain_raw(y),
)
_qn_fit_a, _qn_forward_a, _qn_inverse_a, _qn_domain_a = _make_unary_registry_adapter(
    _qn_y_fit_raw, _qn_y_forward_raw, _qn_y_inverse_raw,
    lambda y: _qn_y_domain_raw(y),
)


_TRANSFORMS_REGISTRY: dict[str, Transform] = {
    "diff": Transform(
        name="diff",
        forward=_diff_forward,
        inverse=_diff_inverse,
        fit=_diff_fit,
        domain_check=_diff_domain,
        description="T = y - base. Inverse y_hat = T_hat + base. No fitted parameters.",
        tags=frozenset({TAG_CORE, TAG_REGRESSION}),
    ),
    "additive_residual": Transform(
        name="additive_residual",
        forward=_additive_residual_forward,
        inverse=_additive_residual_inverse,
        fit=_additive_residual_fit,
        domain_check=_additive_residual_domain,
        description=(
            "T = y - base - beta (alpha=1.0 fixed, beta=mean(y_train - base_train) learned). "
            "Inverse y_hat = T_hat + base + beta. Strict-AR-1 sweet spot between ``diff`` "
            "(no offset) and ``linear_residual`` (alpha+beta both learned). Pure additive "
            "inverse keeps the composite MLP-friendly: no nonlinear inverse to learn."
        ),
        tags=frozenset({TAG_CORE, TAG_REGRESSION}),
    ),
    "median_residual": Transform(
        name="median_residual",
        forward=_median_residual_forward,
        inverse=_median_residual_inverse,
        fit=_median_residual_fit,
        domain_check=_median_residual_domain,
        description=(
            "T = y - median(y | bin(base)) using 20 quantile-bins of base. "
            "Inverse y_hat = T_hat + median_bin[base]. Non-parametric residual "
            "with PURE additive inverse (constant-per-bin lookup) -- distinct "
            "from monotonic_residual (PCHIP nonlinear inverse) and "
            "quantile_residual (divides by IQR, also nonlinear). MLP-friendly: "
            "any T_hat extrapolation maps back to y via simple addition."
        ),
        tags=frozenset({TAG_CORE, TAG_REGRESSION}),
    ),
    "y_quantile_clip": Transform(
        name="y_quantile_clip",
        forward=_y_quantile_clip_forward,
        inverse=_y_quantile_clip_inverse,
        fit=_y_quantile_clip_fit,
        domain_check=_y_quantile_clip_domain,
        description=(
            "T = clip(y, q_0.005, q_0.995) -- unary y-only limit-damage "
            "transform. Bounds downstream model's effective target range "
            "to [q_lo, q_hi] of train y; predictions stay bounded by the "
            "same clip on inverse. Useful for neural / linear downstream "
            "models that might extrapolate wildly outside train range."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        requires_base=False,
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
    "linear_residual_robust": Transform(
        name="linear_residual_robust",
        forward=_linear_residual_forward,
        inverse=_linear_residual_inverse,
        fit=_linear_residual_robust_fit,
        domain_check=_linear_residual_domain,
        description=(
            "Outlier-robust variant of linear_residual via trimmed-LS: OLS first "
            "pass -> drop rows where |resid| > 3 * sigma_MAD -> refit OLS on the "
            "inlier set. Forward / inverse identical to linear_residual once "
            "(alpha, beta) are fitted. Bench (1M rows, 5% Cauchy outliers): "
            "0.12s, alpha err 0.01%, beta err 2.40% -- vs plain OLS 95% beta err "
            "and Huber/RANSAC/LAD 30-80x slower for similar accuracy."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "linear_residual_multi": Transform(
        name="linear_residual_multi",
        forward=_linear_residual_multi_forward,
        inverse=_linear_residual_multi_inverse,
        fit=_linear_residual_multi_fit,
        domain_check=_linear_residual_multi_domain,
        description=(
            "T = y - sum_j(alpha_j * base_j) - beta with joint-OLS (alphas, beta) "
            "over a K-column base matrix. Inverse y_hat = T_hat + base @ alphas + beta. "
            "Falls back to zero-alpha + train mean intercept when the design-matrix "
            "condition number exceeds _MULTI_BASE_COND_NUMBER_MAX (multicollinearity guard)."
        ),
        tags=frozenset({TAG_CORE, TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "linear_residual_grouped": Transform(
        name="linear_residual_grouped",
        forward=_linear_residual_grouped_forward,
        inverse=_linear_residual_grouped_inverse,
        fit=_linear_residual_grouped_fit,
        domain_check=_linear_residual_grouped_domain,
        description=(
            "T = y - alpha_g * base - beta_g where g = group(row). Per-group OLS "
            "with James-Stein shrinkage toward global (alpha, beta). Small groups "
            "(n < _GROUPED_MIN_GROUP_SIZE) and unseen groups at predict time "
            "fall back to global. Requires a 'groups' kwarg threaded through "
            "fit/forward/inverse (wrapper extracts from configured group_column)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        requires_groups=True,
    ),
    "quantile_residual": Transform(
        name="quantile_residual",
        forward=_quantile_residual_forward,
        inverse=_quantile_residual_inverse,
        fit=_quantile_residual_fit,
        domain_check=_quantile_residual_domain,
        description=(
            "Non-parametric heteroscedasticity-aware residual: T = (y - median_bin(y)) / IQR_bin(y) with ``n_bins`` quantile bins of ``base``. Inverse y_hat = T_hat * IQR_bin + median_bin. Under-populated bins (< min_bin_n train rows) and constant-y bins fall back to the global median(y) / IQR(y); out-of-range base values at predict map to the edge bin (no separate OOR bucket)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "monotonic_residual": Transform(
        name="monotonic_residual",
        forward=_monotonic_residual_forward,
        inverse=_monotonic_residual_inverse,
        fit=_monotonic_residual_fit,
        domain_check=_monotonic_residual_domain,
        description=(
            "T = y - g(base) where g is a monotone PCHIP spline fitted on quantile-knot medians. Generalises linear_residual to capture saturating / sigmoidal monotonic relationships that an OLS line leaves in the residual; PCHIP is monotone-preserving and the per-knot y-values are forced monotone (cumulative max/min along the Spearman-correlation orientation) so the interpolant is globally monotone. Out-of-range base values at predict clip to edge knot values (no PCHIP extrapolation)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "ewma_residual": Transform(
        name="ewma_residual",
        forward=_ewma_residual_forward,
        inverse=_ewma_residual_inverse,
        fit=_ewma_residual_fit,
        domain_check=_ewma_residual_domain,
        description=(
            "Time-ordered exponentially-weighted moving-average residual: T = y - EWMA_k(base) with alpha = 2/(k+1). Captures slow drift / regime persistence beyond a single lag. Caller is responsible for chronological row order at fit and predict; non-finite base values carry the previous EWMA state forward."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "rolling_quantile_ratio": Transform(
        name="rolling_quantile_ratio",
        forward=_rolling_quantile_ratio_forward,
        inverse=_rolling_quantile_ratio_inverse,
        fit=_rolling_quantile_ratio_fit,
        domain_check=_rolling_quantile_ratio_domain,
        description=(
            "Localised multiplicative residual: T = y / RollingMedian_k(base), with a centred window of ``k`` rows and an eps floor derived from train base scale to keep division safe at near-zero rolling medians. Inverse: y_hat = T_hat * RollingMedian_k(base). Like logratio but tracks the LOCAL base level instead of the global scale -- useful when y scales with a windowed median of base rather than the instantaneous value."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "frac_diff": Transform(
        name="frac_diff",
        forward=_frac_diff_forward,
        inverse=_frac_diff_inverse,
        fit=_frac_diff_fit,
        domain_check=_frac_diff_domain,
        description=(
            "Lopez de Prado fractional differencing: T_i = sum_k w_k * y_{i-k} with w_k = -w_{k-1} * (d - k + 1) / k truncated at ``lags`` terms. Preserves long-memory while making the target stationary. Inverse iteratively reconstructs y from T + the previously-reconstructed past terms. Pre-window padding uses the train-y mean."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    # ------------------------------------------------------------------
    # Pack J: unary y-only transforms. ``requires_base=False`` tells the
    # wrapper to skip base-column extraction. Composite-target name has
    # no ``base`` segment (e.g. ``y-cbrtY``) since there is no base.
    # ------------------------------------------------------------------
    "cbrt_y": Transform(
        name="cbrt_y",
        forward=_cbrt_forward,
        inverse=_cbrt_inverse,
        fit=_cbrt_fit,
        domain_check=_cbrt_domain,
        description=(
            "Signed cube-root unary y-transform: T = sign(y) * |y|^(1/3). "
            "Inverse y = T^3. Defined for all real y, no fitted parameters. "
            "Compresses heavy tails without breaking sign -- particularly "
            "useful when an upstream bivariate composite has absorbed the "
            "dominant feature but the residual is still Laplace-leptokurtic."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        requires_base=False,
    ),
    "log_y": Transform(
        name="log_y",
        forward=_log_forward_a,
        inverse=_log_inverse_a,
        fit=_log_fit_a,
        domain_check=_log_domain_a,
        description=(
            "Shifted log unary y-transform: T = log(y + offset) where offset is fitted so "
            "min(y_train) + offset > 0. Inverse y = exp(T) - offset. Compresses right-skewed "
            "targets (typical for non-negative regression targets like duration / count / cost)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        requires_base=False,
    ),
    "yeo_johnson_y": Transform(
        name="yeo_johnson_y",
        forward=_yj_forward_a,
        inverse=_yj_inverse_a,
        fit=_yj_fit_a,
        domain_check=_yj_domain_a,
        description=(
            "Yeo-Johnson power transform with lambda fitted by MLE (scipy Brent, range "
            "clipped to [-2, 4]). Works on mixed-sign y unlike Box-Cox. Inverse is the "
            "closed-form YJ inverse with the same lambda."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        requires_base=False,
    ),
    "quantile_normal_y": Transform(
        name="quantile_normal_y",
        forward=_qn_forward_a,
        inverse=_qn_inverse_a,
        fit=_qn_fit_a,
        domain_check=_qn_domain_a,
        description=(
            "Empirical-CDF -> standard Normal: T = Phi^-1(rank(y) / (n + 1)) via knot "
            "interpolation. Inverse interpolates the fitted CDF. Robust to any monotone "
            "distortion of y but loses absolute scale -- use when the noise-distribution "
            "hypothesis is itself uncertain."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        requires_base=False,
    ),
    # ------------------------------------------------------------------
    # Pack K: chain transforms. Bivariate residual + unary tail
    # compression, composed by the chain factory above.
    # ------------------------------------------------------------------
    "chain_linres_cbrt": _make_chain_transform(
        name="chain_linres_cbrt", short_name="linres+cbrt",
        bivariate_name="linear_residual",
        bivariate_fit=_linear_residual_fit,
        bivariate_forward=_linear_residual_forward,
        bivariate_inverse=_linear_residual_inverse,
        bivariate_domain=_linear_residual_domain,
        unary_fit=_cbrt_y_fit_raw,
        unary_forward=_cbrt_y_forward_raw,
        unary_inverse=_cbrt_y_inverse_raw,
        description=(
            "Chain: T1 = y - alpha*base - beta (linear_residual, OLS-fitted alpha+beta), then "
            "T2 = sign(T1) * |T1|^(1/3) (signed cube root). Inverse runs cbrt^-1 then "
            "y = T1 + alpha*base + beta. Targets heavy-tailed residual on top of a "
            "single-base linear regression -- a real heavy-residual case "
            "with excess_kurt=+2.40."
        ),
    ),
    "chain_linres_yj": _make_chain_transform(
        name="chain_linres_yj", short_name="linres+yj",
        bivariate_name="linear_residual",
        bivariate_fit=_linear_residual_fit,
        bivariate_forward=_linear_residual_forward,
        bivariate_inverse=_linear_residual_inverse,
        bivariate_domain=_linear_residual_domain,
        unary_fit=_yj_y_fit_raw,
        unary_forward=_yj_y_forward_raw,
        unary_inverse=_yj_y_inverse_raw,
        description=(
            "Chain: linear_residual + Yeo-Johnson(lambda MLE). YJ adapts to the actual "
            "residual skew + tail shape so the inner boosting sees a near-Gaussian target."
        ),
    ),
    "chain_monres_cbrt": _make_chain_transform(
        name="chain_monres_cbrt", short_name="monres+cbrt",
        bivariate_name="monotonic_residual",
        bivariate_fit=_monotonic_residual_fit,
        bivariate_forward=_monotonic_residual_forward,
        bivariate_inverse=_monotonic_residual_inverse,
        bivariate_domain=_monotonic_residual_domain,
        unary_fit=_cbrt_y_fit_raw,
        unary_forward=_cbrt_y_forward_raw,
        unary_inverse=_cbrt_y_inverse_raw,
        description=(
            "Chain: monotonic_residual (PCHIP-fitted g(base)) + signed cube root. "
            "Combines a nonlinear-monotone base absorber with tail compression."
        ),
    ),
    "chain_monres_yj": _make_chain_transform(
        name="chain_monres_yj", short_name="monres+yj",
        bivariate_name="monotonic_residual",
        bivariate_fit=_monotonic_residual_fit,
        bivariate_forward=_monotonic_residual_forward,
        bivariate_inverse=_monotonic_residual_inverse,
        bivariate_domain=_monotonic_residual_domain,
        unary_fit=_yj_y_fit_raw,
        unary_forward=_yj_y_forward_raw,
        unary_inverse=_yj_y_inverse_raw,
        description=(
            "Chain: monotonic_residual + Yeo-Johnson power transform."
        ),
    ),
    # Pack K extension: 3-stage chain. For VERY heavy-tail residuals
    # where a single unary still leaves leptokurtosis, follow up with
    # quantile-normalisation to map any remaining structure to a
    # standard Normal. Lossy on absolute scale (quantile_normal forgets
    # the original units) but RMSE on the doubly-compressed T is
    # cleaner for boosting inners.
    "chain_linres_cbrt_qn": _make_multi_chain_transform(
        name="chain_linres_cbrt_qn", short_name="linresCbrtQn",
        bivariate_fit=_linear_residual_fit,
        bivariate_forward=_linear_residual_forward,
        bivariate_inverse=_linear_residual_inverse,
        bivariate_domain=_linear_residual_domain,
        unary_stages=[
            (_cbrt_y_fit_raw, _cbrt_y_forward_raw, _cbrt_y_inverse_raw),
            (_qn_y_fit_raw, _qn_y_forward_raw, _qn_y_inverse_raw),
        ],
        description=(
            "3-stage chain: T1 = y - alpha*base - beta (linear_residual); "
            "T2 = sign(T1) * |T1|^(1/3) (signed cbrt); "
            "T3 = Phi^-1(rank(T2) / (n+1)) (quantile-normal). "
            "For VERY heavy-tail residuals where a single unary still leaves "
            "leptokurtosis. Lossy on absolute scale; RMSE on T3 cleaner for "
            "boosting inners."
        ),
    ),
}
