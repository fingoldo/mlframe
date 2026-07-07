"""Transforms registry carved out of
``mlframe.training.composite_transforms``.

Builds the ``_TRANSFORMS_REGISTRY`` dict mapping transform name -> :class:`Transform`. The four functional clusters (simple / linear / nonlinear / unary) define the underlying fit/forward/inverse/domain functions; this module wires them into the public registry.

Imported by the parent AFTER all four functional siblings load (init-order matters: the registry literal references every per-transform function by binding-resolution at module-import time).

Bound back into the parent's namespace via re-export at the parent's
module bottom so historical
``from mlframe.training.composite_transforms import _TRANSFORMS_REGISTRY`` resolves.
"""
from __future__ import annotations


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
)


def _make_unary_registry_adapter(
    fit_fn, forward_fn, inverse_fn, domain_fn, domain_fitted_fn=None,
):
    """Adapt a unary (y, params) signature to the registry's (y, base, params) signature by ignoring ``base``. Returns (fit_adapter, forward_adapter, inverse_adapter, domain_adapter[, domain_fitted_adapter]).

    ``domain_fitted_fn`` (optional, signature ``(y, params) -> mask``) wires
    the fitted-params-aware domain hook for unary transforms whose validity
    depends on a learned parameter (e.g. ``log_y``'s ``offset``: rows with
    ``y + offset <= 0`` are out of domain only once ``offset`` is known). When
    ``None`` the returned 5th element is ``None`` and the registry entry leaves
    ``domain_check_fitted`` unset (params-free ``domain_check`` is exact)."""

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

    if domain_fitted_fn is None:
        return _fit, _forward, _inverse, _domain, None

    def _domain_fitted(y, base, params):  # noqa: ARG001
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
            "Requires y, base > 0. The soft-cap (|T-median| > 10*MAD) trades exact "
            "round-trip on extreme in-domain train rows for bounded inverse blow-up."
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
    "theilsen_residual": Transform(
        name="theilsen_residual",
        forward=_linear_residual_forward,
        inverse=_linear_residual_inverse,
        fit=_theilsen_residual_fit,
        domain_check=_linear_residual_domain,
        description=(
            "High-breakdown robust variant of linear_residual via Theil-Sen: "
            "alpha = median over point-pairs of (y_j - y_i)/(base_j - base_i), "
            "beta = median(y - alpha*base). Tolerates ~29% gross outliers in "
            "EITHER y or base without the OLS-seed fragility of "
            "linear_residual_robust (whose trimming starts from an OLS pass a "
            "clustered outlier mass can already drag). Forward / inverse "
            "identical to linear_residual once (alpha, beta) are fitted. "
            "Theil-Sen is O(n^2) in pairs; for large n the pairwise-slope "
            "computation is subsampled to a fixed-seed random pair set capped "
            "at _THEILSEN_MAX_PAIRS, keeping the breakdown robustness."
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
        # EWMA carries state forward across the row sequence; the fit-time
        # domain filter must run the forward over the full sequence then mask,
        # so train T near a filtered gap matches predict-time T (see Transform.recurrent).
        recurrent=True,
    ),
    "rolling_quantile_ratio": Transform(
        name="rolling_quantile_ratio",
        forward=_rolling_quantile_ratio_forward,
        inverse=_rolling_quantile_ratio_inverse,
        fit=_rolling_quantile_ratio_fit,
        domain_check=_rolling_quantile_ratio_domain,
        description=(
            "Localised multiplicative residual: T = y / RollingMedian_k(base), with a centred window of ``k`` rows and an eps floor derived from train base scale to keep division safe at near-zero rolling medians. Inverse: y_hat = T_hat * RollingMedian_k(base). Like logratio but tracks the LOCAL base level instead of the global scale -- useful when y scales with a windowed median of base rather than the instantaneous value. LOOK-AHEAD: the centred window reads FUTURE base rows, so in time-ordered deployment T leaks forward; gated out of default discovery for this reason (a trailing-only mode is not yet implemented)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        # Centred rolling median reads neighbouring rows; compacting the
        # sequence before the forward would shrink each window across a
        # filtered gap, so the forward runs full-then-mask (see Transform.recurrent).
        recurrent=True,
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
        # y-only transform: forward/inverse never read base, so a single spec must be emitted (not one per base) and base-finiteness must not drop y rows.
        requires_base=False,
        # Fractional-difference weights convolve each row with its lagged
        # predecessors; a compacted gap would re-align the weight tail onto the
        # wrong rows, so the forward runs full-then-mask (see Transform.recurrent).
        recurrent=True,
    ),
    # Unary y-only transforms. ``requires_base=False`` tells the wrapper to skip base-column extraction, and the base is unused by forward/inverse here.
    # Discovery scores these against the full feature matrix via a dedicated sentinel context and emits a BASE-FREE 2-segment composite name (``y-cbrtY``),
    # with an empty ``base_column`` -- no spurious base segment, and the spec's mi_gain no longer depends on auto-base ranking order.
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
    "signed_power_y": Transform(
        name="signed_power_y",
        forward=_sp_forward_a,
        inverse=_sp_inverse_a,
        fit=_sp_fit_a,
        domain_check=_sp_domain_a,
        description=(
            "Tweedie-style signed power unary y-transform: T = sign(y) * |y|^p "
            "with p fitted at fit-time to minimise |skew(T)| over a 1-D grid in "
            "[0.1, 0.9]. Inverse y = sign(T) * |T|^(1/p). Defined for all real y. "
            "Generalises ``cbrt_y`` (fixed p=1/3): the fitted exponent adapts the "
            "tail-compression strength to the target's actual skew, mapping a "
            "strongly right-skewed y (lognormal duration / cost / count) to a "
            "near-symmetric T that RMSE-trained downstream models fit cleanly."
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
        # Fitted-domain gates ``y + offset > 0`` once offset is learned, so screening/fit never forward log() over a NaN-producing row.
        domain_check_fitted=_log_domain_fitted_a,
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
    # Chain transforms: bivariate residual + unary tail compression, composed by the chain factory above.
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
    # 3-stage chain. For VERY heavy-tail residuals where a single unary still leaves leptokurtosis, follow up with quantile-normalisation to map
    # any remaining structure to a standard Normal. Lossy on absolute scale (quantile_normal forgets the original units) but RMSE on the
    # doubly-compressed T is cleaner for boosting inners.
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
    # Extended bivariate + multi-base transforms. Plug specific failure modes observed in production where the existing core/extended set didn't reach:
    # signed bases for log-like residuals (``asinh_residual``), expanded ratio domain via learned shift (``centered_ratio``), curvature beyond OLS line
    # (``polynomial_residual_deg2``), distribution-free monotone (``rank_residual``), arbitrary smooth non-monotone (``smoothing_spline_residual``),
    # multiplicative-jump dynamics (``reciprocal_residual``), and two multi-base variants (``geometric_mean_residual``, ``pairwise_interaction_residual``).
    "asinh_residual": Transform(
        name="asinh_residual",
        forward=_asinh_residual_forward,
        inverse=_asinh_residual_inverse,
        fit=_asinh_residual_fit,
        domain_check=_asinh_residual_domain,
        description=(
            "T = arcsinh(y) - alpha*arcsinh(base) - beta with (alpha, beta) "
            "fitted via OLS on the arcsinh-transformed train pairs. "
            "Inverse y_hat = sinh(T_hat + alpha*arcsinh(base) + beta). "
            "Generalises ``logratio`` to signed bases: arcsinh is log-like "
            "for |base| >> 1 and linear for |base| << 1, defined on all real "
            "numbers."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "centered_ratio": Transform(
        name="centered_ratio",
        forward=_centered_ratio_forward,
        inverse=_centered_ratio_inverse,
        fit=_centered_ratio_fit,
        domain_check=_centered_ratio_domain,
        # Fitted-domain gates ``|base + c| >= eps`` once the shift c and eps-floor are learned, so screening/fit drop rows whose denominator
        # would be clamped to the eps-floor (T then no longer the true ratio).
        domain_check_fitted=_centered_ratio_domain_fitted,
        description=(
            "T = y / (base + c) with ``c`` fitted on train so (base + c) > 0 "
            "subject to an eps floor. Extension of ``ratio`` to signed bases. "
            "Inverse y_hat = T_hat * (base + c)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "polynomial_residual_deg2": Transform(
        name="polynomial_residual_deg2",
        forward=_polynomial_residual_deg2_forward,
        inverse=_polynomial_residual_deg2_inverse,
        fit=_polynomial_residual_deg2_fit,
        domain_check=_polynomial_residual_deg2_domain,
        description=(
            "T = y - alpha1*base - alpha2*base^2 - beta with (alpha1, alpha2, beta) "
            "fitted via ridge-stabilised OLS on the (1, base, base^2) design "
            "matrix. Adds curvature that ``linear_residual`` leaves in the "
            "residual. Inverse y_hat = T_hat + alpha1*base + alpha2*base^2 + beta."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "rank_residual": Transform(
        name="rank_residual",
        forward=_rank_residual_forward,
        inverse=_rank_residual_inverse,
        fit=_rank_residual_fit,
        domain_check=_rank_residual_domain,
        description=(
            "Distribution-free monotone residual: T = rank(y)/n - alpha*rank(base)/n - beta. "
            "Forward uses the train-fitted (sorted-y, sorted-base) lookup; "
            "inverse clips the recovered y-rank to [0, 1] and maps back via "
            "the train sorted-y table. Heavy-tail targets where Yeo-Johnson "
            "doesn't fully whiten still respond to rank-space linear residual."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "smoothing_spline_residual": Transform(
        name="smoothing_spline_residual",
        forward=_smoothing_spline_residual_forward,
        inverse=_smoothing_spline_residual_inverse,
        fit=_smoothing_spline_residual_fit,
        domain_check=_smoothing_spline_residual_domain,
        description=(
            "T = y - g(base) where g is a scipy ``UnivariateSpline`` fitted on "
            "deduped train pairs with smoothing factor s = n_unique * std(y) * 1.0. "
            "Generalises ``monotonic_residual`` to arbitrary smooth "
            "(non-monotone) dependence. Inverse y_hat = T_hat + g(base). "
            "Params store knots + ``s`` only; spline is rebuilt on call "
            "(matches ``monotonic_residual`` pickle convention)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "reciprocal_residual": Transform(
        name="reciprocal_residual",
        forward=_reciprocal_residual_forward,
        inverse=_reciprocal_residual_inverse,
        fit=_reciprocal_residual_fit,
        domain_check=_reciprocal_residual_domain,
        description=(
            "T = 1/y - 1/base with train-scale-derived eps floors guarding "
            "near-zero divisions. Inverse y_hat = 1 / (T_hat + 1/base). "
            "Niche but useful when y has multiplicative-jump dynamics or "
            "reciprocal-scale noise."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "geometric_mean_residual": Transform(
        name="geometric_mean_residual",
        forward=_geometric_mean_residual_forward,
        inverse=_geometric_mean_residual_inverse,
        fit=_geometric_mean_residual_fit,
        domain_check=_geometric_mean_residual_domain,
        description=(
            "Multi-base: T = y / geomean(bases) via log-mean-exp on a K-column "
            "base matrix. Requires every base column > 0 on the row "
            "(strict positivity). Inverse y_hat = T_hat * geomean(bases). "
            "Multiplicative multi-base variant of ``ratio``. Not in default "
            "auto-discovery list (needs multi-base orchestration like "
            "``linear_residual_multi``)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "pairwise_interaction_residual": Transform(
        name="pairwise_interaction_residual",
        forward=_pairwise_interaction_residual_forward,
        inverse=_pairwise_interaction_residual_inverse,
        fit=_pairwise_interaction_residual_fit,
        domain_check=_pairwise_interaction_residual_domain,
        description=(
            "Multi-base: T = y - alpha*prod(bases) - beta with (alpha, beta) "
            "fitted via OLS on (1, prod(bases)) train pairs. Bilinear / "
            "multilinear residual; captures pure interaction term that "
            "``linear_residual_multi`` (additive) misses. Inverse y_hat = "
            "T_hat + alpha*prod(bases) + beta. Not in default auto-discovery "
            "list (needs multi-base orchestration like "
            "``linear_residual_multi``)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "causal_anchor_residual": Transform(
        name="causal_anchor_residual",
        forward=_causal_anchor_residual_forward,
        inverse=_causal_anchor_residual_inverse,
        fit=_causal_anchor_residual_fit,
        domain_check=_causal_anchor_residual_domain,
        description=(
            "T = y - alpha*base with alpha a robustly-fitted SHRINK coefficient "
            "clamped to [0, 1] (two-pass trimmed-LS slope, scarce-data blend toward "
            "a 0.5 prior). Inverse y_hat = T_hat + alpha*base. For NOISY causal "
            "anchors (rolling / EWMA central values) where diff (implicit alpha=1) "
            "over-commits and a free linear_residual can fit a fragile large alpha "
            "that extrapolates badly on unseen groups: the [0, 1] clamp keeps the "
            "additive inverse bounded relative to the anchor (cannot over-commit "
            "past it, cannot invert its sign). Pure-additive inverse (no beta), "
            "MLP-friendly. NOT in the default transform list -- reach it via "
            "explicit transforms=(..., 'causal_anchor_residual')."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "second_diff": Transform(
        name="second_diff",
        forward=_second_diff_forward,
        inverse=_second_diff_inverse,
        fit=_second_diff_fit,
        domain_check=_second_diff_domain,
        description=(
            "T = y - 2*b1 + b2 with b1 the lag-1 anchor (base_prev) and b2 the "
            "lag-2 anchor (base_prev2), supplied as a linear_residual_multi-style "
            "(n, K) base (base_column=lag1 + extra_base_columns=[lag2]). Cancels "
            "level AND linear drift of a doubly-integrated (I(2)) series that a "
            "single diff leaves trending. Inverse y_hat = T_hat + 2*b1 - b2 is "
            "pure-additive and in-range on real per-row lags; no fitted parameters. "
            "A 1-D base degenerates to T = y - 2*b1. NOT in the default transform "
            "list -- reach it via explicit transforms=(..., 'second_diff')."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "rank_ecdf_residual": Transform(
        name="rank_ecdf_residual",
        forward=_rank_ecdf_residual_forward,
        inverse=_rank_ecdf_residual_inverse,
        fit=_rank_ecdf_residual_fit,
        domain_check=_rank_ecdf_residual_domain,
        description=(
            "Rank-space residual: T = ecdf_y(y) - ecdf_base(base) with the TRAIN "
            "empirical CDFs; inverse y_hat = quantile_y(T_hat + ecdf_base(base)) "
            "via the stored inverse-ECDF (quantile function) of y. Collapses any "
            "monotone / heavy-tailed distortion where a linear_residual line leaves "
            "structure and extrapolates on the tails; the quantile inverse cannot "
            "leave the train y-support (out-of-support base/rank clamps to edge "
            "knots). Train ECDF knots stored in fitted_params. NOT in the default "
            "transform list -- reach it via explicit transforms=(..., 'rank_ecdf_residual')."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "target_encoding_residual": Transform(
        name="target_encoding_residual",
        forward=_target_encoding_residual_forward,
        inverse=_target_encoding_residual_inverse,
        fit=_target_encoding_residual_fit,
        domain_check=_target_encoding_residual_domain,
        description=(
            "High-cardinality categorical target-encoding residual: "
            "T = y - smoothed_category_mean(cat) where the per-category mean is "
            "an empirical-Bayes / additive-smoothing estimate shrinking each "
            "category mean toward the global mean by strength ``a`` "
            "(enc[g] = (sum_y[g] + a*global_mean)/(count[g] + a)). Inverse "
            "y_hat = T_hat + smoothed_category_mean(cat). Means are fitted "
            "TRAIN-ONLY; unseen categories at predict fall back to the global "
            "mean. The smoothing strength ``a`` keeps a tiny category from "
            "overfitting its lone y; for leakage-sensitive discovery the "
            "category mean should ideally be OUT-OF-FOLD. Reuses the "
            "``group_column`` plumbing (the category column is the groups "
            "array); ``requires_base=False`` since the encoding is "
            "category-driven, not base-driven."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        requires_groups=True,
        requires_base=False,
    ),
}
