"""Second half of the transforms registry, carved out of registry.py to keep it under the 1k-LOC ceiling.

Merged into _TRANSFORMS_REGISTRY at the parent module bottom.

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
    TAG_EXTENDED,
    TAG_REGRESSION,
    Transform,
)
from .linear import (
    _linear_residual_multi_domain,
    _linear_residual_multi_forward,
    _linear_residual_multi_inverse,
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
):
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


_TRANSFORMS_REGISTRY_EXTENDED: dict[str, Transform] = {
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
    # Grouped variants of the recurrent trio: recurrence state resets at every group boundary (rows of one group need not be contiguous; each group is
    # processed in its stable original order). For stacked panels where the ungrouped recurrences bleed one entity's level into the next entity's first rows.
    "ewma_residual_grouped": Transform(
        name="ewma_residual_grouped",
        forward=_ewma_residual_grouped_forward,
        inverse=_ewma_residual_grouped_inverse,
        fit=_ewma_residual_grouped_fit,
        domain_check=_ewma_residual_grouped_domain,
        description=(
            "Per-group ewma_residual: T = y - EWMA_k(base) with the EWMA recursion reset at each group boundary and seeded by the GROUP's train-base mean (per-group tail state under recurrence continuation). Unseen groups at predict fall back to the global anchor. Caller is responsible for chronological order within each group."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        requires_groups=True,
        recurrent=True,
    ),
    "rolling_quantile_ratio_grouped": Transform(
        name="rolling_quantile_ratio_grouped",
        forward=_rolling_quantile_ratio_grouped_forward,
        inverse=_rolling_quantile_ratio_grouped_inverse,
        fit=_rolling_quantile_ratio_grouped_fit,
        domain_check=_rolling_quantile_ratio_grouped_domain,
        description=(
            "Per-group rolling_quantile_ratio: T = y / max(RollingMedian_k(base), eps) with the (trailing, past-only) window confined to each row's group, so the local level of one entity never leaks into another. eps floor is fitted on the global train base scale."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        requires_groups=True,
        recurrent=True,
    ),
    "frac_diff_grouped": Transform(
        name="frac_diff_grouped",
        forward=_frac_diff_grouped_forward,
        inverse=_frac_diff_grouped_inverse,
        fit=_frac_diff_grouped_fit,
        domain_check=_frac_diff_grouped_domain,
        description=(
            "Per-group frac_diff: the truncated (1-L)^d convolution runs independently within each group, padding each group's pre-window history with ITS OWN train-y mean (per-group tail mean under recurrence continuation) so entity-level differences never contaminate the weight tail across a boundary. y-only like frac_diff (requires_base=False)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        requires_groups=True,
        requires_base=False,
        recurrent=True,
    ),
    "quantile_residual_grouped": Transform(
        name="quantile_residual_grouped",
        forward=_quantile_residual_grouped_forward,
        inverse=_quantile_residual_grouped_inverse,
        fit=_quantile_residual_grouped_fit,
        domain_check=_quantile_residual_grouped_domain,
        description=(
            "Per-group quantile_residual: each group with >= _GROUPED_MIN_GROUP_SIZE train rows gets its own per-bin median/IQR fit, with a James-Stein-style shrinkage of the per-group level (bin medians) toward the global fit; smaller and unseen-at-predict groups fall back to the global params entirely."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        requires_groups=True,
    ),
    "monotonic_residual_grouped": Transform(
        name="monotonic_residual_grouped",
        forward=_monotonic_residual_grouped_forward,
        inverse=_monotonic_residual_grouped_inverse,
        fit=_monotonic_residual_grouped_fit,
        domain_check=_monotonic_residual_grouped_domain,
        description=(
            "Per-group monotonic_residual: each group with >= _GROUPED_MIN_GROUP_SIZE train rows gets its own monotone PCHIP g(base), with a James-Stein-style shrinkage of the per-group level (knot values) toward the global fit; smaller and unseen-at-predict groups fall back to the global spline entirely."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        requires_groups=True,
    ),
    "box_cox_y": Transform(
        name="box_cox_y",
        forward=_bc_forward_a,
        inverse=_bc_inverse_a,
        fit=_bc_fit_a,
        domain_check=_bc_domain_a,
        description=(
            "Box-Cox power transform for STRICTLY-POSITIVE targets with lambda fitted by MLE (scipy.stats.boxcox). "
            "T = (y^lambda - 1) / lambda (log(y) at lambda=0); inverse is the closed form (t*lambda + 1)^(1/lambda) via scipy's inv_boxcox algebra "
            "with an asymptote floor. Sibling of yeo_johnson_y: on y > 0 the classical Box-Cox likelihood is a slightly tighter fit; mixed-sign "
            "targets must use yeo_johnson_y instead (domain_check gates y > 0)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        requires_base=False,
    ),
    "seasonal_residual": Transform(
        name="seasonal_residual",
        forward=_seasonal_residual_forward,
        inverse=_seasonal_residual_inverse,
        fit=_seasonal_residual_fit,
        domain_check=_seasonal_residual_domain,
        description=(
            "T = y - seasonal_mean(phase) with phase = row_index % period. ``period`` may be supplied via fit kwargs or is selected on train by minimum residual variance over a small grid ({4, 5, 7, 12, 24, 52} capped at n/3). Index-position-based like ewma_residual: phase is the row's position in the batch, not a calendar field -- caller supplies chronological, gap-free rows and a predict batch starts at phase 0. Pointwise given the phase (recurrent=False)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        requires_base=False,
    ),
    "volatility_normalized_residual": Transform(
        name="volatility_normalized_residual",
        forward=_volatility_normalized_residual_forward,
        inverse=_volatility_normalized_residual_inverse,
        fit=_volatility_normalized_residual_fit,
        domain_check=_volatility_normalized_residual_domain,
        description=(
            "T = (y - EWMA_k(base)) / max(EWMA_k(|base - EWMA_k(base)|), floor): the ewma_residual level residual normalised by a recency-weighted volatility of the BASE series (volatility must be base-derived for the inverse to exist). Gives downstream models a unit-variance-ish target across calm and turbulent regimes. Inverse: y_hat = T_hat * vol + EWMA_k(base)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        # Both EWMA traces carry state across the row sequence (see ewma_residual).
        recurrent=True,
    ),
    "asinh_residual_multi": Transform(
        name="asinh_residual_multi",
        forward=_asinh_residual_multi_forward,
        inverse=_asinh_residual_multi_inverse,
        fit=_asinh_residual_multi_fit,
        domain_check=_asinh_residual_multi_domain,
        description=(
            "Multi-base sibling of asinh_residual: T = arcsinh(y) - sum_j(alpha_j * arcsinh(base_j)) - beta with joint OLS in arcsinh space over a K-column base matrix. Inverse y_hat = sinh(T_hat + arcsinh(base) @ alphas + beta). Inherits linear_residual_multi's condition-number guard (falls back to zero-alpha + intercept above _MULTI_BASE_COND_NUMBER_MAX)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "linear_residual_multi_robust": Transform(
        name="linear_residual_multi_robust",
        forward=_linear_residual_multi_forward,
        inverse=_linear_residual_multi_inverse,
        fit=_linear_residual_multi_robust_fit,
        domain_check=_linear_residual_multi_domain,
        description=(
            "Trimmed-LS variant of linear_residual_multi: joint OLS first pass -> drop rows where |resid| > 3 * sigma_MAD -> refit on the inlier set. Forward / inverse identical to linear_residual_multi once (alphas, beta) are fitted; keeps the condition-number guard of both passes. Stamped ``is_redundant_with_linres_multi=True`` when no row is trimmed so discovery can skip the duplicate evaluation."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "nadaraya_watson_residual": Transform(
        name="nadaraya_watson_residual",
        forward=_nadaraya_watson_residual_forward,
        inverse=_nadaraya_watson_residual_inverse,
        fit=_nadaraya_watson_residual_fit,
        domain_check=_nadaraya_watson_residual_domain,
        description=(
            "T = y - g(base) where g is a Gaussian-kernel Nadaraya-Watson regression with Silverman's-rule bandwidth on the train base. fit stores (base, y) knots subsampled to ~2000 points evenly along the base-sorted order (bounded O(n*m) predict); far-from-support rows converge to the nearest knot's y. Captures arbitrary non-monotone local dependence that monotonic_residual (monotone) and smoothing_spline_residual (global smoothness) miss. Inverse y_hat = T_hat + g(base)."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
    "gaussian_copula_residual": Transform(
        name="gaussian_copula_residual",
        forward=_gaussian_copula_residual_forward,
        inverse=_gaussian_copula_residual_inverse,
        fit=_gaussian_copula_residual_fit,
        domain_check=_gaussian_copula_residual_domain,
        description=(
            "Gaussian-copula residual: T = Phi^-1(ecdf_y(y)) - alpha * Phi^-1(ecdf_base(base)) - beta with (alpha, beta) OLS-fitted in normal-scores space on the TRAIN empirical CDFs. Collapses any monotone marginal distortion (like rank_ecdf_residual) while keeping the residual on a Gaussian scale RMSE inners like. Inverse maps back through the stored y-ECDF knots (y_hat = quantile_y(Phi(T_hat + alpha*z_b + beta))), so reconstructions cannot leave the train y-support."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    ),
}
