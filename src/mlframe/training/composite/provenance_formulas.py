"""Per-transform provenance formula builders (carved sibling of ``provenance.py``).

Carved VERBATIM out of ``provenance.py`` (sibling re-export pattern) to bring that module under the
1k-LOC ceiling. Holds ``_TRANSFORM_DESCRIPTIONS``, every ``_f_*`` per-transform human-readable
forward/inverse formula builder, ``_TRANSFORM_FORMULA_BUILDERS`` (the name -> builder registry),
``register_chain_provenance`` (dynamic auto-chain registration), ``_registered_transform_names``, and
``_format_transform_formulas`` (the public lookup used by :class:`CompositeProvenance` and by
``model_card.py`` / ``report.py``). ``provenance.py`` re-exports every symbol here at the bottom so all
existing ``from .provenance import X`` paths still resolve byte-for-byte. No formula-string or
registration behavior changed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_TRANSFORM_DESCRIPTIONS: dict[str, str] = {
    "diff": ("predicts the residual after subtracting the base feature " "from the target"),
    "additive_residual": ("predicts the residual after subtracting the base " "feature and a fitted constant offset (alpha fixed at 1)"),
    "median_residual": ("predicts the residual after subtracting the per-bin " "median of the target within quantile bins of the base"),
    "y_quantile_clip": ("clips the target to its train 0.5%/99.5% quantiles to " "bound the downstream model's effective range"),
    "ratio": ("predicts the multiplicative factor relating target to " "base feature"),
    "logratio": ("predicts the log-ratio of target to base feature, " "stabilising heavy-tail distributions"),
    "linear_residual": ("predicts the residual after subtracting a " "fitted linear contribution of the base feature"),
    "linear_residual_robust": ("predicts the residual after subtracting an " "outlier-robust (trimmed-LS) linear contribution of the base"),
    "theilsen_residual": ("predicts the residual after subtracting a high-breakdown " "robust (Theil-Sen median-of-slopes) linear contribution of the base"),
    "linear_residual_multi": ("predicts the residual after subtracting a fitted " "linear combination of several base features"),
    "linear_residual_grouped": ("predicts the residual after subtracting a " "per-group linear contribution of the base (James-Stein shrunk to global)"),
    "causal_anchor_residual": ("predicts the residual after subtracting a robust " "shrink coefficient (clamped to [0,1]) times the base anchor"),
    "second_diff": ("predicts the second difference of the target (level and linear " "drift removed via the lag-1 and lag-2 anchors)"),
    "rank_ecdf_residual": ("predicts the empirical-CDF (rank-space) residual between " "target and base, robust to monotone / heavy-tailed distortion"),
    "quantile_residual": ("predicts the heteroscedasticity-standardised residual: " "per-bin median removed then divided by per-bin IQR"),
    "monotonic_residual": ("predicts the residual after subtracting a fitted " "monotone (PCHIP) function of the base feature"),
    "ewma_residual": ("predicts the residual after subtracting an exponentially-" "weighted moving average of the base feature"),
    "rolling_quantile_ratio": ("predicts the ratio of target to a rolling median " "of the base feature (local multiplicative level)"),
    "frac_diff": ("predicts the fractionally-differenced target (long-memory-" "preserving stationarising transform)"),
    "cbrt_y": ("predicts the signed cube root of the target, compressing heavy " "tails without breaking sign"),
    "log_y": ("predicts the shifted log of the target, compressing right-skew"),
    "yeo_johnson_y": ("predicts the Yeo-Johnson power transform of the target " "(lambda fitted by MLE), normalising mixed-sign targets"),
    "quantile_normal_y": ("predicts the target mapped through its empirical CDF " "into a standard Normal"),
    "chain_linres_cbrt": ("predicts the cube-root of the linear-residual of the " "target (linear base absorber then tail compression)"),
    "chain_linres_yj": ("predicts the Yeo-Johnson transform of the linear-residual " "of the target"),
    "chain_monres_cbrt": ("predicts the cube-root of the monotone-residual of the " "target"),
    "chain_monres_yj": ("predicts the Yeo-Johnson transform of the monotone-residual " "of the target"),
    "chain_linres_cbrt_qn": ("predicts the quantile-normalised cube-root of the " "linear-residual of the target (3-stage chain)"),
    "asinh_residual": ("predicts the residual after subtracting a fitted linear " "contribution of the base in arcsinh space (signed-base logratio)"),
    "centered_ratio": ("predicts the ratio of target to a shifted base feature " "(ratio extended to signed bases)"),
    "polynomial_residual_deg2": ("predicts the residual after subtracting a fitted " "quadratic (degree-2) function of the base feature"),
    "rank_residual": ("predicts the distribution-free rank-space linear residual " "of the target against the base feature"),
    "smoothing_spline_residual": ("predicts the residual after subtracting a fitted " "smoothing spline of the base feature"),
    "reciprocal_residual": ("predicts the residual of 1/target against 1/base " "(reciprocal-scale dynamics)"),
    "geometric_mean_residual": ("predicts the ratio of target to the geometric mean " "of several positive base features"),
    "pairwise_interaction_residual": ("predicts the residual after subtracting a " "fitted multiple of the product of several base features"),
    "signed_power_y": ("predicts the signed power of the target, |y|^p with p fitted " "to minimise skew, symmetrising a heavy-tailed target"),
    "target_encoding_residual": ("predicts the residual after subtracting the " "empirical-Bayes smoothed per-category mean of the target"),
    "box_cox_y": ("predicts the Box-Cox power transform of the target (lambda " "fitted by MLE via scipy), a strictly-positive-only sibling of yeo_johnson_y"),
    "nadaraya_watson_residual": ("predicts the residual after subtracting a fitted " "Gaussian-kernel Nadaraya-Watson regression of the base feature"),
    "gaussian_copula_residual": ("predicts the residual in Gaussian-copula (normal-" "scores) space, collapsing monotone marginal distortion on both target and base"),
    "seasonal_residual": ("predicts the residual after subtracting the per-phase " "(row_index modulo a fitted period) mean of the target"),
    "volatility_normalized_residual": ("predicts the EWMA-level residual of the base " "feature normalised by its own recency-weighted volatility"),
    "asinh_residual_multi": ("predicts the residual after subtracting a fitted " "linear combination of several base features in arcsinh space"),
    "linear_residual_multi_robust": ("predicts the residual after subtracting a " "trimmed-LS (outlier-robust) fitted linear combination of several base features"),
    "rolling_quantile_ratio_centered": ("predicts the ratio of target to a CENTRED " "rolling median of the base feature (look-ahead; non-chronological use only)"),
    "ewma_residual_grouped": ("predicts the residual after subtracting a per-group " "exponentially-weighted moving average of the base feature"),
    "frac_diff_grouped": ("predicts the fractionally-differenced target with " "per-group pre-window anchors (long-memory-preserving, per-entity)"),
    "quantile_residual_grouped": ("predicts the per-group heteroscedasticity-" "standardised residual, James-Stein shrunk toward the global fit"),
    "monotonic_residual_grouped": ("predicts the residual after subtracting a " "per-group monotone (PCHIP) function of the base, James-Stein shrunk to the global fit"),
    "rolling_quantile_ratio_grouped": ("predicts the ratio of target to a per-group " "rolling median of the base feature (entity-local level, no cross-entity leakage)"),
}


def _fmt(value: Any, default: float = 0.0) -> str:
    """Render a fitted scalar param compactly (``%.4g``), tolerating a
    missing key or a non-numeric stored value (returns the raw repr)."""
    if value is None:
        value = default
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _join_base_columns(base_column: str, fitted_params: dict[str, Any]) -> str:
    """Render the multi-base matrix label. The wrapper threads the extra
    base columns via the spec (``extra_base_columns``); at formula-render
    time we only have the primary ``base_column`` so we annotate the count
    when the fit recorded ``n_bases`` / ``alphas`` for honesty."""
    n = fitted_params.get("n_bases")
    if n is None:
        alphas = fitted_params.get("alphas")
        n = len(alphas) if alphas is not None and hasattr(alphas, "__len__") else None
    if n and n > 1:
        return f"[{base_column}, ...{int(n) - 1} more]"
    return base_column


# Per-transform formula builders. Each returns (forward_human, inverse_human) given
# (target_col, base_column, fitted_params). The ``diff`` / ``ratio`` / ``logratio`` /
# ``linear_residual`` four keep exact strings (bit-identical -- pinned by ``TestFormulas``); the
# rest carry faithful, fitted-param-interpolating formulas so an inverse can be reproduced at
# serving time without source.


def _f_diff(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'diff': target is simply base subtracted from the raw target; inverse adds the base back to the residual prediction."""
    return (f"T = {t} - {b}", f"y_hat = T_hat + {b}")


def _f_additive_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'additive_residual': residual after subtracting the base feature and a fitted constant offset (alpha fixed at 1)."""
    beta = _fmt(p.get("beta"))
    return (
        f"T = {t} - {b} - ({beta})",
        f"y_hat = T_hat + {b} + ({beta})",
    )


def _f_median_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'median_residual': residual after subtracting the per-bin median of the target within quantile bins of the base."""
    return (
        f"T = {t} - median({t} | quantile_bin({b}))",
        f"y_hat = T_hat + bin_median[bin({b})]  (per-bin lookup, edge-bin fallback)",
    )


def _f_y_quantile_clip(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'y_quantile_clip': target clipped to its train 0.5%/99.5% quantiles to bound the downstream model's effective range; inverse re-clips the prediction to the same bounds."""
    q_lo = _fmt(p.get("q_lo"))
    q_hi = _fmt(p.get("q_hi"))
    return (
        f"T = clip({t}, {q_lo}, {q_hi})  (train 0.5%/99.5% quantiles)",
        f"y_hat = clip(T_hat, {q_lo}, {q_hi})",
    )


def _f_ratio(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'ratio': target predicts the multiplicative factor relating target to base feature; inverse multiplies the predicted ratio back by the base."""
    eps = float(p.get("eps", 1e-12))
    return (
        f"T = {t} / {b}  (with |{b}| >= {eps:.3g})",
        f"y_hat = T_hat * {b}",
    )


def _f_logratio(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'logratio': target predicts the log-ratio of target to base feature (heavy-tail stabilising); inverse soft-caps the prediction then exponentiates and rescales by the base."""
    median_t = float(p.get("median_t", 0.0))
    mad_eff = float(p.get("mad_eff", 0.0))
    k = float(p.get("soft_cap_k", 10.0))
    return (
        f"T = log({t}) - log({b})  (requires {t}, {b} > 0)",
        f"y_hat = {b} * exp(clip(T_hat, {median_t:.4g} +/- {k:.4g}*{mad_eff:.4g}))",
    )


def _f_linear_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'linear_residual': residual after subtracting a fitted linear contribution (alpha*base + beta) of the base feature."""
    alpha = float(p.get("alpha", 0.0))
    beta = float(p.get("beta", 0.0))
    return (
        f"T = {t} - {alpha:.4g} * {b} - ({beta:.4g})",
        f"y_hat = T_hat + {alpha:.4g} * {b} + ({beta:.4g})",
    )


def _f_linear_residual_multi(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'linear_residual_multi': residual after subtracting a fitted linear combination of several base features (alphas dotted with the bases, plus beta)."""
    bases = _join_base_columns(b, p)
    beta = _fmt(p.get("beta"))
    if p.get("collinear_fallback"):
        note = "  (collinear fallback: alphas=0, beta=train_mean)"
    else:
        note = ""
    return (
        f"T = {t} - alphas . {bases} - ({beta}){note}",
        f"y_hat = T_hat + alphas . {bases} + ({beta})",
    )


def _f_linear_residual_grouped(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'linear_residual_grouped': residual after subtracting a per-group linear contribution of the base (James-Stein shrunk toward the reported global alpha/beta)."""
    a_g = _fmt(p.get("alpha_global"))
    b_g = _fmt(p.get("beta_global"))
    return (
        f"T = {t} - alpha_g * {b} - beta_g  (per-group OLS, James-Stein shrunk " f"to global alpha={a_g}, beta={b_g}; small/unseen groups use global)",
        f"y_hat = T_hat + alpha_g * {b} + beta_g",
    )


def _f_quantile_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'quantile_residual': heteroscedasticity-standardised residual, per-bin median removed then divided by per-bin IQR."""
    return (
        f"T = ({t} - median_bin({t})) / IQR_bin({t})  (quantile bins of {b})",
        "y_hat = T_hat * IQR_bin + median_bin  (sparse/constant bins -> global)",
    )


def _f_monotonic_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'monotonic_residual': residual after subtracting a fitted monotone (PCHIP) function of the base feature."""
    return (
        f"T = {t} - g({b})  (g = monotone PCHIP fitted on quantile-knot medians)",
        f"y_hat = T_hat + g({b})  (out-of-range {b} clips to edge knot)",
    )


def _f_ewma_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'ewma_residual': residual after subtracting an exponentially-weighted moving average of the base feature (chronological order required)."""
    k = int(p.get("k", 7))
    alpha = 2.0 / (k + 1.0)
    return (
        f"T = {t} - EWMA_k({b})  (k={k}, alpha=2/(k+1)={alpha:.4g}; chronological order required)",
        f"y_hat = T_hat + EWMA_k({b})",
    )


def _f_rolling_quantile_ratio(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'rolling_quantile_ratio': ratio of target to a rolling median of the base feature (local multiplicative level); mode-aware -- trailing (past-only) is the default, the centred window (params without a mode key predate the field) is flagged LOOK-AHEAD."""
    k = int(p.get("k", 0))
    eps = float(p.get("eps", 0.0))
    mode = str(p.get("mode", "centered"))
    window = f"trailing k={k} (past-only)" if mode == "trailing" else f"centred k={k}; LOOK-AHEAD"
    return (
        f"T = {t} / RollingMedian_k({b})  ({window}, eps floor {eps:.3g})",
        f"y_hat = T_hat * RollingMedian_k({b})",
    )


def _f_frac_diff(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'frac_diff': fractionally-differenced target (Lopez de Prado long-memory-preserving stationarising transform); inverse reconstructs iteratively from the fitted weights."""
    d = _fmt(p.get("d"))
    lags = int(p.get("lags", 0))
    return (
        f"T_i = sum_k w_k * {t}_(i-k)  (Lopez de Prado frac-diff, d={d}, {lags} lags; " f"w_k = -w_(k-1)*(d-k+1)/k, pre-window padded with train mean)",
        "y_hat_i = (T_i - sum_(k>=1) w_k * y_hat_(i-k)) / w_0  (iterative reconstruction)",
    )


def _f_cbrt_y(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'cbrt_y': target predicts the signed cube root of the raw target, compressing heavy tails without breaking sign; inverse cubes the prediction."""
    return (f"T = sign({t}) * |{t}|^(1/3)", "y_hat = T_hat^3")


def _f_log_y(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'log_y': target predicts the shifted log of the raw target, compressing right-skew; inverse exponentiates and removes the shift."""
    offset = _fmt(p.get("offset"))
    return (
        f"T = log({t} + {offset})  (offset fitted so min({t}_train) + offset > 0)",
        f"y_hat = exp(T_hat) - {offset}",
    )


def _f_yeo_johnson_y(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'yeo_johnson_y': target predicts the Yeo-Johnson power transform of the raw target (lambda fitted by MLE), normalising mixed-sign targets."""
    lam = _fmt(p.get("lambda"))
    return (
        f"T = YeoJohnson({t}; lambda={lam})  (lambda fitted by MLE, clipped to [-2, 4])",
        f"y_hat = YeoJohnson_inverse(T_hat; lambda={lam})",
    )


def _f_quantile_normal_y(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'quantile_normal_y': target mapped through its empirical CDF into a standard Normal; inverse maps back via the train empirical-CDF knots."""
    return (
        f"T = Phi^-1(rank({t}) / (n+1))  (empirical-CDF -> standard Normal via knots)",
        "y_hat = empirical_CDF_inverse(Phi(T_hat))  (knot interpolation, scale lost)",
    )


def _f_asinh_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'asinh_residual': residual after subtracting a fitted linear contribution of the base in arcsinh space (signed-base logratio)."""
    alpha = _fmt(p.get("alpha"))
    beta = _fmt(p.get("beta"))
    return (
        f"T = asinh({t}) - {alpha} * asinh({b}) - ({beta})",
        f"y_hat = sinh(T_hat + {alpha} * asinh({b}) + ({beta}))",
    )


def _f_centered_ratio(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'centered_ratio': ratio of target to a shifted base feature (ratio extended to signed bases via a fitted shift c so base+c>0)."""
    c = _fmt(p.get("c"))
    eps = float(p.get("eps", 0.0))
    return (
        f"T = {t} / ({b} + {c})  (c fitted so {b}+c>0, eps floor {eps:.3g})",
        f"y_hat = T_hat * ({b} + {c})",
    )


def _f_polynomial_residual_deg2(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'polynomial_residual_deg2': residual after subtracting a fitted quadratic (degree-2) function of the base feature."""
    a1 = _fmt(p.get("alpha1"))
    a2 = _fmt(p.get("alpha2"))
    beta = _fmt(p.get("beta"))
    return (
        f"T = {t} - {a1} * {b} - {a2} * {b}^2 - ({beta})",
        f"y_hat = T_hat + {a1} * {b} + {a2} * {b}^2 + ({beta})",
    )


def _f_rank_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'rank_residual': distribution-free rank-space linear residual of the target against the base feature; inverse looks up the recovered rank in the train sorted-y array."""
    alpha = _fmt(p.get("alpha"))
    beta = _fmt(p.get("beta"))
    return (
        f"T = rank({t})/n - {alpha} * rank({b})/n - ({beta})  (train sorted-y/base lookup)",
        "y_hat = train_sorted_y[clip(rank_recovered, 0, 1)]",
    )


def _f_smoothing_spline_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'smoothing_spline_residual': residual after subtracting a fitted smoothing spline (UnivariateSpline) of the base feature."""
    s = _fmt(p.get("s"))
    return (
        f"T = {t} - g({b})  (g = UnivariateSpline, smoothing s={s})",
        f"y_hat = T_hat + g({b})",
    )


def _f_reciprocal_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'reciprocal_residual': residual of 1/target against 1/base (reciprocal-scale dynamics)."""
    eps_y = float(p.get("eps_y", 0.0))
    eps_b = float(p.get("eps_b", 0.0))
    return (
        f"T = 1/{t} - 1/{b}  (eps floors {eps_y:.3g}/{eps_b:.3g})",
        f"y_hat = 1 / (T_hat + 1/{b})",
    )


def _f_geometric_mean_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'geometric_mean_residual': ratio of target to the geometric mean of several positive base features."""
    bases = _join_base_columns(b, p)
    return (
        f"T = {t} / geomean({bases})  (requires every base > 0)",
        f"y_hat = T_hat * geomean({bases})",
    )


def _f_pairwise_interaction_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'pairwise_interaction_residual': residual after subtracting a fitted multiple of the product of several base features."""
    bases = _join_base_columns(b, p)
    alpha = _fmt(p.get("alpha"))
    beta = _fmt(p.get("beta"))
    return (
        f"T = {t} - {alpha} * prod({bases}) - ({beta})",
        f"y_hat = T_hat + {alpha} * prod({bases}) + ({beta})",
    )


def _f_linear_residual_robust(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'linear_residual_robust': delegates to ``_f_linear_residual`` since both fit the same linear form once alpha/beta are computed, just with an outlier-robust (trimmed-LS) estimator."""
    # Same forward/inverse algebra as linear_residual once (alpha, beta) fitted.
    return _f_linear_residual(t, b, p)


def _f_theilsen_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'theilsen_residual': same linear forward/inverse algebra as linear_residual, but alpha is the high-breakdown robust Theil-Sen median-of-pairwise-slopes and beta the median intercept."""
    # Same forward/inverse algebra as linear_residual once the Theil-Sen
    # (median-of-pairwise-slopes) alpha + median-intercept beta are fitted.
    alpha = float(p.get("alpha", 0.0))
    beta = float(p.get("beta", 0.0))
    return (
        f"T = {t} - {alpha:.4g} * {b} - ({beta:.4g})  (alpha=Theil-Sen median slope)",
        f"y_hat = T_hat + {alpha:.4g} * {b} + ({beta:.4g})",
    )


def _f_causal_anchor_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'causal_anchor_residual': residual after subtracting a robust shrink coefficient (alpha, clamped to [0,1]) times the base anchor."""
    # Pure-additive shrink: single fitted alpha clamped to [0,1], no beta.
    alpha = float(p.get("alpha", 0.0))
    return (
        f"T = {t} - {alpha:.4g} * {b}  (alpha in [0,1] robust anchor shrink)",
        f"y_hat = T_hat + {alpha:.4g} * {b}",
    )


def _f_second_diff(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'second_diff': second difference of the target (level and linear drift removed via the lag-1 and lag-2 anchors); parameter-free."""
    # b1 = lag-1 anchor (base_prev), b2 = lag-2 anchor (base_prev2).
    return (
        f"T = {t} - 2*{b}_lag1 + {b}_lag2  (second difference; no fitted params)",
        f"y_hat = T_hat + 2*{b}_lag1 - {b}_lag2",
    )


def _f_rank_ecdf_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'rank_ecdf_residual': empirical-CDF (rank-space) residual between target and base, robust to monotone / heavy-tailed distortion."""
    return (
        f"T = ecdf_y({t}) - ecdf_base({b})  (train empirical-CDF / rank space)",
        f"y_hat = quantile_y(T_hat + ecdf_base({b}))  (inverse-ECDF of train y)",
    )


def _f_chain_linres_cbrt(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'chain_linres_cbrt': cube-root of the linear-residual of the target (linear base absorber then tail compression), a two-stage chain."""
    bp = p.get("bivariate_params", {}) or {}
    alpha = _fmt(bp.get("alpha"))
    beta = _fmt(bp.get("beta"))
    return (
        f"T1 = {t} - {alpha} * {b} - ({beta}); T = sign(T1) * |T1|^(1/3)",
        f"T1_hat = T_hat^3; y_hat = T1_hat + {alpha} * {b} + ({beta})",
    )


def _f_chain_linres_yj(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'chain_linres_yj': Yeo-Johnson transform of the linear-residual of the target, a two-stage chain."""
    bp = p.get("bivariate_params", {}) or {}
    up = p.get("unary_params", {}) or {}
    alpha = _fmt(bp.get("alpha"))
    beta = _fmt(bp.get("beta"))
    lam = _fmt(up.get("lambda"))
    return (
        f"T1 = {t} - {alpha} * {b} - ({beta}); T = YeoJohnson(T1; lambda={lam})",
        f"T1_hat = YeoJohnson_inverse(T_hat; lambda={lam}); y_hat = T1_hat + {alpha} * {b} + ({beta})",
    )


def _f_chain_monres_cbrt(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'chain_monres_cbrt': cube-root of the monotone-residual of the target, a two-stage chain."""
    return (
        f"T1 = {t} - g({b}) (monotone PCHIP); T = sign(T1) * |T1|^(1/3)",
        f"T1_hat = T_hat^3; y_hat = T1_hat + g({b})",
    )


def _f_chain_monres_yj(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'chain_monres_yj': Yeo-Johnson transform of the monotone-residual of the target, a two-stage chain."""
    up = p.get("unary_params", {}) or {}
    lam = _fmt(up.get("lambda"))
    return (
        f"T1 = {t} - g({b}) (monotone PCHIP); T = YeoJohnson(T1; lambda={lam})",
        f"T1_hat = YeoJohnson_inverse(T_hat; lambda={lam}); y_hat = T1_hat + g({b})",
    )


def _f_chain_linres_cbrt_qn(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'chain_linres_cbrt_qn': quantile-normalised cube-root of the linear-residual of the target, a 3-stage chain (linear residual, then cube root, then quantile-normal)."""
    bp = p.get("bivariate_params", {}) or {}
    alpha = _fmt(bp.get("alpha"))
    beta = _fmt(bp.get("beta"))
    return (
        f"T1 = {t} - {alpha} * {b} - ({beta}); T2 = sign(T1) * |T1|^(1/3); " f"T = Phi^-1(rank(T2)/(n+1))",
        f"T2_hat = empirical_CDF_inverse(Phi(T_hat)); T1_hat = T2_hat^3; " f"y_hat = T1_hat + {alpha} * {b} + ({beta})  (scale lost in quantile-normal stage)",
    )


def _f_signed_power_y(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'signed_power_y': signed power of the target, |y|^p with p fitted to minimise skew, symmetrising a heavy-tailed target."""
    pw = _fmt(p.get("p"))
    return (
        f"T = sign({t}) * |{t}|^p  (p={pw}, fitted to minimise skew)",
        "y_hat = sign(T_hat) * |T_hat|^(1/p)",
    )


def _f_target_encoding_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'target_encoding_residual': residual after subtracting the empirical-Bayes smoothed per-category mean of the target."""
    sm = _fmt(p.get("smoothing"))
    return (
        f"T = {t} - cat_mean(group)  (empirical-Bayes smoothed per-category mean, " f"smoothing={sm}; unseen categories use the global mean)",
        "y_hat = T_hat + cat_mean(group)",
    )


def _f_box_cox_y(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'box_cox_y': Box-Cox power transform of the raw target (lambda fitted by MLE), strictly-positive-only; inverse is the closed-form power, floored to stay finite."""
    lam = _fmt(p.get("lambda"))
    return (
        f"T = ({t}^lambda - 1) / lambda  (lambda={lam}, log({t}) at lambda=0; requires {t} > 0)",
        f"y_hat = (T_hat*lambda + 1)^(1/lambda)  (lambda={lam}, exp(T_hat) at lambda=0; base floored to stay finite)",
    )


def _f_nadaraya_watson_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'nadaraya_watson_residual': residual after subtracting a fitted Gaussian-kernel Nadaraya-Watson regression of the base feature."""
    bw = _fmt(p.get("bandwidth"))
    return (
        f"T = {t} - g({b})  (g = Gaussian-kernel Nadaraya-Watson, bandwidth={bw}, " "subsampled knots; far-from-support rows converge to the nearest knot)",
        f"y_hat = T_hat + g({b})",
    )


def _f_gaussian_copula_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'gaussian_copula_residual': residual in Gaussian-copula (normal-scores) space, OLS-fitted, collapsing monotone marginal distortion on both target and base."""
    alpha = _fmt(p.get("alpha"))
    beta = _fmt(p.get("beta"))
    return (
        f"T = z_y - {alpha} * z_b - ({beta})  (z_y=Phi^-1(ecdf_{t}), z_b=Phi^-1(ecdf_{b}), " "train empirical CDFs)",
        f"y_hat = quantile_{t}(Phi(T_hat + {alpha} * z_b + ({beta})))  (train y-ECDF knot lookup)",
    )


def _f_seasonal_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'seasonal_residual': residual after subtracting the per-phase (row_index modulo a fitted period) mean of the target; parameter-free forward besides the fitted period."""
    period = p.get("period")
    return (
        f"T = {t} - phase_mean[row_index mod {period}]  (period={period}, phase 0 = first row of the batch)",
        f"y_hat = T_hat + phase_mean[row_index mod {period}]",
    )


def _f_volatility_normalized_residual(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'volatility_normalized_residual': EWMA-level residual of the base feature normalised by its own recency-weighted (EWMA of |base - level|) volatility, floored to stay finite."""
    k = int(p.get("k", 0))
    floor = _fmt(p.get("floor"))
    return (
        f"T = ({t} - EWMA_k({b})) / max(EWMA_k(|{b} - EWMA_k({b})|), {floor})  (k={k})",
        f"y_hat = T_hat * max(EWMA_k(|{b} - EWMA_k({b})|), {floor}) + EWMA_k({b})",
    )


def _f_asinh_residual_multi(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'asinh_residual_multi': multi-base sibling of asinh_residual, joint OLS in arcsinh space over several base columns."""
    bases = _join_base_columns(b, p)
    beta = _fmt(p.get("beta"))
    if p.get("collinear_fallback"):
        note = "  (collinear fallback: alphas=0, beta=train_mean)"
    else:
        note = ""
    return (
        f"T = asinh({t}) - alphas . asinh({bases}) - ({beta}){note}",
        f"y_hat = sinh(T_hat + alphas . asinh({bases}) + ({beta}))",
    )


def _f_rolling_quantile_ratio_grouped(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'rolling_quantile_ratio_grouped': per-group sibling of rolling_quantile_ratio -- same window/eps math, but the rolling median resets at each group boundary (delegates to the ungrouped renderer since the interpolated params are identical)."""
    forward, inverse = _f_rolling_quantile_ratio(t, b, p)
    return (forward + "  (window confined to each row's group)", inverse)


def _f_ewma_residual_grouped(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'ewma_residual_grouped': per-group sibling of ewma_residual -- same EWMA math, recursion reset at each group boundary and seeded by the group's own train-base mean."""
    k = int(p.get("k", 7))
    alpha = 2.0 / (k + 1.0)
    return (
        f"T = {t} - EWMA_k({b})  (k={k}, alpha=2/(k+1)={alpha:.4g}; per-group recursion reset, " "unseen groups use the global anchor)",
        f"y_hat = T_hat + EWMA_k({b})",
    )


def _f_frac_diff_grouped(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'frac_diff_grouped': per-group sibling of frac_diff -- same fractional-differencing weights, but each group's history pads with its own train-y mean at the group boundary."""
    d = _fmt(p.get("d"))
    lags = int(p.get("lags", 0))
    return (
        f"T_i = sum_k w_k * {t}_(i-k)  (Lopez de Prado frac-diff, d={d}, {lags} lags; " "per-group pre-window padding, each group padded with its OWN train-y mean)",
        "y_hat_i = (T_i - sum_(k>=1) w_k * y_hat_(i-k)) / w_0  (iterative reconstruction, per group)",
    )


def _f_quantile_residual_grouped(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'quantile_residual_grouped': per-group sibling of quantile_residual -- per-group bin medians/IQR, James-Stein shrunk toward the global fit; small/unseen groups fall back to the global fit entirely."""
    return (
        f"T = ({t} - median_bin_group({t})) / IQR_bin_group({t})  (quantile bins of {b}, " "per-group James-Stein shrunk toward the global fit)",
        "y_hat = T_hat * IQR_bin_group + median_bin_group  (small/unseen groups use the global fit)",
    )


def _f_monotonic_residual_grouped(t: str, b: str, p: dict) -> tuple[str, str]:
    """Formula strings for 'monotonic_residual_grouped': per-group sibling of monotonic_residual -- per-group monotone PCHIP, James-Stein shrunk toward the global fit; small/unseen groups fall back to the global fit entirely."""
    return (
        f"T = {t} - g_group({b})  (g_group = per-group monotone PCHIP, James-Stein " "shrunk toward the global fit)",
        f"y_hat = T_hat + g_group({b})  (small/unseen groups use the global g)",
    )


# name -> builder. Source of truth alongside ``_TRANSFORM_DESCRIPTIONS``;
# both are coverage-pinned against the live registry by the formula-coverage meta-test.
_TRANSFORM_FORMULA_BUILDERS: dict[str, Any] = {
    "diff": _f_diff,
    "additive_residual": _f_additive_residual,
    "median_residual": _f_median_residual,
    "y_quantile_clip": _f_y_quantile_clip,
    "ratio": _f_ratio,
    "logratio": _f_logratio,
    "linear_residual": _f_linear_residual,
    "linear_residual_robust": _f_linear_residual_robust,
    "theilsen_residual": _f_theilsen_residual,
    "linear_residual_multi": _f_linear_residual_multi,
    "linear_residual_grouped": _f_linear_residual_grouped,
    "causal_anchor_residual": _f_causal_anchor_residual,
    "second_diff": _f_second_diff,
    "rank_ecdf_residual": _f_rank_ecdf_residual,
    "quantile_residual": _f_quantile_residual,
    "monotonic_residual": _f_monotonic_residual,
    "ewma_residual": _f_ewma_residual,
    "rolling_quantile_ratio": _f_rolling_quantile_ratio,
    "frac_diff": _f_frac_diff,
    "cbrt_y": _f_cbrt_y,
    "log_y": _f_log_y,
    "yeo_johnson_y": _f_yeo_johnson_y,
    "quantile_normal_y": _f_quantile_normal_y,
    "asinh_residual": _f_asinh_residual,
    "centered_ratio": _f_centered_ratio,
    "polynomial_residual_deg2": _f_polynomial_residual_deg2,
    "rank_residual": _f_rank_residual,
    "smoothing_spline_residual": _f_smoothing_spline_residual,
    "reciprocal_residual": _f_reciprocal_residual,
    "geometric_mean_residual": _f_geometric_mean_residual,
    "pairwise_interaction_residual": _f_pairwise_interaction_residual,
    "chain_linres_cbrt": _f_chain_linres_cbrt,
    "chain_linres_yj": _f_chain_linres_yj,
    "chain_monres_cbrt": _f_chain_monres_cbrt,
    "chain_monres_yj": _f_chain_monres_yj,
    "chain_linres_cbrt_qn": _f_chain_linres_cbrt_qn,
    "signed_power_y": _f_signed_power_y,
    "target_encoding_residual": _f_target_encoding_residual,
    "box_cox_y": _f_box_cox_y,
    "nadaraya_watson_residual": _f_nadaraya_watson_residual,
    "gaussian_copula_residual": _f_gaussian_copula_residual,
    "seasonal_residual": _f_seasonal_residual,
    "volatility_normalized_residual": _f_volatility_normalized_residual,
    "asinh_residual_multi": _f_asinh_residual_multi,
    "linear_residual_multi_robust": _f_linear_residual_multi,
    "rolling_quantile_ratio_centered": _f_rolling_quantile_ratio,
    "ewma_residual_grouped": _f_ewma_residual_grouped,
    "frac_diff_grouped": _f_frac_diff_grouped,
    "quantile_residual_grouped": _f_quantile_residual_grouped,
    "monotonic_residual_grouped": _f_monotonic_residual_grouped,
    "rolling_quantile_ratio_grouped": _f_rolling_quantile_ratio_grouped,
}


def register_chain_provenance(chain_name: str, residual_name: str, unary_name: str) -> None:
    """Register a formula builder + description for a DYNAMICALLY composed
    auto-chain transform (residual stage then tail-unary stage), composing both
    from the two stages' static entries. auto-chain registers its composed
    Transform into the live registry at fit time; this keeps that transform
    self-describing so it appears in provenance/reports with a real formula
    instead of the opaque generic stub (and satisfies the coverage invariant
    that every registered transform has provenance)."""
    if chain_name in _TRANSFORM_FORMULA_BUILDERS:
        return
    res_desc = _TRANSFORM_DESCRIPTIONS.get(residual_name, residual_name)
    un_desc = _TRANSFORM_DESCRIPTIONS.get(unary_name, unary_name)
    _TRANSFORM_DESCRIPTIONS[chain_name] = f"two-stage chain -- {un_desc}, applied to the residual that {res_desc}"

    def _builder(t: str, b: str, p: dict, _r: str = residual_name, _u: str = unary_name) -> tuple[str, str]:
        """Per-call formula-string builder for one dynamically registered two-stage chain transform (residual stage then tail-unary stage)."""
        return (
            f"T = {_u}( {_r}({t}, {b}) )  (residual then tail-compression)",
            f"y_hat = {_r}^-1( {_u}^-1(T_hat) )",
        )

    _TRANSFORM_FORMULA_BUILDERS[chain_name] = _builder


def _registered_transform_names() -> frozenset[str]:
    """Live registry keys, lazily imported (no import-time coupling; the
    registry is only needed for the formula-coverage meta-test and graceful
    fallback bookkeeping, never at provenance module load)."""
    try:
        from .transforms import TRANSFORMS_REGISTRY  # local import: avoid cycle
        return frozenset(TRANSFORMS_REGISTRY.keys())
    except Exception as e:  # pragma: no cover - registry should always import
        logger.debug("composite transforms registry import failed, treating as empty (%s)", e)
        return frozenset()


def _format_transform_formulas(
    transform_name: str, base_column: str, target_col: str,
    fitted_params: dict[str, Any],
) -> tuple[str, str, str]:
    """Return (forward_human, inverse_human, description) strings.

    Strings interpolate fitted parameter values where applicable. Used
    by :class:`CompositeProvenance` to render audit-friendly formula
    descriptions without forcing the caller to know the registry.

    Every transform registered in ``TRANSFORMS_REGISTRY`` has a dedicated
    builder in ``_TRANSFORM_FORMULA_BUILDERS``; a genuinely unknown / future
    name falls back to the generic ``forward(...)`` / ``inverse(...)`` stub
    so the function never raises.
    """
    description = _TRANSFORM_DESCRIPTIONS.get(transform_name, "")
    _fallback_description = description if description else f"transform '{transform_name}'"
    builder = _TRANSFORM_FORMULA_BUILDERS.get(transform_name)
    if builder is not None:
        forward, inverse = builder(target_col, base_column, fitted_params if fitted_params else {})
        return (forward, inverse, _fallback_description)
    # Unknown / future transform: fall back to a generic description.
    return (
        f"T = forward({target_col}, {base_column}) [{transform_name}]",
        f"y_hat = inverse(T_hat, {base_column}) [{transform_name}]",
        _fallback_description,
    )
