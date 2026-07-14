"""First half of the transforms registry dict literal (diff .. y_quantile_clip), split out of
``registry.py`` to keep it under the 1k-line monolith threshold. See ``_registry_setup.py`` for
the shared imports/adapters the entries below reference."""
from __future__ import annotations

from ._registry_setup import (
    TAG_CORE,
    TAG_EXTENDED,
    TAG_REGRESSION,
    Transform,
    _additive_residual_domain,
    _additive_residual_fit,
    _additive_residual_forward,
    _additive_residual_inverse,
    _cbrt_domain,
    _cbrt_fit,
    _cbrt_forward,
    _cbrt_inverse,
    _diff_domain,
    _diff_fit,
    _diff_forward,
    _diff_inverse,
    _ewma_residual_domain,
    _ewma_residual_fit,
    _ewma_residual_forward,
    _ewma_residual_inverse,
    _frac_diff_domain,
    _frac_diff_fit,
    _frac_diff_forward,
    _frac_diff_inverse,
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
    _log_domain_a,
    _log_domain_fitted_a,
    _log_fit_a,
    _log_forward_a,
    _log_inverse_a,
    _logratio_domain,
    _logratio_fit,
    _logratio_forward,
    _logratio_inverse,
    _median_residual_domain,
    _median_residual_fit,
    _median_residual_forward,
    _median_residual_inverse,
    _monotonic_residual_domain,
    _monotonic_residual_fit,
    _monotonic_residual_forward,
    _monotonic_residual_inverse,
    _quantile_residual_domain,
    _quantile_residual_fit,
    _quantile_residual_forward,
    _quantile_residual_inverse,
    _ratio_domain,
    _ratio_fit,
    _ratio_forward,
    _ratio_inverse,
    _rolling_quantile_ratio_centered_fit,
    _rolling_quantile_ratio_domain,
    _rolling_quantile_ratio_fit,
    _rolling_quantile_ratio_forward,
    _rolling_quantile_ratio_inverse,
    _sp_domain_a,
    _sp_fit_a,
    _sp_forward_a,
    _sp_inverse_a,
    _theilsen_residual_fit,
    _y_quantile_clip_domain,
    _y_quantile_clip_fit,
    _y_quantile_clip_forward,
    _y_quantile_clip_inverse,
)

_REGISTRY_PART1: dict[str, Transform] = {
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
        description=("T = y / base. Inverse y_hat = T_hat * base. Requires |base| > 0; " "fitted eps stored from train scale."),
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
        description=("T = y - alpha*base - beta with (alpha, beta) fitted via OLS on train. " "Inverse y_hat = T_hat + alpha*base + beta."),
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
            "Localised multiplicative residual: T = y / RollingMedian_k(base), with a TRAILING (past-only) window of ``k`` rows and an eps floor derived from train base scale to keep division safe at near-zero rolling medians. Inverse: y_hat = T_hat * RollingMedian_k(base). Like logratio but tracks the LOCAL base level instead of the global scale -- useful when y scales with a windowed median of base rather than the instantaneous value. The trailing window never reads future rows, so the transform is safe in time-ordered deployment; the legacy look-ahead centred window remains available as ``rolling_quantile_ratio_centered``. Params fitted before the mode field existed keep their historical centred behaviour on load."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
        # The rolling median reads neighbouring rows (past rows in trailing
        # mode); compacting the sequence before the forward would shrink each
        # window across a filtered gap, so the forward runs full-then-mask
        # (see Transform.recurrent).
        recurrent=True,
    ),
    "rolling_quantile_ratio_centered": Transform(
        name="rolling_quantile_ratio_centered",
        forward=_rolling_quantile_ratio_forward,
        inverse=_rolling_quantile_ratio_inverse,
        fit=_rolling_quantile_ratio_centered_fit,
        domain_check=_rolling_quantile_ratio_domain,
        description=(
            "Centred-window variant of ``rolling_quantile_ratio``: T = y / RollingMedian_k(base) with a CENTRED window of ``k`` rows. LOOK-AHEAD: the centred window reads FUTURE base rows, so in time-ordered deployment T leaks forward -- use only on non-chronological / cross-sectional row sequences; the trailing default is the time-safe choice."
        ),
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
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
}
