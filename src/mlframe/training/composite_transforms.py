"""Composite-target transform registry: 11 transforms (diff, ratio, logratio, linear_residual + multi/grouped/quantile/monotonic/ewma/rolling_quantile/frac_diff extended set) + Transform dataclass + UnknownTransformError / DomainViolationError. Split out of composite.py to keep transform-implementation surface separate from the wrapper + discovery surface; composite.py re-exports every symbol below at its bottom for full back-compat."""


from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
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
    forward: Callable[..., np.ndarray]
    inverse: Callable[..., np.ndarray]
    fit: Callable[..., dict[str, Any]]
    domain_check: Callable[[np.ndarray, np.ndarray], np.ndarray]
    description: str
    tags: frozenset[str] = field(default_factory=frozenset)
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


# ----------------------------------------------------------------------
# diff: T = y - base. Always defined, no params, no domain restrictions.
# ----------------------------------------------------------------------

def _diff_fit(y: np.ndarray, base: np.ndarray) -> dict[str, Any]:
    return {}


def _diff_forward(y: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return y - base


def _diff_inverse(t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return t_hat + base


def _diff_domain(y: np.ndarray | None, base: np.ndarray) -> np.ndarray:
    base_ok = np.isfinite(base)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)


# ----------------------------------------------------------------------
# additive_residual: T = y - base - beta (alpha=1.0 fixed, beta learned).
# Strict-AR-1 sweet spot between ``diff`` (no offset) and
# ``linear_residual`` (alpha+beta both learned). Pure additive
# inverse: y = T + base + beta. Distinct from ``diff`` when the
# (y, base) relationship has a non-zero level shift.
# ----------------------------------------------------------------------
def _additive_residual_fit(y: np.ndarray, base: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(y) & np.isfinite(base)
    if not finite.any():
        return {"beta": 0.0}
    beta = float(np.mean(y[finite] - base[finite]))
    return {"beta": beta}


def _additive_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    return y - base - float(params.get("beta", 0.0))


def _additive_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    return t_hat + base + float(params.get("beta", 0.0))


def _additive_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(base)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)


# ----------------------------------------------------------------------
# median_residual: T = y - median(y | bin(base)). Non-parametric
# bin-conditional residual; no fitted alpha/beta, no PCHIP spline.
# Pure additive inverse y = T + median_bin[base], MLP-friendly because
# the inverse is a constant lookup per row instead of a nonlinear
# function. Distinct from ``quantile_residual`` (which divides by IQR
# and reintroduces nonlinear inverse scaling) and ``monotonic_residual``
# (which fits a PCHIP -- nonlinear inverse).
# ----------------------------------------------------------------------
_MEDIAN_RESIDUAL_N_BINS: int = 20


def _median_residual_per_bin_medians_v1_pyloop(
    y_f: np.ndarray, bin_idx: np.ndarray, n_bins: int,
) -> np.ndarray:
    """Reference implementation: build per-bin boolean masks and take ``np.median`` on each."""
    out = np.full(n_bins, np.median(y_f), dtype=np.float64)
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.any():
            out[i] = float(np.median(y_f[mask]))
    return out


def _median_residual_per_bin_medians_v2_pandas_groupby(
    y_f: np.ndarray, bin_idx: np.ndarray, n_bins: int,
) -> np.ndarray:
    """Single hash-groupby pass via pandas; bins with no rows keep the global median."""
    import pandas as _pd
    global_med = float(np.median(y_f))
    out = np.full(n_bins, global_med, dtype=np.float64)
    grp = _pd.Series(y_f).groupby(bin_idx, sort=True).median()
    out[grp.index.to_numpy()] = grp.to_numpy()
    return out


def _median_residual_per_bin_medians(
    y_f: np.ndarray, bin_idx: np.ndarray, n_bins: int,
) -> np.ndarray:
    """Size-aware dispatcher across the per-bin median variants.

    Bench (bench_median_quantile_residual.py, n in {100k, 1M} x n_bins in {10, 20}): v2 pandas-groupby is the best CPU variant at n=100k (~1.2-1.45x over v1 numpy mask loop) but ties / loses at n=1M (~1.15x). v4 numba-sort-based was slower at every size (extra argsort dominates). Routes by total element count: pandas wins on small n; numpy mask-loop wins on large n. Threshold derived from measurement.
    """
    if y_f.size <= 200_000:
        try:
            return _median_residual_per_bin_medians_v2_pandas_groupby(y_f, bin_idx, n_bins)
        except Exception:
            pass
    return _median_residual_per_bin_medians_v1_pyloop(y_f, bin_idx, n_bins)


def _median_residual_fit(y: np.ndarray, base: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(y) & np.isfinite(base)
    if not finite.any():
        return {
            "bin_edges": np.array([0.0]), "bin_medians": np.array([0.0]),
            "fallback_median": 0.0,
        }
    y_f = y[finite].astype(np.float64)
    b_f = base[finite].astype(np.float64)
    n_bins = int(_MEDIAN_RESIDUAL_N_BINS)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(b_f, quantiles)
    bin_edges = np.unique(bin_edges)
    # Heavily-discretised base (e.g. integer counts with <20 distinct values) causes ``np.unique`` to collapse the n_bins+1 quantile edges to fewer slots. ``np.digitize`` then routes most rows to bin 0 and the per-bin median lookup defeats the residual residual modelling. Warn so downstream callers know the granularity collapsed before treating the residual as a useful signal.
    if bin_edges.size - 1 < n_bins:
        import warnings as _w
        _w.warn(
            f"_median_residual_fit: base has only {bin_edges.size - 1} distinct quantile edges (requested n_bins={n_bins}); residual granularity collapsed. Transform falls back to a coarse per-bin median - consider a different base or transform when this fires.",
            RuntimeWarning,
            stacklevel=2,
        )
    if bin_edges.size < 2:
        bin_edges = np.array([bin_edges[0], bin_edges[0] + 1e-9])
    bin_idx = np.digitize(b_f, bin_edges[1:-1])
    bin_medians = _median_residual_per_bin_medians(y_f, bin_idx, bin_edges.size - 1)
    return {
        "bin_edges": bin_edges,
        "bin_medians": bin_medians,
        "fallback_median": float(np.median(y_f)),
    }


def _median_residual_g(base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    bin_edges = np.asarray(params["bin_edges"], dtype=np.float64)
    bin_medians = np.asarray(params["bin_medians"], dtype=np.float64)
    fallback = float(params.get("fallback_median", 0.0))
    if bin_edges.size < 2:
        return np.full_like(base, fallback, dtype=np.float64)
    idx = np.digitize(base, bin_edges[1:-1])
    idx = np.clip(idx, 0, bin_medians.size - 1)
    return bin_medians[idx]


def _median_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    return y - _median_residual_g(base, params)


def _median_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    return t_hat + _median_residual_g(base, params)


def _median_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(base)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)


# ----------------------------------------------------------------------
# y_quantile_clip: T = clip(y, q_lo, q_hi). Unary y-only transform
# (requires_base=False) that limits the target range to [q_lo, q_hi]
# at fit time. Inverse is the identity (no un-clip possible since
# clipping is lossy on extremes). Used as a limit-damage transform
# for neural / linear downstream models that might extrapolate
# wildly outside train-y range; downstream model's predictions
# stay bounded by the clipped y_train range.
# ----------------------------------------------------------------------
_Y_QUANTILE_CLIP_LO: float = 0.005
_Y_QUANTILE_CLIP_HI: float = 0.995


def _y_quantile_clip_fit(y: np.ndarray, base: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(y)
    if not finite.any():
        return {"q_lo": 0.0, "q_hi": 0.0}
    y_f = y[finite].astype(np.float64)
    q_lo = float(np.quantile(y_f, _Y_QUANTILE_CLIP_LO))
    q_hi = float(np.quantile(y_f, _Y_QUANTILE_CLIP_HI))
    return {"q_lo": q_lo, "q_hi": q_hi}


def _y_quantile_clip_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    q_lo = float(params.get("q_lo", -np.inf))
    q_hi = float(params.get("q_hi", np.inf))
    return np.clip(y.astype(np.float64), q_lo, q_hi)


def _y_quantile_clip_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    q_lo = float(params.get("q_lo", -np.inf))
    q_hi = float(params.get("q_hi", np.inf))
    return np.clip(np.asarray(t_hat, dtype=np.float64), q_lo, q_hi)


def _y_quantile_clip_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    if y is None:
        return np.isfinite(base) | ~np.isfinite(base)  # all True
    return np.isfinite(y)


# ----------------------------------------------------------------------
# ratio: T = y / base. Requires |base| >= eps.
# ----------------------------------------------------------------------

def _ratio_fit(y: np.ndarray, base: np.ndarray) -> dict[str, Any]:
    # eps relative to the typical scale of base on train -- small enough
    # not to bias the transform but large enough to keep division
    # numerically clean. Stored in params so predict time uses the
    # SAME eps (no train/test drift).
    scale = float(np.median(np.abs(base[np.isfinite(base) & (base != 0)])))
    eps = max(scale * 1e-6, 1e-12) if scale > 0 else 1e-12
    return {"eps": eps}


def _ratio_forward(y: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    eps = float(params["eps"])
    safe_base = np.where(np.abs(base) < eps, np.sign(base + 1e-300) * eps, base)
    return y / safe_base


def _ratio_inverse(t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return t_hat * base


def _ratio_domain(y: np.ndarray | None, base: np.ndarray) -> np.ndarray:
    base_ok = np.isfinite(base) & (np.abs(base) > 0)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)


# ----------------------------------------------------------------------
# logratio: T = log(y) - log(base). Requires y, base > 0.
# Inverse uses MAD-soft-cap on T_hat to prevent exp() blow-up.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# linear_residual: T = y - alpha*base - beta. OLS on train.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# linear_residual_robust: trimmed-LS (M-estimator with 3*MAD hard threshold).
# Bench (benchmarks/bench_robust_linres_1M.py): 0.12s on 1M rows with 5%
# Cauchy outliers; alpha error 0.01% vs OLS 0.51%, beta error 2.40% vs
# OLS 95.25%. Beats sklearn HuberRegressor (3.28s), RANSACRegressor
# (4.53s), TheilSenRegressor (O(n^2), MemoryError on 1M), statsmodels
# QuantReg L1 (9.81s) on time-vs-accuracy. Implementation: OLS first
# pass, drop rows where |residual| > 3 * MAD (robust scale via
# 1.4826 * MAD = sigma-equivalent under Normality), refit OLS on the
# survivors. Two passes total.
# ----------------------------------------------------------------------

_LINRES_ROBUST_MAD_K: float = 3.0
"""Hard threshold: rows with |residual| > k * sigma_MAD are dropped before the second-pass OLS. k=3 keeps ~99.7% of inliers under Normality and drops outliers heavier than 3-sigma."""

_LINRES_ROBUST_MIN_KEEP_FRAC: float = 0.5
"""Safety: if the MAD-trim would drop more than (1 - this) of rows, fall back to plain OLS. Protects against pathological all-outlier targets where MAD is too small and trimming becomes destructive."""


# ----------------------------------------------------------------------
# linear_residual_multi (#1 from R10c brainstorm).
# T = y - Σⱼ αⱼ·baseⱼ - β. Joint OLS on (n, K) base matrix.
#
# Same inverse contract as single-base linear_residual; the only thing
# that changes is the shape of ``base``: 2-D ``(n, K)`` instead of 1-D
# ``(n,)``. Callers materialise the matrix from ``spec.base_columns``
# at predict time. ``fit`` accepts 1-D or 2-D ``base`` for ergonomics
# (1-D is degenerate-K=1 multi-base, equivalent to plain
# ``linear_residual``).
#
# Joint OLS guard: when K bases are near-collinear (e.g. ``TVT_prev``
# and ``TVT_prev_smoothed`` together), the design-matrix condition
# number explodes and αs become unstable. We compute condition number
# upfront and reject (returning the zero-alpha fallback) above a
# configurable threshold (default 30) so the spec downstream gets a
# safe identity-like inverse rather than blowing up at predict time.
# ----------------------------------------------------------------------

# Condition-number gate above which joint OLS is considered unstable
# and the multi-base transform falls back to zero-alpha + intercept.
# 30 is the conventional threshold (Belsley/Kuh/Welsch); above that,
# multicollinearity inflates standard errors enough that the alpha
# estimates carry no useful information. Exposed as module-level so
# tests can monkey-patch without recompiling.
_MULTI_BASE_COND_NUMBER_MAX: float = 30.0


# ----------------------------------------------------------------------
# linear_residual_grouped (#3 from R10c brainstorm).
#
# T = y - alpha_g * base - beta_g, where g = group(row). Alphas are fit
# per group via OLS, then shrunk toward the global alpha via empirical-
# Bayes (James-Stein style). The shrinkage protects small-n groups
# from over-fitted alphas while still letting large-n groups follow
# their own distinct AR dynamics.
#
# Use case (TVT): per-well α. Wells differ in autoregressive strength;
# one-α-across-wells is the wrong model. Per-well α + shrinkage handles
# the 5-row wells safely without losing the per-well structure that a
# global fit averages out.
#
# Failure handling:
# - At fit: small groups (n < min_group_size) skip OLS and use the
#   global α (i.e. fully shrunk to global).
# - At predict: unseen group IDs fall back to the global α.
# Both paths are silent + tracked in fitted_params for diagnostics.
# ----------------------------------------------------------------------

# Minimum rows per group below which the group falls back to the global
# fit. Belsley/Kuh/Welsch rule of thumb is "≥10× predictors" for stable
# OLS; for 1-base + intercept that's 20-30. We use 30 as a safe default;
# exposed as module-level so tests can monkey-patch.
_GROUPED_MIN_GROUP_SIZE: int = 30


# ----------------------------------------------------------------------
# quantile_residual (#6 from R10c brainstorm).
#
# Non-parametric, heteroscedasticity-aware residual. Bin ``base`` into ``n_bins`` quantile buckets on train; for each bucket compute median(y) and IQR(y). T = (y - median_bin) / IQR_bin. Inverse: y_hat = T_hat * IQR_bin + median_bin. The bucket-conditional centring + scaling makes T near-iid (mean 0, scale 1) regardless of how Var(y|base) depends on base -- which linear_residual cannot do.
#
# Use case (TVT): when Var(TVT|TVT_prev) scales with TVT_prev (heteroscedastic, common in flow / queue / well-log data), conditional-quantile residual produces a near-iid target which tree models predict more cleanly than a heteroscedastic linear residual.
#
# Failure handling:
# - Bins with < ``min_bin_n`` train rows reuse the GLOBAL median(y) / IQR(y) so under-populated bins don't carry noisy bin-local estimates. Tracked in ``params["bin_sizes"]`` for diagnostics.
# - At predict, ``base`` values outside the train range fall back to the EDGE bin (first or last), not a separate "out-of-range" bucket. Consistent with the quantile-binning contract.
# - IQR == 0 for a bin (constant y) is replaced with the global IQR or a small floor (1e-6 * std(y_train)) to keep the inverse well-defined.
# ----------------------------------------------------------------------

_QUANTILE_RESIDUAL_DEFAULT_N_BINS: int = 10
_QUANTILE_RESIDUAL_DEFAULT_MIN_BIN_N: int = 50


# ----------------------------------------------------------------------
# monotonic_residual (non-parametric monotonic-spline residual).
#
# T = y - g(base), where g is a monotonic PCHIP interpolant fitted to per-knot median(y) on quantile-based knots of base. Generalises linear_residual to capture saturating / sigmoidal / convex-concave relationships that the linear OLS leaves in the residual.
#
# Use case (TVT-style well-log): if TVT ~= a*TVT_prev grows linearly at low values but plateaus at high values, linear_residual leaves a wedge of curvature in T. Monotonic-spline residual sucks up the curvature so the inner model sees a near-iid T.
#
# PCHIP (Piecewise Cubic Hermite Interpolating Polynomial; scipy) is monotone-preserving by construction -- the interpolant between two adjacent knots is monotone if the knot values are monotone. We force per-knot y-values to be monotone (cumulative max or cumulative min depending on the train-data slope) so the spline g is monotone overall. This regularises the fit against noise-driven non-monotonicities at small per-knot sample sizes.
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# ewma_residual (R10c brainstorm #5a; exponentially-weighted moving average residual).
#
# T = y - EWMA_k(base), where EWMA_k(x_i) = (1 - alpha) * EWMA_k(x_{i-1}) + alpha * x_i with alpha = 2 / (k + 1) (standard half-life convention). Time-ordered: rows are assumed to be in chronological order (caller responsible). Inverse: y_hat = T_hat + EWMA_k(base), reusing the SAME EWMA recursion at predict time.
#
# Use case: lag-1 ``TVT_prev`` captures only one autoregressive horizon. EWMA(k=3..21) captures slow drift / regime persistence beyond a single lag. Adds transform diversity to the cross-target ensemble which NNLS exploits.
#
# Sequence contract: ``base`` must be in chronological order at both fit and predict. The transform DOES NOT receive a time-column kwarg; the caller (or wrapper-level orchestration) is responsible for ensuring row order matches time order. Out-of-order data produces a valid EWMA but with semantics the caller did not intend.
# ----------------------------------------------------------------------

_EWMA_RESIDUAL_DEFAULT_K: int = 7  # half-life-like span; alpha = 2 / (k + 1) ~= 0.25
_FRAC_DIFF_DEFAULT_D: float = 0.5  # Lopez de Prado standard order
_FRAC_DIFF_DEFAULT_LAGS: int = 30  # maximum weight tail used in the truncated series


# ----------------------------------------------------------------------
# rolling_quantile_ratio (R10c brainstorm #5b).
#
# T = y / max(RollingQ50_k(base), eps), where RollingQ50_k is the centred rolling median of ``base`` over a window of ``k`` rows. Inverse: y_hat = T_hat * RollingQ50_k(base). For the first ``k-1`` rows the window is left-truncated, falling back to the median of available rows.
#
# Use case: multiplicative DGP where the local-median of base sets the y-scale. Logratio captures global multiplicative structure but not local windowing. ``rolling_quantile_ratio`` is the localised version.
# ----------------------------------------------------------------------

_ROLLING_QUANTILE_DEFAULT_K: int = 7


def _rolling_quantile_ratio_fit(
    y: np.ndarray, base: np.ndarray, k: int = _ROLLING_QUANTILE_DEFAULT_K,
) -> dict[str, Any]:
    """Stores the window span ``k`` and an eps floor derived from train base scale to keep division safe at predict time on near-zero rolling medians."""
    k = max(1, int(k))
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    finite = np.isfinite(base_f) & (base_f != 0)
    scale = float(np.median(np.abs(base_f[finite]))) if finite.any() else 1.0
    eps = max(scale * 1e-6, 1e-12)
    return {"k": k, "eps": eps}


def _rolling_quantile_ratio_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    k = int(params["k"])
    eps = float(params["eps"])
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    roll_med = _rolling_median(base_f, k)
    safe = np.where(np.abs(roll_med) < eps, np.sign(roll_med + 1e-300) * eps, roll_med)
    return np.asarray(y, dtype=np.float64) / safe


def _rolling_quantile_ratio_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    k = int(params["k"])
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    roll_med = _rolling_median(base_f, k)
    return np.asarray(t_hat, dtype=np.float64) * roll_med


def _rolling_quantile_ratio_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1))
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1))


# ----------------------------------------------------------------------
# frac_diff (R10c brainstorm #5c; fractional differencing, Lopez de Prado).
#
# T = (1 - L)^d y where L is the lag operator and ``d`` is a fractional order in (0, 1). The expansion truncates to ``lags`` terms with weight w_k = -w_{k-1} * (d - k + 1) / k starting at w_0 = 1. Inverse: y_hat = T_hat + (sum_{k=1}^{lags} w_k * y_{i-k}) using the SAME weight series.
#
# Preserves long-memory while making the target stationary -- proven win in finance / queue-length regimes. The inverse re-uses the past y values to reconstruct the level. Note: this is a UNARY-y transform (``base`` is unused but accepted for signature uniformity).
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Registry and lookup
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Pack J: unary y-only transforms (no base column required).
# Implementation lives in composite_unary_transforms.py; we wrap the
# unary signatures (y, params) into the registry's (y, base, params)
# signature by ignoring the base arg.
# ----------------------------------------------------------------------
from .composite_unary_transforms import (  # noqa: E402
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


# ----------------------------------------------------------------------
# Pack K: chain factory. Build a registry-compatible Transform that
# composes a bivariate (y, base) -> T1 with a unary T1 -> T2.
# ----------------------------------------------------------------------


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
from ._composite_transforms_linear import (  # noqa: E402,F401
    _linear_residual_domain, _linear_residual_fit, _linear_residual_forward,
    _linear_residual_grouped_domain, _linear_residual_grouped_fit,
    _linear_residual_grouped_forward, _linear_residual_grouped_inverse,
    _linear_residual_inverse, _linear_residual_multi_domain,
    _linear_residual_multi_fit, _linear_residual_multi_forward,
    _linear_residual_multi_inverse, _linear_residual_robust_fit,
    _logratio_domain, _logratio_fit, _logratio_forward, _logratio_inverse,
)
from ._composite_transforms_nonlinear import (  # noqa: E402,F401
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
    # no ``base`` segment (e.g. ``TVT-cbrtY``) since there is no base.
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
            "single-base linear regression -- the production TVT-linres-TVT_prev case "
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


from types import MappingProxyType as _MappingProxyType

# Read-only view exported to callers; the underlying ``_TRANSFORMS_REGISTRY`` is module-private and any extension layer must edit it explicitly. Prevents test / extension code from pop-ing a transform and silently corrupting subsequent suite runs in the same process.
TRANSFORMS_REGISTRY: "Mapping[str, Transform]" = _MappingProxyType(_TRANSFORMS_REGISTRY)


def get_transform(name: str) -> Transform:
    """Lookup helper. Raises :exc:`UnknownTransformError` for typos."""
    try:
        return _TRANSFORMS_REGISTRY[name]
    except KeyError as exc:
        raise UnknownTransformError(
            f"Unknown transform '{name}'. Registered: {sorted(_TRANSFORMS_REGISTRY)}"
        ) from exc


# Short-name aliases for composite-target naming. Used in
# ``compose_target_name`` to keep displayed target names compact;
# previously composites were named ``TVT__linear_residual__TVT_prev``
# which read ugly in logs / report headings / dict keys. The dash
# separator + short aliases give us e.g. ``TVT-linres-TVT_prev``.
#
# Order: declared transforms only -- if a transform is missing from
# this map we fall back to the full name in ``compose_target_name`` so
# adding a new transform never silently breaks naming.
TRANSFORM_NAME_SHORT: dict[str, str] = {
    "diff": "diff",
    "additive_residual": "addres",
    "median_residual": "medres",
    "y_quantile_clip": "yqclip",
    "ratio": "ratio",
    "logratio": "logr",
    "linear_residual": "linres",
    "linear_residual_robust": "linresR",
    "linear_residual_multi": "linresM",
    "linear_residual_grouped": "linresG",
    "quantile_residual": "qres",
    "monotonic_residual": "monres",
    "ewma_residual": "ewma",
    "rolling_quantile_ratio": "rqr",
    "frac_diff": "fdiff",
    # Pack J unary y-transforms (no base segment in the composite name).
    "cbrt_y": "cbrtY",
    "log_y": "logY",
    "yeo_johnson_y": "yjY",
    "quantile_normal_y": "qnY",
    # Pack K chain transforms.
    "chain_linres_cbrt": "linresCbrt",
    "chain_linres_yj": "linresYj",
    "chain_monres_cbrt": "monresCbrt",
    "chain_monres_yj": "monresYj",
    "chain_linres_cbrt_qn": "linresCbrtQn",
}


def compose_target_name(target_col: str, transform_name: str, base: str) -> str:
    """Build the canonical composite-target name from its three components.

    Uses ``-`` as the separator and the short transform alias from
    ``TRANSFORM_NAME_SHORT``. Falls back to the full transform name if
    the alias is missing (so brand-new transforms get a longer-but-correct
    name on day one instead of silent collision).

    Examples:
        compose_target_name('TVT', 'linear_residual', 'TVT_prev')
            -> 'TVT-linres-TVT_prev'
        compose_target_name('TVT', 'monotonic_residual', 'Y')
            -> 'TVT-monres-Y'
    """
    short = TRANSFORM_NAME_SHORT.get(transform_name, transform_name)
    return f"{target_col}-{short}-{base}"


# Reverse-lookup pattern fragments: ``f"-{short}-"`` and ``f"-{full}-"``
# both appear as substrings in canonical composite-target names. Used by
# ``is_composite_target_name`` to detect "this target name came from
# discovery, NOT a user-supplied column called ``y-base`` that happens
# to have one dash".
_COMPOSITE_NAME_FRAGMENTS: frozenset = frozenset(
    f"-{alias}-" for alias in TRANSFORM_NAME_SHORT.values()
) | frozenset(
    f"-{full}-" for full in TRANSFORM_NAME_SHORT.keys()
)


def is_composite_target_name(name: str) -> bool:
    """True if ``name`` matches the canonical composite-target naming
    convention ``{target}-{transform_short}-{base}`` for any registered
    transform.

    Used by per-target metric / chart helpers to switch their label from
    ``MTTR`` (raw mean target) to ``MTRESID`` (residual mean ~= 0 by
    construction). Robust to both the post-2026-05-13 short-alias format
    AND the legacy ``{target}__{transform}__{base}`` double-underscore
    format -- so loading a v1 suite-pickle still routes correctly.
    """
    if not name:
        return False
    if any(frag in name for frag in _COMPOSITE_NAME_FRAGMENTS):
        return True
    # Legacy double-underscore format (older pickles).
    for full in TRANSFORM_NAME_SHORT.keys():
        if f"__{full}__" in name:
            return True
    return False


def list_transforms(*, tags: frozenset[str] | None = None) -> list[str]:
    """Return registered transform names, optionally filtered by tag
    intersection (any-of: a transform passes if it has at least one of
    the requested tags)."""
    if tags is None:
        return sorted(_TRANSFORMS_REGISTRY)
    return sorted(
        name for name, t in _TRANSFORMS_REGISTRY.items() if t.tags & tags
    )


