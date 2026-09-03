"""Non-linear residual + chain / EWMA / frac-diff / monotonic / quantile composite transforms carved out of ``mlframe.training.composite_transforms``.

Bound back into the parent's namespace via re-export at the parent's module bottom so historical ``from mlframe.training.composite_transforms import _monotonic_residual_fit`` resolves transparently.
"""
from __future__ import annotations

from ._domain_shared import residual_domain_reshaped

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
)

import numpy as np

try:
    import numba as _numba
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _numba = None
    _HAS_NUMBA = False

logger = logging.getLogger("mlframe.training.composite_transforms")

if TYPE_CHECKING:
    from . import Transform

# Parent-resident constants referenced as default-arg values in signatures below (the EWMA/frac-diff
# constants moved to _nonlinear_ewma_fracdiff.py along with the functions that default on them).
# Signature defaults evaluate at module load, so these MUST be top-level (a lazy in-body import
# wouldn't see them). The parent defines all five BEFORE its bottom-of-module sibling import, so this
# static cycle resolves at runtime. Whitelisted in tests/test_meta/test_no_import_cycles.py.
from . import (
    _QUANTILE_RESIDUAL_DEFAULT_N_BINS,
    _QUANTILE_RESIDUAL_DEFAULT_MIN_BIN_N,
    _MONOTONIC_RESIDUAL_DEFAULT_N_KNOTS,
    _MONOTONIC_RESIDUAL_DEFAULT_MIN_KNOT_N,
    _MONOTONIC_DEGENERACY_RATIO,
)

# EWMA / frac-diff transforms carved out to keep this file under the ~1000 LOC guideline; re-exported
# here so historical ``from .nonlinear import _ewma_kernel`` (and the composite_transforms public
# re-exports built on top of it) keep resolving. See _nonlinear_ewma_fracdiff.py's own docstring.
from ._nonlinear_ewma_fracdiff import (
    _ewma_kernel as _ewma_kernel,
    _ewma_kernel_njit_par_batched as _ewma_kernel_njit_par_batched,
    _frac_diff_inverse_kernel as _frac_diff_inverse_kernel,
    _frac_diff_inverse_kernel_njit_par_batched as _frac_diff_inverse_kernel_njit_par_batched,
    _ewma_residual_fit as _ewma_residual_fit,
    _ewma_anchor as _ewma_anchor,
    _ewma_compute as _ewma_compute,
    _ewma_force_backend as _ewma_force_backend,
    _frac_diff_inv_force_backend as _frac_diff_inv_force_backend,
    _lookup_ewma_backend as _lookup_ewma_backend,
    _lookup_frac_diff_inv_backend as _lookup_frac_diff_inv_backend,
    _ewma_dispatch as _ewma_dispatch,
    _ewma_compute_batched as _ewma_compute_batched,
    _frac_diff_inverse_compute as _frac_diff_inverse_compute,
    _frac_diff_inverse_dispatch as _frac_diff_inverse_dispatch,
    _frac_diff_inverse_compute_batched as _frac_diff_inverse_compute_batched,
    _ewma_residual_forward as _ewma_residual_forward,
    _ewma_residual_inverse as _ewma_residual_inverse,
    _ewma_residual_domain as _ewma_residual_domain,
    _rolling_median_pandas as _rolling_median_pandas,
    _rolling_median as _rolling_median,
    _frac_diff_weights as _frac_diff_weights,
    _frac_diff_fit as _frac_diff_fit,
    _frac_diff_forward as _frac_diff_forward,
    _frac_diff_inverse as _frac_diff_inverse,
    _frac_diff_domain as _frac_diff_domain,
)

# Module-level numba kernels (JIT compile on first call); pure-Python fallback is the in-line recursion below when numba is absent. The EWMA / frac-diff-inverse recurrence kernels live in the ``_nonlinear_ewma_fracdiff`` sibling (re-exported below); only the quantile-bin kernel stays here.
if _HAS_NUMBA:

    @_numba.njit(cache=True, parallel=True)
    def _quantile_assign_bins_kernel(base_f: np.ndarray, inner_edges: np.ndarray, n_bins: int) -> np.ndarray:
        """Parallel linear-scan equivalent of ``np.clip(np.searchsorted(inner_edges, base_f, "right"), 0, n_bins-1)``.

        ``inner_edges`` (the n_bins-1 ascending cut points) is tiny, so a branch-light forward count beats a per-element binary search and avoids the separate ``np.clip`` pass; ``prange`` then scales it across cores. Bit-identical to searchsorted including the NaN edge (NaN sorts as +inf -> top bin) and +/-inf. Bench: 3.9x@10k / 6.6x@200k / 8.9x@1M (bench_quantile_assign_bins_searchsorted.py).
        """
        n = base_f.size
        out = np.empty(n, dtype=np.intp)
        m = inner_edges.size
        for i in _numba.prange(n):
            x = base_f[i]
            if x != x:  # NaN: np.searchsorted sorts it as +inf -> top bin after clip
                out[i] = n_bins - 1
                continue
            b = 0
            for j in range(m):
                if inner_edges[j] <= x:
                    b += 1
                else:
                    break
            if b > n_bins - 1:
                b = n_bins - 1
            out[i] = b
        return out
else:
    _quantile_assign_bins_kernel = None


logger = logging.getLogger("mlframe.training.composite_transforms")

if TYPE_CHECKING:
    from . import Transform

def _james_stein_shrinkage_factor(
    per_group_alphas: np.ndarray,
    global_alpha: float,
    group_sizes: np.ndarray,
    sigma2_total: float,
    base_vars: np.ndarray | None = None,
) -> float:
    """Estimate the James-Stein shrinkage factor toward ``global_alpha``.

    Returns a scalar c ∈ [0, 1]: c=0 keeps per-group alphas as-is (no shrinkage); c=1 collapses all per-group alphas to global_alpha (full shrinkage).

    The classic JS estimator for K estimators ``θ_g`` with known sampling variance σ²_g is ``c = max(0, (K - 3) · mean_g(σ²_g) / Σ_g (θ_g - global)²)`` (clamped to [0, 1]).

    Here the shrunk estimators are per-group OLS *slopes* ``α_g``, whose sampling variance is ``Var(α_g) = σ² / (n_g · Var(base_g))`` -- NOT ``σ² / n_g``. The ``Var(base_g)`` term is essential: it makes the JS factor SCALE-INVARIANT. Rescale ``base`` by a factor ``s`` and every ``α_g`` scales by ``1/s`` (so ``Σ (α_g - global)²`` scales by ``1/s²``); the correct noise proxy ``σ² / (n_g · Var(base_g))`` ALSO scales by ``1/s²`` (since ``Var(base_g)`` scales by ``s²``), leaving ``c`` unchanged. Dropping ``Var(base_g)`` (the historic ``base_vars=None`` path) leaves the numerator fixed while the denominator moves with the unit, so the SAME data on a different ``base`` unit shrinks a different set of groups -- a unit-dependent bug.

    Pass ``base_vars`` = per-group ``Var(base_g)`` (aligned 1:1 with ``per_group_alphas`` / ``group_sizes``) to get the correct, scale-invariant slope-variance proxy. When ``base_vars`` is ``None`` the legacy size-only proxy (``σ² / mean(n_g)``) is used for backward compatibility; callers shrinking OLS slopes should always supply it.

    A degenerate case (K < 4 groups, or all alphas equal) returns c=0 so the JS correction can't reduce K below the JS-applicability threshold; the per-group estimates pass through unmodified.
    """
    k = per_group_alphas.size
    if k < 4:
        return 0.0
    deviations = per_group_alphas - global_alpha
    sum_sq = float(np.sum(deviations * deviations))
    if sum_sq <= 0:
        return 0.0
    sizes = np.asarray(group_sizes, dtype=np.float64).reshape(-1)
    if base_vars is not None:
        # Correct slope-variance proxy: Var(α_g) = σ² / (n_g · Var(base_g)).
        # Average over the K shrunk groups -> mean_g(σ²_g). Var(base_g) below a
        # tiny floor (a near-constant base inside a group) would blow the proxy
        # up; floor it so a single degenerate group can't force full shrinkage.
        bvar = np.asarray(base_vars, dtype=np.float64).reshape(-1)
        denom = np.maximum(sizes, 1.0) * np.maximum(bvar, 1e-12)
        per_group_variance = sigma2_total / denom
        mean_per_group_variance = float(np.mean(per_group_variance))
    else:
        # Legacy (unit-dependent) proxy: σ²_per_group ≈ σ²_total / mean(n_g).
        # Retained only for callers that predate the scale-invariant fix.
        mean_per_group_variance = float(sigma2_total / max(float(np.mean(sizes)), 1.0))
    # Classic JS factor c in α_shrunk = (1-c) α_g + c α_global; c = (K-3) · mean_g(σ²_g) / Σ_g (α_g - α_global)², clamped to [0, 1]. High noise / low spread => c->1 (full shrink); low noise / high spread => c->0 (keep per-group).
    raw = (k - 3) * mean_per_group_variance / sum_sq
    return float(max(0.0, min(1.0, raw)))
def _row_alpha_beta(
    groups: np.ndarray, params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Materialise per-row (alpha, beta) from the grouped params dict.

    Vectorised: ``np.unique`` collapses the n-row groups vector to K unique labels, looks them up in the params dict ONCE per unique label, then inverse-indexes to broadcast back to n rows. A naive ``for i, g in enumerate(groups)`` is ~30x slower on 200K rows; cProfile measured the loop at 88% of total fit+predict cost pre-optimisation. Unseen group labels (at predict but not fit) fall back to global alpha/beta -- a safe identity-like inverse.
    """
    alpha_global = float(params["alpha_global"])
    beta_global = float(params["beta_global"])
    pg_alphas = params["per_group_alphas"]
    pg_betas = params["per_group_betas"]
    # K unique labels; inv maps each row to an index into uniq. Per-unique-label alpha / beta built with global as fallback.
    # Canonical key matches the grouped-fit keying so int<->float dtype drift at
    # predict does not miss every group and silently fall back to global alpha/beta.
    from . import _canonical_group_key
    uniq, inv = np.unique(groups, return_inverse=True)
    uniq_alpha = np.array(
        [pg_alphas.get(_canonical_group_key(g), alpha_global) for g in uniq],
        dtype=np.float64,
    )
    uniq_beta = np.array(
        [pg_betas.get(_canonical_group_key(g), beta_global) for g in uniq],
        dtype=np.float64,
    )
    return uniq_alpha[inv], uniq_beta[inv]
def _quantile_residual_per_bin_stats_v1_pyloop(
    y_clean: np.ndarray, bin_idx: np.ndarray, actual_n_bins: int,
    min_bin_n: int, global_median: float, global_iqr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference: build per-bin boolean masks; under-populated bins keep global fallback."""
    bin_medians = np.full(actual_n_bins, global_median, dtype=np.float64)
    bin_iqrs = np.full(actual_n_bins, global_iqr, dtype=np.float64)
    bin_sizes = np.zeros(actual_n_bins, dtype=np.int64)
    for b in range(actual_n_bins):
        mask = bin_idx == b
        bin_n = int(mask.sum())
        bin_sizes[b] = bin_n
        if bin_n < min_bin_n:
            continue
        bin_y = y_clean[mask]
        bin_medians[b] = float(np.median(bin_y))
        bin_iqr = float(np.subtract(*np.percentile(bin_y, [75, 25])))
        bin_iqrs[b] = bin_iqr if bin_iqr > 1e-6 else global_iqr
    return bin_medians, bin_iqrs, bin_sizes


def _quantile_residual_per_bin_stats_v2_pandas_groupby(
    y_clean: np.ndarray, bin_idx: np.ndarray, actual_n_bins: int,
    min_bin_n: int, global_median: float, global_iqr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised via pandas groupby quantile([.25, .5, .75]); under-populated bins (count < min_bin_n) are reset to global fallback to match the v1 semantics exactly."""
    import pandas as _pd
    bin_medians = np.full(actual_n_bins, global_median, dtype=np.float64)
    bin_iqrs = np.full(actual_n_bins, global_iqr, dtype=np.float64)
    bin_sizes = np.zeros(actual_n_bins, dtype=np.int64)
    ser = _pd.Series(y_clean)
    gb = ser.groupby(bin_idx, sort=True)
    counts = gb.count()
    qs = gb.quantile([0.25, 0.5, 0.75]).unstack()
    idx = counts.index.to_numpy()
    bin_sizes[idx] = counts.to_numpy()
    keep = counts.to_numpy() >= min_bin_n
    if keep.any():
        kept_idx = idx[keep]
        q25 = qs[0.25].to_numpy()[keep]
        q50 = qs[0.5].to_numpy()[keep]
        q75 = qs[0.75].to_numpy()[keep]
        bin_medians[kept_idx] = q50
        raw_iqr = q75 - q25
        bin_iqrs[kept_idx] = np.where(raw_iqr > 1e-6, raw_iqr, global_iqr)
    return bin_medians, bin_iqrs, bin_sizes


def _quantile_residual_per_bin_stats(
    y_clean: np.ndarray, bin_idx: np.ndarray, actual_n_bins: int,
    min_bin_n: int, global_median: float, global_iqr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Size-aware dispatcher across per-bin quantile-stats variants. Bench (bench_median_quantile_residual.py): v2 pandas-groupby wins at small n / large n_bins (n=100k+20bins: 1.72x over v1) and ties / loses on large n with few bins, so route to v2 when ``y_clean.size <= 200_000`` else v1. Sort-based numba variant tried and rejected (extra argsort dominated -- see bench-attempt-rejected note in bench_median_quantile_residual.py)."""
    if y_clean.size <= 200_000:
        try:
            return _quantile_residual_per_bin_stats_v2_pandas_groupby(
                y_clean, bin_idx, actual_n_bins, min_bin_n, global_median, global_iqr,
            )
        except Exception as _exc:
            logger.warning("composite_transforms: pandas-groupby fast path failed (%s); using numpy fallback.", _exc)
    return _quantile_residual_per_bin_stats_v1_pyloop(
        y_clean, bin_idx, actual_n_bins, min_bin_n, global_median, global_iqr,
    )


def _quantile_residual_fit(
    y: np.ndarray, base: np.ndarray,
    n_bins: int = _QUANTILE_RESIDUAL_DEFAULT_N_BINS,
    min_bin_n: int = _QUANTILE_RESIDUAL_DEFAULT_MIN_BIN_N,
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit per-bucket median(y) + IQR(y) over ``n_bins`` quantile bins of ``base``.

    Returns a dict with keys: ``bin_edges`` (1-D ndarray len ``n_bins+1``, open at -inf, +inf), ``bin_medians`` (len ``n_bins``; median(y) per bin, global median for under-populated bins), ``bin_iqrs`` (len ``n_bins``; IQR(y) per bin, global IQR with floor for under-populated / constant bins), ``bin_sizes`` (list[int] len ``n_bins``, train rows per bin), ``global_median``/``global_iqr`` (float fallbacks from train y), ``n_bins`` (int, recorded for predict-time validation).
    """
    # Lazy import: ``.predict`` re-imports this sibling at its bottom, so a top-level ``from .predict import ...`` would create a hard cycle the meta-test flags.
    n_bins = max(2, int(n_bins))
    min_bin_n = max(2, int(min_bin_n))
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    finite = _finite_mask if _finite_mask is not None else (np.isfinite(y_f) & np.isfinite(base_f))
    if finite.sum() < n_bins * 2:
        # Degenerate: fall back to global stats so the inverse is still safe.
        med = float(np.median(y_f[finite])) if finite.any() else 0.0
        iqr_v = float(np.subtract(*np.percentile(y_f[finite], [75, 25]))) if finite.sum() >= 4 else 1.0
        iqr_v = max(iqr_v, 1e-6)
        return {
            "bin_edges": np.array([-np.inf, np.inf], dtype=np.float64),
            "bin_medians": np.array([med], dtype=np.float64),
            "bin_iqrs": np.array([iqr_v], dtype=np.float64),
            "bin_sizes": [int(finite.sum())],
            "global_median": med,
            "global_iqr": iqr_v,
            "n_bins": 1,
        }
    y_clean = y_f[finite]
    base_clean = base_f[finite]
    # Quantile edges on train base; ``np.quantile`` with linspace covers the open-open envelope, and the outermost edges become +/-inf below so predict-time digitize never produces an out-of-range bucket.
    inner_qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(base_clean, inner_qs)
    # Deduplicate edges (ties at one quantile collapse several edges, else empty bins emerge); tolerate up to n_bins-1 unique edges, clip n_bins downstream.
    edges = np.unique(edges)
    if edges.size < 2:
        # All base values identical: degenerate single bucket.
        med = float(np.median(y_clean))
        iqr_v = max(float(np.subtract(*np.percentile(y_clean, [75, 25]))), 1e-6)
        return {
            "bin_edges": np.array([-np.inf, np.inf], dtype=np.float64),
            "bin_medians": np.array([med], dtype=np.float64),
            "bin_iqrs": np.array([iqr_v], dtype=np.float64),
            "bin_sizes": [int(y_clean.size)],
            "global_median": med,
            "global_iqr": iqr_v,
            "n_bins": 1,
        }
    edges[0] = -np.inf
    edges[-1] = np.inf
    actual_n_bins = edges.size - 1
    # Global stats: fallback for under-populated bins.
    global_median = float(np.median(y_clean))
    global_iqr = max(float(np.subtract(*np.percentile(y_clean, [75, 25]))), 1e-6)
    # Per-bin assignment via np.searchsorted (right-side: edges[i-1] <= x < edges[i]).
    bin_idx = np.clip(np.searchsorted(edges[1:-1], base_clean, side="right"), 0, actual_n_bins - 1)
    bin_medians, bin_iqrs, bin_sizes_arr = _quantile_residual_per_bin_stats(
        y_clean, bin_idx, actual_n_bins, min_bin_n, global_median, global_iqr,
    )
    bin_sizes: list[int] = bin_sizes_arr.tolist()
    return {
        "bin_edges": edges,
        "bin_medians": bin_medians,
        "bin_iqrs": bin_iqrs,
        "bin_sizes": bin_sizes,
        "global_median": global_median,
        "global_iqr": global_iqr,
        "n_bins": int(actual_n_bins),
    }
def _quantile_residual_assign_bins(base: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map each row of ``base`` to a bin index in [0, n_bins-1]. Out-of-range values map to the edge bin (no separate OOR bucket), per the contract documented on the transform."""
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    n_bins = edges.size - 1
    if n_bins <= 1:
        return np.zeros(base_f.size, dtype=np.intp)
    # ``edges[1:-1]`` are the INNER cut points; searchsorted returns 0..n_bins.
    if _HAS_NUMBA:
        return np.asarray(_quantile_assign_bins_kernel(
            np.ascontiguousarray(base_f), np.ascontiguousarray(edges[1:-1]), n_bins,
        ))
    bin_idx = np.searchsorted(edges[1:-1], base_f, side="right")
    return np.clip(bin_idx, 0, n_bins - 1)
def _quantile_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Apply ``T = (y - median_bin(base)) / IQR_bin(base)``: bin-conditional median-centring scaled by the bin's IQR."""
    edges = np.asarray(params["bin_edges"], dtype=np.float64)
    medians = np.asarray(params["bin_medians"], dtype=np.float64)
    iqrs = np.asarray(params["bin_iqrs"], dtype=np.float64)
    bin_idx = _quantile_residual_assign_bins(base, edges)
    return np.asarray((np.asarray(y, dtype=np.float64) - medians[bin_idx]) / iqrs[bin_idx])
def _quantile_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Undo the transform: ``y = T_hat * IQR_bin(base) + median_bin(base)``."""
    edges = np.asarray(params["bin_edges"], dtype=np.float64)
    medians = np.asarray(params["bin_medians"], dtype=np.float64)
    iqrs = np.asarray(params["bin_iqrs"], dtype=np.float64)
    bin_idx = _quantile_residual_assign_bins(base, edges)
    return np.asarray(np.asarray(t_hat, dtype=np.float64) * iqrs[bin_idx] + medians[bin_idx])
_quantile_residual_domain: Callable[[Optional[np.ndarray], np.ndarray], np.ndarray] = residual_domain_reshaped
def _spearman_sign(a: np.ndarray, b: np.ndarray) -> int:
    """Return +1 / -1: the sign of the Spearman rank correlation between ``a`` and ``b``.

    Equivalent (in sign) to ``np.sign(scipy.stats.spearmanr(a, b).statistic)`` but without
    computing the magnitude. Ranks are ordinal (``argsort``-of-``argsort``, ties broken
    positionally) rather than scipy's tie-averaged ranks; tie-averaging rescales the rank
    vectors but cannot change the sign of their covariance, so the returned direction is
    identical to scipy's on continuous and tied data alike. A non-positive covariance
    (including the degenerate constant-input case where it is exactly 0) maps to +1,
    matching the legacy ``rho >= 0 -> increasing`` / ``rho is None -> increasing`` rule.
    """
    n = a.size
    ra = np.empty(n, dtype=np.float64)
    ra[np.argsort(a, kind="stable")] = np.arange(n, dtype=np.float64)
    rb = np.empty(n, dtype=np.float64)
    rb[np.argsort(b, kind="stable")] = np.arange(n, dtype=np.float64)
    # Mean-centred rank covariance; sign matches Spearman rho's sign.
    ra -= ra.mean()
    rb -= rb.mean()
    cov = float(np.dot(ra, rb))
    return 1 if cov >= 0.0 else -1


def _monotonic_residual_fit(
    y: np.ndarray, base: np.ndarray,
    n_knots: int = _MONOTONIC_RESIDUAL_DEFAULT_N_KNOTS,
    min_knot_n: int = _MONOTONIC_RESIDUAL_DEFAULT_MIN_KNOT_N,
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit a monotone PCHIP spline g(base) via per-quantile-knot medians and orient by the sign of the global Spearman correlation between y and base. Stores the knot x/y arrays + the global y mean as a fallback. Domain at predict time: base values outside [knots_x[0], knots_x[-1]] are clipped to the edge knots (PCHIP extrapolation is not safe -- it can run off to +/- inf rapidly).

    Auto-knot tuning: when ``base`` has few unique values (categorical / discrete), the default knots oversmooth -- several quantile knots collapse to identical x positions, leaving < n_eff effective knots and a wobbly spline that often goes degenerate. The cap is driven by the base's *distinctness*, NOT its row count: at most ``n_unique_base`` distinct quantile knots can be placed (beyond that, ties collapse them), so ``n_knots`` is capped at ``min(n_knots, n_unique_base)`` (with a floor of 3). A continuous base keeps the full default regardless of n -- the historic ``n_unique_base // 200`` rule wrongly conflated cardinality with discreteness and starved continuous mid-/small-n bases (e.g. 600 distinct continuous values -> 3 knots) of resolution they could use.
    """
    # Lazy import: ``.predict`` re-imports this sibling at its bottom, so a top-level ``from .predict import ...`` would create a hard cycle the meta-test flags.
    base_f_for_unique = np.asarray(base, dtype=np.float64).reshape(-1)
    _n_unique_base = int(np.unique(base_f_for_unique[np.isfinite(base_f_for_unique)]).size)
    # Cap by distinctness only: a base with K distinct values supports at most K
    # distinct quantile knots. Continuous bases (K >= n_knots) keep the full
    # requested count; only genuinely discrete / low-cardinality bases reduce.
    _auto_knots = _n_unique_base if _n_unique_base else n_knots
    n_knots = min(int(n_knots), _auto_knots)
    n_knots = max(3, int(n_knots))
    min_knot_n = max(2, int(min_knot_n))
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    finite = _finite_mask if _finite_mask is not None else (np.isfinite(y_f) & np.isfinite(base_f))
    if finite.sum() < n_knots * 2:
        y_med = float(np.median(y_f[finite])) if finite.any() else 0.0
        return {
            "knots_x": np.array([0.0, 1.0], dtype=np.float64),
            "knots_y": np.array([y_med, y_med], dtype=np.float64),
            "y_train_mean": y_med,
            "monotone_direction": 0,
            "n_knots_effective": 2,
            "is_degenerate": True,
            "var_explained": 0.0,
        }
    y_clean = y_f[finite]
    base_clean = base_f[finite]
    # Knot x positions on quantile cuts of base (NOT linearly-spaced; uneven base distributions benefit from quantile placement). Deduplicate ties (many identical base values collapse to fewer knots).
    qs = np.linspace(0.0, 1.0, n_knots)
    knots_x = np.quantile(base_clean, qs)
    knots_x = np.unique(knots_x)
    if knots_x.size < 3:
        y_med = float(np.median(y_clean))
        return {
            "knots_x": np.array([base_clean.min(), base_clean.max()], dtype=np.float64),
            "knots_y": np.array([y_med, y_med], dtype=np.float64),
            "y_train_mean": y_med,
            "monotone_direction": 0,
            "n_knots_effective": 2,
            "is_degenerate": True,
            "var_explained": 0.0,
        }
    # Per-knot y values: median(y) for rows assigned to each knot's quantile slab. Slab boundaries are midpoints between adjacent knots (left/right edges extend to +/-inf so every row maps to a slab).
    n_eff = knots_x.size
    knots_y = np.empty(n_eff, dtype=np.float64)
    y_global_med = float(np.median(y_clean))
    slab_edges = np.empty(n_eff + 1, dtype=np.float64)
    slab_edges[0] = -np.inf
    slab_edges[-1] = np.inf
    slab_edges[1:-1] = 0.5 * (knots_x[:-1] + knots_x[1:])
    slab_idx = np.clip(np.searchsorted(slab_edges[1:-1], base_clean, side="right"), 0, n_eff - 1)
    for k in range(n_eff):
        mask = slab_idx == k
        n_in_slab = int(mask.sum())
        if n_in_slab < min_knot_n:
            knots_y[k] = y_global_med
        else:
            knots_y[k] = float(np.median(y_clean[mask]))
    # Orient monotonicity by the SIGN of the Spearman correlation between y and base;
    # tie -> increasing (arbitrary but stable). Only the sign is consumed (it flips the
    # orientation, never scales it), so the full scipy.stats.spearmanr (tie-averaged
    # rankdata + Pearson-on-ranks, dominated by two argsorts plus rankdata machinery)
    # is wasted work. ``_spearman_sign`` computes the sign via ordinal ranks
    # (argsort-of-argsort) + a rank covariance: ~1.38x faster than spearmanr on the
    # (base, y) shape this fit sees, sign-identical to scipy across continuous AND tied
    # data (tie-averaging shifts rank magnitudes but never the covariance sign).
    if y_clean.size >= 3 and base_clean.size >= 3:
        try:
            direction = _spearman_sign(base_clean, y_clean)
        except Exception as e:
            logger.debug("_spearman_sign failed, defaulting to positive orientation: %s", e)
            direction = 1
    else:
        direction = 1
    # Enforce monotonicity by cumulative max / min over knots in the orientation direction; protects against per-knot median noise creating local non-monotonicities PCHIP would otherwise honour (PCHIP is monotone PER SEGMENT but only if the knot values are monotone overall).
    if direction == 1:
        knots_y = np.maximum.accumulate(knots_y)
    else:
        knots_y = np.minimum.accumulate(knots_y)
    # Degeneracy detection: measure the actual variance reduction g(base) provides on the TRAIN sample. The composite T = y - g(base) is useful iff g captures a non-trivial fraction of y's variance (``var_explained = 1 - var(T) / var(y)``). When < ``_MONOTONIC_DEGENERACY_RATIO`` the spline is noise / a near-constant fit -- downstream models on T produce SAME predictions as on raw y (observed in prod: CB/XGB/LGB MAE identical to raw on a monres-Y spec). Surface the degeneracy so discovery can drop the spec early instead of paying for full training that produces no win.
    _y_var = float(np.var(y_clean)) if y_clean.size > 1 else 0.0
    if _y_var > 0.0:
        # Reconstruct g(base_clean) via the same PCHIP helper the inverse path uses and measure var(y - g), keeping semantics aligned with the actual transform.
        _g_train = _monotonic_residual_g(
            base_clean,
            {
                "knots_x": knots_x, "knots_y": knots_y,
                "y_train_mean": float(np.mean(y_clean)),
                "monotone_direction": direction,
            },
        )
        _t_train = y_clean - _g_train
        _var_explained = max(0.0, 1.0 - float(np.var(_t_train)) / _y_var)
    else:
        _var_explained = 0.0
    _is_degenerate = _var_explained < _MONOTONIC_DEGENERACY_RATIO
    return {
        "knots_x": knots_x,
        "knots_y": knots_y,
        "y_train_mean": float(np.mean(y_clean)),
        "monotone_direction": direction,
        "n_knots_effective": int(n_eff),
        "is_degenerate": _is_degenerate,
        "var_explained": _var_explained,
    }
def _monotonic_residual_g(base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Evaluate the monotone PCHIP interpolant at the requested base values. Out-of-range values clip to the edge knot value (NOT extrapolated)."""
    knots_x = np.asarray(params["knots_x"], dtype=np.float64)
    knots_y = np.asarray(params["knots_y"], dtype=np.float64)
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    if knots_x.size < 2:
        return np.full(base_f.shape, float(params.get("y_train_mean", 0.0)), dtype=np.float64)
    if knots_x.size == 2:
        # Degenerate: linear interpolation between the two anchor knots; out-of-range clamps to edge value.
        clipped = np.clip(base_f, knots_x[0], knots_x[-1])
        slope = (knots_y[-1] - knots_y[0]) / max(knots_x[-1] - knots_x[0], 1e-12)
        return np.asarray(knots_y[0] + slope * (clipped - knots_x[0]))
    from scipy.interpolate import PchipInterpolator  # lazy
    # extrapolate=False yields NaN outside [x[0], x[-1]]; fill those with the edge knot values to keep predict-time well-defined.
    interp = PchipInterpolator(knots_x, knots_y, extrapolate=False)
    out = interp(base_f)
    if np.any(~np.isfinite(out)):
        low_mask = base_f < knots_x[0]
        high_mask = base_f > knots_x[-1]
        out[low_mask] = knots_y[0]
        out[high_mask] = knots_y[-1]
    return np.asarray(out)
def _monotonic_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Apply ``T = y - g(base)`` where ``g`` is the fitted monotone PCHIP spline."""
    return np.asarray(np.asarray(y, dtype=np.float64) - _monotonic_residual_g(base, params))
def _monotonic_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Undo the transform: ``y = T_hat + g(base)``."""
    return np.asarray(np.asarray(t_hat, dtype=np.float64) + _monotonic_residual_g(base, params))
_monotonic_residual_domain: Callable[[Optional[np.ndarray], np.ndarray], np.ndarray] = residual_domain_reshaped
def _delegate_domain_check(bivariate_domain):
    """Domain-check factory shared by ``_make_chain_transform`` / ``_make_multi_chain_transform``:
    delegates entirely to the bivariate half, since the unary stage(s) have no base-dependent constraint."""

    def _domain(y, base):
        """Delegate to the bivariate domain check bound by the enclosing factory."""
        return bivariate_domain(y, base)

    return _domain


def _make_chain_transform(
    *, name: str, short_name: str,
    bivariate_fit, bivariate_forward, bivariate_inverse, bivariate_domain,
    unary_fit, unary_forward, unary_inverse,
    description: str,
) -> "Transform":
    """Create a registry Transform for ``chain(bivariate, unary)``.

    The chain inherits ``requires_base=True`` from the bivariate half (it still needs a base column at fit + predict). At fit-time it first fits the bivariate, applies forward to get T1, then fits the unary on T1; the joint params dict stores both. Forward / inverse run in the matching order. Domain check delegates to the bivariate's check since the unary half has no base-dependent constraint at predict.
    """
    # Lazy import: ``.predict`` re-imports this sibling at its bottom, so a top-level ``from .predict import ...`` would create a hard cycle the meta-test flags.
    from . import TAG_EXTENDED, TAG_REGRESSION, Transform, _chain_fit_raw, _chain_forward_raw, _chain_inverse_raw
    unary_tup = (unary_fit, unary_forward, unary_inverse)

    def _fit(y, base):
        """Fit the bivariate half then the unary half on its output, per :func:`_chain_fit_raw`."""
        return _chain_fit_raw(
            y=y, base=base,
            bivariate_fit=bivariate_fit,
            bivariate_forward=bivariate_forward,
            unary=unary_tup,
        )

    def _forward(y, base, params):
        """Apply bivariate forward then unary forward, per :func:`_chain_forward_raw`."""
        return _chain_forward_raw(
            y=y, base=base, params=params,
            bivariate_forward=bivariate_forward,
            unary=unary_tup,
        )

    def _inverse(t_hat, base, params):
        """Undo unary then bivariate (reverse order), per :func:`_chain_inverse_raw`."""
        return _chain_inverse_raw(
            t2=t_hat, base=base, params=params,
            bivariate_inverse=bivariate_inverse,
            unary=unary_tup,
        )

    _domain = _delegate_domain_check(bivariate_domain)

    return Transform(
        name=name,
        forward=_forward,
        inverse=_inverse,
        fit=_fit,
        domain_check=_domain,
        description=description,
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    )
def _make_multi_chain_transform(
    *, name: str, short_name: str,
    bivariate_fit, bivariate_forward, bivariate_inverse, bivariate_domain,
    unary_stages: list,
    description: str,
):
    """Multi-stage chain: bivariate + N unary stages. ``unary_stages`` is a list of ``(fit, forward, inverse)`` tuples; each runs in order at forward, in reverse at inverse. Used to register e.g. ``chain([linres, cbrt, quantile_normal])`` for very heavy-tail residuals."""
    # Lazy import: ``.predict`` re-imports this sibling at its bottom, so a top-level ``from .predict import ...`` would create a hard cycle the meta-test flags.
    from . import TAG_EXTENDED, TAG_REGRESSION, Transform, _chain_multi_fit_raw, _chain_multi_forward_raw, _chain_multi_inverse_raw

    def _fit(y, base):
        """Fit the bivariate half then each unary stage in order on its predecessor's output, per :func:`_chain_multi_fit_raw`."""
        return _chain_multi_fit_raw(
            y=y, base=base,
            bivariate_fit=bivariate_fit,
            bivariate_forward=bivariate_forward,
            unary_stages=unary_stages,
        )

    def _forward(y, base, params):
        """Apply bivariate forward then every unary stage in order, per :func:`_chain_multi_forward_raw`."""
        return _chain_multi_forward_raw(
            y=y, base=base, params=params,
            bivariate_forward=bivariate_forward,
            unary_stages=unary_stages,
        )

    def _inverse(t_hat, base, params):
        """Undo the unary stages in reverse order, then the bivariate half, per :func:`_chain_multi_inverse_raw`."""
        return _chain_multi_inverse_raw(
            t_final=t_hat, base=base, params=params,
            bivariate_inverse=bivariate_inverse,
            unary_stages=unary_stages,
        )

    _domain = _delegate_domain_check(bivariate_domain)

    return Transform(
        name=name,
        forward=_forward,
        inverse=_inverse,
        fit=_fit,
        domain_check=_domain,
        description=description,
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    )
