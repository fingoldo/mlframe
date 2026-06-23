"""Non-linear residual + chain / EWMA / frac-diff / monotonic / quantile composite transforms carved out of ``mlframe.training.composite_transforms``.

Bound back into the parent's namespace via re-export at the parent's module bottom so historical ``from mlframe.training.composite_transforms import _monotonic_residual_fit`` resolves transparently.
"""
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple,
)

import numpy as np

try:
    import numba as _numba  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _numba = None  # type: ignore
    _HAS_NUMBA = False

logger = logging.getLogger("mlframe.training.composite_transforms")

# Parent-resident constants referenced as default-arg values in signatures below. Signature defaults evaluate at module load, so these MUST be top-level (a lazy in-body import wouldn't see them). The parent defines all five BEFORE its bottom-of-module sibling import, so this static cycle resolves at runtime. Whitelisted in tests/test_meta/test_no_import_cycles.py.
from . import (  # noqa: E402
    _QUANTILE_RESIDUAL_DEFAULT_N_BINS,
    _QUANTILE_RESIDUAL_DEFAULT_MIN_BIN_N,
    _MONOTONIC_RESIDUAL_DEFAULT_N_KNOTS,
    _MONOTONIC_RESIDUAL_DEFAULT_MIN_KNOT_N,
    _MONOTONIC_DEGENERACY_RATIO,
    _EWMA_RESIDUAL_DEFAULT_K,
    _FRAC_DIFF_DEFAULT_D,
    _FRAC_DIFF_DEFAULT_LAGS,
)


# Module-level numba kernels (JIT compile on first call); pure-Python fallback is the in-line recursion below when numba is absent.
# Backend ladder: EWMA + frac-diff-inverse are LEFT-RECURRENT in row order (out[i] = f(out[i-1], ...)) so prange over rows is impossible; the win comes from a BATCHED kernel (K, N) parallelising across K specs while each row recurrence stays serial. CUDA RawKernel (one block per spec) tried and rejected: sequential per-thread recurrence is bandwidth-bound + host-device transfer kills it (5-100x SLOWER than njit at every size, see _benchmarks/_results/bench_ewma_frac_diff_backends_*.json). Two backends retained: single-spec njit (production) + parallel-batched njit.
if _HAS_NUMBA:

    @_numba.njit(cache=True)
    def _ewma_kernel(base_f: np.ndarray, alpha: float, anchor: float) -> np.ndarray:
        """v1 single-spec EWMA recurrence kernel; production default for K=1 path."""
        n = base_f.size
        out = np.empty(n, dtype=np.float64)
        state = anchor
        for i in range(n):
            x = base_f[i]
            if np.isfinite(x):
                state = (1.0 - alpha) * state + alpha * x
            out[i] = state
        return out

    @_numba.njit(cache=True, parallel=True)
    def _ewma_kernel_njit_par_batched(
        base_batch: np.ndarray, alphas: np.ndarray, anchors: np.ndarray,
    ) -> np.ndarray:
        """v2 batched EWMA: K specs in parallel via prange over the spec axis. Each row-recurrence inside one spec stays serial (left-recurrence). Bench: 2.7-3.8x over per-spec v1 at K>=10, N>=100k."""
        K, N = base_batch.shape
        out = np.empty((K, N), dtype=np.float64)
        for s in _numba.prange(K):
            state = anchors[s]
            a = alphas[s]
            for i in range(N):
                x = base_batch[s, i]
                if np.isfinite(x):
                    state = (1.0 - a) * state + a * x
                out[s, i] = state
        return out

    @_numba.njit(cache=True)
    def _frac_diff_inverse_kernel(
        t_f: np.ndarray, lags: int, weights: np.ndarray, anchor: float,
    ) -> np.ndarray:
        """v1 single-spec frac-diff-inverse recurrence kernel; production default for K=1 path."""
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

    @_numba.njit(cache=True, parallel=True)
    def _frac_diff_inverse_kernel_njit_par_batched(
        t_batch: np.ndarray, lags: int, weights_batch: np.ndarray, anchors: np.ndarray,
    ) -> np.ndarray:
        """v2 batched frac-diff-inverse: K specs in parallel via prange. Each spec carries its own (weights, anchor) row; row-recurrence inside one spec stays serial. Bench: 3.8-5.4x over per-spec v1 at K>=10."""
        K, N = t_batch.shape
        out = np.empty((K, N), dtype=np.float64)
        for s in _numba.prange(K):
            inv_w0 = 1.0 / weights_batch[s, 0]
            anchor = anchors[s]
            for i in range(N):
                lag_sum = 0.0
                upper = min(i + 1, lags + 1)
                for k_idx in range(1, upper):
                    lag_sum += weights_batch[s, k_idx] * out[s, i - k_idx]
                for k_idx in range(upper, lags + 1):
                    lag_sum += weights_batch[s, k_idx] * anchor
                out[s, i] = (t_batch[s, i] - lag_sum) * inv_w0
        return out
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
    _ewma_kernel = None  # type: ignore
    _ewma_kernel_njit_par_batched = None  # type: ignore
    _frac_diff_inverse_kernel = None  # type: ignore
    _frac_diff_inverse_kernel_njit_par_batched = None  # type: ignore
    _quantile_assign_bins_kernel = None  # type: ignore


# Soft-cap MAD floor: when MAD(T_train) is below ``_MAD_FLOOR_FRAC * std(y_train)``, substitute the latter to keep the soft-cap bound numerically meaningful even if the transform produced a degenerate (near-constant) T on train. Without this, logratio's MAD-cap collapses to zero on degenerate train and every prediction inverts to ``base * exp(0) = base`` silently.
_MAD_FLOOR_FRAC: float = 1e-3

# Multiplier for MAD-soft-cap on T_hat (logratio in particular).
_MAD_SOFT_CAP_K: float = 10.0


logger = logging.getLogger("mlframe.training.composite_transforms")

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
        mean_per_group_variance = float(sigma2_total / max(np.mean(sizes), 1.0))
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
    from . import _QUANTILE_RESIDUAL_DEFAULT_MIN_BIN_N, _QUANTILE_RESIDUAL_DEFAULT_N_BINS
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
        return _quantile_assign_bins_kernel(
            np.ascontiguousarray(base_f), np.ascontiguousarray(edges[1:-1]), n_bins,
        )
    bin_idx = np.searchsorted(edges[1:-1], base_f, side="right")
    return np.clip(bin_idx, 0, n_bins - 1)
def _quantile_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    edges = np.asarray(params["bin_edges"], dtype=np.float64)
    medians = np.asarray(params["bin_medians"], dtype=np.float64)
    iqrs = np.asarray(params["bin_iqrs"], dtype=np.float64)
    bin_idx = _quantile_residual_assign_bins(base, edges)
    return (np.asarray(y, dtype=np.float64) - medians[bin_idx]) / iqrs[bin_idx]
def _quantile_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    edges = np.asarray(params["bin_edges"], dtype=np.float64)
    medians = np.asarray(params["bin_medians"], dtype=np.float64)
    iqrs = np.asarray(params["bin_iqrs"], dtype=np.float64)
    bin_idx = _quantile_residual_assign_bins(base, edges)
    return np.asarray(t_hat, dtype=np.float64) * iqrs[bin_idx] + medians[bin_idx]
def _quantile_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1))
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1))
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
    from . import _MONOTONIC_RESIDUAL_DEFAULT_MIN_KNOT_N, _MONOTONIC_RESIDUAL_DEFAULT_N_KNOTS
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
    # Orient monotonicity by Spearman correlation between y and base; tie -> increasing (arbitrary but stable).
    if y_clean.size >= 3 and base_clean.size >= 3:
        from scipy.stats import spearmanr  # lazy import
        try:
            rho, _ = spearmanr(base_clean, y_clean)
            direction = 1 if (rho is None or not np.isfinite(rho) or rho >= 0) else -1
        except Exception:
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
        return knots_y[0] + slope * (clipped - knots_x[0])
    from scipy.interpolate import PchipInterpolator  # lazy
    # extrapolate=False yields NaN outside [x[0], x[-1]]; fill those with the edge knot values to keep predict-time well-defined.
    interp = PchipInterpolator(knots_x, knots_y, extrapolate=False)
    out = interp(base_f)
    if np.any(~np.isfinite(out)):
        low_mask = base_f < knots_x[0]
        high_mask = base_f > knots_x[-1]
        out[low_mask] = knots_y[0]
        out[high_mask] = knots_y[-1]
    return out
def _monotonic_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    return np.asarray(y, dtype=np.float64) - _monotonic_residual_g(base, params)
def _monotonic_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    return np.asarray(t_hat, dtype=np.float64) + _monotonic_residual_g(base, params)
def _monotonic_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1))
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1))
def _ewma_residual_fit(
    y: np.ndarray, base: np.ndarray, k: int = _EWMA_RESIDUAL_DEFAULT_K,
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit stores only the EWMA half-life span ``k``; the EWMA itself is re-computed at forward / inverse time, keeping the fitted params JSON-serialisable and stateless (storing the full N-row EWMA trace would bloat metadata and break predict-on-new-data). The first-row anchor is the train-base mean: ``ewma[0] = mean(base_train)``."""
    # Lazy import: ``.predict`` re-imports this sibling at its bottom, so a top-level ``from .predict import ...`` would create a hard cycle the meta-test flags.
    from . import _EWMA_RESIDUAL_DEFAULT_K
    k = max(1, int(k))
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    finite = _finite_mask if _finite_mask is not None else np.isfinite(base_f)
    anchor = float(np.mean(base_f[finite])) if finite.any() else 0.0
    # tail_anchor is the EWMA state at the LAST train row -- the recency-correct
    # seed for a predict batch that CONTINUES the training series. Opt-in via the
    # estimator's recurrence_continuation flag; the default mean ``anchor`` keeps
    # predict stateless (a fresh batch is not assumed to follow train).
    tail_anchor = anchor
    if finite.any():
        _trace = _ewma_compute(base_f, k, anchor)
        _tf = _trace[np.isfinite(_trace)]
        if _tf.size:
            tail_anchor = float(_tf[-1])
    return {"k": k, "anchor": anchor, "tail_anchor": tail_anchor}


def _ewma_anchor(params: dict[str, Any]) -> float:
    """Mean anchor by default; the train-tail state when the estimator opted into
    recurrence-continuation seeding (streaming a continuation of the train series)."""
    if params.get("recurrence_continuation") and "tail_anchor" in params:
        return float(params["tail_anchor"])
    return float(params["anchor"])
def _ewma_compute(base: np.ndarray, k: int, anchor: float) -> np.ndarray:
    """Exponentially-weighted moving average using ``alpha = 2 / (k + 1)``. Non-finite base values inherit the previous EWMA state (carry-forward), keeping the recursion well-defined on rows the upstream domain check did not yet flag. Single-spec public API; routes through :func:`_ewma_dispatch` so a future force-override or HW-tuned threshold can replace the default njit path without touching every caller. Numba kernel ~300x over pure Python on n=1M; pure-Python fallback otherwise.
    """
    base_f = np.ascontiguousarray(np.asarray(base, dtype=np.float64).reshape(-1))
    return _ewma_dispatch(base_f, float(k), float(anchor))


# EWMA / frac-diff-inverse backend dispatch. Crossover constants are measurement-derived (bench_ewma_frac_diff_backends.py on GTX 1050 Ti + i7-7700k): batched-parallel is a net win once K>=2 AND N>=50k; below that the prange spawn cost (~50us) overshoots the per-spec serial work. Module-level so kernel_tuning_cache can override via :func:`_lookup_ewma_backend` / :func:`_lookup_frac_diff_inv_backend`.
_EWMA_PAR_MIN_K: int = 2
_EWMA_PAR_MIN_N: int = 50_000
_FRAC_DIFF_INV_PAR_MIN_K: int = 2
_FRAC_DIFF_INV_PAR_MIN_N: int = 10_000


def _ewma_force_backend() -> str:
    """Read env-var override (``MLFRAME_EWMA_BACKEND=njit|njit_par``). Returns empty string when unset / unknown -- dispatcher then uses the size-based default."""
    import os
    v = os.environ.get("MLFRAME_EWMA_BACKEND", "").strip().lower()
    return v if v in ("njit", "njit_par") else ""


def _frac_diff_inv_force_backend() -> str:
    """Read env-var override (``MLFRAME_FRAC_DIFF_INV_BACKEND=njit|njit_par``). Returns empty string when unset / unknown -- dispatcher then uses the size-based default."""
    import os
    v = os.environ.get("MLFRAME_FRAC_DIFF_INV_BACKEND", "").strip().lower()
    return v if v in ("njit", "njit_par") else ""


def _lookup_ewma_backend(K: int, N: int) -> str:
    """Return ``"njit"`` or ``"njit_par"`` via the kernel_tuning_cache when available, else the measurement-backed size-based fallback (K=1 -> njit; K>=2 AND N>=50k -> njit_par). Cache key axes are (K, N); HW-tuned crossovers persist via the same pyutilz KernelTuningCache that powers joint_hist_batched."""
    forced = _ewma_force_backend()
    if forced:
        return forced
    try:
        from mlframe.feature_selection.filters import get_kernel_tuning_cache
        cache = get_kernel_tuning_cache()
        if cache is not None:
            choice = cache.lookup("ewma_dispatch", K=K, N=N)
            if choice is not None:
                bc = str(choice.get("backend_choice", "")).strip().lower()
                if bc in ("njit", "njit_par"):
                    return bc
    except Exception:
        pass
    if K >= _EWMA_PAR_MIN_K and N >= _EWMA_PAR_MIN_N:
        return "njit_par"
    return "njit"


def _lookup_frac_diff_inv_backend(K: int, N: int) -> str:
    """Same contract as :func:`_lookup_ewma_backend` for the frac-diff-inverse kernel. Cache key: ``frac_diff_inverse_dispatch``."""
    forced = _frac_diff_inv_force_backend()
    if forced:
        return forced
    try:
        from mlframe.feature_selection.filters import get_kernel_tuning_cache
        cache = get_kernel_tuning_cache()
        if cache is not None:
            choice = cache.lookup("frac_diff_inverse_dispatch", K=K, N=N)
            if choice is not None:
                bc = str(choice.get("backend_choice", "")).strip().lower()
                if bc in ("njit", "njit_par"):
                    return bc
    except Exception:
        pass
    if K >= _FRAC_DIFF_INV_PAR_MIN_K and N >= _FRAC_DIFF_INV_PAR_MIN_N:
        return "njit_par"
    return "njit"


def _ewma_dispatch(base_f: np.ndarray, k_param: float, anchor: float) -> np.ndarray:
    """Single-spec dispatcher: 1-D ``base_f`` shape (N,) -> EWMA(N,). Routes to the scalar njit kernel unless the env-var force-override picks ``njit_par`` (in which case the batched kernel runs with K=1 -- useful for A/B testing the par-batched path on a single spec; size-based default never picks njit_par for K=1)."""
    alpha = 2.0 / (k_param + 1.0)
    if not _HAS_NUMBA:
        out = np.empty(base_f.size, dtype=np.float64)
        state = anchor
        for i in range(base_f.size):
            x = base_f[i]
            if np.isfinite(x):
                state = (1.0 - alpha) * state + alpha * x
            out[i] = state
        return out
    backend = _lookup_ewma_backend(1, int(base_f.size))
    if backend == "njit_par":
        base_batch = base_f.reshape(1, -1)
        alphas = np.array([alpha], dtype=np.float64)
        anchors = np.array([anchor], dtype=np.float64)
        return _ewma_kernel_njit_par_batched(base_batch, alphas, anchors)[0]
    return _ewma_kernel(base_f, alpha, float(anchor))


def _ewma_compute_batched(
    base_batch: np.ndarray, ks: np.ndarray, anchors: np.ndarray,
) -> np.ndarray:
    """Batched public API: run K independent EWMA specs on a (K, N) base matrix and return the (K, N) EWMA result. Each row carries its own ``k`` (half-life) and ``anchor`` (state-zero value). When K>=2 AND N is sufficiently large the parallel-batched njit kernel kicks in -- routed through :func:`_lookup_ewma_backend` so HW-tuned thresholds persist via kernel_tuning_cache. Bench: 2.7-3.8x over per-spec dispatch at K>=10, N>=100k; callers evaluating many EWMA specs on the same series (e.g. cross-target discovery scanning k in [3, 5, 7, 14, 21]) should batch via this entry point.
    """
    base_batch = np.ascontiguousarray(np.asarray(base_batch, dtype=np.float64))
    if base_batch.ndim == 1:
        base_batch = base_batch.reshape(1, -1)
    K, N = base_batch.shape
    ks_a = np.ascontiguousarray(np.asarray(ks, dtype=np.float64).reshape(-1))
    anchors_a = np.ascontiguousarray(np.asarray(anchors, dtype=np.float64).reshape(-1))
    if ks_a.size != K or anchors_a.size != K:
        raise ValueError(
            f"_ewma_compute_batched: ks shape {ks_a.shape} and anchors shape {anchors_a.shape} must each equal (K={K},)"
        )
    alphas = 2.0 / (ks_a + 1.0)
    if not _HAS_NUMBA:
        out = np.empty((K, N), dtype=np.float64)
        for s in range(K):
            state = float(anchors_a[s])
            a = float(alphas[s])
            for i in range(N):
                x = float(base_batch[s, i])
                if np.isfinite(x):
                    state = (1.0 - a) * state + a * x
                out[s, i] = state
        return out
    backend = _lookup_ewma_backend(K, N)
    if backend == "njit_par":
        return _ewma_kernel_njit_par_batched(base_batch, alphas, anchors_a)
    out = np.empty((K, N), dtype=np.float64)
    for s in range(K):
        out[s] = _ewma_kernel(
            np.ascontiguousarray(base_batch[s]), float(alphas[s]), float(anchors_a[s]),
        )
    return out


def _frac_diff_inverse_compute(
    t_f: np.ndarray, lags: int, weights: np.ndarray, anchor: float,
) -> np.ndarray:
    """Single-spec frac-diff-inverse public API; routes through :func:`_frac_diff_inverse_dispatch`."""
    t_f = np.ascontiguousarray(np.asarray(t_f, dtype=np.float64).reshape(-1))
    weights = np.ascontiguousarray(np.asarray(weights, dtype=np.float64).reshape(-1))
    return _frac_diff_inverse_dispatch(t_f, int(lags), weights, float(anchor))


def _frac_diff_inverse_dispatch(
    t_f: np.ndarray, lags: int, weights: np.ndarray, anchor: float,
) -> np.ndarray:
    """Single-spec dispatcher (1-D in, 1-D out). Default routes to scalar njit kernel; env-var force-override or KTC entry can pick the par-batched path with K=1."""
    if not _HAS_NUMBA:
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
    backend = _lookup_frac_diff_inv_backend(1, int(t_f.size))
    if backend == "njit_par":
        t_batch = t_f.reshape(1, -1)
        weights_batch = weights.reshape(1, -1)
        anchors = np.array([anchor], dtype=np.float64)
        return _frac_diff_inverse_kernel_njit_par_batched(t_batch, lags, weights_batch, anchors)[0]
    return _frac_diff_inverse_kernel(t_f, lags, weights, anchor)


def _frac_diff_inverse_compute_batched(
    t_batch: np.ndarray, lags: int, weights_batch: np.ndarray, anchors: np.ndarray,
) -> np.ndarray:
    """Batched public API: K independent frac-diff-inverse specs on a (K, N) t_hat matrix. ``weights_batch`` is (K, lags+1), ``anchors`` is (K,). Bench: 3.8-5.4x over per-spec dispatch at K>=10."""
    t_batch = np.ascontiguousarray(np.asarray(t_batch, dtype=np.float64))
    if t_batch.ndim == 1:
        t_batch = t_batch.reshape(1, -1)
    K, N = t_batch.shape
    weights_batch = np.ascontiguousarray(np.asarray(weights_batch, dtype=np.float64))
    if weights_batch.ndim == 1:
        weights_batch = np.tile(weights_batch, (K, 1))
    anchors_a = np.ascontiguousarray(np.asarray(anchors, dtype=np.float64).reshape(-1))
    if anchors_a.size != K or weights_batch.shape[0] != K:
        raise ValueError(
            f"_frac_diff_inverse_compute_batched: anchors shape {anchors_a.shape} and weights_batch shape {weights_batch.shape} must each have K={K} rows"
        )
    if not _HAS_NUMBA:
        out = np.empty((K, N), dtype=np.float64)
        for s in range(K):
            anchor = float(anchors_a[s])
            inv_w0 = 1.0 / float(weights_batch[s, 0])
            for i in range(N):
                lag_sum = 0.0
                upper = min(i + 1, lags + 1)
                for k_idx in range(1, upper):
                    lag_sum += float(weights_batch[s, k_idx]) * float(out[s, i - k_idx])
                for k_idx in range(upper, lags + 1):
                    lag_sum += float(weights_batch[s, k_idx]) * anchor
                out[s, i] = (float(t_batch[s, i]) - lag_sum) * inv_w0
        return out
    backend = _lookup_frac_diff_inv_backend(K, N)
    if backend == "njit_par":
        return _frac_diff_inverse_kernel_njit_par_batched(t_batch, lags, weights_batch, anchors_a)
    out = np.empty((K, N), dtype=np.float64)
    for s in range(K):
        out[s] = _frac_diff_inverse_kernel(
            np.ascontiguousarray(t_batch[s]), lags,
            np.ascontiguousarray(weights_batch[s]),
            float(anchors_a[s]),
        )
    return out
def _ewma_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    return np.asarray(y, dtype=np.float64) - _ewma_compute(
        base, int(params["k"]), float(params["anchor"]),
    )
def _ewma_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    return np.asarray(t_hat, dtype=np.float64) + _ewma_compute(
        base, int(params["k"]), _ewma_anchor(params),
    )
def _ewma_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1))
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1))
def _rolling_median_pandas(arr_f: np.ndarray, k: int) -> np.ndarray:
    """Reference centred rolling median: pandas ``rolling(window=k, center=True, min_periods=1).median()``. This is the CONTRACT both backends reproduce. ``arr_f`` must already be float64 / 1-D / non-empty; ``k`` already clamped to ``>= 1``."""
    import pandas as pd  # lazy
    return pd.Series(arr_f).rolling(window=k, center=True, min_periods=1).median().to_numpy()


def _rolling_median(arr: np.ndarray, k: int) -> np.ndarray:
    """Centred rolling median with truncation at boundaries.

    Reference semantics (the cross-environment CONTRACT) = pandas ``rolling(window=k, center=True, min_periods=1).median()``: position ``i`` is the median of ``arr[i - k//2 .. i + (k-1)//2]`` clipped to ``[0, n-1]`` (so head/tail windows truncate, and NaN cells inside a window are SKIPPED, never poisoning the window).

    Fast path: ``bottleneck.move_median`` (forward-window O(n log k) quickselect; ~8-10x faster than pandas at k in [7, 21] on n=100k) re-centred to that contract. The forward window ending at index ``j`` is the centred window for ``i = j - (k-1)//2``, so the correct LEFT shift is ``(k-1)//2`` (NOT ``k//2`` -- the historic ``k//2`` shift was off-by-one for every EVEN ``k``). Head positions and tail positions whose centred window would run past the array end carry directly-computed truncated medians (the historic code constant-filled the tail with the last full-window median -- wrong for both even and odd ``k``). ``move_median`` also REQUIRES ``window <= n``, so ``k`` is clamped to ``min(k, n)`` for the kernel call (a centred window wider than the array is identical to ``k = n``; the historic code passed ``k > n`` straight through and ``move_median`` raised, silently dropping the whole result to the non-finite fallback).

    NaN parity: ``bottleneck.move_median`` does NOT skip NaN inside a window (one NaN poisons the window to NaN), whereas pandas' ``min_periods=1`` median skips them. So the fast path is bit-identical to the pandas contract ONLY when the input is all-finite; non-finite input routes to the pandas reference to preserve identical results regardless of whether bottleneck is installed. (The downstream callers domain-check ``base`` finite, so the all-finite fast path is the common case.) After either path, any residual NaN (an entirely-non-finite window under the pandas route) is replaced with the row's own value (or 0.0 if also non-finite) to match the legacy fallback.
    """
    arr_f = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr_f.size == 0:
        return arr_f.copy()
    n = arr_f.size
    k = max(1, int(k))
    out: np.ndarray | None = None
    # Fast path requires all-finite input (move_median can't NaN-skip within a
    # window the way pandas does); otherwise fall through to the pandas reference
    # so results are identical across environments.
    if np.isfinite(arr_f).all():
        try:
            import bottleneck as _bn  # lazy; optional dep but present in mlframe[all]
            k_eff = min(k, n)  # move_median requires 1 <= window <= n
            _fwd = _bn.move_median(arr_f, window=k_eff, min_count=1)
            _shift = (k - 1) // 2  # forward index j = i + (k-1)//2 (NOT k//2)
            _left = k // 2
            out = np.empty(n, dtype=np.float64)
            # Interior positions ``i`` carry the full ``k_eff`` window AND are
            # forward-readable: i >= _left, i + _shift >= k_eff - 1 (full kernel
            # window), i + _shift <= n - 1 (in range). These are a single
            # vectorised slice of the kernel output -- no per-row Python work.
            lo_i = max(_left, k_eff - 1 - _shift)
            hi_i = n - 1 - _shift
            if hi_i >= lo_i:
                out[lo_i:hi_i + 1] = _fwd[lo_i + _shift:hi_i + _shift + 1]
            # Boundary positions (head + tail, O(k) of them): centred window
            # truncates to the array; compute its median directly to match
            # pandas exactly (the historic constant tail-fill was wrong).
            for i in range(0, min(max(lo_i, 0), n)):
                lo = i - _left if i - _left > 0 else 0
                hi = i + _shift if i + _shift < n - 1 else n - 1
                out[i] = np.median(arr_f[lo:hi + 1])
            for i in range(max(hi_i + 1, 0), n):
                lo = i - _left if i - _left > 0 else 0
                hi = i + _shift if i + _shift < n - 1 else n - 1
                out[i] = np.median(arr_f[lo:hi + 1])
        except ImportError:
            out = None
    if out is None:
        out = _rolling_median_pandas(arr_f, k)
    bad = ~np.isfinite(out)
    if bad.any():
        fallback = np.where(np.isfinite(arr_f), arr_f, 0.0)
        out = np.where(bad, fallback, out)
    return out
def _frac_diff_weights(d: float, lags: int) -> np.ndarray:
    """Truncated weight series for (1 - L)^d expansion."""
    lags = max(1, int(lags))
    w = np.empty(lags + 1, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, lags + 1):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return w
def _frac_diff_fit(
    y: np.ndarray, base: np.ndarray,
    d: float = _FRAC_DIFF_DEFAULT_D, lags: int = _FRAC_DIFF_DEFAULT_LAGS,
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Store fractional order ``d``, lag truncation ``lags``, and the train-y mean used as a pre-window anchor (rows whose lag history is shorter than ``lags`` need a fallback value for the missing past terms)."""
    # Lazy import: ``.predict`` re-imports this sibling at its bottom, so a top-level ``from .predict import ...`` would create a hard cycle the meta-test flags.
    from . import _FRAC_DIFF_DEFAULT_D, _FRAC_DIFF_DEFAULT_LAGS
    d = float(d)
    lags = max(1, int(lags))
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = _finite_mask if _finite_mask is not None else np.isfinite(y_f)
    anchor = float(np.mean(y_f[finite])) if finite.any() else 0.0
    # tail_anchor pads the pre-window history from the LAST ``lags`` train rows
    # (recency-correct seed for a continuation predict batch) instead of the
    # whole-train mean. Opt-in via recurrence_continuation; default stays mean.
    tail_anchor = anchor
    _yt = y_f[finite]
    if _yt.size:
        tail_anchor = float(np.mean(_yt[-lags:]))
    return {
        "d": d, "lags": lags, "anchor": anchor, "tail_anchor": tail_anchor,
        "weights": _frac_diff_weights(d, lags).tolist(),
    }
def _frac_diff_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """T_i = sum_{k=0}^{lags} w_k * y_{i-k}, padding y_{i-k} with the train anchor for k > i. Vectorised via ``np.convolve(y_padded, weights, 'valid')`` after left-padding ``y`` with ``lags`` copies of the train anchor (~340x over the nested Python loop on n=1M, lags=30)."""
    lags = int(params["lags"])
    weights = np.asarray(params["weights"], dtype=np.float64)
    anchor = float(params["anchor"])
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    if y_f.size == 0:
        return y_f.copy()
    y_padded = np.concatenate([np.full(lags, anchor, dtype=np.float64), y_f])
    return np.convolve(y_padded, weights, mode="valid")
def _frac_diff_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Invert: T_i = w_0 * y_i + sum_{k=1}^{lags} w_k * y_{i-k}, so y_i = (T_i - sum_{k=1}^{lags} w_k * y_{i-k}) / w_0. w_0 == 1 by construction. Past y values are unknown at predict, so we ITERATIVELY reconstruct them: y_0 from T_0 + lag-anchors, y_1 from T_1 + y_0 + lag-anchors, etc. Routes through :func:`_frac_diff_inverse_compute` -> :func:`_frac_diff_inverse_dispatch` so kernel_tuning_cache + env-var force-override choose the backend; default keeps the scalar njit kernel (~260x over pure Python on n=1M, lags=30)."""
    lags = int(params["lags"])
    weights = np.ascontiguousarray(np.asarray(params["weights"], dtype=np.float64))
    anchor = _ewma_anchor(params)  # mean by default, train-tail when opted in
    t_f = np.ascontiguousarray(np.asarray(t_hat, dtype=np.float64).reshape(-1))
    return _frac_diff_inverse_compute(t_f, lags, weights, anchor)
def _frac_diff_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Frac-diff is y-only; base is accepted for signature uniformity but never read. Domain when y is present is finite-y only (a non-finite UNUSED base must not drop rows and compact the y sequence). The base-finite mask is kept solely for the ``y is None`` predict-side call."""
    if y is None:
        return np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1))
    return np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1))
def _make_chain_transform(
    *, name: str, short_name: str,
    bivariate_name: str,
    bivariate_fit, bivariate_forward, bivariate_inverse, bivariate_domain,
    unary_fit, unary_forward, unary_inverse,
    description: str,
):
    """Create a registry Transform for ``chain(bivariate, unary)``.

    The chain inherits ``requires_base=True`` from the bivariate half (it still needs a base column at fit + predict). At fit-time it first fits the bivariate, applies forward to get T1, then fits the unary on T1; the joint params dict stores both. Forward / inverse run in the matching order. Domain check delegates to the bivariate's check since the unary half has no base-dependent constraint at predict.
    """
    # Lazy import: ``.predict`` re-imports this sibling at its bottom, so a top-level ``from .predict import ...`` would create a hard cycle the meta-test flags.
    from . import TAG_EXTENDED, TAG_REGRESSION, Transform, _chain_fit_raw, _chain_forward_raw, _chain_inverse_raw
    unary_tup = (unary_fit, unary_forward, unary_inverse)

    def _fit(y, base):
        return _chain_fit_raw(
            y=y, base=base,
            bivariate_fit=bivariate_fit,
            bivariate_forward=bivariate_forward,
            unary=unary_tup,
        )

    def _forward(y, base, params):
        return _chain_forward_raw(
            y=y, base=base, params=params,
            bivariate_forward=bivariate_forward,
            unary=unary_tup,
        )

    def _inverse(t_hat, base, params):
        return _chain_inverse_raw(
            t2=t_hat, base=base, params=params,
            bivariate_inverse=bivariate_inverse,
            unary=unary_tup,
        )

    def _domain(y, base):
        return bivariate_domain(y, base)

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
    """Multi-stage chain: bivariate + N unary stages. ``unary_stages`` is a list of ``(fit, forward, inverse)`` tuples; each runs in order at forward, in reverse at inverse. Used to register e.g. ``chain([linres, cbrt, quantile_normal])`` for very heavy-tail residuals.
    """
    # Lazy import: ``.predict`` re-imports this sibling at its bottom, so a top-level ``from .predict import ...`` would create a hard cycle the meta-test flags.
    from . import TAG_EXTENDED, TAG_REGRESSION, Transform, _chain_multi_fit_raw, _chain_multi_forward_raw, _chain_multi_inverse_raw

    def _fit(y, base):
        return _chain_multi_fit_raw(
            y=y, base=base,
            bivariate_fit=bivariate_fit,
            bivariate_forward=bivariate_forward,
            unary_stages=unary_stages,
        )

    def _forward(y, base, params):
        return _chain_multi_forward_raw(
            y=y, base=base, params=params,
            bivariate_forward=bivariate_forward,
            unary_stages=unary_stages,
        )

    def _inverse(t_hat, base, params):
        return _chain_multi_inverse_raw(
            t_final=t_hat, base=base, params=params,
            bivariate_inverse=bivariate_inverse,
            unary_stages=unary_stages,
        )

    def _domain(y, base):
        return bivariate_domain(y, base)

    return Transform(
        name=name,
        forward=_forward,
        inverse=_inverse,
        fit=_fit,
        domain_check=_domain,
        description=description,
        tags=frozenset({TAG_EXTENDED, TAG_REGRESSION}),
    )
