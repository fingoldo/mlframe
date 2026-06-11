"""Simple composite transforms carved out of
``mlframe.training.composite_transforms``: diff, additive_residual,
median_residual, y_quantile_clip, ratio, rolling_quantile_ratio.

Bound back into the parent's namespace via re-export at the parent's
module bottom so historical
``from mlframe.training.composite_transforms import _diff_fit`` resolves.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

logger = logging.getLogger("mlframe.training.composite_transforms")


__all__ = [
    "_MEDIAN_RESIDUAL_N_BINS",
    "_Y_QUANTILE_CLIP_LO",
    "_Y_QUANTILE_CLIP_HI",
    "_ROLLING_QUANTILE_DEFAULT_K",
    "_diff_fit",
    "_diff_forward",
    "_diff_inverse",
    "_diff_domain",
    "_additive_residual_fit",
    "_additive_residual_forward",
    "_additive_residual_inverse",
    "_additive_residual_domain",
    "_median_residual_per_bin_medians_v1_pyloop",
    "_median_residual_per_bin_medians_v2_pandas_groupby",
    "_median_residual_per_bin_medians",
    "_median_residual_fit",
    "_median_residual_g",
    "_median_residual_forward",
    "_median_residual_inverse",
    "_median_residual_domain",
    "_y_quantile_clip_fit",
    "_y_quantile_clip_forward",
    "_y_quantile_clip_inverse",
    "_y_quantile_clip_domain",
    "_ratio_fit",
    "_ratio_forward",
    "_ratio_inverse",
    "_ratio_domain",
    "_rolling_quantile_ratio_fit",
    "_rolling_quantile_ratio_forward",
    "_rolling_quantile_ratio_inverse",
    "_rolling_quantile_ratio_domain",
]


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
def _additive_residual_fit(
    y: np.ndarray, base: np.ndarray, _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    # Optional precomputed joint finite mask: a caller that already knows the (y, base) finite mask can pass it to skip the per-call np.isfinite. No current call site supplies it (discovery fits plain (y, base)); the recompute below is the live path.
    finite = _finite_mask if _finite_mask is not None else (np.isfinite(y) & np.isfinite(base))
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
        except Exception as _exc:
            logger.warning("composite_transforms: pandas-groupby fast path failed (%s); using numpy fallback.", _exc)
    return _median_residual_per_bin_medians_v1_pyloop(y_f, bin_idx, n_bins)


def _median_residual_fit(
    y: np.ndarray, base: np.ndarray, _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    finite = _finite_mask if _finite_mask is not None else (np.isfinite(y) & np.isfinite(base))
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


def _y_quantile_clip_fit(
    y: np.ndarray, base: np.ndarray, _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    # Unary y-only transform; an optional caller-supplied ``_finite_mask`` is treated as the y-side finite gate (the base argument is ignored either way). No current call site supplies it.
    finite = _finite_mask if _finite_mask is not None else np.isfinite(y)
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
    y: np.ndarray | None, base: np.ndarray | None,
) -> np.ndarray:
    if y is None:
        # ``y_quantile_clip`` is ``requires_base=False`` and ignores ``base``; at predict-time the wrapper passes ``base=None``, so size the all-True mask from whichever array is present (never call ``np.isfinite`` on ``None``).
        n = len(base) if base is not None and hasattr(base, "__len__") else 1
        return np.ones(n, dtype=bool)
    return np.isfinite(np.asarray(y, dtype=np.float64))


# ----------------------------------------------------------------------
# ratio: T = y / base. Requires |base| >= eps.
# ----------------------------------------------------------------------

def _ratio_fit(
    y: np.ndarray, base: np.ndarray, _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    # eps relative to the typical scale of base on train -- small enough not to bias the transform but large enough to keep division numerically clean.
    # Stored in params so predict time uses the SAME eps (no train/test drift). An optional caller-supplied ``_finite_mask`` skips the np.isfinite recompute on base; no current call site supplies it.
    if _finite_mask is not None:
        base_finite = _finite_mask & (base != 0)
    else:
        base_finite = np.isfinite(base) & (base != 0)
    # `np.median([])` emits "Mean of empty slice" RuntimeWarning and returns NaN; on NumPy 2.x this may raise outright. Guard so the all-non-finite / all-zero base path is silently safe.
    if base_finite.any():
        scale = float(np.median(np.abs(base[base_finite])))
    else:
        scale = 0.0
    eps = max(scale * 1e-6, 1e-12) if scale > 0 else 1e-12
    return {"eps": eps}


def _ratio_forward(y: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    eps = float(params["eps"])
    safe_base = np.where(np.abs(base) < eps, np.sign(base + 1e-300) * eps, base)
    return y / safe_base


def _ratio_inverse(t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    # Mirror the forward eps-floor so the round-trip is exact on in-domain near-zero base rows (0<|base|<eps would otherwise yield unbounded relative error).
    eps = float(params["eps"])
    safe_base = np.where(np.abs(base) < eps, np.sign(base + 1e-300) * eps, base)
    return t_hat * safe_base


def _ratio_domain(y: np.ndarray | None, base: np.ndarray) -> np.ndarray:
    base_ok = np.isfinite(base) & (np.abs(base) > 0)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)


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
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Stores the window span ``k`` and an eps floor derived from train base scale to keep division safe at predict time on near-zero rolling medians."""
    k = max(1, int(k))
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    if _finite_mask is not None:
        finite = _finite_mask & (base_f != 0)
    else:
        finite = np.isfinite(base_f) & (base_f != 0)
    scale = float(np.median(np.abs(base_f[finite]))) if finite.any() else 1.0
    eps = max(scale * 1e-6, 1e-12)
    return {"k": k, "eps": eps}


def _rolling_quantile_ratio_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    # Lazy import: ``_rolling_median`` lives in the nonlinear sibling, which
    # imports the parent at top, so this top-level import would cycle.
    from .nonlinear import _rolling_median
    k = int(params["k"])
    eps = float(params["eps"])
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    roll_med = _rolling_median(base_f, k)
    safe = np.where(np.abs(roll_med) < eps, np.sign(roll_med + 1e-300) * eps, roll_med)
    return np.asarray(y, dtype=np.float64) / safe


def _rolling_quantile_ratio_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    from .nonlinear import _rolling_median
    k = int(params["k"])
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    roll_med = _rolling_median(base_f, k)
    # Mirror the forward eps-floor so the round-trip is exact on near-zero rolling medians.
    eps = float(params["eps"])
    safe = np.where(np.abs(roll_med) < eps, np.sign(roll_med + 1e-300) * eps, roll_med)
    return np.asarray(t_hat, dtype=np.float64) * safe


def _rolling_quantile_ratio_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1))
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1))
