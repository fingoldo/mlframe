"""Simple composite transforms carved out of
``mlframe.training.composite_transforms``: diff, additive_residual,
median_residual, y_quantile_clip, ratio, rolling_quantile_ratio.

Bound back into the parent's namespace via re-export at the parent's
module bottom so historical
``from mlframe.training.composite_transforms import _diff_fit`` resolves.
"""
from __future__ import annotations

import logging
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
    "_rolling_quantile_ratio_centered_fit",
    "_rolling_quantile_ratio_forward",
    "_rolling_quantile_ratio_inverse",
    "_rolling_quantile_ratio_domain",
    "_rolling_median_trailing",
]


# ----------------------------------------------------------------------
# diff: T = y - base. Always defined, no params, no domain restrictions.
# ----------------------------------------------------------------------

def _diff_fit(y: np.ndarray, base: np.ndarray) -> dict[str, Any]:
    """No learnable parameters: ``diff`` is a fixed alpha=1/beta=0 transform, so fitting is a no-op returning an empty params dict."""
    return {}


def _diff_forward(y: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply the fixed transform ``T = y - base``."""
    return np.asarray(y - base)


def _diff_inverse(t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Undo the fixed transform: ``y = T_hat + base``."""
    return np.asarray(t_hat + base)


def _diff_domain(y: np.ndarray | None, base: np.ndarray) -> np.ndarray:
    """Return the boolean mask of rows where the transform is defined: finite ``base`` (and finite ``y`` when provided)."""
    base_ok = np.isfinite(base)
    if y is None:
        return np.asarray(base_ok)
    return np.asarray(base_ok & np.isfinite(y))


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
    """Learn the constant offset ``beta = mean(y - base)`` over finite rows (0.0 if none are finite).

    # Optional precomputed joint finite mask: a caller that already knows the (y, base) finite mask can pass it to skip the per-call np.isfinite. No current call site supplies it (discovery fits plain (y, base)); the recompute below is the live path.
    """
    finite = _finite_mask if _finite_mask is not None else (np.isfinite(y) & np.isfinite(base))
    if not finite.any():
        return {"beta": 0.0}
    beta = float(np.mean(y[finite] - base[finite]))
    return {"beta": beta}


def _additive_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Apply ``T = y - base - beta`` using the beta learned at fit time."""
    return np.asarray(y - base - float(params.get("beta", 0.0)))


def _additive_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Undo the transform: ``y = T_hat + base + beta``."""
    return np.asarray(t_hat + base + float(params.get("beta", 0.0)))


def _additive_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Return the boolean mask of rows where the transform is defined: finite ``base`` (and finite ``y`` when provided)."""
    base_ok = np.isfinite(base)
    if y is None:
        return np.asarray(base_ok)
    return np.asarray(base_ok & np.isfinite(y))


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
    """Learn per-quantile-bin medians of ``y`` conditioned on ``base``: quantile-bin edges over ``base``, plus the median of ``y`` within each bin (fallback median for empty bins)."""
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
    """Look up the fitted per-bin median of ``y`` for each row's ``base`` value, digitizing into the fitted bin edges."""
    bin_edges = np.asarray(params["bin_edges"], dtype=np.float64)
    bin_medians = np.asarray(params["bin_medians"], dtype=np.float64)
    fallback = float(params.get("fallback_median", 0.0))
    if bin_edges.size < 2:
        return np.full_like(base, fallback, dtype=np.float64)
    idx = np.digitize(base, bin_edges[1:-1])
    idx = np.clip(idx, 0, bin_medians.size - 1)
    return np.asarray(bin_medians[idx])


def _median_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Apply ``T = y - median(y | bin(base))`` using the fitted per-bin medians."""
    return np.asarray(y - _median_residual_g(base, params))


def _median_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Undo the transform: ``y = T_hat + median(y | bin(base))``."""
    return np.asarray(t_hat + _median_residual_g(base, params))


def _median_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Return the boolean mask of rows where the transform is defined: finite ``base`` (and finite ``y`` when provided)."""
    base_ok = np.isfinite(base)
    if y is None:
        return np.asarray(base_ok)
    return np.asarray(base_ok & np.isfinite(y))


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
    """Learn the [q_lo, q_hi] clip bounds as the ``_Y_QUANTILE_CLIP_LO``/``_Y_QUANTILE_CLIP_HI`` quantiles of ``y``.

    # Unary y-only transform; an optional caller-supplied ``_finite_mask`` is treated as the y-side finite gate (the base argument is ignored either way). No current call site supplies it.
    """
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
    """Clip ``y`` to the fitted [q_lo, q_hi] range; ``base`` is ignored (unary transform)."""
    q_lo = float(params.get("q_lo", -np.inf))
    q_hi = float(params.get("q_hi", np.inf))
    return np.clip(y.astype(np.float64), q_lo, q_hi)


def _y_quantile_clip_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Identity-like inverse: clipping is lossy so the inverse just re-clips ``t_hat`` to [q_lo, q_hi] rather than truly inverting."""
    q_lo = float(params.get("q_lo", -np.inf))
    q_hi = float(params.get("q_hi", np.inf))
    return np.clip(np.asarray(t_hat, dtype=np.float64), q_lo, q_hi)


def _y_quantile_clip_domain(
    y: np.ndarray | None, base: np.ndarray | None,
) -> np.ndarray:
    """Return the boolean mask of rows where the transform is defined: finite ``y`` when provided, else all-True sized off ``base`` (unary transform ignores ``base`` content)."""
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
    """Learn an eps floor (relative to the median |base|) used to keep ``y / base`` numerically safe near zero."""
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
    """Apply ``T = y / base``, flooring ``|base|`` at the fitted eps (sign-preserving) to avoid division blowup."""
    eps = float(params["eps"])
    safe_base = np.where(np.abs(base) < eps, np.sign(base + 1e-300) * eps, base)
    return np.asarray(y / safe_base)


def _ratio_inverse(t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Undo the transform: ``y = T_hat * base`` (eps-floored, mirroring forward).

    # Mirror the forward eps-floor so the round-trip is exact on in-domain near-zero base rows (0<|base|<eps would otherwise yield unbounded relative error).
    """
    eps = float(params["eps"])
    safe_base = np.where(np.abs(base) < eps, np.sign(base + 1e-300) * eps, base)
    return np.asarray(t_hat * safe_base)


def _ratio_domain(y: np.ndarray | None, base: np.ndarray) -> np.ndarray:
    """Return the boolean mask of rows where the transform is defined: finite, non-zero ``base`` (and finite ``y`` when provided)."""
    base_ok = np.isfinite(base) & (np.abs(base) > 0)
    if y is None:
        return np.asarray(base_ok)
    return np.asarray(base_ok & np.isfinite(y))


# ----------------------------------------------------------------------
# rolling_quantile_ratio (R10c brainstorm #5b).
#
# T = y / max(RollingQ50_k(base), eps), where RollingQ50_k is the rolling median of ``base`` over a window of ``k`` rows. Inverse: y_hat = T_hat * RollingQ50_k(base).
#
# Two window modes, selected at fit time and stored in params:
# - ``"trailing"`` (default): position ``i`` sees rows ``[i-k+1 .. i]`` only -- past-only, no look-ahead, safe for time-ordered deployment.
# - ``"centered"``: position ``i`` sees rows ``[i-k//2 .. i+(k-1)//2]`` -- reads FUTURE base rows (leaks forward in time order); kept for
#   non-chronological / cross-sectional use via the ``rolling_quantile_ratio_centered`` registry entry.
# Params fitted before the mode field existed carry no ``"mode"`` key and keep their historical centred behaviour on load.
#
# Use case: multiplicative DGP where the local-median of base sets the y-scale. Logratio captures global multiplicative structure but not local windowing. ``rolling_quantile_ratio`` is the localised version.
# ----------------------------------------------------------------------

_ROLLING_QUANTILE_DEFAULT_K: int = 7


def _rolling_median_trailing(arr: np.ndarray, k: int) -> np.ndarray:
    """Trailing (past-only) rolling median: position ``i`` is the median of ``arr[max(0, i-k+1) .. i]``.

    Contract = pandas ``rolling(window=k, min_periods=1).median()`` (head windows truncate; NaN inside a window is skipped).
    Fast path: ``bottleneck.move_median`` computes EXACTLY this trailing window; like :func:`~mlframe.training.composite.transforms.nonlinear._rolling_median`
    it does not NaN-skip, so non-finite input routes to the pandas reference. Residual NaN (all-non-finite window) falls back to the row's own value / 0.0.
    """
    arr_f = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr_f.size == 0:
        return np.asarray(arr_f.copy())
    n = arr_f.size
    k = max(1, int(k))
    out: np.ndarray | None = None
    if np.isfinite(arr_f).all():
        try:
            import bottleneck as _bn  # lazy; optional dep but present in mlframe[all]
            out = np.asarray(_bn.move_median(arr_f, window=min(k, n), min_count=1), dtype=np.float64)
        except ImportError:
            out = None
    if out is None:
        import pandas as pd  # lazy
        out = np.asarray(pd.Series(arr_f).rolling(window=k, min_periods=1).median().to_numpy())
    bad = ~np.isfinite(out)
    if bad.any():
        fallback = np.where(np.isfinite(arr_f), arr_f, 0.0)
        out = np.where(bad, fallback, out)
    return out


def _rqr_rolling_median(base_f: np.ndarray, k: int, mode: str) -> np.ndarray:
    """Route to the trailing or centred rolling median per the fitted ``mode``."""
    if mode == "trailing":
        return _rolling_median_trailing(base_f, k)
    # Lazy import: ``_rolling_median`` lives in the nonlinear sibling, which
    # imports the parent at top, so a top-level import would cycle.
    from .nonlinear import _rolling_median
    return _rolling_median(base_f, k)


def _rolling_quantile_ratio_fit(
    y: np.ndarray, base: np.ndarray, k: int = _ROLLING_QUANTILE_DEFAULT_K,
    mode: str = "trailing",
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Stores the window span ``k``, the window ``mode`` (trailing = past-only default; centered = legacy look-ahead) and an eps floor derived from train base scale to keep division safe at predict time on near-zero rolling medians."""
    k = max(1, int(k))
    if mode not in ("trailing", "centered"):
        raise ValueError(f"rolling_quantile_ratio: mode must be 'trailing' or 'centered', got {mode!r}")
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    if _finite_mask is not None:
        finite = _finite_mask & (base_f != 0)
    else:
        finite = np.isfinite(base_f) & (base_f != 0)
    scale = float(np.median(np.abs(base_f[finite]))) if finite.any() else 1.0
    eps = max(scale * 1e-6, 1e-12)
    return {"k": k, "eps": eps, "mode": mode}


def _rolling_quantile_ratio_centered_fit(
    y: np.ndarray, base: np.ndarray, k: int = _ROLLING_QUANTILE_DEFAULT_K,
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit for the ``rolling_quantile_ratio_centered`` registry entry: same params with ``mode='centered'`` pinned."""
    return _rolling_quantile_ratio_fit(y, base, k=k, mode="centered", _finite_mask=_finite_mask)


def _rolling_quantile_ratio_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Apply ``T = y / max(RollingQ50_k(base), eps)`` with the rolling median of ``base`` over window ``k`` in the fitted mode (params without a ``mode`` key predate the field and keep the historical centred window)."""
    k = int(params["k"])
    eps = float(params["eps"])
    mode = str(params.get("mode", "centered"))
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    roll_med = _rqr_rolling_median(base_f, k, mode)
    safe = np.where(np.abs(roll_med) < eps, np.sign(roll_med + 1e-300) * eps, roll_med)
    return np.asarray(np.asarray(y, dtype=np.float64) / safe)


def _rolling_quantile_ratio_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Undo the transform: ``y = T_hat * max(RollingQ50_k(base), eps)`` with the same fitted window mode as the forward."""
    k = int(params["k"])
    mode = str(params.get("mode", "centered"))
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    roll_med = _rqr_rolling_median(base_f, k, mode)
    # Mirror the forward eps-floor so the round-trip is exact on near-zero rolling medians.
    eps = float(params["eps"])
    safe = np.where(np.abs(roll_med) < eps, np.sign(roll_med + 1e-300) * eps, roll_med)
    return np.asarray(np.asarray(t_hat, dtype=np.float64) * safe)


def _rolling_quantile_ratio_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Return the boolean mask of rows where the transform is defined: finite ``base`` (and finite ``y`` when provided)."""
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1))
    if y is None:
        return base_ok
    return np.asarray(base_ok & np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1)))
