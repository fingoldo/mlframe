"""Grouped composite transforms: per-group variants of the recurrent trio (``ewma_residual_grouped`` / ``rolling_quantile_ratio_grouped`` /
``frac_diff_grouped``) and of the non-parametric pair (``quantile_residual_grouped`` / ``monotonic_residual_grouped``).

Recurrent trio. On a stacked panel (many entities in one frame) the ungrouped recurrences bleed state across entity boundaries: the EWMA seeded on
entity A's level contaminates entity B's first rows, a rolling window straddles the boundary, and frac-diff lag weights convolve across entities.
The grouped variants reset the recurrence at every group boundary: rows are processed PER GROUP in their stable original order (rows of one group
need not be contiguous), each group carrying its own anchor. All three are ``requires_groups=True`` and keep ``recurrent=True`` (within a group the
forward still reads row neighbours, so the fit-time full-then-mask contract applies unchanged).

Non-parametric pair. Per-group ``quantile_residual`` / ``monotonic_residual`` fits with a James-Stein-style partial pooling of the per-group LEVEL:
groups with fewer than ``_GROUPED_MIN_GROUP_SIZE`` train rows (and groups unseen at predict time) fall back to the global fit; eligible groups run
their own fit and have their level (bin medians / spline knots) shrunk toward the global level by the classic JS factor computed on the per-group
median deviations (reusing :func:`~mlframe.training.composite.transforms.nonlinear._james_stein_shrinkage_factor`).

All parent / sibling imports are lazy (function-body) so this leaf module stays out of the whitelisted transforms import SCC.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _group_segments(groups: np.ndarray) -> list[tuple[Any, np.ndarray]]:
    """Return ``[(label, row_indices), ...]`` per unique group; ``row_indices`` are ascending (stable original order within the group)."""
    groups = np.asarray(groups).reshape(-1)
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=uniq.size)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    return [(uniq[i], order[offsets[i] : offsets[i + 1]]) for i in range(uniq.size)]


def _require_groups(groups: np.ndarray | None, name: str, op: str) -> np.ndarray:
    """Shared guard: grouped transforms need the ``groups`` kwarg on every call (wrapper extracts it from ``group_column``)."""
    if groups is None:
        raise ValueError(f"{name}.{op}: groups kwarg is required (configure ``group_column`` on the wrapper).")
    return np.asarray(groups).reshape(-1)


# ----------------------------------------------------------------------
# ewma_residual_grouped
# ----------------------------------------------------------------------

def _ewma_residual_grouped_fit(
    y: np.ndarray, base: np.ndarray, k: int | None = None,
    groups: np.ndarray | None = None,
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-group EWMA anchors (train-mean of the group's base; train-tail state as the continuation seed) + the shared span ``k``. Unseen groups at predict fall back to the global anchors."""
    from . import _EWMA_RESIDUAL_DEFAULT_K, _canonical_group_key
    from .nonlinear import _ewma_compute
    groups_arr = _require_groups(groups, "ewma_residual_grouped", "fit")
    k = max(1, int(k if k is not None else _EWMA_RESIDUAL_DEFAULT_K))
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    finite = np.isfinite(base_f)
    anchor = float(np.mean(base_f[finite])) if finite.any() else 0.0
    per_group_anchors: dict[str, float] = {}
    per_group_tail_anchors: dict[str, float] = {}
    tail_anchor = anchor
    for g, idx in _group_segments(groups_arr):
        seg = base_f[idx]
        seg_finite = seg[np.isfinite(seg)]
        a_g = float(seg_finite.mean()) if seg_finite.size else anchor
        key = _canonical_group_key(g)
        per_group_anchors[key] = a_g
        trace = _ewma_compute(seg, k, a_g)
        tf = trace[np.isfinite(trace)]
        per_group_tail_anchors[key] = float(tf[-1]) if tf.size else a_g
    return {
        "k": k, "anchor": anchor, "tail_anchor": tail_anchor,
        "per_group_anchors": per_group_anchors,
        "per_group_tail_anchors": per_group_tail_anchors,
    }


def _ewma_grouped_anchor(params: dict[str, Any], key: str) -> float:
    """Per-group anchor selection mirroring ``_ewma_anchor``: group mean anchor by default, group train-tail state under recurrence continuation; unseen groups fall back to the global anchor."""
    if params.get("recurrence_continuation"):
        tails = params.get("per_group_tail_anchors", {})
        if key in tails:
            return float(tails[key])
        return float(params.get("tail_anchor", params["anchor"]))
    return float(params.get("per_group_anchors", {}).get(key, params["anchor"]))


def _ewma_residual_grouped_apply(
    arr: np.ndarray, base: np.ndarray, params: dict[str, Any], groups: np.ndarray, sign: float,
) -> np.ndarray:
    """Shared forward/inverse body: ``arr + sign * EWMA_k(base_g)`` per group with per-group seeds (sign=-1 forward, +1 inverse)."""
    from . import _canonical_group_key
    from .nonlinear import _ewma_compute
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    arr_f = np.asarray(arr, dtype=np.float64).reshape(-1)
    out = np.empty(arr_f.size, dtype=np.float64)
    k = int(params["k"])
    for g, idx in _group_segments(groups):
        key = _canonical_group_key(g)
        anchor = _ewma_grouped_anchor(params, key) if sign > 0 else float(params.get("per_group_anchors", {}).get(key, params["anchor"]))
        trace = _ewma_compute(base_f[idx], k, anchor)
        out[idx] = arr_f[idx] + sign * trace
    return out


def _ewma_residual_grouped_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Apply ``T = y - EWMA_k(base)`` with the recurrence reset per group."""
    groups_arr = _require_groups(groups, "ewma_residual_grouped", "forward")
    return _ewma_residual_grouped_apply(y, base, params, groups_arr, sign=-1.0)


def _ewma_residual_grouped_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Undo the transform: ``y = T_hat + EWMA_k(base)`` per group (per-group tail seed under recurrence continuation)."""
    groups_arr = _require_groups(groups, "ewma_residual_grouped", "inverse")
    return _ewma_residual_grouped_apply(t_hat, base, params, groups_arr, sign=1.0)


def _ewma_residual_grouped_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Finite ``base`` (and finite ``y`` when provided), matching ``ewma_residual``."""
    from .nonlinear import _ewma_residual_domain
    return _ewma_residual_domain(y, base)


# ----------------------------------------------------------------------
# rolling_quantile_ratio_grouped
# ----------------------------------------------------------------------

def _rolling_quantile_ratio_grouped_fit(
    y: np.ndarray, base: np.ndarray, k: int | None = None, mode: str = "trailing",
    groups: np.ndarray | None = None,
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Same params as ``rolling_quantile_ratio`` (k / eps / mode); the eps floor is global (train base scale), the window is applied per group."""
    from .simple import _ROLLING_QUANTILE_DEFAULT_K, _rolling_quantile_ratio_fit
    _require_groups(groups, "rolling_quantile_ratio_grouped", "fit")
    k = int(k if k is not None else _ROLLING_QUANTILE_DEFAULT_K)
    return _rolling_quantile_ratio_fit(y, base, k=k, mode=mode, _finite_mask=_finite_mask)


def _rqr_grouped_median(base: np.ndarray, params: dict[str, Any], groups: np.ndarray) -> np.ndarray:
    """Rolling median of ``base`` computed independently within each group's stable-order subsequence."""
    from .simple import _rqr_rolling_median
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    out = np.empty(base_f.size, dtype=np.float64)
    k = int(params["k"])
    mode = str(params.get("mode", "trailing"))
    for _g, idx in _group_segments(groups):
        out[idx] = _rqr_rolling_median(base_f[idx], k, mode)
    return out


def _rolling_quantile_ratio_grouped_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Apply ``T = y / max(RollingMedian_k(base), eps)`` with the window confined to each row's group."""
    groups_arr = _require_groups(groups, "rolling_quantile_ratio_grouped", "forward")
    roll_med = _rqr_grouped_median(base, params, groups_arr)
    eps = float(params["eps"])
    safe = np.where(np.abs(roll_med) < eps, np.sign(roll_med + 1e-300) * eps, roll_med)
    return np.asarray(np.asarray(y, dtype=np.float64).reshape(-1) / safe)


def _rolling_quantile_ratio_grouped_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Undo the transform: ``y = T_hat * max(RollingMedian_k(base), eps)`` with the same per-group window."""
    groups_arr = _require_groups(groups, "rolling_quantile_ratio_grouped", "inverse")
    roll_med = _rqr_grouped_median(base, params, groups_arr)
    eps = float(params["eps"])
    safe = np.where(np.abs(roll_med) < eps, np.sign(roll_med + 1e-300) * eps, roll_med)
    return np.asarray(np.asarray(t_hat, dtype=np.float64).reshape(-1) * safe)


def _rolling_quantile_ratio_grouped_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Finite ``base`` (and finite ``y`` when provided), matching ``rolling_quantile_ratio``."""
    from .simple import _rolling_quantile_ratio_domain
    return _rolling_quantile_ratio_domain(y, base)


# ----------------------------------------------------------------------
# frac_diff_grouped (y-only, requires_base=False)
# ----------------------------------------------------------------------

def _frac_diff_grouped_fit(
    y: np.ndarray, base: np.ndarray | None,
    d: float | None = None, lags: int | None = None,
    groups: np.ndarray | None = None,
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Shared (d, lags, weights) + per-group pre-window anchors: each group's history pads with ITS OWN train-y mean (tail-mean as the continuation seed), so entity-level differences never leak across the boundary."""
    from . import _FRAC_DIFF_DEFAULT_D, _FRAC_DIFF_DEFAULT_LAGS, _canonical_group_key
    from .nonlinear import _frac_diff_weights
    groups_arr = _require_groups(groups, "frac_diff_grouped", "fit")
    d = float(d if d is not None else _FRAC_DIFF_DEFAULT_D)
    lags = max(1, int(lags if lags is not None else _FRAC_DIFF_DEFAULT_LAGS))
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(y_f)
    anchor = float(np.mean(y_f[finite])) if finite.any() else 0.0
    per_group_anchors: dict[str, float] = {}
    per_group_tail_anchors: dict[str, float] = {}
    for g, idx in _group_segments(groups_arr):
        seg = y_f[idx]
        seg_finite = seg[np.isfinite(seg)]
        key = _canonical_group_key(g)
        a_g = float(seg_finite.mean()) if seg_finite.size else anchor
        per_group_anchors[key] = a_g
        per_group_tail_anchors[key] = float(seg_finite[-lags:].mean()) if seg_finite.size else a_g
    return {
        "d": d, "lags": lags, "anchor": anchor, "tail_anchor": anchor,
        "weights": _frac_diff_weights(d, lags).tolist(),
        "per_group_anchors": per_group_anchors,
        "per_group_tail_anchors": per_group_tail_anchors,
    }


def _frac_diff_grouped_forward(
    y: np.ndarray, base: np.ndarray | None, params: dict[str, Any],
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Per-group truncated frac-diff convolution, padding each group's pre-window history with its own anchor."""
    from . import _canonical_group_key
    groups_arr = _require_groups(groups, "frac_diff_grouped", "forward")
    lags = int(params["lags"])
    weights = np.asarray(params["weights"], dtype=np.float64)
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    out = np.empty(y_f.size, dtype=np.float64)
    for g, idx in _group_segments(groups_arr):
        key = _canonical_group_key(g)
        anchor = float(params.get("per_group_anchors", {}).get(key, params["anchor"]))
        seg = y_f[idx]
        padded = np.concatenate([np.full(lags, anchor, dtype=np.float64), seg])
        out[idx] = np.convolve(padded, weights, mode="valid")
    return out


def _frac_diff_grouped_inverse(
    t_hat: np.ndarray, base: np.ndarray | None, params: dict[str, Any],
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Per-group iterative reconstruction via the shared njit-dispatched frac-diff-inverse kernel, each group seeded by its own anchor."""
    from . import _canonical_group_key
    from .nonlinear import _frac_diff_inverse_compute
    groups_arr = _require_groups(groups, "frac_diff_grouped", "inverse")
    lags = int(params["lags"])
    weights = np.ascontiguousarray(np.asarray(params["weights"], dtype=np.float64))
    t_f = np.asarray(t_hat, dtype=np.float64).reshape(-1)
    out = np.empty(t_f.size, dtype=np.float64)
    continuation = bool(params.get("recurrence_continuation"))
    for g, idx in _group_segments(groups_arr):
        key = _canonical_group_key(g)
        if continuation and key in params.get("per_group_tail_anchors", {}):
            anchor = float(params["per_group_tail_anchors"][key])
        else:
            anchor = float(params.get("per_group_anchors", {}).get(key, params["anchor"]))
        out[idx] = _frac_diff_inverse_compute(t_f[idx], lags, weights, anchor)
    return out


def _frac_diff_grouped_domain(
    y: np.ndarray | None, base: np.ndarray | None,
) -> np.ndarray:
    """y-only domain (finite y at fit time; base finiteness must not drop y rows), matching ``frac_diff``; at predict time (y=None) sized off whichever array is present."""
    if y is None:
        if base is None or not hasattr(base, "__len__"):
            return np.ones(1, dtype=bool)
        return np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1))
    return np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1))


# ----------------------------------------------------------------------
# Shared per-group fit machinery for quantile_residual_grouped / monotonic_residual_grouped
# ----------------------------------------------------------------------

def _grouped_level_shrinkage(
    y_f: np.ndarray, segments: list[tuple[Any, np.ndarray]], eligible_keys: set[str], global_median: float,
) -> tuple[float, dict[str, float]]:
    """Classic James-Stein factor on the per-group median deviations from the global median, plus the per-group level offsets ``c * (global - median_g)`` to add to that group's fitted level. Small K / zero spread returns c=0 (no shrink)."""
    from . import _canonical_group_key
    from .nonlinear import _james_stein_shrinkage_factor
    medians: list[float] = []
    sizes: list[float] = []
    keys: list[str] = []
    resid_sq = 0.0
    total_n = 0
    for g, idx in segments:
        key = _canonical_group_key(g)
        if key not in eligible_keys:
            continue
        seg = y_f[idx]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue
        med_g = float(np.median(seg))
        medians.append(med_g)
        sizes.append(float(seg.size))
        keys.append(key)
        resid_sq += float(np.sum((seg - med_g) ** 2))
        total_n += int(seg.size)
    if len(medians) < 4 or total_n <= len(medians):
        return 0.0, {}
    sigma2 = resid_sq / max(total_n - len(medians), 1)
    c = _james_stein_shrinkage_factor(
        np.asarray(medians, dtype=np.float64), global_median,
        np.asarray(sizes, dtype=np.float64), sigma2,
    )
    offsets = {k: c * (global_median - m) for k, m in zip(keys, medians)}
    return c, offsets


def _grouped_np_fit(
    y: np.ndarray, base: np.ndarray, groups: np.ndarray,
    fit_fn: Callable[..., dict[str, Any]],
    level_keys: tuple[str, ...],
    min_group_size: int | None,
    global_median_key: str,
) -> dict[str, Any]:
    """Per-group fit with global fallback + JS level shrinkage. ``level_keys`` name the ndarray/float params carrying the group's LEVEL (shifted by the JS offset); ``global_median_key`` names the global fit's level scalar used as the shrink center."""
    from . import _GROUPED_MIN_GROUP_SIZE, _canonical_group_key
    mgs = int(min_group_size if min_group_size is not None else _GROUPED_MIN_GROUP_SIZE)
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    global_params = fit_fn(y_f, base_f)
    segments = _group_segments(groups)
    per_group: dict[str, dict[str, Any]] = {}
    group_sizes: dict[str, int] = {}
    eligible: set[str] = set()
    for g, idx in segments:
        key = _canonical_group_key(g)
        n_g = int(idx.size)
        group_sizes[key] = n_g
        if n_g < mgs:
            continue
        try:
            per_group[key] = fit_fn(y_f[idx], base_f[idx])
            eligible.add(key)
        except Exception:  # pragma: no cover - defensive; group falls back to global
            continue
    global_median = float(global_params.get(global_median_key, 0.0))
    c, offsets = _grouped_level_shrinkage(y_f, segments, eligible, global_median)
    if c > 0:
        for key, off in offsets.items():
            p_g = per_group.get(key)
            if p_g is None or off == 0.0:
                continue
            for lk in level_keys:
                if lk in p_g:
                    if isinstance(p_g[lk], np.ndarray):
                        p_g[lk] = p_g[lk] + off
                    else:
                        p_g[lk] = float(p_g[lk]) + off
    return {
        "global": global_params,
        "per_group": per_group,
        "group_sizes": group_sizes,
        "min_group_size": mgs,
        "shrinkage_factor": float(c),
    }


def _grouped_np_apply(
    arr: np.ndarray, base: np.ndarray, params: dict[str, Any], groups: np.ndarray,
    apply_fn: Callable[[np.ndarray, np.ndarray, dict[str, Any]], np.ndarray],
) -> np.ndarray:
    """Route each group's rows through its own fitted params (global fallback for small / unseen groups) and scatter back in original order."""
    from . import _canonical_group_key
    arr_f = np.asarray(arr, dtype=np.float64).reshape(-1)
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    out = np.empty(arr_f.size, dtype=np.float64)
    per_group = params.get("per_group", {})
    global_params = params["global"]
    for g, idx in _group_segments(groups):
        p = per_group.get(_canonical_group_key(g), global_params)
        out[idx] = apply_fn(arr_f[idx], base_f[idx], p)
    return out


# ----------------------------------------------------------------------
# quantile_residual_grouped
# ----------------------------------------------------------------------

def _quantile_residual_grouped_fit(
    y: np.ndarray, base: np.ndarray,
    groups: np.ndarray | None = None,
    n_bins: int | None = None, min_bin_n: int | None = None,
    min_group_size: int | None = None,
) -> dict[str, Any]:
    """Per-group ``quantile_residual`` fits with JS level shrinkage of the per-group bin medians toward the global level; small / unseen groups use the global fit."""
    from . import _QUANTILE_RESIDUAL_DEFAULT_N_BINS, _QUANTILE_RESIDUAL_DEFAULT_MIN_BIN_N
    from .nonlinear import _quantile_residual_fit
    groups_arr = _require_groups(groups, "quantile_residual_grouped", "fit")
    nb = int(n_bins if n_bins is not None else _QUANTILE_RESIDUAL_DEFAULT_N_BINS)
    mbn = int(min_bin_n if min_bin_n is not None else _QUANTILE_RESIDUAL_DEFAULT_MIN_BIN_N)

    def _fit(y_g: np.ndarray, base_g: np.ndarray) -> dict[str, Any]:
        return _quantile_residual_fit(y_g, base_g, n_bins=nb, min_bin_n=mbn)

    return _grouped_np_fit(
        y, base, groups_arr, _fit,
        level_keys=("bin_medians", "global_median"),
        min_group_size=min_group_size,
        global_median_key="global_median",
    )


def _quantile_residual_grouped_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the per-group ``(y - median_bin) / IQR_bin`` (global params for small / unseen groups)."""
    from .nonlinear import _quantile_residual_forward
    groups_arr = _require_groups(groups, "quantile_residual_grouped", "forward")
    return _grouped_np_apply(y, base, params, groups_arr, _quantile_residual_forward)


def _quantile_residual_grouped_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Undo the per-group transform: ``y = T_hat * IQR_bin + median_bin`` with each row's group params."""
    from .nonlinear import _quantile_residual_inverse
    groups_arr = _require_groups(groups, "quantile_residual_grouped", "inverse")
    return _grouped_np_apply(t_hat, base, params, groups_arr, _quantile_residual_inverse)


def _quantile_residual_grouped_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Delegate to the ungrouped ``quantile_residual`` domain; grouping adds no restriction."""
    from .nonlinear import _quantile_residual_domain
    return _quantile_residual_domain(y, base)


# ----------------------------------------------------------------------
# monotonic_residual_grouped
# ----------------------------------------------------------------------

def _monotonic_residual_grouped_fit(
    y: np.ndarray, base: np.ndarray,
    groups: np.ndarray | None = None,
    n_knots: int | None = None, min_knot_n: int | None = None,
    min_group_size: int | None = None,
) -> dict[str, Any]:
    """Per-group monotone PCHIP fits with JS level shrinkage of the per-group knot values toward the global level; small / unseen groups use the global fit."""
    from . import _MONOTONIC_RESIDUAL_DEFAULT_N_KNOTS, _MONOTONIC_RESIDUAL_DEFAULT_MIN_KNOT_N
    from .nonlinear import _monotonic_residual_fit
    groups_arr = _require_groups(groups, "monotonic_residual_grouped", "fit")
    nk = int(n_knots if n_knots is not None else _MONOTONIC_RESIDUAL_DEFAULT_N_KNOTS)
    mkn = int(min_knot_n if min_knot_n is not None else _MONOTONIC_RESIDUAL_DEFAULT_MIN_KNOT_N)

    def _fit(y_g: np.ndarray, base_g: np.ndarray) -> dict[str, Any]:
        return _monotonic_residual_fit(y_g, base_g, n_knots=nk, min_knot_n=mkn)

    return _grouped_np_fit(
        y, base, groups_arr, _fit,
        level_keys=("knots_y", "y_train_mean"),
        min_group_size=min_group_size,
        global_median_key="y_train_mean",
    )


def _monotonic_residual_grouped_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the per-group ``T = y - g(base)`` with each row's group spline (global for small / unseen groups)."""
    from .nonlinear import _monotonic_residual_forward
    groups_arr = _require_groups(groups, "monotonic_residual_grouped", "forward")
    return _grouped_np_apply(y, base, params, groups_arr, _monotonic_residual_forward)


def _monotonic_residual_grouped_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Undo the per-group transform: ``y = T_hat + g(base)`` with each row's group spline."""
    from .nonlinear import _monotonic_residual_inverse
    groups_arr = _require_groups(groups, "monotonic_residual_grouped", "inverse")
    return _grouped_np_apply(t_hat, base, params, groups_arr, _monotonic_residual_inverse)


def _monotonic_residual_grouped_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Delegate to the ungrouped ``monotonic_residual`` domain; grouping adds no restriction."""
    from .nonlinear import _monotonic_residual_domain
    return _monotonic_residual_domain(y, base)
