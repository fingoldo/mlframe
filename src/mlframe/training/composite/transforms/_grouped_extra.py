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

import logging
from typing import Any, Callable, Optional, cast

import numpy as np

logger = logging.getLogger(__name__)


def _group_segments(groups: np.ndarray) -> list[tuple[Any, np.ndarray]]:
    """Return ``[(label, row_indices), ...]`` per unique group; ``row_indices`` are ascending (stable original order within the group)."""
    groups = np.asarray(groups).reshape(-1)
    from . import _unique_group_labels
    uniq, inv = _unique_group_labels(groups)
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
# Shared history plumbing for the grouped recurrent trio
# ----------------------------------------------------------------------

def _group_history(history: np.ndarray | None, history_groups: np.ndarray | None, key: str) -> np.ndarray | None:
    """Rows of ``history`` (the rows immediately preceding a batch, in order) that belong to group ``key``; ``None`` when no history was given.

    ``history_groups`` must accompany ``history`` (a history without labels cannot be split per group)."""
    if history is None:
        return None
    if history_groups is None:
        raise ValueError("grouped recurrent transform: a history array needs matching history_groups labels.")
    from . import _canonical_group_key
    h = np.asarray(history, dtype=np.float64).reshape(-1)
    hg = np.asarray(history_groups).reshape(-1)
    if hg.size != h.size:
        raise ValueError(f"grouped recurrent transform: history has {h.size} rows but history_groups has {hg.size}.")
    for g, idx in _group_segments(hg):
        if _canonical_group_key(g) == key:
            return cast(Optional[np.ndarray], h[idx])
    return h[:0]


# ----------------------------------------------------------------------
# ewma_residual_grouped
# ----------------------------------------------------------------------

def _ewma_residual_grouped_fit(
    y: np.ndarray, base: np.ndarray, k: int | None = None,
    groups: np.ndarray | None = None,
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-group EWMA anchors (train-mean of the group's base; train-tail state as the continuation seed) + the shared span ``k``. Unseen groups at
    predict fall back to the global anchors: the global mean by default, and under recurrence continuation the train-tail state of the ungrouped
    EWMA over the whole base series (the same seed ``ewma_residual`` would use), not the whole-history mean."""
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
    if finite.any():
        _global_trace = _ewma_compute(base_f, k, anchor)
        _gtf = _global_trace[np.isfinite(_global_trace)]
        if _gtf.size:
            tail_anchor = float(_gtf[-1])
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
    history_base: np.ndarray | None = None, history_groups: np.ndarray | None = None,
) -> np.ndarray:
    """Shared forward/inverse body: ``arr + sign * EWMA_k(base_g)`` per group with per-group seeds (sign=-1 forward, +1 inverse); each group's
    EWMA first runs over that group's ``history_base`` rows, so a batch scored with its preceding rows equals the same rows inside a longer batch."""
    from . import _canonical_group_key
    from .nonlinear import _ewma_compute
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    arr_f = np.asarray(arr, dtype=np.float64).reshape(-1)
    out = np.empty(arr_f.size, dtype=np.float64)
    k = int(params["k"])
    for g, idx in _group_segments(groups):
        key = _canonical_group_key(g)
        anchor = _ewma_grouped_anchor(params, key) if sign > 0 else float(params.get("per_group_anchors", {}).get(key, params["anchor"]))
        hist = _group_history(history_base, history_groups, key)
        seg = base_f[idx] if hist is None else np.concatenate([hist, base_f[idx]])
        h = 0 if hist is None else hist.size
        trace = _ewma_compute(seg, k, anchor)[h:]
        out[idx] = arr_f[idx] + sign * trace
    return out


def _ewma_residual_grouped_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
    groups: np.ndarray | None = None,
    history_base: np.ndarray | None = None, history_groups: np.ndarray | None = None,
) -> np.ndarray:
    """Apply ``T = y - EWMA_k(base)`` with the recurrence reset per group."""
    groups_arr = _require_groups(groups, "ewma_residual_grouped", "forward")
    return _ewma_residual_grouped_apply(y, base, params, groups_arr, sign=-1.0, history_base=history_base, history_groups=history_groups)


def _ewma_residual_grouped_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
    groups: np.ndarray | None = None,
    history_base: np.ndarray | None = None, history_groups: np.ndarray | None = None,
) -> np.ndarray:
    """Undo the transform: ``y = T_hat + EWMA_k(base)`` per group (per-group tail seed under recurrence continuation)."""
    from ._nonlinear_ewma_fracdiff import _warn_cold_recurrence
    groups_arr = _require_groups(groups, "ewma_residual_grouped", "inverse")
    _warn_cold_recurrence("ewma_residual_grouped", int(np.asarray(t_hat).size), int(params["k"]), params, history_base)
    return _ewma_residual_grouped_apply(t_hat, base, params, groups_arr, sign=1.0, history_base=history_base, history_groups=history_groups)


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
    """Same params as ``rolling_quantile_ratio`` (k / eps / mode); the eps floor is global (train base scale), the window is applied per group.
    Each group's last k-1 train base values are stored as its continuation window history."""
    from . import _canonical_group_key
    from .simple import _ROLLING_QUANTILE_DEFAULT_K, _rolling_quantile_ratio_fit
    groups_arr = _require_groups(groups, "rolling_quantile_ratio_grouped", "fit")
    k = int(k if k is not None else _ROLLING_QUANTILE_DEFAULT_K)
    params = _rolling_quantile_ratio_fit(y, base, k=k, mode=mode, _finite_mask=_finite_mask)
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    per_group_tail_base: dict[str, list[float]] = {}
    kk = int(params["k"])
    for g, idx in _group_segments(groups_arr):
        seg = base_f[idx]
        seg = seg[np.isfinite(seg)]
        per_group_tail_base[_canonical_group_key(g)] = [float(v) for v in (seg[-(kk - 1) :] if kk > 1 else seg[:0])]
    params["per_group_tail_base"] = per_group_tail_base
    return params


def _rqr_grouped_median(
    base: np.ndarray, params: dict[str, Any], groups: np.ndarray,
    history_base: np.ndarray | None = None, history_groups: np.ndarray | None = None, continuation: bool = False,
) -> np.ndarray:
    """Rolling median of ``base`` computed independently within each group's stable-order subsequence; each group's window also reads its
    ``history_base`` rows and, under recurrence continuation (inverse only), its stored train tail (an unseen group: the ungrouped tail)."""
    from . import _canonical_group_key
    from .simple import _rqr_rolling_median
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    out = np.empty(base_f.size, dtype=np.float64)
    k = int(params["k"])
    mode = str(params.get("mode", "trailing"))
    use_tail = continuation and bool(params.get("recurrence_continuation"))
    for g, idx in _group_segments(groups):
        key = _canonical_group_key(g)
        parts = []
        if use_tail:
            # An unseen group continues from the ungrouped series' tail, the seed ``rolling_quantile_ratio`` uses; an empty
            # prefix restarted its window cold, so its first k-1 rows took the median of a truncated window.
            parts.append(np.asarray(params.get("per_group_tail_base", {}).get(key, params.get("tail_base", [])), dtype=np.float64))
        hist = _group_history(history_base, history_groups, key)
        if hist is not None:
            parts.append(hist)
        prefix = np.concatenate(parts) if parts else base_f[:0]
        out[idx] = _rqr_rolling_median(np.concatenate([prefix, base_f[idx]]), k, mode)[prefix.size :]
    return out


def _rolling_quantile_ratio_grouped_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
    groups: np.ndarray | None = None,
    history_base: np.ndarray | None = None, history_groups: np.ndarray | None = None,
) -> np.ndarray:
    """Apply ``T = y / max(RollingMedian_k(base), eps)`` with the window confined to each row's group."""
    groups_arr = _require_groups(groups, "rolling_quantile_ratio_grouped", "forward")
    roll_med = _rqr_grouped_median(base, params, groups_arr, history_base, history_groups, continuation=False)
    eps = float(params["eps"])
    safe = np.where(np.abs(roll_med) < eps, np.sign(roll_med + 1e-300) * eps, roll_med)
    return np.asarray(np.asarray(y, dtype=np.float64).reshape(-1) / safe)


def _rolling_quantile_ratio_grouped_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
    groups: np.ndarray | None = None,
    history_base: np.ndarray | None = None, history_groups: np.ndarray | None = None,
) -> np.ndarray:
    """Undo the transform: ``y = T_hat * max(RollingMedian_k(base), eps)`` with the same per-group window."""
    from ._nonlinear_ewma_fracdiff import _warn_cold_recurrence
    groups_arr = _require_groups(groups, "rolling_quantile_ratio_grouped", "inverse")
    _warn_cold_recurrence("rolling_quantile_ratio_grouped", int(np.asarray(t_hat).size), int(params["k"]), params, history_base)
    roll_med = _rqr_grouped_median(base, params, groups_arr, history_base, history_groups, continuation=True)
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

def _tail_values(seg: np.ndarray, lags: int, pad: float) -> list[float]:
    """The last ``lags`` finite values of ``seg`` (oldest first), left-padded with ``pad`` when the segment is shorter."""
    fin = seg[np.isfinite(seg)][-lags:]
    out = np.full(lags, pad, dtype=np.float64)
    out[lags - fin.size :] = fin
    return [float(v) for v in out]


def _frac_diff_grouped_fit(
    y: np.ndarray, base: np.ndarray | None,
    d: float | None = None, lags: int | None = None,
    groups: np.ndarray | None = None,
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Shared (d, lags, weights) + per-group pre-window anchors: each group's history pads with ITS OWN train-y mean, so entity-level differences
    never leak across the boundary. Continuation seeds are each group's actual last ``lags`` train values; an unseen group under continuation
    falls back to the ungrouped series' tail (the ``frac_diff`` continuation seed), not the whole-history mean."""
    from . import _FRAC_DIFF_DEFAULT_D, _FRAC_DIFF_DEFAULT_LAGS, _canonical_group_key
    from .nonlinear import _frac_diff_weights
    groups_arr = _require_groups(groups, "frac_diff_grouped", "fit")
    d = float(d if d is not None else _FRAC_DIFF_DEFAULT_D)
    lags = max(1, int(lags if lags is not None else _FRAC_DIFF_DEFAULT_LAGS))
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(y_f)
    anchor = float(np.mean(y_f[finite])) if finite.any() else 0.0
    y_fin = y_f[finite]
    tail_anchor = float(np.mean(y_fin[-lags:])) if y_fin.size else anchor
    per_group_anchors: dict[str, float] = {}
    per_group_tail_anchors: dict[str, float] = {}
    per_group_tail_y: dict[str, list[float]] = {}
    for g, idx in _group_segments(groups_arr):
        seg = y_f[idx]
        seg_finite = seg[np.isfinite(seg)]
        key = _canonical_group_key(g)
        a_g = float(seg_finite.mean()) if seg_finite.size else anchor
        per_group_anchors[key] = a_g
        per_group_tail_anchors[key] = float(seg_finite[-lags:].mean()) if seg_finite.size else a_g
        per_group_tail_y[key] = _tail_values(seg, lags, a_g)
    return {
        "d": d, "lags": lags, "anchor": anchor, "tail_anchor": tail_anchor,
        "tail_y": _tail_values(y_f, lags, anchor),
        "weights": _frac_diff_weights(d, lags).tolist(),
        "per_group_anchors": per_group_anchors,
        "per_group_tail_anchors": per_group_tail_anchors,
        "per_group_tail_y": per_group_tail_y,
    }


def _frac_diff_grouped_forward(
    y: np.ndarray, base: np.ndarray | None, params: dict[str, Any],
    groups: np.ndarray | None = None,
    history_y: np.ndarray | None = None, history_groups: np.ndarray | None = None,
) -> np.ndarray:
    """Per-group truncated frac-diff convolution, padding each group's pre-window history with its own anchor (or its ``history_y`` rows)."""
    from . import _canonical_group_key
    from ._nonlinear_ewma_fracdiff import _history_prefix
    groups_arr = _require_groups(groups, "frac_diff_grouped", "forward")
    lags = int(params["lags"])
    weights = np.asarray(params["weights"], dtype=np.float64)
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    out = np.empty(y_f.size, dtype=np.float64)
    for g, idx in _group_segments(groups_arr):
        key = _canonical_group_key(g)
        anchor = float(params.get("per_group_anchors", {}).get(key, params["anchor"]))
        prefix = _history_prefix(np.full(lags, anchor, dtype=np.float64), _group_history(history_y, history_groups, key), lags)
        padded = np.concatenate([prefix, y_f[idx]])
        out[idx] = np.convolve(padded, weights, mode="valid")
    return out


def _frac_diff_grouped_inverse(
    t_hat: np.ndarray, base: np.ndarray | None, params: dict[str, Any],
    groups: np.ndarray | None = None,
    history_y: np.ndarray | None = None, history_groups: np.ndarray | None = None,
) -> np.ndarray:
    """Per-group iterative reconstruction via the shared njit-dispatched frac-diff-inverse kernel, each group seeded by its own anchor, by its
    actual train tail under recurrence continuation, and by its observed ``history_y`` rows when supplied."""
    from . import _canonical_group_key
    from .nonlinear import _frac_diff_inverse_compute
    from ._nonlinear_ewma_fracdiff import _frac_diff_inverse_prefix, _history_prefix, _warn_cold_recurrence
    groups_arr = _require_groups(groups, "frac_diff_grouped", "inverse")
    lags = int(params["lags"])
    weights = np.ascontiguousarray(np.asarray(params["weights"], dtype=np.float64))
    t_f = np.asarray(t_hat, dtype=np.float64).reshape(-1)
    _warn_cold_recurrence("frac_diff_grouped", t_f.size, lags, params, history_y)
    out = np.empty(t_f.size, dtype=np.float64)
    continuation = bool(params.get("recurrence_continuation"))
    for g, idx in _group_segments(groups_arr):
        key = _canonical_group_key(g)
        hist = _group_history(history_y, history_groups, key)
        if continuation and "per_group_tail_y" in params:
            default_prefix = np.asarray(params["per_group_tail_y"].get(key, params.get("tail_y", [params["anchor"]] * lags)), dtype=np.float64)
        elif continuation and key in params.get("per_group_tail_anchors", {}):
            default_prefix = np.full(lags, float(params["per_group_tail_anchors"][key]), dtype=np.float64)
        else:
            default_prefix = np.full(lags, float(params.get("per_group_anchors", {}).get(key, params["anchor"])), dtype=np.float64)
        if hist is None and not (continuation and "per_group_tail_y" in params):
            out[idx] = _frac_diff_inverse_compute(t_f[idx], lags, weights, float(default_prefix[-1]))
        else:
            out[idx] = _frac_diff_inverse_prefix(t_f[idx], lags, weights, _history_prefix(default_prefix, hist, lags))
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
        except Exception as exc:  # pragma: no cover - defensive; group falls back to global
            logger.debug("grouped-transform per-group fit failed for group %r, falling back to global: %s", key, exc)
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
        """Fit the ungrouped ``quantile_residual`` transform on one group's rows."""
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
        """Fit the ungrouped monotonic-residual PCHIP transform on one group's rows."""
        return _monotonic_residual_fit(y_g, base_g, n_knots=nk, min_knot_n=mkn)

    return _grouped_np_fit(
        y, base, groups_arr, _fit,
        level_keys=("knots_y", "y_train_mean"),
        min_group_size=min_group_size,
        # Per-group levels are MEDIANS, so the shrink centre must be the global median: against the global mean, a skewed target's mean-median gap
        # moved every group's knots the same way, even a group drawn from the global population.
        global_median_key="y_train_median",
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
