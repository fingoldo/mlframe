"""Per-group anchor-based linear extrapolation features.

Use case: in panel / time-series / clustered data you have a sparse
ground-truth label that's known on SOME rows ("anchors") but missing
on others. Standard rolling features can't access the latent label
trend because it's mostly NaN. This module computes per-row features
that anchor on the LAST KNOWN ground-truth in the same group and
project forward linearly.

Examples of "anchor" rows:

* sparse manually-labelled rows in a high-frequency stream
* periodic ground-truth measurements between which the target is
  unobserved (medical vitals, geomagnetic survey)
* pre-PS rows in wellbore geosteering (the case this was first
  developed for)

Features emitted:

* ``rows_since_last_anchor`` -- integer rows since the last
  anchor in the same group (0 on anchor rows themselves).
* ``last_anchor_value`` -- the label value of the last anchor
  (a "stale ground truth" feature).
* ``last_anchor_local_slope`` -- slope of the linear regression
  through the last ``K`` anchors at that position (estimate of
  the local label trend).
* ``linear_extrap_pred`` -- naive linear projection of the label
  from the last anchor: ``last_anchor_value + slope *
  rows_since_last_anchor``.

All features are per-row, leak-safe (only past anchors visible to
each row), and computed via a single per-group pass.
"""

from __future__ import annotations

__all__ = [
    "add_anchor_extrapolation_features",
    "anchor_residual_rmse_features",
    "anchor_quadratic_extrapolation_features",
    "anchor_ewm_features",
    "anchor_density_features",
    "rows_until_next_anchor",
]

import math
from typing import Callable, Optional

import numpy as np

from .grouped import iter_group_segments

try:
    import numba as _numba
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:
    # numba cores for the per-segment anchor recursions. The Python
    # versions use growing/shrinking lists (append/pop); these mirror the
    # exact semantics with preallocated buffers + window indices so a
    # @njit can compile them. Each writes into pre-sliced output arrays.
    _ANCHOR_FASTMATH = {"reassoc", "contract", "arcp", "afn"}

    @_numba.njit(cache=True, fastmath=_ANCHOR_FASTMATH)
    def _anchor_rmse_core(label, is_anchor, K_slope, K_rmse, residual_out, rmse_out):
        """Numba core for ``anchor_residual_rmse_features``: walks one segment once, maintaining a growing window of up to ``K_slope`` past anchors to compute a leave-one-out linear-extrapolation residual at each new anchor, and a rolling RMSE over the last ``K_rmse`` such residuals. Writes ``residual_out``/``rmse_out`` in place (last-known-value carried forward between anchors)."""
        m = label.size
        pos = np.empty(m, dtype=np.float64)
        val = np.empty(m, dtype=np.float64)
        res = np.empty(m, dtype=np.float64)
        n_anch = 0
        n_res = 0
        last_carry = np.nan
        for i in range(m):
            if is_anchor[i] and np.isfinite(label[i]):
                if n_anch >= 2:
                    lo = n_anch - K_slope
                    if lo < 0:
                        lo = 0
                    w = n_anch - lo
                    xm = 0.0; ym = 0.0
                    for j in range(lo, n_anch):
                        xm += pos[j]; ym += val[j]
                    xm /= w; ym /= w
                    num = 0.0; den = 0.0
                    for j in range(lo, n_anch):
                        dx = pos[j] - xm
                        num += dx * (val[j] - ym)
                        den += dx * dx
                    slope = num / (den + 1e-12)
                    pred = val[n_anch - 1] + slope * (i - pos[n_anch - 1])
                    e_j = label[i] - pred
                    res[n_res] = e_j
                    n_res += 1
                    last_carry = e_j
                pos[n_anch] = i
                val[n_anch] = label[i]
                n_anch += 1
            residual_out[i] = last_carry
            if n_res > 0:
                lo = n_res - K_rmse
                if lo < 0:
                    lo = 0
                ss = 0.0; cnt = 0
                for j in range(lo, n_res):
                    ss += res[j] * res[j]
                    cnt += 1
                rmse_out[i] = math.sqrt(ss / cnt)

    @_numba.njit(cache=True, fastmath=_ANCHOR_FASTMATH)
    def _anchor_quadratic_core(label, is_anchor, K_window, accel_out, quad_out):
        """Numba core for ``anchor_quadratic_extrapolation_features``: fits a quadratic ``y = a + b*dx + c*dx^2`` (dx centred on the most recent anchor) via the 3x3 normal equations over the last ``K_window`` anchors, solved by closed-form Cramer's rule, and writes the acceleration coefficient ``c`` and the extrapolated prediction at each row. No-ops (leaves NaN) while fewer than 3 anchors are available or the normal-equations matrix is singular."""
        m = label.size
        pos = np.empty(m, dtype=np.float64)
        val = np.empty(m, dtype=np.float64)
        n_anch = 0
        for i in range(m):
            if is_anchor[i] and np.isfinite(label[i]):
                pos[n_anch] = i
                val[n_anch] = label[i]
                n_anch += 1
            lo = n_anch - K_window
            if lo < 0:
                lo = 0
            cnt = n_anch - lo
            if cnt >= 3:
                last_r = pos[n_anch - 1]
                # Normal equations for y = a + b*dx + c*dx^2 (dx centred on last).
                s0 = 0.0; s1 = 0.0; s2 = 0.0; s3 = 0.0; s4 = 0.0
                b0 = 0.0; b1 = 0.0; b2 = 0.0
                for j in range(lo, n_anch):
                    dx = pos[j] - last_r
                    dx2 = dx * dx
                    yj = val[j]
                    s0 += 1.0; s1 += dx; s2 += dx2
                    s3 += dx2 * dx; s4 += dx2 * dx2
                    b0 += yj; b1 += yj * dx; b2 += yj * dx2
                # Solve 3x3 [[s0,s1,s2],[s1,s2,s3],[s2,s3,s4]] @ [a,b,c] = [b0,b1,b2]
                m00 = s0; m01 = s1; m02 = s2
                m10 = s1; m11 = s2; m12 = s3
                m20 = s2; m21 = s3; m22 = s4
                det = (m00 * (m11 * m22 - m12 * m21)
                       - m01 * (m10 * m22 - m12 * m20)
                       + m02 * (m10 * m21 - m11 * m20))
                if abs(det) > 1e-300:
                    inv_det = 1.0 / det
                    # Cramer for c (coef[2]) and full solve for prediction.
                    a = (b0 * (m11 * m22 - m12 * m21) - m01 * (b1 * m22 - m12 * b2) + m02 * (b1 * m21 - m11 * b2)) * inv_det
                    b = (m00 * (b1 * m22 - m12 * b2) - b0 * (m10 * m22 - m12 * m20) + m02 * (m10 * b2 - b1 * m20)) * inv_det
                    c = (m00 * (m11 * b2 - b1 * m21) - m01 * (m10 * b2 - b1 * m20) + b0 * (m10 * m21 - m11 * m20)) * inv_det
                    dx_now = i - last_r
                    accel_out[i] = c
                    quad_out[i] = a + b * dx_now + c * dx_now * dx_now

    @_numba.njit(cache=True, fastmath=_ANCHOR_FASTMATH)
    def _anchor_ewm_core(label, is_anchor, half_life, ewm_val_out, ewm_slope_out):
        """Numba core for ``anchor_ewm_features``: O(1)-per-row exponentially-weighted anchor mean + weighted-OLS slope, using the recurrence documented below (decay-then-shift of five running accumulators S0/Sy/Su/Suy/Suu) instead of an O(A) resum over all anchors at every row."""
        # O(1)-per-step EWMA recurrence. Instead of recomputing decayed sums over
        # ALL anchors at every row (O(A) per row => O(A^2) per segment), keep
        # running accumulators in coordinates centred on the CURRENT row i, so the
        # x-magnitudes stay bounded and the per-step update is O(1).
        #
        # Let r = 0.5^(1/H) be the per-row decay and u_j = pos_j - i (<= 0). At
        # row i the weight of anchor j is w_j = r^(i - pos_j) = r^(-u_j). We track:
        #   S0  = sum_j w_j
        #   Sy  = sum_j w_j * y_j
        #   Su  = sum_j w_j * u_j
        #   Suy = sum_j w_j * u_j * y_j
        #   Suu = sum_j w_j * u_j^2
        # Stepping i -> i+1: every weight gains a factor r (decay), and every u_j
        # shifts by -1. Decay scales all accumulators by r; the u-shift transforms
        #   Su  -> Su  - S0
        #   Suu -> Suu - 2*Su + S0
        #   Suy -> Suy - Sy
        # (order: shift uses the pre-shift Su/Sy). A new anchor at u=0 adds
        # (1, y, 0, 0, 0). Then, since the slope is translation-invariant in x,
        #   w_mean = Sy/S0, xm_u = Su/S0,
        #   num = Suy - Su*Sy/S0,  den = Suu - Su^2/S0.
        m = label.size
        r = 0.5 ** (1.0 / half_life)
        S0 = 0.0; Sy = 0.0; Su = 0.0; Suy = 0.0; Suu = 0.0
        n_anch = 0
        for i in range(m):
            if i > 0 and n_anch > 0:
                # decay (factor r on every w_j) then shift (u_j -> u_j - 1)
                S0 *= r; Sy *= r; Su *= r; Suy *= r; Suu *= r
                Suu = Suu - 2.0 * Su + S0
                Suy = Suy - Sy
                Su = Su - S0
            if is_anchor[i] and np.isfinite(label[i]):
                y = label[i]
                S0 += 1.0
                Sy += y
                # new anchor sits at u = 0 -> contributes nothing to Su/Suy/Suu
                n_anch += 1
            if n_anch == 0:
                continue
            w_mean = Sy / S0
            ewm_val_out[i] = w_mean
            if n_anch >= 2:
                num = Suy - Su * Sy / S0
                den = Suu - Su * Su / S0
                ewm_slope_out[i] = num / (den + 1e-12)

    @_numba.njit(cache=True, fastmath=_ANCHOR_FASTMATH)
    def _anchor_density_core(is_anchor, window_rows, count_out, gap_out):
        """Numba core for ``anchor_density_features``: sliding-window (two-pointer) count of anchors within the trailing ``window_rows`` and the mean gap between them, O(n) total via a monotonic head index instead of rescanning the window at each row."""
        m = is_anchor.size
        pos = np.empty(m, dtype=np.float64)
        n_anch = 0
        head = 0
        for i in range(m):
            if is_anchor[i]:
                pos[n_anch] = i
                n_anch += 1
            while head < n_anch and (i - pos[head]) >= window_rows:
                head += 1
            cnt = n_anch - head
            count_out[i] = float(cnt)
            if cnt >= 2:
                gap_out[i] = (pos[n_anch - 1] - pos[head]) / (cnt - 1)


def _anchor_features_for_segment(
    label: np.ndarray,
    is_anchor: np.ndarray,
    K_slope: int,
) -> dict:
    """Compute the four anchor features for ONE group segment.

    ``label[i]`` is meaningful only when ``is_anchor[i]`` is True; the
    function ignores label values at non-anchor rows.

    Returns dict of 4 arrays each of length ``len(label)``.
    """
    n = label.size
    rows_since = np.full(n, np.nan, dtype=np.float64)
    last_val = np.full(n, np.nan, dtype=np.float64)
    local_slope = np.full(n, np.nan, dtype=np.float64)
    extrap = np.full(n, np.nan, dtype=np.float64)

    # Walk the segment once, tracking the rolling window of last K anchors.
    anchor_positions: list = []
    anchor_values: list = []
    last_anchor_row: int = -1
    last_anchor_val: float = np.nan
    for i in range(n):
        if is_anchor[i] and np.isfinite(label[i]):
            anchor_positions.append(i)
            anchor_values.append(float(label[i]))
            # Trim window to last K anchors
            if len(anchor_positions) > K_slope:
                anchor_positions.pop(0)
                anchor_values.pop(0)
            last_anchor_row = i
            last_anchor_val = float(label[i])
            rows_since[i] = 0.0
            last_val[i] = last_anchor_val
            extrap[i] = last_anchor_val  # on the anchor itself
        else:
            if last_anchor_row >= 0:
                rows_since[i] = float(i - last_anchor_row)
                last_val[i] = last_anchor_val
        # Compute / cache slope over the rolling anchor window.
        if len(anchor_positions) >= 2:
            xs = np.asarray(anchor_positions, dtype=np.float64)
            ys = np.asarray(anchor_values, dtype=np.float64)
            xm = xs.mean()
            ym = ys.mean()
            num = ((xs - xm) * (ys - ym)).sum()
            den = ((xs - xm) ** 2).sum()
            slope = num / den if den > 1e-12 else 0.0
            local_slope[i] = slope
            if last_anchor_row >= 0 and not is_anchor[i]:
                extrap[i] = last_anchor_val + slope * (i - last_anchor_row)
        elif len(anchor_positions) == 1:
            local_slope[i] = 0.0
            if last_anchor_row >= 0 and not is_anchor[i]:
                # No slope estimate yet; flat extrapolation.
                extrap[i] = last_anchor_val
    return {
        "rows_since": rows_since,
        "last_anchor_value": last_val,
        "local_slope": local_slope,
        "linear_extrap_pred": extrap,
    }


def add_anchor_extrapolation_features(
    label: np.ndarray,
    is_anchor: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    *,
    K_slope: int = 10,
) -> dict:
    """Compute four leak-safe anchor-based features per row.

    Parameters
    ----------
    label
        1-D float array. Values are MEANINGFUL only at rows where
        ``is_anchor`` is True; values on non-anchor rows are ignored.
        Use NaN at non-anchor rows as a sentinel if convenient.
    is_anchor
        1-D bool array. ``True`` = this row has a known label value
        in ``label``; ``False`` = label is missing / unobserved.
    group_ids
        Optional per-row group identifier. When supplied, anchors do
        NOT cross group boundaries: row ``i`` only sees anchors from
        the same group. When ``None``, the whole array is one group.
    K_slope
        Rolling-window length (in anchors, NOT rows) for the local
        slope estimate. Default 10. Smaller = more responsive to
        recent anchors; larger = smoother / more biased estimate.

    Returns
    -------
    dict
        ``{
            "rows_since_last_anchor": np.ndarray,
            "last_anchor_value": np.ndarray,
            "last_anchor_local_slope_K{K}": np.ndarray,
            "linear_extrap_pred_K{K}": np.ndarray,
        }``

    All four arrays have shape ``(len(label),)`` with NaN on rows
    that have no prior anchor in the same group.
    """
    label = np.ascontiguousarray(label, dtype=np.float64)
    is_anchor = np.ascontiguousarray(is_anchor, dtype=bool)
    n = label.size
    if is_anchor.size != n:
        raise ValueError(f"label / is_anchor length mismatch: {n} vs {is_anchor.size}")
    if K_slope < 2:
        raise ValueError(f"K_slope must be >= 2, got {K_slope}")

    out = {
        "rows_since_last_anchor": np.full(n, np.nan, dtype=np.float64),
        "last_anchor_value": np.full(n, np.nan, dtype=np.float64),
        f"last_anchor_local_slope_K{K_slope}": np.full(n, np.nan, dtype=np.float64),
        f"linear_extrap_pred_K{K_slope}": np.full(n, np.nan, dtype=np.float64),
    }

    if group_ids is None:
        feats = _anchor_features_for_segment(label, is_anchor, K_slope)
        out["rows_since_last_anchor"] = feats["rows_since"]
        out["last_anchor_value"] = feats["last_anchor_value"]
        out[f"last_anchor_local_slope_K{K_slope}"] = feats["local_slope"]
        out[f"linear_extrap_pred_K{K_slope}"] = feats["linear_extrap_pred"]
        return out

    sort_idx, starts, ends = iter_group_segments(group_ids)
    for s, e in zip(starts, ends):
        idx_seg = sort_idx[s:e]
        seg_label = label[idx_seg]
        seg_anchor = is_anchor[idx_seg]
        feats = _anchor_features_for_segment(seg_label, seg_anchor, K_slope)
        out["rows_since_last_anchor"][idx_seg] = feats["rows_since"]
        out["last_anchor_value"][idx_seg] = feats["last_anchor_value"]
        out[f"last_anchor_local_slope_K{K_slope}"][idx_seg] = feats["local_slope"]
        out[f"linear_extrap_pred_K{K_slope}"][idx_seg] = feats["linear_extrap_pred"]
    return out


def _per_group_anchor_loop(
    label: np.ndarray,
    is_anchor: np.ndarray,
    group_ids: Optional[np.ndarray],
    per_segment_fn: Callable[..., dict],
) -> dict:
    """Shared per-group dispatch for the anchor-feature family."""
    label = np.ascontiguousarray(label, dtype=np.float64)
    is_anchor = np.ascontiguousarray(is_anchor, dtype=bool)
    if group_ids is None:
        return per_segment_fn(label, is_anchor, np.arange(label.size))
    sort_idx, starts, ends = iter_group_segments(group_ids)
    out_keys: dict = {}
    for s, e in zip(starts, ends):
        idx_seg = sort_idx[s:e]
        seg_label = label[idx_seg]
        seg_anchor = is_anchor[idx_seg]
        seg_out = per_segment_fn(seg_label, seg_anchor, idx_seg)
        if not out_keys:
            for k, _v in seg_out.items():
                out_keys[k] = np.full(label.size, np.nan, dtype=np.float64)
        for k, v in seg_out.items():
            out_keys[k][idx_seg] = v
    return out_keys


def _run_anchor_numba_2out(label, is_anchor, group_ids, runner, key1, key2) -> dict:
    """Per-group dispatch for a 2-output numba anchor core. ``runner`` is a
    closure ``(seg_label, seg_anchor, out1_slice, out2_slice) -> None`` that
    invokes the njit core with its captured K / half-life / window params.
    Groups are independent, so this loops per sorted segment (the heavy work
    is inside the njit core; the per-segment call overhead is negligible)."""
    label = np.ascontiguousarray(label, dtype=np.float64)
    is_anchor = np.ascontiguousarray(is_anchor, dtype=np.bool_)
    n = label.size
    o1 = np.full(n, np.nan, dtype=np.float64)
    o2 = np.full(n, np.nan, dtype=np.float64)
    if group_ids is None:
        runner(label, is_anchor, o1, o2)
        return {key1: o1, key2: o2}
    sort_idx, starts, ends = iter_group_segments(group_ids)
    ls = np.ascontiguousarray(label[sort_idx])
    as_ = np.ascontiguousarray(is_anchor[sort_idx])
    s1 = np.full(n, np.nan, dtype=np.float64)
    s2 = np.full(n, np.nan, dtype=np.float64)
    for s, e in zip(starts, ends):
        runner(ls[s:e], as_[s:e], s1[s:e], s2[s:e])
    o1[sort_idx] = s1
    o2[sort_idx] = s2
    return {key1: o1, key2: o2}


def anchor_residual_rmse_features(
    label: np.ndarray,
    is_anchor: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    *,
    K_slope: int = 10,
    K_rmse: int = 10,
) -> dict:
    """Leave-one-out residual + rolling RMSE of linear-extrap predictions.

    For each anchor ``a_j`` (using only anchors with row < j), compute
    the linear-extrap residual ``e_j = y_j - (last_val + slope * dr)``.
    For each row ``i``, emit:
    * ``anchor_loo_residual`` — residual at the last anchor with
      ``row <= i`` (carries forward for non-anchor rows)
    * ``anchor_rmse_K{N}`` — RMSE of last K anchor residuals available
      at row i (= predictive uncertainty of ``linear_extrap_pred``)

    Use case: HONEST uncertainty estimate for ``linear_extrap_pred``;
    downstream GBMs can learn "extrapolation is unreliable here" and
    gate predictions accordingly. Leak-safe by construction.
    """
    if K_slope < 2 or K_rmse < 1:
        raise ValueError(f"K_slope >= 2 and K_rmse >= 1; got {K_slope}, {K_rmse}")

    def _seg(seg_label: np.ndarray, seg_anchor: np.ndarray, _idx: np.ndarray) -> dict:
        """Pure-Python fallback (used when numba is unavailable) mirroring ``_anchor_rmse_core`` for a single segment."""
        m = seg_label.size
        anchor_positions: list = []
        anchor_values: list = []
        anchor_residuals: list = []  # past LOO residuals
        last_residual_carry = np.nan
        residual_out = np.full(m, np.nan, dtype=np.float64)
        rmse_out = np.full(m, np.nan, dtype=np.float64)
        for i in range(m):
            if seg_anchor[i] and np.isfinite(seg_label[i]):
                # LOO residual at THIS anchor using anchors STRICTLY before.
                if len(anchor_positions) >= 2:
                    xs = np.asarray(anchor_positions[-K_slope:], dtype=np.float64)
                    ys = np.asarray(anchor_values[-K_slope:], dtype=np.float64)
                    xm = xs.mean()
                    ym = ys.mean()
                    num = ((xs - xm) * (ys - ym)).sum()
                    den = ((xs - xm) ** 2).sum()
                    slope = num / (den + 1e-12)
                    last_v = ys[-1]
                    last_r = xs[-1]
                    pred = last_v + slope * (i - last_r)
                    e_j = float(seg_label[i] - pred)
                    anchor_residuals.append(e_j)
                    if len(anchor_residuals) > K_rmse:
                        anchor_residuals.pop(0)
                    last_residual_carry = e_j
                anchor_positions.append(i)
                anchor_values.append(float(seg_label[i]))
                if len(anchor_positions) > K_slope:
                    anchor_positions.pop(0)
                    anchor_values.pop(0)
            residual_out[i] = last_residual_carry
            if anchor_residuals:
                arr_r = np.asarray(anchor_residuals, dtype=np.float64)
                rmse_out[i] = float(np.sqrt(np.mean(arr_r * arr_r)))
        return {
            "anchor_loo_residual": residual_out,
            f"anchor_rmse_K{K_rmse}": rmse_out,
        }

    if _NUMBA_AVAILABLE:
        return _run_anchor_numba_2out(
            label, is_anchor, group_ids,
            lambda lab, anc, o1, o2: _anchor_rmse_core(lab, anc, K_slope, K_rmse, o1, o2),
            "anchor_loo_residual", f"anchor_rmse_K{K_rmse}",
        )
    return _per_group_anchor_loop(label, is_anchor, group_ids, _seg)


def anchor_quadratic_extrapolation_features(
    label: np.ndarray,
    is_anchor: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    *,
    K_window: int = 10,
) -> dict:
    """Quadratic extrapolation from last K anchors.

    Fits ``y = a + b*dx + c*dx^2`` over the last K anchors (using only
    anchors with row <= current). Emits per row:
    * ``anchor_acceleration_K{N}`` — 2nd-order coefficient ``c``
    * ``quadratic_extrap_pred_K{N}`` — predicted ``y`` at current row

    Falls back to linear (c=0) when fewer than 3 anchors are available.
    Use case: signals with curvature where linear extrap systematically
    lags inflections (vitals deceleration, geosteering layers with
    curvature, deceleration of a process). Leak-safe.
    """
    if K_window < 3:
        raise ValueError(f"K_window must be >= 3, got {K_window}")

    def _seg(seg_label: np.ndarray, seg_anchor: np.ndarray, _idx: np.ndarray) -> dict:
        """Pure-Python fallback (used when numba is unavailable) mirroring ``_anchor_quadratic_core`` for a single segment, via ``np.linalg.lstsq`` instead of the closed-form Cramer solve."""
        m = seg_label.size
        anchor_positions: list = []
        anchor_values: list = []
        accel_out = np.full(m, np.nan, dtype=np.float64)
        quad_out = np.full(m, np.nan, dtype=np.float64)
        for i in range(m):
            if seg_anchor[i] and np.isfinite(seg_label[i]):
                anchor_positions.append(i)
                anchor_values.append(float(seg_label[i]))
                if len(anchor_positions) > K_window:
                    anchor_positions.pop(0)
                    anchor_values.pop(0)
            if len(anchor_positions) >= 3:
                xs = np.asarray(anchor_positions, dtype=np.float64)
                ys = np.asarray(anchor_values, dtype=np.float64)
                last_r = xs[-1]
                # quadratic fit in dx-coordinates (centred at last anchor)
                dx = xs - last_r
                X = np.stack([np.ones_like(dx), dx, dx * dx], axis=1)
                try:
                    coef, _, _, _ = np.linalg.lstsq(X, ys, rcond=None)
                    c = float(coef[2])
                    dx_now = i - last_r
                    pred = float(coef[0] + coef[1] * dx_now + coef[2] * dx_now * dx_now)
                    accel_out[i] = c
                    quad_out[i] = pred
                except np.linalg.LinAlgError:
                    pass
        return {
            f"anchor_acceleration_K{K_window}": accel_out,
            f"quadratic_extrap_pred_K{K_window}": quad_out,
        }

    if _NUMBA_AVAILABLE:
        return _run_anchor_numba_2out(
            label, is_anchor, group_ids,
            lambda lab, anc, o1, o2: _anchor_quadratic_core(lab, anc, K_window, o1, o2),
            f"anchor_acceleration_K{K_window}", f"quadratic_extrap_pred_K{K_window}",
        )
    return _per_group_anchor_loop(label, is_anchor, group_ids, _seg)


def anchor_ewm_features(
    label: np.ndarray,
    is_anchor: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    *,
    half_life_rows: float = 30.0,
) -> dict:
    """Exponentially-weighted anchor mean + slope (adaptive memory).

    Replaces the fixed-K hyperparameter with a half-life H (in row
    units): older anchors get weight ``0.5^((i - r_j) / H)``. Emits:
    * ``ewm_anchor_value_H{H}`` — weighted mean of past anchor values
    * ``ewm_anchor_slope_H{H}`` — weighted-OLS slope through past anchors

    Use case: panels with mixed regime speeds where K_slope=10 is
    either too myopic or too laggy. Smooth alternative to the hard
    window. Leak-safe.
    """
    if half_life_rows <= 0:
        raise ValueError(f"half_life_rows must be > 0, got {half_life_rows}")

    def _seg(seg_label: np.ndarray, seg_anchor: np.ndarray, _idx: np.ndarray) -> dict:
        """Pure-Python fallback (used when numba is unavailable) mirroring ``_anchor_ewm_core`` for a single segment, recomputing the weighted mean/slope from scratch each row instead of the O(1) recurrence."""
        m = seg_label.size
        anchor_positions: list = []
        anchor_values: list = []
        ewm_val_out = np.full(m, np.nan, dtype=np.float64)
        ewm_slope_out = np.full(m, np.nan, dtype=np.float64)
        for i in range(m):
            if seg_anchor[i] and np.isfinite(seg_label[i]):
                anchor_positions.append(i)
                anchor_values.append(float(seg_label[i]))
            if not anchor_positions:
                continue
            xs = np.asarray(anchor_positions, dtype=np.float64)
            ys = np.asarray(anchor_values, dtype=np.float64)
            w = 0.5 ** ((i - xs) / half_life_rows)
            w_sum = w.sum() + 1e-12
            w_mean = (ys * w).sum() / w_sum
            ewm_val_out[i] = float(w_mean)
            if xs.size >= 2:
                xm = (xs * w).sum() / w_sum
                ym = w_mean
                num = (w * (xs - xm) * (ys - ym)).sum()
                den = (w * (xs - xm) ** 2).sum()
                ewm_slope_out[i] = float(num / (den + 1e-12))
        return {
            f"ewm_anchor_value_H{int(half_life_rows)}": ewm_val_out,
            f"ewm_anchor_slope_H{int(half_life_rows)}": ewm_slope_out,
        }

    if _NUMBA_AVAILABLE:
        return _run_anchor_numba_2out(
            label, is_anchor, group_ids,
            lambda lab, anc, o1, o2: _anchor_ewm_core(lab, anc, float(half_life_rows), o1, o2),
            f"ewm_anchor_value_H{int(half_life_rows)}",
            f"ewm_anchor_slope_H{int(half_life_rows)}",
        )
    return _per_group_anchor_loop(label, is_anchor, group_ids, _seg)


def anchor_density_features(
    is_anchor: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    *,
    window_rows: int = 100,
) -> dict:
    """Trailing-window anchor density + mean inter-anchor gap.

    Emits per row:
    * ``anchor_count_in_W{W}`` — count of anchors in the trailing W rows
    * ``anchor_mean_gap_in_W{W}`` — mean gap (in rows) between
      consecutive anchors inside the window

    Distinguishes "5 anchors clustered 10 rows ago, none since" from
    "5 anchors evenly spaced over last W rows": same `rows_since_last
    _anchor` but very different extrapolation reliability. Also a
    direct proxy for labeling-effort regime.
    """
    if window_rows < 2:
        raise ValueError(f"window_rows must be >= 2, got {window_rows}")
    is_anchor = np.ascontiguousarray(is_anchor, dtype=bool)
    n = is_anchor.size

    def _seg(seg_anchor: np.ndarray, idx_seg: np.ndarray) -> tuple:
        """Pure-Python fallback (used when numba is unavailable) mirroring ``_anchor_density_core`` for a single segment; ``idx_seg`` is unused (kept for signature symmetry with other ``_seg`` fallbacks)."""
        m = seg_anchor.size
        count_out = np.zeros(m, dtype=np.float64)
        gap_out = np.full(m, np.nan, dtype=np.float64)
        recent_positions: list = []  # rolling anchor positions inside window
        for i in range(m):
            if seg_anchor[i]:
                recent_positions.append(i)
            # Drop positions that fell out of window.
            while recent_positions and (i - recent_positions[0]) >= window_rows:
                recent_positions.pop(0)
            count_out[i] = float(len(recent_positions))
            if len(recent_positions) >= 2:
                gaps = np.diff(np.asarray(recent_positions, dtype=np.float64))
                gap_out[i] = float(gaps.mean())
        return count_out, gap_out

    if _NUMBA_AVAILABLE:
        # label slot is unused by the density core (count/gap depend only on
        # anchor POSITIONS); pass is_anchor as the placeholder.
        return _run_anchor_numba_2out(
            is_anchor, is_anchor, group_ids,
            lambda lab, anc, o1, o2: _anchor_density_core(anc, window_rows, o1, o2),
            f"anchor_count_in_W{window_rows}", f"anchor_mean_gap_in_W{window_rows}",
        )
    if group_ids is None:
        count, gap = _seg(is_anchor, np.arange(n))
        return {
            f"anchor_count_in_W{window_rows}": count,
            f"anchor_mean_gap_in_W{window_rows}": gap,
        }
    sort_idx, starts, ends = iter_group_segments(group_ids)
    count_out = np.zeros(n, dtype=np.float64)
    gap_out = np.full(n, np.nan, dtype=np.float64)
    for s, e in zip(starts, ends):
        idx_seg = sort_idx[s:e]
        c, g = _seg(is_anchor[idx_seg], idx_seg)
        count_out[idx_seg] = c
        gap_out[idx_seg] = g
    return {
        f"anchor_count_in_W{window_rows}": count_out,
        f"anchor_mean_gap_in_W{window_rows}": gap_out,
    }


def rows_until_next_anchor(
    is_anchor: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    *,
    emit_forward_features: bool = False,
) -> np.ndarray:
    """FORWARD-LOOKING distance to the NEXT anchor in same group.

    USE ONLY when anchor POSITIONS are knowable at inference time
    (scheduled checkups, planned PS depths, etc.) -- the anchor VALUE
    is never emitted, only the row distance to its position.

    Default ``emit_forward_features=False`` returns an all-NaN array
    so callers cannot accidentally use this without opting in. The
    suffix ``_FORWARD`` in the OUTPUT column name is REQUIRED in any
    downstream wrapper that emits this as a DataFrame column.

    Use case: medical (next scheduled visit), geosteering (next
    planned PS), maintenance (next scheduled inspection). NEVER use
    when anchor positions themselves come from labels.
    """
    is_anchor = np.ascontiguousarray(is_anchor, dtype=bool)
    n = is_anchor.size
    out = np.full(n, np.nan, dtype=np.float64)
    if not emit_forward_features:
        return out

    def _seg(idx_seg: np.ndarray) -> None:
        """Right-to-left pass over one segment writing rows-until-next-anchor into ``out[idx_seg]`` in place."""
        seg = is_anchor[idx_seg]
        m = seg.size
        # Walk RIGHT TO LEFT, track rows-until-next-anchor.
        next_dist = np.iinfo(np.int64).max
        local = np.full(m, np.nan, dtype=np.float64)
        for i in range(m - 1, -1, -1):
            if seg[i]:
                next_dist = 0
            elif next_dist != np.iinfo(np.int64).max:
                next_dist += 1
            if next_dist != np.iinfo(np.int64).max:
                local[i] = float(next_dist)
        out[idx_seg] = local

    if group_ids is None:
        _seg(np.arange(n))
    else:
        sort_idx, starts, ends = iter_group_segments(group_ids)
        for s, e in zip(starts, ends):
            _seg(sort_idx[s:e])
    return out
