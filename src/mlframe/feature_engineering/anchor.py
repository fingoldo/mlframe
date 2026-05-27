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
]

from typing import Optional

import numpy as np

from .grouped import iter_group_segments


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
        raise ValueError(
            f"label / is_anchor length mismatch: {n} vs {is_anchor.size}"
        )
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
