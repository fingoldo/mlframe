"""Unit tests for the group-blocked MI kernel (`_group_mi`).

Anchors: (1) with a SINGLE group + no Miller-Madow, group-blocked MI EQUALS the global `compute_mi_from_classes`
(same joint over all rows) -- the correctness tie to the existing kernel; (2) a feature constant WITHIN each group
(pure between-group level) has within-group MI ~0 (leakage demoted); (3) a within-group relationship whose SIGN FLIPS
across groups still has HIGH per-group MI in every group, so it survives (the case a global-demean approach drops).
"""
from __future__ import annotations

import math

import numpy as np

from mlframe.feature_selection.filters.info_theory._class_mi_kernels import compute_mi_from_classes
from mlframe.feature_selection.filters.info_theory._group_mi import group_blocked_mi, prepare_group_segments


def _codes(x, nbins):
    """Equal-mass quantile bin `x` into `[0, nbins)` integer codes (matches MRMR's binning shape)."""
    q = np.quantile(x, np.linspace(0, 1, nbins + 1)[1:-1])
    c = np.searchsorted(q, x, side="right").astype(np.int64)
    np.clip(c, 0, nbins - 1, out=c)
    return c


def _global_mi(cx, cy, K_x, K_y):
    fx = np.bincount(cx[cx >= 0], minlength=K_x).astype(np.float64)
    fx /= fx.sum()
    fy = np.bincount(cy[cy >= 0], minlength=K_y).astype(np.float64)
    fy /= fy.sum()
    return compute_mi_from_classes(cx.astype(np.int32), fx, cy.astype(np.int32), fy)


def test_single_group_equals_global_mi():
    rng = np.random.default_rng(0)
    n, nb = 4000, 8
    x = rng.normal(size=n)
    y = x + rng.normal(scale=0.5, size=n)  # real dependence
    cx, cy = _codes(x, nb), _codes(y, nb)
    groups = np.zeros(n, dtype=np.int64)  # one group
    sort_idx, offsets = prepare_group_segments(groups)
    gb = group_blocked_mi(cx, cy, sort_idx, offsets, nb, nb, min_rows=10, size_weighted=True, use_mm=False)
    glob = _global_mi(cx, cy, nb, nb)
    assert math.isclose(gb, glob, rel_tol=1e-9, abs_tol=1e-12), f"1-group blocked MI {gb} != global {glob}"


def test_between_group_level_has_zero_within_group_mi():
    # x = group id (constant within group); y depends on x -> global MI high, within-group MI ~0.
    rng = np.random.default_rng(1)
    n_groups, per, nb = 30, 200, 8
    g = np.repeat(np.arange(n_groups), per)
    level = rng.uniform(0, 100, n_groups)[g]
    x = level                                   # pure between-group level
    y = level + rng.normal(scale=1.0, size=g.size)
    cx, cy = _codes(x, nb), _codes(y, nb)
    sort_idx, offsets = prepare_group_segments(g)
    gb = group_blocked_mi(cx, cy, sort_idx, offsets, nb, nb, min_rows=20, size_weighted=True, use_mm=True)
    glob = _global_mi(cx, cy, nb, nb)
    assert glob > 0.5, f"global MI should be high for a level feature; got {glob}"
    assert gb < 0.05, f"within-group MI of a constant-per-group feature must be ~0; got {gb}"


def test_sign_flip_within_group_is_retained():
    # x correlates +y in even groups, -y in odd groups. Global (pooled) association is weak; per-group MI is high in
    # EVERY group -> group-blocked MI stays high (the case a demean/linear approach would collapse).
    rng = np.random.default_rng(2)
    n_groups, per, nb = 20, 300, 8
    g = np.repeat(np.arange(n_groups), per)
    x = rng.normal(size=g.size)
    slope = np.where(g % 2 == 0, 1.0, -1.0)
    y = slope * x + rng.normal(scale=0.3, size=g.size)
    cx, cy = _codes(x, nb), _codes(y, nb)
    sort_idx, offsets = prepare_group_segments(g)
    gb = group_blocked_mi(cx, cy, sort_idx, offsets, nb, nb, min_rows=20, size_weighted=True, use_mm=True)
    assert gb > 0.3, f"sign-flipping within-group signal must be retained (high per-group MI); got {gb}"


def test_min_rows_skips_small_groups():
    rng = np.random.default_rng(3)
    # one big informative group + many size-1 noise groups; min_rows drops the singletons.
    big = np.zeros(500, dtype=np.int64)
    small = np.arange(1, 51)  # 50 singleton groups
    g = np.concatenate([big, small])
    x = np.concatenate([rng.normal(size=500), rng.normal(size=50)])
    y = np.concatenate([x[:500] + rng.normal(scale=0.3, size=500), rng.normal(size=50)])
    cx, cy = _codes(x, 6), _codes(y, 6)
    sort_idx, offsets = prepare_group_segments(g)
    gb = group_blocked_mi(cx, cy, sort_idx, offsets, 6, 6, min_rows=20, size_weighted=True, use_mm=True)
    assert gb > 0.2, "the one big informative group should dominate once singletons are skipped"


def test_sentinel_and_empty():
    g = np.array([0, 0, 0, 1, 1, 1] * 20)
    cx = np.array([-1, 0, 1, 2, -1, 0] * 20, dtype=np.int64)  # some NaN sentinels
    cy = np.array([0, 1, 2, 0, 1, -1] * 20, dtype=np.int64)
    sort_idx, offsets = prepare_group_segments(g)
    v = group_blocked_mi(cx, cy, sort_idx, offsets, 3, 3, min_rows=2, size_weighted=True, use_mm=False)
    assert np.isfinite(v) and v >= 0.0
    # all-sentinel -> 0
    cx2 = np.full(120, -1, dtype=np.int64)
    v2 = group_blocked_mi(cx2, cy, sort_idx, offsets, 3, 3, min_rows=2, size_weighted=True, use_mm=False)
    assert v2 == 0.0


def test_equal_vs_size_weight_differ_when_groups_unequal():
    rng = np.random.default_rng(4)
    # a huge noise group + a small informative group: size-weighted ~0, equal-weight lifts the informative one.
    gbig = np.zeros(4000, dtype=np.int64)
    gsmall = np.ones(200, dtype=np.int64)
    g = np.concatenate([gbig, gsmall])
    x = np.concatenate([rng.normal(size=4000), rng.normal(size=200)])
    y = np.concatenate([rng.normal(size=4000), x[4000:] + rng.normal(scale=0.2, size=200)])
    cx, cy = _codes(x, 6), _codes(y, 6)
    sort_idx, offsets = prepare_group_segments(g)
    sw = group_blocked_mi(cx, cy, sort_idx, offsets, 6, 6, min_rows=20, size_weighted=True, use_mm=True)
    ew = group_blocked_mi(cx, cy, sort_idx, offsets, 6, 6, min_rows=20, size_weighted=False, use_mm=True)
    assert ew > sw, f"equal-weight should surface the small informative group more than size-weight; ew={ew} sw={sw}"
