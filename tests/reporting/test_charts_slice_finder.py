"""Tests for the multi-dimensional weak-slice finder (charts/slice_finder.py).

Covers: aggregation correctness (slice mean error matches a brute-force groupby),
support-floor / no-weak-slice handling, the cap-logging contract, spec shape, and
biz_value -- a synthetic with an injected bad 2-feature region must surface that
exact slice as the #1 ranked weak region with error far above global.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.charts.slice_finder import (
    _bin_matrix,
    find_weak_slices,
)
from mlframe.reporting.spec import BarPanelSpec, FigureSpec


def _flat(fig: FigureSpec):
    return [p for row in fig.panels for p in row if p is not None]


# ----------------------------------------------------------------------------
# Unit: binning + aggregation
# ----------------------------------------------------------------------------


def test_bin_matrix_constant_column_collapses_to_one_bin():
    mat = np.column_stack([np.arange(100.0), np.full(100, 5.0)])
    codes, _edges = _bin_matrix(mat, nbins=4)
    assert set(np.unique(codes[:, 1])) == {0}  # constant col -> single bin
    assert codes[:, 0].max() == 3  # 4 quartile bins on the linear feature


def test_single_feature_slice_mean_matches_groupby():
    rng = np.random.default_rng(0)
    n = 5000
    f0 = rng.random(n)
    err_proxy = np.where(f0 > 0.75, 10.0, 1.0)  # top quartile is bad
    X = pd.DataFrame({"f0": f0, "f1": rng.random(n)})
    # Regression: |y_true - y_pred| == err_proxy when y_pred = y_true + err_proxy.
    y_true = np.zeros(n)
    y_pred = err_proxy
    res = find_weak_slices(X, y_true, y_pred, task="regression", nbins=4, max_arity=1)
    top = res.worst_slice
    assert top[0] == ("f0",)
    # Worst quartile mean error ~10; global ~ 0.25*10 + 0.75*1 = 3.25.
    assert top[2] > 8.0
    assert res.global_error == pytest.approx(np.mean(err_proxy), rel=1e-6)


def test_support_floor_drops_tiny_slices():
    rng = np.random.default_rng(1)
    n = 2000
    X = pd.DataFrame({"f0": rng.random(n), "f1": rng.random(n)})
    y_true = np.zeros(n)
    y_pred = rng.random(n) * 0.1  # uniform small error, no real weak slice
    res = find_weak_slices(X, y_true, y_pred, min_support_fraction=0.5, max_arity=2)
    # With a 50% support floor, no fine slice qualifies -> empty / degenerate.
    assert len(res.table) <= 2


def test_spec_shape_and_orientation():
    rng = np.random.default_rng(2)
    n = 3000
    f0, f1 = rng.random(n), rng.random(n)
    bad = (f0 > 0.5) & (f1 > 0.5)
    X = pd.DataFrame({"f0": f0, "f1": f1})
    res = find_weak_slices(X, np.zeros(n), np.where(bad, 5.0, 1.0), max_arity=2)
    panels = _flat(res.figure)
    assert len(panels) == 1 and isinstance(panels[0], BarPanelSpec)
    assert panels[0].orientation == "horizontal"


def test_bar_categories_are_worst_first_to_match_title():
    """The horizontal-bar categories must be in worst-first order (matching ``table`` / the worst_slice).

    The renderer inverts the y-axis for horizontal bars, so the FIRST category lands on TOP; pre-reversing the
    data here put the worst slice at the BOTTOM, contradicting the panel's "worst-on-top" title."""
    rng = np.random.default_rng(11)
    n = 6000
    f0, f1 = rng.random(n), rng.random(n)
    bad = (f0 > 0.6) & (f1 > 0.6)
    X = pd.DataFrame({"f0": f0, "f1": f1})
    res = find_weak_slices(X, np.zeros(n), np.where(bad, 7.0, 1.0), task="regression", max_arity=2)
    panel = _flat(res.figure)[0]
    # Worst slice (largest mean error) is the first row of the worst-first table and the first bar category.
    assert panel.values[0] == max(panel.values)
    assert str(res.worst_slice[1]) in panel.categories[0]


def test_three_way_cap_is_logged(caplog):
    rng = np.random.default_rng(3)
    n = 4000
    cols = {f"f{i}": rng.random(n) for i in range(10)}
    X = pd.DataFrame(cols)
    bad = (X["f0"] > 0.5) & (X["f1"] > 0.5)
    res = find_weak_slices(X, np.zeros(n), np.where(bad, 5.0, 1.0), max_arity=3, three_way_top_features=4)
    assert any("3-way restricted to top" in c for c in res.capped)


# ----------------------------------------------------------------------------
# biz_value: injected bad 2-feature region ranks #1
# ----------------------------------------------------------------------------


def test_biz_val_injected_bad_2feature_region_ranks_first():
    """A synthetic with a deliberately bad region (f_a high AND f_b high) must surface
    that exact 2-feature slice as the #1 worst, with slice error >> global. Measured:
    slice error ~6.0 vs global ~1.6 (ratio ~3.8x). Floors: ratio >= 3.0, score #1 is the
    (f_a, f_b) pair. A regression in the aggregation / ranking drops the ratio toward 1."""
    rng = np.random.default_rng(42)
    n = 20_000
    f_a = rng.random(n)
    f_b = rng.random(n)
    f_c = rng.random(n)  # irrelevant decoy feature
    base_err = 1.0 + 0.2 * rng.random(n)
    bad_region = (f_a > 0.7) & (f_b > 0.7)  # ~9% of rows
    err = np.where(bad_region, 6.0, base_err)
    X = pd.DataFrame({"f_a": f_a, "f_b": f_b, "f_c": f_c})
    res = find_weak_slices(X, np.zeros(n), err, task="regression", nbins=4, max_arity=2)

    features, bounds, mean_err, _support = res.worst_slice
    assert set(features) == {"f_a", "f_b"}, features
    ratio = mean_err / res.global_error
    assert ratio >= 3.0, ratio
    # The injected region is the upper quartile of both features.
    assert "f_a" in bounds and "f_b" in bounds


def test_biz_val_single_dominant_feature_region():
    """When only ONE feature drives the error, the finder must still localise it (the 1-feature
    slice should appear with high ratio). Ensures the multi-dim search does not bury a strong
    1-feature signal under spurious pairs."""
    rng = np.random.default_rng(7)
    n = 15_000
    f0 = rng.random(n)
    err = np.where(f0 > 0.8, 8.0, 1.0)
    X = pd.DataFrame({"f0": f0, "f1": rng.random(n), "f2": rng.random(n)})
    res = find_weak_slices(X, np.zeros(n), err, task="regression", nbins=5, max_arity=2)
    # f0 must appear in the top-ranked slice's features.
    assert "f0" in res.worst_slice[0]
    assert res.worst_slice[2] / res.global_error >= 2.5


def test_aggregate_combo_fused_kernel_bit_identical_to_two_bincount():
    """The fused single-pass sum+count kernel must be bit-identical to the prior two ``np.bincount``
    walks. Both accumulate in row order, so the float64 per-cell error sums must match exactly --
    including adversarial magnitudes (huge / tiny / negative errors). Guards against a future
    "just always bincount" revert masking a kernel regression, and against an accumulation-order
    change that would silently shift which slice ranks worst."""
    import mlframe.reporting.charts.slice_finder as sf

    rng = np.random.default_rng(3)
    max_diff = 0.0
    for _ in range(120):
        n = int(rng.integers(50, 4000))
        ncells = int(rng.integers(2, 64))
        flat = rng.integers(0, ncells, size=n).astype(np.int64)
        err = (rng.random(n) - 0.5) * rng.choice([1.0, 1e6, 1e-6, 1e12])
        ref_counts = np.bincount(flat, minlength=ncells).astype(np.float64)
        ref_sums = np.bincount(flat, weights=err, minlength=ncells)
        got_sums, got_counts = sf._fused_sum_count(flat, np.ascontiguousarray(err), ncells)
        assert np.array_equal(ref_counts, got_counts)
        if not np.array_equal(ref_sums, got_sums):
            max_diff = max(max_diff, float(np.max(np.abs(ref_sums - got_sums))))
    assert max_diff == 0.0, f"fused kernel sums diverged from bincount by {max_diff}"


def test_aggregate_combo_2col_fast_path_bit_identical_to_bincount():
    """The arity-2 fast path (``_fused_sum_count_2col``) folds the mixed-radix flatten ``c0*stride0 + c1`` into the
    njit reduction. It must be bit-identical to the prior flatten + two-``np.bincount`` path across adversarial
    error magnitudes and bin-count grids. Guards against a future "always use the generic flatten" revert and
    against an accumulation-order change that would shift which pair slice ranks worst."""
    import mlframe.reporting.charts.slice_finder as sf

    rng = np.random.default_rng(11)
    max_diff = 0.0
    for _ in range(120):
        n = int(rng.integers(50, 4000))
        nb0 = int(rng.integers(2, 9))
        nb1 = int(rng.integers(2, 9))
        ncells = nb0 * nb1
        c0 = rng.integers(0, nb0, size=n).astype(np.int64)
        c1 = rng.integers(0, nb1, size=n).astype(np.int64)
        err = (rng.random(n) - 0.5) * rng.choice([1.0, 1e6, 1e-6, 1e12])
        flat = c0 * nb1 + c1
        ref_counts = np.bincount(flat, minlength=ncells).astype(np.float64)
        ref_sums = np.bincount(flat, weights=err, minlength=ncells)
        got_sums, got_counts = sf._fused_sum_count_2col(c0, c1, nb1, np.ascontiguousarray(err), ncells)
        assert np.array_equal(ref_counts, got_counts)
        if not np.array_equal(ref_sums, got_sums):
            max_diff = max(max_diff, float(np.max(np.abs(ref_sums - got_sums))))
    assert max_diff == 0.0, f"2-col fast path sums diverged from bincount by {max_diff}"


def test_slice_decode_vectorized_bit_identical_to_per_cell_loop():
    """CPX25-A: the mixed-radix cell-id decode in ``find_weak_slices`` was vectorized from a per-cell Python
    loop to a batched floor-div/mod over the stride vector. The batched decode must be exactly equal to the
    reference per-cell recurrence across arities (1/2/3), stride grids, and edge cases (empty selection,
    single-feature combo). Pure integer index arithmetic -> bit-identical by construction; this guards a
    future "revert to the python loop / change the digit order" from silently shifting decoded bin labels."""
    import numpy as np

    def decode_old(cell_ids, strides):
        m = len(strides)
        rows = []
        for cid in cell_ids:
            rem = int(cid)
            row = []
            for k in range(m):
                row.append(rem // int(strides[k]))
                rem = rem % int(strides[k])
            rows.append(row)
        return np.asarray(rows, dtype=np.int64).reshape(len(cell_ids), m)

    def decode_new(cell_ids, strides):
        m = len(strides)
        rem = cell_ids.astype(np.int64, copy=True)
        decoded = np.empty((cell_ids.size, m), dtype=np.int64)
        for k in range(m):
            sk = int(strides[k])
            decoded[:, k] = rem // sk
            rem %= sk
        return decoded

    rng = np.random.default_rng(7)
    for _ in range(200):
        arity = int(rng.integers(1, 4))
        nbins = rng.integers(2, 21, size=arity).astype(np.int64)
        strides = np.ones(arity, dtype=np.int64)
        for k in range(arity - 1, 0, -1):
            strides[k - 1] = strides[k] * nbins[k]
        ncells = int(strides[0]) * int(nbins[0])
        size = int(rng.integers(0, 50))  # includes the empty-selection edge case (size == 0)
        cell_ids = rng.integers(0, ncells, size=size, dtype=np.int64)
        old = decode_old(cell_ids, strides)
        new = decode_new(cell_ids, strides)
        assert old.shape == new.shape == (size, arity)
        assert np.array_equal(old, new)
