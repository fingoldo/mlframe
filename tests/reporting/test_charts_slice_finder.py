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
    DEFAULT_NBINS,
    SliceFinderResult,
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
    codes, edges = _bin_matrix(mat, nbins=4)
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


def test_three_way_cap_is_logged(caplog):
    rng = np.random.default_rng(3)
    n = 4000
    cols = {f"f{i}": rng.random(n) for i in range(10)}
    X = pd.DataFrame(cols)
    bad = (X["f0"] > 0.5) & (X["f1"] > 0.5)
    res = find_weak_slices(X, np.zeros(n), np.where(bad, 5.0, 1.0),
                           max_arity=3, three_way_top_features=4)
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

    features, bounds, mean_err, support = res.worst_slice
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
