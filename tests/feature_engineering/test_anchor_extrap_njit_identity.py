"""FE_ROOT_A-5 (2026-08-05 audit): add_anchor_extrapolation_features -- the anchor module's flagship,
most-used function -- had no numba-accelerated path, unlike its four siblings (anchor_residual_rmse_features,
anchor_quadratic_extrapolation_features, anchor_ewm_features, anchor_density_features), which were all
explicitly converted from Python list append/pop to preallocated-buffer njit cores.

Added ``_anchor_extrap_core`` (njit, mirrors the other four cores' pattern) plus a dispatch wrapper
(``_anchor_features_for_segment_dispatch``) that routes to it when numba is available, falling back to the
original ``_anchor_features_for_segment`` (now the documented numba-unavailable fallback) otherwise.

These tests pin the njit core's output against the original pure-Python per-segment walk across varied
scenarios (sparse/dense anchors, K_slope edge cases, NaN-poisoned labels, single/no anchors, grouped
input). ``rows_since``/``last_anchor_value`` are plain row copies -- exact equality. The slope/extrapolation
outputs involve a floating-point reduction (OLS sums), and the njit core runs under the module's
``_ANCHOR_FASTMATH`` (reassoc/contract/arcp/afn) like its four siblings -- so, mirroring
``test_cpx_anchor_ewm_spatial_dispersion_identity.py``'s own convention for this exact fastmath set, those
are compared with ``atol=1e-9`` rather than exact equality.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.anchor import _anchor_features_for_segment, add_anchor_extrapolation_features


def _make_stream(rng, n, anchor_rate, nan_rate=0.0):
    """Random label/is_anchor pair: anchors at anchor_rate density, optionally some anchors NaN-poisoned."""
    is_anchor = rng.uniform(0, 1, n) < anchor_rate
    label = np.where(is_anchor, rng.normal(0, 5, n).cumsum() * 0.1, np.nan)
    if nan_rate > 0:
        poison = is_anchor & (rng.uniform(0, 1, n) < nan_rate)
        label = np.where(poison, np.nan, label)
    return label.astype(np.float64), is_anchor


@pytest.mark.parametrize("K_slope", [2, 3, 5, 10, 50])
@pytest.mark.parametrize("anchor_rate", [0.05, 0.2, 0.5, 0.9])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_njit_core_matches_python_fallback_bit_identical(K_slope, anchor_rate, seed):
    """The njit-dispatched path must match the pure-Python per-segment walk exactly."""
    rng = np.random.default_rng(seed)
    n = 400
    label, is_anchor = _make_stream(rng, n, anchor_rate, nan_rate=0.1)

    python_ref = _anchor_features_for_segment(label, is_anchor, K_slope)
    dispatched = add_anchor_extrapolation_features(label, is_anchor, K_slope=K_slope)

    np.testing.assert_array_equal(dispatched["rows_since_last_anchor"], python_ref["rows_since"])
    np.testing.assert_array_equal(dispatched["last_anchor_value"], python_ref["last_anchor_value"])
    np.testing.assert_allclose(dispatched[f"last_anchor_local_slope_K{K_slope}"], python_ref["local_slope"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(dispatched[f"linear_extrap_pred_K{K_slope}"], python_ref["linear_extrap_pred"], atol=1e-9, rtol=0)


def test_njit_core_matches_python_fallback_no_anchors():
    """Degenerate: zero anchors in the whole segment -- all outputs must stay NaN, both paths agree."""
    n = 100
    label = np.full(n, np.nan)
    is_anchor = np.zeros(n, dtype=bool)
    python_ref = _anchor_features_for_segment(label, is_anchor, K_slope=3)
    dispatched = add_anchor_extrapolation_features(label, is_anchor, K_slope=3)
    assert np.isnan(dispatched["rows_since_last_anchor"]).all()
    np.testing.assert_array_equal(dispatched["rows_since_last_anchor"], python_ref["rows_since"])


def test_njit_core_matches_python_fallback_single_anchor():
    """Degenerate: exactly one anchor -- slope defined as 0, flat extrapolation."""
    n = 50
    label = np.full(n, np.nan)
    is_anchor = np.zeros(n, dtype=bool)
    label[10] = 3.5
    is_anchor[10] = True
    python_ref = _anchor_features_for_segment(label, is_anchor, K_slope=5)
    dispatched = add_anchor_extrapolation_features(label, is_anchor, K_slope=5)
    np.testing.assert_allclose(dispatched["last_anchor_local_slope_K5"], python_ref["local_slope"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(dispatched["linear_extrap_pred_K5"], python_ref["linear_extrap_pred"], atol=1e-9, rtol=0)
    assert dispatched["linear_extrap_pred_K5"][30] == 3.5


@pytest.mark.parametrize("seed", [3, 4])
def test_njit_core_matches_python_fallback_grouped(seed):
    """Grouped input: each group must get its own independent anchor history (no cross-group bleed),
    and the grouped dispatch path must still match the per-group pure-Python walk exactly."""
    rng = np.random.default_rng(seed)
    n = 300
    label, is_anchor = _make_stream(rng, n, anchor_rate=0.15, nan_rate=0.05)
    group_ids = rng.integers(0, 4, n)

    dispatched = add_anchor_extrapolation_features(label, is_anchor, group_ids, K_slope=4)

    expected = {
        "rows_since_last_anchor": np.full(n, np.nan),
        "last_anchor_value": np.full(n, np.nan),
        "last_anchor_local_slope_K4": np.full(n, np.nan),
        "linear_extrap_pred_K4": np.full(n, np.nan),
    }
    for g in np.unique(group_ids):
        idx = np.flatnonzero(group_ids == g)
        ref = _anchor_features_for_segment(label[idx], is_anchor[idx], K_slope=4)
        expected["rows_since_last_anchor"][idx] = ref["rows_since"]
        expected["last_anchor_value"][idx] = ref["last_anchor_value"]
        expected["last_anchor_local_slope_K4"][idx] = ref["local_slope"]
        expected["linear_extrap_pred_K4"][idx] = ref["linear_extrap_pred"]

    np.testing.assert_array_equal(dispatched["rows_since_last_anchor"], expected["rows_since_last_anchor"])
    np.testing.assert_array_equal(dispatched["last_anchor_value"], expected["last_anchor_value"])
    np.testing.assert_allclose(dispatched["last_anchor_local_slope_K4"], expected["last_anchor_local_slope_K4"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(dispatched["linear_extrap_pred_K4"], expected["linear_extrap_pred_K4"], atol=1e-9, rtol=0)
