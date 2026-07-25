"""Regression tests for the discretization findings of the mrmr audit fix wave.

Pins the uniform NaN-sentinel dtype auto-widen (the sentinel at code n_bins must not wrap negative when
n_bins == dtype.max+1), the row-chunk VRAM budget honoring the widened output code width, and adds the
missing optimal_bin_edges coverage.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_discretize_uniform_nan_sentinel_no_overflow_at_int8_boundary():
    """discretize_array(method='uniform', n_bins=128) must widen past int8 so the NaN sentinel (code 128) is
    stored as 128, not wrapped to -128 by an int8 cast."""
    from mlframe.feature_selection.filters.discretization import discretize_array

    arr = np.array([1.0, np.nan, 100.0], dtype=np.float64)
    out = discretize_array(arr, n_bins=128, method="uniform")
    assert np.iinfo(out.dtype).max >= 128, "dtype must widen to hold the NaN sentinel at code 128"
    assert out[1] == 128, "NaN sentinel must be 128, not a wrapped negative"
    assert (out[[0, 2]] >= 0).all()


def test_discretize_uniform_nan_sentinel_no_overflow_at_int16_boundary():
    """Same overflow guard at the int16 boundary (n_bins == 32768)."""
    from mlframe.feature_selection.filters.discretization import discretize_array

    arr = np.array([1.0, np.nan, 100.0], dtype=np.float64)
    out = discretize_array(arr, n_bins=32768, method="uniform")
    assert np.iinfo(out.dtype).max >= 32768
    assert out[1] == 32768


def test_safe_code_dtype_reserves_nan_slot_only_when_asked():
    """_safe_code_dtype must widen for the uniform NaN slot (reserve_nan_slot) but keep the tight dtype for the
    quantile path where NaN routes to the rightmost real code n_bins-1."""
    from mlframe.feature_selection.filters.discretization import _safe_code_dtype

    assert _safe_code_dtype(128, np.int8, reserve_nan_slot=False) == np.int8  # quantile: 127 fits int8
    assert _safe_code_dtype(128, np.int8, reserve_nan_slot=True) == np.int16  # uniform: sentinel 128 needs int16


def test_row_chunk_rows_shrink_with_wider_output_dtype():
    """_choose_discretize_row_chunk_rows must account for the widened output code width: a 4-byte output must
    yield fewer rows than the historical 2-byte margin at the same free-VRAM budget."""
    from mlframe.feature_selection.filters.discretization import _choose_discretize_row_chunk_rows

    free = 8_000_000_000
    rows_2b = _choose_discretize_row_chunk_rows(1000, 4, free, out_itemsize=2)
    rows_4b = _choose_discretize_row_chunk_rows(1000, 4, free, out_itemsize=4)
    assert rows_4b < rows_2b, "wider output dtype must shrink the chunk to stay within the VRAM budget"
    # default preserves the historical 2-byte margin
    assert _choose_discretize_row_chunk_rows(1000, 4, free) == rows_2b


def test_optimal_bin_edges_recovers_monotonic_threshold():
    """optimal_bin_edges must return inf-sentinel, strictly increasing edges that split near a clean threshold,
    and apply_bin_edges must replay them leak-safely on held-out data."""
    pytest.importorskip("optbinning")
    from mlframe.feature_selection.filters.supervised_binning import apply_bin_edges, optimal_bin_edges

    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, 5000)
    # Monotonically-increasing P(y=1) in x with real overlap, so optbinning finds a genuine WoE split.
    y = (rng.uniform(0.0, 1.0, 5000) < x).astype(int)
    edges = optimal_bin_edges(x[:4000], y[:4000], max_n_bins=10, monotonic_trend="ascending")
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert np.all(np.diff(edges) > 0), "edges must be strictly increasing"
    assert len(edges) <= 11
    assert len(edges) >= 3, "a monotonic signal should produce at least one interior split"
    codes_val = apply_bin_edges(x[4000:], edges)  # replay on held-out, no refit
    assert codes_val.min() >= 0 and codes_val.max() < len(edges) - 1
