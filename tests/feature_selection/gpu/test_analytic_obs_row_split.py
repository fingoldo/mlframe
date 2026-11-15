"""The row-split mode of the fused observed-MI kernel (narrow matrices) returns the unsplit kernel's MI and occupied bins bit for bit."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cupy")

from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_analytic_gpu as g


@pytest.mark.parametrize("n_cols", [3, 40, 300, 700])
def test_split_equals_unsplit(monkeypatch, n_cols):
    """Forcing 1 split and forcing 6 splits give identical outputs, including a row count that does not divide evenly."""
    import cupy as cp

    rng = np.random.default_rng(n_cols)
    n = 50_003
    codes = cp.asarray(rng.integers(0, 12, (n, n_cols)).astype(np.int8))
    yc = rng.integers(0, 3, n).astype(np.int64)
    monkeypatch.setattr(g, "_row_splits", lambda cp_, blocks, n_: 1)
    mi1, b1 = g._fused_observed_mi_and_bins(codes, yc, 3)
    monkeypatch.setattr(g, "_row_splits", lambda cp_, blocks, n_: 6)
    mi6, b6 = g._fused_observed_mi_and_bins(codes, yc, 3)
    assert np.array_equal(mi1, mi6) and np.array_equal(b1, b6)
    assert (b1 > 1).all()


def test_split_chosen_for_a_narrow_matrix_only():
    """A 40-column matrix is cut into row splits; a wide one keeps the unsplit kernel."""
    import cupy as cp

    assert g._row_splits(cp, 2, 100_000) > 1
    assert g._row_splits(cp, 100000, 100_000) == 1
    assert g._row_splits(cp, 2, 4_000) == 1
