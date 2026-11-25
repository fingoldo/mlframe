"""The fused projections of the unit columns equal the numpy chain they replaced (finite, NaN-bearing, infinite, constant and sign-flipped columns, a block boundary included)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _mrmr_degenerate as M


def _frame(n=3001, seed=0):
    """Columns of every kind the audit meets."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n)
    d = {"cont": base, "scaled": base * 3.5 - 2.0, "flip": -base, "round": np.round(rng.normal(size=n), 1), "const": np.full(n, 7.0), "few": rng.integers(0, 4, n).astype(float)}
    nan_col = rng.normal(size=n)
    nan_col[[3, 9, 1000]] = np.nan
    inf_col = rng.normal(size=n)
    inf_col[5] = np.inf
    all_nan = np.full(n, np.nan)
    d.update({"nan": nan_col, "inf": inf_col, "all_nan": all_nan})
    return pd.DataFrame(d).astype(np.float32)


def test_projections_equal_the_unit_vector_times_the_basis():
    """Per column: the same variance verdict and projections within the single-precision rounding of the former float32 chain."""
    X = _frame()
    live = list(X.columns)
    basis = np.random.default_rng(0).standard_normal((len(X), 3))
    good, proj = M._unit_projections(X, live, basis)
    for k, name in enumerate(live):
        unit = M._unit_column(X, name)
        assert good[k] == (unit is not None), name
        if unit is not None:
            np.testing.assert_allclose(proj[k], unit @ basis, atol=2e-4, err_msg=name)
        else:
            assert not proj[k].any()


def test_a_block_boundary_changes_nothing(monkeypatch):
    """With the block cut to two columns the result is the one of a single block."""
    X = _frame(n=1500, seed=3)
    live = list(X.columns)
    basis = np.random.default_rng(1).standard_normal((len(X), 3))
    whole = M._unit_projections(X, live, basis)
    monkeypatch.setattr(M, "_UNIT_PROJECTION_BLOCK_BYTES", 2 * len(X) * 4)
    cut = M._unit_projections(X, live, basis)
    np.testing.assert_array_equal(whole[0], cut[0])
    np.testing.assert_array_equal(whole[1], cut[1])
