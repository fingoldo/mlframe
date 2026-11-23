"""The adaptive-degree host path builds its candidate matrix lean (PLAN-10: 1.4 GB retained at 500000 x 1000) and the matrix, hence the scores and winners, are those of the old list-stack-cast form."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _orthogonal_adaptive_degree_fe as A
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype


def _frame(n: int = 1500):
    """Gaussian columns, one with NaNs, one constant, and a target that depends on two of them."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((n, 6)), columns=[f"x{i}" for i in range(6)])
    X.loc[::17, "x3"] = np.nan
    X["x5"] = 1.0
    y = ((X["x0"] ** 2 + 0.5 * X["x1"] + 0.3 * rng.standard_normal(n)) > 1).astype(np.int64).to_numpy()
    return X, y


def test_the_lean_matrix_equals_the_stacked_candidates():
    """Same names, same cells, same values as ``column_stack(cand_values).astype(criterion dtype)``."""
    X, y = _frame()
    cols = list(X.columns)
    dt = np.float32
    c1, vals, cells1 = A._build_degree_candidates(X, cols, y, (2, 3, 4), "auto", dt)
    c2, mat, cells2 = A._build_degree_candidate_matrix(X, cols, y, (2, 3, 4), "auto", dt)
    assert c1 == c2 and cells1 == cells2 and len(c1) > 3
    ref = np.column_stack(vals).astype(_crit_np_dtype(), copy=False)
    assert mat.dtype == ref.dtype
    np.testing.assert_array_equal(mat, ref)


def test_more_than_one_block_of_a_source_is_assembled_in_order(monkeypatch):
    """A source with more cells than the block size (here 2 sources x 3 degrees at a block of 64 stays one block; the degree range is widened to cross it)."""
    X, y = _frame(800)
    cols = ["x0", "x1"]
    degrees = tuple(range(2, 2 + 70))
    c1, vals, _ = A._build_degree_candidates(X, cols, y, degrees, "hermite", np.float32)
    c2, mat, _ = A._build_degree_candidate_matrix(X, cols, y, degrees, "hermite", np.float32)
    assert c1 == c2
    np.testing.assert_array_equal(mat, np.column_stack(vals).astype(_crit_np_dtype(), copy=False))
