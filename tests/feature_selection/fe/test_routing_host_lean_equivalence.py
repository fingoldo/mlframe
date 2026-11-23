"""The lean host path of the conditional routing FE gives the SAME winners as the legacy host path (float64 vectors, staged stack).

The legacy path held every cell as a float64 vector (``cand_values``) and stacked them into the criterion-dtype matrix; the lean path casts each cell at once into a block and rebuilds the winners through
``LazyRoutingValues``. The cast is elementwise (float64 to the criterion dtype), so the scored matrix is the same; the blocked statistics of ``_routing_scores`` are per column, so they agree to rounding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _orthogonal_routing_device as dev
from mlframe.feature_selection.filters import _orthogonal_routing_fe as R


def _frame(n: int = 2500, seed: int = 0):
    """Four numeric columns, a target that depends non-linearly on two of them."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=["a", "b", "c", "d"])
    y = ((X["a"] ** 2 + 0.5 * np.sin(2 * X["b"]) + 0.3 * rng.standard_normal(n)) > 0.8).astype(np.int64).to_numpy()
    return X, y


@pytest.fixture
def host_only(monkeypatch):
    """Force the host path: the device scorer declines."""
    monkeypatch.setattr(dev, "routing_scores_on_device", lambda *a, **k: None)


def test_the_scored_matrix_is_identical(host_only):
    """``_enumerate_routing_blocks`` builds exactly the matrix the legacy enumeration plus ``_stack_candidates`` builds, in the same cell order."""
    X, _ = _frame()
    cols, mat, meta = R._enumerate_routing_blocks(X, list(X.columns), ("identity", "zscore"), ("hermite", "chebyshev"), (2, 3, 4))
    old_cols, old_values, old_meta = R._enumerate_routing_cells(X, list(X.columns), ("identity", "zscore"), ("hermite", "chebyshev"), (2, 3, 4))
    assert cols == old_cols and meta == old_meta
    assert np.array_equal(mat, R._stack_candidates(old_values))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_winners_and_their_values_are_identical(host_only, monkeypatch, seed):
    """The public function returns the same winner columns, meta and float64 values with the lean path and with the legacy one."""
    X, y = _frame(seed=seed)
    kwargs = dict(cols=list(X.columns), candidate_bases=("hermite", "chebyshev"), degrees=(2, 3, 4), min_uplift=0.0, min_abs_mi_frac=0.0, top_k=8)
    monkeypatch.setenv("MLFRAME_ROUTING_HOST_LEAN", "1")
    lean_df, lean_meta = R.generate_conditional_basis_routing_features(X, y, **kwargs)
    monkeypatch.setenv("MLFRAME_ROUTING_HOST_LEAN", "0")
    old_df, old_meta = R.generate_conditional_basis_routing_features(X, y, **kwargs)
    assert list(lean_df.columns) == list(old_df.columns)
    assert lean_meta == old_meta
    for c in old_df.columns:
        assert np.array_equal(lean_df[c].to_numpy(), old_df[c].to_numpy())
    assert len(old_df.columns) >= 1, "the fixture must produce a winner for this to prove anything"
