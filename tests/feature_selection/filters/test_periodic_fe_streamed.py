"""Modular FE scored in RAM-budget chunks of source columns returns the same winners and scores as the one-pass pipeline."""

import numpy as np
import pandas as pd
import pytest

import mlframe.feature_selection.filters._feature_engineering_mem_budget as budget
from mlframe.feature_selection.filters._periodic_fe import hybrid_modular_fe


def _frame(n=4000, seed=0):
    """Frame."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"c{i}": rng.integers(0, 10_000, n).astype(np.float64) for i in range(6)})
    y = ((X["c2"] % 7) < 3).astype(np.int64).to_numpy().copy()
    flip = rng.random(n) < 0.1
    y[flip] = 1 - y[flip]
    return X, y


@pytest.mark.parametrize("cols_per_chunk", [1, 2])
def test_streamed_matches_one_pass(monkeypatch, cols_per_chunk):
    """Streamed matches one pass."""
    X, y = _frame()
    full_aug, full_scores = hybrid_modular_fe(X, y, top_k=4)
    per_src = len(X) * 5 * 3 * 3 * 4
    real = budget._can_hoist_shared_buffer
    monkeypatch.setattr(
        budget, "_can_hoist_shared_buffer", lambda nb, n_workers=1: (False, nb, 0) if nb > cols_per_chunk * per_src else real(nb, n_workers=n_workers)
    )
    st_aug, st_scores = hybrid_modular_fe(X, y, top_k=4)
    appended = [c for c in full_aug.columns if c not in X.columns]
    assert "c2__mod_7" in appended or "c2__modsin_7" in appended or "c2__modcos_7" in appended
    assert [c for c in st_aug.columns if c not in X.columns] == appended
    pd.testing.assert_frame_equal(st_aug[appended], full_aug[appended])
    a = full_scores.set_index("engineered_col").sort_index()
    b = st_scores.set_index("engineered_col").sort_index()
    pd.testing.assert_frame_equal(a, b)
