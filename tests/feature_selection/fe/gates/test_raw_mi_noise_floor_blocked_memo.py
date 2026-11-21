"""When the raw matrix is scored in column blocks, the floor is memoised: the gates that ask for it within a fit do not repeat the scan, and the memoised value is the unblocked one."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _unified_fe_gate as G


def _frame(n=6000, p=12, seed=0):
    """A numeric frame and a binary target that depends on the first column."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)])
    y = (X["x0"] + 0.5 * rng.standard_normal(n) > 0).astype(int).to_numpy()
    return X, y


def test_blocked_floor_is_computed_once_and_equals_the_unblocked_value(monkeypatch):
    """Four blocks of three columns: the first call scores four blocks, the second and a call with an equal frame score none; the value is the one the whole-matrix scan gives."""
    X, y = _frame()
    G._RAW_MI_FLOOR_MEMO.clear()
    whole = G.raw_mi_noise_floor(X, y)
    G._RAW_MI_FLOOR_MEMO.clear()
    monkeypatch.setattr("mlframe.feature_selection.filters._binned_numeric_agg_batches.column_blocks", lambda cols, n: [list(cols[i : i + 3]) for i in range(0, len(cols), 3)])
    import mlframe.feature_selection.filters._orthogonal_univariate_fe as O

    real = O._mi_classif_batch
    calls = []
    monkeypatch.setattr(O, "_mi_classif_batch", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    first = G.raw_mi_noise_floor(X, y)
    n_first = len(calls)
    second = G.raw_mi_noise_floor(X, y)
    third = G.raw_mi_noise_floor(X.copy(), y.copy())
    assert n_first == 4 and len(calls) == 4, calls
    assert first == second == third
    np.testing.assert_allclose(first, whole, rtol=0, atol=1e-9)


def test_blocked_floor_memo_separates_different_data(monkeypatch):
    """A frame with different values (same column names) does not hit the memo of another."""
    X, y = _frame(seed=1)
    X2, y2 = _frame(seed=2)
    G._RAW_MI_FLOOR_MEMO.clear()
    monkeypatch.setattr("mlframe.feature_selection.filters._binned_numeric_agg_batches.column_blocks", lambda cols, n: [list(cols[i : i + 4]) for i in range(0, len(cols), 4)])
    G.raw_mi_noise_floor(X, y)
    G.raw_mi_noise_floor(X2, y2)
    keys = [k for k in G._RAW_MI_FLOOR_MEMO if k and k[0] == "blocked"]
    assert len(keys) == 2 and keys[0][3] != keys[1][3]  # the floor itself can coincide (the analytic null depends on n and the bins), the sample hash does not
