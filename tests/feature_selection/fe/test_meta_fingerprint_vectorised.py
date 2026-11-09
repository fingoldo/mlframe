"""The vectorised fingerprint statistics equal the per-column pandas loops (constant, tied and NaN-bearing columns included)."""

import numpy as np
import pandas as pd
import pytest

import mlframe.feature_selection.filters._orthogonal_meta_scorer_fe as meta


def _frame(n, seed, nan=False):
    """Mixed columns: normal, rounded (ties), constant, skewed, one with a NaN when requested; a binary target tied to the first column."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": r.standard_normal(n) for i in range(8)})
    X["x2"] = np.round(X["x2"] * 2)
    X["x3"] = 1.0
    X["x4"] = np.exp(X["x4"])
    if nan:
        X.loc[X.index[3], "x5"] = np.nan
    y = ((X["x0"] + r.standard_normal(n)) > 0).astype(np.int64).to_numpy()
    return X, y


@pytest.mark.parametrize("nan", [False, True])
@pytest.mark.parametrize("n", [400, 5000])
def test_fingerprint_equals_the_loop(n, nan, monkeypatch):
    """Every fingerprint entry agrees between the vectorised and the loop path (the NaN frame takes the loop path in both)."""
    X, y = _frame(n, 0, nan)
    X = X.astype({c: np.float32 for c in X.columns if c != "x1"})  # float32 columns take the radix distinct count; x1 stays float64
    monkeypatch.setattr(meta, "_VECTORISED", True)
    got = meta.fingerprint_signal(X, y)
    monkeypatch.setattr(meta, "_VECTORISED", False)
    want = meta.fingerprint_signal(X, y)
    assert set(got) == set(want)
    for k, v in want.items():
        assert got[k] == pytest.approx(v, rel=1e-6, abs=1e-8), k  # pandas accumulates float32 columns in float32; the vectorised pass is the more accurate in double
