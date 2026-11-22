"""Audit 2026-11-20 finding 37-F2: a lag supplied as an array becomes a column, so it can be the causal base candidate."""

import numpy as np
import pandas as pd

from mlframe.training.core._achievable_ceiling import _SUPPLIED_LAG_COLUMN, _pick_base_candidates, _with_supplied_lag


def test_supplied_lag_attaches_without_touching_the_callers_frame():
    """The returned frame has the lag column; the original frame does not."""
    df = pd.DataFrame({"x": np.arange(6.0), "y": np.arange(6.0)})
    lag = np.arange(6.0) * 0.5
    out, name = _with_supplied_lag(df, lag)
    assert name == _SUPPLIED_LAG_COLUMN
    np.testing.assert_allclose(out[name].to_numpy(), lag)
    assert name not in df.columns


def test_supplied_lag_is_the_first_base_candidate():
    """With the supplied lag attached, pick_base_candidates offers it first."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
    lag = rng.normal(size=n)
    out, name = _with_supplied_lag(df, lag)
    cands = _pick_base_candidates(out, ["x", name], "y", name, rng.normal(size=n), np.arange(n), max_bases=3)
    assert cands[0] == name
