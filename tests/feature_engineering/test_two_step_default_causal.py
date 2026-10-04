"""two_step_recency_weighted_target_encode defaults to the causal expanding-window aggregate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_engineering.two_step_target_encode import two_step_recency_weighted_target_encode


def _events(seed: int = 0) -> pd.DataFrame:
    """Events of 30 entities over 20 time steps with a category that tracks the label only in the later half."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(30):
        label = e % 2
        for t in range(20):
            cat = ("A" if label else "B") if t >= 10 else rng.choice(["A", "B"])
            rows.append({"entity": e, "t": float(t), "cat": cat, "y": float(label if t >= 10 else rng.integers(0, 2))})
    return pd.DataFrame(rows)


def test_default_row_value_ignores_later_targets_of_the_same_entity():
    """Permuting the targets of later events (global target mean unchanged) leaves every earlier row's encoding untouched under the default."""
    df = _events()
    y = df["y"].to_numpy().copy()
    late = (df["t"] >= 8).to_numpy()
    y_perm = y.copy()
    y_perm[late] = np.random.default_rng(1).permutation(y[late])
    base = two_step_recency_weighted_target_encode(df, "entity", ["cat"], y, "t", decay_half_life=3.0)
    shuffled = two_step_recency_weighted_target_encode(df, "entity", ["cat"], y_perm, "t", decay_half_life=3.0)
    np.testing.assert_allclose(base[~late], shuffled[~late], rtol=1e-9, atol=1e-12)
    terminal = two_step_recency_weighted_target_encode(df, "entity", ["cat"], y, "t", decay_half_life=3.0, causal=False)
    terminal_shuffled = two_step_recency_weighted_target_encode(df, "entity", ["cat"], y_perm, "t", decay_half_life=3.0, causal=False)
    assert not np.allclose(terminal[~late], terminal_shuffled[~late])


def test_opt_out_restores_terminal_per_entity_value_that_leaks_the_future():
    """``causal=False`` gives every row of an entity the same terminal aggregate, which differs from the default on early rows."""
    df = _events()
    y = df["y"].to_numpy()
    default = two_step_recency_weighted_target_encode(df, "entity", ["cat"], y, "t", decay_half_life=3.0)
    terminal = two_step_recency_weighted_target_encode(df, "entity", ["cat"], y, "t", decay_half_life=3.0, causal=False)
    assert (pd.Series(terminal).groupby(df["entity"].to_numpy()).nunique() == 1).all()
    assert not np.allclose(default[df["t"].to_numpy() < 5], terminal[df["t"].to_numpy() < 5])
