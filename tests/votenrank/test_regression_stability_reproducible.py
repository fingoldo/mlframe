"""Regression: spearman_exp used the process-global np.random.choice (non-reproducible and
order-dependent on prior RNG use). It now uses a per-repeat default_rng(seed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.votenrank import Leaderboard
from mlframe.votenrank.stability_exp import spearman_exp


def _lb():
    rng = np.random.default_rng(0)
    idx = [f"M{i}" for i in range(8)]
    cols = [f"t{j}" for j in range(6)]
    table = pd.DataFrame(rng.uniform(size=(8, 6)), index=idx, columns=cols)
    return Leaderboard(table, weights={c: 1.0 for c in cols})


def test_spearman_exp_reproducible_across_runs():
    exp_range = [0.1, 0.2]
    # Perturb the global RNG between runs: a reproducible implementation must be unaffected.
    np.random.seed(123)
    res_a = spearman_exp(_lb(), num_repeats=2, exp_range=exp_range)
    np.random.seed(999)
    for _ in range(5):
        np.random.random()
    res_b = spearman_exp(_lb(), num_repeats=2, exp_range=exp_range)

    keys = set(res_a) | set(res_b)
    assert keys, "experiment produced no correlation columns"
    for k in keys:
        np.testing.assert_allclose(np.asarray(res_a[k]), np.asarray(res_b[k]), rtol=0, atol=0)
