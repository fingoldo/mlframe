"""Synthetic MRMR fit fixture shared by the FE-stage benchmark and profiling scripts."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def make(seed, n):
    """Novel data per call: MRMR memoises by content hash, so a repeated frame would time the cache replay."""
    rng = np.random.default_rng(seed)
    a, b, c, d, e, f = (rng.uniform(0.1, 1.1, n) for _ in range(6))
    df = pd.DataFrame({k: v for k, v in zip("abcde", (a, b, c, d, e))})
    y = a**2 / b + f / 5.0 + np.log(np.abs(c) + 1e-9) * np.sin(d)
    return df, y


def fit(seed, n):
    df, y = make(seed, n)
    return MRMR(full_npermutations=10, baseline_npermutations=20, fe_max_steps=2, fe_min_pair_mi_prevalence=1.05, verbose=0, n_jobs=1, random_seed=seed).fit(df, y)
