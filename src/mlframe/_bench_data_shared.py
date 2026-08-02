"""Shared micro-benchmark synthetic-dataset helper used by several unrelated ``_benchmarks/``
packages (feature_selection, evaluation): independently duplicated across those scripts,
consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_regression_data(n: int, n_features: int, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic (X, y): ``n_features`` Gaussian columns, ``y`` a noisy sum of the first 3."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = (X.iloc[:, :3].sum(axis=1) + rng.normal(scale=0.5, size=n)).to_numpy()
    return X, y


def make_meta_stacker_oof(n_rows: int, n_components: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic OOF-prediction matrix (X) + target (y) for the lasso/elasticnet meta-stacker benches: the first two
    columns carry the signal (0.7/0.3 weighted mix), the rest are pure noise components."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n_rows)
    b = rng.normal(size=n_rows)
    y = 0.7 * a + 0.3 * b + 0.05 * rng.normal(size=n_rows)
    extra = [rng.normal(size=n_rows) for _ in range(n_components - 2)]
    X = np.column_stack([a, b, *extra])
    return X, y
