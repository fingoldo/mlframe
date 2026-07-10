"""biz_value test for ``feature_selection.drop_near_noise_univariate_auc.drop_near_noise_univariate_auc``.

The win (4th_santander-customer-transaction-prediction.md): a cheap univariate-AUC prescreen correctly
identifies features whose OWN individual AUC sits at chance (no linear/monotone signal in isolation) before
running the expensive MRMR/DCD pipeline. This test confirms the prescreen correctly flags pure-noise columns
while preserving genuinely predictive ones, and that dropping the flagged columns doesn't meaningfully hurt a
downstream classifier's AUC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_selection.drop_near_noise_univariate_auc import drop_near_noise_univariate_auc


def _make_mixed_signal_dataset(n: int, n_signal: int, n_noise: int, seed: int):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    signal_direction = np.where(y == 1, 1.0, -1.0)

    cols = {}
    for i in range(n_signal):
        cols[f"signal{i}"] = signal_direction + rng.normal(scale=0.5, size=n)
    for i in range(n_noise):
        cols[f"noise{i}"] = rng.normal(size=n)  # independent of y

    df = pd.DataFrame(cols)
    return df, y


def test_biz_val_drop_near_noise_correctly_identifies_noise_columns():
    df, y = _make_mixed_signal_dataset(n=3000, n_signal=5, n_noise=15, seed=0)

    dropped = drop_near_noise_univariate_auc(df, y, tolerance=0.03)

    n_signal_dropped = sum(1 for c in dropped if c.startswith("signal"))
    n_noise_dropped = sum(1 for c in dropped if c.startswith("noise"))
    assert n_signal_dropped == 0, f"expected no genuinely-predictive signal columns to be flagged, got {[c for c in dropped if c.startswith('signal')]}"
    assert n_noise_dropped >= 12, f"expected most pure-noise columns to be correctly flagged, got {n_noise_dropped}/15"


def test_biz_val_dropping_near_noise_columns_preserves_downstream_auc():
    df, y = _make_mixed_signal_dataset(n=3000, n_signal=5, n_noise=15, seed=1)
    dropped = drop_near_noise_univariate_auc(df, y, tolerance=0.03)
    kept_cols = [c for c in df.columns if c not in dropped]

    auc_full = cross_val_score(LogisticRegression(max_iter=500), df, y, cv=5, scoring="roc_auc").mean()
    auc_pruned = cross_val_score(LogisticRegression(max_iter=500), df[kept_cols], y, cv=5, scoring="roc_auc").mean()

    assert len(kept_cols) < df.shape[1]
    assert auc_pruned >= auc_full - 0.02, f"expected pruning near-noise columns to not meaningfully hurt downstream AUC, got pruned={auc_pruned:.4f} full={auc_full:.4f}"
