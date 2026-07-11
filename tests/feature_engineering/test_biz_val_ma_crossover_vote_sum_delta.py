"""biz_value test for ``ma_crossover_features``'s ``vote_sum_delta`` output.

Source: extension of "MA-crossover difference and sign-comparison features" -- the row-over-row CHANGE in
``vote_sum`` (momentum-of-momentum) should spike distinctly AT a regime-shift ONSET moment (many pairs
flipping direction simultaneously), whereas the raw ``vote_sum`` level itself changes slowly/smoothly
during an already-established trend and doesn't single out the onset row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.ma_crossover import ma_crossover_features


def _make_regime_shift_series(n: int, shift_at: int, seed: int):
    rng = np.random.default_rng(seed)
    trend = np.where(np.arange(n) < shift_at, -0.4, 0.4)
    x = np.cumsum(trend + rng.normal(scale=0.3, size=n)) + 100
    label = np.zeros(n, dtype=int)
    label[shift_at : shift_at + 3] = 1  # "onset window": the moment consensus flips
    return pd.Series(x), label


def test_biz_val_ma_crossover_vote_sum_delta_detects_onset_better_than_vote_sum_level():
    s, label = _make_regime_shift_series(n=400, shift_at=200, seed=1)
    mas = {w: s.rolling(w).mean() for w in [3, 5, 10, 20]}
    feats = ma_crossover_features(mas)

    valid = ~(feats["ma_crossover_vote_sum"].isna() | feats["ma_crossover_vote_sum_delta"].isna())
    y = label[valid.to_numpy()]

    auc_delta = roc_auc_score(y, feats.loc[valid, "ma_crossover_vote_sum_delta"].abs())
    auc_level = roc_auc_score(y, feats.loc[valid, "ma_crossover_vote_sum"].abs())

    assert auc_delta >= 0.75, f"expected vote_sum_delta to flag the onset row, got auc={auc_delta:.4f}"
    assert auc_delta > auc_level + 0.1, f"expected delta to beat raw level at onset detection, got delta={auc_delta:.4f} level={auc_level:.4f}"


def test_ma_crossover_features_vote_sum_delta_is_first_difference_of_vote_sum():
    idx = pd.RangeIndex(6)
    mas = {3: pd.Series([1.0, 2.0, 4.0, 4.0, 1.0, 1.0], index=idx), 5: pd.Series([2.0] * 6, index=idx)}
    feats = ma_crossover_features(mas)

    expected = feats["ma_crossover_vote_sum"].diff().to_numpy()
    np.testing.assert_allclose(feats["ma_crossover_vote_sum_delta"].to_numpy()[1:], expected[1:])
    assert np.isnan(feats["ma_crossover_vote_sum_delta"].to_numpy()[0])


def test_ma_crossover_features_vote_sum_delta_resets_at_group_boundary():
    idx = pd.RangeIndex(6)
    mas = {3: pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], index=idx), 5: pd.Series([2.0] * 6, index=idx)}
    group_ids = np.array([0, 0, 0, 1, 1, 1])
    feats = ma_crossover_features(mas, group_ids=group_ids)

    delta = feats["ma_crossover_vote_sum_delta"].to_numpy()
    assert np.isnan(delta[0])
    assert np.isnan(delta[3])  # first row of the second group must not see the first group's last row
