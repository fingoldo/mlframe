"""biz_value test for ``feature_engineering.ma_crossover.ma_crossover_features``.

Source: av_top3_rampaging_datahulk_minihack2017.md -- MA-difference and sign-comparison ("1, 0, -1") features
summed into a composite crossover score, a MACD-style technical-analysis pattern. A raw non-stationary price
series carries almost no trend-DIRECTION signal on its own (its level is dominated by the random-walk
component, not the current regime), while the MA-crossover vote_sum should cleanly separate uptrend vs
downtrend regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.ma_crossover import ma_crossover_features


def _make_alternating_trend_series(n: int, segment_len: int, seed: int):
    rng = np.random.default_rng(seed)
    trend_dir = np.repeat(rng.choice([-1, 1], n // segment_len + 1), segment_len)[:n]
    x = np.cumsum(trend_dir * 0.5 + rng.normal(scale=1.0, size=n)) + 100
    label = (trend_dir > 0).astype(int)
    return x, label


def test_biz_val_ma_crossover_vote_sum_separates_trend_regimes_better_than_raw_price():
    x, label = _make_alternating_trend_series(n=500, segment_len=40, seed=0)
    s = pd.Series(x)
    mas = {w: s.rolling(w).mean() for w in [3, 5, 10, 20]}
    feats = ma_crossover_features(mas)

    valid = ~feats["ma_crossover_vote_sum"].isna()
    auc_vote_sum = roc_auc_score(label[valid.to_numpy()], feats["ma_crossover_vote_sum"][valid])
    auc_raw = roc_auc_score(label[valid.to_numpy()], x[valid.to_numpy()])

    assert auc_vote_sum >= 0.9, f"expected the crossover vote_sum to cleanly separate trend regimes, got auc={auc_vote_sum:.4f}"
    assert auc_vote_sum > auc_raw + 0.3, f"expected vote_sum to beat raw price by a wide margin, got vote_sum={auc_vote_sum:.4f} raw={auc_raw:.4f}"


def test_ma_crossover_features_emits_all_pairs_and_correct_signs():
    idx = pd.RangeIndex(5)
    mas = {3: pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx), 5: pd.Series([2.0, 2.0, 2.0, 2.0, 2.0], index=idx)}
    feats = ma_crossover_features(mas)

    assert "ma_crossover_diff_3_5" in feats.columns
    assert "ma_crossover_vote_3_5" in feats.columns
    assert "ma_crossover_vote_sum" in feats.columns

    np.testing.assert_allclose(feats["ma_crossover_diff_3_5"].to_numpy(), [-1.0, 0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(feats["ma_crossover_vote_3_5"].to_numpy(), [-1.0, 0.0, 1.0, 1.0, 1.0])


def test_ma_crossover_features_requires_at_least_two_windows():
    import pytest

    with pytest.raises(ValueError):
        ma_crossover_features({3: pd.Series([1.0, 2.0, 3.0])})
