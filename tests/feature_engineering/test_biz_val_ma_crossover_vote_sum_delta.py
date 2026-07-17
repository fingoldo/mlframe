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


def test_ma_crossover_features_default_matches_zero_weight_power_bit_identical():
    # long_window_weight_power's default (0.0) must reproduce the original equal-weight vote exactly.
    s, _ = _make_regime_shift_series(n=200, shift_at=100, seed=2)
    mas = {w: s.rolling(w).mean() for w in [3, 5, 10, 20]}

    default_feats = ma_crossover_features(mas)
    explicit_zero_feats = ma_crossover_features(mas, long_window_weight_power=0.0)

    pd.testing.assert_frame_equal(default_feats, explicit_zero_feats)


def _consensus_false_flip_rate(vote_sum: np.ndarray, trend_sign: np.ndarray) -> float:
    # a "false flip" = the sign of vote_sum disagreeing with the true underlying trend direction.
    valid = ~np.isnan(vote_sum)
    consensus_sign = np.sign(vote_sum[valid])
    return float(np.mean(consensus_sign != trend_sign[valid]))


def test_biz_val_ma_crossover_long_window_weight_power_reduces_false_flips_vs_equal_weight():
    # Synthetic: the price level is a deterministic drift ramp (slope alternates sign every `segment`
    # bars -- a regime-switching trend) plus STATIONARY i.i.d. observation noise added ON TOP of the
    # ramp (not cumsummed together with it -- `cumsum(rate + noise)` would turn the noise itself into an
    # unbounded random walk that swamps the trend at every window length, defeating any pair-weighting).
    # A rolling mean of window w recovers the ramp's slope (up to a small lag term) while shrinking the
    # noise's contribution by ~1/sqrt(w). The short windows here (2,3,4,5) barely average any noise away
    # -- each short-short vote is close to a coin flip -- while every pair anchored on the single long
    # window (100) averages the noise down enough to track the true drift direction reliably. Equal-
    # weight voting lets the 6 noisy short-short votes swamp the 4 reliable long-anchored votes; weighting
    # by the long window size recovers a consensus with materially fewer false flips (verified stable
    # across many random seeds during development).
    rng = np.random.default_rng(7)
    n = 3000
    segment = 500
    slope = 0.4
    rate = np.where((np.arange(n) // segment) % 2 == 0, slope, -slope)
    trend_sign = np.sign(rate)
    ramp = np.cumsum(rate)  # deterministic drift, no noise accumulated into the walk itself
    noise = rng.normal(scale=8.0, size=n)  # stationary observation noise, added on top (not integrated)
    x = pd.Series(ramp + noise)

    windows = [2, 3, 4, 5, 100]
    mas = {w: x.rolling(w).mean() for w in windows}

    equal_feats = ma_crossover_features(mas)
    weighted_feats = ma_crossover_features(mas, long_window_weight_power=2.0)

    equal_rate = _consensus_false_flip_rate(equal_feats["ma_crossover_vote_sum"].to_numpy(), trend_sign)
    weighted_rate = _consensus_false_flip_rate(weighted_feats["ma_crossover_vote_sum"].to_numpy(), trend_sign)

    assert weighted_rate <= 0.30, f"expected weighted consensus false-flip rate <=0.30, got {weighted_rate:.4f}"
    assert weighted_rate < equal_rate - 0.15, (
        f"expected weighted voting to cut false flips well below equal-weight, got weighted={weighted_rate:.4f} equal={equal_rate:.4f}"
    )
