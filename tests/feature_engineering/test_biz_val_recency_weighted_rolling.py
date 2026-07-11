"""biz_value test for ``feature_engineering.recency_weighted_rolling.recency_weighted_rolling_mean``.

Synthetic: each entity's underlying rate steps from a LOW regime to a HIGH regime partway through its trailing
window. A plain uniform rolling mean averages old (low) and new (high) observations equally, lagging the
regime shift; a recency-weighted rolling mean (favoring recent observations within the same fixed window) tracks
the shift faster, so its per-row estimate is closer to the TRUE current-regime value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_engineering.recency_weighted_rolling import recency_weighted_rolling_mean


def _make_regime_shift_dataset(n_entities: int, rows_per_entity: int, window: int, seed: int):
    rng = np.random.default_rng(seed)
    entity_ids = np.repeat(np.arange(n_entities), rows_per_entity)
    order = np.tile(np.arange(rows_per_entity), n_entities)
    # Each entity: first half at rate 0.2, second half at rate 0.8 (step regime shift).
    shift_point = rows_per_entity // 2
    true_rate = np.where(order < shift_point, 0.2, 0.8)
    values = np.clip(true_rate + rng.normal(scale=0.05, size=n_entities * rows_per_entity), 0, 1)
    return pd.DataFrame({"entity": entity_ids, "order": order, "value": values, "true_rate": true_rate})


def test_biz_val_recency_weighted_rolling_tracks_regime_shift_better_than_uniform():
    window = 10
    df = _make_regime_shift_dataset(n_entities=200, rows_per_entity=30, window=window, seed=0)

    weighted = recency_weighted_rolling_mean(df["value"].to_numpy(), df["entity"].to_numpy(), window=window, order=df["order"].to_numpy(), scheme="exp", param=0.6)
    uniform = df.groupby("entity")["value"].transform(lambda s: s.rolling(window=window, min_periods=1).mean()).to_numpy()

    # Only score rows shortly AFTER the regime shift (where the window still spans both regimes) -- that is
    # exactly where recency weighting should help; far from the shift both estimators converge to the truth.
    shift_point = 30 // 2
    near_shift = (df["order"].to_numpy() >= shift_point) & (df["order"].to_numpy() < shift_point + window)

    mse_weighted = float(np.mean((weighted[near_shift] - df["true_rate"].to_numpy()[near_shift]) ** 2))
    mse_uniform = float(np.mean((uniform[near_shift] - df["true_rate"].to_numpy()[near_shift]) ** 2))

    assert mse_weighted < mse_uniform * 0.8, f"expected recency-weighted rolling mean to track the regime shift better, got weighted={mse_weighted:.5f} uniform={mse_uniform:.5f}"


def test_recency_weighted_rolling_mean_identity_param_matches_uniform_rolling():
    window = 5
    df = _make_regime_shift_dataset(n_entities=20, rows_per_entity=15, window=window, seed=1)

    weighted = recency_weighted_rolling_mean(df["value"].to_numpy(), df["entity"].to_numpy(), window=window, order=df["order"].to_numpy(), scheme="poly", param=0.0)
    uniform = df.groupby("entity")["value"].transform(lambda s: s.rolling(window=window, min_periods=1).mean()).to_numpy()

    np.testing.assert_allclose(weighted, uniform, atol=1e-10)


def test_recency_weighted_rolling_mean_rejects_invalid_window():
    df = _make_regime_shift_dataset(n_entities=5, rows_per_entity=10, window=3, seed=2)
    import pytest

    with pytest.raises(ValueError):
        recency_weighted_rolling_mean(df["value"].to_numpy(), df["entity"].to_numpy(), window=0)


def test_recency_weighted_rolling_mean_preserves_original_row_order():
    df = _make_regime_shift_dataset(n_entities=10, rows_per_entity=8, window=4, seed=3)
    shuffled = df.sample(frac=1.0, random_state=0)
    out_shuffled = recency_weighted_rolling_mean(shuffled["value"].to_numpy(), shuffled["entity"].to_numpy(), window=4, order=shuffled["order"].to_numpy(), scheme="exp", param=0.5)
    out_original = recency_weighted_rolling_mean(df["value"].to_numpy(), df["entity"].to_numpy(), window=4, order=df["order"].to_numpy(), scheme="exp", param=0.5)
    # Same row, regardless of input order, must get the same computed value -- align by (entity, order) key.
    shuffled_lookup = dict(zip(zip(shuffled["entity"], shuffled["order"]), out_shuffled))
    for i in range(len(df)):
        key = (df["entity"].iloc[i], df["order"].iloc[i])
        assert np.isclose(shuffled_lookup[key], out_original[i])
