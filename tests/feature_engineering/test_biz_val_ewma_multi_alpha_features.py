"""biz_value test for ``feature_engineering.ewma_multi_alpha_features.ewma_multi_alpha_features``.

Source: 4th_santander-product-recommendation.md -- "Exponential weighted average of each product's presence
per client as time goes. I've used two different alphas - 0.5 and 0.1 ... least susceptible to the amount of
given data points." A fixed-window rolling mean (e.g. last-3) estimates an entity's persistent rate from very
few samples (high variance); a low-alpha (long-memory) EWMA effectively averages over MORE of the entity's
history without needing a fixed window size, giving a lower-variance rate estimate -- this should predict a
persistent hidden per-entity trait better than a small fixed window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.ewma_multi_alpha_features import ewma_multi_alpha_features


def _make_entities_with_hidden_rate(n_entities: int, seed: int):
    """Helper: Make entities with hidden rate."""
    rng = np.random.default_rng(seed)
    presence_list, group_list = [], []
    true_rate = {}
    for e in range(n_entities):
        hist_len = rng.integers(10, 30)
        rate = rng.beta(2, 2)
        presence = (rng.random(hist_len) < rate).astype(float)
        presence_list.append(presence)
        group_list.append(np.full(hist_len, e))
        true_rate[e] = rate
    return presence_list, np.concatenate(group_list), true_rate


def test_biz_val_low_alpha_ewma_beats_fixed_window_rolling_mean():
    """Biz val low alpha ewma beats fixed window rolling mean."""
    presence_list, group_ids, true_rate = _make_entities_with_hidden_rate(n_entities=500, seed=1)
    presence = np.concatenate(presence_list)

    res = ewma_multi_alpha_features(presence, group_ids, alphas=[0.5, 0.1])
    df = pd.DataFrame(res)
    df["group"] = group_ids
    last_per_entity = df.groupby("group").last()

    label = (pd.Series(true_rate).reindex(last_per_entity.index) > 0.5).astype(int)
    roll3 = pd.Series({e: p[-3:].mean() for e, p in enumerate(presence_list)}).reindex(last_per_entity.index)

    auc_ewma_long = roc_auc_score(label, last_per_entity["ewma_alpha_0.1"])
    auc_roll3 = roc_auc_score(label, roll3)

    assert (
        auc_ewma_long > auc_roll3 + 0.05
    ), f"expected low-alpha EWMA to beat a fixed 3-window rolling mean by >=0.05 AUC, got ewma={auc_ewma_long:.4f} roll3={auc_roll3:.4f}"


def test_ewma_multi_alpha_features_matches_pandas_ewm():
    """Ewma multi alpha features matches pandas ewm."""
    values = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 5.0, 5.0, 4.0])
    groups = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    res = ewma_multi_alpha_features(values, groups, alphas=[0.5])

    expected_0 = pd.Series(values[:5]).ewm(alpha=0.5, adjust=False).mean().to_numpy()
    expected_1 = pd.Series(values[5:]).ewm(alpha=0.5, adjust=False).mean().to_numpy()
    np.testing.assert_allclose(res["ewma_alpha_0.5"], np.concatenate([expected_0, expected_1]))


def test_ewma_multi_alpha_features_rejects_invalid_alpha():
    """Ewma multi alpha features rejects invalid alpha."""
    import pytest

    with pytest.raises(ValueError):
        ewma_multi_alpha_features(np.array([1.0, 2.0]), np.array([0, 0]), alphas=[1.5])
