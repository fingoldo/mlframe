"""biz_value test for ``per_group_recency_weighted_agg(agg='std'|'var')``.

Same "recent event, plain aggregate misses it" pattern as the sibling max/min biz_value test, applied to
dispersion: an entity whose volatility REGIME changed recently (was calm, then a burst of high-variance
observations in the last few rows) has the same overall/global std as a calm-throughout entity, once the
burst is a small fraction of the whole history -- a plain per-entity std averages over the entire history and
is blind to WHEN the volatility happened. Recency-weighted std, by concentrating weight on the newest rows,
should surface the regime change and separate "recently turned volatile" entities from "calm throughout" ones,
while plain std (== recency-weighted std at the identity poly param=0) stays near chance.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.recency_aggregation import per_group_recency_weighted_agg


def _make_recent_volatility_regime_change_data(n_entities: int, n_hist: int, n_recent_burst: int, sigma_calm: float, sigma_burst: float, seed: int):
    """Every entity gets exactly one volatility burst of the same magnitude somewhere in its history --
    only its POSITION (recent vs old) differs by label -- so the two classes have identical GLOBAL variance
    by construction and a plain std is uninformative; only recency-weighting can tell them apart.
    """
    rng = np.random.default_rng(seed)
    group_ids = np.repeat(np.arange(n_entities), n_hist)
    order = np.tile(np.arange(n_hist), n_entities)
    label = rng.integers(0, 2, n_entities)
    values = rng.normal(0.0, sigma_calm, size=n_entities * n_hist)
    for e in range(n_entities):
        base = e * n_hist
        if label[e] == 1:
            burst_start = n_hist - n_recent_burst  # burst lands in the most recent slots -> regime just changed.
        else:
            burst_start = rng.integers(0, n_hist - n_recent_burst)  # burst lands somewhere old -> already back to calm.
        values[base + burst_start : base + burst_start + n_recent_burst] = rng.normal(0.0, sigma_burst, size=n_recent_burst)
    return values, group_ids, order, label


def test_biz_val_recency_weighted_std_separates_recent_volatility_regime_change_from_plain_std():
    """Biz val recency weighted std separates recent volatility regime change from plain std."""
    n_entities, n_hist, n_recent_burst = 4000, 30, 4
    values, group_ids, order, label = _make_recent_volatility_regime_change_data(n_entities, n_hist, n_recent_burst, sigma_calm=1.0, sigma_burst=8.0, seed=0)

    plain_std = per_group_recency_weighted_agg(values, group_ids, agg="std", order=order, scheme="poly", param=0.0, broadcast=False)
    weighted_std = per_group_recency_weighted_agg(values, group_ids, agg="std", order=order, scheme="poly", param=10.0, broadcast=False)

    auc_plain = roc_auc_score(label, plain_std)
    auc_weighted = roc_auc_score(label, weighted_std)

    assert auc_weighted >= 0.90, f"expected recency-weighted std to separate recent-regime-change entities, got auc={auc_weighted:.4f}"
    assert auc_weighted > auc_plain + 0.3, (
        f"expected recency-weighted std to beat plain std by a wide margin, got weighted={auc_weighted:.4f} plain={auc_plain:.4f}"
    )


def test_per_group_recency_weighted_agg_var_is_squared_std():
    """Per group recency weighted agg var is squared std."""
    n_entities, n_hist, n_recent_burst = 500, 20, 3
    values, group_ids, order, _ = _make_recent_volatility_regime_change_data(n_entities, n_hist, n_recent_burst, sigma_calm=1.0, sigma_burst=5.0, seed=1)

    weighted_std = per_group_recency_weighted_agg(values, group_ids, agg="std", order=order, scheme="exp", param=0.8, broadcast=False)
    weighted_var = per_group_recency_weighted_agg(values, group_ids, agg="var", order=order, scheme="exp", param=0.8, broadcast=False)

    np.testing.assert_allclose(weighted_std**2, weighted_var, rtol=1e-10)


def test_per_group_recency_weighted_agg_std_matches_manual_weighted_formula():
    """Per group recency weighted agg std matches manual weighted formula."""
    values = np.array([1.0, 3.0, 2.0, 10.0, 4.0], dtype=np.float64)
    group_ids = np.zeros(5, dtype=np.int64)
    order = np.arange(5)

    scheme, param = "poly", 2.0
    m = 5
    weights = np.array([((m - i + 1) / m) ** param for i in range(m, 0, -1)])
    mean = np.sum(weights * values) / np.sum(weights)
    expected_var = np.sum(weights * (values - mean) ** 2) / np.sum(weights)

    got_var = per_group_recency_weighted_agg(values, group_ids, agg="var", order=order, scheme=scheme, param=param, broadcast=False)
    got_std = per_group_recency_weighted_agg(values, group_ids, agg="std", order=order, scheme=scheme, param=param, broadcast=False)

    np.testing.assert_allclose(got_var[0], expected_var, rtol=1e-10)
    np.testing.assert_allclose(got_std[0], np.sqrt(expected_var), rtol=1e-10)


def test_per_group_recency_weighted_agg_std_single_observation_group_is_nan():
    """Per group recency weighted agg std single observation group is nan."""
    values = np.array([1.0])
    group_ids = np.array([0])
    order = np.array([0])

    out = per_group_recency_weighted_agg(values, group_ids, agg="std", order=order, broadcast=False)
    assert np.isnan(out[0])
