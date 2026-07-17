"""biz_value test for ``feature_engineering.recency_aggregation.per_group_recency_weighted_agg``.

Source: 9th_home-credit-default-risk.md -- ``MULTIPLIER=1.00-MONTHS_BALANCE/MIN_VALUES`` multiplied onto raw
values before aggregating with mean/min/max ("killer" feature per other competitors). Plain (unweighted) max
cannot distinguish a spike in the MOST RECENT observation from an equally large spike in an OLD, no-longer-
relevant observation -- both produce the same max. Recency-weighting each value toward 0 for older observations
before taking max makes only recent spikes survive, so a label driven purely by RECENT spikes should be far
more separable via recency-weighted max than via a plain per-entity max.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.recency_aggregation import per_group_recency_weighted_agg, per_group_recency_weighted_mean


def _make_recent_vs_old_spike_data(n_entities: int, n_hist: int, seed: int):
    rng = np.random.default_rng(seed)
    group_ids = np.repeat(np.arange(n_entities), n_hist)
    order = np.tile(np.arange(n_hist), n_entities)
    label = rng.integers(0, 2, n_entities)
    values = rng.normal(0.0, 1.0, size=n_entities * n_hist)
    for e in range(n_entities):
        base = e * n_hist
        if label[e] == 1:
            values[base + n_hist - 1] += 8.0  # recent spike -> should count
        else:
            values[base + 0] += 8.0  # old spike -> should NOT count
    return values, group_ids, order, label


def test_biz_val_recency_weighted_max_separates_recent_from_old_spikes_better_than_plain_max():
    n_entities, n_hist = 2000, 12
    values, group_ids, order, label = _make_recent_vs_old_spike_data(n_entities, n_hist, seed=0)

    plain_max = np.array([values[e * n_hist : (e + 1) * n_hist].max() for e in range(n_entities)])
    weighted_max = per_group_recency_weighted_agg(values, group_ids, agg="max", order=order, scheme="poly", param=1.0, broadcast=False)

    auc_plain = roc_auc_score(label, plain_max)
    auc_weighted = roc_auc_score(label, weighted_max)

    assert auc_weighted >= 0.95, f"expected recency-weighted max to cleanly separate recent-spike entities, got auc={auc_weighted:.4f}"
    assert auc_weighted > auc_plain + 0.3, (
        f"expected recency-weighted max to beat plain max by a wide margin, got weighted={auc_weighted:.4f} plain={auc_plain:.4f}"
    )


def test_per_group_recency_weighted_agg_mean_matches_dedicated_mean_function():
    n_entities, n_hist = 300, 8
    values, group_ids, order, _ = _make_recent_vs_old_spike_data(n_entities, n_hist, seed=1)

    agg_mean = per_group_recency_weighted_agg(values, group_ids, agg="mean", order=order, scheme="exp", param=0.7)
    dedicated_mean = per_group_recency_weighted_mean(values, group_ids, order=order, scheme="exp", param=0.7)

    np.testing.assert_allclose(agg_mean, dedicated_mean)


def test_per_group_recency_weighted_agg_sum_and_min_identity_at_param_extremes():
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    group_ids = np.array([0, 0, 0, 0])
    order = np.array([0, 1, 2, 3])

    # poly, param=0 -> weight == 1 everywhere (identity), so sum/min/max reduce to the plain aggregate.
    agg_sum = per_group_recency_weighted_agg(values, group_ids, agg="sum", order=order, scheme="poly", param=0.0, broadcast=False)
    agg_min = per_group_recency_weighted_agg(values, group_ids, agg="min", order=order, scheme="poly", param=0.0, broadcast=False)
    agg_max = per_group_recency_weighted_agg(values, group_ids, agg="max", order=order, scheme="poly", param=0.0, broadcast=False)

    assert agg_sum[0] == 10.0
    assert agg_min[0] == 1.0
    assert agg_max[0] == 4.0
