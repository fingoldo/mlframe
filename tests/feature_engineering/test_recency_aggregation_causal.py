"""Causal mode of the per-entity recency-weighted aggregates."""

import numpy as np

from mlframe.feature_engineering.recency_aggregation import per_group_recency_weighted_agg, per_group_recency_weighted_mean


def test_causal_mean_row_does_not_see_later_observations():
    """The first row of an entity whose last value is 1 must not be influenced by that later value."""
    values = np.array([0.0, 0.0, 0.0, 1.0])
    groups = np.zeros(4, dtype=np.int64)
    out = per_group_recency_weighted_mean(values, groups, order=np.arange(4), scheme="exp", param=0.5)
    assert out[0] == 0.0 and out[2] == 0.0
    assert out[3] > 0.0


def test_causal_false_broadcasts_whole_history():
    """The one-shot mode gives every row the entity's whole-history value."""
    values = np.array([0.0, 0.0, 0.0, 1.0])
    groups = np.zeros(4, dtype=np.int64)
    out = per_group_recency_weighted_mean(values, groups, order=np.arange(4), scheme="exp", param=0.5, causal=False)
    assert np.all(out == out[0]) and out[0] > 0.0


def test_causal_last_row_equals_whole_history_value_and_order_is_respected():
    """Under shuffled input rows the last chronological row carries the whole-history aggregate; early rows carry prefixes."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=12)
    groups = np.repeat([1, 2, 3], 4)
    order = np.tile(np.arange(4), 3)
    perm = rng.permutation(12)
    values, groups, order = values[perm], groups[perm], order[perm]
    for agg in ("mean", "max", "std"):
        causal = per_group_recency_weighted_agg(values, groups, agg=agg, order=order, scheme="poly", param=1.0)
        whole = per_group_recency_weighted_agg(values, groups, agg=agg, order=order, scheme="poly", param=1.0, causal=False)
        last = order == 3
        np.testing.assert_allclose(causal[last], whole[last])
        first = order == 0
        np.testing.assert_allclose(causal[first][~np.isnan(causal[first])], values[first][~np.isnan(causal[first])] if agg != "max" else causal[first])


def test_causal_density_mode_and_stability_ignore_later_rows():
    """A late jump must not move the mode or stability score of earlier rows, and the one-shot mode still broadcasts one value."""
    from mlframe.feature_engineering.recency_density import per_group_behavioral_stability, per_group_recency_weighted_mode

    values = np.array([1.0, 1.1, 0.9, 1.0, 50.0])
    groups = np.zeros(5, dtype=np.int64)
    kw = dict(order=np.arange(5), scheme="exp", param=0.8)
    mode = per_group_recency_weighted_mode(values, groups, **kw)
    assert mode[3] < 2.0
    whole = per_group_recency_weighted_mode(values, groups, causal=False, **kw)
    assert np.all(whole == whole[0])
    stab = per_group_behavioral_stability(values, groups, **kw)
    assert stab.shape == (5,) and np.all(np.isfinite(stab[1:]))
