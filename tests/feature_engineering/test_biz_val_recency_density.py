"""biz_value tests for recency-weighted Parzen density: mode-vs-mean and behavioral stability."""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.recency_density import (
    per_group_behavioral_stability,
    per_group_recency_weighted_mode,
)
from mlframe.feature_engineering.recency_aggregation import per_group_recency_weighted_mean


def test_biz_val_mode_beats_mean_on_skewed_bimodal_spend():
    """On entities whose spend is a tight cluster plus rare large spikes, the density MODE predicts the typical
    (cluster) value with lower error than the mean, which is dragged up by the spikes.

    Target = each entity's cluster center. Measured: mode RMSE ~ 3-6, mean RMSE ~ 12-20 => mode wins clearly.
    """
    rng = np.random.default_rng(0)
    n_ent, hist = 250, 20
    vals, groups, order, centers = [], [], [], []
    for e in range(n_ent):
        center = rng.uniform(10, 60)
        base = rng.normal(center, 2.0, size=hist)
        # inject rare large spikes (~15% of points) that skew the mean but not the mode
        n_spikes = max(1, int(0.15 * hist))
        spike_idx = rng.choice(hist, size=n_spikes, replace=False)
        base[spike_idx] += rng.uniform(80, 200, size=n_spikes)
        vals.append(base)
        groups.append(np.full(hist, e))
        order.append(np.arange(hist, dtype=float))
        centers.append(center)
    vals = np.concatenate(vals)
    groups = np.concatenate(groups)
    order = np.concatenate(order)
    centers = np.array(centers)

    mode = per_group_recency_weighted_mode(vals, groups, order=order, scheme="poly", param=0.0, broadcast=False)
    mean = per_group_recency_weighted_mean(vals, groups, order=order, scheme="poly", param=0.0, broadcast=False)
    rmse_mode = float(np.sqrt(np.mean((mode - centers) ** 2)))
    rmse_mean = float(np.sqrt(np.mean((mean - centers) ** 2)))
    assert rmse_mode < 0.7 * rmse_mean, f"mode RMSE {rmse_mode:.2f} should beat mean RMSE {rmse_mean:.2f} on skewed spend"


def test_biz_val_stability_separates_predictable_from_erratic():
    """Concentrated (predictable) entities must score higher stability than spread-out (erratic) ones."""
    rng = np.random.default_rng(1)
    hist = 24
    vals, groups, order = [], [], []
    labels = []  # True = concentrated
    for e in range(200):
        concentrated = e % 2 == 0
        if concentrated:
            series = rng.normal(50, 1.5, size=hist)
        else:
            series = rng.uniform(0, 100, size=hist)
        vals.append(series)
        groups.append(np.full(hist, e))
        order.append(np.arange(hist, dtype=float))
        labels.append(concentrated)
    vals = np.concatenate(vals)
    groups = np.concatenate(groups)
    order = np.concatenate(order)
    labels = np.array(labels)

    stab = per_group_behavioral_stability(vals, groups, order=order, scheme="poly", param=0.0, broadcast=False)
    assert np.nanmean(stab[labels]) > np.nanmean(stab[~labels]) + 0.05, "concentrated entities must score higher stability"


def test_biz_val_mode_identity_stable_on_unimodal():
    """On clean unimodal data mode and mean agree closely (sanity: no spurious multimodality artifacts)."""
    rng = np.random.default_rng(2)
    hist = 30
    vals, groups, order = [], [], []
    for e in range(40):
        series = rng.normal(e, 1.0, size=hist)
        vals.append(series)
        groups.append(np.full(hist, e))
        order.append(np.arange(hist, dtype=float))
    vals = np.concatenate(vals)
    groups = np.concatenate(groups)
    order = np.concatenate(order)
    mode = per_group_recency_weighted_mode(vals, groups, order=order, scheme="poly", param=0.0, broadcast=False)
    mean = per_group_recency_weighted_mean(vals, groups, order=order, scheme="poly", param=0.0, broadcast=False)
    assert np.max(np.abs(mode - mean)) < 3.0, "mode and mean should be close on clean unimodal data"
