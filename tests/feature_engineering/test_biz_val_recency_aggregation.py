"""biz_value tests: recency-weighted per-entity aggregation beats the flat mean on a drifting panel.

The lecture's premise: "more recent data about a client matters more than stale data". On a panel where
each entity's value DRIFTS over time, a recency-weighted historical mean predicts the entity's NEXT value
with lower error than the flat mean. These tests pin that measurable win so a regression in the weighting
(e.g. weights silently uniform) fails the assertion instead of a shape check.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.recency_aggregation import per_group_recency_weighted_mean


def _make_drifting_panel(n_entities=400, hist=12, seed=0):
    """Each entity: a linearly drifting series + noise. Return (values, group_ids, order, next_value)."""
    rng = np.random.default_rng(seed)
    values, groups, orders, targets = [], [], [], []
    for e in range(n_entities):
        level = rng.uniform(-2, 2)
        slope = rng.uniform(0.3, 1.0) * rng.choice([-1, 1])
        t = np.arange(hist + 1, dtype=float)
        series = level + slope * t + rng.normal(0, 0.4, size=hist + 1)
        values.append(series[:hist])
        orders.append(t[:hist])
        groups.append(np.full(hist, e))
        targets.append(series[hist])  # the NEXT (held-out) value to predict
    return (
        np.concatenate(values),
        np.concatenate(groups),
        np.concatenate(orders),
        np.array(targets),
    )


def _rmse_of_prediction(scheme, param):
    """Helper: Rmse of prediction."""
    values, groups, order, targets = _make_drifting_panel()
    per_entity = per_group_recency_weighted_mean(values, groups, order=order, scheme=scheme, param=param, broadcast=False)
    return float(np.sqrt(np.mean((per_entity - targets) ** 2)))


def test_biz_val_recency_weighted_beats_flat_mean_on_drift():
    """Recency-weighted mean must predict the next value with >=15% lower RMSE than the flat mean on a drifting panel.

    Measured: flat ~ 4.9, exp(0.55) ~ 3.0 => ~0.61 ratio. Floor at 0.85 to absorb seed noise while catching a
    weighting regression (which would push the ratio back to ~1.0).
    """
    flat = _rmse_of_prediction("poly", 0.0)  # identity == flat mean
    weighted = _rmse_of_prediction("exp", 0.55)
    assert weighted < 0.85 * flat, f"recency-weighted RMSE {weighted:.3f} should beat flat {flat:.3f} by >=15%"


def test_biz_val_identity_param_equals_flat_mean():
    """poly delta=0 must reproduce the plain unweighted per-entity mean (bit-close)."""
    values, groups, order, _ = _make_drifting_panel(n_entities=50, hist=8, seed=3)
    weighted = per_group_recency_weighted_mean(values, groups, order=order, scheme="poly", param=0.0, broadcast=True)
    # Plain per-group mean via bincount.
    _, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    flat = (sums / counts)[inv]
    assert np.allclose(weighted, flat)


def test_biz_val_binary_event_rate_recency_weighted():
    """On a binary event series whose probability rises over time, recency weighting yields a higher (closer-to-current) rate."""
    rng = np.random.default_rng(7)
    n_ent, hist = 300, 16
    vals, groups, order = [], [], []
    for e in range(n_ent):
        p = np.linspace(0.1, 0.85, hist)  # event prob rises
        ev = (rng.random(hist) < p).astype(float)
        vals.append(ev)
        groups.append(np.full(hist, e))
        order.append(np.arange(hist, dtype=float))
    vals = np.concatenate(vals)
    groups = np.concatenate(groups)
    order = np.concatenate(order)
    flat = per_group_recency_weighted_mean(vals, groups, order=order, scheme="poly", param=0.0, broadcast=False)
    weighted = per_group_recency_weighted_mean(vals, groups, order=order, scheme="exp", param=0.6, broadcast=False)
    # Current-regime event rate ~0.85; recency-weighted estimate should sit closer to it than the flat ~0.475.
    assert np.nanmean(weighted) > np.nanmean(flat) + 0.1
