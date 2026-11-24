"""Cluster 24 regression: the causal exp-scheme aggregation uses a one-pass recurrence and matches the prefix re-walk (audit 2026-11-20, finding 24-F16).

The causal broadcast re-walked each row's prefix (O(m^2) per group). For the exp scheme the prefix means now come from one recurrence per group.
The test rebuilds the old path (prefix bounds + the per-prefix kernel) as a reference and compares, including NaN values and several groups.
"""

import numpy as np

from mlframe.feature_engineering.recency_aggregation import (
    _group_recency_weighted_mean_sorted,
    _prefix_bounds,
    per_group_recency_weighted_mean,
)


def _reference(values, group_ids, param):
    """The old causal path: per-row prefix windows through the per-group kernel, mapped back to original order."""
    n = values.shape[0]
    sort_idx = np.argsort(group_ids, kind="stable")
    g_sorted = group_ids[sort_idx]
    v_sorted = values[sort_idx]
    bnd = np.where(g_sorted[1:] != g_sorted[:-1])[0] + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), bnd.astype(np.int64)))
    ends = np.concatenate((bnd.astype(np.int64), np.array([n], dtype=np.int64)))
    p_starts, p_ends = _prefix_bounds(starts, ends)
    out = np.empty(n, dtype=np.float64)
    out[sort_idx] = _group_recency_weighted_mean_sorted(v_sorted, p_starts, p_ends, 1, float(param))
    return out


def test_exp_causal_matches_prefix_reference_with_nans():
    """The recurrence reproduces the prefix-walk values for several parameters, with NaN rows (24-F16)."""
    rng = np.random.default_rng(0)
    n = 300
    values = rng.normal(size=n)
    values[rng.random(n) < 0.15] = np.nan
    group_ids = rng.integers(0, 7, size=n)
    for param in (0.9, 1.0, 1.3):
        got = per_group_recency_weighted_mean(values, group_ids, scheme="exp", param=param, causal=True, broadcast=True)
        want = _reference(values, group_ids, param)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=f"param={param}")


def test_all_nan_prefix_is_nan():
    """A prefix with no observed value is NaN, as in the reference (24-F16)."""
    values = np.array([np.nan, np.nan, 2.0])
    group_ids = np.array([0, 0, 0])
    got = per_group_recency_weighted_mean(values, group_ids, scheme="exp", param=0.5, causal=True, broadcast=True)
    assert np.isnan(got[0]) and np.isnan(got[1])
    assert np.isclose(got[2], 2.0)
