"""Cluster 22 regression: the factorised zero-sum sweeps match a plain pandas groupby reference (audit 2026-11-20, finding 22-F17)."""

import numpy as np
import pandas as pd

from mlframe.calibration.group_zero_sum_constraint import apply_group_zero_sum_constraint, apply_group_zero_sum_constraint_multi


def _reference_single(pred, group, w, target):
    """The pandas groupby formulation the sweeps used before."""
    df = pd.DataFrame({"w": w, "wp": pred * w, "g": group})
    agg = df.groupby("g", sort=False).agg(ws=("w", "sum"), wps=("wp", "sum"))
    off = pd.Series(np.where(agg["ws"] != 0, (agg["wps"] - target) / agg["ws"], 0.0), index=agg.index)
    return pred - pd.Series(group).map(off).fillna(0.0).to_numpy()


def test_single_constraint_matches_the_groupby_reference_with_weights_and_nan_labels():
    """Weighted shift per group, NaN-labelled rows untouched, a zero-weight group left alone."""
    rng = np.random.default_rng(0)
    n = 500
    pred = rng.normal(size=n)
    group = rng.integers(0, 20, n).astype(float)
    group[:7] = np.nan
    w = rng.uniform(0.5, 2.0, n)
    w[group == 3] = 0.0
    got = apply_group_zero_sum_constraint(pred, group, weights=w, target_sum=0.25)
    np.testing.assert_allclose(got, _reference_single(pred, group, w, 0.25), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(got[:7], pred[:7])


def test_multi_constraint_still_satisfies_every_grouping():
    """After the sweeps both groupings' weighted sums are zero to tolerance, as the convergence tests promise."""
    rng = np.random.default_rng(1)
    n = 2000
    pred = rng.normal(size=n)
    g1 = rng.integers(0, 15, n)
    g2 = rng.integers(0, 12, n)
    out = apply_group_zero_sum_constraint_multi(pred, [g1, g2], max_iterations=200, tol=1e-10)
    for g in (g1, g2):
        sums = pd.Series(out).groupby(g).sum().to_numpy()
        assert np.abs(sums).max() < 1e-6
