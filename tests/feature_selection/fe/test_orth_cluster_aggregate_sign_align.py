"""Regression guard for the orthogonal-basis cluster aggregator sign-alignment
(audit 2026-06-03: gap-03).

detect_clusters_by_correlation links members on |corr|, so a cluster may
contain an anticorrelated reflection ({z, -z+eps}). The old compute_cluster_-
aggregate z-scored each member with NO sign alignment, so under mean_z the two
reflections cancelled to a near-constant (dead) aggregate. The fix sign-aligns
members to the reference before combining, so reflections add constructively
(matching the canonical filters._cluster_aggregate path).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_cluster_basis_fe import (
    compute_cluster_aggregate,
)


def _anticorrelated_reflections(seed=0, n=500):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    X = pd.DataFrame({
        "a": z + 0.05 * rng.standard_normal(n),
        "b": -z + 0.05 * rng.standard_normal(n),   # anticorrelated reflection
        "c": z + 0.05 * rng.standard_normal(n),
    })
    return X, z


def test_mean_z_does_not_cancel_anticorrelated_members():
    X, z = _anticorrelated_reflections()
    agg = compute_cluster_aggregate(X, ["a", "b", "c"], aggregator="mean_z")
    # Pre-fix: a,c ~ +u and b ~ -u -> mean ~ (u - u + u)/3, partial cancellation
    # and for the 2-member {a,b} case ~0. Post-fix the aligned mean tracks z.
    assert float(np.std(agg)) > 0.5, f"aggregate cancelled (std={np.std(agg):.4f})"
    assert abs(float(np.corrcoef(agg, z)[0, 1])) > 0.9


def test_two_member_anticorrelated_not_dead():
    X, z = _anticorrelated_reflections()
    agg = compute_cluster_aggregate(X, ["a", "b"], aggregator="mean_z")
    assert float(np.std(agg)) > 0.5, (
        f"2-member anticorrelated cluster cancelled to a dead column "
        f"(std={np.std(agg):.4f})"
    )


def test_pc1_recovers_latent_for_anticorrelated_cluster():
    X, z = _anticorrelated_reflections()
    agg = compute_cluster_aggregate(X, ["a", "b", "c"], aggregator="pc1")
    assert abs(float(np.corrcoef(agg, z)[0, 1])) > 0.9
