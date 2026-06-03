"""Leakage-firewall regression (audit 2026-06-03: integration-defaults-1).

``dcd_distance='sotoca_pla'`` is a TARGET-AWARE distance (it subtracts
I(X_a;Y) + I(X_b;Y)). It was being used by ``pair_su`` to decide UNSUPERVISED
cluster membership / pool pruning -- so the target was silently deciding which
candidate features get pruned from the support, with no later y-aware
accept/reject re-screen for the pruned ones. That breaches the leakage firewall
the rest of the clustering machinery honours.

Fix: ``pair_su`` always computes the unsupervised symmetric uncertainty for the
``sotoca_pla`` distance. These tests assert that the membership score is now
identical to ``su`` (y no longer changes which candidates are pruned), including
on a pair that is BOTH mutually redundant AND target-informative -- exactly the
case where the old y-aware score diverged from SU.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
    make_dcd_state,
    pair_su,
)


def _state(distance, fd, fn, target_indices=None):
    return make_dcd_state(
        X_raw=None, factors_data=fd, factors_nbins=fn,
        cols=[f"c{i}" for i in range(fd.shape[1])], nbins=fn,
        target_indices=target_indices, distance=distance,
    )


def _redundant_and_target_informative(seed=0, n=3000):
    """Two features that are highly mutually redundant AND both informative
    about y -- the regime where the old sotoca_pla score (1 - d, with d
    subtracting I(X;Y)) diverged from SU."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 4, n).astype(np.int32)
    a = base.copy()
    flip = rng.random(n) < 0.1
    b = np.where(flip, (base + 1) % 4, base).astype(np.int32)
    y = (base >= 2).astype(np.int32)  # target informative about base -> a, b
    fd = np.column_stack([a, b, y])
    fn = np.array([4, 4, 2], dtype=np.int64)
    return fd, fn


def test_sotoca_pla_membership_score_equals_su():
    fd, fn = _redundant_and_target_informative()
    tgt = np.array([2], dtype=np.int64)
    s_soto = _state("sotoca_pla", fd, fn, target_indices=tgt)
    s_su = _state("su", fd, fn, target_indices=tgt)
    sc_soto = pair_su(s_soto, 0, 1, entropy_cache=None, factors_data=fd, factors_nbins=fn)
    sc_su = pair_su(s_su, 0, 1, entropy_cache=None, factors_data=fd, factors_nbins=fn)
    assert abs(float(sc_soto) - float(sc_su)) < 1e-9, (
        f"sotoca_pla membership must equal unsupervised SU (firewall); "
        f"got sotoca={sc_soto} vs su={sc_su}"
    )


def test_sotoca_pla_score_invariant_to_target_choice():
    """The membership score must NOT change when the target column changes,
    since membership is unsupervised. Pre-fix it varied with I(X;Y)."""
    fd, fn = _redundant_and_target_informative()
    # Two different "targets": the real y (col 2), and a constant pseudo-target.
    fd2 = np.column_stack([fd[:, 0], fd[:, 1], np.zeros(fd.shape[0], dtype=np.int32)])
    fn2 = np.array([4, 4, 1], dtype=np.int64)
    s1 = _state("sotoca_pla", fd, fn, target_indices=np.array([2], dtype=np.int64))
    s2 = _state("sotoca_pla", fd2, fn2, target_indices=np.array([2], dtype=np.int64))
    sc1 = pair_su(s1, 0, 1, entropy_cache=None, factors_data=fd, factors_nbins=fn)
    sc2 = pair_su(s2, 0, 1, entropy_cache=None, factors_data=fd2, factors_nbins=fn2)
    assert abs(float(sc1) - float(sc2)) < 1e-9, (
        f"sotoca_pla membership score changed with the target column "
        f"({sc1} vs {sc2}) -- y is still leaking into unsupervised pruning"
    )
