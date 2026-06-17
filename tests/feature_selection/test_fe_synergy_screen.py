# -*- coding: utf-8 -*-
"""Detection-vs-noise gate for the joint-synergy pair screen (2026-06-17).

The MARGINAL FE pair screen cannot see a pure-synergy pair (XOR: both operands ~zero marginal MI),
the documented I4/I5 barrier. These tests pin that the bias-corrected JOINT MI:
  1. detects the XOR signal pair (high joint MI where both marginals are ~0), and
  2. ranks it ABOVE every noise pair with a wide margin (no noise admission), at small + moderate n.

This is the gate the I4/I5 re-platform requires before the screen is wired into the rung-0 search.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_synergy_screen import joint_synergy_mi
from mlframe.feature_selection.filters.discretization import discretize_array


def _q(arr, nb):
    return discretize_array(arr=np.asarray(arr, dtype=float), n_bins=nb, method="quantile", dtype=np.int32)


@pytest.mark.parametrize("n", [8000, 25000])
@pytest.mark.parametrize("nb", [6, 8])
def test_joint_mi_detects_xor_marginals_miss(n, nb):
    """XOR has ~zero marginal MI on each operand but large JOINT MI; a noise pair has ~zero both."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    y = ((a > 0) ^ (b > 0)).astype(np.int32)
    ca, cb = _q(a, nb), _q(b, nb)
    # both marginals near zero (XOR is balanced w.r.t. each operand alone)
    assert joint_synergy_mi(ca, ca[::-1] * 0, y) >= 0.0  # smoke: degenerate operand -> finite, >=0
    sig = joint_synergy_mi(ca, cb, y)
    # a genuine noise pair
    nz = joint_synergy_mi(_q(rng.standard_normal(n), nb), _q(rng.standard_normal(n), nb), rng.integers(0, 2, n))
    assert sig > 0.25, f"XOR joint synergy not detected: {sig:.4f}"
    assert sig > 20.0 * max(nz, 1e-4), f"signal {sig:.4f} not separated from noise {nz:.4f}"


@pytest.mark.parametrize("n", [8000, 25000])
def test_xor_pair_ranks_first_among_noise(n):
    """1 XOR signal pair hidden among 189 noise pairs (P=20) must rank #1 by joint synergy MI."""
    nb = 8
    rng = np.random.default_rng(1)
    P = 20
    cols = [_q(rng.standard_normal(n), nb) for _ in range(P)]
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    cols[0], cols[1] = _q(a, nb), _q(b, nb)
    y = ((a > 0) ^ (b > 0)).astype(np.int32)
    scored = []
    for i in range(P):
        for j in range(i + 1, P):
            scored.append(((i, j), joint_synergy_mi(cols[i], cols[j], y)))
    scored.sort(key=lambda t: -t[1])
    assert scored[0][0] == (0, 1), f"XOR pair did not rank #1: top={scored[0]}"
    sig = dict(scored)[(0, 1)]
    max_noise = max(v for p, v in scored if p != (0, 1))
    assert sig > 20.0 * max(max_noise, 1e-4), f"signal {sig:.4f} vs max_noise {max_noise:.4f} too close"
