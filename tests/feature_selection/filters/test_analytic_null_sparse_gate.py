"""Regression: the analytic batched FE noise gate must only trust the G-test tail to REJECT where
the chi-square asymptotic is valid. On a sparse / high-cardinality contingency table
(analytic_null_applicable False) the analytic p is unreliable, so a genuine-signal candidate must be
KEPT, not zeroed. Pre-fix the gate ran the chi-square unconditionally and could reject on an invalid test.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._analytic_mi_null import (
    analytic_batch_noise_gate,
    analytic_null_applicable,
)


def test_sparse_high_cardinality_candidate_not_rejected():
    """On a sparse high-cardinality contingency table (chi-square asymptotic invalid), the noise gate keeps a tiny-but-positive candidate rather than zeroing it."""
    n = 100
    rng = np.random.default_rng(0)
    disc = rng.integers(0, 50, size=(n, 1)).astype(np.int64)  # ~50 occupied x bins (high cardinality)
    classes_y = rng.integers(0, 3, size=n).astype(np.int64)  # by = 3
    bx = int(np.unique(disc[:, 0]).size)
    assert not analytic_null_applicable(n, bx, 3)  # sparse: n/(bx*by) << min-cell floor
    observed = np.array([0.001])  # tiny but positive observed MI
    fe = analytic_batch_noise_gate(disc, observed, classes_y, n, min_nonzero_confidence=0.5)
    assert fe[0] == observed[0], "sparse-table candidate wrongly zeroed -- the gate must not reject where the chi-square is invalid"


def test_dense_table_still_rejects_pure_noise():
    """Where the asymptotic IS valid (large n past the min-n floor, low cardinality), a ~zero-MI
    candidate is still rejected."""
    n = 60000
    rng = np.random.default_rng(1)
    disc = rng.integers(0, 4, size=(n, 1)).astype(np.int64)
    classes_y = rng.integers(0, 2, size=n).astype(np.int64)
    bx = int(np.unique(disc[:, 0]).size)
    assert analytic_null_applicable(n, bx, 2)
    observed = np.array([1e-9])
    fe = analytic_batch_noise_gate(disc, observed, classes_y, n, min_nonzero_confidence=0.5)
    assert fe[0] == 0.0, "near-zero-MI candidate on a valid dense table should be rejected"
