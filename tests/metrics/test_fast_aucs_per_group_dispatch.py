"""Regression: public ``fast_aucs_per_group`` routes to the presort twin and stays
AUC-identical on valid groups.

CPX-P0-1: the public entry used to do O(n*G) ``group_ids == g`` masking + a per-group
argsort. It now dispatches to ``fast_aucs_per_group_optimized`` (presort + numba
boundary-walk). For every VALID group (>=2 samples, both classes present) the
(roc_auc, pr_auc) must be bit-identical between the two paths. Degenerate groups are
allowed to differ (the dispatch corrects the prior silent ``(0.0, 0.0)`` to NaN).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics._auc_per_group import (
    fast_aucs_per_group,
    fast_aucs_per_group_optimized,
)


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("n,n_groups", [(2_000, 20), (5_000, 500)])
def test_public_matches_optimized_on_valid_groups(seed, n, n_groups):
    """Public matches optimized on valid groups."""
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < 0.4).astype(np.float64)
    y_score = rng.random(n)
    group_ids = rng.integers(0, n_groups, size=n).astype(np.int64)

    roc_p, pr_p, ga_p = fast_aucs_per_group(y_true, y_score, group_ids)
    roc_o, pr_o, ga_o = fast_aucs_per_group_optimized(y_true, y_score, group_ids)

    assert roc_p == roc_o
    assert pr_p == pr_o
    assert ga_p.keys() == ga_o.keys()

    n_valid = 0
    for gid, (r_o, pr_oo) in ga_o.items():
        r_p, pr_pp = ga_p[gid]
        if np.isnan(r_o):
            assert np.isnan(r_p)  # degenerate: both NaN now
            continue
        n_valid += 1
        assert r_p == r_o
        assert pr_pp == pr_oo
    assert n_valid > 0, "fixture must contain at least one valid (2-class, >=2 sample) group"


def test_single_sample_group_is_nan_not_zero():
    """Degenerate (single-sample) groups emit NaN, not the prior silent (0.0, 0.0)."""
    y_true = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    y_score = np.array([0.9, 0.1, 0.5], dtype=np.float64)
    group_ids = np.array([10, 10, 99], dtype=np.int64)  # group 99 is single-sample

    _, _, ga = fast_aucs_per_group(y_true, y_score, group_ids)
    assert np.isnan(ga[99][0]) and np.isnan(ga[99][1])
