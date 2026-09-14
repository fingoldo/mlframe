"""The escalation S5 gate must compare marginal MIs estimated on one row set (mrmr_audit_2026-09-14 FE_STEP-3).

With ``fe_check_pairs_subsample_n`` active, FE auto-escalation decides on a row subsample. It sliced the already-admitted support's VALUES
to that subsample but kept each column's marginal MI from the FULL frame, then put those next to survivor MIs estimated on the subsample
in one redundancy-gate pool. Plug-in MI is biased upward by roughly (k_x - 1)(k_y - 1) / 2n, so the subsample estimates run higher: the
admitted support looked weaker than it is relative to the candidates, and the gate's bar (a fraction of the weakest admitted MI) was set
from the wrong scale, in the direction that admits more candidates.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._fe_auto_escalation import _slice_admitted_pool
from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin


def _pool_and_slice(seed=0, n=4000, m=600, nbins=10):
    """Two admitted columns carrying a deliberately wrong full-frame marginal, a sorted row subsample, and the target."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    a = y + rng.normal(scale=1.0, size=n)
    b = rng.normal(size=n)
    pool = {"eng_a": (a, 999.0), "eng_b": (b, 999.0)}
    idx = np.sort(rng.choice(n, size=m, replace=False))
    return pool, idx, y, nbins


def test_admitted_marginals_are_rescored_on_the_subsample():
    """Every admitted column's marginal must be the gate estimator applied to its sliced values and the sliced target."""
    pool, idx, y, nbins = _pool_and_slice()
    y_sub = y[idx]
    out = _slice_admitted_pool(pool, idx, y_sub, nbins)
    _, y_dense = np.unique(y_sub.astype(np.int64), return_inverse=True)
    for name, (full_values, _) in pool.items():
        vals, marg = out[name]
        np.testing.assert_array_equal(vals, np.asarray(full_values)[idx])
        expected = float(_cmi_from_binned(_quantile_bin(np.asarray(vals, dtype=np.float64), nbins=nbins), y_dense.astype(np.int64), None))
        assert marg == expected, f"{name}: marginal {marg} was carried over instead of re-estimated on the subsample ({expected})"


def test_rescored_marginal_keeps_the_signal_ordering():
    """Control: the informative admitted column still scores above the noise column after re-estimation."""
    pool, idx, y, nbins = _pool_and_slice()
    out = _slice_admitted_pool(pool, idx, y[idx], nbins)
    assert out["eng_a"][1] > out["eng_b"][1]
