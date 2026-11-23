"""A block correlated with itself is row-centred once (PLAN-10: two identical full copies of the block were held at 500000 rows) and the correlations are unchanged."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_dedup as D


def test_self_block_is_centred_once_and_the_result_is_unchanged(monkeypatch):
    """One centring call for ``(Q, Q)``, two for ``(Q, copy of Q)``; the same values either way."""
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((12, 500)).astype(np.float32)
    Q[3] = Q[1] + 0.01 * rng.standard_normal(500)
    Q[2, :40] = np.nan
    calls = []
    real = D._row_center_finite

    def spy(a):
        """Count the centring calls."""
        calls.append(1)
        return real(a)

    monkeypatch.setattr(D, "_row_center_finite", spy)
    same = D._pairwise_complete_abs_corr(Q, Q)
    assert len(calls) == 1
    calls.clear()
    other = D._pairwise_complete_abs_corr(Q, Q.copy())
    assert len(calls) == 2
    np.testing.assert_array_equal(np.nan_to_num(same, nan=-1.0), np.nan_to_num(other, nan=-1.0))
    assert same[1, 3] > 0.99
