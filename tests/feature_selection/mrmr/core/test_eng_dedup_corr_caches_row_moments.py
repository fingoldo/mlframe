"""The dedup correlation kernel reads each buffer row once per comparison, not three times.

The buffer is append-only and its rows never move, so a row's mean and centred sum-of-squares are constants of that row. The kernel
recomputed both on every candidate the row was ever compared against, which over an O(K^2) dedup scan is O(K^2 * n) redundant reduction
passes, the same order as the ``np.corrcoef`` calls the kernel was written to replace. The candidate's own centring was also redone inside
the per-row loop, once per row.

The correlations themselves must not move, so they are pinned against ``np.corrcoef``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_fit_impl._eng_dedup_batch_corr import (
    one_vs_many_abs_corr_masked,
    row_mean_and_centred_ss,
)


@pytest.fixture
def buffer():
    """A candidate and a buffer of rows, some correlated with it, some not, plus a near-constant one."""
    rng = np.random.default_rng(0)
    n, k = 500, 7
    a = rng.normal(size=n)
    mat = rng.normal(size=(k, n))
    mat[0] = a * 2.0 + 1.0  # perfectly correlated
    mat[1] = -a  # perfectly anti-correlated, |r| must still be 1
    mat[2] = np.full(n, 3.0)  # no variance at all
    return a, mat


def _reference(a, mat, active):
    """|Pearson| of ``a`` against each active row, straight from numpy."""
    out = np.zeros(mat.shape[0])
    for j in range(mat.shape[0]):
        if not active[j]:
            continue
        if np.ptp(mat[j]) == 0.0:
            continue
        out[j] = abs(np.corrcoef(a, mat[j])[0, 1])
    return out


def test_the_correlations_match_numpy(buffer):
    """Whatever the kernel caches, it must still compute Pearson."""
    a, mat = buffer
    active = np.ones(mat.shape[0], dtype=np.bool_)
    got = one_vs_many_abs_corr_masked(a, mat, active)
    assert np.allclose(got, _reference(a, mat, active), atol=1e-12), f"{got} vs {_reference(a, mat, active)}"
    assert got[0] == pytest.approx(1.0, abs=1e-12) and got[1] == pytest.approx(1.0, abs=1e-12)
    assert got[2] == 0.0, "a row with no variance must read as uncorrelated, not as noise"


def test_cached_moments_give_the_same_answer_as_recomputing_them(buffer):
    """The cached path and the compute-it-here path are the same arithmetic, so they must agree exactly."""
    a, mat = buffer
    active = np.ones(mat.shape[0], dtype=np.bool_)
    means = np.empty(mat.shape[0])
    sss = np.empty(mat.shape[0])
    for j in range(mat.shape[0]):
        means[j], sss[j] = row_mean_and_centred_ss(mat[j])
    cached = one_vs_many_abs_corr_masked(a, mat, active, means, sss)
    recomputed = one_vs_many_abs_corr_masked(a, mat, active)
    assert np.array_equal(cached, recomputed), f"cached {cached} vs recomputed {recomputed}"


def test_inactive_rows_stay_zero_and_are_not_read(buffer):
    """The mask is what lets the caller pass its whole history buffer, so an inactive row must contribute nothing."""
    a, mat = buffer
    active = np.ones(mat.shape[0], dtype=np.bool_)
    active[0] = False
    got = one_vs_many_abs_corr_masked(a, mat, active)
    assert got[0] == 0.0, "a masked-out row was still scored"
    assert got[1] == pytest.approx(1.0, abs=1e-12), "masking one row changed another row's answer"


def test_a_candidate_with_no_variance_correlates_with_nothing():
    """The guard is on the candidate too: a constant candidate has no correlation to report."""
    rng = np.random.default_rng(1)
    mat = rng.normal(size=(4, 300))
    got = one_vs_many_abs_corr_masked(np.full(300, 2.0), mat, np.ones(4, dtype=np.bool_))
    assert np.array_equal(got, np.zeros(4))


def test_row_moments_match_numpy(buffer):
    """The cached moments are the row's own mean and centred sum-of-squares."""
    _a, mat = buffer
    for j in range(mat.shape[0]):
        mean, ss = row_mean_and_centred_ss(mat[j])
        assert mean == pytest.approx(float(mat[j].mean()), abs=1e-12)
        assert ss == pytest.approx(float(((mat[j] - mat[j].mean()) ** 2).sum()), rel=1e-12)


def test_an_empty_buffer_is_not_scored():
    """Nothing to compare against means an empty result, without invoking the kernel."""
    assert one_vs_many_abs_corr_masked(np.arange(10.0), np.empty((0, 10)), np.zeros(0, dtype=np.bool_)).shape == (0,)
