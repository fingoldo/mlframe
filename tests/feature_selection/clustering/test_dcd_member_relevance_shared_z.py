"""``batched_member_relevance`` builds the conditioning classes once and equals the per-candidate CMI loop for any support width."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._dynamic_cluster_discovery import _dcd_member_rank_batch as D
from mlframe.feature_selection.filters.info_theory.shared import cpu_cmi_loop_parallel as per_candidate_loop


def _data(dtype, n=4000, f=40, nb=8, seed=0):
    """Random codes with column 5 tied to column 6."""
    rng = np.random.default_rng(seed)
    fd = rng.integers(0, nb, (n, f)).astype(dtype)
    fd[:, 5] = (fd[:, 6] + (rng.random(n) < 0.3) * 3) % nb
    return fd, np.full(f, nb, dtype=np.int64)


@pytest.mark.parametrize("dtype", [np.int8, np.int32])
@pytest.mark.parametrize("kz", [1, 3, 12])
def test_equals_the_per_candidate_loop(dtype, kz):
    """Supports of one, three and twelve columns: the same relevance vector as the loop that melts the support per candidate."""
    fd, nbins = _data(dtype)
    cand = np.arange(20, 28, dtype=np.int64)
    target = np.array([5], dtype=np.int64)
    s = list(range(10, 10 + kz))
    got = D.batched_member_relevance(fd, cand, target, s, nbins)
    want = np.asarray(per_candidate_loop(fd, cand, target, np.array(s, dtype=np.int64), nbins, np.empty(0, dtype=np.int64)))
    assert np.allclose(got, want, rtol=0, atol=1e-12)


def test_overlapping_columns_take_the_old_loop():
    """A candidate that is also in the support keeps the per-candidate path (its joint is not the disjoint union)."""
    fd, nbins = _data(np.int32)
    cand = np.array([10, 21], dtype=np.int64)
    target = np.array([5], dtype=np.int64)
    s = [10, 11]
    got = D.batched_member_relevance(fd, cand, target, s, nbins)
    want = np.asarray(per_candidate_loop(fd, cand, target, np.array(s, dtype=np.int64), nbins, np.empty(0, dtype=np.int64)))
    assert np.array_equal(got, want)
