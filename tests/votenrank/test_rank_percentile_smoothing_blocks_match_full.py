"""The kernel-smoothed percentile is the same whatever block size builds its (query, reference) matrix.

The matrix used to be built whole, n_query x n_oof floats. It is now built in query blocks; a tiny block budget must give the same
percentiles as one block.
"""

from __future__ import annotations

import numpy as np

from mlframe.votenrank import rank_percentile_stacking as rps


def test_smoothed_percentiles_do_not_depend_on_the_block_size(monkeypatch):
    """A budget of a few cells forces many query blocks and returns exactly the values one block returns."""
    rng = np.random.default_rng(5)
    oof = rng.normal(size=60)
    query = rng.normal(size=500)

    monkeypatch.setattr(rps, "_SMOOTH_BLOCK_BYTES", 1 << 30)
    whole = rps.rank_percentile_transform(oof, query, smoothing=0.35)[1]

    monkeypatch.setattr(rps, "_SMOOTH_BLOCK_BYTES", 3 * 60 * 8)
    blocked = rps.rank_percentile_transform(oof, query, smoothing=0.35)[1]

    np.testing.assert_array_equal(blocked, whole)
