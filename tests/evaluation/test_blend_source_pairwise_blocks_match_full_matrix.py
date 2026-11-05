"""The rank-agreement count is the same whatever row-block size builds the pairwise comparison matrices.

The matrices used to be built whole, which is n^2 booleans each. They are now built in row blocks; a tiny block budget must give the same
agreement as a single block.
"""

from __future__ import annotations

import numpy as np

from mlframe.evaluation import blend_source_selection as bss


def test_rank_agreement_is_independent_of_the_row_block_size(monkeypatch):
    """A block budget of a few cells, forcing many blocks, returns exactly the value one block returns."""
    rng = np.random.default_rng(3)
    n = 37
    a = rng.normal(size=n)
    b = a + rng.normal(scale=0.8, size=n)

    monkeypatch.setattr(bss, "_PAIRWISE_BLOCK_ELEMENTS", 10**9)
    whole = bss.check_pairwise_score_correlation(a, b)["rank_agreement"]

    monkeypatch.setattr(bss, "_PAIRWISE_BLOCK_ELEMENTS", 3 * n)
    blocked = bss.check_pairwise_score_correlation(a, b)["rank_agreement"]

    assert blocked == whole
    assert 0.0 <= whole <= 1.0
