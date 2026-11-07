"""The per-pair family-wise level is Bonferroni-divided by the number of prospective pairs.

Without the division every pair runs its family test at the full level, so across a pool of noise pairs the chance that at least
one passes grows with the pool size (audit 2026-10-14, F01). The division keeps the family-wise error near the pool-wide level.
"""

import numpy as np

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core_parts import _check_prospective_fe_pairs_part5


def _alpha_for_pool(n_pairs: int) -> float:
    """Family alpha the pair scan would use for a pool of ``n_pairs`` prospective pairs."""
    prospective_pairs = {((f"a{i}", f"b{i}"), 0): None for i in range(n_pairs)}
    out = _check_prospective_fe_pairs_part5({}, 10, np.array([0, 1, 0, 1]), np.array([0.5, 0.5]), prospective_pairs)
    return out[-1]


def test_pair_family_alpha_is_bonferroni_over_pool():
    """A 28-pair pool runs each family test at 0.001 / 28."""
    assert np.isclose(_alpha_for_pool(28), 0.001 / 28)


def test_pair_family_alpha_shrinks_with_pool_size():
    """A larger pool gets a strictly smaller per-pair alpha."""
    assert _alpha_for_pool(100) < _alpha_for_pool(10)
