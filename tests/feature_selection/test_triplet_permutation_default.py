"""The cross-basis triplet test keeps its permutation count at 199, enough to resolve the 0.99 null quantile (tracker 2026-10-13, D10).

Twenty permutations cannot resolve a 0.99 quantile at all; the false-winner probability of about 0.04 at 199 was derived, and the default is pinned so a later edit cannot quietly lower it.
"""

import inspect

from mlframe.feature_selection.filters._orthogonal_triplet_fe import score_triplet_cross_basis_by_cmi


def test_default_permutation_count_resolves_the_null_quantile():
    """199 permutations at least resolve a 0.99 quantile (1 / (1 - 0.99) = 100), and the quantile default is 0.99."""
    params = inspect.signature(score_triplet_cross_basis_by_cmi).parameters
    assert params["n_perm"].default == 199
    assert params["quantile"].default == 0.99
    assert params["n_perm"].default >= 1.0 / (1.0 - params["quantile"].default)
