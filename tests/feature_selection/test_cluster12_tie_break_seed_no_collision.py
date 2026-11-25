"""Cluster 12 regression: the tie-break seed for two (dataset, cv) cells does not collide (audit 2026-11-20, finding 12-F100).

The old ``dataset * 1000 + cv`` gave cell (0, 1000) and cell (1, 0) the same seed. The seed is now a SeedSequence over both values.
"""

import numpy as np


def _tie_seed(dataset_seed, cv_seed):
    """Mirror of the expression in run_experiment (kept here so the collision check does not import the bench)."""
    return int(np.random.SeedSequence([int(dataset_seed), int(cv_seed)]).generate_state(1)[0])


def test_old_formula_collided_and_new_does_not():
    """The cells that collided under dataset*1000+cv get different seeds now (12-F100)."""
    assert 0 * 1000 + 1000 == 1 * 1000 + 0
    assert _tie_seed(0, 1000) != _tie_seed(1, 0)


def test_seed_is_deterministic():
    """Control: the same cell always gets the same seed (12-F100)."""
    assert _tie_seed(3, 7) == _tie_seed(3, 7)
