"""The batched modulus permutation null must return exactly what ``escalate_modulus`` returns for each permuted target."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _pairwise_modular_fe as pm
from mlframe.feature_selection.filters._modulus_perm_null import perm_best_mis
from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi


def _hit_and_data(seed: int):
    """Build a seeded frame with a modular target and return it with its coarse modular hit and the RNG used for permutations."""
    rng = np.random.default_rng(seed)
    n = 1500
    a = rng.integers(0, 40, n)
    b = rng.integers(0, 25, n)
    y = ((a + b) % 7).astype(int) if seed % 2 == 0 else rng.integers(0, 4, n)
    X = pd.DataFrame({"a": a, "b": b})
    hits = pm.cheap_modular_scan(X, y, moduli=pm.COARSE_MODULI)
    return X, y, hits[0], rng


def test_perm_best_mis_equals_escalate_modulus_per_permutation():
    """Each permuted target's batched best MI must be exactly the value escalate_modulus returns for it."""
    for seed in (0, 1, 2):
        X, y, hit, rng = _hit_and_data(seed)
        perms = [rng.permutation(y) for _ in range(6)]
        ref = np.array([pm.escalate_modulus(X, p, hit, nbins=12)[1] for p in perms])
        yi = np.stack([encode_y_for_classif_mi(p) for p in perms])
        got = perm_best_mis(hit, X, yi, 12)
        assert np.array_equal(ref, got), f"seed {seed}: batched null diverged from escalate_modulus"


def test_modulus_grid_matches_escalate_search_window():
    """The grid around modulus 12 is ascending and covers the window, the small multiples and the divisors."""
    grid = pm.modulus_grid(12)
    assert grid == sorted(grid)
    assert {10, 12, 18, 24, 36, 2, 3, 4, 6} <= set(grid)
