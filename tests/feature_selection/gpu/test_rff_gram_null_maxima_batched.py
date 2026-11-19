"""The batched null maxima of the RFF Gram energies equal the maxima of the single-surrogate device route for every surrogate of the stack."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from itertools import combinations

from mlframe.feature_selection.filters import _gradient_interaction_seeder as seeder
from mlframe.feature_selection.filters import _rff_gram_energy_gpu as rg


def test_batched_null_maxima_equal_single_route():
    """Several permutation surrogates scored together give the per-surrogate maximum energy, and a pair-less or surrogate-less call gives None."""
    rng = np.random.default_rng(0)
    n, p = 600, 14
    Xs = rng.standard_normal((n, p)).astype(np.float32)
    y = (Xs[:, 0] * Xs[:, 1] + 0.3 * rng.standard_normal(n)).astype(np.float32)
    pairs = list(combinations(range(p), 2))
    fits = [seeder._fit_rff_ridge(Xs, np.random.default_rng(1000 + s).permutation(y), 120, 1.0 / p, 1.0, s) for s in range(5)]
    batched = rg.rff_gram_null_maxima_device([f[0] for f in fits], [f[1] for f in fits], Xs, pairs, chunk=40)
    assert batched is not None and batched.shape == (5,)
    single = [max(rg.rff_gram_energies_device(f[0], f[1], Xs, pairs).values()) for f in fits]
    np.testing.assert_allclose(batched, single, rtol=2e-5)
    assert rg.rff_gram_null_maxima_device([], [], Xs, pairs) is None
    assert rg.rff_gram_null_maxima_device([fits[0][0]], [fits[0][1]], Xs, []) is None
