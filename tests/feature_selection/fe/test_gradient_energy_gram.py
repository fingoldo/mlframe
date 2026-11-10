"""The Gram-matrix mixed-partial energy equals the batched and the per-pair kernels up to float rounding and keeps the pair order."""

from itertools import combinations

import numpy as np

from mlframe.feature_selection.filters import _gradient_interaction_seeder as S


def test_gram_energy_matches_batched_and_loop_kernels():
    """Same energies (relative 1e-4) and the same top pairs on a surrogate fitted to an interaction target."""
    r = np.random.default_rng(3)
    n, p = 1500, 14
    Xs = r.standard_normal((n, p)).astype(np.float32)
    y = (Xs[:, 0] * Xs[:, 1] + 0.3 * Xs[:, 2] + 0.1 * r.standard_normal(n)).astype(np.float32)
    rbf, ridge = S._fit_rff_ridge(Xs, y, 200, 1.0 / p, 1.0, 0)
    pairs = list(combinations(range(p), 2))
    gram = S._rff_analytic_mixed_partial_energy_gram(rbf, ridge, Xs, pairs, chunk=40)
    batched = S._rff_analytic_mixed_partial_energy(rbf, ridge, Xs, pairs)
    loop = S._rff_analytic_mixed_partial_energy_loop(rbf, ridge, Xs, pairs)
    g, b, l = (np.array([d[k] for k in pairs]) for d in (gram, batched, loop))
    np.testing.assert_allclose(g, b, rtol=1e-4, atol=1e-12)
    np.testing.assert_allclose(g, l, rtol=1e-3, atol=1e-12)
    assert list(np.argsort(-g)[:5]) == list(np.argsort(-b)[:5])
