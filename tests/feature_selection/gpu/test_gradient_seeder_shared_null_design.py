"""The gradient seeder's permutation null on ONE shared random-feature design: the multi-target ridge, the device and host maxima, and the verdicts."""

from itertools import combinations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _gradient_interaction_seeder as seeder
from mlframe.feature_selection.filters._fe_probe_ridge import fit_ridge_closed_form, fit_ridge_closed_form_multi
from mlframe.feature_selection.filters._linear_probes import PlainRidge, RffSampler


def _surrogate_case(p=12, n=800, seed=0):
    """A standardised matrix, a target with one interaction, the shared design fitted on it and every column pair."""
    r = np.random.default_rng(seed)
    X = r.standard_normal((n, p))
    y = X[:, 0] * X[:, 1] + 0.3 * X[:, 2] + 0.2 * r.standard_normal(n)
    rbf = RffSampler(gamma=1.0 / p, n_components=120, random_state=seed)
    rbf.fit(X)
    return X, y, rbf, list(combinations(range(p), 2))


def test_multi_target_ridge_equals_the_single_target_fit_of_every_column():
    """One factorisation for all targets gives the weights and intercepts of the per-column closed form."""
    r = np.random.default_rng(1)
    X = r.standard_normal((500, 40))
    Y = np.stack([r.standard_normal(500) + X[:, j] for j in range(5)], axis=1)
    weights, intercepts = fit_ridge_closed_form_multi(X, Y, 1.0)
    for j in range(Y.shape[1]):
        w, b = fit_ridge_closed_form(X, Y[:, j], 1.0)
        np.testing.assert_allclose(weights[:, j], w, rtol=1e-10, atol=1e-12)
        assert intercepts[j] == pytest.approx(b, rel=1e-10, abs=1e-12)
    assert fit_ridge_closed_form_multi(X, Y[:, 0], 1.0)[0].shape == (40, 1)


def test_multi_target_ridge_reports_a_non_finite_system_as_none():
    """A NaN in the design is not solved."""
    X = np.random.default_rng(2).standard_normal((50, 20))
    X[3, 4] = np.nan
    assert fit_ridge_closed_form_multi(X, np.ones((50, 2)), 1.0) is None


@pytest.mark.parametrize("device", ["0", "1"])
def test_shared_design_maxima_equal_the_per_surrogate_energies(monkeypatch, device):
    """The null maxima of ``k`` permutation surrogates on one design are the largest pair energy of each surrogate scored alone (host, and the device twin when a GPU is there)."""
    if device == "1":
        pytest.importorskip("cupy")
    monkeypatch.setenv("MLFRAME_RFF_GRAM_DEVICE", device)
    X, y, rbf, pairs = _surrogate_case()
    k_perm, seed, alpha = 6, 3, 1.0
    got = seeder._shared_design_null_maxima(rbf, X, y, pairs, k_perm, alpha, seed, chunk=40)
    assert got is not None and got.shape == (k_perm,)
    monkeypatch.setenv("MLFRAME_RFF_GRAM_DEVICE", "0")
    for s in range(k_perm):
        permuted = np.random.default_rng(seed + 1000 + s).permutation(y)
        ridge = PlainRidge(alpha=alpha).fit(rbf.transform(X), permuted)
        alone = seeder._rff_analytic_mixed_partial_energy_gram(rbf, ridge, X, pairs)
        assert got[s] == pytest.approx(max(alone.values()), rel=1e-3 if device == "1" else 1e-9)


def _data(kind, seed, n=2000, p=20):
    """An additive target, or the same plus an ``x2 * x3`` interaction."""
    r = np.random.default_rng(seed)
    X = r.standard_normal((n, p))
    y = X[:, 0] + 0.7 * np.sin(X[:, 1]) + 0.3 * r.standard_normal(n)
    return X, (y + 0.8 * X[:, 2] * X[:, 3] if kind == "saddle" else y)


def test_shared_design_finds_the_interaction_and_stays_quiet_on_an_additive_target():
    """The interaction is proposed in every seed tried and an additive target proposes nothing, with the shared design (the default) as with one design per surrogate."""
    for seed in range(6):
        X, y = _data("saddle", seed)
        for shared in (True, False):
            proposed, _, diag = seeder.rank_gradient_interaction_pairs(X, y, list(range(X.shape[1])), seed=seed, shared_null_design=shared)
            assert (2, 3) in proposed, (seed, shared)
            assert diag["learned"]
        Xa, ya = _data("additive", seed)
        proposed, _, _ = seeder.rank_gradient_interaction_pairs(Xa, ya, list(range(Xa.shape[1])), seed=seed)
        assert proposed == [], (seed, proposed)


def test_the_diagnostics_name_the_null_design():
    """The default null is on the shared design, and ``shared_null_design=False`` keeps the per-surrogate designs (no ``null_design`` entry)."""
    X, y = _data("saddle", 0)
    _, _, diag = seeder.rank_gradient_interaction_pairs(X, y, list(range(X.shape[1])), seed=0)
    assert diag["null_design"] == "shared"
    _, _, old = seeder.rank_gradient_interaction_pairs(X, y, list(range(X.shape[1])), seed=0, shared_null_design=False)
    assert "null_design" not in old
