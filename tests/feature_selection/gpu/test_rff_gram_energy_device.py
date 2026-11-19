"""The device RFF Gram energies match the host kernel to float32 rounding and rank the pairs the same way."""

from itertools import combinations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytest.importorskip("sklearn")

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge

from mlframe.feature_selection.filters import _gradient_interaction_seeder as seeder
from mlframe.feature_selection.filters import _rff_gram_energy_gpu as dev


def _case(p=30, n=1500, seed=0):
    """A fitted RFF + ridge surrogate of a target with one interaction, its standardised matrix and all column pairs."""
    r = np.random.default_rng(seed)
    X = r.standard_normal((n, p)).astype(np.float32)
    y = X[:, 0] * X[:, 1] + 0.3 * X[:, 2] + 0.2 * r.standard_normal(n)
    rbf = RBFSampler(gamma=1.0 / p, n_components=300, random_state=0).fit(X)
    ridge = Ridge(alpha=1.0).fit(rbf.transform(X), y)
    return rbf, ridge, X, list(combinations(range(p), 2))


def test_device_energies_equal_host_and_rank_alike(monkeypatch):
    """Energies agree to a relative 1e-4 and the ten strongest pairs are the same set."""
    rbf, ridge, X, pairs = _case()
    monkeypatch.setenv("MLFRAME_RFF_GRAM_DEVICE", "0")
    host = seeder._rff_analytic_mixed_partial_energy_gram(rbf, ridge, X, pairs)
    monkeypatch.setenv("MLFRAME_RFF_GRAM_DEVICE", "1")
    got = dev.rff_gram_energies_device(rbf, ridge, X, pairs)
    assert got is not None and set(got) == set(host)
    h = np.array([host[q] for q in pairs])
    d = np.array([got[q] for q in pairs])
    np.testing.assert_allclose(d, h, rtol=1e-4, atol=1e-12)
    assert set(np.argsort(-h)[:10]) == set(np.argsort(-d)[:10])


def test_forced_host_and_small_work_return_none(monkeypatch):
    """The switch ``0`` and a pair list below the floor leave the work to the host kernel; no pairs gives an empty result through the host."""
    rbf, ridge, X, pairs = _case(p=8)
    monkeypatch.setenv("MLFRAME_RFF_GRAM_DEVICE", "0")
    assert dev.rff_gram_energies_device(rbf, ridge, X, pairs) is None
    monkeypatch.delenv("MLFRAME_RFF_GRAM_DEVICE")
    assert dev.rff_gram_energies_device(rbf, ridge, X, pairs[:2]) is None  # 2 pairs x 300 features is below the floor
    assert seeder._rff_analytic_mixed_partial_energy_gram(rbf, ridge, X, []) == {}
