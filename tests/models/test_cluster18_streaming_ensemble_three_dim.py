"""Cluster 18 regression: the streaming ensemble accepts (N, K, C) members like the materialised path (audit 2026-11-20, 18-F11)."""

import numpy as np
import pytest

from mlframe.models.ensembling.predict import ensemble_probabilistic_predictions_streaming


def test_three_dimensional_members_are_averaged_over_members():
    """Three (N, K, C) members give the (N, K, C) arithmetic mean and an (N,) uncertainty (18-F11)."""
    rng = np.random.default_rng(0)
    members = [rng.uniform(0.1, 0.9, size=(20, 3, 2)) for _ in range(3)]
    ens, unc, conf = ensemble_probabilistic_predictions_streaming(*members, ensemble_method="arithm", verbose=False)
    np.testing.assert_allclose(ens, np.mean(members, axis=0), atol=1e-12)
    assert unc.shape == (20,) and conf is None


def test_two_dimensional_and_one_dimensional_members_unchanged():
    """(N, K) keeps its shape and (N,) comes back as (N,) (18-F11)."""
    a = np.linspace(0.1, 0.9, 10)
    ens, unc, _ = ensemble_probabilistic_predictions_streaming(a, a[::-1], ensemble_method="arithm", verbose=False)
    assert ens.shape == (10,) and unc.shape == (10,)
    m = np.tile(a[:, None], (1, 4))
    ens2, _, _ = ensemble_probabilistic_predictions_streaming(m, m, ensemble_method="arithm", verbose=False)
    assert ens2.shape == (10, 4)


def test_scalar_members_are_refused():
    """A 0-d member raises a clear ValueError (18-F11)."""
    with pytest.raises(ValueError, match="at least 1-D"):
        ensemble_probabilistic_predictions_streaming(np.float64(0.5), np.float64(0.4), ensemble_method="arithm", verbose=False)
