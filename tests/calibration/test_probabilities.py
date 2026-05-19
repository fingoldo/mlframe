"""Smoke tests for mlframe.calibration.probabilities (E-P1.4)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


@pytest.mark.fast
def test_import_probabilities_module():
    """Module imports cleanly with all public callables present."""
    from mlframe.calibration import probabilities as pm

    for name in (
        "generate_probs_from_outcomes",
        "generate_similar_probs_logit_space",
        "generate_similar_probs_random_walk",
        "generate_similar_probs",
        "generate_similar_probs_by_ranking",
    ):
        assert callable(getattr(pm, name)), f"{name} not callable"


@pytest.mark.fast
def test_generate_probs_from_outcomes_happy_path():
    """generate_probs_from_outcomes returns probs in [0, 1] with same length as input."""
    pytest.importorskip("numba")
    from mlframe.calibration.probabilities import generate_probs_from_outcomes

    rng = np.random.default_rng(42)
    outcomes = (rng.random(200) > 0.5).astype(np.int64)
    probs = generate_probs_from_outcomes(outcomes, chunk_size=20, nbins=5)
    assert probs.shape == (200,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


@pytest.mark.fast
def test_generate_similar_probs_logit_space():
    """Logit-space perturbation returns probs in (0, 1) of matching shape."""
    from mlframe.calibration.probabilities import generate_similar_probs_logit_space

    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=50)
    y = (rng.random(50) > 0.5).astype(int)
    out = generate_similar_probs_logit_space(p, y, noise_scale=0.05)
    assert out.shape == p.shape
    assert (out > 0).all() and (out < 1).all()
