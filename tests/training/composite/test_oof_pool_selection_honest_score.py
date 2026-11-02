"""select_oof_pool_ensemble can report a score on rows the selection never saw (tracker 2026-11-02 wave 15, LEAK15-4)."""

from types import SimpleNamespace

import numpy as np

from mlframe.training.composite.hpo_ensembling import select_oof_pool_ensemble


def _noise_pool(n_trials=60, n=300, seed=0):
    """A pool of pure-noise 'predictions' around a target they carry no information about."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    return SimpleNamespace(trial_oof_pool=[rng.normal(size=n) for _ in range(n_trials)]), y


def test_honest_score_is_worse_than_the_in_sample_score_on_a_noise_pool():
    """Greedy selection over 60 noise columns fits chance on its own rows; scored on held-out rows the RMSE is higher."""
    result, y = _noise_pool()
    out = select_oof_pool_ensemble(result, y, honest_folds=5)
    assert out.honest_score is not None
    assert out.honest_score > out.score + 0.02


def test_honest_score_is_absent_unless_requested():
    """The default result carries no nested estimate."""
    result, y = _noise_pool(n_trials=10)
    assert select_oof_pool_ensemble(result, y).honest_score is None
