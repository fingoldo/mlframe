"""Audit 2026-11-20 finding 34-F10: the upgraded multi-base spec carries the multi-set CV score, not the single-base seed's.

The forward stepwise pass measures the final kept set; its ``rmse_after`` is what the upgraded composite should be ranked by.
With no accepted base the kept set is the seed, so the seed's score is still correct.
"""

from types import SimpleNamespace

import numpy as np

from mlframe.training.composite.discovery._fit_multibase import _upgrade_spec

_NEW = "linear_residual_multi"


def _run(diag):
    """Upgrade a seed named 'seed' with bases a and b; return the score stored under the new name."""
    rng = np.random.default_rng(0)
    n = 200
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = 2 * a + b + 0.1 * rng.normal(size=n)
    self_ = SimpleNamespace(_tiny_rerank_scores={"seed": 1.0})
    spec = SimpleNamespace(name="seed")
    pool = {"a": a, "b": b}
    upgraded = _upgrade_spec(self_, spec, ["a", "b"], diag, pool, y, "y")
    return self_._tiny_rerank_scores[upgraded.name]


def test_accepted_step_score_replaces_the_seed_score():
    """An accepted step with rmse_after=0.42 makes the upgraded spec score 0.42, not the seed's 1.0."""
    diag = [{"accepted": True, "candidate_added": "b", "marginal_gain": 0.1, "rmse_after": 0.42}]
    assert _run(diag) == 0.42


def test_no_accepted_step_keeps_the_seed_score():
    """With no accepted step the kept set is the seed, so the seed's score stays."""
    assert _run([{"accepted": False, "candidate_added": "b", "marginal_gain": 0.0, "rmse_after": 9.9}]) == 1.0
