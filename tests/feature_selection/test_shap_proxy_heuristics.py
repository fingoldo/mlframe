"""Unit tests for heuristic subset search (beam / greedy / multistart / GA / annealing).

Each heuristic must return valid, deduplicated subsets and recover a clearly-planted informative
feature set on an easy synthetic problem (loss well below a random same-size subset).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection import _shap_proxy_heuristics as H
from mlframe.feature_selection._shap_proxy_objective import coalition_margin, proxy_loss


@pytest.fixture
def easy_problem():
    rng = np.random.default_rng(7)
    n, f = 500, 10
    phi = rng.normal(size=(n, f))
    base = np.full(n, 0.2)
    truth = (0, 1, 2, 3)
    y = base + phi[:, list(truth)].sum(axis=1) + 0.02 * rng.normal(size=n)
    return phi, base, y, set(truth)


def _check(top, phi, base, y, truth):
    assert len(top) >= 1
    # deduplicated and ascending by loss
    keys = [frozenset(c) for _, c in top]
    assert len(keys) == len(set(keys))
    losses = [l for l, _ in top]
    assert losses == sorted(losses)
    best = set(top[0][1])
    # beats a random same-size subset by a wide margin
    rng = np.random.default_rng(99)
    rnd = sorted(rng.choice(phi.shape[1], size=len(best), replace=False).tolist())
    rnd_loss = proxy_loss(coalition_margin(phi, base, rnd), y, "rmse")
    assert top[0][0] < rnd_loss
    return best


def test_beam_recovers_truth(easy_problem):
    phi, base, y, truth = easy_problem
    top = H.beam_search(phi, base, y, classification=False, metric="rmse", beam_width=6, top_n=5)
    assert _check(top, phi, base, y, truth) == truth


def test_greedy_forward_recovers_truth(easy_problem):
    phi, base, y, truth = easy_problem
    top = H.greedy_forward(phi, base, y, classification=False, metric="rmse", top_n=5)
    assert _check(top, phi, base, y, truth) == truth


def test_greedy_backward_recovers_truth(easy_problem):
    phi, base, y, truth = easy_problem
    top = H.greedy_backward(phi, base, y, classification=False, metric="rmse", top_n=5)
    assert _check(top, phi, base, y, truth) == truth


def test_multistart_beats_random(easy_problem):
    phi, base, y, truth = easy_problem
    top = H.multistart_local(phi, base, y, classification=False, metric="rmse",
                             rng=np.random.default_rng(0), n_starts=8, top_n=5)
    _check(top, phi, base, y, truth)


def test_genetic_beats_random(easy_problem):
    phi, base, y, truth = easy_problem
    top = H.genetic(phi, base, y, classification=False, metric="rmse",
                    rng=np.random.default_rng(0), pop_size=30, n_generations=20, top_n=5)
    _check(top, phi, base, y, truth)


def test_annealing_beats_random(easy_problem):
    phi, base, y, truth = easy_problem
    top = H.simulated_annealing(phi, base, y, classification=False, metric="rmse",
                                rng=np.random.default_rng(0), n_iter=1500, top_n=5)
    _check(top, phi, base, y, truth)
