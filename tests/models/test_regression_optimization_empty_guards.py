"""Regression tests for MBHOptimizer empty-input guards and dead-loop removal.

Bug 3a: ``compute_candidates_exploration_scores`` left ``r``/``l`` unbound when ``known_candidates`` is empty,
        raising ``NameError`` at ``distances[r:]``.
Bug 3b: ``suggest_candidate`` indexed ``self.known_evaluations[0]`` without guarding the empty case (IndexError).
Bug 4 : the ``for best_idx in np.argsort(...)`` greedy scan was wrapped in a dead ``for _ in range(100)`` with no
        state change between iterations; removing the wrapper leaves selection identical.
"""

import numpy as np

from mlframe.models.optimization import (
    MBHOptimizer,
    NOT_READY,
    compute_candidates_exploration_scores,
)


def _make_optimizer(**overrides):
    """Helper: Make optimizer."""
    kwargs = dict(
        search_space=np.arange(0, 20),
        model_name="ETR",
        model_params={},
        init_num_samples=5,
        random_state=0,
    )
    kwargs.update(overrides)
    return MBHOptimizer(**kwargs)


def test_exploration_scores_empty_known_candidates_returns_zeros():
    """Exploration scores empty known candidates returns zeros."""
    search_space = np.arange(10)
    # Pre-fix this raised NameError (r/l unbound because the loop never ran).
    distances = compute_candidates_exploration_scores(search_space=search_space, known_candidates=[])
    assert distances.shape == (10,)
    assert np.all(distances == 0.0)


def test_exploration_scores_nonempty_unchanged():
    """Exploration scores nonempty unchanged."""
    search_space = np.arange(10)
    distances = compute_candidates_exploration_scores(search_space=search_space, known_candidates=[5])
    np.testing.assert_array_equal(distances, np.abs(search_space - 5))


def test_suggest_candidate_empty_known_evaluations_returns_not_ready():
    # Inconsistent state (candidates present, evaluations empty) must not IndexError on known_evaluations[0].
    # Post-API23 the transient "surrogate not yet trainable" case returns the NOT_READY sentinel (NOT None),
    # so the driving loop does not mistake it for search-space exhaustion and truncate the search.
    """Suggest candidate empty known evaluations returns not ready."""
    opt = _make_optimizer(greedy_prob=0.0)
    opt.pre_seeded_candidates = []  # skip the pre-seed shortcut so the surrogate-fit branch is reached
    opt.known_candidates = np.array([1, 2, 3])
    opt.known_evaluations = np.array([])
    opt.last_retrain_ninputs = 0
    opt.best_candidate = 1
    assert opt.suggest_candidate() is NOT_READY


def test_suggest_candidate_greedy_picks_nearest_unchecked():
    # Greedy path (greedy_prob forced to 1.0) must return the closest unchecked candidate to the best known one.
    """Suggest candidate greedy picks nearest unchecked."""
    opt = _make_optimizer(greedy_prob=1.0)
    opt.pre_seeded_candidates = []  # skip the pre-seed shortcut so the greedy branch is reached
    opt.known_candidates = np.array([10])
    opt.best_candidate = 10
    opt.best_evaluation = 1.0
    opt.n_noimproving_iters = 1
    opt.suggested_candidates.clear()
    nxt = opt.suggest_candidate()
    # Nearest unchecked to 10 (distance 1): 9 or 11. The argsort scan returns one of them, never a far point.
    assert nxt is not None
    assert abs(int(nxt) - 10) == 1
    assert int(nxt) not in opt.known_candidates
