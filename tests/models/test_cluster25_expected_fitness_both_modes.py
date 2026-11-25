"""Cluster 25 regression: ``expected_fitness`` is set in both search modes (audit 2026-11-20, finding 25-F62).

It used to be assigned only inside the exploitation branch, so the exploration plot showed a stale score. The helper now stores it on the
optimizer after either branch.
"""

from types import SimpleNamespace

import numpy as np

from mlframe.models import _optimization_search as opt

_N = 6


def _owner(mode):
    """A minimal optimizer state for the candidate helper in the given mode."""
    return SimpleNamespace(
        mode=mode,
        direction=opt.OptimizationDirection.Maximize,
        use_stds_for_exploitation=True,
        search_space=list(range(_N)),
        use_distances_on_preds_collision=False,
        additional_info="",
    )


def _arrays():
    """Predictions, uncertainty and distance arrays for the helper."""
    return np.linspace(0.1, 0.6, _N), np.linspace(0.0, 1.0, _N), np.linspace(0.0, 1.0, _N)


def test_exploitation_mode_stores_expected_fitness_on_the_optimizer():
    """In exploitation mode the returned scores are also stored on the optimizer, so the plot reads the current values (25-F62)."""
    owner = _owner("exploitation")
    y_std, distances, y_pred = _arrays()
    returned = opt.MBHOptimizer._suggest_exploration_candidate(owner, y_std, distances, y_pred, set())
    assert owner.expected_fitness is returned


def test_exploration_mode_stores_expected_fitness_on_the_optimizer():
    """Control: exploration mode stores its scores on the optimizer the same way (25-F62)."""
    owner = _owner("exploration")
    y_std, distances, y_pred = _arrays()
    returned = opt.MBHOptimizer._suggest_exploration_candidate(owner, y_std, distances, y_pred, set())
    assert owner.expected_fitness is returned
