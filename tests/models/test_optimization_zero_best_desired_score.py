"""A best_desired_score of 0.0 is a legitimate target (e.g. an R2 or a loss of 0) and must stop the search once reached."""

from __future__ import annotations

import numpy as np

from mlframe.models import optimization as opt_mod


def _run(best_desired_score):
    """Run the one-dimensional search with a constant-0 objective and return how many evaluations it made."""
    calls = {"n": 0}

    def evalfn(x):
        """Count the call and return a constant score of 0.0."""
        calls["n"] += 1
        return 0.0

    opt_mod.optimize_finite_onedimensional_search_space(
        eval_candidate_func=evalfn,
        search_space=np.arange(0, 30),
        direction=opt_mod.OptimizationDirection.Maximize,
        init_num_samples=3,
        max_fevals=20,
        best_desired_score=best_desired_score,
        model_name="ETR",
        model_params={},
        random_state=0,
        verbose=0,
    )
    return calls["n"]


def test_zero_best_desired_score_stops_search_early():
    """A 0.0 target reached by the objective stops the search before the unbounded run does."""
    unbounded = _run(None)
    with_zero_target = _run(0.0)
    assert with_zero_target < unbounded, (with_zero_target, unbounded)
