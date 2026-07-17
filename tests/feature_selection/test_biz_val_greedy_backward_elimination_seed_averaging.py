"""biz_value test for the opt-in ``n_repeats`` seed-averaging extension to ``greedy_backward_elimination``.

Gap: with a single CV run per removal decision, the accept/reject delta for a borderline feature is itself
noisy -- across repeated top-level runs (different CV shuffles) the same weak-but-real feature can be kept
in some runs and wrongly dropped in others, purely because one split happened to make its marginal
contribution look negative. Averaging the removal delta across ``n_repeats`` independently-shuffled splits
before deciding should make that "keep the weak signal feature" decision land correctly far more often.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from mlframe.feature_selection.greedy_backward_elimination import greedy_backward_elimination


def _make_borderline_regression(n: int, weak_coef: float, noise_scale: float, seed: int):
    rng = np.random.default_rng(seed)
    strong = rng.normal(size=(n, 2))
    weak = rng.normal(size=(n, 1))
    y = strong[:, 0] * 2.0 + strong[:, 1] * 2.0 + weak[:, 0] * weak_coef + rng.normal(scale=noise_scale, size=n)
    X = pd.DataFrame(np.hstack([strong, weak]), columns=["strong0", "strong1", "weak"])
    return X, y


def test_biz_val_greedy_backward_elimination_seed_averaging_more_consistent_than_single_run():
    # 2 strong features drown out "weak"'s small, real contribution on a tiny 2-fold split, so a SINGLE
    # noisy CV run frequently (mis)judges dropping "weak" as an improvement; averaging over several
    # independently-shuffled splits before deciding should surface its real, if small, contribution far
    # more consistently. Both X/y and every CV seed are fixed, so the counts below are deterministic.
    n_trials = 10
    X, y = _make_borderline_regression(n=30, weak_coef=0.45, noise_scale=2.0, seed=7)

    single_run_keeps = 0
    seed_averaged_keeps = 0
    for trial in range(n_trials):
        cv = KFold(n_splits=2, shuffle=True, random_state=trial)
        survivors_single = greedy_backward_elimination(Ridge(alpha=1.0), X, y, scoring=r2_score, cv=cv, min_features=1)
        if "weak" in survivors_single:
            single_run_keeps += 1

        survivors_averaged = greedy_backward_elimination(Ridge(alpha=1.0), X, y, scoring=r2_score, min_features=1, n_repeats=5, seed_base=trial * 100)
        if "weak" in survivors_averaged:
            seed_averaged_keeps += 1

    single_run_rate = single_run_keeps / n_trials
    seed_averaged_rate = seed_averaged_keeps / n_trials

    assert seed_averaged_rate >= single_run_rate + 0.25, (
        "expected seed-averaging (n_repeats=5) to keep the weak-but-real feature materially more often "
        f"across repeated top-level runs than a single CV run: single_run_rate={single_run_rate:.3f} "
        f"seed_averaged_rate={seed_averaged_rate:.3f}"
    )
    assert seed_averaged_rate >= 0.6, f"expected seed-averaging to keep the weak feature in most runs, got {seed_averaged_rate:.3f}"


def test_greedy_backward_elimination_n_repeats_default_is_bit_identical_to_single_run():
    X, y = _make_borderline_regression(n=50, weak_coef=0.6, noise_scale=1.0, seed=3)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)

    survivors_default = greedy_backward_elimination(Ridge(alpha=1.0), X, y, scoring=r2_score, cv=cv, min_features=1)
    survivors_explicit_n_repeats_1 = greedy_backward_elimination(Ridge(alpha=1.0), X, y, scoring=r2_score, cv=cv, min_features=1, n_repeats=1)
    assert survivors_default == survivors_explicit_n_repeats_1
