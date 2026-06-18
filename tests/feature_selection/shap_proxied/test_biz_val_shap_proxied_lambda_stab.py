"""biz_val: ``lambda_stab`` makes revalidation prefer a STABLE subset over an unstable lower-mean one.

The honest re-validation winner is chosen by ``stable_score = mean(honest_loss) + lambda_stab * std``
across the per-candidate seed retrains. With ``lambda_stab=0`` the pick is mean-only, so a candidate
whose mean is marginally lower but whose loss swings wildly across seeds (high variance => unreliable
on unseen folds) wins. A positive ``lambda_stab`` penalises that variance and instead selects the
candidate whose loss is consistently low across seeds -- the one that generalises predictably.

This pins the decision kernel ``_winner_from_per_candidate`` directly (deterministic, no model fit):
an unstable candidate with the lower MEAN but high std vs a stable candidate with a slightly higher
mean but near-zero std. ``lambda_stab=0`` -> unstable wins; ``lambda_stab=0.5`` -> stable wins.
"""

from __future__ import annotations

import numpy as np


def _setup():
    # cand 0 (UNSTABLE): mean 0.20, std 0.155 -- lower mean, high seed-to-seed swing.
    # cand 1 (STABLE):   mean 0.23, std ~0.003 -- slightly higher mean, consistent.
    per_candidate = {0: [0.05, 0.35, 0.04, 0.36], 1: [0.23, 0.235, 0.225, 0.23]}
    candidates = [(0.0, (0, 1, 2)), (0.0, (3, 4))]
    member_cols = [[0, 1, 2], [3, 4]]
    return per_candidate, candidates, member_cols


def _winner(lambda_stab):
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_refine import (
        _winner_from_per_candidate,
    )

    per_candidate, candidates, member_cols = _setup()
    return _winner_from_per_candidate(per_candidate, candidates, member_cols, lambda_stab, parsimony_tol=0.0)


def test_biz_val_lambda_stab_off_picks_unstable_lower_mean_subset():
    per, _, _ = _setup()
    assert np.mean(per[0]) < np.mean(per[1])  # unstable cand has the lower mean
    assert np.std(per[0]) > 10 * np.std(per[1])  # but is far more variable
    assert _winner(0.0) == (0, 1, 2), "lambda_stab=0 must pick the mean-best (unstable) candidate"


def test_biz_val_lambda_stab_on_switches_to_stable_subset():
    assert _winner(0.5) == (3, 4), "lambda_stab=0.5 must penalise variance and pick the stable candidate"
