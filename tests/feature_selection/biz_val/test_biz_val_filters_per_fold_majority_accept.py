"""biz_value test for ``feature_selection.filters.per_fold_majority_accept`` / ``seed_averaged_fold_scores``.

The win: a feature that happens to help ONE fold a lot (by chance) while slightly hurting the rest can still
show a POSITIVE aggregate mean CV delta -- aggregate-only acceptance would wrongly keep it. The per-fold-
majority criterion correctly rejects it (since it only improves a minority of folds), while still accepting
a genuine improvement that helps most folds.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._per_fold_majority_accept import per_fold_majority_accept, seed_averaged_fold_scores


def test_biz_val_per_fold_majority_rejects_lucky_single_fold_despite_positive_mean_delta():
    baseline = [0.80, 0.80, 0.80, 0.80, 0.80]
    candidate = [0.95, 0.78, 0.78, 0.78, 0.78]  # one lucky fold, mild hurt elsewhere

    result = per_fold_majority_accept(baseline, candidate, maximize=True)
    assert result["mean_delta"] > 0, "aggregate mean delta should be positive (the misleading signal)"
    assert result["fraction_folds_improved"] < 0.6
    assert result["accept"] is False


def test_biz_val_per_fold_majority_accepts_genuine_majority_improvement():
    baseline = [0.80, 0.81, 0.79, 0.80, 0.82]
    candidate = [0.83, 0.84, 0.78, 0.83, 0.85]  # improves 4/5 folds

    result = per_fold_majority_accept(baseline, candidate, maximize=True)
    assert result["fraction_folds_improved"] >= 0.6
    assert result["accept"] is True


def test_seed_averaged_fold_scores_reduces_noise_vs_single_seed():
    rng = np.random.default_rng(0)
    true_fold_scores = np.array([0.80, 0.81, 0.79, 0.82, 0.80])

    def noisy_score_fn(seed):
        local_rng = np.random.default_rng(seed)
        return true_fold_scores + local_rng.normal(0, 0.05, size=5)

    single_seed_error = float(np.mean(np.abs(noisy_score_fn(0) - true_fold_scores)))
    averaged = seed_averaged_fold_scores(noisy_score_fn, n_repeats=8, base_seed=100)
    averaged_error = float(np.mean(np.abs(averaged - true_fold_scores)))

    assert averaged_error < single_seed_error


def test_per_fold_majority_accept_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        per_fold_majority_accept([0.8, 0.9], [0.8])
