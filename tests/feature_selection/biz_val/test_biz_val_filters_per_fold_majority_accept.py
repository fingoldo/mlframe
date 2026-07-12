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


def test_biz_val_per_fold_majority_agreement_score_distinguishes_borderline_from_strong_majority():
    """The binary ``accept`` flag treats a bare 3/5 majority and a clean 5/5 sweep identically (both True at
    the default ``min_fraction=0.6``) -- it can't tell a caller how MUCH to trust the decision. The opt-in
    continuous ``agreement_score`` (Wilson lower bound) must separate them clearly, so a stricter caller can
    reject the borderline case without touching the binary vote.
    """
    baseline = [0.80, 0.81, 0.79, 0.80, 0.82]

    # Borderline: exactly 3/5 folds improved (bare majority, still >= default min_fraction=0.6).
    candidate_borderline = [0.83, 0.79, 0.78, 0.83, 0.85]
    borderline = per_fold_majority_accept(baseline, candidate_borderline, maximize=True, compute_agreement_score=True)

    # Strong: 5/5 folds improved.
    candidate_strong = [0.83, 0.84, 0.82, 0.83, 0.85]
    strong = per_fold_majority_accept(baseline, candidate_strong, maximize=True, compute_agreement_score=True)

    # Both cross the same binary bar -- the flag alone can't distinguish them.
    assert borderline["fraction_folds_improved"] == 0.6
    assert strong["fraction_folds_improved"] == 1.0
    assert borderline["accept"] is True
    assert strong["accept"] is True

    # The continuous agreement_score must separate them with a real numeric margin.
    assert borderline["agreement_score"] < 0.4, "bare 3/5 majority on 5 folds should score low confidence"
    assert strong["agreement_score"] > 0.55, "clean 5/5 sweep should score high confidence"
    assert strong["agreement_score"] - borderline["agreement_score"] >= 0.2

    # A caller tuning a stricter self-chosen threshold (e.g. 0.5) rejects the borderline case even though
    # the binary majority vote accepted it -- this is the whole point of exposing the continuous score.
    strict_threshold = 0.5
    assert borderline["agreement_score"] < strict_threshold
    assert strong["agreement_score"] >= strict_threshold


def test_per_fold_majority_accept_default_unchanged_when_agreement_score_omitted():
    baseline = [0.80, 0.81, 0.79, 0.80, 0.82]
    candidate = [0.83, 0.84, 0.78, 0.83, 0.85]

    default_result = per_fold_majority_accept(baseline, candidate, maximize=True)
    explicit_off_result = per_fold_majority_accept(baseline, candidate, maximize=True, compute_agreement_score=False)

    assert default_result == explicit_off_result
    assert "agreement_score" not in default_result
    assert set(default_result.keys()) == {"fraction_folds_improved", "mean_delta", "accept"}
