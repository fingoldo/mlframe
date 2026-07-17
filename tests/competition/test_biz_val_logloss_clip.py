"""Unit + biz_value tests for the competition-only log-loss clipping trick.

See src/mlframe/competition/logloss_clip.py for the full metric-gaming warning.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.competition.logloss_clip import clip_probabilities_for_logloss

EPS = 1e-15


def _log_loss(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Compute binary cross-entropy log-loss with a fixed EPS floor/ceiling clip."""
    p = np.clip(probs, EPS, 1 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Compute the mean squared error between predicted probabilities and labels (a proper scoring rule)."""
    return float(np.mean((probs - y_true) ** 2))


def _make_overconfident_miss_scenario(n_correct: int = 200, n_wrong: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Model that's well-calibrated on most rows but confidently wrong on a few.

    A realistic "occasional overconfident miss" pattern: most predictions track
    the true label closely, but a handful of rows get near-0/near-1 predictions
    for the OPPOSITE of the true class (the classic log-loss-tanking scenario).
    """
    rng = np.random.default_rng(seed)
    y_correct = rng.integers(0, 2, size=n_correct)
    probs_correct = np.where(
        y_correct == 1,
        rng.uniform(0.6, 0.9, size=n_correct),
        rng.uniform(0.1, 0.4, size=n_correct),
    )

    y_wrong = rng.integers(0, 2, size=n_wrong)
    probs_wrong = np.where(y_wrong == 1, 1e-6, 1 - 1e-6)

    y_true = np.concatenate([y_correct, y_wrong])
    probs = np.concatenate([probs_correct, probs_wrong])
    return y_true.astype(np.float64), probs.astype(np.float64)


def test_unit_clip_bounds_respected() -> None:
    """Clipped output is bounded to [lower, upper] and passes through mid-range values unchanged."""
    probs = np.array([0.0, 0.001, 0.5, 0.999, 1.0])
    clipped = clip_probabilities_for_logloss(probs, lower=0.05, upper=0.95)
    assert clipped.min() >= 0.05
    assert clipped.max() <= 0.95
    np.testing.assert_allclose(clipped, [0.05, 0.05, 0.5, 0.95, 0.95])


def test_unit_default_bounds_are_tight() -> None:
    """Default lower/upper bounds clip to 1e-4 / 1-1e-4, a tight but nonzero margin."""
    probs = np.array([0.0, 1.0])
    clipped = clip_probabilities_for_logloss(probs)
    assert clipped[0] == pytest.approx(1e-4)
    assert clipped[1] == pytest.approx(1 - 1e-4)


def test_unit_invalid_bounds_raise() -> None:
    """lower>upper, negative lower, or upper>1 each raise ValueError."""
    probs = np.array([0.1, 0.9])
    with pytest.raises(ValueError):
        clip_probabilities_for_logloss(probs, lower=0.9, upper=0.1)
    with pytest.raises(ValueError):
        clip_probabilities_for_logloss(probs, lower=-0.1)
    with pytest.raises(ValueError):
        clip_probabilities_for_logloss(probs, upper=1.5)


def test_biz_val_logloss_clip_reduces_logloss_under_overconfident_misses() -> None:
    """Clipping measurably reduces total log-loss when a few predictions are
    confidently wrong - this is the leaderboard trick working as intended."""
    y_true, probs = _make_overconfident_miss_scenario()

    unclipped_loss = _log_loss(y_true, probs)
    clipped = clip_probabilities_for_logloss(probs, lower=0.05, upper=0.95)
    clipped_loss = _log_loss(y_true, clipped)

    assert clipped_loss < unclipped_loss
    # measured improvement is large (the few near-0/1 misses dominate unclipped loss);
    # assert a real, threshold'd reduction, not just "some" improvement
    relative_improvement = (unclipped_loss - clipped_loss) / unclipped_loss
    assert relative_improvement >= 0.40, (
        f"expected >=40% log-loss reduction from clipping, got {relative_improvement:.2%} (unclipped={unclipped_loss:.4f}, clipped={clipped_loss:.4f})"
    )


def _make_mostly_confident_correct_scenario(n_correct: int = 300, n_wrong: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Many extremely-confident-and-CORRECT predictions plus a few confidently-wrong.

    Tuned so log-loss (unbounded, dominated by the few wrong predictions) still
    improves under clipping, while Brier score (bounded in [0, 1], so the many
    confident-correct predictions dragged toward 0.5 dominate the aggregate)
    gets WORSE - the metric-gaming signature the tracker critique warns about.
    """
    rng = np.random.default_rng(seed)
    y_correct = rng.integers(0, 2, size=n_correct)
    probs_correct = np.where(
        y_correct == 1,
        rng.uniform(0.98, 0.999, size=n_correct),
        rng.uniform(0.001, 0.02, size=n_correct),
    )

    y_wrong = rng.integers(0, 2, size=n_wrong)
    probs_wrong = np.where(y_wrong == 1, 1e-6, 1 - 1e-6)

    y_true = np.concatenate([y_correct, y_wrong])
    probs = np.concatenate([probs_correct, probs_wrong])
    return y_true.astype(np.float64), probs.astype(np.float64)


def test_biz_val_logloss_clip_does_not_improve_brier_score_honest_check() -> None:
    """Honesty check demanded by the tracker critique: clipping is metric-gaming.

    Log-loss improves under clipping (the few confidently-wrong predictions
    dominate the unbounded metric), but the Brier score - a proper scoring
    rule not fooled by truncating confidence - measurably WORSENS, because
    clipping also drags the many well-calibrated CONFIDENT-CORRECT predictions
    toward 0.5. This demonstrates clipping optimizes the log-loss metric
    specifically, not the model's actual calibration quality.
    """
    y_true, probs = _make_mostly_confident_correct_scenario()

    unclipped_loss = _log_loss(y_true, probs)
    unclipped_brier = _brier_score(y_true, probs)
    clipped = clip_probabilities_for_logloss(probs, lower=0.05, upper=0.95)
    clipped_loss = _log_loss(y_true, clipped)
    clipped_brier = _brier_score(y_true, clipped)

    # the gaming trick "works" on log-loss...
    assert clipped_loss < unclipped_loss
    # ...but a proper scoring rule (Brier) is NOT fooled - it gets measurably worse
    assert clipped_brier > unclipped_brier
    relative_worsening = (clipped_brier - unclipped_brier) / unclipped_brier
    assert relative_worsening >= 0.05, (
        f"expected clipping to measurably worsen Brier score (proper-scoring-rule honesty check), "
        f"got relative change {relative_worsening:.2%} "
        f"(unclipped={unclipped_brier:.4f}, clipped={clipped_brier:.4f})"
    )


def test_biz_val_logloss_clip_no_miss_scenario_only_hurts() -> None:
    """When there are no confidently-wrong predictions, clipping only ever hurts
    both log-loss and Brier score - reinforcing this is a narrow leaderboard
    trick, not a general improvement."""
    rng = np.random.default_rng(1)
    n = 200
    y_true = rng.integers(0, 2, size=n).astype(np.float64)
    probs = np.where(y_true == 1, rng.uniform(0.55, 0.75, size=n), rng.uniform(0.25, 0.45, size=n))

    unclipped_loss = _log_loss(y_true, probs)
    unclipped_brier = _brier_score(y_true, probs)

    clipped = clip_probabilities_for_logloss(probs, lower=0.05, upper=0.95)
    clipped_loss = _log_loss(y_true, clipped)
    clipped_brier = _brier_score(y_true, clipped)

    assert clipped_loss >= unclipped_loss
    assert clipped_brier >= unclipped_brier
