"""Regression tests for the ML-sampler objective->weight arithmetic in ParamsOptimizer.suggest_trials.

The weighting is now factored into ``objective_to_sampling_weights`` (called from suggest_trials), so these
tests exercise the real production code, not a re-implemented mirror. Three pre-fix bugs are pinned:

  (a) `probs = predictions + predictions.min()` doubled the negative offset instead of shifting to >=0.
  (b) `if minimize: probs = 1 - probs` on raw (unnormalized) predictions produced wrong-sign weights.
  (c) the minimize and maximize threshold branches were identical (both anchored on y.max()).
"""

import numpy as np

from mlframe.models.tuning import objective_to_sampling_weights


def test_shift_to_nonnegative_does_not_double_offset():
    """With min == -2, the shifted weights must start at 0, not -4 (the pre-fix `+ min` bug)."""
    predictions = np.array([-2.0, 0.0, 3.0])
    probs = objective_to_sampling_weights(predictions, y=np.array([-2.0, 3.0]), minimize=False, improving_by_atleast=0.0)
    assert probs.min() >= 0.0
    assert np.isclose(probs.min(), 0.0)


def test_maximize_path_weights_high_objective_candidates():
    """Maximize path weights high objective candidates."""
    predictions = np.array([0.1, 0.5, 0.9])
    y = np.array([0.0, 1.0])
    probs = objective_to_sampling_weights(predictions, y, minimize=False, improving_by_atleast=0.0)
    # Highest-objective candidate must carry the largest sampling weight.
    assert np.argmax(probs) == 2


def test_minimize_path_weights_low_objective_candidates():
    """Minimize path weights low objective candidates."""
    predictions = np.array([0.1, 0.5, 0.9])
    y = np.array([0.0, 1.0])
    probs = objective_to_sampling_weights(predictions, y, minimize=True, improving_by_atleast=0.0)
    # Lowest-objective candidate must carry the largest sampling weight (pre-fix `1 - probs` got this wrong).
    assert np.argmax(probs) == 0


def test_minimize_and_maximize_select_opposite_bands():
    """The two branches must NOT compute the identical threshold; they select opposite candidate bands."""
    predictions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y = np.array([0.0, 1.0])

    probs_min = objective_to_sampling_weights(predictions, y, minimize=True, improving_by_atleast=0.5)
    probs_max = objective_to_sampling_weights(predictions, y, minimize=False, improving_by_atleast=0.5)

    # Minimize keeps the LOW band (objective < 0.5), maximize keeps the HIGH band (objective > 0.5).
    kept_min = np.where(probs_min > 0)[0]
    kept_max = np.where(probs_max > 0)[0]
    assert kept_min.tolist() == [0, 1]
    assert kept_max.tolist() == [3, 4]
