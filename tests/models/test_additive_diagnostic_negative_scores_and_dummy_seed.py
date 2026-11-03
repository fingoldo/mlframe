"""The interaction recommendation keeps its direction for negative scores, and the dummy baseline is reproducible."""

from __future__ import annotations

import numpy as np

from mlframe.estimators.baselines import get_best_dummy_score
from mlframe.models.additive_interaction_diagnostic import _additive_deficit_exceeds
from sklearn.dummy import DummyClassifier


def test_negative_loss_scores_recommend_when_the_additive_model_is_clearly_worse():
    """-RMSE: full -1.0, additive -2.0 (twice the error): recommend (the plain ratio was 2.0 and said no)."""
    assert _additive_deficit_exceeds(-1.0, -2.0, 0.1) is True


def test_positive_scores_keep_the_historical_threshold():
    """AUC full 0.90, additive 0.80: deficit 11% -> recommend; additive 0.85: 5.6% -> no (matches ratio < 0.9)."""
    assert _additive_deficit_exceeds(0.90, 0.80, 0.1) is True
    assert _additive_deficit_exceeds(0.90, 0.85, 0.1) is False


def test_undefined_inputs_do_not_recommend():
    """Zero or non-finite full score: nothing to compare against."""
    assert _additive_deficit_exceeds(0.0, 0.5, 0.1) is False
    assert _additive_deficit_exceeds(float("nan"), 0.5, 0.1) is False


def test_dummy_baseline_is_identical_across_calls():
    """Stratified / uniform dummies were unseeded, so the best-dummy anchor moved between identical runs."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (rng.random(200) < 0.5).astype(int)
    scores = {get_best_dummy_score(DummyClassifier(), X[:100], y[:100], X[100:], y[100:], scoring=lambda m, Xt, yt, **k: float((m.predict(Xt) == yt).mean())) for _ in range(5)}
    assert len(scores) == 1
