"""The auto-k selection error is the weighted loss when sample weights are given.

The models are trained with the weights, and the selection error was an unweighted mean of the squared errors, so ``k`` was tuned for a different
loss than the one the blend was fitted on. The estimators here ignore ``sample_weight``, so the predictions do not move with the weights and any
change in the selection scores comes from the error weighting alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training.composite.count_weighted_blend import CountWeightedBlendEnsemble


class _MeanIgnoresWeights(BaseEstimator, RegressorMixin):
    """Predicts the training mean of the target; accepts ``sample_weight`` and ignores it."""

    def fit(self, X, y, sample_weight=None):
        """Remember the unweighted training mean."""
        self.mean_ = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X):
        """The training mean for every row."""
        return np.full(len(X), self.mean_)


def _frame(n: int = 120, seed: int = 0):
    """Ten entities with a shared meta column and a noisy target."""
    rng = np.random.default_rng(seed)
    entity = rng.integers(0, 10, size=n)
    X = pd.DataFrame({"entity": entity, "meta": rng.normal(size=n)})
    y = rng.normal(loc=entity * 0.5, scale=1.0, size=n)
    return X, y


def _k_scores(sample_weight):
    """The auto-k scores of one ensemble fitted with the given sample weights."""
    X, y = _frame()
    ens = CountWeightedBlendEnsemble(
        entity_estimator=_MeanIgnoresWeights(), global_estimator=_MeanIgnoresWeights(), entity_col="entity",
        metadata_cols=["meta"], auto_k=True, k_grid=[1.0, 10.0, 100.0], k_cv=3, random_state=0,
    )
    ens.fit(X, y, sample_weight=sample_weight)
    return ens.k_cv_scores_


def test_selection_scores_change_with_the_sample_weights_when_the_models_ignore_them():
    """With uneven weights and weight-blind models, the selection scores differ from the uniform-weight scores."""
    n = 120
    uniform = _k_scores(np.ones(n))
    uneven = _k_scores(np.where(np.arange(n) % 2 == 0, 1.0, 50.0))
    assert any(abs(uniform[k] - uneven[k]) > 1e-9 for k in uniform)


def test_uniform_weights_reproduce_the_unweighted_scores():
    """Equal weights give the same scores as no weights, so existing unweighted behaviour is unchanged."""
    n = 120
    assert _k_scores(np.ones(n)) == _k_scores(None)
