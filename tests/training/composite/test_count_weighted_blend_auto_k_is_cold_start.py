"""The auto-k selection is scored on entities unseen in its training folds, as a new entity is at predict time.

The folds were row-shuffled, so almost every validation entity was also in training and its entity model was fitted on it. The selection then
favoured the entity model. With entities as the fold unit, every validation row is a new entity: its blend weight is zero for every ``k``, so the
scores do not depend on ``k``. On the previous code they do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training.composite.count_weighted_blend import CountWeightedBlendEnsemble


class _EntityMean(BaseEstimator, RegressorMixin):
    """Predicts each entity's training mean from the first column; an unseen entity gets the overall mean."""

    def fit(self, X, y, sample_weight=None):
        """Record the per-entity means of ``y``."""
        ent = np.asarray(X.iloc[:, 0])
        y = np.asarray(y, dtype=np.float64)
        self.overall_ = float(y.mean())
        self.means_ = {e: float(y[ent == e].mean()) for e in np.unique(ent)}
        return self

    def predict(self, X):
        """Each row's entity mean, or the overall mean for an entity not seen in fit."""
        ent = np.asarray(X.iloc[:, 0])
        return np.array([self.means_.get(e, self.overall_) for e in ent], dtype=np.float64)


class _GlobalMean(BaseEstimator, RegressorMixin):
    """Predicts the training mean for every row."""

    def fit(self, X, y, sample_weight=None):
        """Record the mean of ``y``."""
        self.mean_ = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X):
        """The training mean for every row."""
        return np.full(len(X), self.mean_)


def _data(n_entities: int = 30, rows_per_entity: int = 6, seed: int = 0):
    """Strong entity effects: each entity has its own level, and the rows are noisy around it."""
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), rows_per_entity)
    level = rng.normal(scale=5.0, size=n_entities)
    y = level[entity] + rng.normal(scale=0.5, size=entity.size)
    X = pd.DataFrame({"entity": entity, "meta": rng.normal(size=entity.size)})
    return X, y


def test_auto_k_scores_do_not_depend_on_k_when_every_validation_entity_is_unseen():
    """With entity-grouped folds, every validation row is a cold start, so all candidate k give the same selection error."""
    X, y = _data()
    ens = CountWeightedBlendEnsemble(
        entity_estimator=_EntityMean(), global_estimator=_GlobalMean(), entity_col="entity",
        metadata_cols=["meta"], auto_k=True, k_grid=[1.0, 10.0, 100.0], k_cv=3, random_state=0,
    )
    ens.fit(X, y)
    scores = np.array(list(ens.k_cv_scores_.values()))
    assert scores.max() - scores.min() < 1e-9
