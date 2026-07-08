"""Noise-injected multi-model ensemble for NON-TREE estimators (Workstream B3).

Trains ``k`` clones of a base estimator, each on the train rows + fresh per-feature Gaussian jitter, and
averages their predictions. Input noise is approximately Tikhonov/L2 regularisation, so it helps
variance-prone NON-TREE learners (neural nets, linear, kNN); tree estimators are insensitive to small
input jitter and are already covered by row-resampling bagging, so this wrapper is scoped to non-trees by
convention (the suite routes trees to the existing bagging path). Picklable; sklearn-clone friendly.

Per the plan this ships only where its biz_value beats the baseline. Measured: a clear win on a
high-variance learner (1-NN, ~34% honest-RMSE cut at sigma=0.2/k=20) but a measured LOSS on low-variance
OLS (n>p, low label noise) where feature noise just attenuates coefficients (errors-in-variables) -- so
the scope is variance-prone non-tree models, not OLS (REJECTED!=DELETED: both results pinned in tests).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class NoiseAugmentedEnsemble:
    """Average ``k`` clones each fit on jittered inputs. ``predict`` (+ ``predict_proba`` if the base has it)."""

    def __init__(self, base_estimator: Any, *, k: int = 5, sigma_scale: float = 0.05, seed: int = 0) -> None:
        self.base_estimator = base_estimator
        self.k = int(k)
        self.sigma_scale = float(sigma_scale)
        self.seed = int(seed)

    def fit(self, X, y):
        """Fit ``k`` independent clones of ``base_estimator``, each on ``X`` perturbed by its own Gaussian jitter stream (scaled per-feature by ``sigma_scale * std``); jitter streams are spawned from one seed via ``SeedSequence.spawn`` for reproducible, mutually-independent member noise."""
        from sklearn.base import clone

        Xf = np.asarray(X, dtype=np.float64)
        std = Xf.std(axis=0).reshape(1, -1)
        # Per-member independent RNG streams derived from the parent seed via
        # SeedSequence.spawn: each clone draws statistically independent noise
        # (proper ensemble diversity), and the whole set is reproducible under
        # a fixed `seed`. spawn() is preferred over a single shared Generator
        # because independence is guaranteed by construction rather than by
        # sequential-draw ordering, and it composes when NoiseAugmentedEnsembles
        # are stacked (distinct parent seeds -> disjoint child streams).
        n_members = max(1, self.k)
        child_rngs = np.random.default_rng(self.seed).spawn(n_members)
        self.estimators_ = []
        for member_rng in child_rngs:
            Xn = Xf + member_rng.standard_normal(Xf.shape) * (self.sigma_scale * std)
            est = clone(self.base_estimator)
            est.fit(Xn, y)
            self.estimators_.append(est)
        if hasattr(self.estimators_[0], "classes_"):
            self.classes_ = self.estimators_[0].classes_
        return self

    def predict(self, X):
        """Average the ``k`` members' predictions on the unperturbed query ``X``."""
        Xf = np.asarray(X, dtype=np.float64)
        preds = np.stack([np.asarray(e.predict(Xf)) for e in self.estimators_], axis=0)
        return preds.mean(axis=0)

    def predict_proba(self, X):
        """Average the ``k`` members' predicted class probabilities on the unperturbed query ``X``."""
        Xf = np.asarray(X, dtype=np.float64)
        probs = np.stack([np.asarray(e.predict_proba(Xf)) for e in self.estimators_], axis=0)
        return probs.mean(axis=0)
