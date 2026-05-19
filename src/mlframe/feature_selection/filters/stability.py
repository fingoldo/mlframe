"""Stability selection wrapper around MRMR.

mRMR with permutation-test confidence is unstable for small N: the support depends on the seed of the permutation pass. Stability
selection (Meinshausen-Buhlmann 2010 -- analog ``RandomizedLasso`` in old sklearn) addresses this by running mRMR on ``n_bootstraps``
subsamples and recommending only features that appear in the support of at least ``support_threshold`` (default 0.6 = 60%) of runs.

Public class
------------
``StabilityMRMR(estimator, n_bootstraps=20, sample_fraction=0.75, support_threshold=0.6, random_state=None)``

Same ``.fit / .transform / .support_ / .selection_probabilities_`` surface as ``MRMR``. ``selection_probabilities_`` exposes per-feature
inclusion frequency as a numpy float vector for downstream stability plots.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone

logger = logging.getLogger(__name__)


class StabilityMRMR(BaseEstimator, TransformerMixin):
    """Bootstrap-stability wrapper for mRMR-family selectors.

    Each bootstrap iteration:
    1. Sample ``sample_fraction * n_samples`` rows without replacement (with ``random_state + iteration`` seed).
    2. Fit a clone of ``estimator`` on the subsample.
    3. Record ``estimator.support_`` (set of selected feature indices).

    After all iterations:
    * ``selection_probabilities_[j] = P(feature_j in support across bootstraps)``.
    * ``support_`` = features with prob >= ``support_threshold``.

    Per Meinshausen-Buhlmann, ``support_threshold=0.6`` controls the expected number of false positives at a known FDR level given mild
    assumptions on the base selector; tune up to ~0.8 for stricter control.

    Parameters
    ----------
    estimator : BaseEstimator
        Any selector with ``.fit(X, y)`` and ``.support_`` attributes (typically an ``MRMR`` instance).
    n_bootstraps : int, default 20
    sample_fraction : float, default 0.75
        Fraction of rows to subsample per bootstrap.
    support_threshold : float, default 0.6
    random_state : int, default None
    n_jobs : int, default 1
        Passes through to ``joblib.Parallel`` for the bootstrap loop.
    """
    def __init__(
        self,
        estimator,
        n_bootstraps: int = 20,
        sample_fraction: float = 0.75,
        support_threshold: float = 0.6,
        random_state: int = None,
        n_jobs: int = 1,
    ):
        self.estimator = estimator
        self.n_bootstraps = n_bootstraps
        self.sample_fraction = sample_fraction
        self.support_threshold = support_threshold
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from joblib import Parallel, delayed
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        sub_size = int(self.sample_fraction * n_samples)

        # Generate seeds upfront so the bootstrap is deterministic for a given ``random_state`` regardless of joblib worker order.
        seeds = rng.integers(0, 2 ** 31 - 1, size=self.n_bootstraps)

        def _one_bootstrap(seed: int) -> np.ndarray:
            local_rng = np.random.default_rng(seed)
            idx = local_rng.choice(n_samples, size=sub_size, replace=False)
            X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
            y_sub = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
            est = clone(self.estimator)
            est.fit(X_sub, y_sub)
            return np.asarray(est.support_, dtype=np.int64)

        if self.n_jobs == 1:
            supports = [_one_bootstrap(s) for s in seeds]
        else:
            # backend="threading": each bootstrap fits an estimator clone on a
            # resampled (X, y) tuple. loky's default would deep-copy the whole
            # (X, y) into each worker process -- with the stability path
            # already running inside MRMR's outer FE loop, that's a recipe for
            # the iter-371 paging cascade. Estimator clones share the parent
            # X/y arrays under threading; the inner fit holds the GIL anyway
            # so threading doesn't speed up the fits themselves, but it
            # eliminates the OOM risk and removes loky process-spawn cost.
            supports = Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(_one_bootstrap)(s) for s in seeds)

        # Accumulate per-feature inclusion counts.
        counts = np.zeros(n_features, dtype=np.int64)
        for sup in supports:
            counts[sup] += 1

        self.selection_probabilities_ = counts / self.n_bootstraps
        self.support_ = np.where(self.selection_probabilities_ >= self.support_threshold)[0]
        self.n_features_ = len(self.support_)
        self.n_features_in_ = n_features
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X, y=None):
        if hasattr(X, "iloc"):
            return X.iloc[:, self.support_]
        return X[:, self.support_]
