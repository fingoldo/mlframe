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
        # 2026-05-30 Wave 9.1 fix (loop iter 41): input validation.
        # Pre-fix:
        #   * sample_fraction=0.05 with n=10 -> sub_size=int(0.5)=0,
        #     every clone got X[0:0] (empty fit), silent garbage
        #     ``selection_probabilities_``.
        #   * n_bootstraps=0 -> counts / 0 = NaN, ``support_=[]``
        #     indistinguishable from a legitimate "all below threshold"
        #     result.
        #   * Negative / out-of-range params leaked to numpy with
        #     unhelpful errors.
        if not isinstance(self.n_bootstraps, (int, np.integer)) or self.n_bootstraps < 1:
            raise ValueError(
                f"StabilityMRMR: n_bootstraps must be a positive integer; "
                f"got {self.n_bootstraps!r}."
            )
        if not (0.0 < float(self.sample_fraction) <= 1.0):
            raise ValueError(
                f"StabilityMRMR: sample_fraction must be in (0, 1]; "
                f"got {self.sample_fraction!r}."
            )
        if not (0.0 < float(self.support_threshold) <= 1.0):
            raise ValueError(
                f"StabilityMRMR: support_threshold must be in (0, 1]; "
                f"got {self.support_threshold!r}."
            )
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        # Floor at 2 so a clone never sees 0/1-row fit (numerically
        # degenerate; MI estimators raise or return NaN at n<2).
        sub_size = max(2, int(round(self.sample_fraction * n_samples)))
        if sub_size > n_samples:
            raise ValueError(
                f"StabilityMRMR: sub_size ({sub_size}) exceeds n_samples "
                f"({n_samples}); reduce sample_fraction."
            )

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
        # 2026-05-30 Wave 9.1 fix (loop iter 42): validate fit-time
        # column semantics at transform. Pre-fix the function
        # positional-indexed via ``X.iloc[:, self.support_]`` with no
        # check that the columns AT transform time match the columns
        # AT fit time. Reordering, renaming, or dropping columns
        # silently returned the wrong slice - downstream models saw
        # feature ``d`` labelled as ``b`` and vice versa.
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            cols = list(X.columns)
            fit_cols = list(self.feature_names_in_)
            if cols != fit_cols:
                # Realign by name when all fit columns are present
                # (sklearn ``_check_feature_names(reset=False)`` semantics).
                if set(fit_cols).issubset(cols):
                    X = X[fit_cols]
                else:
                    missing = sorted(set(fit_cols) - set(cols))
                    raise ValueError(
                        f"StabilityMRMR.transform: X columns differ from "
                        f"fit; missing {missing!r}."
                    )
        elif hasattr(self, "n_features_in_"):
            _ncols = int(X.shape[1])
            if _ncols != int(self.n_features_in_):
                raise ValueError(
                    f"StabilityMRMR.transform: X has {_ncols} features, "
                    f"but StabilityMRMR is expecting "
                    f"{int(self.n_features_in_)} features as input."
                )
        if hasattr(X, "iloc"):
            return X.iloc[:, self.support_]
        return X[:, self.support_]
