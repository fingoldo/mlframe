"""Aims at giving overfitting detection capability to models that do not support it natively.

``EarlyStoppingWrapper`` adds held-out-validation early stopping to estimators that do NOT expose it
natively. Three backends, picked automatically in priority order (cheapest correct path first):

  1. ``partial_fit`` -- drive the estimator's own incremental fit loop one mini-epoch at a time
     (SGD*/Perceptron/PassiveAggressive*/MLP*/NB...).
  2. ``staged_predict`` -- a single fit, then evaluate the val metric at EVERY boosting stage and pick
     the best stage (GradientBoosting* expose ``staged_predict``/``staged_predict_proba``). This is the
     efficient form of the warm-start "dichotomy": one fit yields the whole score-vs-stage curve, so no
     refitting is needed -- far cheaper than re-growing the ensemble step by step.
  3. ``warm_start`` incremental -- for estimators with a settable ``warm_start`` flag plus an incremental
     count attribute (``n_estimators`` for ensembles/boosters, ``max_iter`` for linear/MLP) but no
     ``staged_predict`` (RandomForest/ExtraTrees/Bagging...). Grow the count in batches, refit (warm-start
     reuses the already-built estimators), score the val fold each step, snapshot the best, stop on patience.

The capability is PROBED on attributes (never a hardcoded class list), mirroring
``training/diagnostics/learning_curve._supports_warm_start`` so lgb/xgb/sklearn wrappers and future
learners with the same API work automatically. The public contract -- ``fit`` /
``predict`` / ``predict_proba`` / ``best_model_`` / ``best_score_`` -- is identical across all backends.

Classifiers and regressors are both supported: ``partial_fit``'s ``classes=`` kwarg is dropped for
regressors, and the default scorer becomes NEGATIVE RMSE (greater-is-better, so maximising it minimises
RMSE) instead of accuracy. The same greater-is-better improvement rule drives all three backends.

This is a thin control loop (a handful of fits + scores), not a numeric kernel, so the cProfile /
acceleration-ladder rule does not apply -- the cost is entirely inside the wrapped estimator's fit.
"""

from __future__ import annotations

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

import copy
import time

from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, is_regressor, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import SGDClassifier
from mlframe.metrics.core import accuracy_ratio, fast_root_mean_squared_error

from pyutilz.pythonlib import store_params_in_object, get_parent_func_args

from .early_stopping_monotonic import MonotonicDeclineStopper

# Incremental-count attributes a warm-start learner uses to grow capacity, in probe order: boosters/ensembles
# expose ``n_estimators``, linear/MLP expose ``max_iter``. Mirrors learning_curve._WARM_START_N_ATTRS.
_WARM_START_N_ATTRS = ("n_estimators", "max_iter")


class EarlyStoppingWrapper(BaseEstimator):
    def __init__(
        self,
        base_model: object,
        start_iter: int = 1,
        max_iter: int = None,
        # stopping conditions
        max_runtime_mins: float = None,
        patience: int = 5,
        # Monotonic strict-decline overfitting stop, COMPLEMENTARY to ``patience``: stop once the val
        # score has STRICTLY worsened for this many CONSECUTIVE iterations since the global best (a
        # confident overfitting signal). A new best, a plateau, or a bounce-up resets the run. Training
        # stops when EITHER patience OR this streak fires. Default-on at 5 (per CLAUDE.md "enable
        # corrective mechanisms by default"; N=5 calibrated by bench_worsening_vs_monotonic_stop.py --
        # ties patience-level holdout accuracy while stopping ~3.6x earlier than the full budget).
        # Set to ``None`` to disable.
        monotonic_decline_patience: int = 7,
        tolerance: float = 0.0,
        # CV
        validation_fraction: float = 0.1,
        scoring: Callable = accuracy_ratio,
        warm_start_step: int = None,
        random_state: int = None,
    ):
        store_params_in_object(obj=self, params=get_parent_func_args())

    # ------------------------------------------------------------------ helpers

    def _resolve_scoring(self):
        """Pick the scorer: caller-supplied, else accuracy (classifier) / negative-RMSE (regressor).

        The improvement rule is greater-is-better (``score > best_score_``), so maximising ``-RMSE``
        minimises RMSE. RMSE is preferred over R^2 -- R^2's variance denominator is unstable on the small
        validation folds the wrapper holds out (a low-variance val slice inflates/explodes it). Only swapped
        in when the caller left the default scorer in place.
        """
        scoring = self.scoring
        if self._is_regressor and scoring is accuracy_ratio:
            scoring = lambda _yt, _yp: -float(fast_root_mean_squared_error(_yt, _yp))  # noqa: E731
        return scoring

    def _split(self, X, y):
        """Shuffle + (for classifiers) stratify a ``validation_fraction`` holdout; guard against a zero-row split.

        Previously this held out the LAST rows with no shuffle/stratify/seed, which on data sorted by class
        (the common case) produced a degenerate single-class validation fold and a biased early-stopping signal.
        Mirrors ``base.SplitFitEstimator`` (shuffled, stratified, seeded train/val split) for a representative fold.
        """
        # int(len(X) * frac) could round down to 0 on small X (e.g. len=9, frac=0.1 -> int(0.9)=0); clamp to >=1.
        n_val_samples = max(1, int(len(X) * self.validation_fraction))
        if n_val_samples >= len(X):
            raise ValueError(
                f"early-stopping: validation_fraction={self.validation_fraction} with len(X)={len(X)} leaves "
                f"zero training rows (n_val_samples={n_val_samples}). Use a smaller validation_fraction or more samples."
            )
        from sklearn.model_selection import train_test_split

        # Stratify only when classification AND every class has >=2 rows AND the val fold can hold at least one
        # row per class (StratifiedShuffleSplit rejects test_size < n_classes and a stratify vector with a
        # singleton class). When the requested val size is too small to stratify, fall back to a plain shuffled
        # split rather than crashing -- a 1-row val on a 2-class target is a legitimate tiny-fraction request.
        stratify = None
        if not self._is_regressor:
            classes, counts = np.unique(np.asarray(y), return_counts=True)
            if counts.min() >= 2 and n_val_samples >= classes.size:
                stratify = y
        return train_test_split(
            X, y, test_size=n_val_samples, random_state=self.random_state, shuffle=True, stratify=stratify,
        )

    def _capability(self, model):
        """Return the early-stopping backend for ``model``: 'partial_fit' | 'staged' | ('warm_start', attr) | None.

        Probed on attributes, not class identity. ``partial_fit`` wins first (cheapest -- the estimator's own
        incremental loop). Else ``staged_predict``/``staged_predict_proba`` (one fit, whole score curve). Else a
        settable ``warm_start`` flag plus an incremental count attr (``n_estimators``/``max_iter``).
        """
        if hasattr(model, "partial_fit"):
            return "partial_fit"
        staged = "staged_predict_proba" if hasattr(model, "predict_proba") else "staged_predict"
        if hasattr(model, staged):
            return "staged"
        if hasattr(model, "set_params") and hasattr(model, "get_params"):
            try:
                params = model.get_params()
            except Exception:
                params = {}
            if "warm_start" in params:
                for attr in _WARM_START_N_ATTRS:
                    if attr in params:
                        return ("warm_start", attr)
        return None

    def _deadline(self):
        """Wall-clock deadline from ``max_runtime_mins`` (None -> +inf). Checked at the top of each iteration."""
        if self.max_runtime_mins is None:
            return np.inf
        return time.monotonic() + self.max_runtime_mins * 60.0

    def _consider(self, score, model_provider):
        """Apply the greater-is-better improvement rule; snapshot a deep copy of the best model.

        ``model_provider`` is a zero-arg callable returning the model to snapshot -- deferred so we only
        deep-copy when the score actually improves. Returns True once EITHER patience is exhausted OR the
        monotonic strict-decline streak fires (whichever first). Patience only starts counting at/after
        ``start_iter`` (the warm-up grace window).
        """
        if score > self.best_score_ + self.tolerance:
            self.best_score_ = score
            # Snapshot AT the best iteration: a live reference would keep mutating in later iterations, so
            # best_model_ would hold the FINAL (often degraded) weights -- silently defeating ES.
            self.best_model_ = copy.deepcopy(model_provider())
            self.no_improvement_count_ = 0
        else:
            self.no_improvement_count_ += 1
        # Count every scored iteration so callers / tests can see the work the stop saved.
        self.n_iterations_ += 1
        # Monotonic strict-decline streak (greater-is-better -- the wrapper's scorer is always
        # greater-is-better: accuracy or negative-RMSE). Fires independently of patience; best_model_
        # already holds the global-best snapshot so a streak-stop keeps the right model.
        monotonic_stop = self._monotonic_stopper.update(score)
        return (self.no_improvement_count_ >= self.patience) or monotonic_stop

    # ------------------------------------------------------------------ backends

    def _fit_partial(self, X_train, y_train, X_val, y_val, y, scoring, deadline):
        pf_kwargs = {} if self._is_regressor else {"classes": np.unique(y)}
        for i in range(1, self.max_iter + 1):
            if time.monotonic() > deadline:
                logger.info("Early stopping (max_runtime) at iteration %d", i)
                break
            self.estimator_.partial_fit(X_train, y_train, **pf_kwargs)
            score = scoring(y_val, self.estimator_.predict(X_val))
            if i >= self.start_iter and self._consider(score, lambda: self.estimator_):
                logger.info("Early stopping at iteration %d", i)
                break

    def _fit_staged(self, X_train, y_train, X_val, y_val, scoring):
        """One fit, then evaluate the val metric at every boosting stage; keep the best-scoring prefix.

        ``staged_predict*`` yields the prediction after each added stage from a SINGLE fit, so the whole
        score-vs-stage curve costs one fit. We snapshot the estimator truncated to the best stage by copying
        it and setting ``n_estimators`` to that stage (the fast variant of the warm-start dichotomy).
        """
        # Grow to the full budget first: staged_predict yields one prediction per existing stage, so the
        # estimator must be built with max_iter stages for the curve to span the whole search range.
        if "n_estimators" in self.estimator_.get_params():
            self.estimator_.set_params(n_estimators=self.max_iter)
        self.estimator_.fit(X_train, y_train)
        if not self._is_regressor and hasattr(self.estimator_, "staged_predict_proba"):
            classes = self.estimator_.classes_
            stages = (classes[np.argmax(p, axis=1)] for p in self.estimator_.staged_predict_proba(X_val))
        else:
            stages = self.estimator_.staged_predict(X_val)
        best_stage = 0
        for stage, y_pred in enumerate(stages, start=1):
            score = scoring(y_val, y_pred)
            stop = stage >= self.start_iter and self._consider(score, lambda: self.estimator_)
            if self.no_improvement_count_ == 0:
                best_stage = stage
            if stop:
                logger.info("Early stopping at stage %d", stage)
                break
        # Truncate the snapshot to the winning stage so predict() uses exactly that prefix of estimators.
        if self.best_model_ is not None and "n_estimators" in self.best_model_.get_params():
            self.best_model_.set_params(n_estimators=best_stage)

    def _fit_warm(self, X_train, y_train, X_val, y_val, scoring, n_attr, deadline):
        """Warm-start incremental: grow ``n_attr`` in batches, refit (reusing prior estimators), score, snapshot.

        ``warm_start=True`` makes each ``fit`` continue from the previous one (adds trees / runs more
        iterations) instead of starting over, so the loop costs ~one full fit of incremental work total.
        """
        self.estimator_.set_params(warm_start=True)
        step = self.warm_start_step
        if step is None:
            try:
                base_n = int(self.estimator_.get_params().get(n_attr) or 0)
            except Exception:
                base_n = 0
            step = max(1, base_n // 10) if base_n else 10
        i = 0
        n = 0
        while n < self.max_iter:
            if time.monotonic() > deadline:
                logger.info("Early stopping (max_runtime) at count %d", n)
                break
            i += 1
            n = min(self.max_iter, i * step)
            self.estimator_.set_params(**{n_attr: n})
            self.estimator_.fit(X_train, y_train)
            score = scoring(y_val, self.estimator_.predict(X_val))
            if i >= self.start_iter and self._consider(score, lambda: self.estimator_):
                logger.info("Early stopping at count %d", n)
                break

    # ------------------------------------------------------------------ public API

    def fit(self, X, y):
        # sklearn contract: never mutate the caller-supplied ``base_model``. Under clone / GridSearchCV every
        # clone shares the SAME ``base_model`` object, so fitting it in place would train an already-fitted
        # estimator on a later fold (and leave the user's instance fitted). Operate on a fresh clone instead.
        self.estimator_ = clone(self.base_model)
        _shape = getattr(X, "shape", None)
        if _shape is not None and len(_shape) >= 2:
            self.n_features_in_ = int(_shape[1])
        self.best_score_ = -np.inf
        self.best_model_ = None
        self.no_improvement_count_ = 0
        self.n_iterations_ = 0
        # Scorer is always greater-is-better here, so the monotonic detector runs in mode="max".
        self._monotonic_stopper = MonotonicDeclineStopper(self.monotonic_decline_patience, mode="max")

        # ``is_regressor`` on a base model that does not inherit ``BaseEstimator`` raises under recent
        # sklearn (no ``__sklearn_tags__`` in the MRO). The wrapper supports any duck-typed partial_fit /
        # warm_start object, so fall back to "not a regressor" (classifier scoring) when tags are absent.
        # Determined BEFORE the split so ``_split`` can stratify on classification targets.
        try:
            self._is_regressor = is_regressor(self.estimator_)
        except AttributeError:
            self._is_regressor = False
        # Validate / build the train/val split (shuffled + stratified, see ``_split``) so a degenerate
        # validation_fraction raises the clear "zero training rows" error rather than failing later.
        X_train, X_val, y_train, y_val = self._split(X, y)
        scoring = self._resolve_scoring()
        if self.max_iter is None:
            raise ValueError("EarlyStoppingWrapper requires max_iter to bound the iteration / growth schedule.")
        deadline = self._deadline()

        cap = self._capability(self.estimator_)
        if cap == "partial_fit":
            self._fit_partial(X_train, y_train, X_val, y_val, y, scoring, deadline)
        elif cap == "staged":
            self._fit_staged(X_train, y_train, X_val, y_val, scoring)
        elif isinstance(cap, tuple) and cap[0] == "warm_start":
            self._fit_warm(X_train, y_train, X_val, y_val, scoring, cap[1], deadline)
        else:
            raise TypeError(
                f"{type(self.estimator_).__name__} cannot be early-stopped: it exposes neither partial_fit, nor "
                f"staged_predict, nor a warm_start flag with an incremental count attribute "
                f"({' / '.join(_WARM_START_N_ATTRS)})."
            )

        if self.best_model_ is None:
            # No iteration produced a finite score (e.g. max_iter exhausted before start_iter, or all-NaN scores).
            # Fall back to the last-fit model so predict() still works rather than raising on a None snapshot.
            self.best_model_ = copy.deepcopy(self.estimator_)
        return self

    def predict(self, X):
        check_is_fitted(self, "best_model_")
        return self.best_model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "best_model_")
        return self.best_model_.predict_proba(X)


# Demo / smoke test kept for reference. Guarded behind __main__ so it no longer
# trains a model + prints to stdout at import time.
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    base_model = SGDClassifier(max_iter=1, tol=None)  # max_iter=1 so partial_fit drives iteration
    early_stopping_model = EarlyStoppingWrapper(base_model, patience=5, max_iter=100)

    early_stopping_model.fit(X_train, y_train)
    y_pred = early_stopping_model.predict(X_test)

    print(f"Accuracy: {accuracy_ratio(y_test, y_pred)}")
