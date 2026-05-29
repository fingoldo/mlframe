"""Bring patience + curve-shape early stopping to estimators that don't natively support it.

Two ES strategies dispatched by capability:

  - **partial_fit-based** (preferred when the estimator has ``partial_fit``). Train on the full
    train set in mini-epochs: each ``partial_fit`` call processes the data once, then we
    evaluate on a held-out val set, push the metric into a ``UniversalCallback``, and stop
    when patience or the curve-shape detector fires. Cost: O(N * epochs), each epoch is O(N).
    Applicable to sklearn ``SGDRegressor``/``SGDClassifier``, ``MultinomialNB``,
    ``PassiveAggressive*``, and pytorch / lightning models that expose ``partial_fit`` shims.

  - **dichotomic search on a budget hyperparameter** (when no ``partial_fit`` but the
    estimator exposes a budget knob like ``n_estimators``, ``max_iter``, etc.). Bisect:
    train at ``budget=hi``, train at ``budget=hi/2``, train at ``budget=(hi+lo)/2``, ... pick
    the budget that minimises val loss. Cost: O(log(budget_max) * N) refits from scratch.
    Useful for expensive single-shot models without an iterative interface (Ridge, Lasso with
    LASSO-LARS, etc.). The wrapper picks the budget that minimised val loss, then refits ONE
    final time at that budget on the full train.

The wrapper exposes the standard sklearn estimator surface (``fit`` / ``predict`` /
``predict_proba`` / ``get_params`` / ``set_params``) so it drops into any pipeline that
takes an estimator. ``best_iter`` / ``best_metric`` / ``stopped_via`` are stored on the
wrapper after ``fit`` for diagnostics.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


def _split_train_val(X, y, val_size: float, random_state: int):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=val_size, random_state=random_state)


def _resolve_metric(metric: str | Callable | None, is_classification: bool) -> tuple[Callable, str, str]:
    """Return (fn, name, mode) where mode is 'min' or 'max' per is_greater_better."""
    if callable(metric):
        return metric, getattr(metric, "__name__", "custom"), "min"
    if metric in (None, ""):
        if is_classification:
            from sklearn.metrics import log_loss
            return (lambda y, p: log_loss(y, p, labels=np.unique(y))), "log_loss", "min"
        else:
            return (lambda y, p: float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))), "rmse", "min"
    if metric == "rmse":
        return (lambda y, p: float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))), "rmse", "min"
    if metric == "mae":
        return (lambda y, p: float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))), "mae", "min"
    if metric == "logloss":
        from sklearn.metrics import log_loss
        return (lambda y, p: log_loss(y, p)), "log_loss", "min"
    if metric == "auc":
        from sklearn.metrics import roc_auc_score
        return (lambda y, p: roc_auc_score(y, p)), "auc", "max"
    raise ValueError(f"Unknown metric: {metric}")


class PartialFitESWrapper:
    """Early-stopping wrapper for sklearn-style estimators without native ES support.

    Parameters
    ----------
    estimator
        Sklearn-compatible estimator. Wrapper picks ``partial_fit`` strategy if available,
        else ``budget_param`` dichotomic search.
    metric
        Either a string (``"rmse"``, ``"mae"``, ``"logloss"``, ``"auc"``), a ``callable``
        ``(y_true, y_pred) -> float`` (min-direction by default), or ``None`` for sensible
        defaults (log_loss for classification, RMSE for regression).
    patience
        Iterations without best-metric improvement before stopping.
    min_delta
        Minimum improvement to count as such.
    val_size
        Fraction of train data held out for ES val (when ``X_val``/``y_val`` not supplied).
    random_state
        Seed for the internal train/val split when one is needed.
    max_iter
        For ``partial_fit``: max number of epochs (each = one full pass over the train set).
        For dichotomic search: ignored; the search uses ``budget_param`` bounds instead.
    is_classification
        When True, ``predict_proba`` is used during evaluation if available; otherwise
        ``predict`` is used.
    budget_param
        Name of the integer hyperparameter (on ``estimator``) representing the iteration
        budget for the dichotomic search path (e.g. ``"n_estimators"`` for RF,
        ``"max_iter"`` for Ridge). Only used when the estimator lacks ``partial_fit``.
    budget_min, budget_max
        Bounds for the dichotomic search.
    worsening_enabled, worsening_coeff, worsening_min_iters
        Curve-shape detector knobs (forwarded to ``UniversalCallback``). Default ON.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        metric: str | Callable | None = None,
        patience: int = 10,
        min_delta: float = 0.0,
        val_size: float = 0.15,
        random_state: int = 42,
        max_iter: int = 200,
        is_classification: bool = False,
        budget_param: str | None = None,
        budget_min: int = 1,
        budget_max: int = 1000,
        worsening_enabled: bool = True,
        worsening_coeff: int = 5,
        worsening_min_iters: int = 5,
        verbose: int = 0,
        # When the caller (typically the mlframe training suite) already has a held-out val
        # set, passing it here means fit(X_tr, y_tr) -- WITHOUT the X_val=/y_val= kwargs --
        # still drives ES off the external val instead of re-splitting X_tr. Lets the wrapper
        # plug into ``model.fit(X, y, **fit_params)``-style calling conventions where the
        # caller can't easily inject X_val/y_val into the fit signature.
        external_X_val: Any = None,
        external_y_val: Any = None,
    ) -> None:
        self.estimator = estimator
        self.metric = metric
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.val_size = float(val_size)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.is_classification = bool(is_classification)
        self.budget_param = budget_param
        self.budget_min = int(budget_min)
        self.budget_max = int(budget_max)
        self.worsening_enabled = bool(worsening_enabled)
        self.worsening_coeff = int(worsening_coeff)
        self.worsening_min_iters = int(worsening_min_iters)
        self.verbose = int(verbose)
        self.external_X_val = external_X_val
        self.external_y_val = external_y_val
        # Filled after fit:
        self.best_iter: int | None = None
        self.best_metric: float | None = None
        self.stopped_via: str | None = None  # "partial_fit_es" | "dichotomic" | "max_iter_hit"
        self.history: list[float] = []
        self._fitted: bool = False

    # -- sklearn-style API ----------------------------------------------------

    def get_params(self, deep: bool = True) -> dict:
        return dict(estimator=self.estimator, metric=self.metric, patience=self.patience,
                    min_delta=self.min_delta, val_size=self.val_size,
                    random_state=self.random_state, max_iter=self.max_iter,
                    is_classification=self.is_classification, budget_param=self.budget_param,
                    budget_min=self.budget_min, budget_max=self.budget_max,
                    worsening_enabled=self.worsening_enabled,
                    worsening_coeff=self.worsening_coeff,
                    worsening_min_iters=self.worsening_min_iters, verbose=self.verbose)

    def set_params(self, **kw) -> "PartialFitESWrapper":
        for k, v in kw.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y, *, X_val=None, y_val=None, **_unused) -> "PartialFitESWrapper":
        # Resolution order for the val set used by the ES callback:
        #   1. explicit X_val= / y_val= kwargs win,
        #   2. external_X_val / external_y_val supplied at construction,
        #   3. internal train/val split via val_size.
        # ``**_unused`` swallows fit_params keys the suite may pass (eval_set, callbacks etc.)
        # that don't apply to non-native-ES estimators.
        if X_val is None or y_val is None:
            if self.external_X_val is not None and self.external_y_val is not None:
                X_tr, y_tr = X, y
                X_val, y_val = self.external_X_val, self.external_y_val
            else:
                X_tr, X_val, y_tr, y_val = _split_train_val(X, y, self.val_size, self.random_state)
        else:
            X_tr, y_tr = X, y
        metric_fn, metric_name, mode = _resolve_metric(self.metric, self.is_classification)

        if hasattr(self.estimator, "partial_fit"):
            self._fit_partial(X_tr, y_tr, X_val, y_val, metric_fn, metric_name, mode)
        elif self.budget_param is not None and hasattr(self.estimator, self.budget_param):
            self._fit_dichotomic(X_tr, y_tr, X_val, y_val, metric_fn, metric_name, mode)
        else:
            raise ValueError(
                f"Estimator {type(self.estimator).__name__} has no ``partial_fit`` and "
                f"no ``budget_param`` was supplied; cannot apply ES wrapper."
            )
        self._fitted = True
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        # Fallback for estimators with decision_function only (e.g. RidgeClassifier, LinearSVC).
        # Reporting/calibration callers expect a probabilistic surface; raising AttributeError
        # would force them to fall back to hard-label predict() which feeds class indices to
        # log_loss (values >1 in multiclass). Synthesise probs via sigmoid/softmax instead.
        if hasattr(self.estimator, "decision_function"):
            from scipy.special import expit, softmax
            dec = np.asarray(self.estimator.decision_function(X))
            if dec.ndim == 1:
                p = expit(dec)
                return np.column_stack([1.0 - p, p])
            return softmax(dec, axis=1)
        raise AttributeError("Underlying estimator has no predict_proba or decision_function")

    def score(self, X, y):
        return self.estimator.score(X, y)

    def __getattr__(self, name: str) -> Any:
        """Fall through to the wrapped estimator for attributes we don't explicitly own.

        Lets downstream code (feature-importance plotting, calibration, SHAP, etc.) read
        ``.coef_`` / ``.feature_importances_`` / ``.classes_`` / etc. without knowing the
        wrapper exists. ``__getattr__`` is only called when the normal attribute lookup
        fails so wrapper-owned attributes (best_iter, history, etc.) take precedence.
        """
        # Guard against early access during unpickling when self.estimator may not be set yet.
        if name == "estimator":
            raise AttributeError(name)
        est = self.__dict__.get("estimator")
        if est is None:
            raise AttributeError(name)
        return getattr(est, name)

    # -- internal: partial_fit strategy ---------------------------------------

    def _fit_partial(self, X_tr, y_tr, X_val, y_val, metric_fn, metric_name, mode) -> None:
        from ._callbacks import UniversalCallback
        # partial_fit on classification often needs the class list on first call.
        classes = np.unique(y_tr) if self.is_classification else None
        cb = UniversalCallback(
            patience=self.patience, min_delta=self.min_delta,
            monitor_dataset="val", monitor_metric=metric_name, mode=mode,
            worsening_enabled=self.worsening_enabled,
            worsening_coeff=self.worsening_coeff,
            worsening_min_iters=self.worsening_min_iters,
            worsening_max_iter=self.max_iter, verbose=0,
        )
        cb.start_time = time.time(); cb.last_reporting_ts = time.time(); cb.iter = 0
        for epoch in range(self.max_iter):
            if classes is not None and epoch == 0:
                self.estimator.partial_fit(X_tr, y_tr, classes=classes)
            else:
                self.estimator.partial_fit(X_tr, y_tr)
            val_pred = self._predict_for_eval(X_val)
            v = float(metric_fn(y_val, val_pred))
            self.history.append(v)
            cb.metric_history.setdefault("val", {}).setdefault(metric_name, []).append(v)
            if cb.should_stop():
                self.stopped_via = "curve_shape" if cb._worsening_stopped else "patience"
                break
        else:
            self.stopped_via = "max_iter_hit"
        self.best_iter = cb.best_iter
        self.best_metric = cb.best_metric
        if self.verbose > 0:
            logger.info("PartialFitESWrapper: stopped via %s at epoch %d, best=%s @ epoch %d",
                        self.stopped_via, len(self.history) - 1, self.best_metric, self.best_iter)

    def _predict_for_eval(self, X_val):
        """For classification with probabilistic metrics, prefer predict_proba."""
        if not self.is_classification:
            return self.estimator.predict(X_val)
        if hasattr(self.estimator, "predict_proba"):
            p = self.estimator.predict_proba(X_val)
            # Binary case: caller expects (n,) for AUC / log-loss positive-class
            if p.ndim == 2 and p.shape[1] == 2:
                return p[:, 1]
            return p
        if hasattr(self.estimator, "decision_function"):
            from scipy.special import expit, softmax
            dec = np.asarray(self.estimator.decision_function(X_val))
            return expit(dec) if dec.ndim == 1 else softmax(dec, axis=1)
        # Last-resort fallback: predict (hard labels) -- log_loss will warn but won't crash.
        return self.estimator.predict(X_val)

    # -- internal: dichotomic-search strategy ---------------------------------

    def _fit_dichotomic(self, X_tr, y_tr, X_val, y_val, metric_fn, metric_name, mode) -> None:
        """Bisect the budget hyperparameter to find the best val metric.

        Strategy:
          1. Evaluate at ``budget_max`` and at ``budget_min``. Both extremes scored.
          2. Bisect: test midpoint, drop the half whose endpoint is worse.
          3. Stop when the bracket shrinks below ``log_step_floor`` (default 1).
          4. Final fit at the winning budget on full train data.
        """
        lo, hi = self.budget_min, self.budget_max
        scores: dict[int, tuple[float, Any]] = {}  # budget -> (val_score, fitted_estimator)

        def score_at(budget: int) -> float:
            from sklearn.base import clone
            est = clone(self.estimator)
            est.set_params(**{self.budget_param: int(budget)})
            est.fit(X_tr, y_tr)
            if not self.is_classification:
                pred = est.predict(X_val)
            elif hasattr(est, "predict_proba"):
                p = est.predict_proba(X_val)
                pred = p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p
            elif hasattr(est, "decision_function"):
                # RidgeClassifier / LinearSVC: produce probabilities so log_loss never sees
                # raw class indices (which exceed [0,1] in multiclass and crash with
                # "y_prob contains values greater than 1: 2.0").
                from scipy.special import expit, softmax
                dec = np.asarray(est.decision_function(X_val))
                pred = expit(dec) if dec.ndim == 1 else softmax(dec, axis=1)
            else:
                pred = est.predict(X_val)
            v = float(metric_fn(y_val, pred))
            scores[budget] = (v, est)
            return v

        better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
        score_at(lo)
        score_at(hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            score_at(mid)
            # Drop the half whose endpoint is worse than the midpoint.
            if better(scores[lo][0], scores[hi][0]):
                hi = mid
            else:
                lo = mid

        # Winning budget = argmin/argmax score over all tested points
        comp_fn = min if mode == "min" else max
        best_budget = comp_fn(scores.keys(), key=lambda b: scores[b][0])
        best_score, best_est = scores[best_budget]
        self.estimator = best_est
        self.best_iter = int(best_budget)
        self.best_metric = float(best_score)
        self.stopped_via = "dichotomic"
        self.history = sorted(scores.items())  # type: ignore[assignment]
        if self.verbose > 0:
            logger.info("PartialFitESWrapper dichotomic: best %s = %s at %s=%d (tested %d budgets)",
                        metric_name, self.best_metric, self.budget_param, self.best_iter, len(scores))
