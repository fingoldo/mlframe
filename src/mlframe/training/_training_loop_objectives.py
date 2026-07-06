"""Training loop helpers extracted from ``trainer.py``.

Core training path: model fitting with CatBoost/LGB/XGB fallbacks,
early stopping, OOM recovery, and post-hoc probability calibration.
"""

from __future__ import annotations

import logging

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


# Refit helpers + their module-level constants moved to sibling
# ``_training_loop_refit.py`` to drop this file below the 1k-LOC
# monolith threshold; imported here so callers keep using
# ``from mlframe.training._training_loop import _maybe_refit_on_*``.
from ._training_loop_refit import (  # noqa: F401
    _maybe_refit_on_collapsed_predictions,
    _maybe_refit_on_degenerate_best_iter,
)

logger = logging.getLogger(__name__)


def _ensure_xgb_classification_objective(model, train_target) -> None:
    """When XGB reaches .fit() without objective matching target shape, set it pre-fit.
    XGB sklearn wrapper auto-fills binary:logistic; for multiclass/multilabel that
    produces N*K preds vs N labels -> Invalid shape. No-op when objective already
    multiclass/multilabel, model isn't XGB classifier, or target inspection fails."""
    if model is None:
        return
    _mt = type(model).__name__
    if "XGB" not in _mt or "Classifier" not in _mt:
        return
    try:
        params = model.get_params() if hasattr(model, "get_params") else {}
    except Exception as _params_exc:
        logger.debug("_ensure_xgb_classification_objective: get_params() raised %s; skipping objective adjust.", _params_exc)
        return
    obj = params.get("objective")
    if isinstance(obj, str) and ("multi" in obj or "multilabel" in obj):
        return
    arr = np.asarray(train_target) if train_target is not None else None
    if arr is None:
        return
    if arr.dtype == object and arr.ndim == 1 and arr.shape[0] > 0:
        _first = arr[0]
        if hasattr(_first, "shape") or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes))):
            try:
                arr = np.stack([np.asarray(c) for c in arr], axis=0)
            except Exception as _stack_exc:
                logger.debug("_ensure_xgb_classification_objective: object-dtype stack raised %s; skipping.", _stack_exc)
                return
    if arr.ndim == 2 and arr.shape[1] >= 2:
        model.set_params(objective="binary:logistic", eval_metric="logloss")
    elif arr.ndim == 1:
        n_unique = len(np.unique(arr))
        if n_unique > 2:
            model.set_params(objective="multi:softprob", num_class=n_unique)

def _maybe_wrap_for_2d_target(model, train_target):
    """When ``train_target`` is 2-D ``(N, K)`` but ``model`` is a single-
    output sklearn classifier (e.g., ``LogisticRegression``), wrap it
    with ``MultiOutputClassifier`` inline at fit time. The upstream
    ``_wrap_for_multilabel_if_needed`` guard SHOULD already have wrapped
    via the strategy hook, but a few combos slip through (e.g., 3-way
    fuzz c0008: cb_linear / multilabel / heavy preprocessing). Acts as a
    last-line defence so the fit doesn't raise sklearn's
    ``y should be a 1d array, got an array of shape (N, K)``.

    Returns the (possibly wrapped) model. No-op for already-wrapped
    estimators (MultiOutputClassifier / ClassifierChain / CB / XGB /
    HGB / LGB), regression targets, or 1-D targets.
    """
    if model is None or train_target is None:
        return model
    arr = np.asarray(train_target)
    if arr.ndim != 2:
        return model
    _mt = type(model).__name__
    # Already a multi-output wrap or a strategy with native 2-D support.
    if _mt in (
        "MultiOutputClassifier",
        "MultiOutputRegressor",
        "ClassifierChain",
        "RegressorChain",
        "_ChainEnsemble",
    ):
        return model
    if _mt in ("CatBoostClassifier", "CatBoostRegressor"):
        return model  # CB native multilabel via MultiLogloss
    if _mt in ("PytorchLightningClassifier", "PytorchLightningRegressor"):
        # MLP supports multilabel natively via per-label sigmoid +
        # BCEWithLogitsLoss (NeuralNetStrategy.supports_native_
        # multilabel=True). Output layer is K units, predict_step applies
        # sigmoid when task_type='multilabel'. No wrap needed.
        return model
    # NOTE: RidgeClassifier / RidgeClassifierCV technically accept 2-D y
    # natively (sklearn quirk: multi-output ridge regression internally),
    # but the eval-pipeline expects predict_proba which RidgeClassifier
    # lacks. Until eval is generalised to fall back to decision_function
    # for multilabel, leave Ridge in the wrapper path. Tracked in
    # LinearModelStrategy docstring.
    # HGB / LGB / Linear / XGB sklearn wrappers all require 1-D y for
    # classifiers. XGB has native multi-output via the binary:logistic
    # loss but the sklearn wrapper still rejects 2-D y; only the native
    # ``xgb.train()`` API takes (N, K) labels directly. So wrap every
    # non-CB classifier inline as the safe last-line defense (the
    # upstream ``_wrap_for_multilabel_if_needed`` strategy hook already
    # wraps these in the happy path; this guard catches the combos that
    # slip through). Surfaced 3-way fuzz c0036 (cb_hgb_lgb_linear_xgb /
    # pl_enum / multilabel) - HGB raised ``y should be a 1d array, got
    # an array of shape (N, K)`` before this guard.
    try:
        from sklearn.multioutput import MultiOutputClassifier

        # Disable inner-estimator features that depend on a held-out
        # eval / val split: ``MultiOutputClassifier`` splits ``y`` per
        # label but does NOT propagate eval_set / val_data, so
        # early-stopping-via-internal-split paths in HGB / LGB collide
        # with the per-label fit. Concretely: HGB with
        # ``early_stopping=True`` + ``validation_fraction=None`` calls
        # ``_check_classification_targets`` on its own internal val
        # fold's y, which the inner-clone receives as the sliced 1-D
        # column - except that inner check then passes the FULL
        # train+val (N, K) along into a deeper sklearn validation step
        # that does ``column_or_1d`` on it. Easiest fix: turn off the
        # inner's early stopping so no internal val split happens.
        try:
            inner_params = model.get_params() if hasattr(model, "get_params") else {}
            patch = {}
            # HGB: ``early_stopping=True`` + no val => internal split on
            # 2-D y is incompatible with the per-label MOC slice.
            if "early_stopping" in inner_params and inner_params.get("early_stopping"):
                patch["early_stopping"] = False
            # LGB: callbacks (e.g. ``early_stopping(rounds=N)``) require
            # an eval_set, but MOC cannot pass eval_set per label, so the
            # callback raises ``For early stopping, at least one dataset
            # and eval metric is required for evaluation``. Drop the
            # callbacks so the per-label fit completes without
            # early-stopping.
            if "callbacks" in inner_params and inner_params.get("callbacks"):
                patch["callbacks"] = None
            # XGB / LGB ``early_stopping_rounds`` keyword: same issue,
            # the per-label fit has no eval_set to evaluate against.
            if "early_stopping_rounds" in inner_params and inner_params.get("early_stopping_rounds"):
                patch["early_stopping_rounds"] = None
            if patch:
                model.set_params(**patch)
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in _training_loop_objectives.py:156: %s", e)
            pass
        return MultiOutputClassifier(model, n_jobs=1)
    except ImportError as _import_err:
        # sklearn missing or version skew. Wrapping is impossible -> caller's bare model
        # will explode at fit with "y should be a 1d array, got (N, K)" but at least the
        # operator will see the underlying ImportError explanation rather than the cryptic
        # downstream message. Surface the failure rather than silently returning the
        # unwrapped model, which would defeat this whole guard block.
        logger.error(
            "_maybe_wrap_for_multilabel: sklearn.multioutput.MultiOutputClassifier "
            "import failed (%s); returning bare model unwrapped. Caller fit will likely "
            "raise on (N, K) target shape -- upgrade scikit-learn or supply CatBoost.",
            _import_err,
        )
        return model
