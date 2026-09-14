"""Early-stopping callback setup, carved out of ``_data_helpers.py`` to keep that module under the LOC budget.

Covers the dichotomic/partial-fit ES auto-wrap for sklearn-API models without native ES, the
GPU-CatBoost callback-skip probe, and ``_setup_early_stopping_callback`` itself (lgb/cb/xgb callback wiring).
"""

from __future__ import annotations

import logging
from typing import Any

from .callbacks import LightGBMCallback, CatBoostCallback, XGBoostCallback

try:
    from xgboost.callback import TrainingCallback as XGBTrainingCallback
except ImportError:
    XGBTrainingCallback = None  # type: ignore[assignment,misc]  # only used when xgboost is the chosen backend

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

# Model categories that already wire val/eval_set through ``_setup_eval_set`` and have
# their own native ES path (booster callbacks or sklearn-style validation_fraction).
# Models OUTSIDE this set get nothing val-driven by default; the auto-wrap helper below
# folds them into a ``PartialFitESWrapper`` so val is no longer wasted.
_NATIVE_ES_CATEGORIES: frozenset[str] = frozenset({"lgb", "hgb", "ngb", "cb", "xgb", "tabnet", "mlp"})

# Per-model budget parameter for the dichotomic-search ES strategy when the model lacks
# ``partial_fit``. ``None`` means no usable budget knob (e.g. plain LinearRegression which
# is closed-form -- no ES is possible at all, the wrapper degrades to a single fit-and-score).
_BUDGET_PARAM_BY_CATEGORY: dict[str, str | None] = {
    "ridge": "max_iter",
    "lasso": "max_iter",
    "elasticnet": "max_iter",
    "huber": "max_iter",
    "ransac": "max_trials",
    "linear": None,  # LinearRegression/LogisticRegression closed-form -- no budget
    # Composite registry entries (_trainer_configure.py): the generic ``get_params()`` probe below would
    # otherwise pick up ``n_estimators`` on ``BaggedCompositeEstimator`` (bag COUNT, not boosting rounds)
    # or ``CompositeClassificationEstimator`` (delegates to its inner LGBM's own natively-ES'd boosting
    # rounds) and dichotomic-search over it -- each candidate value re-fits the WHOLE composite (all N bags,
    # or the full base-margin + residual-boost chain), multiplying fit cost by the search's candidate count
    # on top of the composite's own internal per-member cost. Neither composite exposes a budget knob safe
    # for this wrapper's single-scalar dichotomic search; each already manages its own iteration budget
    # internally (per-bag native ES / base-margin residual boosting).
    "bagging": None,
    "composite_classification": None,
}


def _detect_budget_param(model_category: str, model_obj: Any) -> str | None:
    """Return the integer budget kwarg name on ``model_obj`` for dichotomic ES, or None.

    Priority: explicit per-category mapping first (so we don't accidentally pick the wrong
    knob on an estimator with multiple iterative params), then a runtime ``get_params``
    probe for common names.
    """
    # ``.get()`` + ``is not None`` can't tell "explicitly mapped to None" (no usable budget knob,
    # e.g. "linear"/"bagging"/"composite_classification") apart from "key absent" -- both look like
    # ``None`` -- so a category explicitly opted OUT of the runtime probe below would silently fall
    # through to it anyway. Use membership to distinguish the two cases.
    if model_category in _BUDGET_PARAM_BY_CATEGORY:
        return _BUDGET_PARAM_BY_CATEGORY[model_category]
    if model_obj is None:
        return None
    try:
        params = model_obj.get_params() if hasattr(model_obj, "get_params") else {}
    except Exception as exc:
        logger.debug("budget param probe: get_params() failed, no usable budget knob detected: %s", exc)
        return None
    for cand in ("max_iter", "n_estimators", "max_trials"):
        if cand in params and isinstance(params.get(cand), int):
            return cand
    return None


def maybe_wrap_for_partial_fit_es(
    model_obj: Any,
    *,
    model_category: str,
    X_val: Any,
    y_val: Any,
    is_classification: bool,
    behavior_kwargs: dict[str, Any] | None = None,
    random_state: int | None = None,
) -> tuple[Any, bool]:
    """Wrap a non-native-ES sklearn model in ``PartialFitESWrapper`` when feasible.

    Returns (possibly_wrapped_model, was_wrapped). The wrapper drives val-based ES via
    either ``partial_fit`` (preferred when available) or a dichotomic budget search on
    ``max_iter`` / ``n_estimators`` / ``max_trials``. Models in ``_NATIVE_ES_CATEGORIES``
    are passed through untouched (they already use val via ``_setup_eval_set``). Closed-
    form models with neither capability (plain ``LinearRegression``) are passed through
    too -- no ES is possible.

    Parameters
    ----------
    behavior_kwargs
        Optional dict carrying ``TrainingBehaviorConfig`` ES knobs (patience, min_delta,
        max_iter, budget bounds) forwarded to the wrapper.
    random_state
        Outer suite seed forwarded to the wrapper's internal train/val ES split so ES is
        reproducible per-seed and independent across seeds. ``None`` lets the split vary.
    """
    if model_obj is None or X_val is None or y_val is None:
        return model_obj, False
    if model_category in _NATIVE_ES_CATEGORIES:
        return model_obj, False
    # Already wrapped by a previous call (e.g. nested suite invocation) -- no-op.
    if type(model_obj).__name__ == "PartialFitESWrapper":
        return model_obj, False

    has_partial_fit = hasattr(model_obj, "partial_fit")
    budget_param = None if has_partial_fit else _detect_budget_param(model_category, model_obj)
    if not has_partial_fit and budget_param is None:
        # Closed-form / no usable knob -- nothing to early-stop.
        return model_obj, False

    from ._partial_fit_es_wrapper import PartialFitESWrapper

    kw = dict(behavior_kwargs or {})
    wrapper = PartialFitESWrapper(
        model_obj,
        metric=kw.pop("metric", None),
        patience=int(kw.pop("patience", 10)),
        min_delta=float(kw.pop("min_delta", 0.0)),
        max_iter=int(kw.pop("max_iter", 200)),
        is_classification=is_classification,
        random_state=random_state,
        budget_param=budget_param,
        budget_min=int(kw.pop("budget_min", 1)),
        budget_max=int(kw.pop("budget_max", 1000)),
        external_X_val=X_val,
        external_y_val=y_val,
        verbose=int(kw.pop("verbose", 0)),
    )
    return wrapper, True


def _detect_max_iter(model_category: str, model_obj: Any) -> int | None:
    """Best-effort extraction of the iteration budget from a sklearn-API booster.

    Used by the "best_iter hit max_iter" diagnostic (``UniversalCallback.max_iter``).
    Returns None when the budget is not discoverable.
    """
    if model_obj is None:
        return None
    try:
        params = model_obj.get_params() if hasattr(model_obj, "get_params") else {}
    except Exception as e:
        logger.debug("get_params() failed for %s, treating as empty: %s", type(model_obj).__name__, e)
        params = {}
    if model_category == "cb":
        return params.get("iterations") or params.get("n_estimators")
    if model_category in {"lgb", "xgb"}:
        return params.get("n_estimators")
    return None


def _build_cb_iteration_metrics_callback(fit_params, model_obj, stride):
    """Build the CatBoost per-iteration metric-capture callback from the val eval_set + model target type.

    Requires the build's ``callbacks=`` support and an eval_set in fit_params (the val Pool source). The callback's
    ``iteration_metrics_`` dict is bound by reference onto ``model_obj.iteration_metrics_`` at wiring time so the
    trajectory is readable on the fitted estimator without a post-fit stamp step (CatBoost callbacks have no
    after-training hook). Returns None when capture is not wireable (no eval_set / unsupported build / import fail).
    """
    from .callbacks.monotonic_decline import catboost_callbacks_supported

    if not catboost_callbacks_supported():
        return None
    eval_set = fit_params.get("eval_set")
    if not eval_set:
        return None
    pair = eval_set[0] if isinstance(eval_set, (list, tuple)) and eval_set and isinstance(eval_set[0], (list, tuple)) else eval_set
    try:
        X_val, y_val = pair[0], pair[1]
    except (TypeError, IndexError, KeyError):
        return None
    try:
        from sklearn.base import is_classifier as _sk_is_classifier
        from catboost import Pool

        from .callbacks.iteration_metrics import CBIterationMetricsCallback

        import numpy as _np

        if model_obj is not None and not _sk_is_classifier(model_obj):
            target_type, n_classes = "regression", None
        else:
            n_classes = int(_np.unique(_np.asarray(y_val)).shape[0])
            target_type = "binary_classification" if n_classes <= 2 else "multiclass_classification"
        val_pool = Pool(X_val, y_val)
        cb = CBIterationMetricsCallback(val_pool, y_val, target_type, stride=stride, n_classes=n_classes)
        if model_obj is not None:
            model_obj.iteration_metrics_ = cb.iteration_metrics_  # bound by reference; filled during fit
        return cb
    except Exception as exc:
        logger.debug("CatBoost iteration-metrics capture not wired: %s", exc)
        return None


def _cb_is_gpu(model_obj) -> bool:
    """True when ``model_obj`` is a CatBoost estimator configured for GPU training.

    CatBoost rejects ANY ``callbacks=`` list on GPU ("User defined callbacks are not supported for GPU") --
    every GPU CatBoost fit therefore used to attempt the CatBoostCallback/CBMonotonicDeclineStop wiring
    below, fail with a CatBoostError on the first attempt, and get silently retried without callbacks by
    ``_train_model_with_fallback``'s reactive fallback. That fallback still has to exist (a caller-supplied
    ``task_type`` can arrive by other paths this probe does not see), but checking here avoids paying for the
    guaranteed-to-fail first Pool build + fit attempt on the common path.
    """
    if model_obj is None:
        return False
    try:
        return str(model_obj.get_params().get("task_type", "")).upper() == "GPU"
    except Exception as exc:
        # Best-effort probe: an unreadable task_type just keeps the callback-wiring path (the reactive
        # fallback in _training_loop.py still catches a genuine GPU CatBoostError and retries), so this
        # never breaks a fit -- but it DOES silently defeat this function's whole optimization (paying for
        # the guaranteed-to-fail first attempt again) with no signal that it happened. Throttled warning
        # (this can run once per model in a suite) so a probe that starts failing (e.g. a CatBoost API
        # change to get_params()) is discoverable instead of a silent, permanent efficiency regression.
        log_throttle(
            logger, "cb_is_gpu_probe_failed", logging.WARNING,
            "_cb_is_gpu: task_type probe failed on %s (%s: %s); keeping the callback-wiring path.",
            type(model_obj).__name__, type(exc).__name__, exc,
        )
        return False


def _setup_early_stopping_callback(model_category, fit_params, callback_params, model_obj=None):
    """Set up early stopping callback for the given model category."""
    no_callback_list_models = {"xgb", "hgb", "ngb"}
    if model_category == "cb" and _cb_is_gpu(model_obj):
        no_callback_list_models = no_callback_list_models | {"cb"}

    if model_category not in no_callback_list_models:
        if "callbacks" not in fit_params:
            fit_params["callbacks"] = []

    # Auto-inject the booster's iteration budget so the "best_iter hit max_iter" diagnostic
    # can fire. Caller-provided ``max_iter`` wins if set.
    if isinstance(callback_params, dict) and callback_params.get("max_iter") is None:
        budget = _detect_max_iter(model_category, model_obj)
        if budget:
            callback_params = {**callback_params, "max_iter": int(budget)}

    # Pull the monotonic-decline patience out of callback_params (mirroring the lgb / xgb shims' own default)
    # before splatting the rest into the UniversalCallback subclass, which does not accept this kwarg.
    # ``None`` disables the fixed-N monotonic stop, leaving the booster's native detector.
    _mono_patience = 20
    if isinstance(callback_params, dict) and "monotonic_decline_patience" in callback_params:
        callback_params = dict(callback_params)
        _mono_patience = callback_params.pop("monotonic_decline_patience")

    # Pull the per-iteration metric-capture knobs out before splatting callback_params into the UniversalCallback
    # subclass (which does not accept them). Wired below for CatBoost (the lgb / xgb shims read them as fit kwargs).
    _cap_iter = False
    _iter_stride = 1
    if isinstance(callback_params, dict) and "capture_iteration_metrics" in callback_params:
        callback_params = dict(callback_params)
        _cap_iter = bool(callback_params.pop("capture_iteration_metrics"))
        _iter_stride = int(callback_params.pop("iteration_metrics_stride", 1))

    # None on a GPU CatBoost model (no callback wired at all -- CatBoost rejects the whole list on GPU), or
    # any other category the if/elif chain below does not recognise; guarded at the bottom before use.
    es_callback: LightGBMCallback | CatBoostCallback | XGBoostCallback | None = None
    if model_category == "lgb":
        es_callback = LightGBMCallback(**callback_params)
        fit_params["callbacks"].append(es_callback)
    elif model_category == "cb" and not _cb_is_gpu(model_obj):
        es_callback = CatBoostCallback(**callback_params)
        fit_params["callbacks"].append(es_callback)
        # Monotonic strict-decline stop for CatBoost (default-on) -- same shared rule as lgb / xgb / mlp.
        # Gated on a runtime probe of the installed build's ``callbacks=`` support; older builds fall back to
        # the native od_wait detector gracefully.
        if _mono_patience is not None:
            from .callbacks.monotonic_decline import CBMonotonicDeclineStop, catboost_callbacks_supported
            if catboost_callbacks_supported():
                fit_params["callbacks"].append(CBMonotonicDeclineStop(patience=_mono_patience))
        if _cap_iter:
            _cb = _build_cb_iteration_metrics_callback(fit_params, model_obj, _iter_stride)
            if _cb is not None:
                fit_params["callbacks"].append(_cb)
    elif model_category == "xgb" and model_obj is not None:
        es_callback = XGBoostCallback(**callback_params)
        existing_callbacks = model_obj.get_params().get("callbacks", []) or []
        # Keep only valid TrainingCallback instances, excluding stale XGBoostCallback instances.
        # This also filters out any legacy callbacks (e.g. from xgb_kwargs in XGB_GENERAL_PARAMS)
        # that do not inherit from xgboost.callback.TrainingCallback, which would cause a
        # TypeError in XGBoost >= 2.x where CallbackContainer validates isinstance strictly.
        callbacks = [cb for cb in existing_callbacks if isinstance(cb, XGBTrainingCallback) and not isinstance(cb, XGBoostCallback)]
        callbacks.append(es_callback)
        model_obj.set_params(callbacks=callbacks)
    if model_obj is not None and es_callback is not None:
        # Expose the per-iteration trajectory on the estimator for the run metadata. The containers are bound
        # BY REFERENCE at wiring time (the same idiom ``_build_cb_iteration_metrics_callback`` uses for
        # ``iteration_metrics_``) so they fill during fit with no post-fit harvest step. Recorded regardless
        # of whether the widget drew it or the log printed it, which is what makes ``live_trainperf_report``
        # default to False without losing anything.
        try:
            model_obj._mlframe_es_callback = es_callback
            model_obj.training_curves_ = {
                "iterations": es_callback.iter_history,
                "metrics": es_callback.metric_history,
                "ram_gb": es_callback.ram_history,
            }
        except AttributeError:
            pass  # best-effort: an estimator with __slots__ simply does not carry the trajectory
