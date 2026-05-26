"""Refit helpers extracted from ``_training_loop.py``.

Two narrow refit policies:

- ``_maybe_refit_on_degenerate_best_iter`` -- if a CatBoost / LGB / XGB
  fit converged at ``best_iter < threshold`` under a robust loss
  (Huber/MAE/L1/quantile), refit with the RMSE-family default.
- ``_maybe_refit_on_collapsed_predictions`` -- if a Lightning MLP /
  recurrent regressor emits near-constant predictions on the train set,
  refit with ``output_activation='linear'``.

Carved out to drop the parent ``_training_loop.py`` below the 1k-LOC
monolith threshold; re-exported from the parent so existing callers
keep working unchanged.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mlframe.core.helpers import get_model_best_iter

_MIN_BEST_ITER_HEALTHY: int = 3
"""Absolute floor below which we consider a booster to have failed to learn (constant prediction, ES at iter=0/1)."""

_MIN_BEST_ITER_FRACTION: float = 0.05
"""``best_iter`` MUST also be below this fraction of the configured ``max_iterations`` to trigger a retry. Protects users who deliberately request a tiny budget (``iterations=2`` etc) from getting their fit silently re-run on a different loss. Example: max_iter=2 -> threshold = 0.1 -> best_iter < 0.1 never fires; max_iter=1000 -> threshold = 50 -> best_iter=2 fires (and is also < 3, the absolute floor)."""

_BOOSTING_FAMILIES: tuple[str, ...] = ("CatBoost", "LGB", "XGB")

# Per-backend parameter name carrying the max-iterations budget.
# Used to derive the adaptive threshold so we don't refit users who
# explicitly chose a tiny iteration budget.
_MAX_ITER_PARAM: dict[str, tuple[str, ...]] = {
    "CatBoost": ("iterations", "n_estimators", "num_boost_round"),
    "LGB": ("n_estimators", "num_boost_round", "num_iterations"),
    "XGB": ("n_estimators", "num_boost_round"),
}

# Per-backend RMSE-family fallback (the most stable training surface).
_RMSE_FALLBACK: dict[str, tuple[str, str, str, str]] = {
    "CatBoost": ("loss_function", "RMSE", "eval_metric", "RMSE"),
    "LGB": ("objective", "regression", "metric", "l2"),
    "XGB": ("objective", "reg:squarederror", "eval_metric", "rmse"),
}

# Lowercase substrings that indicate the current loss is a robust /
# non-default surface (Huber / MAE / L1 / quantile). Only models
# currently on such a loss are refit candidates -- a legitimate
# iter=2 convergence under RMSE is left alone.
_NON_DEFAULT_LOSS_TOKENS: tuple[str, ...] = (
    "huber", "mae", "absoluteerror", "_l1", "pseudohuber", "quantile",
)


def _maybe_refit_on_degenerate_best_iter(
    *,
    model_obj: Any,
    model_type_name: str,
    best_iter: int,
    train_df: Any,
    train_target: Any,
    fit_params: dict[str, Any],
    logger_: logging.Logger,
) -> int | None:
    """Detect degenerate ES + refit booster with RMSE-family default.

    Returns the post-refit ``best_iteration`` when a refit happened,
    else ``None`` (callers keep their original best_iter). The
    in-place ``set_params`` + ``fit`` is the same lifecycle the trainer
    used for the first fit so downstream code (predict, save) is
    transparent.
    """
    _backend_prefix = next(
        (p for p in _BOOSTING_FAMILIES if model_type_name.startswith(p)),
        None,
    )
    if _backend_prefix is None:
        return None
    _fallback = _RMSE_FALLBACK.get(_backend_prefix)
    if _fallback is None:
        return None
    _loss_key, _loss_val, _metric_key, _metric_val = _fallback
    try:
        _cur_params = model_obj.get_params() if hasattr(model_obj, "get_params") else {}
    except Exception:
        _cur_params = {}
    # Adaptive threshold: respect the user's iteration budget. We only
    # treat best_iter as degenerate if BOTH (a) it is absolutely small
    # AND (b) it is a tiny fraction of the configured max-iterations.
    # A user who set ``iterations=2`` and got best_iter=1 chose that
    # budget deliberately; we must not silently swap their loss.
    _max_iter = None
    for _max_key in _MAX_ITER_PARAM.get(_backend_prefix, ()):
        _v = _cur_params.get(_max_key)
        if _v is not None:
            try:
                _max_iter = int(_v)
                break
            except (TypeError, ValueError):
                continue
    _adaptive_floor = max(1, int(_max_iter * _MIN_BEST_ITER_FRACTION)) if _max_iter else _MIN_BEST_ITER_HEALTHY
    _threshold = min(_MIN_BEST_ITER_HEALTHY, _adaptive_floor)
    if best_iter >= _threshold:
        return None
    _cur_loss = str(_cur_params.get(_loss_key, "")).lower()
    if not any(tok in _cur_loss for tok in _NON_DEFAULT_LOSS_TOKENS):
        return None
    logger_.warning(
        "[loss-fallback] %s training degenerate (best_iter=%d < %d; "
        "max_iter=%s, adaptive_floor=%d) under loss=%r. Refitting with "
        "%s=%r + %s=%r. Heavy-kurt targets can collapse the Huber/L1 "
        "gradient; RMSE is the stable training surface even if less "
        "robust to outliers downstream.",
        model_type_name, int(best_iter), _threshold,
        str(_max_iter) if _max_iter is not None else "?", _adaptive_floor,
        _cur_loss, _loss_key, _loss_val, _metric_key, _metric_val,
    )
    try:
        model_obj.set_params(**{_loss_key: _loss_val, _metric_key: _metric_val})
        model_obj.fit(train_df, train_target, **fit_params)
    except (ValueError, TypeError) as _refit_err:
        logger_.warning(
            "[loss-fallback] %s refit with RMSE rejected: %s. Keeping the "
            "degenerate fit; downstream charts will show the truth "
            "(R^2 < 0, near-constant pred).",
            model_type_name, _refit_err,
        )
        return None
    try:
        _new_best_iter = get_model_best_iter(model_obj)
    except (AttributeError, TypeError, ValueError):
        _new_best_iter = None
    logger_.warning(
        "[loss-fallback] %s refit complete: best_iter %d -> %s.",
        model_type_name, int(best_iter),
        str(_new_best_iter) if _new_best_iter is not None else "?",
    )
    return _new_best_iter


_COLLAPSED_PRED_STD_FRACTION: float = 0.1
"""Threshold on ``predict(X_train).std() / y_train.std()``. Below this the model is essentially a constant predictor (no signal learned). Architecture-agnostic -- catches MLP / recurrent / boost collapse modes that don't surface as ``best_iter < 3``."""

_MIN_MAX_EPOCHS_FOR_RETRY: int = 3
"""If the user explicitly requested ``max_epochs <= 2`` we treat the collapse as intentional (tiny-budget run) and skip the refit. Symmetric to the ``_MIN_BEST_ITER_FRACTION`` gate on the booster path."""


def _maybe_refit_on_collapsed_predictions(
    *,
    model: Any,
    model_obj: Any,
    model_type_name: str,
    train_df: Any,
    train_target: Any,
    fit_params: dict[str, Any],
    logger_: logging.Logger,
) -> bool:
    """Detect near-constant prediction collapse on Lightning MLP /
    recurrent regressors and refit with the bounded output activation
    removed.

    Production failure mode (2026-05-26): the BN-equipped MLP defaults
    use ``output_activation='tanh_train_range'`` to hard-cap regression
    outputs. On extreme-AR / extreme-kurt targets the inner pre-
    activation saturates and the destandardised prediction collapses
    to a single rail (observed R^2 = -30 with predictions clustered at
    +-scale around y_train_mean). The booster ``best_iter < 3``
    detector doesn't catch this -- Lightning has no ``best_iteration``
    attribute on regression heads.

    Detection: ``predict(X_train).std() / y_train.std() <
    _COLLAPSED_PRED_STD_FRACTION``. Architecture-agnostic; any model
    that ends up emitting a near-constant prediction triggers.

    Retry strategy (regression-only): set ``network_params['output_-
    activation'] = 'linear'`` on the inner Lightning estimator and
    refit. The bounded output is the most common collapse vector on
    this codebase; removing it lets the inner net find a non-trivial
    fit. If that also fails, the chart still shows the (bad) refit so
    the operator sees truthful R^2 < 0.

    Skipped when:
    - Train target is not 1-D (multi-output / classification heads).
    - Train target std is zero (degenerate; pred ratio undefined).
    - Model has no ``network_params`` attribute (non-Lightning path).
    - Lightning ``max_epochs <= 2`` (user chose a tiny budget on
      purpose; symmetric to ``_MIN_BEST_ITER_FRACTION`` gate on
      boosters).
    """
    try:
        _y = np.asarray(train_target)
    except Exception:
        return False
    if _y.ndim != 1 or _y.size < 10:
        return False
    _y_finite = _y[np.isfinite(_y)]
    if _y_finite.size < 10:
        return False
    _y_std = float(_y_finite.std())
    if _y_std <= 0.0:
        return False
    # ``network_params`` lives on the Lightning estimator wrapped by
    # ``CompositeTargetEstimator`` / ``_TTRWithEvalSetScaling`` /
    # Pipeline; walk the common nesting paths to find it.
    _inner = model_obj
    _visited: set[int] = set()
    while _inner is not None and id(_inner) not in _visited:
        _visited.add(id(_inner))
        if hasattr(_inner, "network_params") and isinstance(getattr(_inner, "network_params"), dict):
            break
        _next = None
        for _attr in ("regressor_", "regressor", "estimator_", "estimator"):
            _candidate = getattr(_inner, _attr, None)
            if _candidate is not None and id(_candidate) not in _visited:
                _next = _candidate
                break
        if _next is None and hasattr(_inner, "named_steps"):
            for _step in _inner.named_steps.values():
                if id(_step) not in _visited:
                    _next = _step
                    break
        _inner = _next
    if _inner is None or not hasattr(_inner, "network_params"):
        return False
    # Tiny-budget gate.
    _max_epochs = None
    if hasattr(_inner, "trainer_params") and isinstance(_inner.trainer_params, dict):
        _v = _inner.trainer_params.get("max_epochs")
        if _v is not None:
            try:
                _max_epochs = int(_v)
            except (TypeError, ValueError):
                _max_epochs = None
    if _max_epochs is not None and _max_epochs < _MIN_MAX_EPOCHS_FOR_RETRY:
        return False
    # Predict on train and check variance ratio.
    try:
        _preds = np.asarray(model.predict(train_df)).reshape(-1)
    except Exception:
        return False
    if _preds.size != _y.size:
        return False
    _preds_finite = _preds[np.isfinite(_preds)]
    if _preds_finite.size < 10:
        return False
    _pred_std = float(_preds_finite.std())
    _ratio = _pred_std / _y_std
    if _ratio >= _COLLAPSED_PRED_STD_FRACTION:
        return False
    # Only swap output_activation if it's currently the bounded variant
    # (linear default doesn't have anywhere to go; user-chosen non-
    # linear activations may be intentional).
    _net_params = dict(_inner.network_params)
    _cur_act = str(_net_params.get("output_activation", "linear"))
    if _cur_act == "linear":
        logger_.warning(
            "[pred-collapse] %s prediction collapse detected "
            "(pred_std/y_std=%.4f < %.2f) but output_activation already "
            "'linear'; no refit knob available. Downstream chart will "
            "show the truth (near-constant predictions).",
            model_type_name, _ratio, _COLLAPSED_PRED_STD_FRACTION,
        )
        return False
    logger_.warning(
        "[pred-collapse] %s prediction collapse detected "
        "(pred_std/y_std=%.4f < %.2f; max_epochs=%s) under "
        "output_activation=%r. Refitting with output_activation='linear' "
        "(removed bound). Tanh_train_range can saturate inner pre-"
        "activations on extreme-AR / extreme-kurt targets.",
        model_type_name, _ratio, _COLLAPSED_PRED_STD_FRACTION,
        str(_max_epochs) if _max_epochs is not None else "?", _cur_act,
    )
    _net_params["output_activation"] = "linear"
    _net_params.pop("output_activation_scale", None)
    _net_params.pop("output_activation_center", None)
    try:
        _inner.network_params = _net_params
        # Reset the trained network so the next fit rebuilds from scratch.
        if hasattr(_inner, "network"):
            try:
                _inner.network = None
            except Exception:
                pass
        model_obj.fit(train_df, train_target, **fit_params)
    except Exception as _refit_err:
        logger_.warning(
            "[pred-collapse] %s refit with output_activation='linear' "
            "failed: %s. Keeping the collapsed fit; downstream chart "
            "will show the truth.",
            model_type_name, _refit_err,
        )
        return False
    # Sanity-check post-refit prediction spread.
    try:
        _preds2 = np.asarray(model.predict(train_df)).reshape(-1)
        _preds2_finite = _preds2[np.isfinite(_preds2)]
        _ratio2 = float(_preds2_finite.std()) / _y_std if _preds2_finite.size >= 10 else 0.0
    except Exception:
        _ratio2 = 0.0
    logger_.warning(
        "[pred-collapse] %s refit complete: pred_std/y_std %.4f -> %.4f.",
        model_type_name, _ratio, _ratio2,
    )
    return True
