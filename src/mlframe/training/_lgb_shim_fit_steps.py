"""Step helpers for ``_DatasetReuseMixin.fit`` in ``lgb_shim.py``, split out to keep that method under the function-length gate."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._lgb_shim_helpers import (
    _build_dataset,
    _lgb_cache_get,
    _lgb_cache_put,
    _reset_weight_to_uniform,
)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore[assignment]

logger = logging.getLogger("mlframe.training.lgb_shim")


def _without_stale_init_score(dtrain: Any, source: str, init_score: Any) -> "tuple[Any, str]":
    """The cached train Dataset and its source, or ``(None, "miss")`` when this fit has no init_score but it carries one.

    LightGBM cannot clear an init_score (set_init_score(None) keeps the old array), and explicit zeros are not "no
    init_score": they switch off boost_from_average, so the same fit trained a different booster on a cache hit than
    on a miss.
    """
    if dtrain is not None and init_score is None and dtrain.get_init_score() is not None:
        return None, "miss"
    return dtrain, source


def _reuse_cached_train_dataset(est: Any, dtrain: Any, train_key: Any, train_source: str, y_for_fit: Any, sample_weight: Any, init_score: Any) -> None:
    """Re-apply label / weight / init_score on a cached train Dataset and promote it to the instance cache."""
    # Validate that label cardinality matches the cached Dataset
    # before any set_label/set_weight: cache hit on ``X`` does NOT
    # guarantee ``y`` has the same length (caller can pass a
    # filtered y while keeping the cached frame). Without this
    # guard, LightGBM accepts the mismatched label silently and
    # downstream set_weight(ones(num_data)) crashes - or worse,
    # produces aligned-by-position garbage when sample_weight is
    # supplied at the wrong length.
    _y_arr = np.asarray(y_for_fit)
    if _y_arr.shape[0] != dtrain.num_data():
        raise ValueError(
            f"lgb_shim: cached train Dataset has "
            f"{dtrain.num_data()} rows but y has "
            f"{_y_arr.shape[0]} rows. Invalidate the cache before "
            f"refitting (call ._init_cache())."
        )
    dtrain.set_label(_y_arr)
    if sample_weight is not None:
        _w_arr = np.asarray(sample_weight)
        if _w_arr.shape[0] != dtrain.num_data():
            raise ValueError(f"lgb_shim: sample_weight length " f"{_w_arr.shape[0]} != Dataset.num_data() " f"{dtrain.num_data()}")
        dtrain.set_weight(_w_arr)
    else:
        # When sample_weight was set previously and now None is
        # passed, "uniform weights" is the desired semantic. Uses
        # _reset_weight_to_uniform (set_field, not set_weight) --
        # plain set_weight(ones) silently NO-OPS when a real prior
        # weight is already set at the C++ side (see that helper's
        # docstring), which would otherwise leave the stale weight
        # in place despite the caller asking for uniform.
        _reset_weight_to_uniform(dtrain)
    # set_label/set_weight were re-applied above on a cache hit, but set_init_score was
    # not -- a subsequent .fit(X, y, init_score=new_value) call reusing this cached
    # Dataset silently trained against the STALE (or entirely absent) init_score baked in
    # from whichever earlier .fit() call first built it, instead of the new one. Always
    # re-apply on every cache hit, same as label/weight: an explicit new init_score
    # overwrites. A fit without one reaches here only when the cached Dataset has none
    # either (see the rebuild above), so there is nothing to clear.
    # Multiclass boosters need init_score flattened to num_data() * num_class rows (LightGBM's
    # C++ side raises "Number of class for initial score error" for anything shorter) -- a
    # binary/regression booster's init_score is exactly num_data() long. _n_classes is set by
    # _pre_fit_bookkeeping above (classifier only; absent/<=2 means not multiclass).
    _ncls_for_init = getattr(est, "_n_classes", None)
    _expected_init_len = dtrain.num_data() * _ncls_for_init if _ncls_for_init is not None and _ncls_for_init > 2 else dtrain.num_data()
    if init_score is not None:
        _init_arr = np.asarray(init_score)
        if _init_arr.shape[0] != _expected_init_len:
            raise ValueError(f"lgb_shim: init_score length " f"{_init_arr.shape[0]} != expected " f"{_expected_init_len}")
        dtrain.set_init_score(_init_arr)

    # Promote a module-cache hit to instance-level for the next call on this instance.
    est._cached_train_dataset = dtrain
    est._cached_train_key = train_key
    logger.debug(
        "[lgb-shim] reused %s cached train Dataset (id=%d)",
        train_source, id(dtrain),
    )


def _prepare_train_dataset(
    est: Any,
    X: Any,
    y_for_fit: Any,
    sample_weight: Any,
    init_score: Any,
    categorical_feature: Any,
    feature_name: Any,
    train_key: Any,
) -> Any:
    """Return the train Dataset for ``est.fit``: reused from the instance or module cache (inputs re-applied) or freshly built and cached."""
    dtrain = None
    _train_source = "miss"
    if est._cached_train_key == train_key and est._cached_train_dataset is not None:
        dtrain = est._cached_train_dataset
        _train_source = "instance"
    else:
        _global_train_hit = _lgb_cache_get(train_key)
        if _global_train_hit is not None:
            dtrain = _global_train_hit
            _train_source = "module"
    dtrain, _train_source = _without_stale_init_score(dtrain, _train_source, init_score)
    if dtrain is not None:
        _reuse_cached_train_dataset(est, dtrain, train_key, _train_source, y_for_fit, sample_weight, init_score)
    else:
        dtrain = _build_dataset(
            X, y_for_fit, sample_weight,
            categorical_feature=categorical_feature,
            feature_name=feature_name,
            init_score=init_score,
        )
        est._cached_train_dataset = dtrain
        est._cached_train_key = train_key
        _lgb_cache_put(train_key, dtrain)
        logger.debug(
            "[lgb-shim] built fresh train Dataset (id=%d); stored in module cache for cross-clone reuse",
            id(dtrain),
        )
    return dtrain


def _build_train_callbacks(callbacks: Any, valid_sets: list[Any], evals_result: dict, monotonic_decline_patience: Any) -> "tuple[list[Any], Any]":
    """Assemble the lgb.train callback list and return it with the monotonic-decline callback (None when absent)."""
    train_callbacks: list[Any] = list(callbacks) if callbacks else []
    if valid_sets:
        train_callbacks.append(lgb.record_evaluation(evals_result))
        # Default-on monotonic strict-decline overfitting stop, COMPLEMENTARY to native
        # early_stopping_rounds: stop once the first val set's metric strictly worsens for
        # ``monotonic_decline_patience`` consecutive rounds since the best. Skipped when the
        # caller already supplied one, or when disabled (None). The native best-iteration
        # rollback (EarlyStopException) keeps the global-best booster.
        from .callbacks.monotonic_decline import LGBMonotonicDeclineStop

        if monotonic_decline_patience is not None and not any(isinstance(cb, LGBMonotonicDeclineStop) for cb in train_callbacks):
            train_callbacks.append(LGBMonotonicDeclineStop(patience=monotonic_decline_patience))
        # Kept for the post-fit stop-reason log below: the ONLY way to tell "the monotonic-decline
        # detector stopped this fit" apart from "native early_stopping_round did" from the outside,
        # since both roll the booster back to the same best_iteration and neither is otherwise
        # distinguishable from iteration count alone.
        _mono_cb_ref = next((cb for cb in train_callbacks if isinstance(cb, LGBMonotonicDeclineStop)), None)
    else:
        _mono_cb_ref = None
    return train_callbacks, _mono_cb_ref


def _make_iter_metrics_callback(est: Any, Xval: Any, yval: Any, stride: Any) -> Any:
    """Build the per-iteration full-metric-suite capture callback for the raw first eval pair."""
    from .callbacks.iteration_metrics import LGBIterationMetricsCallback
    from sklearn.base import is_classifier as _sk_is_classifier
    _ncls = getattr(est, "_n_classes", None)
    if not _sk_is_classifier(est):
        _tt = "regression"
    elif _ncls is not None and _ncls > 2:
        _tt = "multiclass_classification"
    else:
        _tt = "binary_classification"
    return LGBIterationMetricsCallback(Xval, yval, _tt, stride=int(stride), n_classes=_ncls)


def _attach_booster(
    est: Any,
    booster: Any,
    evals_result: dict,
    valid_sets: list[Any],
    n_estimators: Any,
    mono_cb_ref: Any,
    monotonic_decline_patience: Any,
    iter_metrics_cb: Any,
) -> None:
    """Attach the trained Booster plus the sklearn-convention bookkeeping attributes to ``est``."""
    est._Booster = booster
    est._n_features = booster.num_feature()
    est._n_features_in = est._n_features
    est._evals_result = evals_result
    est._best_iteration = booster.best_iteration
    est._best_score = booster.best_score
    # Which of the two independent stop mechanisms actually ended this fit -- the monotonic-decline
    # detector already logs its own firing (above), so this only adds the case that had NO log line at
    # all: LightGBM's native early_stopping_round (silent unless verbosity is turned up), and the
    # trivial "ran to the full budget" case. Read from valid_sets so a train-only fit (no eval data,
    # nothing CAN have stopped it early) stays silent instead of stating a reason for a fit that had
    # neither ES mechanism wired at all.
    if valid_sets and booster.best_iteration and int(booster.best_iteration) < int(n_estimators) - 1:
        if mono_cb_ref is not None and getattr(mono_cb_ref, "fired", False):
            pass  # already logged inline by LGBMonotonicDeclineStop itself
        else:
            logger.info(
                "[lgb] stopped at iteration %d of %d requested: native early_stopping_round fired "
                "(monotonic-decline patience=%s did not trigger first).",
                booster.best_iteration, n_estimators, monotonic_decline_patience,
            )
    if iter_metrics_cb is not None:
        est.iteration_metrics_ = iter_metrics_cb.iteration_metrics_
    # ``fitted_`` is the flag ``__sklearn_is_fitted__`` checks --
    # without it predict raises NotFittedError even though _Booster
    # is set. Mirror of LGBMModel.fit's final state-flip line.
    est.fitted_ = True

    # ``_class_weight`` is consumed by predict_proba (multiclass
    # path) and by some sample weighting helpers; set it to a safe
    # default if pre_fit_bookkeeping didn't already.
    if not hasattr(est, "_class_weight"):
        est._class_weight = getattr(est, "class_weight", None)
