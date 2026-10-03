"""Helpers carved out of ``_training_loop`` to keep that module under its size budget."""

from __future__ import annotations

import contextlib
import logging

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


from mlframe.config import CATBOOST_MODEL_TYPES

# Refit helpers + their module-level constants moved to sibling
# ``_training_loop_refit.py`` to drop this file below the 1k-LOC
# monolith threshold; imported here so callers keep using
# ``from mlframe.training._training_loop import _maybe_refit_on_*``.

logger = logging.getLogger(__name__)


def _in_interactive_notebook() -> bool:
    """True only inside an IPython/Jupyter kernel where CB's live plot makes sense."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and type(ip).__name__ == "ZMQInteractiveShell"
    except Exception as exc:
        logger.debug("_in_interactive_notebook: IPython probe failed, assuming non-interactive: %s", exc)
        return False


@contextlib.contextmanager
def _suppress_catboost_noise():
    """Silence CatBoost's own cosmetic UserWarnings for the duration of a fit.

    ``Can't optimze method "evaluate" because self argument is used`` (CatBoost's own typo included) is emitted
    from ``_check_train_params`` on every single fit, says nothing the caller can act on, and is
    indistinguishable in a log from a warning that matters. Only this exact message is filtered -- everything
    else CatBoost has to say still reaches the log.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*optimze method.*", category=UserWarning)
        yield


def _maybe_disable_cb_plot(model_type_name: str, fit_params: dict, verbose: bool) -> None:
    """Set ``plot=False`` for CatBoost fits outside an interactive notebook.

    Without it CB spawns+joins a MetricVisualizer/ipywidget plot thread (~1.5s/fit)
    even headless / ``verbose=0``. Pure-config: no numerics change. Respects an
    explicit user-supplied ``plot`` and never disables inside a live Jupyter kernel
    unless verbose is off.
    """
    if model_type_name not in CATBOOST_MODEL_TYPES or "plot" in fit_params:
        return
    if _in_interactive_notebook() and verbose:
        return
    fit_params["plot"] = False


def _ensure_cb_mtr_loss(model, train_target, pool=None) -> None:
    """When target is (N, K>=2) continuous but CatBoostRegressor lacks an
    MTR-compatible ``loss_function``, set ``loss_function='MultiRMSE'``
    pre-fit.

    CatBoost rejects 2-D continuous ``y`` with the default ``RMSE`` loss
    (``Currently only multi-regression, multilabel and survival
    objectives work with multidimensional target``). The dispatch parallels
    ``_ensure_cb_multilabel_loss`` for CatBoostClassifier+multilabel.
    """
    if model is None:
        return
    if type(model).__name__ != "CatBoostRegressor":
        return
    try:
        # `get_params` FIRST: CatBoost defines both, and `get_param(key)` takes a required argument, so the
        # previous `get_param or get_params` always selected the single-key form and `get()` raised TypeError
        # on every call. That was invisible while the handler substituted an empty dict -- the loss still got
        # set, for the wrong reason -- and became a silent no-op the moment the handler correctly stopped
        # treating "params unknown" as "params empty".
        get = getattr(model, "get_params", None) or getattr(model, "get_param", None)
        params = get() if callable(get) else {}
    except Exception as e:
        # "Params unknown" is not "params empty". Substituting {} made the very next check conclude the caller
        # set no loss, so this function overwrote a deliberately-chosen CatBoost objective with MultiRMSE and the
        # model trained against the wrong one -- with every downstream metric then honestly computed on a
        # wrongly-trained model. Leave the objective alone and say why.
        logger.warning(
            "_ensure_cb_mtr_loss: get_param()/get_params() raised %s (%s); leaving the model's objective untouched rather than " "assuming none was set.",
            type(e).__name__,
            e,
        )
        return
    _existing = params.get("loss_function")
    # Skip when the user already wired a multi-target-compatible loss.
    if _existing is not None and "multi" in str(_existing).lower():
        return
    label_arr = None
    if pool is not None:
        try:
            label_arr = np.asarray(pool.get_label())
        except Exception as e:
            logger.debug("_ensure_cb_mtr_loss: pool.get_label() failed, treating label_arr as unavailable: %s", e)
            label_arr = None
    if label_arr is None:
        label_arr = np.asarray(train_target) if train_target is not None else None
        if label_arr is not None and label_arr.dtype == object and label_arr.ndim == 1 and label_arr.shape[0] > 0:
            try:
                # np.array(<object-array>.tolist()) stacks the per-row label vectors
                # ~2.5x faster than the np.asarray-per-row listcomp at n=100k (object
                # tolist() yields the row arrays as-is, then np.array stacks them).
                # Bit-identical for uniform-width rows; ragged rows still raise here and
                # hit the except below exactly as the prior np.stack did.
                label_arr = np.array(label_arr.tolist())
            except Exception as e:
                logger.debug("_ensure_cb_mtr_loss: ragged label_arr.tolist() -> np.array() failed: %s", e)
                label_arr = None
    if label_arr is None or label_arr.ndim != 2 or label_arr.shape[1] < 2:
        return
    # Only fires for continuous (float) labels; integer 2-D is multilabel
    # and handled by ``_ensure_cb_multilabel_loss`` instead.
    if label_arr.dtype.kind not in ("f",):
        return
    try:
        model.set_params(loss_function="MultiRMSE", eval_metric="MultiRMSE")
    except Exception:
        try:
            model._init_params["loss_function"] = "MultiRMSE"
            model._init_params["eval_metric"] = "MultiRMSE"
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
            pass


def _ensure_cb_multilabel_loss(model, train_target, pool=None) -> None:
    """When target is multilabel-shaped but CatBoost lacks loss_function,
    set loss_function='MultiLogloss' pre-fit."""
    if model is None:
        return
    if type(model).__name__ != "CatBoostClassifier":
        return
    try:
        # `get_params` FIRST: CatBoost defines both, and `get_param(key)` takes a required argument, so the
        # previous `get_param or get_params` always selected the single-key form and `get()` raised TypeError
        # on every call. That was invisible while the handler substituted an empty dict -- the loss still got
        # set, for the wrong reason -- and became a silent no-op the moment the handler correctly stopped
        # treating "params unknown" as "params empty".
        get = getattr(model, "get_params", None) or getattr(model, "get_param", None)
        params = get() if callable(get) else {}
    except Exception as e:
        # "Params unknown" is not "params empty". Substituting {} made the very next check conclude the caller
        # set no loss, so this function overwrote a deliberately-chosen CatBoost objective with MultiLogloss / HammingLoss and the
        # model trained against the wrong one -- with every downstream metric then honestly computed on a
        # wrongly-trained model. Leave the objective alone and say why.
        logger.warning(
            "_ensure_cb_multilabel_loss: get_param()/get_params() raised %s (%s); leaving the model's objective untouched rather than "
            "assuming none was set.",
            type(e).__name__, e,
        )
        return
    if params.get("loss_function") is not None:
        return
    label_arr = None
    if pool is not None:
        try:
            label_arr = np.asarray(pool.get_label())
        except Exception as e:
            logger.debug("_ensure_cb_multilabel_loss: pool.get_label() failed, treating label_arr as unavailable: %s", e)
            label_arr = None
    if label_arr is None:
        label_arr = np.asarray(train_target) if train_target is not None else None
        if label_arr is not None and label_arr.dtype == object and label_arr.ndim == 1 and label_arr.shape[0] > 0:
            try:
                # np.array(<object-array>.tolist()) stacks the per-row label vectors
                # ~2.5x faster than the np.asarray-per-row listcomp at n=100k (object
                # tolist() yields the row arrays as-is, then np.array stacks them).
                # Bit-identical for uniform-width rows; ragged rows still raise here and
                # hit the except below exactly as the prior np.stack did.
                label_arr = np.array(label_arr.tolist())
            except Exception as _e_stack:  # best-effort: falls back to the single-label default loss
                # Stack failure -> label_arr stays None -> the function
                # returns without configuring MultiLogloss / HammingLoss,
                # so a multilabel CatBoost ends up training with the
                # single-label default loss. Caller wouldn't see why.
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "multilabel CB auto-config: failed to stack label rows "
                    "(%s); CatBoost will use the single-label default loss "
                    "instead of MultiLogloss. Pass a 2D label array to bypass.",
                    _e_stack,
                )
                label_arr = None
    if label_arr is None or label_arr.ndim != 2:
        return
    try:
        model.set_params(loss_function="MultiLogloss", eval_metric="HammingLoss")
    except Exception:
        try:
            model._init_params["loss_function"] = "MultiLogloss"
            model._init_params["eval_metric"] = "HammingLoss"
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
            pass


def _handle_oom_error(model_obj, model_type_name: str) -> bool:
    """Attempt to recover from an OOM error by clearing caches and
    returning True if the caller should retry the fit.
    """
    import gc
    gc.collect()
    # Clear LGB/XGB/CB internal caches if accessible.
    for _attr in ("_Booster", "_cached_train_features", "_cached_val_features"):
        if hasattr(model_obj, _attr):
            try:
                delattr(model_obj, _attr)
            except Exception as e:  # nosec B110 - non-fatal by design
                logger.debug("could not delattr %s during OOM recovery: %s", _attr, e)
    logger.warning(
        "OOM during %s.fit; cleared caches and will retry once.",
        model_type_name,
    )
    return True
