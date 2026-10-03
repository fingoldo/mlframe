"""Training loop helpers extracted from ``trainer.py``.

Core training path: model fitting with CatBoost/LGB/XGB fallbacks,
early stopping, OOM recovery, and post-hoc probability calibration.
"""

from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import Any

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


from mlframe.config import CATBOOST_MODEL_TYPES

# Refit helpers + their module-level constants moved to sibling
# ``_training_loop_refit.py`` to drop this file below the 1k-LOC
# monolith threshold; imported here so callers keep using
# ``from mlframe.training._training_loop import _maybe_refit_on_*``.
from ._training_loop_refit import (  # noqa: F401  (re-exported)
    _maybe_refit_on_collapsed_predictions,
    _maybe_refit_on_best_iter_pathology,
    _maybe_refit_on_degenerate_best_iter,
    _maybe_refit_on_saturated_best_iter,
)
from mlframe.training.cb.shared import apply_cb_eval_sample_weights
from mlframe.training.cb.shared import cb_text_features_as_strings
from mlframe.training.cb.shared import (
    maybe_get_or_build_cb_pool as _maybe_get_or_build_cb_pool,
    maybe_rewrite_eval_set_as_cb_pool as _maybe_rewrite_eval_set_as_cb_pool,
)

logger = logging.getLogger(__name__)


from ._training_loop_cb_utils import (  # noqa: F401  -- carved helpers
    _suppress_catboost_noise,
    _maybe_disable_cb_plot,
    _ensure_cb_mtr_loss,
    _ensure_cb_multilabel_loss,
    _handle_oom_error,
    _in_interactive_notebook,
)
from ._training_loop_fallback_helpers import (  # noqa: F401  -- carved helpers
    _train_model_with_f_base_estimator_cloned_per,
    _train_model_with_f_trainer_boundary_regardless_which,
    _train_model_with_f_should_have_been_converted,
    _train_model_with_f_base_estimator_cloned_per_2,
    _train_model_with_f_pandas_model_type_name,
    _train_model_with_f_text_processing_via_cb,
    _train_model_with_f_re_align_here_xgb,
    _train_model_with_f_isinstance_train_df_pd,
    _train_model_with_f_phase,
    _train_model_with_f_model_none,
    _train_model_with_f_focused_unit_testing,
    _ensure_xgb_classification_objective,  # re-export: part of this module's public surface
    _maybe_wrap_for_2d_target,  # re-export: part of this module's public surface
)

# post-hoc calibration wrappers
# (_SigmoidAdapter, _PostHocCalibratedModel, _PerClassIsotonicCalibrator,
# _PostHocMultiCalibratedModel, _maybe_apply_posthoc_calibration) moved
# to sibling file _calibration_models.py to drop this file below the
# 1k-line monolith threshold. Re-exported below so existing callers
# (`from ._training_loop import _PostHocCalibratedModel`, etc.) keep working.
from ._calibration_models import (  # noqa: F401
    _maybe_apply_posthoc_calibration,
    _PerClassIsotonicCalibrator,
    _PostHocCalibratedModel,
    _PostHocMultiCalibratedModel,
    _SigmoidAdapter,
)


def _train_model_with_fallback(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
    """Fit under ``CatBoostGpuFitGuard`` (GPU CatBoost: no callbacks, progress monitor, time budget / runaway stop that keeps the model; else a no-op)."""
    from mlframe.training.cb.shared import fit_with_cb_gpu_guard
    return fit_with_cb_gpu_guard(_train_model_with_fallback_unguarded, model, model_obj, model_type_name, train_df, train_target, fit_params, verbose)


def _train_model_with_fallback_unguarded(
    model: Any,
    model_obj: Any,
    model_type_name: str,
    train_df: pd.DataFrame | np.ndarray,
    train_target: pd.Series | np.ndarray,
    fit_params: dict[str, Any],
    verbose: bool = False,
) -> tuple[Any, int | None]:
    """Train model with automatic GPU->CPU fallback on OOM errors.

    Parameters
    ----------
    model : Any
        Model to train (may be a Pipeline).
    model_obj : Any
        The actual estimator object (extracted from Pipeline if needed).
    model_type_name : str
        Name of the model type (e.g., 'CatBoostClassifier').
    train_df : pd.DataFrame or np.ndarray
        Training features.
    train_target : pd.Series or np.ndarray
        Training target values.
    fit_params : dict
        Additional parameters for model.fit().
    verbose : bool, default=False
        Whether to log verbose output.

    Returns
    -------
    tuple
        (trained_model, best_iteration) where best_iteration may be None.
    """
    t0_fit = timer()
    # 0-feature train frame is unfittable: CatBoost raises ``CatBoostError: Input data must have at least one feature``,
    # XGBoost raises an opaque DMatrix IndexError, and the linear/sklearn estimators raise their own validate_data errors.
    # The suite-level guard at ``_trainer_train_and_evaluate`` already skips the common FS-empties-everything case, but
    # any column-dropping step between that check and this fit primitive (or a direct caller) can still arrive 0-feature.
    # Mirror the empty-FS warning + return ``(None, None)`` so the caller's ``if model is None: skip`` path handles it,
    # rather than letting the per-backend C++ crash abort the whole suite run.
    _n_feat = None
    if train_df is not None and hasattr(train_df, "shape") and len(getattr(train_df, "shape", ())) == 2:
        _n_feat = train_df.shape[1]
    if _n_feat == 0:
        logger.warning(
            "Skipping %s fit: train frame has 0 features (feature selection / column dropping removed every column). "
            "Nothing to fit -- the model is skipped instead of crashing the backend.",
            model_type_name,
        )
        return None, None
    # CB-only: reuse a single ``catboost.Pool`` across weight schemas
    # and same-target_type targets by mutating the Pool's label/weight
    # in place instead of letting the sklearn wrapper rebuild from X on
    # every fit. Gated on:
    #   * model is CatBoost-family;
    #   * installed CatBoost exposes ``Pool.set_label`` and
    #     ``Pool.set_weight`` (callable);
    #   * ``CatBoostClassifier.fit(X=Pool)`` is the idiomatic native path
    #     (short-circuits rebuild in ``_build_train_pool``).
    # XGB/LGB are not yet covered -- their sklearn wrappers don't accept pre-built DMatrix/Dataset yet (upstream FRs drafted in
    # ``D:\Machine Learning\3rdParty\reproducers\upstream_feature_requests\``). The per-build logging makes their rebuild cost visible.
    train_df = cb_text_features_as_strings(model_type_name, train_df, fit_params)  # a polars Categorical text feature crashes CatBoost
    _cb_pool = _maybe_get_or_build_cb_pool(
        model_type_name=model_type_name,
        model=model,
        train_df=train_df,
        train_target=train_target,
        fit_params=fit_params,
    )
    # Also reuse the val Pool across fits: rewrite fit_params['eval_set'] from (val_df, val_target) to a cached Pool so CB's
    # sklearn wrapper short-circuits the val-side rebuild too. Only when _cb_pool is active (train-side reuse succeeded):
    # otherwise the cached-Pool path mixes containers (train=df, eval_set=pool), which confuses CB's fit signature.
    if _cb_pool is not None and model_type_name in CATBOOST_MODEL_TYPES:
        _maybe_rewrite_eval_set_as_cb_pool(fit_params)
    apply_cb_eval_sample_weights(fit_params, model_type_name)  # CB has no eval-weight kwarg; the weights ride on the eval Pool
    # Diagnostic: log the type+module of train_df right before model.fit so
    # silent type drift is visible in the log (Polars vs pandas vs numpy).
    # Critical: type(pl.DataFrame).__name__ == "DataFrame" -- same as pandas --
    # so we log the module too, otherwise "DataFrame" can hide a Polars frame
    # that should have been converted upstream.
    _is_polars = pl is not None and isinstance(train_df, pl.DataFrame)
    _is_pandas = isinstance(train_df, pd.DataFrame)
    _train_model_with_f_should_have_been_converted(_is_polars, _is_pandas, train_df)

    # Polars-frame contract: only CatBoost, XGBoost, and HistGradientBoosting
    # accept a Polars frame natively at fit time -- their strategies carry
    # ``supports_polars=True``. Everyone else (LGB, sklearn, linear, ridge,
    # ...) MUST arrive with pandas; if a pl.DataFrame gets here for them, the
    # upstream lazy-conversion -> pipeline_cache -> process_model chain has a
    # leak. Previously the trainer silently ran a second polars->pandas
    # conversion as a "self-heal" -- which hid a regression where
    # ``pipeline_cache`` crossed streams between XGB (polars-native,
    # ``cache_key="tree" + tier(False,False)``) and LGB (same key) -- LGB kept
    # pulling XGB's polars frame out of cache and paying a duplicate 224 s
    # conversion. The pipeline_cache fix (container-kind in key, core.py) is
    # the real fix; this raise is the guard that ensures future leaks are
    # caught at the trainer boundary instead of being papered over.
    _POLARS_NATIVE_FIT_MODEL_PREFIXES = (
        "CatBoost",  # CatBoostClassifier / CatBoostRegressor / CatBoost
        "XGB",  # XGBClassifier / XGBRegressor / XGBRanker
        "HistGradient",  # HistGradientBoostingClassifier / Regressor
    )
    # Look through MultiOutputClassifier wrapper for the polars-native check.
    # The wrapper's `estimator` is the per-label base; if the base is polars-native
    # (e.g. HGB), each per-label fit will accept polars too.
    # _ChainEnsemble is the multilabel chain ensemble -- its inner is exposed
    # as `base_estimator` (cloned per chain at fit time).
    _effective_model_type_name = model_type_name
    best_iter, model = _train_model_with_f_base_estimator_cloned_per(model_type_name, model, _effective_model_type_name, _is_polars, _POLARS_NATIVE_FIT_MODEL_PREFIXES, train_df, _is_pandas, fit_params, _cb_pool, verbose, train_target, model_obj, t0_fit)

    return model, best_iter


# xgb-objective / 2d-target-wrap helpers carved to _training_loop_objectives.py (1k-LOC ceiling).
