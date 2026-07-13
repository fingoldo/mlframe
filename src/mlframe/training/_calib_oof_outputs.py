"""Post-fit tail helpers for ``train_and_evaluate_model``.

Self-contained sections lifted from ``_trainer_train_and_evaluate.py`` so the
main function stays under the LOC ceiling: the disjoint-calib predict + OOF
mirror outputs, and the optional confidence-analysis dispatch. Each reads an
explicit set of inputs and has no shared-local-namespace coupling.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np

from .pipeline import _prepare_test_split
from ._feature_name_sanitize import sanitize_frame_columns as _sanitize_frame_columns
from .cb import _predict_with_fallback
from ._eval_helpers import run_confidence_analysis

logger = logging.getLogger("mlframe.training.trainer")


def maybe_run_confidence_analysis(
    *,
    run_test: bool,
    confidence: Any,
    test_df: Any,
    test_target: Any,
    test_probs: Any,
    fit_params: Optional[dict],
    model_type_name: str,
    figsize: Any,
    verbose: Any,
) -> None:
    """Run the optional SHAP/confidence analysis on the test split when enabled.

    No-op unless ``run_test`` and ``confidence.include`` are truthy and a test
    frame exists. Produces no return value (plots/diagnostics are side effects).
    """
    if not run_test:
        return
    if not (confidence.include and test_df is not None):
        return
    run_confidence_analysis(
        test_df=test_df,
        test_target=test_target,
        test_probs=test_probs,
        cat_features=fit_params.get("cat_features") if fit_params else None,
        text_features=fit_params.get("text_features") if fit_params else None,
        embedding_features=fit_params.get("embedding_features") if fit_params else None,
        confidence_model_kwargs=dict(confidence.model_kwargs) if confidence.model_kwargs else {},
        fit_params=fit_params if model_type_name == "CatBoostRegressor" else None,
        use_shap=confidence.use_shap,
        max_features=confidence.max_features,
        cmap=confidence.cmap,
        alpha=confidence.alpha,
        title=confidence.title,
        ylabel=confidence.ylabel,
        figsize=figsize,
        verbose=verbose,
    )


def compute_calib_and_oof_outputs(
    *,
    model: object,
    calib_df: Any,
    calib_target: Any,
    real_drop_columns: Any,
    pre_pipeline: Any,
    skip_pre_pipeline_transform: bool,
    skip_preprocessing: bool,
    fit_params: dict,
    model_type_name: str,
    model_name: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Any, Any, Any]:
    """Run the fitted model's calib-slice predict (proba for classification, point for regression) and mirror OOF preds/probs/target.

    Returns ``(calib_probs, calib_target, calib_preds, oof_preds, oof_probs, oof_target)``. ``calib_probs``
    is ``None`` for regression / no-``predict_proba`` models; ``calib_preds`` is the regression point
    prediction on the calib slice (``None`` for classification or when no calib frame exists) -- the
    leakage-free residual source for split-conformal in finalize. The OOF outputs mirror whatever was
    stamped on ``model`` during training.
    """
    # Disjoint-calib predict (TrainingSplitConfig.calib_size > 0): run the fitted base model's predict_proba on the calib
    # slice and stamp (calib_probs, calib_target) so finalize's _auto_calibrate_on_calib_slice fits the post-hoc isotonic
    # calibrator. Mirrors the test path: subset+transform the raw calib rows through the SAME fitted pre_pipeline via
    # _prepare_test_split, then predict_proba via _predict_with_fallback. Gated strictly on a calib frame existing and the
    # model exposing predict_proba (classification) -- regression / no-predict_proba models stamp nothing (consumer no-ops).
    calib_probs = None
    calib_target_out = None
    calib_preds = None
    if model is not None and calib_df is not None and hasattr(model, "predict_proba"):
        try:
            _calib_df_prep, _calib_target_prep, _ = _prepare_test_split(
                df=None,
                test_df=calib_df,
                test_idx=None,
                test_target=calib_target,
                target=None,
                real_drop_columns=real_drop_columns,
                model=model,
                pre_pipeline=pre_pipeline,
                skip_pre_pipeline_transform=skip_pre_pipeline_transform,
                skip_preprocessing=skip_preprocessing,
                selector_passthrough_cols=(list(fit_params.get("text_features") or []) + list(fit_params.get("embedding_features") or [])) or None,
            )
            _calib_df_prep = _sanitize_frame_columns(_calib_df_prep)
            if _calib_df_prep is not None and _calib_target_prep is not None and len(_calib_df_prep) > 0:
                from ._classif_helpers import _canonical_predict_proba_shape
                from ._data_helpers import _prepare_df_for_model

                _calib_X = _prepare_df_for_model(_calib_df_prep, model_type_name)
                _cp = _predict_with_fallback(model, _calib_X, method="predict_proba")
                _cp = _canonical_predict_proba_shape(_cp)
                calib_probs = _cp
                calib_target_out = _calib_target_prep.values if hasattr(_calib_target_prep, "values") else np.asarray(_calib_target_prep)
        except Exception as _calib_err:
            logger.warning("calib-slice predict failed for %s; finalize calibration will no-op for this model: %s", model_name, _calib_err)

    # Regression calib-slice point predictions: the leakage-free residual source for split-conformal in finalize.
    # Runs only when the model has no predict_proba (regression) but can predict; mirrors the classification block's
    # subset+transform+predict path. None on classification / no calib frame so the conformal consumer no-ops.
    if model is not None and calib_df is not None and calib_probs is None and hasattr(model, "predict"):
        try:
            _calib_df_prep, _calib_target_prep, _ = _prepare_test_split(
                df=None,
                test_df=calib_df,
                test_idx=None,
                test_target=calib_target,
                target=None,
                real_drop_columns=real_drop_columns,
                model=model,
                pre_pipeline=pre_pipeline,
                skip_pre_pipeline_transform=skip_pre_pipeline_transform,
                skip_preprocessing=skip_preprocessing,
                selector_passthrough_cols=(list(fit_params.get("text_features") or []) + list(fit_params.get("embedding_features") or [])) or None,
            )
            _calib_df_prep = _sanitize_frame_columns(_calib_df_prep)
            if _calib_df_prep is not None and _calib_target_prep is not None and len(_calib_df_prep) > 0:
                from ._data_helpers import _prepare_df_for_model

                _calib_X = _prepare_df_for_model(_calib_df_prep, model_type_name)
                calib_preds = np.asarray(_predict_with_fallback(model, _calib_X, method="predict")).reshape(-1)
                if calib_target_out is None:
                    calib_target_out = _calib_target_prep.values if hasattr(_calib_target_prep, "values") else np.asarray(_calib_target_prep)
        except Exception as _calib_reg_err:
            logger.warning("calib-slice regression predict failed for %s; finalize conformal will fall back: %s", model_name, _calib_reg_err)

    # OOF preds/probs/target were stamped on ``model`` right after training (see ``_compute_oof_preds`` call above).
    # Mirror them onto the returned namespace so ensemble member shapes carry the OOF signal alongside the
    # per-split predictions without callers having to fish them off the model object. oof_target must be
    # mirrored the same way oof_preds/oof_probs are -- consumers like
    # ``recommend_diversity_additions_in_leaderboard`` require oof_preds/oof_probs AND oof_target together
    # on every member regardless of task type (regression stamps oof_preds+oof_target with no oof_probs).
    oof_preds = getattr(model, "oof_preds", None) if model is not None else None
    oof_probs = getattr(model, "oof_probs", None) if model is not None else None
    oof_target = getattr(model, "oof_target", None) if model is not None else None
    return calib_probs, calib_target_out, calib_preds, oof_preds, oof_probs, oof_target
