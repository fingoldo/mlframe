"""Per-method ensemble dispatcher for ``mlframe.models.ensembling``.

Split out of ``ensembling.py`` to keep the parent below the 1k-line monolith
threshold. ``_process_single_ensemble_method`` is the per-method worker that
``score_ensemble`` fans out via joblib; the parent re-exports it so
``score_ensemble``'s call sites continue to resolve identity-equal.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ._ensembling_base import build_predictive_kwargs
from ._ensembling_predict import ensemble_probabilistic_predictions

logger = logging.getLogger("mlframe.models.ensembling")


def _process_single_ensemble_method(
    ensemble_method: str,
    level_models_and_predictions: Sequence,
    is_regression: bool,
    ensembling_level: int,
    ensemble_name: str,
    target: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    val_idx: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    val_target: np.ndarray,
    target_label_encoder: object,
    max_mae: float,
    max_std: float,
    max_mae_relative: float,
    max_std_relative: float,
    ensure_prob_limits: bool,
    nbins: int,
    uncertainty_quantile: float,
    normalize_stds_by_mean_preds: bool,
    custom_ice_metric: Callable,
    custom_rice_metric: Callable,
    subgroups: dict,
    n_features: int,
    verbose: bool,
    kwargs: dict,
    flag_degenerate_conf_subset: bool = True,
    degenerate_class_ratio: float = 0.01,
    sample_weight: Optional[np.ndarray] = None,
    rrf_k: int = 60,
) -> tuple:
    """Process a single ensemble method. Returns (method_name, results, conf_results, next_level_pred)."""
    # E1.1 opt-out: these lazy imports break a circular dep with mlframe.training.
    # The Parallel dispatch at line 2539 uses ``backend="loky"`` (process workers,
    # not threads), so each fork has its own Python interpreter -- no shared
    # import state to race against. Safe to keep lazy here.
    from mlframe.training import train_and_evaluate_model  # joblib-import-race-ok
    from mlframe.training.trainer import _build_configs_from_params  # joblib-import-race-ok

    # 2026-05-13 (bug fix): val_preds / test_preds may be ``None`` when the
    # corresponding split metric computation was disabled at suite level
    # (``ReportingConfig.compute_valset_metrics=False`` /
    # ``compute_testset_metrics=False``). Pre-fix the bare
    # ``el.val_preds.reshape(-1, 1)`` raised AttributeError on the first
    # ``None``-valued member. Mirrors the existing train-side guard at
    # the bottom of this function (line ~870). When NO members have val
    # preds, the ensemble call gets an empty tuple; downstream
    # ``ensemble_probabilistic_predictions`` already returns
    # ``(None, None, None)`` for that case (line ~438).
    if not is_regression:
        _val_preds = [el.val_probs for el in level_models_and_predictions if el.val_probs is not None]
        predictions = iter(_val_preds)
    else:
        _val_preds = [el.val_preds for el in level_models_and_predictions if el.val_preds is not None]
        predictions = (p.reshape(-1, 1) for p in _val_preds)

    val_ensembled_predictions, _, val_confident_indices = ensemble_probabilistic_predictions(
        *predictions,
        ensemble_method=ensemble_method,
        max_mae=max_mae,
        max_std=max_std,
        max_mae_relative=max_mae_relative,
        max_std_relative=max_std_relative,
        ensure_prob_limits=ensure_prob_limits,
        uncertainty_quantile=uncertainty_quantile,
        normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
        verbose=verbose,
        sample_weight=sample_weight,
        rrf_k=rrf_k,
    )

    # 2026-05-13: same ``None``-guard for test_preds / test_probs.
    if not is_regression:
        _test_preds = [el.test_probs for el in level_models_and_predictions if el.test_probs is not None]
        predictions = iter(_test_preds)
    else:
        _test_preds = [el.test_preds for el in level_models_and_predictions if el.test_preds is not None]
        predictions = (p.reshape(-1, 1) for p in _test_preds)

    test_ensembled_predictions, _, test_confident_indices = ensemble_probabilistic_predictions(
        *predictions,
        ensemble_method=ensemble_method,
        max_mae=max_mae,
        max_std=max_std,
        max_mae_relative=max_mae_relative,
        max_std_relative=max_std_relative,
        ensure_prob_limits=ensure_prob_limits,
        uncertainty_quantile=uncertainty_quantile,
        normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
        verbose=verbose,
        sample_weight=sample_weight,
        rrf_k=rrf_k,
    )

    # Level-1 aggregation reads OOF predictions when present (the K-fold cross_val_predict output stamped by the trainer),
    # not in-sample ``train_preds``/``train_probs`` -- the latter are produced by predicting on rows the model already saw
    # during fit and leak that fit's residual structure into a meta-learner. Falling back to ``train_*`` keeps the path
    # alive when OOF was unavailable (e.g. tiny datasets, multi-output paths where ``cross_val_predict`` skipped); a
    # higher-level guard in ``score_ensemble`` raises when ``max_ensembling_level > 1`` AND any member is missing OOF.
    #
    # Existence check uses ``isinstance(..., np.ndarray)`` rather than ``is not None`` because MagicMock-based test
    # doubles auto-fabricate any attribute access (mock.oof_probs returns a MagicMock, never None); falling back on
    # the array-instance check keeps the production path identical while letting test mocks still hit the train_*
    # branch when they only stamp train_*/val_* arrays.
    def _oof_or_train(el, oof_attr, train_attr):
        _oof = getattr(el, oof_attr, None)
        if isinstance(_oof, np.ndarray):
            return _oof
        return getattr(el, train_attr, None)

    if not is_regression:
        predictions = (_oof_or_train(el, "oof_probs", "train_probs") for el in level_models_and_predictions)
    elif (_oof_or_train(level_models_and_predictions[0], "oof_preds", "train_preds") is not None):
        predictions = (_oof_or_train(el, "oof_preds", "train_preds") for el in level_models_and_predictions)
        predictions = (el.reshape(-1, 1) if (el is not None) else el for el in predictions)
    else:
        predictions = ()

    train_ensembled_predictions, _, train_confident_indices = ensemble_probabilistic_predictions(
        *predictions,
        ensemble_method=ensemble_method,
        max_mae=max_mae,
        max_std=max_std,
        max_mae_relative=max_mae_relative,
        max_std_relative=max_std_relative,
        ensure_prob_limits=ensure_prob_limits,
        uncertainty_quantile=uncertainty_quantile,
        normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
        verbose=verbose,
        sample_weight=sample_weight,
        rrf_k=rrf_k,
    )

    internal_ensemble_method = f"{ensemble_method} L{ensembling_level}" if ensembling_level > 0 else ensemble_method

    predictive_kwargs = build_predictive_kwargs(
        train_data=train_ensembled_predictions, test_data=test_ensembled_predictions, val_data=val_ensembled_predictions, is_regression=is_regression
    )

    if target is not None:
        target_kwargs = dict(target=target)
    else:
        target_kwargs = dict(train_target=train_target, test_target=test_target, val_target=val_target)

    # Pop params not accepted by _build_configs_from_params (they come from common_params in core.py)
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop("trainset_features_stats", None)
    kwargs_copy.pop("train_od_idx", None)
    kwargs_copy.pop("val_od_idx", None)
    # 2026-04-23 (coverage-gap test_ensembles_enabled_produces_ensemble_log):
    # ``common_params`` frequently carries ``drop_columns`` when the user
    # passes ``init_common_params={"drop_columns": [...]}``. The literal
    # ``drop_columns=[]`` below then collides with the ``**kwargs_copy``
    # splat two positions later, raising
    # ``TypeError: dict() got multiple values for keyword argument 'drop_columns'``.
    # Pop the caller's value -- the ensemble scorer intentionally sets
    # ``drop_columns=[]`` to avoid dropping anything its sub-models
    # already trained on (columns already stripped upstream).
    kwargs_copy.pop("drop_columns", None)
    # 2026-04-24 (fuzz extension): init_common_params is a prod
    # convention for passing PIPELINE COMPONENTS (not training
    # hyperparams), e.g.:
    #     init_common_params = {
    #         "category_encoder": ce.CatBoostEncoder(),
    #         "scaler": StandardScaler(),
    #         "imputer": SimpleImputer(strategy="mean"),
    #     }
    # Suite threads these into common_params so per-model pre_pipeline
    # builders pick them up. But the ensemble-scoring helper calls
    # ``_build_configs_from_params(**kwargs_copy)`` -- a function with a
    # declared signature that raises TypeError on any kwarg it doesn't
    # know about. Pop pipeline-component kwargs here so the ensemble
    # path doesn't leak them into the config builder. This isn't
    # feature loss: sub-models have already been fitted BEFORE the
    # ensemble scorer runs; we don't re-apply encoder/scaler/imputer
    # inside ensemble scoring.
    for _pipeline_kwarg in ("category_encoder", "scaler", "imputer"):
        kwargs_copy.pop(_pipeline_kwarg, None)

    # 2026-04-27 typed-config refactor: ``compute_{trainset,valset,testset}_-
    # metrics`` were lifted from trainer-internal to ``ReportingConfig``.
    # core.py now seeds ``common_params_dict`` from ``reporting_config.model_dump()``,
    # which makes those fields part of ``kwargs_copy``. Preserve the caller's
    # switches before popping them to avoid duplicate-key ``dict(...)`` splats;
    # ensemble scoring must respect the same reporting contract as single
    # models.
    _caller_compute_trainset_metrics = bool(
        kwargs_copy.pop("compute_trainset_metrics", False)
    )
    _caller_compute_valset_metrics = bool(
        kwargs_copy.pop("compute_valset_metrics", True)
    )
    _caller_compute_testset_metrics = bool(
        kwargs_copy.pop("compute_testset_metrics", True)
    )

    def _has_split_predictions(_kwargs: dict, _split: str) -> bool:
        return (
            _kwargs.get(f"{_split}_preds") is not None
            or _kwargs.get(f"{_split}_probs") is not None
        )

    # Build config objects from flat params
    flat_params = dict(
        df=None,
        drop_columns=[],
        model_name_prefix=f"Ens{internal_ensemble_method.upper()} {ensemble_name}",
        train_idx=train_idx,
        test_idx=test_idx,
        val_idx=val_idx,
        target_label_encoder=target_label_encoder,
        compute_trainset_metrics=(
            _caller_compute_trainset_metrics
            and _has_split_predictions(predictive_kwargs, "train")
        ),
        compute_valset_metrics=(
            _caller_compute_valset_metrics
            and _has_split_predictions(predictive_kwargs, "val")
        ),
        compute_testset_metrics=(
            _caller_compute_testset_metrics
            and _has_split_predictions(predictive_kwargs, "test")
        ),
        nbins=nbins,
        custom_ice_metric=custom_ice_metric,
        custom_rice_metric=custom_rice_metric,
        subgroups=subgroups,
        n_features=n_features,
        **target_kwargs,
        **predictive_kwargs,
        **kwargs_copy,
    )
    data, control, metrics_cfg, reporting_cfg, naming, confidence, predictions_cfg, output_cfg = _build_configs_from_params(**flat_params)
    next_ens_results = train_and_evaluate_model(
        model=None,
        data=data,
        control=control,
        metrics=metrics_cfg,
        reporting=reporting_cfg,
        naming=naming,
        output=output_cfg,
        confidence=confidence,
        predictions=predictions_cfg,
    )

    conf_results = None
    if uncertainty_quantile:
        if target is not None:
            conf_target_kwargs = dict(target=target)
        else:
            conf_target_kwargs = dict(
                train_target=train_target[train_confident_indices] if (train_target is not None and train_confident_indices is not None) else None,
                test_target=test_target[test_confident_indices] if (test_target is not None and test_confident_indices is not None) else None,
                val_target=val_target[val_confident_indices] if (val_target is not None and val_confident_indices is not None) else None,
            )

        conf_predictive_kwargs = build_predictive_kwargs(
            train_data=(
                train_ensembled_predictions[train_confident_indices]
                if (train_ensembled_predictions is not None and train_confident_indices is not None)
                else None
            ),
            test_data=(
                test_ensembled_predictions[test_confident_indices] if (test_ensembled_predictions is not None and test_confident_indices is not None) else None
            ),
            val_data=(
                val_ensembled_predictions[val_confident_indices] if (val_ensembled_predictions is not None and val_confident_indices is not None) else None
            ),
            is_regression=is_regression,
        )

        # Report the confidence-filter coverage right in the model name so
        # log-grep immediately shows that e.g. "Conf Ensemble ... [VAL
        # COV=10%]" is computed on just 10 % of VAL rows -- previously the
        # 99.77 % accuracy number in the Conf Ensemble block was easy to
        # misread as a headline, because coverage only appeared inside the
        # calibration subsection as ``COV=XX%`` (2026-04-23 review finding).
        # Prefer VAL coverage as the headline (early-stopping + calibration
        # both key on VAL); fall back to TEST coverage then TRAIN.
        # Evaluate (label, full_preds, conf_idx, target_for_label) for each split
        # in priority order. We need the target slice as well because the
        # degenerate-class-balance check (below) operates on the filtered target,
        # not on the prediction array.
        _cov_src = None
        _conf_target = None
        for _label, _full, _conf, _full_target in (
            ("VAL", val_ensembled_predictions, val_confident_indices, val_target),
            ("TEST", test_ensembled_predictions, test_confident_indices, test_target),
            ("TRAIN", train_ensembled_predictions, train_confident_indices, train_target),
        ):
            if _full is not None and _conf is not None and len(_full) > 0:
                _cov_src = (_label, 100.0 * len(_conf) / len(_full))
                if _full_target is not None and len(_full_target) == len(_full):
                    _conf_target = _full_target[_conf]
                break

        # Degenerate-class-balance check on the filtered target. A confidence
        # filter that "keeps the rows the ensemble agrees on" tends to keep
        # almost-all-positive (or almost-all-negative) subsets on imbalanced
        # data РІР‚вЂќ one prod log showed 21 negatives vs 81_815 positives in the
        # 10 % VAL slice, and the resulting ``BR=0.026 %`` looked like a
        # headline win until you noticed it was reporting on a degenerate
        # split. Marker is binary-classification only; regression has no
        # class balance to check.
        _degenerate_marker = ""
        if flag_degenerate_conf_subset and not is_regression and _conf_target is not None and len(_conf_target) > 0:
            _ct = np.asarray(_conf_target)
            if _ct.ndim == 1:
                # Count positives via boolean comparison so float / bool / int
                # targets all behave the same. Ratio is min/max regardless of
                # which class is the minority.
                _n_pos = int((_ct == 1).sum())
                _n_neg = int(_ct.shape[0] - _n_pos)
                _hi = max(_n_pos, _n_neg)
                _lo = min(_n_pos, _n_neg)
                if _hi > 0 and (_lo / _hi) < degenerate_class_ratio:
                    _degenerate_marker = "[DEGENERATE] "

        # Trailing space so the downstream concat ``f"...{ensemble_name}{_cov_tag}"``
        # doesn't slam the next token onto the closing bracket -- the 2026-04-24
        # prod log showed ``[VAL COV=10%]notext prod_jobsdetails ...`` (no space
        # before "notext"). Empty tag stays empty (no double-space when off).
        _cov_tag = f" {_degenerate_marker}[{_cov_src[0]} COV={_cov_src[1]:.0f}%] " if _cov_src else ""

        # Build config objects from flat params for confidence ensemble
        conf_flat_params = dict(
            df=None,
            drop_columns=[],
            model_name_prefix=f"Conf Ensemble {internal_ensemble_method} {ensemble_name}{_cov_tag}",
            train_idx=train_idx[train_confident_indices] if (train_idx is not None and train_confident_indices is not None) else None,
            test_idx=test_idx[test_confident_indices] if (test_idx is not None and test_confident_indices is not None) else None,
            val_idx=val_idx[val_confident_indices] if (val_idx is not None and val_confident_indices is not None) else None,
            target_label_encoder=target_label_encoder,
            compute_trainset_metrics=(
                _caller_compute_trainset_metrics
                and _has_split_predictions(conf_predictive_kwargs, "train")
            ),
            compute_valset_metrics=(
                _caller_compute_valset_metrics
                and _has_split_predictions(conf_predictive_kwargs, "val")
            ),
            compute_testset_metrics=(
                _caller_compute_testset_metrics
                and _has_split_predictions(conf_predictive_kwargs, "test")
            ),
            nbins=nbins,
            custom_ice_metric=custom_ice_metric,
            custom_rice_metric=custom_rice_metric,
            subgroups=subgroups,
            n_features=n_features,
            **conf_predictive_kwargs,
            **conf_target_kwargs,
            **kwargs_copy,
        )
        conf_data, conf_control, conf_metrics, conf_reporting, conf_naming, conf_confidence, conf_predictions, conf_output = _build_configs_from_params(
            **conf_flat_params
        )
        conf_results = train_and_evaluate_model(
            model=None,
            data=conf_data,
            control=conf_control,
            metrics=conf_metrics,
            reporting=conf_reporting,
            naming=conf_naming,
            output=conf_output,
            confidence=conf_confidence,
            predictions=conf_predictions,
        )

    return (internal_ensemble_method, next_ens_results, conf_results)


