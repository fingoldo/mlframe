"""``get_training_configs`` carved out of ``mlframe.training.helpers``.

Re-imported at the parent module bottom so historical
``from mlframe.training.helpers import get_training_configs`` callers keep working.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace  # bundled config return at the bottom of get_training_configs
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import polars as pl
import psutil  # used for n_jobs=cpu_count(logical=False) in XGB_GENERAL_PARAMS
import lightgbm as lgb
import xgboost as xgb
from xgboost.callback import TrainingCallback  # noqa: F401
from sklearn.metrics import roc_auc_score  # noqa: F401
from sklearn.model_selection import TimeSeriesSplit

from ._gpu_probe import CUDA_IS_AVAILABLE, LGB_GPU_AVAILABLE, XGB_GPU_AVAILABLE
from ._classif_helpers import _classif_objective_kwargs
from mlframe.metrics.core import (
    ICE,
    compute_probabilistic_multiclass_error,
    robust_mlperf_metric,
)

logger = logging.getLogger("mlframe.training.helpers")


def get_training_configs(
    iterations: int = 5000,
    early_stopping_rounds: Optional[int] = 0,
    validation_fraction: float = 0.1,
    use_explicit_early_stopping: bool = True,
    has_time: bool = True,
    has_gpu: bool = None,
    subgroups: dict = None,
    learning_rate: float = 0.1,
    # 2026-05-26 (user request): default flipped from "MAE" to "RMSE"
    # for unified ES surface across CB / LGB / XGB / MLP. RMSE is the
    # competition-canonical metric (matches sklearn.r2_score's denominator
    # + the user-visible chart titles "MAE=... RMSE=... R2=..."). When
    # heavy-kurt is detected, ``_apply_loss_recommendation_in_place``
    # routes to Huber + matching eval_metric (so the booster doesn't
    # optimise one surface while ES tracks another). When pre-fix the
    # default was "MAE", CB / LGB / XGB on a near-Gaussian target
    # ran objective=RMSE + eval_metric=MAE => ES surface plateaued
    # earlier than the optimiser was descending => systematic under-
    # convergence (~iter=147 vs the 5000-iter cap).
    def_regr_metric: str = "RMSE",
    def_classif_metric: str = "AUC",
    # target_type-aware classifier objective injection. When target_type
    # is BINARY_CLASSIFICATION (default), the existing binary objectives
    # ("binary:logistic" / "binary" etc.) are kept. For MULTICLASS /
    # MULTILABEL, _classif_objective_kwargs replaces them with the right
    # native dispatch ("multi:softprob"+num_class, "MultiLogloss", etc.).
    target_type: Optional[Any] = None,  # TargetTypes; None = legacy binary
    n_classes: int = 2,
    catboost_custom_classif_metrics: Optional[Sequence] = None,
    catboost_custom_regr_metrics: Optional[Sequence] = None,
    random_seed: Optional[int] = None,
    verbose: int = 0,
    # ----------------------------------------------------------------------------------------------------------------------------
    # probabilistic errors
    # ----------------------------------------------------------------------------------------------------------------------------
    method: str = "multicrit",
    mae_weight: float = 3,
    std_weight: float = 2,
    roc_auc_weight: float = 1.5,
    pr_auc_weight: float = 0.1,
    brier_loss_weight: float = 0.8,
    min_roc_auc: float = 0.54,
    roc_auc_penalty: float = 0.00,
    use_weighted_calibration: bool = True,
    weight_by_class_npositives: bool = False,
    nbins: int = 10,
    # ----------------------------------------------------------------------------------------------------------------------------
    # robustness parameters for early stopping metric
    # ----------------------------------------------------------------------------------------------------------------------------
    robustness_num_ts_splits: int = 0,  # 0 = disabled, >0 = number of consecutive time splits
    robustness_std_coeff: float = 0.1,  # multiplier for std penalty
    robustness_greater_is_better: bool = False,  # False for ICE (lower is better)
    # ----------------------------------------------------------------------------------------------------------------------------
    # model-specific params
    # ----------------------------------------------------------------------------------------------------------------------------
    cb_kwargs: dict = None,
    hgb_kwargs: dict = None,
    lgb_kwargs: dict = None,
    xgb_kwargs: dict = None,
    mlp_kwargs: dict = None,
    ngb_kwargs: dict = None,
    # First-class predict-time MLP batch override. When None (default) the
    # wrapper auto-adapts to memory + input width. Plumbed in from
    # ``ModelHyperparamsConfig.mlp_predict_batch_size`` so callers don't
    # need to dig into ``mlp_kwargs["datamodule_params"]``.
    mlp_predict_batch_size: Optional[int] = None,
    # ----------------------------------------------------------------------------------------------------------------------------
    # featureselectors
    # ----------------------------------------------------------------------------------------------------------------------------
    rfecv_kwargs: dict = None,
    # ----------------------------------------------------------------------------------------------------------------------------
    # Skip MLP-config build (incl. ~14s pytorch / lightning import on first
    # call) when MLP isn't in the requested model list. Default ``None``
    # preserves legacy behaviour (build all configs).
    # ----------------------------------------------------------------------------------------------------------------------------
    enabled_models: Optional[Sequence[str]] = None,
) -> tuple:
    """Returns comparable training configs for different types of models,
    based on general params supplied like learning rate, task type, time budget.
    Useful for more or less fair comparison between different models on the same data/task, and their upcoming ensembling.
    This procedure is good for getting the feeling of what ML models are capable of for a particular task.
    """

    if has_gpu is None:
        has_gpu = CUDA_IS_AVAILABLE

    # Initialize mutable defaults
    if catboost_custom_classif_metrics is None:
        catboost_custom_classif_metrics = ["AUC", "BrierScore", "PRAUC"]
    if catboost_custom_regr_metrics is None:
        catboost_custom_regr_metrics = ["RMSE", "MAPE"]

    # Initialize kwargs dicts with defaults, making copies to avoid mutating caller's dicts
    if cb_kwargs is None:
        cb_kwargs = dict(verbose=0)
    else:
        cb_kwargs = cb_kwargs.copy()  # Don't mutate caller's dict
    if lgb_kwargs is None:
        lgb_kwargs = dict(verbose=-1)
    else:
        lgb_kwargs = lgb_kwargs.copy()  # Don't mutate caller's dict
    if xgb_kwargs is None:
        xgb_kwargs = dict(verbosity=0)
    else:
        xgb_kwargs = xgb_kwargs.copy()  # Don't mutate caller's dict
    if hgb_kwargs is None:
        hgb_kwargs = dict(verbose=0)
    else:
        hgb_kwargs = hgb_kwargs.copy()
    if mlp_kwargs is None:
        mlp_kwargs = dict()
    else:
        mlp_kwargs = mlp_kwargs.copy()
    if ngb_kwargs is None:
        ngb_kwargs = dict(verbose=True)
    else:
        ngb_kwargs = ngb_kwargs.copy()

    # None = disabled (don't pass to model fit at all); 0 = auto (iterations // 3); int = as-is.
    early_stopping_disabled = early_stopping_rounds is None
    if not early_stopping_disabled and not early_stopping_rounds:
        early_stopping_rounds = max(2, iterations // 3)

    def neg_ovr_roc_auc_score(*args, **kwargs):
        return -roc_auc_score(*args, **kwargs, multi_class="ovr")

    # Build defaults, then let caller's kwargs override any of them
    # via .update(). Using **cb_kwargs for merge crashes when the
    # caller passes a key that's already in the defaults dict
    # (TypeError: got multiple values).
    # ``has_gpu`` reports nvidia-smi presence; the installed catboost wheel
    # may still be CPU-only (default PyPI Windows wheels). Confirm via
    # ``_cb_gpu_usable`` (one-shot tiny-fit probe, cached) so machines with
    # a working GPU but a CPU-only CB binary fall back to ``task_type="CPU"``
    # instead of crashing every fit with "Environment for task type [GPU]
    # not found". Skip the probe entirely when CB is not in scope - the
    # tiny CB-GPU fit costs ~150ms per process and is wasted on
    # linear/ridge/lgb/xgb-only suites. ``models_in_scope`` is a hint;
    # when None we keep the conservative behaviour and probe.
    # ``enabled_models`` carries raw ``mlframe_models`` entries -- a CatBoost model may be
    # an alias ("cb"/"catboost") OR an estimator instance (CatBoostClassifier()). A bare
    # ``str(m).lower()`` check mis-routes the instance (stringifies to "<...object at 0x..>")
    # and silently skips the GPU probe -> CB runs on CPU even with a working GPU. Route via
    # the strategy registry so both surfaces classify identically.
    from .strategies import is_catboost_model
    _cb_in_scope = (
        enabled_models is None
        or any(is_catboost_model(m) for m in enabled_models)
    )
    if has_gpu and _cb_in_scope:
        from .cb import _cb_gpu_usable as _cb_gpu_probe
        _cb_task = "GPU" if _cb_gpu_probe() else "CPU"
    else:
        _cb_task = "CPU"
    CB_GENERAL_PARAMS = dict(
        iterations=iterations,
        has_time=has_time,
        learning_rate=learning_rate,
        eval_fraction=(0.0 if use_explicit_early_stopping else validation_fraction),
        task_type=_cb_task,
        early_stopping_rounds=early_stopping_rounds,
        random_seed=random_seed,
        # metric_period=5: evaluate the custom eval metric every 5th boost
        # iteration instead of every iteration. On 1M-row multiclass with
        # the ICE calibration metric this cut suite wall from ~95s to ~60s
        # (the per-iter metric was 110ms and CB ran 350+ boost rounds).
        # Trade-off: early-stopping detects "best iteration" with a 5-iter
        # granularity instead of 1-iter; on a 100+-iter run this is a
        # negligible accuracy hit. CB caller can override via cb_kwargs.
        metric_period=5,
        # allow_writing_files=False: CatBoost's catboost_info/ JSON+TSV side-
        # outputs are unused by mlframe (we have our own metric/plot pipeline)
        # and clash with multi-process pytest-xdist on Windows (every worker
        # contends on the shared cwd path). Booster itself, SHAP, FI, and
        # get_evals_result() are unaffected. Re-enable via
        # cb_kwargs={"allow_writing_files": True, "train_dir": "<unique>"}.
        allow_writing_files=False,
    )
    CB_GENERAL_PARAMS.update(cb_kwargs)

    CB_CLASSIF = CB_GENERAL_PARAMS.copy()
    CB_CLASSIF.update({"eval_metric": def_classif_metric})
    # NOTE: custom_metric breaks sklearn.clone() - CatBoost modifies this param after init.
    # TODO 2026-05-21: Raise issue at https://github.com/catboost/catboost/issues
    # "custom_metric": tuple(catboost_custom_classif_metrics or [])

    CB_REGR = CB_GENERAL_PARAMS.copy()
    CB_REGR.update({"eval_metric": def_regr_metric})
    # NOTE: custom_metric breaks sklearn.clone() - CatBoost modifies this param after init.
    # TODO 2026-05-21: Raise issue at https://github.com/catboost/catboost/issues
    # "custom_metric": tuple(catboost_custom_regr_metrics or [])

    HGB_GENERAL_PARAMS = dict(
        max_iter=iterations,
        learning_rate=learning_rate,
        early_stopping=True,
        validation_fraction=(None if use_explicit_early_stopping else validation_fraction),
        n_iter_no_change=early_stopping_rounds,
        categorical_features="from_dtype",
        random_state=random_seed,
    )
    HGB_GENERAL_PARAMS.update(hgb_kwargs)

    # Device gating reflects XGB build-info, not just CUDA presence.
    # ``has_gpu and XGB_GPU_AVAILABLE`` skips ``cuda`` on CPU-only XGB
    # binaries (avoids per-fit ``Device is changed from GPU to CPU``
    # warning storm). Caller's xgb_kwargs.update overrides win for
    # advanced users.
    _xgb_device = "cuda" if (has_gpu and XGB_GPU_AVAILABLE) else "cpu"
    XGB_GENERAL_PARAMS = dict(
        n_estimators=iterations,
        learning_rate=learning_rate,
        enable_categorical=True,
        max_cat_to_onehot=1,
        max_cat_threshold=100,  # affects model size heavily when high cardinality cat features r present!
        tree_method="hist",
        device=_xgb_device,
        n_jobs=psutil.cpu_count(logical=False),
        early_stopping_rounds=early_stopping_rounds,
        # XGBoost's sklearn API + native trainer key the RNG on ``random_state``
        # (or native ``seed``); ``random_seed`` is silently ignored ("Parameters:
        # { random_seed } are not used"), so a requested seed was dropped and XGB
        # ran non-reproducibly while CB (random_seed) / LGB+HGB (random_state)
        # were correctly seeded.
        random_state=random_seed,
    )
    XGB_GENERAL_PARAMS.update(xgb_kwargs)
    # 2026-05-26 unified ES surface: XGB regression default eval_metric
    # is "rmse" upstream, but mlframe used to leave it implicit -- ES
    # then tracked whatever ``objective`` defaulted to. Stamp the
    # eval_metric explicitly from ``def_regr_metric`` (mapped to xgb
    # naming) so CB / LGB / XGB all early-stop on the same surface
    # the user requested. ``_apply_loss_recommendation_in_place``
    # routes around this when heavy-kurt detection wins.
    _XGB_METRIC_NAME = {
        "RMSE": "rmse", "MAE": "mae", "Huber": "mphe",
        "MSLE": "rmsle", "MAPE": "mape",
    }.get(def_regr_metric, def_regr_metric.lower())
    XGB_GENERAL_PARAMS.setdefault("eval_metric", _XGB_METRIC_NAME)

    XGB_GENERAL_CLASSIF = XGB_GENERAL_PARAMS.copy()
    XGB_GENERAL_CLASSIF.update({"objective": "binary:logistic", "eval_metric": neg_ovr_roc_auc_score})

    # Target-type-aware objective injection. For non-binary classification
    # target types, replace the binary defaults with the native multi-output
    # objective. Binary path is a no-op (helper returns the same kwargs as
    # the explicit defaults above).
    from .configs import TargetTypes

    _resolved_tt = target_type if target_type is not None else TargetTypes.BINARY_CLASSIFICATION
    if _resolved_tt.is_classification and not _resolved_tt.is_binary:
        # Non-binary classification: inject native objective per library.
        cb_obj = _classif_objective_kwargs("catboost", _resolved_tt, n_classes)
        xgb_obj = _classif_objective_kwargs("xgboost", _resolved_tt, n_classes)
        lgb_obj = _classif_objective_kwargs("lightgbm", _resolved_tt, n_classes)
        if cb_obj:
            CB_CLASSIF.update(cb_obj)
            # When loss_function=MultiLogloss (multilabel), CatBoost REJECTS
            # eval_metric='AUC' with "metric AUC and loss MultiLogloss are
            # incompatible". Override to HammingLoss for MultiLogloss,
            # Accuracy for MultiClass. Caller can still override via cb_kwargs.
            if cb_obj.get("loss_function") == "MultiLogloss":
                CB_CLASSIF["eval_metric"] = "HammingLoss"
            elif cb_obj.get("loss_function") == "MultiClass":
                CB_CLASSIF["eval_metric"] = "Accuracy"
        if xgb_obj:
            # For multiclass, multi:softprob conflicts with binary metric.
            # Strip the binary eval_metric — caller can re-set if needed.
            XGB_GENERAL_CLASSIF.update(xgb_obj)
            # XGB multiclass eval_metric: mlogloss aligns with multi:softprob.
            # (binary binary_logloss / AUC don't apply.)
            if xgb_obj.get("objective") == "multi:softprob":
                XGB_GENERAL_CLASSIF["eval_metric"] = "mlogloss"
        if lgb_obj:
            # LGB_GENERAL_PARAMS gets the multiclass objective too — it has
            # no separate _CLASSIF variant currently.
            pass  # applied to LGB after LGB_GENERAL_PARAMS is built (below)
        # NOTE: no _mlframe_target_type metadata tag is attached here.
        # CatBoostClassifier init raises TypeError on unknown kwargs, which
        # would block the entire multilabel path. Downstream observability
        # (which lib + target_type) is covered by the per-model
        # model_schemas metadata record populated in core.py around the
        # fit call.

    def integral_calibration_error(y_true: np.ndarray, y_score: np.ndarray, verbose: bool = False) -> float:
        """Compute integral calibration error for probabilistic predictions.

        Wraps compute_probabilistic_multiclass_error with the outer function's
        configuration parameters (method, weights, etc.).

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.
        y_score : np.ndarray
            Predicted probabilities.
        verbose : bool, default=False
            If True, print calibration error info.

        Returns
        -------
        float
            The computed calibration error (lower is better).
        """
        err = compute_probabilistic_multiclass_error(
            y_true=y_true,
            y_score=y_score,
            method=method,
            mae_weight=mae_weight,
            std_weight=std_weight,
            brier_loss_weight=brier_loss_weight,
            roc_auc_weight=roc_auc_weight,
            pr_auc_weight=pr_auc_weight,
            min_roc_auc=min_roc_auc,
            roc_auc_penalty=roc_auc_penalty,
            use_weighted_calibration=use_weighted_calibration,
            weight_by_class_npositives=weight_by_class_npositives,
            nbins=nbins,
            verbose=verbose,
        )
        if verbose:
            logger.debug("integral_calibration_error=%s (n=%d)", err, len(y_true))
        return err

    def make_robust_ts_metric(
        metric_fn,
        num_splits: int,
        std_coeff: float,
        greater_is_better: bool,
        min_samples_per_split: int = 100,
        ensure_enough_classes: bool = False,
        verbose: int = 0,
    ):
        """Wrap a metric to evaluate across consecutive time splits.

        Returns mean(metric_values) ± std(metric_values) * std_coeff
        where ± is + if greater_is_better=False (penalize variance for minimization)
              and - if greater_is_better=True (penalize variance for maximization)
        """

        def robust_metric(y_true: np.ndarray, y_score: np.ndarray, *args, **kwargs):
            n = len(y_true)

            # Fallback 1: Not enough data for any splits
            if n < min_samples_per_split:
                if verbose:
                    logger.info("make_robust_ts_metric: n=%s < min_samples_per_split=%s, using full data", n, min_samples_per_split)
                return metric_fn(y_true, y_score, *args, **kwargs)

            # Compute actual number of splits we can do
            actual_splits = min(num_splits, n // min_samples_per_split)

            # Fallback 2: Can only do 1 split
            if actual_splits <= 1:
                if verbose:
                    logger.info("make_robust_ts_metric: actual_splits=%s <= 1, using full data", actual_splits)
                return metric_fn(y_true, y_score, *args, **kwargs)

            # Split into consecutive intervals
            split_size = n // actual_splits
            values = []

            for i in range(actual_splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < actual_splits - 1 else n

                y_true_split = y_true[start_idx:end_idx]
                y_score_split = y_score[start_idx:end_idx]

                # Skip split if not enough samples
                if len(y_true_split) < min_samples_per_split:
                    if verbose:
                        logger.info("make_robust_ts_metric: split %s skipped, len=%d < %d", i, len(y_true_split), min_samples_per_split)
                    continue

                # Skip split if single class (classification only)
                if ensure_enough_classes and len(np.unique(y_true_split)) < 2:
                    if verbose:
                        logger.info("make_robust_ts_metric: split %s skipped, single class in y_true", i)
                    continue

                val = metric_fn(y_true_split, y_score_split, *args, **kwargs)
                if not np.isnan(val):
                    values.append(val)

            # Fallback 3: No valid splits computed
            if len(values) == 0:
                if verbose:
                    logger.info("make_robust_ts_metric: no valid splits, using full data")
                return metric_fn(y_true, y_score, *args, **kwargs)

            # Fallback 4: Only one valid split
            if len(values) == 1:
                if verbose:
                    logger.info("make_robust_ts_metric: only 1 valid split, returning %.6f", values[0])
                return values[0]

            mean_val = np.mean(values)
            std_val = np.std(values)

            if verbose:
                logger.info("make_robust_ts_metric: %d splits, mean=%.6f, std=%.6f", len(values), mean_val, std_val)

            # Penalize high variance
            if greater_is_better:
                # For maximization: subtract std penalty (lower result = worse)
                return mean_val - std_val * std_coeff
            else:
                # For minimization: add std penalty (higher result = worse)
                return mean_val + std_val * std_coeff

        return robust_metric

    if subgroups:

        def final_integral_calibration_error(y_true: np.ndarray, y_score: np.ndarray, *args, **kwargs):  # partial won't work with xgboost
            return robust_mlperf_metric(
                y_true,
                y_score,
                *args,
                metric=integral_calibration_error,
                higher_is_better=False,
                subgroups=subgroups,
                **kwargs,
            )

    else:
        final_integral_calibration_error = integral_calibration_error

    # Apply robustness wrapper if enabled
    if robustness_num_ts_splits > 0:
        final_integral_calibration_error = make_robust_ts_metric(
            final_integral_calibration_error,
            num_splits=robustness_num_ts_splits,
            std_coeff=robustness_std_coeff,
            greater_is_better=robustness_greater_is_better,
            ensure_enough_classes=True,  # ICE is for classification
            verbose=verbose,
        )

    def fs_and_hpt_integral_calibration_error(*args, verbose: bool = True, **kwargs):
        err = compute_probabilistic_multiclass_error(
            *args,
            **kwargs,
            mae_weight=mae_weight,
            std_weight=std_weight,
            brier_loss_weight=brier_loss_weight,
            roc_auc_weight=roc_auc_weight,
            pr_auc_weight=pr_auc_weight,
            min_roc_auc=min_roc_auc,
            roc_auc_penalty=roc_auc_penalty,
            use_weighted_calibration=use_weighted_calibration,
            weight_by_class_npositives=weight_by_class_npositives,
            nbins=nbins,
            verbose=verbose,
        )
        return err

    XGB_CALIB_CLASSIF = XGB_GENERAL_CLASSIF.copy()
    XGB_CALIB_CLASSIF.update({"eval_metric": final_integral_calibration_error})

    def lgbm_integral_calibration_error(y_true, y_score):
        metric_name = "integral_calibration_error"
        value = final_integral_calibration_error(y_true, y_score)
        higher_is_better = False
        return metric_name, value, higher_is_better

    CB_CALIB_CLASSIF = CB_CLASSIF.copy()
    # ICE custom-metric only works for single-target CatBoost objectives
    # (binary / multiclass). For MultiLogloss (multilabel) CB asserts that
    # the custom metric inherits from MultiTargetCustomMetric. Until we
    # ship a multi-target ICE variant, fall back to HammingLoss for
    # multilabel (same as CB_CLASSIF so calibrated path == base path).
    if _resolved_tt.is_classification and not _resolved_tt.is_binary and CB_CLASSIF.get("loss_function") == "MultiLogloss":
        # eval_metric already set to HammingLoss above; keep it.
        pass
    else:
        CB_CALIB_CLASSIF.update({"eval_metric": ICE(metric=final_integral_calibration_error, higher_is_better=False, max_arr_size=0)})

    # Same gating story as XGB. ``has_gpu and LGB_GPU_AVAILABLE`` respects
    # the LightGBM build's actual GPU support (default LightGBM wheels are
    # CPU-only; opt-in via env ``MLFRAME_TRUST_LGB_CUDA=1`` if you've built
    # / installed a GPU-enabled LGB binary).
    _lgb_device = "cuda" if (has_gpu and LGB_GPU_AVAILABLE) else "cpu"
    LGB_GENERAL_PARAMS = dict(
        n_estimators=iterations,
        # 2026-05-23: explicit ``learning_rate`` plumbed so the suite's
        # ``learning_rate=`` kwarg actually reaches LightGBM (pre-fix LGB
        # silently inherited library default 0.1 which happened to match
        # the suite default but broke the override surface contract --
        # users passing learning_rate=0.05 got LGB at 0.1).
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds,
        device_type=_lgb_device,
        random_state=random_seed,
        # histogram_pool_size=16384,
    )
    LGB_GENERAL_PARAMS.update(lgb_kwargs)
    # 2026-05-26 unified ES surface (companion to the XGB metric stamp):
    # LGB regression default ``metric`` is unset which means "use
    # objective" -- on RMSE that's fine, but on MAE / Huber objective
    # the ES surface drifts from CB's. Stamp ``metric`` explicitly so
    # all three boosters track the same surface.
    #
    # Stamp the regression metric ONLY when the suite knows the target is
    # regression (or unspecified -- legacy callers). For known-
    # classification targets, leave ``metric`` unset and let LightGBM
    # auto-pick a compatible metric from the inferred objective
    # (binary_logloss for binary, multi_logloss for multiclass). The bug
    # this guards against: LightGBM's LGBMClassifier auto-promotes to
    # ``objective='multiclass'`` based on y's class cardinality at fit
    # time -- so even if the suite never set MULTICLASS_CLASSIFICATION
    # explicitly upstream, a 3-class y still triggers the multiclass
    # path. With a stale regression metric=l2 on the params, the
    # multiclass fit raises "Multiclass objective and metrics don't
    # match" before training starts.
    _is_regression_path = (
        _resolved_tt is None or not _resolved_tt.is_classification
    )
    if _is_regression_path:
        _LGB_METRIC_NAME = {
            "RMSE": "l2", "MAE": "l1", "Huber": "huber",
            "MAPE": "mape", "MSLE": "rmse",  # LGB has no msle; fall back
        }.get(def_regr_metric, def_regr_metric.lower())
        LGB_GENERAL_PARAMS.setdefault("metric", _LGB_METRIC_NAME)
    # Target-type-aware objective for LGB (no separate _CLASSIF variant).
    if _resolved_tt.is_classification and not _resolved_tt.is_binary:
        _lgb_obj = _classif_objective_kwargs("lightgbm", _resolved_tt, n_classes)
        if _lgb_obj:
            LGB_GENERAL_PARAMS.update(_lgb_obj)
            # Even though the regression-metric setdefault is now gated to
            # the regression branch, callers can still pass an explicit
            # regression-shaped metric in ``lgb_kwargs`` (legacy callers
            # that hardcoded ``metric="l2"`` before the gate landed). Force
            # a multiclass-compatible metric in the multiclass path so the
            # objective <-> metric contract holds regardless of caller history.
            if _lgb_obj.get("objective") == "multiclass":
                LGB_GENERAL_PARAMS["metric"] = "multi_logloss"
            LGB_GENERAL_PARAMS["_mlframe_target_type"] = str(_resolved_tt.value)

    NGB_GENERAL_PARAMS = dict(
        n_estimators=iterations,
        learning_rate=learning_rate,
    )
    NGB_GENERAL_PARAMS.update(ngb_kwargs)

    # When caller declares which models are in scope AND mlp / recurrent
    # are NOT among them, skip the entire MLP config block (saves ~14s of
    # pytorch + lightning import overhead on the first call to
    # get_training_configs in a process). Any caller that asks for
    # MLP_GENERAL_PARAMS later will get None.
    # Same instance-vs-alias hazard as the CB probe above: a neural model may be an alias
    # ("mlp"/"recurrent"/"lstm"/...) OR a torch estimator instance. A name-tuple membership
    # test misses the instance and skips the entire MLP config block -> a neural model
    # gets no config. Route via the strategy registry.
    from .strategies import is_neural_model
    _mlp_in_scope = (
        enabled_models is None
        or any(is_neural_model(m) for m in enabled_models)
    )

    if not _mlp_in_scope:
        # Skip the heavy MLP path entirely. Downstream consumers must
        # not assume MLP_GENERAL_PARAMS is non-None when mlp isn't
        # requested -- the dispatch in trainer.py's _get_mlp_imports
        # gates on the presence of 'mlp' in mlframe_models already.
        MLP_GENERAL_PARAMS = None
    else:
        # Note: ``num_sanity_val_steps`` is intentionally left at
        # Lightning's default (2). An A/B benchmark on the tightest MLP
        # combo (cb+mlp pandas n=300 multiclass) showed setting it to 0
        # saves only 32 ms / 1.7% per fit (within combined stdev) because
        # the "Sanity Checking" pass is just the first forward through
        # cuDNN's auto-tuner; disabling it pushes the same kernel
        # selection cost onto epoch 0 instead. Keeping the default
        # preserves Lightning's fail-fast on a broken val pipeline at no
        # measurable cost.
        # AMP precision is auto-resolved on Ampere+ CUDA hosts (RTX
        # 30/40, A100, H100) to "bf16-mixed" for ~2x throughput on
        # tabular MLPs. Falls back to "32-true" on older GPUs / CPU. The
        # user override under mlp_kwargs["trainer_params"]["precision"]
        # always wins.
        from .mlp_runtime_defaults import resolve_mlp_precision_default
        _user_precision = (
            (mlp_kwargs or {}).get("trainer_params", {}).get("precision")
        )
        _resolved_precision = resolve_mlp_precision_default(
            user_override=_user_precision,
        )

        mlp_trainer_params: dict = dict(
            devices=1,  # Always use single device by default to avoid multi-GPU complexity
            # ------------------------------------------------------------------
            # Runtime:
            # ------------------------------------------------------------------
            min_epochs=1,
            max_epochs=iterations,
            # 2h cap (raised from a prior 30-min default that cut a 4M-row
            # regression at ~6 epochs). The caller-side
            # ``early_stopping_rounds`` (default 100 epochs of no val
            # improvement) still terminates well before this on healthy
            # training. Override via
            # ``mlp_kwargs["trainer_params"]["max_time"]``.
            max_time={"days": 0, "hours": 2, "minutes": 0},
            # ------------------------------------------------------------------
            # Intervals:
            # ------------------------------------------------------------------
            check_val_every_n_epoch=1,
            # ------------------------------------------------------------------
            # Flags:
            # ------------------------------------------------------------------
            enable_model_summary=False,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            accumulate_grad_batches=2,
            # ------------------------------------------------------------------
            # Precision & accelerators:
            # ------------------------------------------------------------------
            precision=_resolved_precision,
            num_nodes=1,
            # ------------------------------------------------------------------
            # Logging:
            # ------------------------------------------------------------------
            default_root_dir="logs",
        )

        if mlp_kwargs:
            mlp_trainer_params.update(mlp_kwargs.get("trainer_params", {}))

        # Lazy imports -- only paid when MLP configs are actually being built.
        import torch
        import torch.nn.functional as F
        from mlframe.training.neural.flat import MLPTorchModel
        from mlframe.training.neural.data import TorchDataModule

        # Default loss function and dtype (classification)
        loss_fn = F.cross_entropy
        labels_dtype = torch.int64

        # Defaults: Adam + LR=3e-3 (chosen after a prod-failure root cause
        # analysis; was previously AdamW + LR=1e-3).
        #
        # Why Adam (not AdamW): AdamW's built-in weight_decay=0.01
        # penalises large weights -- which is EXACTLY what an MLP needs
        # to learn a near-linear target with one dominant feature (y =
        # 0.95 * lag_feature + small_residual requires a weight of ~0.95
        # on the dominant input). Weight decay shrinks that weight every
        # step, fighting the loss. Adam (no decay) is the safer default
        # for tabular regression with strong linear / additive signal.
        # Users whose dataset is overfit-prone can opt back in via
        # ``mlp_kwargs["model_params"]["optimizer"]=torch.optim.AdamW``.
        #
        # Why LR=3e-3 (not 1e-3): with the new zero-dropout architecture,
        # gradient flow is unobstructed; the larger LR converges in ~1/3
        # the epochs without overshoot. On a 2-hour prod run with
        # LR=1e-3 + dropout=0.15, MLP plateaued at val_MSE=0.7555 after 9
        # epochs (out of ~20 the time budget allowed); with LR=3e-3 +
        # dropout=0, the same architecture converges to val_MSE ~ 0.15
        # in similar epoch count -- matching linear regression on a
        # near-linear DGP.
        mlp_model_params = dict(
            loss_fn=loss_fn,
            learning_rate=3e-3,
            l1_alpha=0.0,
            optimizer=torch.optim.Adam,
            optimizer_kwargs={},
            lr_scheduler=None,
            lr_scheduler_kwargs={},
        )
        if mlp_kwargs:
            mlp_model_params.update(mlp_kwargs.get("model_params", {}))

        # Cross-platform-safe DataLoader defaults. num_workers stays at 0
        # on EVERY OS because ``TorchDataset`` keeps the full input frame
        # in self.features. On Windows that means each worker pickles
        # 100 GB; on Linux fork's CoW gets broken by Polars indexing +
        # Python refcount writes -> swap death. pin_memory still defaults
        # ON for CUDA hosts (no IPC landmine). User opts in to workers via
        # ``mlp_kwargs["dataloader_params"]["num_workers"]`` once their
        # specific dataset is verified to fit each worker's memory budget.
        from .mlp_runtime_defaults import resolve_mlp_dataloader_defaults
        _user_dataloader_overrides = (
            (mlp_kwargs or {}).get("dataloader_params", {}) or {}
        )
        _resolved_dataloader_extras = resolve_mlp_dataloader_defaults(
            user_overrides=_user_dataloader_overrides,
        )

        mlp_dataloader_params = dict(
            sampler=None,
            batch_sampler=None,
            num_workers=_resolved_dataloader_extras["num_workers"],
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=_resolved_dataloader_extras["prefetch_factor"],
            persistent_workers=_resolved_dataloader_extras["persistent_workers"],
            pin_memory=_resolved_dataloader_extras["pin_memory"],
            batch_size="auto",
            shuffle=False,
        )
        if mlp_kwargs:
            mlp_dataloader_params.update(mlp_kwargs.get("dataloader_params", {}))

        mlp_datamodule_params = dict(
            read_fcn=None, data_placement_device=None,
            features_dtype=torch.float32, labels_dtype=labels_dtype,
            dataloader_params=mlp_dataloader_params,
        )
        # Plumb the suite-level predict-batch-size knob through the
        # datamodule; the wrapper's _predict_raw consults this when no
        # explicit batch_size is passed to .predict(). None (default) lets
        # the adaptive resolver pick based on memory + dataframe width.
        if mlp_predict_batch_size is not None:
            mlp_datamodule_params["predict_batch_size"] = int(mlp_predict_batch_size)
        if mlp_kwargs:
            mlp_datamodule_params.update(mlp_kwargs.get("datamodule_params", {}))

        MLP_GENERAL_PARAMS = dict(
            model_class=MLPTorchModel,
            model_params=mlp_model_params,
            datamodule_class=TorchDataModule,
            datamodule_params=mlp_datamodule_params,
            trainer_params=mlp_trainer_params,
            use_swa=mlp_kwargs.get("use_swa", False) if mlp_kwargs else False,
            swa_params=(
                mlp_kwargs.get("swa_params", dict(swa_epoch_start=5, annealing_epochs=5, swa_lrs=1e-4))
                if mlp_kwargs
                else dict(swa_epoch_start=5, annealing_epochs=5, swa_lrs=1e-4)
            ),
            tune_params=mlp_kwargs.get("tune_params", False) if mlp_kwargs else False,
            float32_matmul_precision=mlp_kwargs.get("float32_matmul_precision", None) if mlp_kwargs else None,
            early_stopping_rounds=early_stopping_rounds,
        )

    if rfecv_kwargs is None:
        rfecv_kwargs = {}
    else:
        rfecv_kwargs = rfecv_kwargs.copy()

    cv = rfecv_kwargs.get("cv")
    _cv_n_splits = rfecv_kwargs.get("cv_n_splits")
    if not cv:
        if has_time:
            cv = TimeSeriesSplit(n_splits=_cv_n_splits if _cv_n_splits is not None else 3)
            logger.info(f"Using TimeSeriesSplit for RFECV...")
        elif _cv_n_splits is not None:
            # Non-time-series + user-supplied cv_n_splits: build a KFold so the kwarg actually
            # propagates instead of getting silently dropped. Prior code only honored cv_n_splits
            # on the time-series branch; users hitting the regular branch saw their tuning
            # ignored, sklearn fell back to its default 5-fold.
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=_cv_n_splits)
        else:
            cv = None
        rfecv_kwargs["cv"] = cv

    if "cv_n_splits" in rfecv_kwargs:
        del rfecv_kwargs["cv_n_splits"]

    COMMON_RFECV_PARAMS = dict(
        early_stopping_rounds=early_stopping_rounds,
        cv=cv,
        cv_shuffle=not has_time,
    )
    COMMON_RFECV_PARAMS.update(rfecv_kwargs)

    # If ES is disabled (early_stopping_rounds=None), strip the key from every per-model
    # constructor-params dict so backends don't register an ES callback.
    # - LGB: omitted from constructor → LightGBMSklearn skips ES on fit
    # - XGB: omitted from constructor → no early_stopping_rounds passed
    # - CB:  omitted → CatBoost runs full iterations (no od_type)
    # - HGB: replace n_iter_no_change with iterations+1 so ES condition never trips
    if early_stopping_disabled:
        for _params in (CB_GENERAL_PARAMS, CB_REGR, CB_CLASSIF, CB_CALIB_CLASSIF,
                        LGB_GENERAL_PARAMS, XGB_GENERAL_PARAMS,
                        XGB_GENERAL_CLASSIF, XGB_CALIB_CLASSIF,
                        MLP_GENERAL_PARAMS, COMMON_RFECV_PARAMS):
            if _params is not None:  # MLP_GENERAL_PARAMS may be None when MLP not in scope
                _params.pop("early_stopping_rounds", None)
        # HGB uses early_stopping=True + n_iter_no_change; force ES off explicitly
        HGB_GENERAL_PARAMS["early_stopping"] = False
        HGB_GENERAL_PARAMS.pop("n_iter_no_change", None)

    return SimpleNamespace(
        integral_calibration_error=integral_calibration_error,
        final_integral_calibration_error=final_integral_calibration_error,
        lgbm_integral_calibration_error=lgbm_integral_calibration_error,
        fs_and_hpt_integral_calibration_error=fs_and_hpt_integral_calibration_error,
        CB_GENERAL_PARAMS=CB_GENERAL_PARAMS,
        CB_REGR=CB_REGR,
        CB_CLASSIF=CB_CLASSIF,
        CB_CALIB_CLASSIF=CB_CALIB_CLASSIF,
        HGB_GENERAL_PARAMS=HGB_GENERAL_PARAMS,
        LGB_GENERAL_PARAMS=LGB_GENERAL_PARAMS,
        XGB_GENERAL_PARAMS=XGB_GENERAL_PARAMS,
        XGB_GENERAL_CLASSIF=XGB_GENERAL_CLASSIF,
        XGB_CALIB_CLASSIF=XGB_CALIB_CLASSIF,
        COMMON_RFECV_PARAMS=COMMON_RFECV_PARAMS,
        MLP_GENERAL_PARAMS=MLP_GENERAL_PARAMS,
        NGB_GENERAL_PARAMS=NGB_GENERAL_PARAMS,
    )


def disable_native_es_for_slice_stable(configs: Any) -> None:
    """Strip native early-stopping settings from per-booster params when slice-stable ES is active.

    When the suite layer registers the slice-stable callback for ES (it aggregates per-shard
    scores into a single stop signal), the booster's *own* early-stopping must not race against
    that decision. CatBoost in particular fires its native ES off the FIRST registered eval_set
    (the full val) on a single-metric basis and can stop training BEFORE the callback's
    aggregate would; LGB/XGB are less aggressive but also expose the same hazard.

    Mutates the configs in-place. Mirrors the ``early_stopping_disabled`` branch above so the
    two paths converge on the same booster behaviour.
    """
    for _params in (
        getattr(configs, "CB_GENERAL_PARAMS", None),
        getattr(configs, "CB_REGR", None),
        getattr(configs, "CB_CLASSIF", None),
        getattr(configs, "CB_CALIB_CLASSIF", None),
        getattr(configs, "LGB_GENERAL_PARAMS", None),
        getattr(configs, "XGB_GENERAL_PARAMS", None),
        getattr(configs, "XGB_GENERAL_CLASSIF", None),
        getattr(configs, "XGB_CALIB_CLASSIF", None),
        getattr(configs, "MLP_GENERAL_PARAMS", None),
        getattr(configs, "COMMON_RFECV_PARAMS", None),
    ):
        if _params is not None:
            _params.pop("early_stopping_rounds", None)
    hgb = getattr(configs, "HGB_GENERAL_PARAMS", None)
    if hgb is not None:
        hgb["early_stopping"] = False
        hgb.pop("n_iter_no_change", None)
