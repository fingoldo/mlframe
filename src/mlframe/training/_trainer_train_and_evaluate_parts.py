"""Helpers carved out of ``_trainer_train_and_evaluate`` to keep that module under its size budget."""
from __future__ import annotations

import logging  # nosec B403 - module used safely in this file, see call sites below (no untrusted input reaches it)
import pickle  # nosec B403 - pickle used only for trusted same-process/dev-local round-trips, see call sites in this file
import os
from os.path import exists
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    pass

import numpy as np
import pandas as pd
import polars as pl

from mlframe.training.io import safe_joblib_load
from .utils import maybe_clean_ram_adaptive as _maybe_clean_ram

# Heavy optional deps: defer failures to first actual use so `import mlframe.training` stays cheap and does not crash when a given backend is not installed.
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


# Optional model backends: lazy/tolerant of missing deps.
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostRegressor = CatBoostClassifier = None
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = LGBMRegressor = None  # type: ignore[assignment,misc]
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover
    XGBClassifier = XGBRegressor = None  # type: ignore[assignment,misc]

from ._eval_helpers import (  # noqa: F401  -- carved helpers
    _align_xgb_cat_categories, _append_split_rate_suffix,
    _compute_split_metrics, _decategorise_float_cat_columns,
    _filter_categorical_features, run_confidence_analysis,
)
from ._data_helpers import (  # noqa: F401  -- carved helpers
    _setup_eval_set, _setup_early_stopping_callback,
)

logger = logging.getLogger("mlframe.training.trainer")

# Split-metrics emitters carved into a sibling to keep this orchestration module under the LOC budget.


logger = logging.getLogger("mlframe.training.trainer")


def _oof_train_timestamps(timestamps: Any, train_idx: Any) -> Any:
    """Timestamps of the train rows the OOF pass folds over (None when the suite has none); a pandas Series stays one so a tz-aware dtype survives."""
    if timestamps is None:
        return None
    if train_idx is None:
        return timestamps
    if hasattr(timestamps, "iloc"):
        return timestamps.iloc[np.asarray(train_idx)]
    return np.asarray(timestamps)[np.asarray(train_idx)]


def _train_envelope_stats(train_target: Any, mase_seasonality: int) -> Any:
    """Train-target envelope stats (with the naive MAE at ``mase_seasonality``), or None when ``train_target`` is None or they cannot be computed.

    A failure is logged at DEBUG: the per-split eval-fallback envelope still applies in the reporter.
    """
    if train_target is None:
        return None
    try:
        from ._prediction_envelope_clip import compute_train_envelope_stats, train_naive_mae

        stats = compute_train_envelope_stats(train_target)
        if stats is not None:
            stats = stats._replace(naive_mae=train_naive_mae(train_target, mase_seasonality))
        return stats
    except Exception as err:
        logger.debug("Could not compute train envelope stats: %s. Per-split eval-fallback envelope still applies in the reporter.", err)
        return None


def _train_and_evaluate_into_non_booster_fits(model_category, fit_params, _mono_patience_cfg):
    """Block of train_and_evaluate_model starting at ``if model_category in ("lgb", "xgb"):``."""
    if model_category in ("lgb", "xgb"):
        if fit_params is None:
            fit_params = {}
        fit_params.setdefault("monotonic_decline_patience", _mono_patience_cfg)
    return fit_params


def _train_and_evaluate_cap_iter_boosters(_cap_iter_boosters, model_category, callback_params, _iter_stride_cfg, fit_params):
    """Block of train_and_evaluate_model starting at ``if _cap_iter_boosters:``."""
    if _cap_iter_boosters:
        if model_category == "cb" or callback_params:
            callback_params = dict(callback_params or {})
            callback_params.setdefault("capture_iteration_metrics", True)
            callback_params.setdefault("iteration_metrics_stride", _iter_stride_cfg)
        if model_category in ("lgb", "xgb"):
            if fit_params is None:
                fit_params = {}
            fit_params.setdefault("capture_iteration_metrics", True)
            fit_params.setdefault("iteration_metrics_stride", _iter_stride_cfg)
    return callback_params, fit_params


def _train_and_evaluate_use_cache_exists_model(use_cache, model_file_name, trusted_root, model, pre_pipeline):
    """Block of train_and_evaluate_model starting at ``if use_cache and exists(model_file_name):``."""
    from mlframe.training.trainer import _validate_trusted_path

    if use_cache and exists(model_file_name):
        logger.info("Loading model from file %s", model_file_name)
        # Security: verify the model path is inside a trusted root before the joblib.load (pickle). Default `trusted_root` to the model file's parent dir when
        # not provided, preserving backward compat for the in-process trained-then-loaded flow (the trainer wrote this file itself). RESIDUAL RISK (audit2 F2):
        # path-containment cannot catch a pickle an attacker plants AT the expected model_file_name (it's exactly where we look) -- only integrity can.
        # safe_joblib_load below closes the RCE-gadget half (denylists eval/exec/os/subprocess/etc. reconstructors), but NOT the authenticity half: the complete
        # fix is to write + verify a sha256 sidecar around this save/load pair via utils.safe_pickle.safe_load (fail-closed on a missing/mismatched sidecar);
        # tracked as an owned follow-up since it needs the paired SAVE site (not in this file) to emit the sidecar first. Until then this default only blocks
        # gross path escapes and known RCE gadgets, not a planted-at-path pickle built from an otherwise-permitted class.
        _root = trusted_root if trusted_root is not None else os.path.dirname(os.path.abspath(model_file_name))
        _validate_trusted_path(model_file_name, _root)
        try:
            model, *_, pre_pipeline = safe_joblib_load(model_file_name)
        except (EOFError, OSError, ModuleNotFoundError, pickle.UnpicklingError, AttributeError):
            # retraining is expensive; preserve traceback so the
            # operator can distinguish pickle-version mismatch / torch attribute drift /
            # disk corruption rather than re-investigating after each fallback.
            logger.warning("Failed to load cached model from %s; will retrain instead.", model_file_name, exc_info=True)
    return model, pre_pipeline


def _train_and_evaluate_df_none_train_df(df, train_df, train_idx, real_drop_columns, val_df, val_idx):
    """Block of train_and_evaluate_model starting at ``if (df is not None) or (train_df is not None):``."""
    from mlframe.training.trainer import _subset_dataframe

    if (df is not None) or (train_df is not None):
        if train_df is None:
            train_df = _subset_dataframe(df, train_idx, real_drop_columns)
        if val_df is None and val_idx is not None:
            val_df = _subset_dataframe(df, val_idx, real_drop_columns)
    return train_df, val_df


def _train_and_evaluate_range_no_test_fix(group_ids, train_idx, train_df, _pre_pipeline_groups):
    """Block of train_and_evaluate_model starting at ``if group_ids is not None:``."""
    if group_ids is not None:
        try:
            _gi = np.asarray(group_ids)
            if train_idx is not None and len(_gi) >= int(np.max(np.asarray(train_idx))) + 1:
                _pre_pipeline_groups = _gi[np.asarray(train_idx)]
            elif train_df is not None and hasattr(train_df, "shape") and len(_gi) == train_df.shape[0]:
                _pre_pipeline_groups = _gi
        except (TypeError, ValueError, IndexError):
            _pre_pipeline_groups = None
    return _pre_pipeline_groups


def _train_and_evaluate_forwarding_happens_through_apply(sample_weight, train_idx, _pre_pipeline_sample_weight):
    """Block of train_and_evaluate_model starting at ``if sample_weight is not None:``."""
    if sample_weight is not None:
        try:
            if isinstance(sample_weight, (pd.Series, pd.DataFrame)):
                if train_idx is not None:
                    _pre_pipeline_sample_weight = np.asarray(sample_weight.iloc[train_idx].values, dtype=np.float64)
                else:
                    _pre_pipeline_sample_weight = np.asarray(sample_weight.values, dtype=np.float64)
            else:
                _sw_arr = np.asarray(sample_weight, dtype=np.float64)
                if train_idx is not None:
                    _pre_pipeline_sample_weight = _sw_arr[train_idx]
                else:
                    _pre_pipeline_sample_weight = _sw_arr
        except (TypeError, ValueError, IndexError):
            _pre_pipeline_sample_weight = None
    return _pre_pipeline_sample_weight


def _train_and_evaluate_model_none_pre_pipeline(model, pre_pipeline, skip_pre_pipeline_transform, train_df, val_df, _orig_train_df, _orig_val_df):
    """Block of train_and_evaluate_model starting at ``if model is not None and pre_pipeline and not skip_pre_pipeline_transf``."""
    if model is not None and pre_pipeline and not skip_pre_pipeline_transform:
        _orig_train_df = train_df
        if val_df is not None:
            _orig_val_df = val_df
    return _orig_train_df, _orig_val_df


def _train_and_evaluate_val_df_none(val_df, val_target, control, model_category, sample_weight, val_idx, group_ids, callback_params, model_obj, model_type_name, verbose, fit_params, model, oof_random_seed):
    """Block of train_and_evaluate_model starting at ``if val_df is not None:``."""
    from mlframe.training.trainer import _disable_xgboost_early_stopping_if_needed

    if val_df is not None:
        if isinstance(val_target, pl.Series):
            val_target = val_target.to_numpy()

        # Slice-stable ES integration (opt-in; ``control.slice_stable_es`` is None for legacy path).
        # When enabled we build per-shard eval-sets from the val frame, inject the slice-aggregator
        # knobs into ``callback_params`` so the UniversalCallback aggregates positionally, and
        # strip the booster's native early_stopping_rounds so it doesn't race the callback.
        # HGB/NGB use a single (X_val, y_val) pair (``value_format='separate'``); when slice-ES
        # is configured for these we fall back to the legacy single-val path per ``on_unsupported``.
        extra_eval_sets = None
        _slice_cfg = getattr(control, "slice_stable_es", None)
        # Slice ES infrastructure activates when either ``enabled`` (drives ES decisions) or
        # ``diagnostic_only`` (per-shard logging + Pareto plot without changing ES) is set.
        _slice_active = _slice_cfg is not None and (getattr(_slice_cfg, "enabled", False) or getattr(_slice_cfg, "diagnostic_only", False))
        _slice_diag_only = _slice_cfg is not None and getattr(_slice_cfg, "diagnostic_only", False) and not getattr(_slice_cfg, "enabled", False)
        if _slice_active:
            from mlframe.training.slicing import build_slice_eval_sets
            _supports_multi_eval = model_category in {"cb", "lgb", "xgb"}
            _policy = getattr(_slice_cfg, "on_unsupported", "posthoc")
            if _supports_multi_eval:
                try:
                    val_target_arr = val_target.values if hasattr(val_target, "values") else val_target
                    _sw_val = sample_weight[val_idx] if (sample_weight is not None and val_idx is not None) else None
                    _grp_val = group_ids[val_idx] if (group_ids is not None and val_idx is not None) else None
                    extra_eval_sets = build_slice_eval_sets(
                        val_df, val_target_arr,
                        source=getattr(_slice_cfg, "source", "random"),
                        k=int(getattr(_slice_cfg, "k", 5)),
                        min_rows_per_shard=int(getattr(_slice_cfg, "min_rows_per_shard", 100)),
                        random_state=int(getattr(_slice_cfg, "random_state", 42)),
                        sample_weight=_sw_val,
                        group_ids=_grp_val,
                    )
                    if extra_eval_sets:
                        callback_params = dict(callback_params or {})
                        callback_params.update({
                            "slice_k": len(extra_eval_sets),
                            "slice_aggregate_mode": getattr(_slice_cfg, "aggregate", "mean"),
                            "slice_aggregate_alpha": float(getattr(_slice_cfg, "alpha", 1.0)),
                            "slice_aggregate_confidence": float(getattr(_slice_cfg, "confidence", 0.9)),
                            "slice_aggregate_quantile_level": float(getattr(_slice_cfg, "quantile_level", 0.9)),
                            "slice_correlation_inflation": float(getattr(_slice_cfg, "correlation_inflation", 1.5)),
                            "slice_min_delta_in_se": getattr(_slice_cfg, "min_delta_in_se", None),
                            "slice_persist_history": bool(getattr(_slice_cfg, "pareto_plot_enabled", True))
                                                     or bool(getattr(_slice_cfg, "pareto_best_iter_selection", False))
                                                     or bool(getattr(_slice_cfg, "pareto_persist_shard_history", False)),
                            "slice_diagnostic_only": _slice_diag_only,
                        })
                        # Strip native ES rounds only when the slice callback OWNS the stop
                        # decision (``enabled=True``). In ``diagnostic_only`` mode the booster's
                        # native ES path stays intact -- slice ES is logging-only here.
                        if model_obj is not None and not _slice_diag_only:
                            try:
                                if "early_stopping_rounds" in model_obj.get_params():
                                    model_obj.set_params(early_stopping_rounds=None)
                            except Exception as _e_strip_es:  # nosec B110 - non-trivial body
                                logger.debug(
                                    "%s does not expose set_params for early_stopping_rounds (%s); native ES stays wired alongside slice ES",
                                    model_type_name, _e_strip_es,
                                )
                except Exception as _slice_err:
                    logger.warning(
                        "slice-stable ES wiring failed for %s (%s); falling back to single-val path",
                        model_type_name, _slice_err,
                    )
                    extra_eval_sets = None
            else:
                if _policy == "raise":
                    raise ValueError(
                        f"slice-stable ES not supported for model_category={model_category!r}; "
                        f"set TrainingConfig.slice_stable_es.on_unsupported='posthoc' or 'skip'"
                    )
                if verbose:
                    logger.info(
                        "slice-stable ES skipped for model_category=%s (uses separate X_val/y_val kwargs); " "on_unsupported=%s",
                        model_category,
                        _policy,
                    )
        _setup_eval_set(
            model_type_name, fit_params, val_df, val_target, callback_params, model_obj, model_category,
            extra_eval_sets=extra_eval_sets,
            sample_weight_val=sample_weight[val_idx] if (sample_weight is not None and val_idx is not None) else None,
            group_ids_val=group_ids[val_idx] if (group_ids is not None and val_idx is not None) else None,
        )

        # Auto-wrap models whose category isn't natively wired into ``_setup_eval_set``
        # (linear / ridge / lasso / elasticnet / huber / sgd / ransac) in PartialFitESWrapper
        # so val drives ES via partial_fit (SGD-family) or dichotomic budget search (the
        # iterative-solver linear-family models). The wrapper transparently delegates
        # attribute access to the underlying estimator via ``__getattr__`` so downstream
        # feature-importance / calibration / SHAP code continues to read ``.coef_`` etc.
        # unchanged. Closed-form models with no usable budget knob (plain LinearRegression)
        # pass through untouched -- no ES is structurally possible there.
        from mlframe.training._data_helpers import maybe_wrap_for_partial_fit_es
        _behavior_kwargs: dict[str, Any] = {}
        _beh = getattr(getattr(control, "behavior", None), "__dict__", None)
        if _beh is None:
            # ``control`` is a TrainingControlConfig; suite-level TrainingBehaviorConfig may
            # live a level up.
            _auto_wrap = getattr(control, "auto_wrap_partial_fit_es", True)
        else:
            _auto_wrap = _beh.get("auto_wrap_partial_fit_es", True)
        # ``auto_wrap_partial_fit_es`` (TrainingBehaviorConfig field, default True)
        # gates the wrap entirely. False reaches the underlying estimator unchanged
        # -- intended for parity bench / off-switch use, not perf.
        if _auto_wrap:
            _wrapped, _did_wrap = maybe_wrap_for_partial_fit_es(
                model_obj if model is None else (model_obj or model),
                model_category=model_category or "",
                X_val=val_df, y_val=val_target,
                is_classification=("Classifier" in (model_type_name or "")),
                behavior_kwargs=_behavior_kwargs,
                random_state=int(oof_random_seed),
            )
        else:
            _wrapped, _did_wrap = None, False
        if _did_wrap:
            logger.info("Auto-wrapped %s in PartialFitESWrapper for val-driven ES " "(model_category=%s)", model_type_name, model_category)
            # Replace both ``model`` (used downstream for predict / metrics / FI) and
            # ``model_obj`` (used for set_params / get_params probes upstream) with the wrapper.
            # The wrapper's __getattr__ forwards everything to the underlying estimator.
            model = _wrapped
            model_obj = _wrapped
        _maybe_clean_ram()
    else:
        _disable_xgboost_early_stopping_if_needed(model_type_name, model_obj)
    return model, model_obj, val_target


def _train_and_evaluate_train_df_none(train_df, model_name, show_feature_names):
    """Block of train_and_evaluate_model starting at ``if train_df is not None:``."""
    if train_df is not None:
        report_title = f"Training {model_name} model on {train_df.shape[1]} feature(s)"
        if show_feature_names:
            report_title += ": " + ", ".join(list(train_df.columns))
        report_title += f", {len(train_df):_} records"


def _train_and_evaluate_nest_lightning_checkpoints_csv(model_file_name, model):
    """Block of train_and_evaluate_model starting at ``try:``."""
    try:
        if model_file_name:
            _ckpt_dir = os.path.splitext(model_file_name)[0]
            _inner = getattr(model, "regressor", model)
            if hasattr(_inner, "trainer_params"):
                _inner.checkpoint_dir_override = _ckpt_dir  # mlframe-injected bookkeeping attr on an arbitrary (object-typed) inner estimator
    except Exception as e:
        logger.debug("swallowed exception in _trainer_train_and_evaluate.py: %s", e)
        pass


def _train_and_evaluate_score_ensemble_can_pick(oof_n_splits, just_evaluate, model_type_name, train_target, model, train_df, oof_random_seed, group_ids, train_idx, oof_has_time, _pre_pipeline_sample_weight, timestamps):
    """Block of train_and_evaluate_model starting at ``if oof_n_splits and oof_n_splits >= 2 and not just_evaluate:``."""
    from mlframe.training.trainer import _compute_oof_preds

    if oof_n_splits and oof_n_splits >= 2 and not just_evaluate:
        _is_clf_for_oof = "Classifier" in model_type_name or model_type_name in ("ClassifierChain", "_ChainEnsemble")
        # Multi-output / multi-label paths skip OOF here; their stackers will raise if level-2 is requested.
        _y_arr = np.asarray(train_target) if train_target is not None else None
        _is_multi_output_target = _y_arr is not None and _y_arr.ndim == 2
        if not _is_multi_output_target:
            _oof_preds, _oof_probs = _compute_oof_preds(
                model=model,
                train_df=train_df,
                train_target=train_target,
                is_classifier_model=_is_clf_for_oof,
                n_splits=int(oof_n_splits),
                random_seed=int(oof_random_seed),
                group_ids=group_ids[train_idx] if (group_ids is not None and train_idx is not None) else None,
                has_time=bool(oof_has_time), sample_weight=_pre_pipeline_sample_weight, timestamps=_oof_train_timestamps(timestamps, train_idx),
            )
            try:
                if _oof_preds is not None:
                    model.oof_preds = _oof_preds  # mlframe-injected bookkeeping attr on an arbitrary (object-typed) model
                if _oof_probs is not None:
                    model.oof_probs = _oof_probs
                if _oof_preds is not None or _oof_probs is not None:
                    # Stamp the train-aligned target so post-hoc OOF
                    # calibration pairs each OOF prob with its OWN row's
                    # label, AND so regression-side consumers (e.g.
                    # ``recommend_diversity_additions_in_leaderboard``,
                    # which requires oof_preds/oof_probs + oof_target on
                    # every member regardless of task type) can find it.
                    # cross_val_predict returns predictions in train-row
                    # order, so train_target (train_idx order) is the
                    # row-for-row match. Without this, post_calibrate_model
                    # fell back to a POSITIONAL target_series slice that is
                    # correct only when train is the leading contiguous
                    # block (wrong under shuffled / group-aware splits).
                    model.oof_target = _y_arr
            except AttributeError:
                # Some frozen-attribute estimators (sklearn 1.4+ slots) refuse new attrs; stamp on the wrapper carrier instead.
                pass


def _train_and_evaluate_train_runs_sequentially_may(splits_config, metrics_out, has_val, has_test, common_metrics_params, columns, train_preds, train_probs):
    """Block of train_and_evaluate_model starting at ``for split_name, split_df, split_target, split_idx, split_preds, split_``."""
    for split_name, split_df, split_target, split_idx, split_preds, split_probs, split_details, should_compute in splits_config:
        if should_compute and split_name == "train":
            preds_result, probs_result, columns = _compute_split_metrics(
                split_name=split_name,
                df=split_df,
                target=split_target,
                idx=split_idx,
                metrics_dict=metrics_out[split_name],
                preds=split_preds,
                probs=split_probs,
                details=split_details,
                has_other_splits=has_val or has_test,
                **common_metrics_params,
            )
            train_preds, train_probs = preds_result, probs_result
    return columns, train_preds, train_probs


def _train_and_evaluate_run_test_df_none(_run_test, df, test_df, train_df):
    """Block of train_and_evaluate_model starting at ``if _run_test and ((df is not None) or (test_df is not None)):``."""
    if _run_test and ((df is not None) or (test_df is not None)):
        try:
            if train_df is not None:
                del train_df
        except NameError:
            pass
        _maybe_clean_ram()


def _train_and_evaluate_was_trained_instead_silently(sample_weight, test_idx, _test_sample_weight):
    """Block of train_and_evaluate_model starting at ``if sample_weight is not None and test_idx is not None:``."""
    if sample_weight is not None and test_idx is not None:
        try:
            if isinstance(sample_weight, (pd.Series, pd.DataFrame)):
                _test_sample_weight = np.asarray(sample_weight.iloc[test_idx].values, dtype=np.float64)
            else:
                _test_sample_weight = np.asarray(sample_weight, dtype=np.float64)[test_idx]
        except (TypeError, ValueError, IndexError):
            _test_sample_weight = None
    return _test_sample_weight
