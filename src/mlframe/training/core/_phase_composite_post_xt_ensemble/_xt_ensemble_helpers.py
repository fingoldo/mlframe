"""Helpers carved out of ``__init__`` to keep that module under its size budget."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ...composite import CompositeCrossTargetEnsemble as _CrossEns
from ...composite.estimator import CompositeTargetEstimator
from ...composite.post_shim import PrePipelinePredictShim
from ..utils import _build_full_column_from_splits
from ._prescreen import (
    PRESCREEN_SAFETY, dummy_floor_from_metadata, leaky_rmse_keep_mask,
)
from .._prediction_memo import memo_predict
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.training.core._phase_composite_post")

_DEFAULT_OOF_RANDOM_STATE = 42


logger = logging.getLogger("mlframe.training.core._phase_composite_post")


_DEFAULT_OOF_RANDOM_STATE = 42


def note_stack_without_oof(metadata: Any, target_type: Any, target_name: Any, strategy: str) -> None:
    """Warn and stamp ``metadata["xt_ensemble_stack_source"]`` when a stacking strategy has no honest OOF matrix and is replaced by a uniform mean.

    Fitting NNLS / linear-stack weights on in-sample component predictions over-weights the most overfit member, so the stack is refused.
    """
    logger.warning(
        "[CompositeCrossTargetEnsemble] target='%s': %s needs honest OOF predictions and none were produced; using a uniform mean "
        "instead of fitting stack weights on in-sample predictions.", target_name, strategy,
    )
    if isinstance(metadata, dict):
        metadata.setdefault("xt_ensemble_stack_source", {})[f"{target_type}/{target_name}"] = f"uniform_mean_no_oof (requested {strategy})"


def _build_cross_target_entry_enumerate_orig_entries(_orig_entries, _components, _component_names):
    """Block of _build_cross_target_ensemble_for_target starting at ``for _i, _entry in enumerate(_orig_entries):``."""
    for _i, _entry in enumerate(_orig_entries):
        _inner = getattr(_entry, "model", None) or _entry
        if not hasattr(_inner, "predict"):
            continue
        _pp = getattr(_entry, "pre_pipeline", None)
        _name = f"raw#{_i}"
        _components.append(PrePipelinePredictShim(_inner, _pp, _name))
        _component_names.append(_name)


def _build_cross_target_unmatched_oof_refit_path(_components, _component_names, _known_spec_keys, _spec_list):
    """Block of _build_cross_target_ensemble_for_target starting at ``for _comp, _name in zip(_components, _component_names):``."""
    for _comp, _name in zip(_components, _component_names):
        _inner_for_adhoc = getattr(_comp, "model", _comp)
        if not isinstance(_inner_for_adhoc, CompositeTargetEstimator):
            continue
        _tn_adhoc = getattr(_inner_for_adhoc, "transform_name", None)
        _bc_adhoc = getattr(_inner_for_adhoc, "base_column", None)
        if _tn_adhoc is None or _bc_adhoc is None or (_tn_adhoc, _bc_adhoc) in _known_spec_keys:
            continue
        _base_cols_adhoc = getattr(_inner_for_adhoc, "base_columns", None) or ()
        _spec_list = [
            *_spec_list,
            {
                "name": f"__adhoc__{_tn_adhoc}__{_bc_adhoc}",
                "transform_name": _tn_adhoc,
                "base_column": _bc_adhoc,
                "extra_base_columns": tuple(_base_cols_adhoc[1:]) if len(_base_cols_adhoc) > 1 else (),
                "fitted_params": getattr(_inner_for_adhoc, "fitted_params_", None),
            },
        ]
        _known_spec_keys.add((_tn_adhoc, _bc_adhoc))
    return _spec_list


def _build_cross_target_train_fold_oof_stack(filtered_train_df, _fit_y_full, filtered_train_idx, models, _tt_e, _orig_tname, composite_target_discovery_config, ctx, _oof_weights):
    """Block of _build_cross_target_ensemble_for_target starting at ``if filtered_train_df is not None and _fit_y_full is not None and filte``."""
    if filtered_train_df is not None and _fit_y_full is not None and filtered_train_idx is not None:
        try:
            _y_arr_mtr = np.asarray(_fit_y_full)[filtered_train_idx]
            # Build the same component shims the equal-mean path uses, then OOF-fit NNLS over them.
            _mtr_entries = (models or {}).get(_tt_e, {}).get(_orig_tname, []) or []
            _mtr_components: list[Any] = []
            for _mi, _mentry in enumerate(_mtr_entries):
                _minner = getattr(_mentry, "model", None) or _mentry
                if not hasattr(_minner, "predict"):
                    continue
                _mpp = getattr(_mentry, "pre_pipeline", None)
                _mtr_components.append(PrePipelinePredictShim(_minner, _mpp, f"raw#{_mi}"))
            if len(_mtr_components) >= 2:
                from mlframe.training.core._phase_composite_post_xt_ensemble._phase_composite_post_xt_mtr_oof import compute_mtr_oof_nnls_weights
                _oof_random_state = int(getattr(
                    composite_target_discovery_config,
                    "oof_random_state", _DEFAULT_OOF_RANDOM_STATE,
                ))
                _oof_kfold_mtr = int(getattr(
                    composite_target_discovery_config, "oof_kfold", 5,
                ))
                # Same ctx.sample_weights[target] -> filtered_train_idx slicing the general (non-MTR)
                # path below uses for `_sw_for_oof` -- computed independently here since this MTR
                # branch runs BEFORE that later block. Without it, the honest-OOF NNLS weights were
                # silently fit as if every row were equally important on a weighted suite.
                _ctx_sw_dict_mtr = getattr(ctx, "sample_weights", None) if ctx is not None else None
                _sw_for_mtr_oof = None
                if isinstance(_ctx_sw_dict_mtr, dict) and _ctx_sw_dict_mtr:
                    _sw_raw_mtr = _ctx_sw_dict_mtr.get(_orig_tname)
                    if _sw_raw_mtr is not None:
                        try:
                            _sw_for_mtr_oof = np.asarray(_sw_raw_mtr)[filtered_train_idx]
                        except (TypeError, IndexError):
                            _sw_for_mtr_oof = None
                _oof_weights = compute_mtr_oof_nnls_weights(
                    _mtr_components, filtered_train_df, _y_arr_mtr,
                    kfold=_oof_kfold_mtr, random_state=_oof_random_state,
                    sample_weight=_sw_for_mtr_oof,
                )
        except Exception as _mtr_oof_err:
            logger.warning(
                "[MTR CT_ENSEMBLE] target='%s': honest-OOF NNLS weighting failed (%s); forfeiting the " "benched ~9%% NNLS win and falling back to equal-mean.",
                _orig_tname,
                _mtr_oof_err,
            )
            _oof_weights = None
    return _oof_weights


def _build_cross_target_inject_lag_predict_dummy(_spec_list, models, _tt_e, _components, _component_names):
    """Block of _build_cross_target_ensemble_for_target starting at ``for _spec in _spec_list:``."""
    for _spec in _spec_list:
        _composite_entries = (models or {}).get(_tt_e, {}).get(_spec["name"], []) or []
        for _i, _entry in enumerate(_composite_entries):
            _inner = getattr(_entry, "model", None) or _entry
            if not hasattr(_inner, "predict"):
                continue
            # CTE wrappers handle the transform; pre_pipeline (if any) is outer frame-prep applied via the same shim.
            _pp = getattr(_entry, "pre_pipeline", None)
            _name = f"{_spec['name']}#{_i}"
            _components.append(PrePipelinePredictShim(_inner, _pp, _name))
            _component_names.append(_name)


def _build_cross_target_per_spec_base_matrix(_spec_list, train_df_pd, val_df_pd, test_df_pd, train_idx, val_idx, test_idx, _oof_y_full, filtered_train_idx, filtered_val_idx, _base_full_per_spec, _base_val_per_spec):
    """Block of _build_cross_target_ensemble_for_target starting at ``for _spec_for_oof in _spec_list:``."""
    for _spec_for_oof in _spec_list:
        _b_primary = _build_full_column_from_splits(
            _spec_for_oof["base_column"],
            train_df_pd, val_df_pd, test_df_pd,
            train_idx, val_idx, test_idx,
            n_total=len(_oof_y_full),
        )
        _extra_for_oof = tuple(_spec_for_oof.get("extra_base_columns") or ())
        if _extra_for_oof:
            _b_cols = [_b_primary]
            _b_cols.extend(
                _build_full_column_from_splits(
                    _eb_oof,
                    train_df_pd, val_df_pd, test_df_pd,
                    train_idx, val_idx, test_idx,
                    n_total=len(_oof_y_full),
                )
                for _eb_oof in _extra_for_oof
            )
            _b_stack_full = np.column_stack(_b_cols)
            _b_filtered = _b_stack_full[filtered_train_idx]
            try:
                _b_val = _b_stack_full[filtered_val_idx]
            except Exception as e:
                logger.debug("indexing _b_stack_full by filtered_val_idx failed: %s", e)
                _b_val = None
        else:
            _b_filtered = _b_primary[filtered_train_idx]
            try:
                _b_val = _b_primary[filtered_val_idx]
            except Exception as e:
                logger.debug("indexing _b_primary by filtered_val_idx failed: %s", e)
                _b_val = None
        # Key by the UNIQUE spec name, not base_column. A multi-base
        # spec and a single-base spec sharing the same PRIMARY base column
        # otherwise collide (last writer wins), so the other spec's OOF
        # refit raises a base-width mismatch and is silently excluded.
        _base_full_per_spec[_spec_for_oof["name"]] = _b_filtered
        if _b_val is not None:
            _base_val_per_spec[_spec_for_oof["name"]] = _b_val


def _build_cross_target_component_no_spec_entry(_component_names, _components, _component_specs, _spec_list):
    """Block of _build_cross_target_ensemble_for_target starting at ``for _name, _comp in zip(_component_names, _components):``."""
    for _name, _comp in zip(_component_names, _components):
        _inner_for_spec = getattr(_comp, "model", _comp)
        if not isinstance(_inner_for_spec, CompositeTargetEstimator):
            _component_specs.append(None)
            continue
        _comp_name = _name.split("#", 1)[0]
        _matching = next(
            (s for s in _spec_list if s["name"] == _comp_name),
            None,
        )
        if _matching is None:
            # The name didn't resolve to a spec (e.g. this is the "raw#N" slot) -- fall back to
            # matching by the wrapper's own (transform_name, base_column), which is stable regardless
            # of which models[..] bucket the entry ended up in.
            _tn = getattr(_inner_for_spec, "transform_name", None)
            _bc = getattr(_inner_for_spec, "base_column", None)
            _matching = next(
                (s for s in _spec_list if s.get("transform_name") == _tn and s.get("base_column") == _bc),
                None,
            )
        _component_specs.append(_matching)


def _build_cross_target_thread_ctx_timestamps_per(_ctx_ts_full, filtered_train_idx, _time_ordering):
    """Block of _build_cross_target_ensemble_for_target starting at ``if _ctx_ts_full is not None:``."""
    if _ctx_ts_full is not None:
        try:
            _time_ordering = np.asarray(_ctx_ts_full)[filtered_train_idx]
        except (TypeError, IndexError):
            _time_ordering = None
    return _time_ordering


def _build_cross_target_isinstance_ctx_sw_dict(_ctx_sw_dict, _orig_tname, filtered_train_idx, _sw_for_oof):
    """Block of _build_cross_target_ensemble_for_target starting at ``if isinstance(_ctx_sw_dict, dict) and _ctx_sw_dict:``."""
    if isinstance(_ctx_sw_dict, dict) and _ctx_sw_dict:
        _sw_raw = _ctx_sw_dict.get(_orig_tname)
        if _sw_raw is not None:
            try:
                _sw_for_oof = np.asarray(_sw_raw)[filtered_train_idx]
            except (TypeError, IndexError):
                _sw_for_oof = None
    return _sw_for_oof


def _build_cross_target_ctx_groups_none(_ctx_groups, filtered_train_idx, _group_ids_for_oof):
    """Block of _build_cross_target_ensemble_for_target starting at ``if _ctx_groups is not None:``."""
    if _ctx_groups is not None:
        try:
            _group_ids_for_oof = np.asarray(_ctx_groups)[filtered_train_idx]
        except (TypeError, IndexError):
            _group_ids_for_oof = None
    return _group_ids_for_oof


def _build_cross_target_speed_up_only_correctness(_prescreen_X, _components, metadata, _tt_e, _orig_tname, _component_names, _prescreen_y, _component_specs):
    """Block of _build_cross_target_ensemble_for_target starting at ``if _prescreen_X is not None and len(_components) >= 4:``."""
    if _prescreen_X is not None and len(_components) >= 4:
        try:
            _dummy_floor_for_prescreen = dummy_floor_from_metadata(metadata, _tt_e, _orig_tname)
            if _dummy_floor_for_prescreen is not None:
                _keep_mask, _dropped_pre = leaky_rmse_keep_mask(
                    _components, _component_names, _prescreen_X, _prescreen_y, _dummy_floor_for_prescreen,
                )
                if _dropped_pre and sum(_keep_mask) >= 2:
                    _kept = [i for i, k in enumerate(_keep_mask) if k]
                    logger.warning(
                        "[CompositeCrossTargetEnsemble] target='%s' "
                        "OOF pre-screen (leaky val-RMSE speed gate) dropped %d/%d component(s) "
                        "whose leaky val_RMSE / %.1f > dummy floor "
                        "%.4g. Dropped: %s. Saves ~%d minute(s) of "
                        "refit time.",
                        _orig_tname, len(_dropped_pre),
                        len(_components), PRESCREEN_SAFETY,
                        _dummy_floor_for_prescreen, _dropped_pre,
                        len(_dropped_pre) * 5,  # ~5 min/component
                    )
                    _components = [_components[i] for i in _kept]
                    _component_names = [_component_names[i] for i in _kept]
                    _component_specs = [_component_specs[i] for i in _kept]
        except Exception as _prescreen_err:
            logger.warning(
                "[CompositeCrossTargetEnsemble] OOF pre-screen " "failed (non-fatal): %s. Continuing with full OOF " "refit.",
                _prescreen_err,
            )
    return _component_names, _component_specs, _components


def _build_cross_target_do_calib(_do_calib, _ensemble, _oof_pred_matrix, _oof_y_holdout, composite_target_discovery_config, _oof_sw, _orig_tname):
    """Block of _build_cross_target_ensemble_for_target starting at ``if (_do_calib``."""
    if (_do_calib
            and isinstance(_ensemble, _CrossEns)
            and _oof_pred_matrix is not None
            and _oof_pred_matrix.shape[1] == len(_ensemble.component_models)
            and _oof_y_holdout is not None
            and _oof_pred_matrix.shape[0] >= 3):
        try:
            _calib_method = str(getattr(
                composite_target_discovery_config,
                "cross_target_calibration_method", "isotonic",
            ))
            _ensemble.fit_output_calibrator(
                _oof_pred_matrix, np.asarray(_oof_y_holdout, dtype=np.float64),
                method=_calib_method, sample_weight=_oof_sw,
            )
            logger.info(
                "[CompositeCrossTargetEnsemble] target='%s' fitted output calibrator " "(method=%s, attached=%s) on %d OOF rows.",
                _orig_tname,
                _calib_method,
                getattr(_ensemble, "_output_calibrator", None) is not None,
                int(_oof_pred_matrix.shape[0]),
            )
        except Exception as _calib_err:  # best-effort: calibration is an optional enhancement, never worth failing the ensemble build over
            logger.warning(
                "[CompositeCrossTargetEnsemble] target='%s' output calibration failed "
                "(%s); ensemble retained uncalibrated.", _orig_tname, _calib_err,
            )


def _build_cross_target_rather_than_each_fallback(filtered_train_idx, _ens_y_arr, _ens_train_envelope):
    """Block of _build_cross_target_ensemble_for_target starting at ``try:``."""
    try:
        _ens_y_train = _ens_y_arr[filtered_train_idx] if filtered_train_idx is not None else None
        if _ens_y_train is not None and len(_ens_y_train) > 0:
            from mlframe.training._prediction_envelope_clip import compute_train_envelope_stats
            _ens_train_envelope = compute_train_envelope_stats(_ens_y_train)
    except Exception as e:
        logger.debug("compute_train_envelope_stats failed, leaving the train envelope unset: %s", e)
        _ens_train_envelope = None
    return _ens_train_envelope


def _build_cross_target_split_name_report_title(_split_plan, _ens_y_arr, _ensemble, _ce_strategy, metadata, _tt_e, _orig_tname, _ens_common, plot_file, report_model_perf):
    """Block of _build_cross_target_ensemble_for_target starting at ``for _split_name, _report_title, _split_idx, _split_df in _split_plan:``."""
    for _split_name, _report_title, _split_idx, _split_df in _split_plan:
        if _split_idx is None or _split_df is None:
            continue
        try:
            _y_split = _ens_y_arr[_split_idx]
            _ens_preds = memo_predict(_ensemble, _split_df)
            # Stamp val/test scalar metrics for this ensemble into metadata so the
            # suite-end verdict block can compare CT_ENSEMBLE against the dummy floor.
            # Without this the verdict only sees the SINGLE best model and falsely
            # flags BEST_MODEL_BELOW_DUMMY when the ensemble (stacked on lag_predict)
            # is the actual winner on strong-AR targets.
            try:
                from mlframe.metrics.core import fast_mean_absolute_error, fast_root_mean_squared_error
                _y_arr = np.asarray(_y_split, dtype=np.float64).reshape(-1)
                _ens_arr = _ens_preds.reshape(-1)
                _ens_scalar_metrics = {
                    f"{_split_name}_RMSE": float(fast_root_mean_squared_error(_y_arr, _ens_arr)),
                    f"{_split_name}_MAE": float(fast_mean_absolute_error(_y_arr, _ens_arr)),
                    "model_name": f"CT_ENSEMBLE[{_ce_strategy}]",
                }
                metadata.setdefault("cross_target_ensemble_metrics", {}).setdefault(str(_tt_e), {}).setdefault(_orig_tname, {}).update(_ens_scalar_metrics)
            except Exception as _metric_err:
                logger.debug(
                    "Could not stamp CT_ENSEMBLE %s metrics for target='%s': %s",
                    _split_name, _orig_tname, _metric_err,
                )
            _common_split = dict(_ens_common)
            if plot_file:
                _common_split["plot_file"] = f"{plot_file}_ct_ensemble_{_orig_tname}_{_split_name}"
            report_model_perf(
                targets=_y_split,
                preds=_ens_preds, probs=None,
                report_title=_report_title,
                **_common_split,
            )
        except Exception as _split_err:
            log_throttle(
                logger,
                "xt_ensemble_report_model_perf_failed",
                logging.WARNING,
                "[CompositeCrossTargetEnsemble] target='%s' " "split='%s' report_model_perf failed: %s. " "Continuing without ensemble chart for this split.",
                _orig_tname,
                _split_name,
                _split_err,
            )
