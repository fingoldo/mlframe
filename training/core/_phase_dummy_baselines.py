"""
Per-target dummy baselines diagnostic.

Computes trivial-baseline floor (mean / median / prior / most_frequent / per_group /
TS-naive / LTR random_within_query / multilabel per-label-prior) for a single target.
Reports strongest baseline through the same ``report_model_perf`` pipeline as real models.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._format import format_metric as _dummy_fmt
from ..composite_transforms import is_composite_target_name
from ..evaluation import report_model_perf
from ..phases import phase
from .utils import _augment_with_dropped_high_card_cols

logger = logging.getLogger(__name__)


def run_dummy_baselines(
    *,
    target_type,
    cur_target_name: str,
    target_name: str,
    model_name: str,
    current_train_target,
    current_val_target,
    current_test_target,
    filtered_train_df,
    filtered_val_df,
    test_df_pd,
    filtered_train_idx,
    filtered_val_idx,
    test_idx,
    timestamps,
    cat_features: Optional[List[str]],
    dummy_baselines_config,
    quantile_regression_config,
    reporting_config,
    _dropped_high_card_data: Optional[Dict],
    train_od_idx,
    val_od_idx,
    plot_file: str,
    metadata: Dict,
    target_by_type: Dict,
    _split_preds_probs,
) -> Dict:
    """Run dummy baselines for one target. Returns updated metadata."""
    try:
        if not (dummy_baselines_config.enabled and (
            str(target_type) in dummy_baselines_config.apply_to_target_types
        )):
            return metadata

        from ..dummy_baselines import compute_dummy_baselines

        _ts_train = (
            timestamps[filtered_train_idx]
            if timestamps is not None and filtered_train_idx is not None
            else None
        )
        _ts_val = (
            timestamps[filtered_val_idx]
            if timestamps is not None and filtered_val_idx is not None
            else None
        )
        _ts_test = (
            timestamps[test_idx]
            if timestamps is not None and test_idx is not None
            else None
        )

        # Re-attach high-card cat cols dropped earlier so per_group_mean can use
        # them as group keys (e.g. well_id with 623 unique values).
        _dummy_train_X = filtered_train_df
        _dummy_val_X = filtered_val_df
        _dummy_test_X = test_df_pd
        _dummy_cat_features = list(cat_features or [])
        if _dropped_high_card_data:
            try:
                _dummy_train_X, _dummy_val_X, _dummy_test_X, _added = _augment_with_dropped_high_card_cols(
                    _dropped_high_card_data,
                    filtered_train_df, filtered_val_df, test_df_pd,
                    train_od_idx=train_od_idx, val_od_idx=val_od_idx,
                )
                if _added:
                    _dummy_cat_features.extend(_added)
                    logger.debug(
                        "[dummy-baselines] re-attached %d auto-dropped high-card cat col(s) for per_group_mean: %s",
                        len(_added), _added,
                    )
            except Exception as _aug_err:
                logger.debug(
                    "[dummy-baselines] failed to re-attach dropped high-card cat cols (%s); per_group_mean may be missing",
                    _aug_err,
                )

        # Auto-pick quantile_alphas from QuantileRegressionConfig
        _q_alphas = None
        if str(target_type) == "quantile_regression":
            _q_alphas = list(getattr(
                quantile_regression_config, "alphas", ()
            ) or ())
            if not _q_alphas:
                _q_alphas = None

        with phase(f"dummy_baselines:{str(target_type)}", target=cur_target_name):
            _db_report = compute_dummy_baselines(
                target_type=str(target_type),
                target_name=cur_target_name,
                train_X=_dummy_train_X,
                val_X=_dummy_val_X,
                test_X=_dummy_test_X,
                train_y=current_train_target,
                val_y=current_val_target,
                test_y=current_test_target,
                timestamps_train=_ts_train,
                timestamps_val=_ts_val,
                timestamps_test=_ts_test,
                cat_features=_dummy_cat_features,
                quantile_alphas=_q_alphas,
                config=dummy_baselines_config,
                plot_file_prefix=(plot_file or ""),
            )

        logger.info(_db_report.format_text())
        logger.debug(
            "[dummy-baselines] target='%s' full table:\n%s",
            cur_target_name, _db_report.table.to_string(),
        )

        # Report strongest dummy through the SAME report_model_perf pipeline
        # that all real models use. Gated on plot_strongest (default ON).
        if (getattr(dummy_baselines_config, "plot_strongest", True)
                and _db_report.strongest is not None):
            try:
                _strongest_val_raw = _db_report.extras.get("strongest_val_preds")
                _strongest_test_raw = _db_report.extras.get("strongest_test_preds")

                # Mirror real-model title format so plots aren't anonymous.
                _dummy_is_composite = is_composite_target_name(cur_target_name)
                _dummy_mt_tag = "MTRESID" if _dummy_is_composite else "MTTR"
                try:
                    if current_train_target is not None and len(current_train_target) > 0:
                        _dummy_mt_val = float(np.asarray(current_train_target).mean())
                        _dummy_mt_suffix = f" {_dummy_mt_tag}={_dummy_fmt(_dummy_mt_val)}"
                    else:
                        _dummy_mt_suffix = ""
                except Exception:
                    _dummy_mt_suffix = ""
                _dummy_name = (
                    f"DummyBaseline:{_db_report.strongest} "
                    f"{target_name} {model_name} {cur_target_name}"
                    f"{_dummy_mt_suffix}"
                )

                _common = dict(
                    columns=list(getattr(filtered_train_df, "columns", []) or []),
                    df=None, model=None,
                    model_name=_dummy_name,
                    plot_outputs=getattr(reporting_config, "plot_outputs", None),
                    plot_dpi=getattr(reporting_config, "plot_dpi", None),
                    show_fi=False,
                    target_type=str(target_type),
                )
                if _strongest_val_raw is not None and current_val_target is not None:
                    _vp, _vpr = _split_preds_probs(_strongest_val_raw)
                    _common_val = dict(_common)
                    if plot_file:
                        _common_val["plot_file"] = f"{plot_file}_dummy_{_db_report.strongest}_val"
                    report_model_perf(
                        targets=current_val_target,
                        preds=_vp, probs=_vpr,
                        report_title="VAL (DUMMY) ",
                        **_common_val,
                    )
                if _strongest_test_raw is not None and current_test_target is not None:
                    _tp, _tpr = _split_preds_probs(_strongest_test_raw)
                    _common_test = dict(_common)
                    if plot_file:
                        _common_test["plot_file"] = f"{plot_file}_dummy_{_db_report.strongest}_test"
                    report_model_perf(
                        targets=current_test_target,
                        preds=_tp, probs=_tpr,
                        report_title="TEST (DUMMY) ",
                        **_common_test,
                    )
            except Exception as _plot_err:
                logger.warning(
                    "[dummy-baselines] target='%s' report_model_perf for dummy "
                    "failed: %s. Training continues without pre-training floor report.",
                    cur_target_name, _plot_err,
                )

        metadata.setdefault("dummy_baselines", {}) \
            .setdefault(str(target_type), {})[cur_target_name] = _db_report.to_dict()

        # y-scale dummy for composite targets: invert strongest dummy predictions
        # to y-scale so suite-end verdict block compares both numbers on the same scale.
        _specs_for_tt = metadata.get("composite_target_specs", {}).get(str(target_type), {})
        _matching_spec = None
        for _tname_specs in _specs_for_tt.values():
            for _s in _tname_specs or []:
                if _s.get("name") == cur_target_name:
                    _matching_spec = _s
                    break
            if _matching_spec is not None:
                break
        if (_matching_spec is not None
                and _db_report.strongest is not None
                and _db_report.extras.get("strongest_val_preds") is not None):
            try:
                from ..composite import get_transform
                _tf = get_transform(_matching_spec["transform_name"])
                _fp = _matching_spec["fitted_params"]
                _base_col = _matching_spec["base_column"]
                _raw_target_col = _matching_spec["target_col"]
                _raw_y_full = target_by_type.get(target_type, {}).get(_raw_target_col)
                _y_scale_dummy_metrics: Dict[str, Dict[str, float]] = {}
                for _split_name, _split_df, _split_idx, _T_preds_key in (
                    ("val", filtered_val_df, filtered_val_idx, "strongest_val_preds"),
                    ("test", test_df_pd, test_idx, "strongest_test_preds"),
                ):
                    _T_preds = _db_report.extras.get(_T_preds_key)
                    if (_T_preds is None or _split_df is None
                            or _split_idx is None
                            or _raw_y_full is None
                            or _base_col not in _split_df.columns):
                        continue
                    _base_split = np.asarray(_split_df[_base_col], dtype=np.float64)
                    _y_dummy_split = _tf.inverse(
                        np.asarray(_T_preds, dtype=np.float64),
                        _base_split, _fp,
                    )
                    _y_true_split = np.asarray(_raw_y_full, dtype=np.float64)[_split_idx]
                    _diff = _y_dummy_split.astype(np.float64) - _y_true_split
                    _finite = np.isfinite(_diff)
                    if _finite.sum() == 0:
                        continue
                    _y_scale_dummy_metrics[_split_name] = {
                        "RMSE": float(np.sqrt(np.mean(_diff[_finite] * _diff[_finite]))),
                        "MAE": float(np.mean(np.abs(_diff[_finite]))),
                        "n_rows_finite": int(_finite.sum()),
                    }
                if _y_scale_dummy_metrics:
                    metadata["dummy_baselines"][str(target_type)][
                        cur_target_name
                    ]["y_scale_strongest_metrics"] = _y_scale_dummy_metrics
                    _ys_log_parts = [
                        f"{k.upper()}=RMSE_y:{v['RMSE']:.4g} MAE_y:{v['MAE']:.4g}"
                        for k, v in _y_scale_dummy_metrics.items()
                    ]
                    logger.info(
                        "[DUMMY_BASELINES] composite='%s' strongest='%s' y-scale metrics "
                        "(inverted from T via %s): %s",
                        cur_target_name, _db_report.strongest,
                        _matching_spec["transform_name"],
                        " | ".join(_ys_log_parts),
                    )
            except Exception as _yscale_err:
                logger.warning(
                    "[DUMMY_BASELINES] failed to compute y-scale dummy for composite '%s': %s. "
                    "T-scale metrics remain in metadata.",
                    cur_target_name, _yscale_err,
                )

    except Exception as _db_err:
        logger.warning(
            "[DUMMY_BASELINES] FAILED target='%s' (%s): %s. "
            "Training continues without baseline floor.",
            cur_target_name, target_type, _db_err,
        )
        metadata.setdefault("dummy_baselines_failures", {}) \
            .setdefault(str(target_type), {})[cur_target_name] = str(_db_err)

    return metadata
