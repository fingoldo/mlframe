"""Per-target dummy baselines diagnostic. Computes trivial-baseline floor and reports the strongest baseline through ``report_model_perf``."""
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
    cat_features: list[str] | None,
    dummy_baselines_config,
    quantile_regression_config,
    reporting_config,
    _dropped_high_card_data: dict | None,
    train_od_idx,
    val_od_idx,
    plot_file: str,
    metadata: dict,
    target_by_type: dict,
    _split_preds_probs,
    group_ids=None,  # ctx.group_ids; required for LTR-Popularity / per-group baselines.
                    # Pre-fix the signature didn't accept it -> LTR suites silently
                    # degraded to regression-style dummy + the LTR baseline table came
                    # back blank with extras["ltr_skip_reason"] = "group_ids missing".
) -> dict:
    try:
        if not (dummy_baselines_config.enabled and (
            str(target_type) in dummy_baselines_config.apply_to_target_types
        )):
            return metadata

        # Skip dummy-baselines for composite targets: their report is in the
        # COMPOSITE/residual scale (lag_predict / per_group_mean on a diff /
        # linres target is residual-scale by construction), and the user
        # requires every emitted/stored metric to be in the ORIGINAL target
        # scale. The raw target already has its dummy baseline computed and
        # stored, so the leaderboard floor for the y-scale metric is already
        # available. Storing a composite-scale dummy in
        # ``metadata["dummy_baselines"][type][composite_name]`` would also
        # mis-rank the suite-end verdict block (model side uses y-scale via
        # composite_target_y_scale_metrics; mixing scales is incorrect).
        if is_composite_target_name(cur_target_name):
            logger.info(
                "[dummy-baselines] target='%s' is a composite target -- "
                "skipping dummy baselines (its lag/mean baselines would be "
                "in the composite/residual scale, not the original-target "
                "scale; the raw target's dummies already cover the y-scale "
                "leaderboard floor).",
                cur_target_name,
            )
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

        # Re-attach high-card cat cols dropped earlier so per_group_mean can use them as group keys.
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

        _q_alphas = None
        if str(target_type) == "quantile_regression":
            _q_alphas = list(getattr(
                quantile_regression_config, "alphas", ()
            ) or ())
            if not _q_alphas:
                _q_alphas = None

        # Slice group_ids to the split-indices so per-split arrays line up with the y / X
        # arrays passed below. Pre-fix group_ids was never received by this function so the
        # LTR-Popularity / per-group baselines hit extras["ltr_skip_reason"] = "group_ids
        # missing" and the LTR baseline table came back blank.
        _gid_train = _gid_val = _gid_test = None
        if group_ids is not None:
            import numpy as _np_gid
            try:
                _gid_arr = _np_gid.asarray(group_ids)
                if filtered_train_idx is not None:
                    _gid_train = _gid_arr[filtered_train_idx]
                if filtered_val_idx is not None:
                    _gid_val = _gid_arr[filtered_val_idx]
                if test_idx is not None:
                    _gid_test = _gid_arr[test_idx]
            except (TypeError, IndexError) as _gid_err:
                logger.warning(
                    "run_dummy_baselines: failed to slice group_ids to split indices "
                    "(%s); LTR-popularity baseline will be skipped.", _gid_err,
                )
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
                group_ids_train=_gid_train,
                group_ids_val=_gid_val,
                group_ids_test=_gid_test,
            )

        logger.info(_db_report.format_text())
        logger.debug(
            "[dummy-baselines] target='%s' full table:\n%s",
            cur_target_name, _db_report.table.to_string(),
        )

        # Report strongest dummy through the same report_model_perf pipeline as real models. Gated on plot_strongest.
        if (getattr(dummy_baselines_config, "plot_strongest", True)
                and _db_report.strongest is not None):
            try:
                _strongest_val_raw = _db_report.extras.get("strongest_val_preds")
                _strongest_test_raw = _db_report.extras.get("strongest_test_preds")

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

                # ``X or []`` triggers pd.Index.__bool__ which raises ValueError; use an
                # explicit None/empty check instead. Was: ``getattr(..., "columns", []) or []``.
                _columns_attr = getattr(filtered_train_df, "columns", None)
                # Train envelope stats so the dummy report's val/test
                # charts share the same prediction-clip bound as the
                # real-model reports (constant-predictor dummies never
                # exceed the train bound by construction; the wiring
                # is mostly for consistency + non-dummy callsites that
                # forward through the same helper).
                _dummy_envelope_stats = None
                if current_train_target is not None:
                    try:
                        from .._prediction_envelope_clip import compute_train_envelope_stats
                        _dummy_envelope_stats = compute_train_envelope_stats(current_train_target)
                    except Exception:
                        _dummy_envelope_stats = None
                _common = dict(
                    columns=list(_columns_attr) if _columns_attr is not None else [],
                    df=None, model=None,
                    model_name=_dummy_name,
                    plot_outputs=getattr(reporting_config, "plot_outputs", None),
                    plot_dpi=getattr(reporting_config, "plot_dpi", None),
                    show_fi=False,
                    target_type=str(target_type),
                    y_train_envelope_stats=_dummy_envelope_stats,
                    # Threaded so report_regression_model_perf can read
                    # regression_title_metrics_tokens; before this the dummy
                    # report path raised NameError on every regression run.
                    reporting_config=reporting_config,
                )
                _emit_val = bool(getattr(reporting_config, "compute_valset_metrics", True))
                _emit_test = bool(getattr(reporting_config, "compute_testset_metrics", True))
                if (_emit_val and _strongest_val_raw is not None
                        and current_val_target is not None):
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
                if (_emit_test and _strongest_test_raw is not None
                        and current_test_target is not None):
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
                # Include input types in the warning so the "truth value of a Index is
                # ambiguous"-style pandas booleanness errors can be triaged without
                # repro. The dummy report path is non-critical; training continues
                # regardless.
                _val_t = type(current_val_target).__name__ if current_val_target is not None else "None"
                _test_t = type(current_test_target).__name__ if current_test_target is not None else "None"
                _strong_val_t = type(_strongest_val_raw).__name__ if _strongest_val_raw is not None else "None"
                _strong_test_t = type(_strongest_test_raw).__name__ if _strongest_test_raw is not None else "None"
                logger.warning(
                    "[dummy-baselines] target='%s' report_model_perf for dummy "
                    "failed: %s [val_target=%s test_target=%s strong_val_preds=%s strong_test_preds=%s]. "
                    "Training continues without pre-training floor report.",
                    cur_target_name, _plot_err,
                    _val_t, _test_t, _strong_val_t, _strong_test_t,
                )

        metadata.setdefault("dummy_baselines", {}) \
            .setdefault(str(target_type), {})[cur_target_name] = _db_report.to_dict()

        # Invert strongest dummy preds to y-scale so the verdict block compares both numbers on the same scale.
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
                # Multi-base specs (linear_residual_multi from forward-stepwise
                # auto-promotion) store the extra base columns alongside the
                # primary; transform.inverse needs the FULL (n, 1+K) matrix
                # whose alpha count matches fitted_params['alphas']. Without
                # this the inverse raises "base has 1 columns but fitted
                # alphas has K entries" -- caught by the outer try/except as
                # a WARNING, but the y-scale dummy metric is then missing
                # from metadata. Reproduced by fuzz c0047 (mode=legacy,
                # multi-base auto-promoted to linresM-num_1+num_dep).
                _extra_bases = tuple(_matching_spec.get("extra_base_columns") or ())
                _raw_target_col = _matching_spec["target_col"]
                _raw_y_full = target_by_type.get(target_type, {}).get(_raw_target_col)
                _y_scale_dummy_metrics: dict[str, dict[str, float]] = {}
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
                    # Skip cleanly when any extra base is missing from this
                    # split (avoid a deep traceback inside the inverse).
                    if any(_eb not in _split_df.columns for _eb in _extra_bases):
                        continue
                    if _extra_bases:
                        _base_split = np.column_stack(
                            [np.asarray(_split_df[_base_col], dtype=np.float64)]
                            + [np.asarray(_split_df[_eb], dtype=np.float64) for _eb in _extra_bases]
                        )
                    else:
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
        # Surface failure on TWO keys so CI gates can detect "baselines silently failed"
        # without having to scan logs:
        #   - dummy_baselines_failures[tt][target_name] = error message (rich detail)
        #   - dummy_baselines_status                    = "failed" (top-level boolean-like sentinel)
        # Pre-fix only the deep-nested failures key was set; a CI assertion checking
        # ``"dummy_baselines" in metadata`` for release-gating saw the slot remain absent on
        # failure (which read as "step never ran" rather than "step failed") AND the failure
        # key was easy to miss because it's not part of the standard happy-path key set.
        metadata.setdefault("dummy_baselines_failures", {}) \
            .setdefault(str(target_type), {})[cur_target_name] = str(_db_err)
        # Top-level status sentinel: any failure flips this from missing/"ok" to "failed".
        # Callers can gate on ``metadata.get("dummy_baselines_status") != "failed"``.
        metadata["dummy_baselines_status"] = "failed"
        metadata.setdefault("dummy_baselines_status_detail", []).append({
            "target_type": str(target_type),
            "target_name": cur_target_name,
            "error_type": type(_db_err).__name__,
            "error": str(_db_err),
        })

    return metadata
