"""
Phase 6-7: composite-target post-processing.

1. Composite-target wrapping — wraps fitted T-scale models in ``CompositeTargetEstimator``
   so predictions are y-scale, then computes y-scale RMSE/MAE/R² per split.
2. Cross-target ensemble — opt-in ensemble over composite + raw components
   (mean / linear_stack / nnls_stack / oof_weighted).
3. Suite-end dummy-baselines summary — cross-target verdict block.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._format import format_metric as _fmt, short_model_tag as _short_tag_fn, strip_shim_suffix as _strip
from ..composite import CompositeCrossTargetEnsemble as _CrossEns, CompositeTargetEstimator
from ..composite import compute_oof_holdout_predictions, get_transform
from ..composite_transforms import is_composite_target_name
from ..dummy_baselines import format_suite_end_summary
from ..evaluation import report_model_perf
from .utils import _build_full_column_from_splits, _entry_metric

logger = logging.getLogger(__name__)

_DEFAULT_OOF_RANDOM_STATE = 42
_PROB_NORM_EPS = 1e-12


def _run_composite_target_wrapping(
    *,
    models: dict,
    metadata: dict,
    target_by_type: dict,
    composite_specs_by_target_type: dict,
    filtered_train_idx,
    filtered_train_df,
    filtered_val_idx,
    filtered_val_df,
    test_idx,
    test_df_pd,
) -> dict[int, np.ndarray]:
    """Wrap T-scale inner models in CompositeTargetEstimator so predict() returns y-scale; record y-scale RMSE/MAE/R2 per split.

    Mutates ``models`` in-place (replaces each composite-target inner with its wrapper) and writes ``metadata["composite_target_y_scale_metrics"]``.
    Returns the train-prediction cache (keyed by ``id(wrapper)``) so the downstream cross-target ensemble block can reuse the predictions
    without re-calling ``.predict`` on the wrapped models.
    """
    _train_pred_cache: dict[int, np.ndarray] = {}
    for _tt_w, _by_name in (models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        _tt_specs = composite_specs_by_target_type.get(str(_tt_w), {})
        if not _tt_specs:
            continue
        _name_to_spec: dict[str, tuple[str, dict[str, Any]]] = {}
        for _orig_tname, _spec_list in _tt_specs.items():
            for _spec in _spec_list:
                _name_to_spec[_spec["name"]] = (_orig_tname, _spec)
        for _composite_name, _entries in list(_by_name.items()):
            if _composite_name not in _name_to_spec:
                continue
            _orig_tname, _spec = _name_to_spec[_composite_name]
            # y_train for wrapping is the ORIGINAL y (not T) at the train rows the wrapper saw at fit time.
            _y_full = target_by_type.get(_tt_w, {}).get(_orig_tname)
            if _y_full is None:
                logger.warning(
                    "[CompositeTargetEstimator] missing original target '%s' "
                    "in target_by_type for composite='%s'; skipping wrap. "
                    "Predictions will remain in T-scale.",
                    _orig_tname, _composite_name,
                )
                continue
            try:
                _y_train_for_wrap = np.asarray(_y_full)[filtered_train_idx]
            except Exception as _y_err:
                logger.warning(
                    "[CompositeTargetEstimator] cannot align y_train for '%s': %s. "
                    "Skipping wrap.",
                    _composite_name, _y_err,
                )
                continue
            if not isinstance(_entries, list):
                continue
            for _i, _entry in enumerate(_entries):
                _inner = getattr(_entry, "model", None) or _entry
                if not hasattr(_inner, "predict"):
                    continue
                try:
                    _wrapper = CompositeTargetEstimator.from_fitted_inner(
                        fitted_inner=_inner,
                        transform_name=_spec["transform_name"],
                        base_column=_spec["base_column"],
                        transform_fitted_params=_spec["fitted_params"],
                        y_train=_y_train_for_wrap,
                    )
                except Exception as _wrap_err:
                    logger.warning(
                        "[CompositeTargetEstimator] wrap failed for '%s' (entry %d): %s. "
                        "Predictions will remain in T-scale.",
                        _composite_name, _i, _wrap_err,
                    )
                    continue
                # Preserve auxiliary metadata (columns, model_name, metrics) by replacing inner on entry.
                if hasattr(_entry, "model"):
                    try:
                        _entry.model = _wrapper
                    except Exception:
                        # Read-only attribute: replace the entry itself.
                        _entries[_i] = _wrapper
                else:
                    _entries[_i] = _wrapper
            logger.info(
                "[CompositeTargetEstimator] wrapped %d model(s) for composite "
                "target '%s'; predictions now y-scale.",
                len(_entries), _composite_name,
            )
            # Compute y-scale RMSE/MAE/R2 per split so composite is comparable to raw (per-target metrics were T-scale).
            _metrics_dict = metadata.setdefault(
                "composite_target_y_scale_metrics", {},
            ).setdefault(str(_tt_w), {}).setdefault(_composite_name, [])
            _metrics_dict.clear()
            _y_full_metric = target_by_type.get(_tt_w, {}).get(_orig_tname)
            if _y_full_metric is None:
                continue
            _y_arr_metric = np.asarray(_y_full_metric)
            for _entry in _entries:
                _wrapper_for_score = getattr(_entry, "model", None) or _entry
                _entry_y_scores: dict[str, dict[str, float]] = {}
                for _split_name, _split_idx, _split_df in (
                    ("train", filtered_train_idx, filtered_train_df),
                    ("val", filtered_val_idx, filtered_val_df),
                    ("test", test_idx, test_df_pd),
                ):
                    if _split_idx is None or _split_df is None:
                        continue
                    try:
                        _y_split = _y_arr_metric[_split_idx]
                        # Wrapped (post-clip) prediction = today's headline value. Train RMSE here is optimistic by construction:
                        # the clip is [y_train_min, y_train_max], train rows are in-envelope, clip is a no-op. Val / test rows
                        # may drift outside; the clip then narrows the headline RMSE. To make that contribution explicit we ALSO
                        # capture the raw (pre-clip) prediction via ``predict_pre_clip`` and emit a parallel metric block.
                        _y_pred_wrapped = np.asarray(
                            _wrapper_for_score.predict(_split_df),
                            dtype=np.float64,
                        ).reshape(-1)
                        if hasattr(_wrapper_for_score, "predict_pre_clip"):
                            _y_pred_raw = np.asarray(
                                _wrapper_for_score.predict_pre_clip(_split_df),
                                dtype=np.float64,
                            ).reshape(-1)
                        else:
                            # Inner is not a CompositeTargetEstimator (raw / passthrough); raw == wrapped is the honest answer.
                            _y_pred_raw = _y_pred_wrapped
                        # Use wrapped predictions for sample-log, cache, and the headline metric block (back-compat).
                        _y_pred = _y_pred_wrapped
                        # Sample-log the first 3 (y_pred, y_true) pairs per split as a leakage / contract sanity check.
                        if _split_idx is not None and len(_y_split) > 0:
                            _n_dbg = min(3, len(_y_split))
                            _pairs = ", ".join(
                                f"({_y_pred[_i]:.3f}, {_y_split[_i]:.3f})"
                                for _i in range(_n_dbg)
                            )
                            _outer_dbg = getattr(_entry, "model", None) or _entry
                            _inner_dbg = getattr(_outer_dbg, "base_estimator", None) or getattr(_outer_dbg, "estimator_", None) or _outer_dbg
                            logger.debug(
                                "[CompositeTargetEstimator.diag] inner=%s split=%s sample(y_hat, y_true) = %s",
                                type(_inner_dbg).__name__, _split_name, _pairs,
                            )
                        if _split_name == "train":
                            _train_pred_cache[id(_wrapper_for_score)] = _y_pred
                        _diff = _y_pred - _y_split.astype(np.float64)
                        _finite = np.isfinite(_diff)
                        if _finite.sum() == 0:
                            continue
                        # Zero-variance y => R2 undefined; emit NaN rather than 0.0 to mark the degenerate case.
                        _y_finite = _y_split.astype(np.float64)[_finite]
                        _ss_tot = float(np.sum(
                            (_y_finite - _y_finite.mean()) ** 2
                        ))
                        _ss_res = float(np.sum(
                            _diff[_finite] * _diff[_finite]
                        ))
                        _r2 = (1.0 - _ss_res / _ss_tot) if _ss_tot > 0 else float("nan")
                        _rmse_wrapped = float(np.sqrt(np.mean(_diff[_finite] * _diff[_finite])))
                        _mae_wrapped = float(np.mean(np.abs(_diff[_finite])))
                        # Raw (pre-clip) RMSE / MAE: align finite mask to raw predictions so any wrapped-only NaN doesn't
                        # bias the comparison. On in-envelope splits (train) raw and wrapped agree exactly.
                        _diff_raw = _y_pred_raw - _y_split.astype(np.float64)
                        _finite_raw = np.isfinite(_diff_raw)
                        if int(_finite_raw.sum()) > 0:
                            _rmse_raw = float(np.sqrt(np.mean(_diff_raw[_finite_raw] * _diff_raw[_finite_raw])))
                            _mae_raw = float(np.mean(np.abs(_diff_raw[_finite_raw])))
                        else:
                            _rmse_raw = float("nan")
                            _mae_raw = float("nan")
                        _entry_y_scores[_split_name] = {
                            "RMSE": _rmse_wrapped,
                            "MAE": _mae_wrapped,
                            "R2": _r2,
                            "n_rows_finite": int(_finite.sum()),
                            "RMSE_raw": _rmse_raw,
                            "RMSE_wrapped": _rmse_wrapped,
                            "MAE_raw": _mae_raw,
                            "MAE_wrapped": _mae_wrapped,
                        }
                    except Exception:
                        continue
                _metrics_dict.append({
                    "model_name": getattr(_entry, "model_name", None),
                    "metrics": _entry_y_scores,
                })
                # Log y-scale summary so composite numbers are comparable to raw-target models in script output.
                if _entry_y_scores:
                    _y_summary_parts: list[str] = []
                    for _split_name in ("train", "val", "test"):
                        _s = _entry_y_scores.get(_split_name)
                        if not _s:
                            continue
                        _y_summary_parts.append(
                            f"{_split_name.upper()}=RMSE_y:{_fmt(_s['RMSE'])} "
                            f"MAE_y:{_fmt(_s['MAE'])} "
                            f"R2_y:{_fmt(_s.get('R2', float('nan')), 4)}"
                        )
                    if _y_summary_parts:
                        # After wrapping _entry.model IS the CompositeTargetEstimator; drill into base_estimator for the actual inner type name.
                        _mn = getattr(_entry, "model_name", None)
                        if not _mn:
                            _outer = getattr(_entry, "model", None) or _entry
                            _inner_actual = getattr(_outer, "base_estimator", None) or getattr(_outer, "estimator_", None) or _outer
                            _mn = _strip(type(_inner_actual).__name__)
                        else:
                            _mn = _strip(_mn)
                        logger.info(
                            "[CompositeTargetEstimator] composite='%s' "
                            "model='%s' y-scale metrics (post-inverse, "
                            "comparable to raw): %s",
                            _composite_name, _mn,
                            " | ".join(_y_summary_parts),
                        )
    return _train_pred_cache


def _run_suite_end_dummy_baselines_summary(
    *,
    models: dict,
    metadata: dict,
    dummy_baselines_config,
) -> None:
    """Log the cross-target verdict block at suite end: best model per (target_type, target_name) vs. dummy baselines.

    Read-only on inputs. Picks the best model per target by val-split primary metric (min for RMSE/MAE/log_loss/pinball,
    max for NDCG/AUC), preferring y-scale composite metrics when available. Wrapped catch-all keeps the suite alive on
    summary errors -- the verdict block is diagnostic, never load-bearing.
    """
    try:
        if not metadata.get("dummy_baselines"):
            return
        from ..dummy_baselines import format_suite_end_summary
        # Build {(target_type, target_name): {primary_metric: best_val, "model_name": ...}} from trained models.
        # Model metrics key is the bare metric name (e.g. "RMSE"); dummy primary_metric is split-prefixed ("val_RMSE").
        _best_metrics: dict[tuple[str, str], dict[str, Any]] = {}
        for _tt, _by_name in metadata.get("dummy_baselines", {}).items():
            for _tname, _rep_dict in _by_name.items():
                _pm = _rep_dict.get("primary_metric")
                if not _pm or not _pm.startswith("val_"):
                    continue
                _metric_name = _pm[len("val_"):]
                _model_list = models.get(_tt, {}).get(_tname, [])
                if not _model_list:
                    continue
                # Minimize for RMSE/MAE/log_loss/pinball; maximize otherwise (NDCG/AUC).
                _is_minimize = (
                    "RMSE" in _metric_name or "MAE" in _metric_name
                    or "log_loss" in _metric_name or "pinball" in _metric_name
                )
                # For composite targets prefer y-scale metrics (post-inverse, comparable to raw / y-scale dummy).
                _yscale_entries = (
                    metadata.get("composite_target_y_scale_metrics", {})
                    .get(str(_tt), {})
                    .get(_tname, [])
                )
                _best_val: float | None = None
                _best_name = "-"
                if _yscale_entries:
                    for _ye in _yscale_entries:
                        _split_metric = _ye.get("metrics", {}).get("val", {})
                        _v = _split_metric.get(_metric_name)
                        if _v is None or not np.isfinite(_v):
                            continue
                        if (
                            _best_val is None
                            or (_is_minimize and _v < _best_val)
                            or (not _is_minimize and _v > _best_val)
                        ):
                            _best_val = float(_v)
                            _best_name = _ye.get("model_name") or "Composite"
                else:
                    for _m in _model_list:
                        _v = _entry_metric(_m, "val", _metric_name)
                        if not np.isfinite(_v):
                            continue
                        if (
                            _best_val is None
                            or (_is_minimize and _v < _best_val)
                            or (not _is_minimize and _v > _best_val)
                        ):
                            _best_val = _v
                            _best_name = getattr(_m, "model_name", None) or type(
                                getattr(_m, "model", _m)
                            ).__name__
                if _best_val is not None:
                    _best_metrics[(str(_tt), str(_tname))] = {
                        _pm: _best_val,
                        "model_name": _best_name,
                    }
        # composite -> raw target map so the verdict block uses the raw median(y_raw) constant as the trivial baseline
        # (not the inverted-T fake baseline that uses fitted alpha).
        _composite_to_raw: dict[tuple[str, str], str] = {}
        for _tt_str, _by_tname in metadata.get(
            "composite_target_specs", {}
        ).items():
            for _raw_tname, _spec_list in _by_tname.items():
                for _s in _spec_list or []:
                    _comp_name = _s.get("name")
                    if _comp_name:
                        _composite_to_raw[(_tt_str, _comp_name)] = _raw_tname
        _summary_text = format_suite_end_summary(
            dummy_baselines_metadata=metadata.get("dummy_baselines", {}),
            failures_metadata=metadata.get("dummy_baselines_failures", {}),
            best_model_metrics_by_target=_best_metrics if _best_metrics else None,
            min_lift=dummy_baselines_config.best_model_min_lift,
            composite_to_raw_target_map=_composite_to_raw if _composite_to_raw else None,
        )
        if _summary_text:
            logger.info(_summary_text)
    except Exception as _db_summary_err:
        logger.warning(
            "[DUMMY_BASELINES] suite-end summary failed: %s",
            _db_summary_err,
        )


def run_composite_post_processing(
    *,
    models: dict,
    metadata: dict,
    target_by_type: dict,
    composite_target_discovery_config,
    target_name: str,
    model_name: str,
    filtered_train_df,
    filtered_val_df,
    test_df_pd,
    filtered_train_idx,
    filtered_val_idx,
    test_idx,
    train_df_pd,
    val_df_pd,
    train_idx,
    val_idx,
    dummy_baselines_config,
    reporting_config,
    plot_file: str | None,
    verbose: bool,
) -> tuple[dict, dict]:
    """Run composite wrapping, cross-target ensemble, and suite-end summary.

    Returns updated (models, metadata).
    """
    # Composite-target wrapping: T-scale inner models get wrapped so predict() returns y-scale.
    composite_specs_by_target_type = metadata.get("composite_target_specs", {}) or {}
    # Train-prediction cache (key = id(wrapper)) populated by the wrapping block and reused by the cross-target ensemble block.
    _train_pred_cache: dict[int, np.ndarray] = {}
    if composite_specs_by_target_type:
        _train_pred_cache = _run_composite_target_wrapping(
            models=models,
            metadata=metadata,
            target_by_type=target_by_type,
            composite_specs_by_target_type=composite_specs_by_target_type,
            filtered_train_idx=filtered_train_idx,
            filtered_train_df=filtered_train_df,
            filtered_val_idx=filtered_val_idx,
            filtered_val_df=filtered_val_df,
            test_idx=test_idx,
            test_df_pd=test_df_pd,
        )

    # Cross-target ensemble (opt-in). Stored as a SimpleNamespace under models[type][f"_CT_ENSEMBLE__{original_target}"].
    _ce_strategy = getattr(
        composite_target_discovery_config, "cross_target_ensemble_strategy", "off",
    )
    # Unconditional banner when discovery is enabled so "no log lines" remains a debuggable signal.
    if composite_target_discovery_config.enabled:
        _n_specs_total = sum(
            sum(len(v) for v in _tt_specs.values())
            for _tt_specs in (composite_specs_by_target_type or {}).values()
        )
        logger.info(
            "[CompositeCrossTargetEnsemble] entry: strategy='%s', "
            "target_types=%d, composite_specs=%d",
            _ce_strategy,
            len(composite_specs_by_target_type or {}),
            _n_specs_total,
        )
    if (composite_target_discovery_config.enabled
            and _ce_strategy != "off"
            and composite_specs_by_target_type):
        from ..composite import CompositeCrossTargetEnsemble as _CrossEns

        # Raw components need entry.pre_pipeline applied before inner.predict (linear models with NaN-imputer pipelines).
        class _PrePipelinePredictShim:
            __slots__ = ("_model", "_pre_pipeline", "_name")

            def __init__(self, model, pre_pipeline, name):
                self._model = model
                self._pre_pipeline = pre_pipeline
                self._name = name

            def predict(self, X):
                X_in = X
                if self._pre_pipeline is not None:
                    try:
                        X_in = self._pre_pipeline.transform(X)
                    except Exception:
                        # Fall through so inner.predict raises the more descriptive error on pd/pl boundary mismatches.
                        X_in = X
                return self._model.predict(X_in)

            def __repr__(self):
                return f"_PrePipelinePredictShim({self._name})"

        for _tt_e, _tt_specs in composite_specs_by_target_type.items():
            if not _tt_specs:
                continue
            # StrEnum: models.get(str_key) is hash-equivalent to models.get(enum_key).
            if _tt_e not in (models or {}):
                logger.info(
                    "[CompositeCrossTargetEnsemble] target_type='%s': no models "
                    "registered; ensemble skipped.", _tt_e,
                )
                continue
            for _orig_tname, _spec_list in _tt_specs.items():
                # Collect raw-target + wrapped composite-target entries for this original target.
                _components: list[Any] = []
                _component_names: list[str] = []
                _orig_entries = (models or {}).get(_tt_e, {}).get(_orig_tname, []) or []
                for _i, _entry in enumerate(_orig_entries):
                    _inner = getattr(_entry, "model", None) or _entry
                    if not hasattr(_inner, "predict"):
                        continue
                    _pp = getattr(_entry, "pre_pipeline", None)
                    _name = f"raw#{_i}"
                    _components.append(
                        _PrePipelinePredictShim(_inner, _pp, _name)
                    )
                    _component_names.append(_name)
                for _spec in _spec_list:
                    _composite_entries = (models or {}).get(_tt_e, {}).get(
                        _spec["name"], []
                    ) or []
                    for _i, _entry in enumerate(_composite_entries):
                        _inner = getattr(_entry, "model", None) or _entry
                        if not hasattr(_inner, "predict"):
                            continue
                        # CTE wrappers handle the transform; pre_pipeline (if any) is outer frame-prep applied via the same shim.
                        _pp = getattr(_entry, "pre_pipeline", None)
                        _name = f"{_spec['name']}#{_i}"
                        _components.append(
                            _PrePipelinePredictShim(_inner, _pp, _name)
                        )
                        _component_names.append(_name)
                if len(_components) < 2:
                    logger.info(
                        "[CompositeCrossTargetEnsemble] target='%s': only %d "
                        "component(s); ensemble skipped.",
                        _orig_tname, len(_components),
                    )
                    continue
                # Score components on the train slice in y-scale (same rows wrappers were fitted on).
                _y_full_for_rmse = target_by_type.get(_tt_e, {}).get(_orig_tname)
                _component_train_rmses: list[float] = []
                if _y_full_for_rmse is not None:
                    _y_train_for_rmse = np.asarray(_y_full_for_rmse)[filtered_train_idx]
                    for _comp, _name in zip(_components, _component_names):
                        try:
                            # Cache key is the INNER model id; shims are built per-pass so id(_comp) never hits the wrap-pass cache.
                            _inner_for_cache = getattr(_comp, "_model", _comp)
                            _pred = _train_pred_cache.get(id(_inner_for_cache))
                            if _pred is None:
                                _pred = _train_pred_cache.get(id(_comp))
                            if _pred is None:
                                _pred = np.asarray(
                                    _comp.predict(filtered_train_df),
                                    dtype=np.float64,
                                ).reshape(-1)
                                _train_pred_cache[id(_inner_for_cache)] = _pred
                            _diff = _pred - _y_train_for_rmse.astype(np.float64)
                            _component_train_rmses.append(
                                float(np.sqrt(np.mean(_diff * _diff)))
                            )
                        except Exception as _rmse_err:
                            logger.warning(
                                "[CompositeCrossTargetEnsemble] could not score "
                                "component '%s' on train: %s. Skipping in "
                                "ensemble weighting.", _name, _rmse_err,
                            )
                            _component_train_rmses.append(float("nan"))
                else:
                    _component_train_rmses = [float("nan")] * len(_components)
                _rmse_arr = np.asarray(_component_train_rmses, dtype=np.float64)
                _finite = np.isfinite(_rmse_arr)
                if _finite.sum() == 0:
                    logger.warning(
                        "[CompositeCrossTargetEnsemble] target='%s': no "
                        "component scored on train; ensemble skipped.",
                        _orig_tname,
                    )
                    continue
                if not _finite.all():
                    _rmse_arr[~_finite] = float(np.median(_rmse_arr[_finite]))
                # If oof_holdout_frac > 0, replace train-RMSE proxy with honest holdout (re-fit on 1-frac, predict on frac).
                _oof_frac = float(getattr(
                    composite_target_discovery_config, "oof_holdout_frac", 0.0,
                ))
                _oof_y_full = _y_full_for_rmse
                _oof_pred_matrix = None
                _oof_y_holdout = None
                _oof_components = _components
                _oof_names = _component_names
                _oof_rmses = _rmse_arr  # train-RMSE proxy by default
                if _oof_frac > 0.0 and _oof_y_full is not None:
                    from ..composite import compute_oof_holdout_predictions
                    # Per-spec base column on filtered_train_df rows for transform.forward inside the OOF helper.
                    _base_full_per_spec: dict[str, np.ndarray] = {}
                    for _spec_for_oof in _spec_list:
                        _b = _build_full_column_from_splits(
                            _spec_for_oof["base_column"],
                            train_df_pd, val_df_pd, test_df_pd,
                            train_idx, val_idx, test_idx,
                            n_total=len(_oof_y_full),
                        )
                        _base_full_per_spec[_spec_for_oof["base_column"]] = (
                            _b[filtered_train_idx]
                        )
                    # Build the spec-or-None list parallel to components.
                    _component_specs: list[dict[str, Any] | None] = []
                    for _name in _component_names:
                        if _name.startswith("raw#"):
                            _component_specs.append(None)
                        else:
                            _comp_name = _name.split("#", 1)[0]
                            _matching = next(
                                (s for s in _spec_list
                                 if s["name"] == _comp_name), None,
                            )
                            _component_specs.append(_matching)
                    try:
                        _oof_pred_matrix, _oof_y_holdout, _surviving = (
                            compute_oof_holdout_predictions(
                                component_models=_components,
                                component_names=_component_names,
                                component_specs=_component_specs,
                                train_X=filtered_train_df,
                                y_train_full=np.asarray(_oof_y_full)[filtered_train_idx],
                                base_train_full_per_spec=_base_full_per_spec,
                                holdout_frac=_oof_frac,
                                random_state=getattr(
                                    composite_target_discovery_config,
                                    "oof_random_state", _DEFAULT_OOF_RANDOM_STATE,
                                ),
                            )
                        )
                    except Exception as _oof_err:
                        logger.warning(
                            "[CompositeCrossTargetEnsemble] OOF computation failed "
                            "for target='%s': %s. Falling back to train-RMSE proxy.",
                            _orig_tname, _oof_err,
                        )
                        _oof_pred_matrix, _oof_y_holdout, _surviving = (
                            None, None, [],
                        )
                    if _oof_pred_matrix is not None and _oof_pred_matrix.shape[1] > 0:
                        # Re-align to the surviving set returned by the OOF helper.
                        _surviving_set = set(_surviving)
                        _oof_components = [
                            c for c, n in zip(_components, _component_names)
                            if n in _surviving_set
                        ]
                        _oof_names = list(_surviving)
                        _oof_rmses_list = []
                        for _i_col in range(_oof_pred_matrix.shape[1]):
                            _diff = _oof_pred_matrix[:, _i_col] - _oof_y_holdout
                            _finite = np.isfinite(_diff)
                            if _finite.sum() == 0:
                                _oof_rmses_list.append(float("nan"))
                            else:
                                _oof_rmses_list.append(float(np.sqrt(np.mean(
                                    _diff[_finite] * _diff[_finite]
                                ))))
                        _oof_rmses = np.asarray(_oof_rmses_list, dtype=np.float64)
                        logger.info(
                            "[CompositeCrossTargetEnsemble] target='%s' using "
                            "honest OOF holdout (frac=%.2f, n=%d) for ensemble "
                            "weights / stacking.",
                            _orig_tname, _oof_frac, len(_oof_y_holdout),
                        )

                try:
                    if _ce_strategy == "mean":
                        _ensemble = _CrossEns.from_uniform_weights(
                            component_models=_oof_components,
                            component_names=_oof_names,
                        )
                    elif _ce_strategy in ("linear_stack", "nnls_stack"):
                        # Honest OOF preds if available, else biased train-set preds.
                        if _oof_pred_matrix is not None and _oof_pred_matrix.shape[1] > 0:
                            _pred_matrix = _oof_pred_matrix
                            _y_for_stack = _oof_y_holdout
                            # Stacking-aware gate (opt-in) -- drop components whose NNLS
                            # weight on the honest OOF preds falls below the configured
                            # threshold BEFORE running the actual stacker. Keeps the final
                            # weight vector concentrated on signal-bearing components.
                            if (getattr(
                                composite_target_discovery_config,
                                "stacking_aware_gate_enabled", False,
                            ) and _pred_matrix.shape[1] >= 2):
                                try:
                                    from ..composite_stacking import stacking_aware_gate
                                    _gate_preds = {
                                        _oof_names[_i]: _pred_matrix[:, _i]
                                        for _i in range(_pred_matrix.shape[1])
                                    }
                                    _gate_min = float(getattr(
                                        composite_target_discovery_config,
                                        "stacking_aware_gate_min_weight", 0.05,
                                    ))
                                    _survivors, _gate_w = stacking_aware_gate(
                                        _gate_preds, _y_for_stack, min_weight=_gate_min,
                                    )
                                    if 2 <= len(_survivors) < len(_oof_names):
                                        _keep_mask = np.array([
                                            n in set(_survivors) for n in _oof_names
                                        ], dtype=bool)
                                        _pred_matrix = _pred_matrix[:, _keep_mask]
                                        _oof_components = [
                                            c for c, k in zip(_oof_components, _keep_mask) if k
                                        ]
                                        _oof_names = [
                                            n for n, k in zip(_oof_names, _keep_mask) if k
                                        ]
                                        _oof_rmses = _oof_rmses[_keep_mask]
                                        logger.info(
                                            "[CompositeCrossTargetEnsemble] target='%s' "
                                            "stacking_aware_gate kept %d of %d components "
                                            "(min_weight=%.3f).",
                                            _orig_tname, len(_survivors),
                                            len(_gate_w), _gate_min,
                                        )
                                except Exception as _gate_err:
                                    logger.warning(
                                        "[CompositeCrossTargetEnsemble] stacking_aware_gate "
                                        "failed for target='%s': %s. Proceeding with full set.",
                                        _orig_tname, _gate_err,
                                    )
                        else:
                            _y_for_stack = (
                                np.asarray(_oof_y_full)[filtered_train_idx]
                                if _oof_y_full is not None else None
                            )
                            if _y_for_stack is None:
                                raise RuntimeError(
                                    "stacking requires train target alignment"
                                )
                            _pred_matrix_cols = []
                            for _comp, _name in zip(_oof_components, _oof_names):
                                # Inner-keyed cache lookup (shim ids are per-pass and never hit the wrap-pass cache).
                                _inner_for_cache = getattr(_comp, "_model", _comp)
                                _pred = _train_pred_cache.get(id(_inner_for_cache))
                                if _pred is None:
                                    _pred = _train_pred_cache.get(id(_comp))
                                if _pred is None:
                                    _pred = np.asarray(
                                        _comp.predict(filtered_train_df),
                                        dtype=np.float64,
                                    ).reshape(-1)
                                    _train_pred_cache[id(_inner_for_cache)] = _pred
                                _pred_matrix_cols.append(_pred)
                            _pred_matrix = np.column_stack(_pred_matrix_cols)
                        if _ce_strategy == "linear_stack":
                            _ensemble = _CrossEns.from_linear_stack(
                                component_models=_oof_components,
                                component_names=_oof_names,
                                component_predictions=_pred_matrix,
                                y_train=_y_for_stack,
                            )
                        else:  # nnls_stack
                            _ensemble = _CrossEns.from_nnls_stack(
                                component_models=_oof_components,
                                component_names=_oof_names,
                                component_predictions=_pred_matrix,
                                y_train=_y_for_stack,
                            )
                    else:  # "oof_weighted"
                        _ensemble = _CrossEns.from_train_metrics(
                            component_models=_oof_components,
                            component_names=_oof_names,
                            component_train_rmse=_oof_rmses.tolist(),
                            baseline_train_rmse=None,
                        )
                    # OOF validation gate: fall back to best single if ensemble holdout RMSE > best-single holdout RMSE.
                    if (_oof_pred_matrix is not None
                            and _oof_pred_matrix.shape[1] > 0
                            and isinstance(_ensemble, _CrossEns)):
                        try:
                            _ens_pred = _ensemble.predict(filtered_train_df)
                            # Recompute ensemble preds on stack_holdout by weighted-combining the cached _oof_pred_matrix.
                            _w_full = np.asarray(_ensemble.weights, dtype=np.float64)
                            if _ce_strategy == "linear_stack":
                                _intercept = float(getattr(
                                    _ensemble, "_linear_stack_intercept", 0.0,
                                ))
                                _ens_holdout = (
                                    (_oof_pred_matrix * _w_full[None, :]).sum(axis=1)
                                    + _intercept
                                )
                            else:
                                _w_norm = _w_full / max(_w_full.sum(), _PROB_NORM_EPS)
                                _ens_holdout = (
                                    _oof_pred_matrix * _w_norm[None, :]
                                ).sum(axis=1)
                            _ens_diff = _ens_holdout - _oof_y_holdout
                            _ens_rmse = float(np.sqrt(np.mean(_ens_diff ** 2)))
                            _best_single_rmse = float(np.nanmin(_oof_rmses))
                            if _ens_rmse > _best_single_rmse:
                                _best_idx = int(np.nanargmin(_oof_rmses))
                                logger.warning(
                                    "[CompositeCrossTargetEnsemble] target='%s' "
                                    "honest OOF gate fired: ensemble RMSE %.4g > "
                                    "best single '%s' RMSE %.4g. Falling back to "
                                    "best single component.",
                                    _orig_tname, _ens_rmse,
                                    _oof_names[_best_idx], _best_single_rmse,
                                )
                                _ensemble = _oof_components[_best_idx]
                        except Exception as _gate_err:
                            logger.info(
                                "[CompositeCrossTargetEnsemble] OOF gate check "
                                "skipped (%s); ensemble retained.", _gate_err,
                            )
                except Exception as _ens_err:
                    logger.warning(
                        "[CompositeCrossTargetEnsemble] target='%s' build failed: "
                        "%s. Skipping.", _orig_tname, _ens_err,
                    )
                    continue
                # Optional top-N cap by weight for latency-bounded serving (0/None preserves full ensemble).
                _max_components = getattr(
                    composite_target_discovery_config,
                    "max_inference_components", None,
                )
                if (_max_components is not None and _max_components > 0
                        and isinstance(_ensemble, _CrossEns)):
                    _ensemble = _ensemble.cap_inference_components(
                        int(_max_components)
                    )
                # SimpleNamespace shim for downstream iterators expecting .model/.columns; columns=None since each component knows its own.
                _ens_entry = SimpleNamespace(
                    model=_ensemble,
                    model_name="CT_ENSEMBLE",
                    columns=None,
                    pre_pipeline=None,
                    metrics={},
                )
                _ens_key = f"_CT_ENSEMBLE__{_orig_tname}"
                _by_name = models.setdefault(_tt_e, {})
                _by_name[_ens_key] = [_ens_entry]
                metadata.setdefault("composite_target_ensemble", {}) \
                    .setdefault(str(_tt_e), {})[_orig_tname] = (
                    _ensemble.export_metadata()
                    if hasattr(_ensemble, "export_metadata")
                    else {"strategy": "single_best_fallback"}
                )
                # Stamp the chosen ensemble flavour into metadata["ensembles_chosen"] so the predict path can replay
                # the right combine for the cross-target slot (predict-path parity). The CT key reuses
                # the _CT_ENSEMBLE__ literal so the predict per-target lookup hits the same slot the loader produces.
                _ce_actual_strategy = getattr(_ensemble, "strategy", None) or _ce_strategy
                metadata.setdefault("ensembles_chosen", {}) \
                    .setdefault(str(_tt_e), {})[_ens_key] = str(_ce_actual_strategy)
                logger.info(
                    "[CompositeCrossTargetEnsemble] target='%s' built strategy='%s' "
                    "over %d component(s); stored at models[%s][%s].",
                    _orig_tname, _ce_strategy, len(_components),
                    _tt_e, _ens_key,
                )

                # Route the ensemble through report_model_perf so val/test get the same scatter + residual charts as real models.
                try:
                    from ..evaluation import report_model_perf
                    _ens_orig_y = target_by_type.get(_tt_e, {}).get(_orig_tname)
                    if _ens_orig_y is not None:
                        _ens_y_arr = np.asarray(_ens_orig_y)
                        _ens_model_name = (
                            f"CT_ENSEMBLE[{_ce_strategy}] {target_name} "
                            f"{model_name} {_orig_tname}"
                        )
                        _ens_columns = (
                            list(getattr(filtered_train_df, "columns", []) or [])
                        )
                        _ens_common = dict(
                            columns=_ens_columns,
                            df=None, model=None,
                            model_name=_ens_model_name,
                            plot_outputs=getattr(reporting_config, "plot_outputs", None),
                            plot_dpi=getattr(reporting_config, "plot_dpi", None),
                            show_fi=False,
                            target_type=str(_tt_e),
                        )
                        for _split_name, _report_title, _split_idx, _split_df in (
                            ("val", "VAL (CT_ENSEMBLE) ", filtered_val_idx, filtered_val_df),
                            ("test", "TEST (CT_ENSEMBLE) ", test_idx, test_df_pd),
                        ):
                            if _split_idx is None or _split_df is None:
                                continue
                            try:
                                _y_split = _ens_y_arr[_split_idx]
                                _ens_preds = np.asarray(
                                    _ensemble.predict(_split_df),
                                    dtype=np.float64,
                                ).reshape(-1)
                                _common_split = dict(_ens_common)
                                if plot_file:
                                    _common_split["plot_file"] = (
                                        f"{plot_file}_ct_ensemble_{_orig_tname}_{_split_name}"
                                    )
                                report_model_perf(
                                    targets=_y_split,
                                    preds=_ens_preds, probs=None,
                                    report_title=_report_title,
                                    **_common_split,
                                )
                            except Exception as _split_err:
                                logger.warning(
                                    "[CompositeCrossTargetEnsemble] target='%s' "
                                    "split='%s' report_model_perf failed: %s. "
                                    "Continuing without ensemble chart for this split.",
                                    _orig_tname, _split_name, _split_err,
                                )
                except Exception as _ens_report_err:
                    logger.warning(
                        "[CompositeCrossTargetEnsemble] target='%s' could not emit "
                        "scatter / log charts: %s. The ensemble entry is still "
                        "stored at models[%s][%s] for downstream consumers.",
                        _orig_tname, _ens_report_err, _tt_e, _ens_key,
                    )

    _run_suite_end_dummy_baselines_summary(
        models=models,
        metadata=metadata,
        dummy_baselines_config=dummy_baselines_config,
    )

    return models, metadata
