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
from types import SimpleNamespace as _SN
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._format import format_metric as _fmt, short_model_tag as _short_tag_fn, strip_shim_suffix as _strip
from ..composite import CompositeCrossTargetEnsemble as _CrossEns, CompositeTargetEstimator as _CTE
from ..composite import compute_oof_holdout_predictions, get_transform
from ..composite_transforms import is_composite_target_name
from ..dummy_baselines import format_suite_end_summary
from ..evaluation import report_model_perf
from .utils import _build_full_column_from_splits, _entry_metric

logger = logging.getLogger(__name__)


def run_composite_post_processing(
    *,
    models: Dict,
    metadata: Dict,
    target_by_type: Dict,
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
    plot_file: Optional[str],
    verbose: bool,
) -> Tuple[Dict, Dict]:
    """Run composite wrapping, cross-target ensemble, and suite-end summary.

    Returns updated (models, metadata).
    """
    # 6. COMPOSITE-TARGET WRAPPING (post-fit, y-scale predictions)
    # ==================================================================================
    #
    # The per-target loop trained models on the T-scale composite
    # target (e.g. ``T = y - alpha*base - beta`` for linear_residual).
    # Predict-time those models return T-scale, which is useless for
    # downstream consumers expecting the original y-scale. Wrap each
    # fitted composite-target model in a ``CompositeTargetEstimator``
    # so ``model.predict(X)`` automatically applies the inverse
    # transform and returns y-scale.
    #
    # Wrapping is post-hoc (no re-training): the wrapper takes the
    # ALREADY-fitted inner model + the spec's fitted_params and adds
    # the inverse / clip / fallback machinery on top.
    composite_specs_by_target_type = metadata.get("composite_target_specs", {}) or {}
    # Train-prediction cache shared across the y-scale-metrics block
    # (post-wrap) and the cross-target-ensemble RMSE block. Key is
    # ``id(model)`` of the wrapped / raw component; value is the
    # y-scale prediction array on ``filtered_train_df``. The y-scale-
    # metrics block fills it; the ensemble block reads from it before
    # falling back to a fresh predict call. Saves K predict calls per
    # target on the LightGBM/XGB hot path.
    _train_pred_cache: Dict[int, np.ndarray] = {}
    if composite_specs_by_target_type:
        from ..composite import CompositeTargetEstimator as _CTE
        for _tt_w, _by_name in (models or {}).items():
            if not isinstance(_by_name, dict):
                continue
            _tt_specs = composite_specs_by_target_type.get(str(_tt_w), {})
            if not _tt_specs:
                continue
            # Build ``composite_name -> (original_target, spec)`` lookup
            # so wrapping is O(K) per pass.
            _name_to_spec: Dict[str, Tuple[str, Dict[str, Any]]] = {}
            for _orig_tname, _spec_list in _tt_specs.items():
                for _spec in _spec_list:
                    _name_to_spec[_spec["name"]] = (_orig_tname, _spec)
            for _composite_name, _entries in list(_by_name.items()):
                if _composite_name not in _name_to_spec:
                    continue  # not a composite target
                _orig_tname, _spec = _name_to_spec[_composite_name]
                # ``y_train`` for wrapping = original y values (NOT T)
                # at the train rows the wrapper saw at fit time.
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
                    # Entries may be plain estimators OR wrapper objects
                    # carrying the model on ``.model``. Try ``.model``
                    # first, fall back to the entry itself.
                    _inner = getattr(_entry, "model", None) or _entry
                    if not hasattr(_inner, "predict"):
                        # Not an estimator -- skip (e.g. metadata-only
                        # placeholder entry).
                        continue
                    try:
                        _wrapper = _CTE.from_fitted_inner(
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
                    # Replace the inner model on the entry (preserve
                    # auxiliary metadata: columns, model_name, metrics).
                    if hasattr(_entry, "model"):
                        try:
                            _entry.model = _wrapper
                        except Exception:
                            # Read-only attribute: replace the entry
                            # itself with the wrapper.
                            _entries[_i] = _wrapper
                    else:
                        _entries[_i] = _wrapper
                logger.info(
                    "[CompositeTargetEstimator] wrapped %d model(s) for composite "
                    "target '%s'; predictions now y-scale.",
                    len(_entries), _composite_name,
                )
                # Compute parallel y-scale RMSE / MAE for each wrapped
                # entry on train + val + test slices. The per-target
                # loop's metrics were computed on T-scale BEFORE wrap;
                # this block fills the y-scale gap so callers can
                # compare composite to raw on the same scale.
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
                    _entry_y_scores: Dict[str, Dict[str, float]] = {}
                    for _split_name, _split_idx, _split_df in (
                        ("train", filtered_train_idx, filtered_train_df),
                        ("val", filtered_val_idx, filtered_val_df),
                        ("test", test_idx, test_df_pd),
                    ):
                        if _split_idx is None or _split_df is None:
                            continue
                        try:
                            _y_split = _y_arr_metric[_split_idx]
                            _y_pred = np.asarray(
                                _wrapper_for_score.predict(_split_df),
                                dtype=np.float64,
                            ).reshape(-1)
                            # F6 diagnostic (2026-05-11): suspicious RMSE_y values in the 05:03 TVT run (MLP-wrapped composite gave 0.49 on a target where init_score AR(1) baseline gives RMSE=11.12 -- impossibly good). Sample-log the first 3 (y_pred, y_true) pairs per split so the next run reveals whether the wrapper is returning y-scale predictions correctly OR there is a leakage / contract mismatch. Single line per entry x split keeps the spam bounded.
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
                            # Cache the train prediction for the
                            # cross-target ensemble RMSE block to
                            # avoid a second predict call on the same
                            # data.
                            if _split_name == "train":
                                _train_pred_cache[id(_wrapper_for_score)] = _y_pred
                            _diff = _y_pred - _y_split.astype(np.float64)
                            _finite = np.isfinite(_diff)
                            if _finite.sum() == 0:
                                continue
                            # R^2 = 1 - SS_res / SS_tot. When the split's
                            # y has zero variance R^2 is undefined; we
                            # emit NaN so the summary explicitly marks
                            # the degenerate case rather than 0.0.
                            _y_finite = _y_split.astype(np.float64)[_finite]
                            _ss_tot = float(np.sum(
                                (_y_finite - _y_finite.mean()) ** 2
                            ))
                            _ss_res = float(np.sum(
                                _diff[_finite] * _diff[_finite]
                            ))
                            _r2 = (1.0 - _ss_res / _ss_tot) if _ss_tot > 0 else float("nan")
                            _entry_y_scores[_split_name] = {
                                "RMSE": float(
                                    np.sqrt(np.mean(_diff[_finite] * _diff[_finite]))
                                ),
                                "MAE": float(np.mean(np.abs(_diff[_finite]))),
                                "R2": _r2,
                                "n_rows_finite": int(_finite.sum()),
                            }
                        except Exception:
                            # Best-effort: any predict failure simply
                            # omits that split's y-scale metrics.
                            continue
                    _metrics_dict.append({
                        "model_name": getattr(_entry, "model_name", None),
                        "metrics": _entry_y_scores,
                    })
                    # User-facing fix (2026-05-11): the per-target loop
                    # printed RMSE / MAE / R^2 on the T-scale (composite
                    # target before inverse), which is apples-to-oranges
                    # vs raw-target models. Log a y-scale summary here
                    # so the user sees the COMPARABLE numbers in the
                    # script output for each wrapped composite model.
                    if _entry_y_scores:
                        from .._format import (
                            format_metric as _fmt,
                            strip_shim_suffix as _strip,
                        )
                        _y_summary_parts: List[str] = []
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
                            # B1-v2 fix (2026-05-11): drill into the WRAPPED model. After wrapping, ``_entry.model`` IS the CompositeTargetEstimator -- using its type name in the log gives the unhelpful ``model='CompositeTargetEstimator'`` (5 entries in a row, all identical). Look one level deeper at ``_entry.model.base_estimator`` (the actual inner cb / xgb / lgb / linear / mlp) for the diagnostic name.
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

    # ==================================================================================
    # 7. CROSS-TARGET ENSEMBLE (post-wrap; opt-in via config)
    # ==================================================================================
    #
    # After every composite-target model is wrapped to y-scale we can
    # combine them into one final predictor per original target. The
    # ensemble is OPT-IN via ``cross_target_ensemble_strategy`` and
    # produces a SimpleNamespace entry under
    # ``models[type][f"_CT_ENSEMBLE__{original_target}"]`` so downstream
    # consumers can pick it without having to know which composite to
    # trust.
    _ce_strategy = getattr(
        composite_target_discovery_config, "cross_target_ensemble_strategy", "off",
    )
    # Diagnostic: emit a one-line state banner whenever the user has
    # composite discovery enabled, regardless of whether the gate
    # actually opens. Without this, users who set
    # ``cross_target_ensemble_strategy="nnls_stack"`` but get no
    # ``[CompositeCrossTargetEnsemble] ...`` lines have no way to tell
    # whether the gate was closed (strategy=off, no specs) or whether
    # the build silently failed for every target. Emitting the banner
    # unconditionally turns "no log lines" into a debuggable signal.
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

        # R10c bug #4 fix: cross-target ensemble must call
        # ``inner.predict`` on input that has already been transformed
        # by the inner's ``pre_pipeline`` (SimpleImputer + StandardScaler
        # for linear models, identity for tree models). Without this,
        # LinearRegression / Ridge components blow up on raw frames
        # with NaN because the imputer never ran. Wrap each raw
        # component in a thin pipeline-aware shim that applies the
        # entry's pre_pipeline before delegating to the model.
        # Composite-target components (wrapped via
        # ``CompositeTargetEstimator``) handle this internally and
        # don't need the shim.
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
                        # Some pipelines reject pandas vs polars
                        # mismatches at the boundary; fall through to
                        # the inner predict which will raise a more
                        # descriptive error.
                        X_in = X
                return self._model.predict(X_in)

            def __repr__(self):
                return f"_PrePipelinePredictShim({self._name})"

        for _tt_e, _tt_specs in composite_specs_by_target_type.items():
            if not _tt_specs:
                continue
            # StrEnum: ``models.get(str_key)`` is hash-equivalent to
            # ``models.get(enum_key)`` so a plain string key works here.
            # Guard with explicit-skip log when the target type has no
            # trained models (e.g. dropped at split time) so users see
            # WHY the ensemble didn't fire for that type rather than a
            # silent skip.
            if _tt_e not in (models or {}):
                logger.info(
                    "[CompositeCrossTargetEnsemble] target_type='%s': no models "
                    "registered; ensemble skipped.", _tt_e,
                )
                continue
            for _orig_tname, _spec_list in _tt_specs.items():
                # Collect all wrapped composite-target entries plus the
                # raw-target entries for this original target. The raw
                # entries are at ``models[tt][orig_tname]``.
                _components: List[Any] = []
                _component_names: List[str] = []
                _orig_entries = (models or {}).get(_tt_e, {}).get(_orig_tname, []) or []
                for _i, _entry in enumerate(_orig_entries):
                    _inner = getattr(_entry, "model", None) or _entry
                    if not hasattr(_inner, "predict"):
                        continue
                    # Raw components: apply entry's pre_pipeline before
                    # predict. ``model_obj.pre_pipeline`` may be None for
                    # tree models (no preprocessing needed).
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
                        # Composite entries: CompositeTargetEstimator
                        # wrappers already manage their own transform;
                        # pre_pipeline (if any) is the OUTER frame-prep
                        # that should also be applied. Same shim.
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
                # Score every component on the train slice in y-scale.
                # Wrapped composite-target components predict in y-scale
                # via their inverse layer; raw-target components predict
                # y-scale directly. Use the same train rows the wrappers
                # were fitted against to keep the comparison fair.
                _y_full_for_rmse = target_by_type.get(_tt_e, {}).get(_orig_tname)
                _component_train_rmses: List[float] = []
                if _y_full_for_rmse is not None:
                    _y_train_for_rmse = np.asarray(_y_full_for_rmse)[filtered_train_idx]
                    for _comp, _name in zip(_components, _component_names):
                        try:
                            # I1 fix (2026-05-11): cache key is the INNER model id, not the shim id. The wrap pass populated ``_train_pred_cache`` keyed by the wrapper / inner ``id()``; the ensemble pass builds NEW shim instances so ``id(_comp)`` never hits. Look up via the inner instead and only fall back to ``id(_comp)`` for safety.
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
                # If oof_holdout_frac > 0, replace train-RMSE proxy
                # with honest holdout predictions: re-fit each
                # component on (1-frac) of train and predict on the
                # held-out frac. This is the correct objective for
                # ensemble weighting / stacking; the cost is one
                # extra fit per component.
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
                    # Build per-spec base column on filtered_train_df
                    # rows (composite components need this for the
                    # transform.forward step inside the OOF helper).
                    _base_full_per_spec: Dict[str, np.ndarray] = {}
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
                    _component_specs: List[Optional[Dict[str, Any]]] = []
                    for _name in _component_names:
                        if _name.startswith("raw#"):
                            _component_specs.append(None)
                        else:
                            # Composite name format "{compname}#{i}".
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
                                    "oof_random_state", 42,
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
                        # Re-align components / names / rmses to the
                        # surviving set returned by the OOF helper.
                        _surviving_set = set(_surviving)
                        _oof_components = [
                            c for c, n in zip(_components, _component_names)
                            if n in _surviving_set
                        ]
                        _oof_names = list(_surviving)
                        # Compute holdout RMSE per surviving component.
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
                        # Use OOF holdout predictions if available
                        # (honest stacking), otherwise the train-set
                        # predictions (biased but always available).
                        if _oof_pred_matrix is not None and _oof_pred_matrix.shape[1] > 0:
                            _pred_matrix = _oof_pred_matrix
                            _y_for_stack = _oof_y_holdout
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
                                # I1 (2026-05-11): inner-keyed cache lookup; see twin block above for the rationale. Shim-id keying never hit because shims are created per-pass.
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
                    # True OOF validation gate: if we have honest
                    # holdout predictions, compare ensemble holdout
                    # RMSE vs best single holdout RMSE. If the
                    # ensemble is worse, fall back to the best single.
                    if (_oof_pred_matrix is not None
                            and _oof_pred_matrix.shape[1] > 0
                            and isinstance(_ensemble, _CrossEns)):
                        try:
                            _ens_pred = _ensemble.predict(filtered_train_df)
                            # Use the SAME train rows used for OOF;
                            # ensemble.predict on filtered_train_df is
                            # in-sample for raw-target components but
                            # the comparison is component-fair (all
                            # components see the same X at same rows).
                            # NB: this is approximate -- the proper
                            # check would predict on stack_holdout. We
                            # do that next:
                            # Recompute ensemble preds on stack_holdout
                            # by weighted-combining the cached
                            # _oof_pred_matrix with the ensemble's
                            # weights.
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
                                _w_norm = _w_full / max(_w_full.sum(), 1e-12)
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
                # Optionally cap to the top-N components by weight
                # for online-latency-bounded serving. Configured via
                # ``max_inference_components``; 0 / None preserves
                # the full ensemble.
                _max_components = getattr(
                    composite_target_discovery_config,
                    "max_inference_components", None,
                )
                if (_max_components is not None and _max_components > 0
                        and isinstance(_ensemble, _CrossEns)):
                    _ensemble = _ensemble.cap_inference_components(
                        int(_max_components)
                    )
                # Wrap as a SimpleNamespace entry so downstream
                # iterators that expect ``.model`` / ``.columns`` keep
                # working. ``columns`` = union of inner columns; we
                # leave it empty -- the ensemble itself does not need
                # a fixed feature list (each component knows its own).
                from types import SimpleNamespace as _SN
                _ens_entry = _SN(
                    model=_ensemble,
                    model_name="CT_ENSEMBLE",
                    columns=None,
                    pre_pipeline=None,
                    metrics={},
                )
                # Dedicated key with prefix ``_CT_ENSEMBLE__`` so
                # downstream code that loops over composite-target
                # entries can trivially detect the ensemble entry and
                # skip / pick it.
                _ens_key = f"_CT_ENSEMBLE__{_orig_tname}"
                _by_name = models.setdefault(_tt_e, {})
                _by_name[_ens_key] = [_ens_entry]
                metadata.setdefault("composite_target_ensemble", {}) \
                    .setdefault(str(_tt_e), {})[_orig_tname] = (
                    _ensemble.export_metadata()
                    if hasattr(_ensemble, "export_metadata")
                    else {"strategy": "single_best_fallback"}
                )
                logger.info(
                    "[CompositeCrossTargetEnsemble] target='%s' built strategy='%s' "
                    "over %d component(s); stored at models[%s][%s].",
                    _orig_tname, _ce_strategy, len(_components),
                    _tt_e, _ens_key,
                )

                # 2026-05-12 (user request): route the cross-target ensemble
                # through the SAME ``report_model_perf`` pipeline that every
                # per-target model goes through. Previously the entry was
                # stored with ``metrics={}`` and no chart / log lines were
                # emitted, so users had no visual confirmation that the
                # ensemble was even built. Each split (val + test) gets a
                # scatter + residual chart + one-line metrics in the log,
                # using the SAME look as the real models. Guarded with a
                # broad try/except because an ensemble that has a
                # component shim that doesn't accept the suite's frame
                # shape would otherwise abort the whole suite.
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

    # 2026-05-10: suite-end dummy-baselines summary (D6) — cross-target
    # verdict block — canonical UPPERCASE WARN tokens.
    try:
        if metadata.get("dummy_baselines"):
            from ..dummy_baselines import format_suite_end_summary
            # Build {(target_type, target_name): {primary_metric: best_val,
            # "model_name": ...}} from the trained models. The model
            # metrics dict is keyed by metric NAME (e.g. "RMSE"); the
            # dummy primary_metric is split-prefixed (e.g. "val_RMSE").
            # Strip the "val_" prefix and look up via _entry_metric.
            _best_metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for _tt, _by_name in metadata.get("dummy_baselines", {}).items():
                for _tname, _rep_dict in _by_name.items():
                    _pm = _rep_dict.get("primary_metric")
                    if not _pm or not _pm.startswith("val_"):
                        continue
                    _metric_name = _pm[len("val_"):]  # "val_RMSE" -> "RMSE"
                    _model_list = models.get(_tt, {}).get(_tname, [])
                    if not _model_list:
                        continue
                    # Pick best model by primary metric. Minimize for
                    # RMSE/MAE/log_loss/pinball; maximize for everything
                    # else (NDCG / AUC).
                    _is_minimize = (
                        "RMSE" in _metric_name or "MAE" in _metric_name
                        or "log_loss" in _metric_name or "pinball" in _metric_name
                    )
                    # For composite targets: prefer the y-scale model
                    # metric (post-inverse, comparable to raw / y-scale
                    # dummy) over the T-scale ``_entry_metric`` value
                    # that was computed during the per-target loop on
                    # the unwrapped inner model. The y-scale numbers
                    # live in metadata["composite_target_y_scale_metrics"]
                    # populated by the wrap pass at section 6.
                    _yscale_entries = (
                        metadata.get("composite_target_y_scale_metrics", {})
                        .get(str(_tt), {})
                        .get(_tname, [])
                    )
                    _best_val: Optional[float] = None
                    _best_name = "-"
                    if _yscale_entries:
                        # y-scale path: iterate stored entries.
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
            # B2 (2026-05-11): build composite->raw target map so the verdict block uses the raw target's dummy (median(y_raw) constant) as the true trivial baseline -- not the inverted-T fake baseline that uses fitted alpha.
            _composite_to_raw: Dict[Tuple[str, str], str] = {}
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
