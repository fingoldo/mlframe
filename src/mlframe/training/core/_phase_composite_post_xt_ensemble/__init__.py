"""Cross-target ensemble per-target helper carved out of ``_phase_composite_post.run_composite_post_processing``.

Holds the body of the inner ``for _orig_tname, _spec_list in _tt_specs.items()`` loop. The suite TrainingContext is passed in as ``ctx`` so the honest OOF split can read ``timestamps`` / ``sample_weights`` / ``group_ids`` (full-data-indexed, subset by ``filtered_train_idx``).
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np

from ...composite import CompositeCrossTargetEnsemble as _CrossEns
from ...composite import compute_oof_holdout_predictions
from ...composite.post_shim import PrePipelinePredictShim
from ..utils import _build_full_column_from_splits
from .._phase_composite_post_lag_predict import _LagPredictDeployableModel
from ._post_xt_ensemble_mtr import _build_mtr_per_column_ensemble

logger = logging.getLogger("mlframe.training.core._phase_composite_post")

_DEFAULT_OOF_RANDOM_STATE = 42


def _oof_subsample_positions(n: int, groups, cap: int, seed: int):
    """Row positions for a bounded OOF weight-estimation subsample.

    Group-aware when ``groups`` is supplied: keep WHOLE groups (sorted by a seeded permutation) until
    ~``cap`` rows are collected, so the group-disjoint OOF structure the weights consume is preserved.
    Falls back to a plain seeded random subsample without groups. Returns sorted positions, or ``None``
    when no subsample is needed / possible.
    """
    if cap <= 0 or n <= cap:
        return None
    rng = np.random.default_rng(seed)
    if groups is not None:
        g = np.asarray(groups)
        if g.shape[0] == n:
            uniq = rng.permutation(np.unique(g))
            keep, total = [], 0
            for gid in uniq:
                keep.append(gid)
                total += int(np.count_nonzero(g == gid))
                if total >= cap:
                    break
            pos = np.nonzero(np.isin(g, keep))[0]
            if 0 < pos.size < n:
                return np.sort(pos)
    return np.sort(rng.choice(n, size=cap, replace=False))


def _slice_frame_rows(frame, pos):
    """Row-subset a polars / pandas / ndarray frame by integer positions (REDUCES rows -- never a full copy)."""
    try:
        import polars as pl
        if isinstance(frame, pl.DataFrame):
            mask = np.zeros(frame.height, dtype=bool)
            mask[pos] = True
            return frame.filter(pl.Series(mask))
    except ImportError:
        pass
    if hasattr(frame, "iloc"):
        return frame.iloc[pos].reset_index(drop=True)
    return frame[pos]


def _build_cross_target_ensemble_for_target(
    *,
    _tt_e,
    _orig_tname,
    _spec_list,
    _ce_strategy: str,
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
    reporting_config,
    plot_file: str | None,
    _train_pred_cache: dict,
    ctx: Any = None,
) -> None:
    """Build CT_ENSEMBLE for one (target_type, original_target_name).

    Mutates ``models`` and ``metadata`` in place; same contract as the original inline loop body. ``ctx`` is the
    suite TrainingContext: its ``timestamps`` / ``sample_weights`` / ``group_ids`` (full-data-indexed) drive the
    time-aware, weighted, group-aware honest OOF split when present.
    """
    # Build-scoped train-prediction cache. The shared ``_train_pred_cache`` carries wrap-pass entries keyed by ``(id(inner_model),) + frame_key``; ``id()``-based keys are only meaningful while the underlying objects are alive, so we never let a builder-computed prediction leak to a sibling build. Reads consult the build-local dict first (this build's own writes), then the shared wrap-pass cache for this exact live frame; all writes go to the build-local dict, discarded when this call returns. This makes a stale cross-build hit impossible without hashing the (potentially TB-scale) frame.
    _build_pred_cache: dict[tuple, np.ndarray] = {}

    def _get_train_pred(_comp, _frame_key):
        _inner = getattr(_comp, "model", _comp)
        _key = (id(_inner),) + _frame_key
        _p = _build_pred_cache.get(_key)
        if _p is None:
            _p = _train_pred_cache.get(_key)
        if _p is None:
            _p = np.asarray(_comp.predict(filtered_train_df), dtype=np.float64).reshape(-1)
            _build_pred_cache[_key] = _p
        return _p

    # MULTI_TARGET_REGRESSION path. The general CT_ENSEMBLE flow below assumes 1-D y per component (sklearn metrics + honest-OOF blender solve a 1-D regression at the component level). For (N, K) MTR targets we build a per-column mean ensemble: stack each component's (N, K) predictions across a "component" axis, then average across components for a single (N, K) output. Equal-weight is the floor; per-column honest-OOF blended weights can swap in without changing the public deployable model interface.
    try:
        from mlframe.training import TargetTypes

        _is_mtr = str(_tt_e) == str(TargetTypes.MULTI_TARGET_REGRESSION) or (hasattr(_tt_e, "is_multi_target_regression") and _tt_e.is_multi_target_regression)
    except Exception:
        _is_mtr = False

    if _is_mtr:
        # Honest train-K-fold OOF NNLS weights (bench: bench_mtr_nnls_oof.py -- beats equal_mean on 8/8 seeds,
        # leak-free vs the old val-fold fit). The val fold was the early-stopping surface for the components, so
        # fitting per-column NNLS on it double-dipped a biased surface; we now derive the weights from a true
        # train-K-fold OOF stack on the TRAIN rows and inject them. Falls back to equal-mean if OOF fails.
        _fit_y_full = (target_by_type or {}).get(_tt_e, {}).get(_orig_tname)
        _oof_weights = None
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
                    from ._phase_composite_post_xt_mtr_oof import compute_mtr_oof_nnls_weights
                    _oof_random_state = int(getattr(
                        composite_target_discovery_config,
                        "oof_random_state", _DEFAULT_OOF_RANDOM_STATE,
                    ))
                    _oof_kfold_mtr = int(getattr(
                        composite_target_discovery_config, "oof_kfold", 5,
                    ))
                    _oof_weights = compute_mtr_oof_nnls_weights(
                        _mtr_components, filtered_train_df, _y_arr_mtr,
                        kfold=_oof_kfold_mtr, random_state=_oof_random_state,
                    )
            except Exception as _mtr_oof_err:
                logger.warning(
                    "[MTR CT_ENSEMBLE] target='%s': honest-OOF NNLS weighting failed (%s); forfeiting the "
                    "benched ~9%% NNLS win and falling back to equal-mean.",
                    _orig_tname, _mtr_oof_err,
                )
                _oof_weights = None
        _build_mtr_per_column_ensemble(
            _tt_e=_tt_e,
            _orig_tname=_orig_tname,
            models=models,
            metadata=metadata,
            target_by_type=target_by_type,
            oof_weights=_oof_weights,
        )
        return

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
        _components.append(PrePipelinePredictShim(_inner, _pp, _name))
        _component_names.append(_name)
    # Inject lag_predict dummy baseline as a free component for the cross-target ensemble pool. On strongly auto-regressive targets (lag1_corr ~0.999 within groups) the dumbest ``y_hat = lag_target_value`` baseline often beats every trained model on RMSE; honest-OOF gate naturally selects it when it dominates. NO trainable parameters; cost is one column read.
    try:
        _dbl_for_target = metadata.get("dummy_baselines", {}).get(str(_tt_e), {}).get(str(_orig_tname), {})
        _dbl_extras = _dbl_for_target.get("extras", {})
        _lag_meta = (
            _dbl_extras.get("lag_predict")
            if isinstance(
                _dbl_extras,
                dict,
            )
            else None
        )
        if _lag_meta is not None:
            _lag_col = _lag_meta.get("feature_used")
            if _lag_col:
                _lag_model = _LagPredictDeployableModel(_lag_col)
                _components.append(PrePipelinePredictShim(_lag_model, None, "lag_predict"))
                _component_names.append("lag_predict")
                logger.info(
                    "[CompositeCrossTargetEnsemble] target='%s' "
                    "injected lag_predict (feature=%s) as a free "
                    "ensemble component. honest-OOF gate will "
                    "auto-select if it dominates trained models.",
                    _orig_tname, _lag_col,
                )
    except Exception as _lag_inj_err:
        logger.debug(
            "[CompositeCrossTargetEnsemble] lag_predict injection " "failed for target='%s' (non-fatal): %s",
            _orig_tname,
            _lag_inj_err,
        )
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
    if len(_components) < 2:
        logger.info(
            "[CompositeCrossTargetEnsemble] target='%s': only %d " "component(s); ensemble skipped.",
            _orig_tname,
            len(_components),
        )
        return
    # Score components on the train slice in y-scale (same rows wrappers were fitted on).
    _y_full_for_rmse = target_by_type.get(_tt_e, {}).get(_orig_tname)

    def _compute_train_rmse_proxy() -> np.ndarray:
        """Full-train predict per component -> y-scale RMSE proxy.

        A component whose train-predict fails keeps a NaN here so the caller
        drops it from the pool entirely (zero ensemble weight). Imputing the
        failed row with the median of the survivors would instead grant the
        broken component mid-pack weight on a fabricated score.
        """
        _rmses: list[float] = []
        if _y_full_for_rmse is not None:
            _y_train_for_rmse = np.asarray(_y_full_for_rmse)[filtered_train_idx]
            _frame_key = (id(filtered_train_df), getattr(filtered_train_df, "shape", None))
            for _comp, _name in zip(_components, _component_names):
                try:
                    _pred = _get_train_pred(_comp, _frame_key)
                    _diff = _pred - _y_train_for_rmse.astype(np.float64)
                    _rmses.append(float(np.sqrt(np.mean(_diff * _diff))))
                except Exception as _rmse_err:
                    logger.warning(
                        "[CompositeCrossTargetEnsemble] could not score "
                        "component '%s' on train: %s. Dropping it from "
                        "ensemble weighting (zero weight, not median).",
                        _name, _rmse_err,
                    )
                    _rmses.append(float("nan"))
        else:
            _rmses = [float("nan")] * len(_components)
        return np.asarray(_rmses, dtype=np.float64)

    def _drop_unscored_from_pool(_rmses: np.ndarray):
        """Filter the component pool to the rows with a finite train-RMSE.

        A component that failed to predict on train (NaN proxy) is excluded so
        it never reaches ``from_train_metrics`` (which raises on non-finite
        rmses) and never earns ensemble weight on a fabricated score. Returns
        the (components, names, rmses) triple restricted to the finite subset;
        all-finite input passes through unchanged.
        """
        _fin = np.isfinite(_rmses)
        if _fin.all():
            return list(_components), list(_component_names), _rmses
        _keep = [int(_i) for _i in np.flatnonzero(_fin)]
        return (
            [_components[_i] for _i in _keep],
            [_component_names[_i] for _i in _keep],
            _rmses[_keep],
        )

    # If oof_holdout_frac > 0, the honest holdout REPLACES the train-RMSE proxy
    # (re-fit on 1-frac, predict on frac), so computing the proxy first is a
    # wasted full-train predict per component on the default honest-OOF path.
    # Defer it; compute only as the fallback when the OOF produces no matrix.
    _oof_frac = float(getattr(
        composite_target_discovery_config, "oof_holdout_frac", 0.0,
    ))
    _defer_train_proxy = _oof_frac > 0.0 and _y_full_for_rmse is not None
    _oof_components = _components
    _oof_names = _component_names
    if _defer_train_proxy:
        _rmse_arr = np.ones(len(_components), dtype=np.float64)
    else:
        _rmse_arr = _compute_train_rmse_proxy()
        if not np.isfinite(_rmse_arr).any():
            logger.warning(
                "[CompositeCrossTargetEnsemble] target='%s': no " "component scored on train; ensemble skipped.",
                _orig_tname,
            )
            return
        # Drop any component whose train-predict failed (NaN proxy) so it gets
        # zero weight rather than a median-imputed mid-pack score.
        _oof_components, _oof_names, _rmse_arr = _drop_unscored_from_pool(_rmse_arr)
        if len(_oof_components) < 2:
            logger.info(
                "[CompositeCrossTargetEnsemble] target='%s': only %d " "component(s) scored on train after dropping failed " "predicts; ensemble skipped.",
                _orig_tname,
                len(_oof_components),
            )
            return
    _oof_y_full = _y_full_for_rmse
    _oof_pred_matrix = None
    _oof_y_holdout = None
    _oof_rmses = _rmse_arr  # train-RMSE proxy by default
    if _oof_frac > 0.0 and _oof_y_full is not None:
        # Per-spec base matrix on filtered_train_df rows for transform.forward inside the OOF helper. Multi-base specs (linear_residual_multi from forward-stepwise auto-promotion) need the FULL (n, 1+K) matrix whose column count matches the fitted alphas.
        _base_full_per_spec: dict[str, np.ndarray] = {}
        _base_val_per_spec: dict[str, np.ndarray] = {}
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
                for _eb_oof in _extra_for_oof:
                    _b_cols.append(
                        _build_full_column_from_splits(
                            _eb_oof,
                            train_df_pd, val_df_pd, test_df_pd,
                            train_idx, val_idx, test_idx,
                            n_total=len(_oof_y_full),
                        )
                    )
                _b_stack_full = np.column_stack(_b_cols)
                _b_filtered = _b_stack_full[filtered_train_idx]
                try:
                    _b_val = _b_stack_full[filtered_val_idx]
                except Exception:
                    _b_val = None
            else:
                _b_filtered = _b_primary[filtered_train_idx]
                try:
                    _b_val = _b_primary[filtered_val_idx]
                except Exception:
                    _b_val = None
            # Key by the UNIQUE spec name, not base_column. A multi-base
            # spec and a single-base spec sharing the same PRIMARY base column
            # otherwise collide (last writer wins), so the other spec's OOF
            # refit raises a base-width mismatch and is silently excluded.
            _base_full_per_spec[_spec_for_oof["name"]] = _b_filtered
            if _b_val is not None:
                _base_val_per_spec[_spec_for_oof["name"]] = _b_val
        # Build the spec-or-None list parallel to components.
        _component_specs: list[dict[str, Any] | None] = []
        for _name in _component_names:
            if _name.startswith("raw#"):
                _component_specs.append(None)
            else:
                _comp_name = _name.split("#", 1)[0]
                _matching = next(
                    (s for s in _spec_list if s["name"] == _comp_name),
                    None,
                )
                _component_specs.append(_matching)
        # Thread ctx.timestamps + per-target sample_weight + group_ids (full-data-indexed) so the honest OOF split becomes time-aware / weighted / group-aware. All three are subset by filtered_train_idx to the train rows the components were fitted on.
        _ctx_ts_full = getattr(ctx, "timestamps", None) if ctx is not None else None
        _time_ordering = None
        if _ctx_ts_full is not None:
            try:
                _time_ordering = np.asarray(_ctx_ts_full)[filtered_train_idx]
            except (TypeError, IndexError):
                _time_ordering = None
        _ctx_sw_dict = getattr(ctx, "sample_weights", None) if ctx is not None else None
        _sw_for_oof = None
        if isinstance(_ctx_sw_dict, dict) and _ctx_sw_dict:
            _sw_raw = _ctx_sw_dict.get(_orig_tname)
            if _sw_raw is not None:
                try:
                    _sw_for_oof = np.asarray(_sw_raw)[filtered_train_idx]
                except (TypeError, IndexError):
                    _sw_for_oof = None
        # Resolve the OOF holdout source. Default ``kfold`` computes a true train-K-fold OOF surface that never
        # reuses the early-stopping (val) split for weighting; ``external_val`` predicts on the suite's val frame
        # (cheaper but the early-stopped components saw that surface, so it biases weights optimistic -> WARN);
        # ``train_tail`` is the legacy trailing-slice carve.
        _oof_source = str(getattr(
            composite_target_discovery_config,
            "oof_holdout_source", "kfold",
        )).lower()
        _oof_kfold = int(getattr(
            composite_target_discovery_config, "oof_kfold", 5,
        ))
        _ext_X = None
        _ext_y = None
        _ext_base_per_spec = None
        _kfold_for_oof = 1
        if _oof_source == "kfold":
            # K-fold OOF is incompatible with time-aware semantics (past-only training rules out shuffled folds);
            # the helper would silently downgrade with a WARN, so drop the time signal here when going K-fold.
            _kfold_for_oof = max(2, _oof_kfold)
            _time_ordering = None
            logger.info(
                "[CompositeCrossTargetEnsemble] target='%s' honest-OOF source='kfold' (K=%d); "
                "stack weights + OOF gate computed on true train-K-fold OOF (no val reuse).",
                _orig_tname, _kfold_for_oof,
            )
        elif _oof_source == "external_val":
            try:
                _ext_y_arr = np.asarray(_oof_y_full)[filtered_val_idx]
            except (TypeError, IndexError):
                _ext_y_arr = None
            if filtered_val_df is not None and _ext_y_arr is not None and len(_ext_y_arr) > 0:
                _ext_X = filtered_val_df
                _ext_y = _ext_y_arr
                _ext_base_per_spec = _base_val_per_spec or None
                logger.warning(
                    "[CompositeCrossTargetEnsemble] target='%s' honest-OOF source='external_val' (n=%d): "
                    "stack weights + OOF gate are computed on the early-stopping (val) split, which the booster "
                    "components were tuned against -- this biases the weighting optimistic. Use "
                    "oof_holdout_source='kfold' for an honest weighting surface; external_val is a "
                    "representativeness cross-check only.",
                    _orig_tname, len(_ext_y_arr),
                )
            else:
                logger.info(
                    "[CompositeCrossTargetEnsemble] target='%s' " "external_val OOF requested but val unavailable; " "falling back to train_tail.",
                    _orig_tname,
                )
        _group_ids_for_oof = None
        _ctx_groups = getattr(ctx, "group_ids", None) if ctx is not None else None
        if _ctx_groups is not None:
            try:
                _group_ids_for_oof = np.asarray(_ctx_groups)[filtered_train_idx]
            except (TypeError, IndexError):
                _group_ids_for_oof = None

        # OOF pre-screen optimisation: the dummy-floor gate
        # at the BOTTOM of this function frequently drops 60-70% of
        # components (observed in prod: 14/21 dropped). Without it, all
        # 21 components are OOF-refit (~10 min/MLP, ~5 min/booster),
        # then 14 immediately discarded -- ~30-50 minutes of pure
        # waste per target. The pre-screen uses already-trained
        # models' predict() on the external_val frame (cheap; no
        # refit) to compute a LEAKY val_RMSE estimate -- leaks only
        # through early-stopping signal, not full training -- and
        # drops components whose leaky RMSE clears the dummy floor
        # with a generous safety margin. Final dummy-floor gate STILL
        # runs after OOF on the honest refit RMSE, so this is a
        # speed-up only; correctness contract unchanged.
        _PRESCREEN_SAFETY = 1.5  # leaky RMSE * 1.5 must still clear floor
        if _ext_X is not None and _ext_y is not None and len(_components) >= 4:
            try:
                _raw_dbl_pre = metadata.get("dummy_baselines", {}).get(str(_tt_e), {}).get(str(_orig_tname), {})
                _data_pre = _raw_dbl_pre.get("data", {}) if isinstance(_raw_dbl_pre, dict) else {}
                _strongest_pre = _raw_dbl_pre.get("strongest") if isinstance(_raw_dbl_pre, dict) else None
                _pm_pre = _raw_dbl_pre.get("primary_metric") if isinstance(_raw_dbl_pre, dict) else None
                _dummy_floor_for_prescreen = None
                # Assumes an RMSE-family regression primary_metric: the dummy's primary_metric value is compared directly against component RMSEs, so this floor is only unit-consistent while the regression primary is RMSE (currently the only option).
                if _strongest_pre and _pm_pre and _strongest_pre in _data_pre:
                    _v = _data_pre[_strongest_pre].get(_pm_pre)
                    if _v is not None and np.isfinite(float(_v)):
                        _dummy_floor_for_prescreen = float(_v)
                if _dummy_floor_for_prescreen is not None:
                    _keep_mask = []
                    _dropped_pre: list[str] = []
                    _ext_y_arr_np = np.asarray(_ext_y, dtype=np.float64)
                    for _comp, _name in zip(_components, _component_names):
                        try:
                            _p = np.asarray(
                                _comp.predict(_ext_X), dtype=np.float64,
                            ).reshape(-1)
                            _finite = np.isfinite(_p) & np.isfinite(_ext_y_arr_np)
                            if _finite.sum() < 10:
                                _keep_mask.append(True)
                                continue
                            _r = _p[_finite] - _ext_y_arr_np[_finite]
                            _leaky_rmse = float(np.sqrt(np.mean(_r * _r)))
                            if _leaky_rmse / _PRESCREEN_SAFETY > _dummy_floor_for_prescreen:
                                _keep_mask.append(False)
                                _dropped_pre.append(f"{_name}(leakyRMSE={_leaky_rmse:.4g})")
                            else:
                                _keep_mask.append(True)
                        except Exception:
                            _keep_mask.append(True)  # err on the safe side
                    if _dropped_pre and sum(_keep_mask) >= 2:
                        _kept = [i for i, k in enumerate(_keep_mask) if k]
                        logger.warning(
                            "[CompositeCrossTargetEnsemble] target='%s' "
                            "OOF pre-screen (leaky val-RMSE speed gate, runs only under "
                            "oof_holdout_source='external_val') dropped %d/%d component(s) "
                            "whose leaky val_RMSE / %.1f > dummy floor "
                            "%.4g. Dropped: %s. Saves ~%d minute(s) of "
                            "refit time.",
                            _orig_tname, len(_dropped_pre),
                            len(_components), _PRESCREEN_SAFETY,
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

        # Bound the OOF weight-estimation refits to a train subsample (group-aware: keep whole groups).
        # The NNLS / dummy-floor blend weights saturate far below millions of rows, so this turns the
        # K-fold x N-component refit from hours to minutes with ensemble RMSE within ~1e-4 of full-data
        # (bench_oof_subsample_speedup.py). Slices REDUCE rows -> bounded, never a 100GB copy. The
        # deployed per-target components are untouched; only the weighting surface is subsampled.
        _oof_train_X = filtered_train_df
        _oof_y_arr = np.asarray(_oof_y_full)[filtered_train_idx]
        _oof_base_per_spec = _base_full_per_spec
        _oof_groups_arg = _group_ids_for_oof
        _oof_rs = int(getattr(composite_target_discovery_config, "oof_random_state", _DEFAULT_OOF_RANDOM_STATE))
        _oof_cap = int(getattr(composite_target_discovery_config, "oof_max_train_rows", 0) or 0)
        _n_oof_rows = len(_oof_y_arr)
        _sub_pos = _oof_subsample_positions(_n_oof_rows, _oof_groups_arg, _oof_cap, _oof_rs)
        if _sub_pos is not None and _sub_pos.size < _n_oof_rows:
            _oof_train_X = _slice_frame_rows(filtered_train_df, _sub_pos)
            _oof_y_arr = _oof_y_arr[_sub_pos]
            _oof_base_per_spec = {k: np.asarray(v)[_sub_pos] for k, v in (_base_full_per_spec or {}).items()}
            _oof_groups_arg = None if _oof_groups_arg is None else np.asarray(_oof_groups_arg)[_sub_pos]
            # Reassign the ctx-derived kwargs IN PLACE (subsampled) so the call site keeps threading
            # ``sample_weight=_sw_for_oof`` / ``time_ordering=_time_ordering`` (pinned by the call-site
            # propagation test) while still honouring the subsample.
            _sw_for_oof = None if _sw_for_oof is None else np.asarray(_sw_for_oof)[_sub_pos]
            _time_ordering = None if _time_ordering is None else np.asarray(_time_ordering)[_sub_pos]
            logger.info(
                "[CompositeCrossTargetEnsemble] target='%s' OOF weight-estimation subsampled %d -> %d "
                "train rows (group-aware cap=%d); deployed components unaffected.",
                _orig_tname, _n_oof_rows, int(_sub_pos.size), _oof_cap,
            )
        try:
            _oof_pred_matrix, _oof_y_holdout, _surviving = compute_oof_holdout_predictions(
                component_models=_components,
                component_names=_component_names,
                component_specs=_component_specs,
                train_X=_oof_train_X,
                y_train_full=_oof_y_arr,
                base_train_full_per_spec=_oof_base_per_spec,
                holdout_frac=_oof_frac,
                random_state=_oof_rs,
                time_ordering=_time_ordering,
                kfold=_kfold_for_oof,
                sample_weight=_sw_for_oof,
                external_holdout_X=_ext_X,
                external_holdout_y=_ext_y,
                external_holdout_base_per_spec=_ext_base_per_spec,
                group_ids=_oof_groups_arg,
            )
        except Exception as _oof_err:
            logger.warning(
                "[CompositeCrossTargetEnsemble] OOF computation failed " "for target='%s': %s. Falling back to train-RMSE proxy.",
                _orig_tname,
                _oof_err,
            )
            _oof_pred_matrix, _oof_y_holdout, _surviving = (
                None, None, [],
            )
        if _oof_pred_matrix is not None and _oof_y_holdout is not None and _oof_pred_matrix.shape[1] > 0:
            # Re-align to the surviving set returned by the OOF helper.
            _surviving_set = set(_surviving)
            _oof_components = [c for c, n in zip(_components, _component_names) if n in _surviving_set]
            _oof_names = list(_surviving)
            # Vectorised per-column RMSE: mask non-finite to 0 in a sum-and-divide pass; all-non-finite columns land as NaN (one pass, K cols).
            _diff_mat = _oof_pred_matrix - _oof_y_holdout[:, None]
            _finite_mat = np.isfinite(_diff_mat)
            _n_fin = _finite_mat.sum(axis=0)
            _sq_sum = np.where(_finite_mat, _diff_mat * _diff_mat, 0.0).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                _oof_rmses = np.where(_n_fin > 0, np.sqrt(_sq_sum / np.maximum(_n_fin, 1)), np.nan)
            _oof_rmses = _oof_rmses.astype(np.float64, copy=False)
        elif _defer_train_proxy:
            # OOF produced no usable matrix and the train-RMSE proxy was
            # deferred -> compute it now as the fallback weighting surface.
            _rmse_arr = _compute_train_rmse_proxy()
            if not np.isfinite(_rmse_arr).any():
                logger.warning(
                    "[CompositeCrossTargetEnsemble] target='%s': honest OOF " "produced no matrix and no component scored on train; " "ensemble skipped.",
                    _orig_tname,
                )
                return
            # Drop any component whose train-predict failed (NaN proxy). No OOF
            # matrix exists on this fallback path, so the pool lists carry the
            # full set; restrict them to the scored subset for zero-weight.
            _oof_components, _oof_names, _rmse_arr = _drop_unscored_from_pool(_rmse_arr)
            if len(_oof_components) < 2:
                logger.info(
                    "[CompositeCrossTargetEnsemble] target='%s': only %d " "component(s) scored on train after dropping failed " "predicts; ensemble skipped.",
                    _orig_tname,
                    len(_oof_components),
                )
                return
            _oof_rmses = _rmse_arr
        if _oof_pred_matrix is not None and _oof_y_holdout is not None and _oof_pred_matrix.shape[1] > 0:
            logger.info(
                "[CompositeCrossTargetEnsemble] target='%s' using " "honest OOF holdout (frac=%.2f, n=%d) for ensemble " "weights / stacking.",
                _orig_tname,
                _oof_frac,
                len(_oof_y_holdout),
            )
            # Dummy-floor gate: drop any component whose honest-OOF RMSE exceeds the raw target's strongest-dummy RMSE by more than the configured tolerance. A trained model that loses to a parameter-free dummy on the honest holdout cannot improve the ensemble; keeping it dilutes NNLS weights and harms test performance.
            # The dummy's primary_metric value is compared directly against component OOF RMSEs, so the floor is unit-consistent only while the regression primary is RMSE (currently the only option).
            _dummy_floor_enabled = bool(getattr(
                composite_target_discovery_config,
                "ct_ensemble_dummy_floor_enabled", True,
            ))
            _dummy_floor_tol = float(getattr(
                composite_target_discovery_config,
                "ct_ensemble_dummy_floor_tolerance", 0.0,
            ))
            if (_dummy_floor_enabled
                    and _oof_pred_matrix is not None
                    and _oof_pred_matrix.shape[1] > 0
                    and len(_oof_rmses) > 0):
                _dummy_floor_rmse = None
                try:
                    _raw_dbl = metadata.get("dummy_baselines", {}).get(str(_tt_e), {}).get(str(_orig_tname), {})
                    _data = _raw_dbl.get("data", {}) if isinstance(_raw_dbl, dict) else {}
                    _strongest = _raw_dbl.get("strongest") if isinstance(_raw_dbl, dict) else None
                    _pm = _raw_dbl.get("primary_metric") if isinstance(_raw_dbl, dict) else None
                    if _strongest and _pm and _strongest in _data:
                        _v = _data[_strongest].get(_pm)
                        if _v is not None and np.isfinite(float(_v)):
                            _dummy_floor_rmse = float(_v) * (1.0 + _dummy_floor_tol)
                except (KeyError, TypeError, ValueError):
                    _dummy_floor_rmse = None
                if _dummy_floor_rmse is not None:
                    _keep_idx = [_i for _i in range(len(_oof_rmses)) if np.isfinite(_oof_rmses[_i]) and _oof_rmses[_i] <= _dummy_floor_rmse]
                    _dropped_idx = [_i for _i in range(len(_oof_rmses)) if _i not in set(_keep_idx)]
                    if _dropped_idx and len(_keep_idx) >= 1:
                        _dropped_names = [f"{_oof_names[_i]}(OOF={_oof_rmses[_i]:.4g})" for _i in _dropped_idx]
                        _floor_base = _dummy_floor_rmse / (1.0 + _dummy_floor_tol)
                        logger.warning(
                            "[CompositeCrossTargetEnsemble] target='%s' "
                            "dummy-floor gate fired: dropping %d/%d "
                            "component(s) whose OOF RMSE > strongest "
                            "dummy ('%s' %s=%.4g) x (1+%.2f) = %.4g. "
                            "Dropped: %s",
                            _orig_tname, len(_dropped_idx),
                            len(_oof_rmses), _strongest, _pm,
                            _floor_base, _dummy_floor_tol,
                            _dummy_floor_rmse, _dropped_names,
                        )
                        _oof_components = [_oof_components[_i] for _i in _keep_idx]
                        _oof_names = [_oof_names[_i] for _i in _keep_idx]
                        _oof_rmses = _oof_rmses[_keep_idx]
                        _oof_pred_matrix = _oof_pred_matrix[:, _keep_idx]
                    elif not _keep_idx:
                        logger.warning(
                            "[CompositeCrossTargetEnsemble] target='%s' "
                            "dummy-floor gate would drop ALL %d "
                            "component(s) (every OOF RMSE > %.4g); "
                            "keeping all to avoid empty pool. The "
                            "honest-OOF gate below will fall back to "
                            "best single.",
                            _orig_tname, len(_oof_rmses),
                            _dummy_floor_rmse,
                        )

        # Residual-correlation dedup (opt-in): drop near-duplicate members (|residual corr| > threshold), keeping
        # the lower-OOF-RMSE one, so a redundant pair can't split + dominate the NNLS weight. Runs on the honest
        # OOF residuals so redundancy is measured on the same surface the stacker fits.
        if (bool(getattr(composite_target_discovery_config, "ct_ensemble_dedup_enabled", False))
                and _oof_pred_matrix is not None
                and _oof_y_holdout is not None
                and _oof_pred_matrix.shape[1] > 2):
            try:
                from ...composite import residual_dedup_indices
                _dedup_thr = float(getattr(
                    composite_target_discovery_config,
                    "ct_ensemble_dedup_corr_threshold", 0.95,
                ))
                _resid = _oof_pred_matrix - _oof_y_holdout[:, None]
                _keep_dd, _drop_dd = residual_dedup_indices(
                    _resid, np.asarray(_oof_rmses, dtype=np.float64),
                    corr_threshold=_dedup_thr,
                )
                if _drop_dd:
                    logger.info(
                        "[CompositeCrossTargetEnsemble] target='%s' residual dedup dropped %d/%d near-duplicate "
                        "component(s) (|resid corr| > %.2f): %s.",
                        _orig_tname, len(_drop_dd), _oof_pred_matrix.shape[1], _dedup_thr,
                        [_oof_names[_i] for _i in _drop_dd],
                    )
                    _oof_components = [_oof_components[_i] for _i in _keep_dd]
                    _oof_names = [_oof_names[_i] for _i in _keep_dd]
                    _oof_rmses = _oof_rmses[_keep_dd]
                    _oof_pred_matrix = _oof_pred_matrix[:, _keep_dd]
            except Exception as _dedup_err:
                logger.warning(
                    "[CompositeCrossTargetEnsemble] residual dedup failed for target='%s': %s. "
                    "Proceeding with full set.", _orig_tname, _dedup_err,
                )

    try:
        if _ce_strategy == "mean":
            _ensemble = _CrossEns.from_uniform_weights(
                component_models=_oof_components,
                component_names=_oof_names,
            )
        elif _ce_strategy in ("linear_stack", "nnls_stack"):
            # Honest OOF preds if available, else biased train-set preds.
            if _oof_pred_matrix is not None and _oof_y_holdout is not None and _oof_pred_matrix.shape[1] > 0:
                _pred_matrix = _oof_pred_matrix
                _y_for_stack = _oof_y_holdout
                # Stacking-aware gate (opt-in) -- drop components whose NNLS weight on the honest OOF preds falls below the configured threshold BEFORE running the actual stacker.
                if (getattr(
                    composite_target_discovery_config,
                    "stacking_aware_gate_enabled", False,
                ) and _pred_matrix.shape[1] >= 2):
                    try:
                        from ...composite import stacking_aware_gate
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
                            _keep_mask_arr = np.array([n in set(_survivors) for n in _oof_names], dtype=bool)
                            _pred_matrix = _pred_matrix[:, _keep_mask_arr]
                            # Keep _oof_pred_matrix aligned with the pruned weights so the OOF gate + AR(1) failsafe below still match column-for-column (mirrors the dedup block).
                            _oof_pred_matrix = _oof_pred_matrix[:, _keep_mask_arr]
                            _oof_components = [c for c, k in zip(_oof_components, _keep_mask_arr) if k]
                            _oof_names = [n for n, k in zip(_oof_names, _keep_mask_arr) if k]
                            _oof_rmses = _oof_rmses[_keep_mask_arr]
                            logger.info(
                                "[CompositeCrossTargetEnsemble] target='%s' " "stacking_aware_gate kept %d of %d components " "(min_weight=%.3f).",
                                _orig_tname,
                                len(_survivors),
                                len(_gate_w),
                                _gate_min,
                            )
                    except Exception as _gate_err:
                        logger.warning(
                            "[CompositeCrossTargetEnsemble] stacking_aware_gate " "failed for target='%s': %s. Proceeding with full set.",
                            _orig_tname,
                            _gate_err,
                        )
            else:
                if _oof_y_full is None:
                    raise RuntimeError("stacking requires train target alignment")
                _y_for_stack = np.asarray(_oof_y_full)[filtered_train_idx]
                # Preallocate (n_rows, K) to skip np.column_stack's per-entry copy doubling peak RAM.
                _frame_key2 = (id(filtered_train_df), getattr(filtered_train_df, "shape", None))
                _n_rows = len(_y_for_stack)
                _pred_matrix = np.empty((_n_rows, len(_oof_components)), dtype=np.float64)
                for _ci, (_comp, _name) in enumerate(zip(_oof_components, _oof_names)):
                    _pred_matrix[:, _ci] = _get_train_pred(_comp, _frame_key2)
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
            # Pipe OOF rmses through component_oof_rmse= so from_train_metrics ranks on the honest holdout signal,
            # and pass the strongest-dummy (lag_predict / naive) OOF RMSE as baseline_oof_rmse so weights are
            # gain-over-naive rather than gain-over-the-worst-component (the class's self-normalising fallback,
            # which discards every below-median component and dilutes against a meaningless baseline).
            _baseline_oof_rmse = None
            try:
                _raw_dbl_base = metadata.get("dummy_baselines", {}).get(str(_tt_e), {}).get(str(_orig_tname), {})
                if isinstance(_raw_dbl_base, dict):
                    _data_base = _raw_dbl_base.get("data", {}) or {}
                    _strongest_base = _raw_dbl_base.get("strongest")
                    _pm_base = _raw_dbl_base.get("primary_metric")
                    if _strongest_base and _pm_base and _strongest_base in _data_base:
                        _v_base = _data_base[_strongest_base].get(_pm_base)
                        if _v_base is not None and np.isfinite(float(_v_base)):
                            _baseline_oof_rmse = float(_v_base)
                # Prefer the in-pool lag_predict OOF RMSE when present: it is the honest, same-split naive floor.
                if "lag_predict" in _oof_names:
                    _lp_b = float(_oof_rmses[_oof_names.index("lag_predict")])
                    if np.isfinite(_lp_b):
                        _baseline_oof_rmse = _lp_b if _baseline_oof_rmse is None else max(_baseline_oof_rmse, _lp_b)
            except (KeyError, TypeError, ValueError):
                _baseline_oof_rmse = None
            _ensemble = _CrossEns.from_train_metrics(
                component_models=_oof_components,
                component_names=_oof_names,
                component_oof_rmse=_oof_rmses.tolist(),
                baseline_oof_rmse=_baseline_oof_rmse,
            )
        # OOF validation gate: fall back to best single if ensemble holdout RMSE > best-single holdout RMSE.
        if _oof_pred_matrix is not None and _oof_pred_matrix.shape[1] > 0 and isinstance(_ensemble, _CrossEns):
            try:
                # Gate==deploy: combine the OOF-holdout component preds with the SAME rule predict() uses, so
                # the gate scores the exact predictor that ships. Non-convex stacks (linear_stack / nnls_stack)
                # use raw solver weights with NO renormalisation (linear_stack also adds the intercept); convex
                # strategies (mean / oof_weighted) renormalise across the surviving columns. Branching on
                # _ensemble.is_convex (not on _ce_strategy) mirrors predict() exactly, including the case where
                # a stacker degenerated and fell back to a convex mean inside the ensemble class.
                _w_full = np.asarray(_ensemble.weights, dtype=np.float64)
                if not getattr(_ensemble, "is_convex", True):
                    _intercept = float(
                        getattr(
                            _ensemble,
                            "_linear_stack_intercept",
                            0.0,
                        )
                    )
                    _ens_holdout = (_oof_pred_matrix * _w_full[None, :]).sum(axis=1) + _intercept
                else:
                    _w_sum = float(_w_full.sum())
                    _w_norm = _w_full / _w_sum if _w_sum > 0 else np.full_like(_w_full, 1.0 / len(_w_full))
                    _ens_holdout = (_oof_pred_matrix * _w_norm[None, :]).sum(axis=1)
                _ens_diff = _ens_holdout - _oof_y_holdout
                _ens_rmse = float(np.sqrt(np.mean(_ens_diff**2)))
                _best_single_rmse = float(np.nanmin(_oof_rmses))
                # AR(1) failsafe: when lag_predict's OOF RMSE ties the best trained component, prefer zero-param lag. But
                # the OOF RMSE is a group-K-fold estimate that UNDERESTIMATES the full-data model (each fold trains on
                # fewer groups), so a tie can ship lag over a trained model that generalises far better (prod: lag test
                # 12.29 vs trained 9.31). Cross-check on the group-disjoint VAL split (same honest regime as test): if a
                # trained component beats lag on VAL by > tolerance the OOF tie is spurious -> veto, deploy the trained.
                _lag_failsafe_tol = float(getattr(
                    composite_target_discovery_config,
                    "lag_predict_failsafe_tolerance", 0.10,
                ))
                _lag_failsafe_taken = False
                from .._ar1_failsafe_veto import compute_val_veto
                _val_veto_idx = compute_val_veto(
                    _oof_names, _oof_rmses, _oof_components, filtered_val_df,
                    filtered_val_idx, _oof_y_full, _lag_failsafe_tol,
                    composite_target_discovery_config,
                )
                if _val_veto_idx is not None:
                    logger.warning(
                        "[CompositeCrossTargetEnsemble] target='%s' AR1 failsafe VETOED by val cross-check: trained "
                        "'%s' beats lag_predict on the group-disjoint val split by >%.0f%% (the OOF tie was group-K-fold "
                        "pessimism); deploying the trained component, NOT lag.",
                        _orig_tname, _oof_names[_val_veto_idx], _lag_failsafe_tol * 100.0,
                    )
                    _deployed = _oof_components[_val_veto_idx]
                    # Per-row OOD-lag routing on top: the trained model wins overall but still extrapolates on unseen
                    # groups whose target level is out of the train range; route those rows (lag out of range) to lag,
                    # but only when it improves the honest val RMSE. Transferable (train-range rule, not group-id).
                    try:
                        from .._ood_lag_router import build_ood_lag_router
                        _ytr = (np.asarray(_oof_y_full)[filtered_train_idx].astype(np.float64)
                                if (_oof_y_full is not None and filtered_train_idx is not None) else None)
                        _yv = (np.asarray(_oof_y_full)[filtered_val_idx].astype(np.float64)
                               if (_oof_y_full is not None and filtered_val_idx is not None) else None)
                        _deployed = build_ood_lag_router(
                            _deployed, _oof_components[_oof_names.index("lag_predict")],
                            _ytr, filtered_val_df, _yv, composite_target_discovery_config,
                        )
                    except Exception as _rr_err:
                        logger.info("[CompositeCrossTargetEnsemble] target='%s' OOD-lag routing skipped (%s).", _orig_tname, _rr_err)
                    # Per-row VOLATILITY-lag routing: on a strong-AR target the lag-wins groups are IN-range but locally
                    # SMOOTH, which the range rule above cannot catch. Route rows whose MD-local target volatility is low
                    # (lag near-perfect) to lag, only when it improves the honest val RMSE. Needs group_column + a MD
                    # order column (time_column) on the frame -- ordering is explicit, never a frame-row-order guess.
                    try:
                        from .._volatility_lag_router import build_volatility_lag_router
                        _yv2 = (np.asarray(_oof_y_full)[filtered_val_idx].astype(np.float64)
                                if (_oof_y_full is not None and filtered_val_idx is not None) else None)
                        _ctx_g2 = getattr(ctx, "group_ids", None) if ctx is not None else None
                        _gids_val2 = np.asarray(_ctx_g2)[filtered_val_idx] if (_ctx_g2 is not None and filtered_val_idx is not None) else None
                        _deployed = build_volatility_lag_router(
                            _deployed, _oof_components[_oof_names.index("lag_predict")],
                            _gids_val2, filtered_val_df, _yv2,
                            getattr(composite_target_discovery_config, "group_column", None),
                            getattr(composite_target_discovery_config, "time_column", None),
                            composite_target_discovery_config,
                        )
                    except Exception as _vr_err:
                        logger.info("[CompositeCrossTargetEnsemble] target='%s' volatility-lag routing skipped (%s).", _orig_tname, _vr_err)
                    _ensemble = _deployed
                    _lag_failsafe_taken = True
                elif _lag_failsafe_tol > 0 and "lag_predict" in _oof_names and np.isfinite(_best_single_rmse):
                    _lp_idx = _oof_names.index("lag_predict")
                    _lp_rmse = float(_oof_rmses[_lp_idx])
                    if np.isfinite(_lp_rmse) and _lp_rmse <= (1.0 + _lag_failsafe_tol) * _best_single_rmse:
                        logger.warning(
                            "[CompositeCrossTargetEnsemble] target='%s' "
                            "AR1 failsafe fired: lag_predict OOF "
                            "RMSE %.4g within +%.0f%% of best single "
                            "%.4g; preferring zero-parameter "
                            "lag_predict over %d-component stack "
                            "(cannot overfit on test).",
                            _orig_tname, _lp_rmse,
                            _lag_failsafe_tol * 100.0,
                            _best_single_rmse, len(_oof_names),
                        )
                        _ensemble = _oof_components[_lp_idx]
                        _lag_failsafe_taken = True
                if not _lag_failsafe_taken and _ens_rmse > _best_single_rmse:
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
                    "[CompositeCrossTargetEnsemble] OOF gate check " "skipped (%s); ensemble retained.",
                    _gate_err,
                )
        # Opt-in post-hoc output recalibration. Fit a monotone map on the SAME OOF holdout surface
        # the weights were derived from (blend the OOF component matrix with the frozen weights, then
        # map that OOF blend onto the OOF truth). Leakage-free (OOF only) and bit-identical when off.
        # Skip when the gate fell back to a single component (no ensemble blend to recalibrate).
        _do_calib = bool(getattr(
            composite_target_discovery_config,
            "calibrate_cross_target_output", False,
        ))
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
                    method=_calib_method,
                )
                logger.info(
                    "[CompositeCrossTargetEnsemble] target='%s' fitted output calibrator " "(method=%s, attached=%s) on %d OOF rows.",
                    _orig_tname,
                    _calib_method,
                    getattr(_ensemble, "_output_calibrator", None) is not None,
                    int(_oof_pred_matrix.shape[0]),
                )
            except Exception as _calib_err:
                logger.warning(
                    "[CompositeCrossTargetEnsemble] target='%s' output calibration failed "
                    "(%s); ensemble retained uncalibrated.", _orig_tname, _calib_err,
                )
    except Exception as _ens_err:
        logger.warning(
            "[CompositeCrossTargetEnsemble] target='%s' build failed: "
            "%s. Skipping.", _orig_tname, _ens_err,
        )
        return
    # Optional top-N cap by weight for latency-bounded serving (0/None preserves full ensemble).
    _max_components = getattr(
        composite_target_discovery_config,
        "max_inference_components", None,
    )
    if _max_components is not None and _max_components > 0 and isinstance(_ensemble, _CrossEns):
        _ensemble = _ensemble.cap_inference_components(int(_max_components))
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
    metadata.setdefault("composite_target_ensemble", {}).setdefault(str(_tt_e), {})[_orig_tname] = (
        _ensemble.export_metadata() if hasattr(_ensemble, "export_metadata") else {"strategy": "single_best_fallback"}
    )
    # Stamp the chosen ensemble flavour into metadata["ensembles_chosen"] so the predict path can replay the right combine for the cross-target slot (predict-path parity).
    _ce_actual_strategy = getattr(_ensemble, "strategy", None) or _ce_strategy
    # Sub-key per ensemble family: cross-target ensembles live under ``ensembles_chosen["cross_target"]``; simple per-target ensembles are stamped by _phase_train_one_target under ``ensembles_chosen["simple"]``.
    metadata.setdefault("ensembles_chosen", {}).setdefault("cross_target", {}).setdefault(str(_tt_e), {})[_ens_key] = str(_ce_actual_strategy)
    logger.info(
        "[CompositeCrossTargetEnsemble] target='%s' built strategy='%s' "
        "over %d component(s); stored at models[%s][%s].",
        _orig_tname, _ce_strategy, len(_components),
        _tt_e, _ens_key,
    )

    # Route the ensemble through report_model_perf so val/test get the same scatter + residual charts as real models.
    try:
        from ...evaluation import report_model_perf
        _ens_orig_y = target_by_type.get(_tt_e, {}).get(_orig_tname)
        if _ens_orig_y is not None:
            _ens_y_arr = np.asarray(_ens_orig_y)
            _ens_model_name = f"CT_ENSEMBLE[{_ce_strategy}] {target_name} " f"{model_name} {_orig_tname}"
            # ``or []`` on a pandas Index raises ``The truth value of a Index is ambiguous``; explicit None+len check keeps both pandas Index and plain lists safe.
            _cols_attr = getattr(filtered_train_df, "columns", None)
            _ens_columns = list(_cols_attr) if _cols_attr is not None and len(_cols_attr) > 0 else []
            # Compute train envelope once so val + test ensemble reports
            # share the same TRAIN-bound prediction clip (k=3 sigma)
            # rather than each fallback-deriving a different eval bound.
            _ens_train_envelope = None
            try:
                _ens_y_train = _ens_y_arr[filtered_train_idx] if filtered_train_idx is not None else None
                if _ens_y_train is not None and len(_ens_y_train) > 0:
                    from ..._prediction_envelope_clip import compute_train_envelope_stats
                    _ens_train_envelope = compute_train_envelope_stats(_ens_y_train)
            except Exception:
                _ens_train_envelope = None
            _ens_common: dict[str, Any] = dict(
                columns=_ens_columns,
                df=None, model=None,
                model_name=_ens_model_name,
                plot_outputs=getattr(reporting_config, "plot_outputs", None),
                plot_dpi=getattr(reporting_config, "plot_dpi", None),
                show_fi=False,
                target_type=str(_tt_e),
                y_train_envelope_stats=_ens_train_envelope,
            )
            _emit_val_ens = bool(getattr(reporting_config, "compute_valset_metrics", True))
            _emit_test_ens = bool(getattr(reporting_config, "compute_testset_metrics", True))
            _split_plan = []
            if _emit_val_ens:
                _split_plan.append(("val", "VAL (CT_ENSEMBLE) ", filtered_val_idx, filtered_val_df))
            if _emit_test_ens:
                _split_plan.append(("test", "TEST (CT_ENSEMBLE) ", test_idx, test_df_pd))
            for _split_name, _report_title, _split_idx, _split_df in _split_plan:
                if _split_idx is None or _split_df is None:
                    continue
                try:
                    _y_split = _ens_y_arr[_split_idx]
                    _ens_preds = np.asarray(
                        _ensemble.predict(_split_df),
                        dtype=np.float64,
                    ).reshape(-1)
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
                        metadata.setdefault("cross_target_ensemble_metrics", {}).setdefault(str(_tt_e), {}).setdefault(_orig_tname, {}).update(
                            _ens_scalar_metrics
                        )
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


from ._post_xt_ensemble_mtr import MTRPerColumnEqualMeanEnsemble
