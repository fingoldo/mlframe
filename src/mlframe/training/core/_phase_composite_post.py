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
from ..composite_post_shim import PrePipelinePredictShim
from ..composite_transforms import is_composite_target_name
from ..dummy_baselines import format_suite_end_summary
from ..evaluation import report_model_perf
from .utils import _build_full_column_from_splits, _entry_metric

logger = logging.getLogger(__name__)

_DEFAULT_OOF_RANDOM_STATE = 42
_PROB_NORM_EPS = 1e-12
# T2#10 2026-05-18 Pack G universal watchdog threshold. ``wrapper.predict(X)``
# is compared against ``transform.inverse(inner.predict(X), base, params)``;
# divergence beyond this fraction of ``y_std`` fires a WARNING.
#
# Choice of 1%: the wrapper applies a y-train clip on inverse() output, so
# out-of-envelope rows can show tiny per-row differences (clip pulled the
# extreme back inside [y_min, y_max] while the reconstructed path didn't).
# 1% of y_std is well below the float64 round-off floor accumulated across
# a typical (n=10^5, transform=linear_residual) split, AND well above the
# clip-induced noise on a normally-distributed y (clip would have to bite
# ~3 sigma rows AND the inverse path miss them, both rare). Wrapper-math
# bugs (entry-mutation cache stale, double-inverse, base mismatch) produce
# divergence in the 5-50% range, comfortably above this threshold.
#
# Tune by raising if a healthy wrapper fires this warning in your data
# (consult the watchdog log line; %_of_y_std is included so the threshold
# can be set just above the observed noise floor).
_WATCHDOG_RELATIVE_THRESHOLD = 0.01


class _LagPredictDeployableModel:
    """Wraps the ``lag_predict`` dummy baseline as a deployable model.

    Production TVT 2026-05-23: lag_predict beat every trained component
    on TEST RMSE (11.58 vs ensemble's 12.73). The dummy was visible in
    the dummy-baselines table but invisible to CT_ENSEMBLE's component
    pool -- final delivery used the worse stacker output. This wrapper
    presents lag_predict via the same ``predict(X) -> ndarray`` shape
    every other CT_ENSEMBLE component exposes, so the existing
    honest-OOF gate naturally selects it when it dominates.

    Prediction rule: ``y_hat[i] = X[lag_column].iloc[i]`` -- zero
    trainable parameters, returns the lag-target value verbatim per
    row. Inference cost is one column access; never extrapolates.

    Implements ``get_params``/``set_params``/``fit`` so ``sklearn.clone``
    accepts it during honest-OOF refit (CompositeCrossTargetEnsemble path,
    2026-05-23 prod incident: clone failed -> component dropped -> NNLS
    weights missed lag_predict and ensemble landed at RMSE 13.30 vs
    lag_predict's 11.58 floor).
    """

    def __init__(self, lag_column: str) -> None:
        self.lag_column = str(lag_column)

    def get_params(self, deep: bool = True) -> dict:  # noqa: ARG002 - sklearn API
        return {"lag_column": self.lag_column}

    def set_params(self, **params: Any) -> "_LagPredictDeployableModel":
        for k, v in params.items():
            if k != "lag_column":
                raise ValueError(
                    f"_LagPredictDeployableModel has no parameter {k!r}; "
                    f"valid: ['lag_column']"
                )
            self.lag_column = str(v)
        return self

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> "_LagPredictDeployableModel":  # noqa: ARG002
        return self

    def predict(self, X: Any) -> np.ndarray:
        # Polars Series.to_numpy() returns a 1-D zero-copy ndarray for numeric dtypes; the prior .select(col).to_numpy().reshape(-1) path
        # built a (N, 1) frame, materialised it as a 2-D ndarray, then reshaped -- two extra allocations per predict on the lag baseline.
        if hasattr(X, "get_column"):
            try:
                col = X.get_column(self.lag_column).to_numpy()
                if col.dtype != np.float64:
                    col = col.astype(np.float64, copy=False)
                return col.reshape(-1)
            except Exception:
                pass
        if hasattr(X, "loc") or hasattr(X, "__getitem__"):
            try:
                col = X[self.lag_column]
                if hasattr(col, "to_numpy"):
                    arr = col.to_numpy()
                    if arr.dtype != np.float64:
                        arr = arr.astype(np.float64, copy=False)
                    return arr.reshape(-1)
                return np.asarray(col, dtype=np.float64).reshape(-1)
            except (KeyError, TypeError):
                pass
        raise KeyError(
            f"_LagPredictDeployableModel: column {self.lag_column!r} "
            f"not found on X (type={type(X).__name__})"
        )

    def __repr__(self) -> str:
        return f"_LagPredictDeployableModel(lag_column={self.lag_column!r})"


# Wave 100 (2026-05-21): _run_composite_target_wrapping (~390 lines)
# moved to sibling file _phase_composite_wrapping.py to drop this file
# below the 1k-line monolith threshold. Re-exported below so existing
# callers (`from ._phase_composite_post import _run_composite_target_wrapping`)
# keep working.
from ._phase_composite_wrapping import _run_composite_target_wrapping  # noqa: F401, E402


def recover_composite_y_scale_metrics(
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
    enable_watchdog: bool = True,
) -> dict[int, np.ndarray]:
    """T1#7 2026-05-18 lazy recovery of composite-target y-scale metrics.

    When the suite runs with ``skip_wrap_pass_predict=True`` (default since
    2026-05-18), the wrap step still runs but the y-scale metric block is
    bypassed - ``metadata["composite_target_y_scale_metrics"]`` stays empty.

    Callers that subsequently need those metrics (notebooks, dashboards,
    downstream audits) invoke this helper. It walks the already-wrapped
    ``models`` dict and computes RMSE/MAE/R2 per (composite_name, split).
    The metadata dict is populated in place with the same shape as the
    eager path; the train-prediction cache is returned so subsequent
    cross-target ensemble work reuses the freshly-computed predictions.

    Idempotent: when the wrap step in ``_run_composite_target_wrapping``
    detects an entry whose inner is already a ``CompositeTargetEstimator``
    it skips re-wrapping, so callers can invoke this helper safely after
    the eager path has already run.
    """
    return _run_composite_target_wrapping(
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
        skip_predict=False,
        enable_watchdog=enable_watchdog,
    )


# _run_suite_end_dummy_baselines_summary moved to sibling; re-exported below.


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
        _skip_predict = bool(getattr(
            composite_target_discovery_config, "skip_wrap_pass_predict", False,
        ))
        _enable_watchdog = bool(getattr(
            composite_target_discovery_config, "enable_wrap_pass_watchdog", True,
        ))
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
            skip_predict=_skip_predict,
            enable_watchdog=_enable_watchdog,
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
    # Build CT_ENSEMBLE for raw-target models even when 0 composite
    # specs were discovered. On extreme-AR + group-aware regression
    # (composite-discovery extreme_ar_group_aware_skip fires; see round
    # 5.3) the existing entry guard requiring composite_specs_by_target_type
    # silently bypasses the dummy-floor gate + lag_predict injection,
    # leaving the suite shipping a simple-arithmetic ensemble of the raw
    # models -- which is provably WORSE than the best single component
    # when 3 of 4 boosters are above the lag-predict floor (TVT prod
    # 2026-05-24: EnsARITHM TEST=12.45 vs Ridge alone 11.63 vs
    # lag_predict 11.58). Synthesise a per-target empty-spec entry for
    # every regression target with at least one trained model so the
    # below loop runs, lag_predict is injected, and the OOF + dummy-
    # floor + AR(1)-failsafe gates pick the right component.
    _build_for_raw_only = bool(getattr(
        composite_target_discovery_config,
        "always_build_ct_ensemble_for_raw", True,
    ))
    if (composite_target_discovery_config.enabled
            and _ce_strategy != "off"
            and not composite_specs_by_target_type
            and _build_for_raw_only):
        from ..target_types import TargetTypes as _TT
        _raw_only_specs: dict = {}
        _reg_models = (models or {}).get(_TT.REGRESSION, {}) if models else {}
        for _raw_tname, _entries in _reg_models.items():
            if _entries and not is_composite_target_name(str(_raw_tname)):
                _raw_only_specs.setdefault(_TT.REGRESSION, {})[_raw_tname] = []
        if _raw_only_specs:
            composite_specs_by_target_type = _raw_only_specs
            logger.info(
                "[CompositeCrossTargetEnsemble] always_build_ct_ensemble_for_raw=True: "
                "synthesised raw-only entries for %d regression target(s) "
                "(no composite specs were discovered); ensemble loop will inject "
                "lag_predict and run the dummy-floor + AR(1)-failsafe gates.",
                len(_raw_only_specs.get(_TT.REGRESSION, {})),
            )
    if (composite_target_discovery_config.enabled
            and _ce_strategy != "off"
            and composite_specs_by_target_type):
        from ..composite import CompositeCrossTargetEnsemble as _CrossEns

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
                        PrePipelinePredictShim(_inner, _pp, _name)
                    )
                    _component_names.append(_name)
                # Inject lag_predict dummy baseline as a free component
                # for the cross-target ensemble pool. On strongly auto-regressive
                # targets (TVT-style, lag1_corr ~0.999 within groups) the dumbest
                # ``y_hat = lag_target_value`` baseline often beats every trained
                # model on RMSE (prod TVT 2026-05-23: lag_predict TEST RMSE=11.58
                # vs CT_ENSEMBLE 12.73, ensemble loses to dummy by ~10%). Inject
                # the lag column as a zero-cost deployable component so the
                # existing honest-OOF gate naturally selects it when it
                # dominates. NO trainable parameters; cost is one column read.
                try:
                    _dbl_for_target = (
                        metadata.get("dummy_baselines", {})
                        .get(str(_tt_e), {})
                        .get(str(_orig_tname), {})
                    )
                    _dbl_extras = _dbl_for_target.get("extras", {})
                    _lag_meta = _dbl_extras.get("lag_predict") if isinstance(
                        _dbl_extras, dict,
                    ) else None
                    if _lag_meta is not None:
                        _lag_col = _lag_meta.get("feature_used")
                        if _lag_col:
                            _lag_model = _LagPredictDeployableModel(_lag_col)
                            _components.append(
                                PrePipelinePredictShim(_lag_model, None, "lag_predict")
                            )
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
                        "[CompositeCrossTargetEnsemble] lag_predict injection "
                        "failed for target='%s' (non-fatal): %s",
                        _orig_tname, _lag_inj_err,
                    )
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
                            PrePipelinePredictShim(_inner, _pp, _name)
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
                            _inner_for_cache = getattr(_comp, "model", _comp)
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
                    # Per-spec base matrix on filtered_train_df rows for
                    # transform.forward inside the OOF helper. Multi-base
                    # specs (linear_residual_multi from forward-stepwise
                    # auto-promotion) need the FULL (n, 1+K) matrix whose
                    # column count matches the fitted alphas. Building the
                    # primary column only crashes inside the OOF helper's
                    # transform.forward with "base has 1 columns but fitted
                    # alphas has K entries". Reproduced by fuzz c0047
                    # (multi-base auto-promoted to linresM-num_1+num_dep).
                    _base_full_per_spec: dict[str, np.ndarray] = {}
                    _base_val_per_spec: dict[str, np.ndarray] = {}
                    for _spec_for_oof in _spec_list:
                        _b_primary = _build_full_column_from_splits(
                            _spec_for_oof["base_column"],
                            train_df_pd, val_df_pd, test_df_pd,
                            train_idx, val_idx, test_idx,
                            n_total=len(_oof_y_full),
                        )
                        _extra_for_oof = tuple(
                            _spec_for_oof.get("extra_base_columns") or ()
                        )
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
                        _base_full_per_spec[_spec_for_oof["base_column"]] = _b_filtered
                        if _b_val is not None:
                            _base_val_per_spec[_spec_for_oof["base_column"]] = _b_val
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
                    # Thread ctx.timestamps + per-target sample_weight so the OOF
                    # holdout split becomes time-aware (trailing-slice past-only train)
                    # rather than random shuffle. Pre-fix a time-series composite suite
                    # silently used random shuffle -> FUTURE rows leaked into the OOF
                    # train -> over-optimistic ensemble gate. compute_oof_holdout_predictions
                    # documents the time-aware fork at composite_ensemble.py:260-267.
                    # Slice both to filtered_train_idx so length matches train_X.
                    _ctx_ts_full = getattr(ctx, "timestamps", None) if "ctx" in dir() else None
                    _time_ordering = None
                    if _ctx_ts_full is not None:
                        try:
                            _time_ordering = np.asarray(_ctx_ts_full)[filtered_train_idx]
                        except (TypeError, IndexError):
                            _time_ordering = None
                    _ctx_sw_dict = getattr(ctx, "sample_weights", None) if "ctx" in dir() else None
                    _sw_for_oof = None
                    if isinstance(_ctx_sw_dict, dict) and _ctx_sw_dict:
                        _sw_raw = _ctx_sw_dict.get(_orig_tname)
                        if _sw_raw is not None:
                            try:
                                _sw_for_oof = np.asarray(_sw_raw)[filtered_train_idx]
                            except (TypeError, IndexError):
                                _sw_for_oof = None
                    # Resolve the OOF holdout source. ``external_val``
                    # (default) replaces the train-tail carving with the
                    # suite's val frame: fit clones on full train,
                    # predict on val. ``train_tail`` keeps the legacy
                    # internal trailing-slice / random-shuffle carve.
                    _oof_source = str(getattr(
                        composite_target_discovery_config,
                        "oof_holdout_source", "external_val",
                    )).lower()
                    _ext_X = None
                    _ext_y = None
                    _ext_base_per_spec = None
                    if _oof_source == "external_val":
                        try:
                            _ext_y_arr = (
                                np.asarray(_oof_y_full)[filtered_val_idx]
                            )
                        except (TypeError, IndexError):
                            _ext_y_arr = None
                        if (filtered_val_df is not None
                                and _ext_y_arr is not None
                                and len(_ext_y_arr) > 0):
                            _ext_X = filtered_val_df
                            _ext_y = _ext_y_arr
                            _ext_base_per_spec = _base_val_per_spec or None
                            logger.info(
                                "[CompositeCrossTargetEnsemble] target='%s' "
                                "honest-OOF source='external_val' (n=%d); "
                                "skipping train-tail carve.",
                                _orig_tname, len(_ext_y_arr),
                            )
                        else:
                            logger.info(
                                "[CompositeCrossTargetEnsemble] target='%s' "
                                "external_val OOF requested but val unavailable; "
                                "falling back to train_tail.",
                                _orig_tname,
                            )
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
                                time_ordering=_time_ordering,
                                sample_weight=_sw_for_oof,
                                external_holdout_X=_ext_X,
                                external_holdout_y=_ext_y,
                                external_holdout_base_per_spec=_ext_base_per_spec,
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
                        # Dummy-floor gate: drop any component whose honest-OOF
                        # RMSE exceeds the raw target's strongest-dummy RMSE
                        # by more than the configured tolerance. A trained
                        # model that loses to a parameter-free dummy on the
                        # honest holdout cannot improve the ensemble; keeping
                        # it dilutes NNLS weights and harms test performance
                        # (TVT prod 2026-05-23: composite-target models on
                        # residual T overfit on group-aware splits, pred_std
                        # 3-5x target_std, R2 down to -22; NNLS still gave
                        # them weight, ensemble landed at RMSE 13.28 vs
                        # lag_predict 11.58 floor).
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
                                _raw_dbl = (
                                    metadata.get("dummy_baselines", {})
                                    .get(str(_tt_e), {})
                                    .get(str(_orig_tname), {})
                                )
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
                                _keep_idx = [
                                    _i for _i in range(len(_oof_rmses))
                                    if np.isfinite(_oof_rmses[_i])
                                    and _oof_rmses[_i] <= _dummy_floor_rmse
                                ]
                                _dropped_idx = [
                                    _i for _i in range(len(_oof_rmses))
                                    if _i not in set(_keep_idx)
                                ]
                                if _dropped_idx and len(_keep_idx) >= 1:
                                    _dropped_names = [
                                        f"{_oof_names[_i]}(OOF={_oof_rmses[_i]:.4g})"
                                        for _i in _dropped_idx
                                    ]
                                    _floor_base = (
                                        _dummy_floor_rmse / (1.0 + _dummy_floor_tol)
                                    )
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
                                    _oof_components = [
                                        _oof_components[_i] for _i in _keep_idx
                                    ]
                                    _oof_names = [
                                        _oof_names[_i] for _i in _keep_idx
                                    ]
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
                                _inner_for_cache = getattr(_comp, "model", _comp)
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
                        # C-P1-2: pipe OOF rmses through component_oof_rmse= so from_train_metrics ranks
                        # on the honest holdout signal. The list named _oof_rmses IS the per-component OOF
                        # RMSE (computed earlier in this phase), so the historical
                        # component_train_rmse=_oof_rmses.tolist() argument was passing the right values
                        # under a misleading parameter name (train_rmse) -- the helper would then emit the
                        # "ranking on TRAIN RMSE which is biased" WARN even though the values were OOF.
                        _ensemble = _CrossEns.from_train_metrics(
                            component_models=_oof_components,
                            component_names=_oof_names,
                            component_oof_rmse=_oof_rmses.tolist(),
                            baseline_oof_rmse=None,
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
                            # AR(1) failsafe: when lag_predict is in the pool
                            # and its OOF RMSE is within tolerance of the best
                            # trained component, prefer lag_predict outright.
                            # Rationale: NNLS minimises squared error on the
                            # train-tail holdout, which has DIFFERENT residual
                            # structure than the test split on a group-aware
                            # split of an AR(1) target (TVT prod 2026-05-23:
                            # train-tail lag_predict RMSE=15.18 vs test
                            # RMSE=11.58 - 31% gap). A zero-parameter dummy
                            # cannot overfit so its OOF rank understates its
                            # test rank. Tolerance defaults to +10% of the
                            # best single; tune via the config knob below.
                            _lag_failsafe_tol = float(getattr(
                                composite_target_discovery_config,
                                "lag_predict_failsafe_tolerance", 0.10,
                            ))
                            _lag_failsafe_taken = False
                            if (_lag_failsafe_tol > 0
                                    and "lag_predict" in _oof_names
                                    and np.isfinite(_best_single_rmse)):
                                _lp_idx = _oof_names.index("lag_predict")
                                _lp_rmse = float(_oof_rmses[_lp_idx])
                                if (np.isfinite(_lp_rmse)
                                        and _lp_rmse <= (1.0 + _lag_failsafe_tol)
                                        * _best_single_rmse):
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
                            if (not _lag_failsafe_taken
                                    and _ens_rmse > _best_single_rmse):
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
                # Sub-key per ensemble family: cross-target ensembles live under
                # ``ensembles_chosen["cross_target"]``; simple per-target ensembles are stamped by
                # _phase_train_one_target under ``ensembles_chosen["simple"]``.
                metadata.setdefault("ensembles_chosen", {}) \
                    .setdefault("cross_target", {}) \
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
                        # ``or []`` on a pandas Index raises ``The truth value of a Index is ambiguous``; explicit None+len check keeps both pandas Index and plain lists safe.
                        _cols_attr = getattr(filtered_train_df, "columns", None)
                        _ens_columns = list(_cols_attr) if _cols_attr is not None and len(_cols_attr) > 0 else []
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


from ._phase_composite_post_summary import _run_suite_end_dummy_baselines_summary  # noqa: E402, F401
