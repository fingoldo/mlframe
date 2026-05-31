"""Cross-target ensemble per-target helper carved out of ``_phase_composite_post.run_composite_post_processing``.

Holds the body of the inner ``for _orig_tname, _spec_list in _tt_specs.items()`` loop. Behavioural-equivalent extract: every closure-captured local from the parent is passed in explicitly and mutations on ``models``/``metadata``/``_train_pred_cache`` happen on the caller-owned dicts (mutation is the existing contract).

Undefined ``ctx`` references in the original body (e.g. ``getattr(ctx, "timestamps", None) if "ctx" in dir() else None``) evaluated to ``None`` because ``ctx`` is not bound inside the function frame; preserved verbatim here so the new path retains the same behaviour.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..composite import CompositeCrossTargetEnsemble as _CrossEns
from ..composite import compute_oof_holdout_predictions
from ..composite_post_shim import PrePipelinePredictShim
from .utils import _build_full_column_from_splits
from ._phase_composite_post_lag_predict import _LagPredictDeployableModel

logger = logging.getLogger("mlframe.training.core._phase_composite_post")

_DEFAULT_OOF_RANDOM_STATE = 42
_PROB_NORM_EPS = 1e-12


class MTRPerColumnEqualMeanEnsemble:
    """E2 (F-34, 2026-05-31) + E3 (F-34, 2026-05-31): per-column ensemble
    for MULTI_TARGET_REGRESSION cross-target ensembling.

    Wraps a list of K trained component models (each producing (N, K)
    predictions on input X). Two strategies:

      * ``strategy="equal_mean"`` (default): equal weight 1 / n_components
        per component per column. No fit() needed; works immediately.
      * ``strategy="nnls"`` (E3): non-negative least-squares per-column
        weights learned from a held-out (X, y) set via .fit(X, y).
        Weights are normalised to sum to 1 (or fall back to equal-mean
        when NNLS returns the all-zero degenerate solution). Per
        target column k, solves: y[:, k] = A_k @ w_k, w_k >= 0 where
        A_k is the (N, n_components) component-prediction matrix on
        the held-out X. Independent solve per column so the K targets
        can have different optimal component mixtures.

    The class keeps its original name (``MTRPerColumnEqualMeanEnsemble``)
    for backward compatibility with the existing ``isinstance`` checks
    and ``models`` dict entries; the ``strategy`` kwarg controls the
    behaviour. Future PR can wire honest-OOF (held-out preds from
    cross-validated folds) by calling .fit() with the OOF stack
    instead of an in-sample held-out set; the predict() contract is
    unchanged.

    The wrapper exposes the standard sklearn-shape predict(X) ->
    np.ndarray so the suite's downstream save / report layers treat
    it as any other regressor.
    """

    def __init__(
        self,
        components,
        component_names,
        n_targets: int,
        *,
        strategy: str = "equal_mean",
        weights: np.ndarray = None,
    ):
        if not components:
            raise ValueError("MTRPerColumnEqualMeanEnsemble requires at least 1 component")
        if strategy not in ("equal_mean", "nnls"):
            raise ValueError(
                f"strategy must be 'equal_mean' or 'nnls'; got {strategy!r}"
            )
        self._components = list(components)
        self._component_names = list(component_names) if component_names else [
            f"comp{i}" for i in range(len(self._components))
        ]
        self._n_targets = int(n_targets)
        self._strategy = strategy
        # Pre-supplied weights take precedence (e.g. a future PR that
        # computes honest-OOF weights externally and injects them).
        # Shape contract: (n_components, n_targets).
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.shape != (len(self._components), self._n_targets):
                raise ValueError(
                    f"weights shape {weights.shape} != "
                    f"({len(self._components)}, {self._n_targets})"
                )
            self._weights = weights
            self._strategy = "nnls"  # caller provided weights -> use them
        else:
            # Equal-mean default: 1 / n_components per (component, target).
            self._weights = np.full(
                (len(self._components), self._n_targets),
                1.0 / len(self._components),
                dtype=np.float64,
            )

    @property
    def components(self):
        return tuple(self._components)

    @property
    def component_names(self):
        return tuple(self._component_names)

    @property
    def n_targets(self) -> int:
        return self._n_targets

    @property
    def strategy(self) -> str:
        return self._strategy

    @property
    def weights(self) -> np.ndarray:
        """(n_components, n_targets) array. Columns are per-target
        weight vectors; rows are component contributions. Each column
        sums to 1.0 by construction (equal_mean) or post-normalisation
        (nnls). Defensive copy."""
        return self._weights.copy()

    def fit(self, X, y) -> "MTRPerColumnEqualMeanEnsemble":
        """E3: fit per-column NNLS weights from a held-out (X, y) set.

        For each target column k, solves:
            y[:, k] = A_k @ w_k,  subject to w_k >= 0
        where A_k[:, j] = self._components[j].predict(X)[:, k].
        Then normalises w_k to sum to 1 (so the prediction is a convex
        combination). When NNLS returns the all-zero degenerate
        solution (no component fits the column), falls back to
        equal-mean for that column.

        No-op when strategy == "equal_mean" (the equal-weight ensemble
        doesn't depend on training data).

        Args:
            X: held-out features (n_holdout, n_features).
            y: held-out targets (n_holdout, n_targets) or (n_holdout,)
               for K=1.

        Returns: self (sklearn convention).
        """
        if self._strategy == "equal_mean":
            return self  # no-op
        from scipy.optimize import nnls as _nnls

        # Gather (n_components, N, K) prediction stack on the held-out X.
        comp_preds = []
        for c in self._components:
            p = np.asarray(c.predict(X), dtype=np.float64)
            if p.ndim == 1:
                p = p.reshape(-1, 1)
            comp_preds.append(p)
        stacked = np.stack(comp_preds, axis=0)  # (n_comp, N, K)

        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if y_arr.shape[1] != self._n_targets:
            raise ValueError(
                f"y.shape[1] = {y_arr.shape[1]} != n_targets = {self._n_targets}"
            )

        n_comp = len(self._components)
        weights = np.zeros((n_comp, self._n_targets), dtype=np.float64)
        for k in range(self._n_targets):
            A_k = stacked[:, :, k].T  # (N, n_comp)
            b_k = y_arr[:, k]
            w_k, _residual = _nnls(A_k, b_k, maxiter=200)
            w_sum = float(w_k.sum())
            if w_sum > 0:
                # Keep RAW NNLS weights (not normalised to sum to 1).
                # Normalising distorts the optimal fit when component
                # predictions don't bracket the target -- e.g. if all
                # components emit values smaller than y, optimal NNLS
                # may produce weights that sum to >1 (boosting the
                # prediction); normalising would force a convex-
                # combination interpretation that loses the optimum.
                # Trade: the per-column weights are no longer
                # interpretable as "probabilities" / a "convex
                # combination", but the predictions stay optimal.
                weights[:, k] = w_k
            else:
                # Degenerate: NNLS returned all-zero (e.g. all
                # component preds are exactly zero for this column).
                # Fall back to equal-mean for THIS column only.
                weights[:, k] = 1.0 / n_comp
        self._weights = weights
        return self

    def predict(self, X) -> np.ndarray:
        preds_stack = []
        for c in self._components:
            p = np.asarray(c.predict(X))
            if p.ndim == 1:
                # Single-target component? Promote (N,) to (N, 1) so the
                # stack shape is consistent; downstream caller must
                # ensure all components have the same n_targets dimension.
                p = p.reshape(-1, 1)
            preds_stack.append(p)
        stacked = np.stack(preds_stack, axis=0)  # (n_components, N, K)
        # Per-column weighted sum: (n_comp, N, K) @ (n_comp, K) -> (N, K).
        # Uses einsum so the per-column weight is applied to the
        # matching column of each component's preds; equal-mean
        # collapses to stacked.mean(axis=0) by construction.
        return np.einsum("cnk,ck->nk", stacked, self._weights)

    def __repr__(self) -> str:
        return (
            f"MTRPerColumnEqualMeanEnsemble("
            f"n_components={len(self._components)}, "
            f"n_targets={self._n_targets}, "
            f"strategy={self._strategy!r}, "
            f"components={self._component_names!r})"
        )


def _build_mtr_per_column_ensemble(
    *, _tt_e, _orig_tname, models, metadata, target_by_type,
    fit_X=None, fit_y=None,
) -> None:
    """E2 + E3 helper: build a per-column ensemble for an MTR target.

    Strategy auto-selected:
      * ``fit_X`` + ``fit_y`` provided AND len(components) >= 2 -> NNLS
        per-column weights learned from the (fit_X, fit_y) hold-out
        (E3). The held-out preds the ensemble's .fit() consumes are
        in-sample by default unless the caller passes a CV / OOF
        stack; future PR will route true honest-OOF here.
      * Otherwise -> equal_mean (the E2 default).

    Mutates ``models`` and ``metadata`` in place to mirror the
    single-target CT_ENSEMBLE registration shape.
    """
    _orig_entries = (models or {}).get(_tt_e, {}).get(_orig_tname, []) or []
    _components: list[Any] = []
    _component_names: list[str] = []
    for _i, _entry in enumerate(_orig_entries):
        _inner = getattr(_entry, "model", None) or _entry
        if not hasattr(_inner, "predict"):
            continue
        _pp = getattr(_entry, "pre_pipeline", None)
        _name = f"raw#{_i}"
        _components.append(PrePipelinePredictShim(_inner, _pp, _name))
        _component_names.append(_name)

    if len(_components) < 2:
        logger.info(
            "[MTR CT_ENSEMBLE] target='%s': only %d component(s) "
            "available; need >=2 for an ensemble. Skipping.",
            _orig_tname, len(_components),
        )
        return

    # Probe K (n_targets) from the routed target_by_type entry.
    try:
        _y_full = (target_by_type or {}).get(_tt_e, {}).get(_orig_tname)
        _y_arr = np.asarray(_y_full) if _y_full is not None else None
        _K = int(_y_arr.shape[1]) if _y_arr is not None and _y_arr.ndim == 2 else 1
    except Exception:
        _K = 1

    # E3 (2026-05-31): auto-pick strategy. NNLS when held-out data is
    # provided; equal_mean otherwise.
    _use_nnls = fit_X is not None and fit_y is not None
    _strategy_label = "per_column_nnls" if _use_nnls else "per_column_equal_mean"
    _ensemble_model = MTRPerColumnEqualMeanEnsemble(
        components=_components,
        component_names=_component_names,
        n_targets=_K,
        strategy=("nnls" if _use_nnls else "equal_mean"),
    )
    if _use_nnls:
        try:
            _ensemble_model.fit(fit_X, fit_y)
        except Exception as _nnls_err:
            # NNLS failed (singular A_k, scipy bug, etc.) -- log + fall
            # back to equal_mean instead of crashing the whole suite.
            logger.warning(
                "[MTR CT_ENSEMBLE] target='%s': NNLS weight-learning "
                "failed (%s); falling back to equal-mean weights.",
                _orig_tname, _nnls_err,
            )
            _ensemble_model = MTRPerColumnEqualMeanEnsemble(
                components=_components,
                component_names=_component_names,
                n_targets=_K,
                strategy="equal_mean",
            )
            _strategy_label = "per_column_equal_mean"
    _ens_entry = SimpleNamespace(
        model=_ensemble_model,
        pre_pipeline=None,
        # mirror the structural keys the standard CT_ENSEMBLE entry
        # carries so downstream consumers that look for these attrs
        # don't KeyError
        model_name=f"CT_ENSEMBLE_MTR[{','.join(_component_names)}]",
        target_name=_orig_tname,
        ct_ensemble=True,
        mtr_ensemble=True,
        ensemble_strategy=_strategy_label,
        n_components=len(_components),
        component_names=tuple(_component_names),
    )
    models.setdefault(_tt_e, {}).setdefault(_orig_tname, []).append(_ens_entry)
    metadata.setdefault("mtr_ct_ensemble", {}).setdefault(
        str(_tt_e), {})[_orig_tname] = {
        "strategy": _strategy_label,
        "n_components": len(_components),
        "component_names": list(_component_names),
        "n_targets": _K,
    }
    logger.info(
        "[MTR CT_ENSEMBLE] target='%s' (K=%d): %s ensemble built "
        "over %d components: %s.",
        _orig_tname, _K, _strategy_label, len(_components), _component_names,
    )


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
) -> None:
    """Build CT_ENSEMBLE for one (target_type, original_target_name).

    Mutates ``models``, ``metadata``, and ``_train_pred_cache`` in place; same contract as the original inline loop body.
    """
    # F-34 (2026-05-31) + E2 (2026-05-31): MULTI_TARGET_REGRESSION
    # cross-target ensemble path. The general CT_ENSEMBLE flow below
    # assumes 1-D y per component (uses sklearn metrics that expect 1-D
    # plus the honest-OOF blender that solves a 1-D regression at the
    # component level). For (N, K) MTR targets we build a SIMPLIFIED
    # per-column equal-weight mean ensemble: stack each component's
    # (N, K) predictions across a "component" axis, then average across
    # components to produce a single (N, K) ensemble output.
    # Equal-weight is the MVP; future PR can swap in per-column honest-
    # OOF blended weights without changing the public deployable model
    # interface.
    try:
        from mlframe.training import TargetTypes
        _is_mtr = (
            str(_tt_e) == str(TargetTypes.MULTI_TARGET_REGRESSION)
            or (hasattr(_tt_e, "is_multi_target_regression")
                and _tt_e.is_multi_target_regression)
        )
    except Exception:
        _is_mtr = False

    if _is_mtr:
        # E3 (2026-05-31): pass the val-fold (X, y) hold-out so the
        # per-column ensemble learns NNLS weights instead of using
        # equal-mean. The val fold is the closest stand-in for an
        # honest-OOF set in the current dispatcher signature; a future
        # PR can route true OOF stacks here without changing the
        # ensemble's .fit() contract.
        _fit_X = filtered_val_df if filtered_val_df is not None else val_df_pd
        _fit_y_full = (target_by_type or {}).get(_tt_e, {}).get(_orig_tname)
        _fit_y_val = None
        if _fit_X is not None and _fit_y_full is not None:
            try:
                _y_arr = np.asarray(_fit_y_full)
                _val_idx_eff = filtered_val_idx if filtered_val_idx is not None else val_idx
                if _val_idx_eff is not None:
                    _fit_y_val = _y_arr[_val_idx_eff]
                else:
                    # No explicit val index -- if the full y rows align
                    # with the val-frame row count, use that directly.
                    if _y_arr.shape[0] == len(_fit_X):
                        _fit_y_val = _y_arr
            except Exception as _val_y_err:
                logger.debug(
                    "[MTR CT_ENSEMBLE] target='%s': val y extraction failed (%s); "
                    "falling back to equal-mean.",
                    _orig_tname, _val_y_err,
                )
                _fit_y_val = None
        _build_mtr_per_column_ensemble(
            _tt_e=_tt_e,
            _orig_tname=_orig_tname,
            models=models,
            metadata=metadata,
            target_by_type=target_by_type,
            fit_X=_fit_X if _fit_y_val is not None else None,
            fit_y=_fit_y_val,
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
        _components.append(
            PrePipelinePredictShim(_inner, _pp, _name)
        )
        _component_names.append(_name)
    # Inject lag_predict dummy baseline as a free component for the cross-target ensemble pool. On strongly auto-regressive targets (lag1_corr ~0.999 within groups) the dumbest ``y_hat = lag_target_value`` baseline often beats every trained model on RMSE; honest-OOF gate naturally selects it when it dominates. NO trainable parameters; cost is one column read.
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
        return
    # Score components on the train slice in y-scale (same rows wrappers were fitted on).
    _y_full_for_rmse = target_by_type.get(_tt_e, {}).get(_orig_tname)
    _component_train_rmses: list[float] = []
    if _y_full_for_rmse is not None:
        _y_train_for_rmse = np.asarray(_y_full_for_rmse)[filtered_train_idx]
        # Frame-content key (id(frame)+shape) shields against id() recycling: wrap-pass objects may be GC'd before we look up here.
        _frame_key = (id(filtered_train_df), getattr(filtered_train_df, "shape", None))
        for _comp, _name in zip(_components, _component_names):
            try:
                # Cache key is the INNER model id; shims are built per-pass so id(_comp) never hits the wrap-pass cache.
                _inner_for_cache = getattr(_comp, "model", _comp)
                _pred = _train_pred_cache.get((id(_inner_for_cache),) + _frame_key)
                if _pred is None:
                    _pred = _train_pred_cache.get((id(_comp),) + _frame_key)
                if _pred is None:
                    _pred = np.asarray(
                        _comp.predict(filtered_train_df),
                        dtype=np.float64,
                    ).reshape(-1)
                    _train_pred_cache[(id(_inner_for_cache),) + _frame_key] = _pred
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
        return
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
        # Thread ctx.timestamps + per-target sample_weight so the OOF holdout split becomes time-aware (trailing-slice past-only train) rather than random shuffle. ``ctx`` is not bound in the original function frame either; the ``if "ctx" in dir()`` guard always evaluated to False so these always resolved to None.
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
        # Resolve the OOF holdout source. ``external_val`` (default) replaces the train-tail carving with the suite's val frame.
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
        _group_ids_for_oof = None
        _ctx_groups = getattr(ctx, "group_ids", None) if "ctx" in dir() else None
        if _ctx_groups is not None:
            try:
                _group_ids_for_oof = (
                    np.asarray(_ctx_groups)[filtered_train_idx]
                )
            except (TypeError, IndexError):
                _group_ids_for_oof = None

        # 2026-05-26 OOF pre-screen optimisation: the dummy-floor gate
        # at the BOTTOM of this function frequently drops 60-70% of
        # components (observed in prod: 14/21 dropped). Pre-fix, all
        # 21 components were OOF-refit (~10 min/MLP, ~5 min/booster),
        # then 14 immediately discarded -- ~30-50 minutes of pure
        # waste per target. Pre-screen now uses Phase-4-trained
        # models' predict() on the external_val frame (cheap; no
        # refit) to compute a LEAKY val_RMSE estimate -- leaks only
        # through early-stopping signal, not full training -- and
        # drops components whose leaky RMSE clears the dummy floor
        # with a generous safety margin. Final dummy-floor gate STILL
        # runs after OOF on the honest refit RMSE, so this is a
        # speed-up only; correctness contract unchanged.
        _PRESCREEN_SAFETY = 1.5  # leaky RMSE * 1.5 must still clear floor
        if (_ext_X is not None and _ext_y is not None
                and len(_components) >= 4):
            try:
                _raw_dbl_pre = (
                    metadata.get("dummy_baselines", {})
                    .get(str(_tt_e), {})
                    .get(str(_orig_tname), {})
                )
                _data_pre = _raw_dbl_pre.get("data", {}) if isinstance(_raw_dbl_pre, dict) else {}
                _strongest_pre = _raw_dbl_pre.get("strongest") if isinstance(_raw_dbl_pre, dict) else None
                _pm_pre = _raw_dbl_pre.get("primary_metric") if isinstance(_raw_dbl_pre, dict) else None
                _dummy_floor_for_prescreen = None
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
                            if (_leaky_rmse / _PRESCREEN_SAFETY
                                    > _dummy_floor_for_prescreen):
                                _keep_mask.append(False)
                                _dropped_pre.append(
                                    f"{_name}(leakyRMSE={_leaky_rmse:.4g})"
                                )
                            else:
                                _keep_mask.append(True)
                        except Exception:
                            _keep_mask.append(True)  # err on the safe side
                    if _dropped_pre and sum(_keep_mask) >= 2:
                        _kept = [i for i, k in enumerate(_keep_mask) if k]
                        logger.warning(
                            "[CompositeCrossTargetEnsemble] target='%s' "
                            "OOF pre-screen dropped %d/%d component(s) "
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
                    "[CompositeCrossTargetEnsemble] OOF pre-screen "
                    "failed (non-fatal): %s. Continuing with full OOF "
                    "refit.", _prescreen_err,
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
                    group_ids=_group_ids_for_oof,
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
            # Vectorised per-column RMSE: mask non-finite to 0 in a sum-and-divide pass; all-non-finite columns land as NaN (one pass, K cols).
            _diff_mat = _oof_pred_matrix - _oof_y_holdout[:, None]
            _finite_mat = np.isfinite(_diff_mat)
            _n_fin = _finite_mat.sum(axis=0)
            _sq_sum = np.where(_finite_mat, _diff_mat * _diff_mat, 0.0).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                _oof_rmses = np.where(_n_fin > 0, np.sqrt(_sq_sum / np.maximum(_n_fin, 1)), np.nan)
            _oof_rmses = _oof_rmses.astype(np.float64, copy=False)
            logger.info(
                "[CompositeCrossTargetEnsemble] target='%s' using "
                "honest OOF holdout (frac=%.2f, n=%d) for ensemble "
                "weights / stacking.",
                _orig_tname, _oof_frac, len(_oof_y_holdout),
            )
            # Dummy-floor gate: drop any component whose honest-OOF RMSE exceeds the raw target's strongest-dummy RMSE by more than the configured tolerance. A trained model that loses to a parameter-free dummy on the honest holdout cannot improve the ensemble; keeping it dilutes NNLS weights and harms test performance.
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
                # Stacking-aware gate (opt-in) -- drop components whose NNLS weight on the honest OOF preds falls below the configured threshold BEFORE running the actual stacker.
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
                # Preallocate (n_rows, K) to skip np.column_stack's per-entry copy doubling peak RAM; frame-content key shields against id() recycling between waves.
                _frame_key2 = (id(filtered_train_df), getattr(filtered_train_df, "shape", None))
                _n_rows = int(len(_y_for_stack))
                _pred_matrix = np.empty((_n_rows, len(_oof_components)), dtype=np.float64)
                for _ci, (_comp, _name) in enumerate(zip(_oof_components, _oof_names)):
                    # Inner-keyed cache lookup (shim ids are per-pass and never hit the wrap-pass cache).
                    _inner_for_cache = getattr(_comp, "model", _comp)
                    _pred = _train_pred_cache.get((id(_inner_for_cache),) + _frame_key2)
                    if _pred is None:
                        _pred = _train_pred_cache.get((id(_comp),) + _frame_key2)
                    if _pred is None:
                        _pred = np.asarray(
                            _comp.predict(filtered_train_df),
                            dtype=np.float64,
                        ).reshape(-1)
                        _train_pred_cache[(id(_inner_for_cache),) + _frame_key2] = _pred
                    _pred_matrix[:, _ci] = _pred
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
            # C-P1-2: pipe OOF rmses through component_oof_rmse= so from_train_metrics ranks on the honest holdout signal.
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
                # AR(1) failsafe: when lag_predict is in the pool and its OOF RMSE is within tolerance of the best trained component, prefer lag_predict outright.
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
        return
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
    # Stamp the chosen ensemble flavour into metadata["ensembles_chosen"] so the predict path can replay the right combine for the cross-target slot (predict-path parity).
    _ce_actual_strategy = getattr(_ensemble, "strategy", None) or _ce_strategy
    # Sub-key per ensemble family: cross-target ensembles live under ``ensembles_chosen["cross_target"]``; simple per-target ensembles are stamped by _phase_train_one_target under ``ensembles_chosen["simple"]``.
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
            # Compute train envelope once so val + test ensemble reports
            # share the same TRAIN-bound prediction clip (k=3 sigma)
            # rather than each fallback-deriving a different eval bound.
            _ens_train_envelope = None
            try:
                _ens_y_train = _ens_y_arr[filtered_train_idx] if filtered_train_idx is not None else None
                if _ens_y_train is not None and len(_ens_y_train) > 0:
                    from .._prediction_envelope_clip import compute_train_envelope_stats
                    _ens_train_envelope = compute_train_envelope_stats(_ens_y_train)
            except Exception:
                _ens_train_envelope = None
            _ens_common = dict(
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
                        from sklearn.metrics import mean_absolute_error, root_mean_squared_error
                        _y_arr = np.asarray(_y_split, dtype=np.float64).reshape(-1)
                        _ens_arr = _ens_preds.reshape(-1)
                        _ens_scalar_metrics = {
                            f"{_split_name}_RMSE": float(root_mean_squared_error(_y_arr, _ens_arr)),
                            f"{_split_name}_MAE": float(mean_absolute_error(_y_arr, _ens_arr)),
                            "model_name": f"CT_ENSEMBLE[{_ce_strategy}]",
                        }
                        metadata.setdefault("cross_target_ensemble_metrics", {}) \
                            .setdefault(str(_tt_e), {}) \
                            .setdefault(_orig_tname, {}) \
                            .update(_ens_scalar_metrics)
                    except Exception as _metric_err:
                        logger.debug(
                            "Could not stamp CT_ENSEMBLE %s metrics for target='%s': %s",
                            _split_name, _orig_tname, _metric_err,
                        )
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
