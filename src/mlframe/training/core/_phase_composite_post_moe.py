"""MoE selection gate + composite VALUE report wired into the composite deploy path.

This is the one place where the three experts -- the deployed composite ensemble, the raw-y model, and the
``lag_predict`` failsafe -- coexist with true ``y`` and ``group_ids`` on an honest split, so it is the correct
home for (1) the per-group composite VALUE report and (2) the ``MoESelectionGate`` that makes the shipped
prediction provably never worse than lag per group.

Both features are flag-gated and no-op cleanly when their inputs are missing: with the MoE flag off, or no lag
expert, or no group ids, or no resolvable predict-time group column, the deployed model is left byte-identical.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..composite import (
    MoESelectionGate,
    build_composite_value_report,
    render_composite_value_report,
)
from ..composite.post_shim import PrePipelinePredictShim
from ._phase_composite_post_lag_predict import _LagPredictDeployableModel

logger = logging.getLogger("mlframe.training.core._phase_composite_post")


def _extract_group_array(frame: Any, group_column: str) -> np.ndarray | None:
    """Narrow, non-raising pull of a single group-key column (polars / pandas) -> 1-D ndarray, else None.

    Reads ONE column (never a frame copy) so a 100GB frame stays untouched; a missing / unreadable column
    degrades to None (the caller then no-ops the group-aware path rather than raising).
    """
    if not group_column or frame is None:
        return None
    try:
        if hasattr(frame, "get_column"):  # polars
            if group_column not in getattr(frame, "columns", []):
                return None
            return np.asarray(frame.get_column(group_column).to_numpy()).reshape(-1)
        if hasattr(frame, "columns") and group_column in list(getattr(frame, "columns", [])):
            col = frame[group_column]
            return np.asarray(col.to_numpy() if hasattr(col, "to_numpy") else col).reshape(-1)
    except Exception as exc:
        logger.debug("_extract_group_array: column read failed for %r, group-aware path disabled: %s", group_column, exc)
        return None
    return None


class _MoEGatedDeployableModel:
    """Wrap a deployed composite ensemble so ``predict`` routes {composite, raw, lag} through a fitted gate.

    At predict time it computes each expert's prediction on ``X``, extracts the group key from ``X`` via
    ``group_column``, and lets the fitted :class:`MoESelectionGate` deploy the per-group winner -- guaranteed
    never worse than the lag failsafe on the selection split. When the group column is absent from ``X`` the
    gate falls back to its global choice (the lag failsafe), preserving the not-worse-than-lag guarantee.
    """

    def __init__(
        self,
        *,
        composite_model: Any,
        raw_model: Any,
        lag_model: Any,
        gate: MoESelectionGate,
        group_column: str | None,
    ) -> None:
        self.composite_model = composite_model
        self.raw_model = raw_model
        self.lag_model = lag_model
        self.gate = gate
        self.group_column = group_column

    def _expert_preds(self, X: Any) -> dict[str, np.ndarray]:
        """Predict with each of the three experts (composite, raw, lag) on ``X``, returning flat float64 arrays keyed by expert name for the gate to select over."""
        return {
            "composite": np.asarray(self.composite_model.predict(X), dtype=np.float64).reshape(-1),
            "raw": np.asarray(self.raw_model.predict(X), dtype=np.float64).reshape(-1),
            "lag": np.asarray(self.lag_model.predict(X), dtype=np.float64).reshape(-1),
        }

    def predict(self, X: Any) -> np.ndarray:
        """Route each row to its fitted gate-selected expert prediction; falls back to the gate's global (lag failsafe) choice when the group column is absent from ``X``."""
        preds = self._expert_preds(X)
        groups = _extract_group_array(X, self.group_column) if self.group_column else None
        return self.gate.predict(preds, group_ids=groups)

    def __repr__(self) -> str:
        return f"_MoEGatedDeployableModel(group_column={self.group_column!r})"


def _resolve_lag_model(metadata: dict, tt_e: Any, orig_tname: str) -> _LagPredictDeployableModel | None:
    """Rebuild the lag_predict deployable from the dummy-baselines ``feature_used`` column, or None if absent."""
    try:
        _dbl = metadata.get("dummy_baselines", {}).get(str(tt_e), {}).get(str(orig_tname), {})
        _extras = _dbl.get("extras", {}) if isinstance(_dbl, dict) else {}
        _lag_meta = _extras.get("lag_predict") if isinstance(_extras, dict) else None
        _lag_col = _lag_meta.get("feature_used") if isinstance(_lag_meta, dict) else None
        if _lag_col:
            return _LagPredictDeployableModel(_lag_col)
    except Exception as exc:
        logger.debug("_resolve_lag_model: dummy-baselines metadata walk failed, no lag model rebuilt: %s", exc)
        return None
    return None


def _first_raw_shim(models: dict, tt_e: Any, orig_tname: str) -> PrePipelinePredictShim | None:
    """First raw-y model entry for the target, wrapped in a predict shim (its pre_pipeline preserved)."""
    for _entry in (models or {}).get(tt_e, {}).get(orig_tname, []) or []:
        _inner = getattr(_entry, "model", None) or _entry
        if hasattr(_inner, "predict"):
            return PrePipelinePredictShim(_inner, getattr(_entry, "pre_pipeline", None), "raw")
    return None


def run_composite_moe_and_value_report(
    *,
    models: dict,
    metadata: dict,
    target_by_type: dict,
    composite_target_discovery_config,
    filtered_train_df,
    filtered_val_df,
    filtered_train_idx,
    filtered_val_idx,
    ctx: Any = None,
) -> None:
    """Emit the composite VALUE report and (flag-gated) wrap deployed ensembles through a MoE selection gate.

    Runs once per deployed ``_CT_ENSEMBLE__<target>`` regression slot. The SELECTION split is the honest val
    split (rows the deployed components were early-stopped on, so this is the same surface the cross-target
    external_val gate uses): the three experts {deployed composite ensemble, raw-y model, lag_predict} are
    predicted there alongside true ``y`` / ``group_ids``. The value report is stashed on
    ``metadata["composite_value_report"]``; the gate wraps the deployed model in place, guaranteeing the
    shipped prediction is never worse than lag per group. Mutates ``models`` / ``metadata`` in place.
    """
    _emit_report = bool(getattr(composite_target_discovery_config, "emit_composite_value_report", True))
    _moe_enabled = bool(getattr(composite_target_discovery_config, "moe_gate_enabled", True))
    if not (_emit_report or _moe_enabled):
        return
    if filtered_val_df is None or filtered_val_idx is None:
        return

    _group_column = getattr(composite_target_discovery_config, "group_column", None)
    _shrink_rtol = float(getattr(composite_target_discovery_config, "moe_gate_shrink_rtol", 0.0))
    _min_group_rows = int(getattr(composite_target_discovery_config, "moe_gate_min_group_rows", 1))

    _ctx_groups = getattr(ctx, "group_ids", None) if ctx is not None else None
    _ctx_sw = getattr(ctx, "sample_weights", None) if ctx is not None else None

    for _tt_e, _by_name in list((models or {}).items()):
        for _key in list(_by_name.keys()):
            if not str(_key).startswith("_CT_ENSEMBLE__"):
                continue
            _orig_tname = str(_key)[len("_CT_ENSEMBLE__") :]
            _entries = _by_name.get(_key) or []
            if not _entries:
                continue
            _ens_model = getattr(_entries[0], "model", None)
            if _ens_model is None or not hasattr(_ens_model, "predict"):
                continue

            _y_full = (target_by_type or {}).get(_tt_e, {}).get(_orig_tname)
            if _y_full is None:
                continue
            try:
                _y_sel = np.asarray(_y_full, dtype=np.float64).reshape(-1)[filtered_val_idx]
            except (TypeError, IndexError):
                continue
            if _y_sel.size == 0:
                continue

            _raw_shim = _first_raw_shim(models, _tt_e, _orig_tname)
            if _raw_shim is None:
                continue
            _lag_model = _resolve_lag_model(metadata, _tt_e, _orig_tname)

            # Predict the experts on the selection (val) split. A single failing expert aborts this target's
            # gate/report (no fabricated numbers) but leaves every other target untouched.
            try:
                _composite_sel = np.asarray(_ens_model.predict(filtered_val_df), dtype=np.float64).reshape(-1)
                _raw_sel = np.asarray(_raw_shim.predict(filtered_val_df), dtype=np.float64).reshape(-1)
                _lag_sel = None
                if _lag_model is not None:
                    if filtered_train_df is not None:
                        try:
                            _lag_model.fit(filtered_train_df)
                        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                            logger.debug("suppressed in _phase_composite_post_moe.py:185: %s", e)
                            pass
                    _lag_sel = np.asarray(_lag_model.predict(filtered_val_df), dtype=np.float64).reshape(-1)
            except Exception as _pred_err:
                logger.warning(
                    "[CompositeMoE] target='%s': expert prediction on selection split failed (%s); "
                    "skipping value report + gate for this target.", _orig_tname, _pred_err,
                )
                continue

            _groups_sel = None
            if _ctx_groups is not None:
                try:
                    _groups_sel = np.asarray(_ctx_groups).reshape(-1)[filtered_val_idx]
                except (TypeError, IndexError):
                    _groups_sel = None
            _sw_sel = None
            if isinstance(_ctx_sw, dict) and _ctx_sw:
                _sw_raw = _ctx_sw.get(_orig_tname)
                if _sw_raw is not None:
                    try:
                        _sw_sel = np.asarray(_sw_raw, dtype=np.float64).reshape(-1)[filtered_val_idx]
                    except (TypeError, IndexError):
                        _sw_sel = None

            if _emit_report:
                try:
                    _report = build_composite_value_report(
                        _y_sel, _raw_sel, _composite_sel, _groups_sel,
                        y_pred_lag=_lag_sel, sample_weight=_sw_sel,
                    )
                    metadata.setdefault("composite_value_report", {}).setdefault(str(_tt_e), {})[_orig_tname] = _report
                    logger.info(
                        "[CompositeValueReport] target='%s'\n%s",
                        _orig_tname, render_composite_value_report(_report),
                    )
                except Exception as _rep_err:
                    logger.warning(
                        "[CompositeValueReport] target='%s': report build failed (%s); continuing.",
                        _orig_tname, _rep_err,
                    )

            # MoE gate wrap: needs the lag failsafe, group ids on the selection split, AND a predict-time group
            # column present in the val frame (else the deployed gate could only route globally to lag, which
            # is worse than the ensemble where composite wins -- so we no-op and ship the ensemble unchanged).
            if not _moe_enabled or _lag_model is None or _groups_sel is None or _group_column is None:
                continue
            if _extract_group_array(filtered_val_df, _group_column) is None:
                continue
            try:
                _gate = MoESelectionGate(
                    failsafe="lag", shrink_rtol=_shrink_rtol, min_group_rows=_min_group_rows,
                ).fit(
                    _y_sel,
                    {"composite": _composite_sel, "raw": _raw_sel, "lag": _lag_sel},
                    group_ids=_groups_sel,
                    sample_weight=_sw_sel,
                )
                _entries[0].model = _MoEGatedDeployableModel(
                    composite_model=_ens_model,
                    raw_model=_raw_shim,
                    lag_model=_lag_model,
                    gate=_gate,
                    group_column=_group_column,
                )
                metadata.setdefault("composite_moe_gate", {}).setdefault(str(_tt_e), {})[_orig_tname] = {
                    "group_choice": dict(_gate.group_choice_),
                    "global_choice": _gate.global_choice_,
                    "guarantee": dict(_gate.guarantee_),
                    "group_column": _group_column,
                }
                logger.info(
                    "[CompositeMoE] target='%s' wrapped deployed ensemble in MoE gate " "(global='%s', not_worse_than_lag=%s, pooled RMSE gate=%s vs lag=%s).",
                    _orig_tname,
                    _gate.global_choice_,
                    _gate.guarantee_.get("not_worse_than_lag"),
                    _gate.guarantee_.get("pooled_rmse_gate"),
                    _gate.guarantee_.get("pooled_rmse_per_expert", {}).get("lag"),
                )
            except Exception as _gate_err:
                logger.warning(
                    "[CompositeMoE] target='%s': gate fit/wrap failed (%s); deploy left unchanged.",
                    _orig_tname, _gate_err,
                )
