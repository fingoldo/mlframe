"""Suite-end summary + verdict-table formatting for ``dummy_baselines``.

Split out of ``dummy_baselines.py`` to keep the parent below the 1k-line
monolith threshold. Behaviour preserved bit-for-bit; the parent re-exports
the two public formatters so historical
``from .dummy import format_suite_end_summary``
imports continue to resolve.

What lives here:
  - ``format_suite_end_summary``
  - ``format_unified_target_verdict_table``
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def format_suite_end_summary(
    dummy_baselines_metadata: dict[str, Any],
    failures_metadata: dict[str, Any] | None = None,
    best_model_metrics_by_target: dict[tuple[str, str], dict[str, float]] | None = None,
    min_lift: float = 1.5,
    composite_to_raw_target_map: dict[tuple[str, str], str] | None = None,
    cross_target_ensemble_metrics: dict[tuple[str, str], dict[str, float]] | None = None,
) -> str:
    """Format the cross-target verdict block emitted at suite end (D6).

    Operator Contract guarantee 2: a single suite-end block with rows per
    target, columns ``(strongest_dummy, dummy_metric, best_model,
    model_metric, lift_x, verdict)``.

    Auto-emits canonical UPPERCASE WARN tokens (D6, B#10):
      - ``BEST_MODEL_BELOW_DUMMY`` when ``model/dummy < min_lift`` for a
        minimize metric (or ``dummy/model < min_lift`` for maximize).
      - ``ALL_BASELINES_BELOW_RANDOM`` for binary when all classifier
        baselines have AUC < 0.5 (label-flip suspect).
      - ``TS_BEATS_TREES`` when the strongest TS baseline beats the best
        model on val.
      - ``PARTIAL_FAILURE`` when ``failures_metadata`` is non-empty for
        the target.

    Parameters
    ----------
    dummy_baselines_metadata
        ``metadata["dummy_baselines"]`` from
        ``train_mlframe_models_suite`` -- nested dict
        ``{target_type: {target_name: report.to_dict()}}``.
    failures_metadata
        ``metadata["dummy_baselines_failures"]`` (same shape) -- used for
        ``PARTIAL_FAILURE`` WARN.
    best_model_metrics_by_target
        Optional ``{(target_type, target_name): {metric: value}}`` -- used
        for the lift-vs-model column and ``BEST_MODEL_BELOW_DUMMY`` /
        ``TS_BEATS_TREES`` WARN. When None, only the dummy verdict block
        is emitted.
    min_lift
        Lift threshold below which the model is flagged as not
        meaningfully beating dummy. Default 1.5 (model must be >=1.5x
        better than dummy on the primary metric).
    """
    lines: list[str] = []
    if not dummy_baselines_metadata:
        return ""

    lines.append("[DUMMY_BASELINES] CROSS-TARGET VERDICT")
    lines.append(
        f"{'target':<24} {'strongest_dummy':<28} {'dummy_metric':<28} "
        f"{'best_model':<12} {'model_metric':<22} {'lift':<8} verdict"
    )

    warn_lines: list[str] = []
    n_total = 0
    n_healthy = 0

    for target_type, by_name in dummy_baselines_metadata.items():
        for target_name, rep_dict in by_name.items():
            n_total += 1
            strongest = rep_dict.get("strongest")
            primary_metric = rep_dict.get("primary_metric")
            data = rep_dict.get("data", {})
            if not strongest or not primary_metric or strongest not in data:
                continue
            strongest_row = data[strongest]
            dummy_val = strongest_row.get(primary_metric)
            if dummy_val is None:
                continue
            # For composite targets the verdict
            # baseline must be a TRULY trivial baseline -- i.e. the
            # raw-y dummy (``median(y_raw)`` constant prediction on
            # the original y-scale), NOT the inverted T-dummy. Previous
            # impl inverted ``median(T_train)`` through fitted
            # ``alpha * base + beta`` to a y-scale value, but that
            # uses fitted-alpha information from training -- not a
            # "trivial" baseline. The composite then "barely beat" it
            # because they're both non-trivial baselines on y-scale.
            #
            # Correct comparison: composite y-scale model RMSE vs
            # raw target's strongest dummy RMSE (same scale, same
            # honesty). Lookup via composite_to_raw_target_map.
            _used_raw_y_dummy = False
            if composite_to_raw_target_map is not None:
                _raw_tname = composite_to_raw_target_map.get(
                    (str(target_type), str(target_name))
                )
                if _raw_tname is not None:
                    _raw_rep = dummy_baselines_metadata.get(
                        target_type, {}
                    ).get(_raw_tname)
                    if _raw_rep is not None:
                        _raw_strongest = _raw_rep.get("strongest")
                        _raw_pm = _raw_rep.get("primary_metric")
                        _raw_data = _raw_rep.get("data", {})
                        if (_raw_strongest and _raw_pm
                                and _raw_strongest in _raw_data):
                            _raw_dummy_val = _raw_data[_raw_strongest].get(
                                _raw_pm
                            )
                            if _raw_dummy_val is not None:
                                dummy_val = _raw_dummy_val
                                # Override strongest name in the
                                # output for clarity (the raw-y dummy
                                # is conceptually different).
                                strongest = (
                                    f"{_raw_strongest} [raw-y trivial]"
                                )
                                _used_raw_y_dummy = True

            # No raw-y trivial baseline available (raw target not separately
            # registered / name mismatch): fall back to this composite's OWN
            # strongest dummy inverted to y-scale (``y_scale_strongest_metrics``,
            # stamped in _phase_dummy_baselines). The model metric below is the
            # composite's y-scale RMSE, so the T-scale ``primary_metric`` value
            # is a silent scale mismatch -- the inverted dummy is the only
            # same-scale comparison. ``primary_metric`` decomposes as
            # ``<split>_<metric>`` (e.g. ``val_RMSE`` -> split ``val``, metric ``RMSE``).
            if not _used_raw_y_dummy:
                _ys = rep_dict.get("y_scale_strongest_metrics")
                if isinstance(_ys, dict) and "_" in primary_metric:
                    _split_key, _metric_key = primary_metric.split("_", 1)
                    _ys_val = _ys.get(_split_key, {}).get(_metric_key)
                    if _ys_val is not None and np.isfinite(_ys_val):
                        dummy_val = float(_ys_val)
                        strongest = f"{strongest} [y-scale inv]"

            # Best model metric lookup (optional). Considers BOTH the single best
            # individual model AND the cross-target ensemble (NNLS-stack etc.), then
            # picks whichever is stronger on the primary metric. On strong-AR targets
            # the ensemble often beats every single model by stacking on top of
            # lag_predict; before this change the verdict ignored the ensemble and
            # falsely flagged BEST_MODEL_BELOW_DUMMY even when CT_ENSEMBLE clearly
            # cleared the dummy floor.
            best_model_name = "-"
            model_val: float | None = None
            from ..metrics_registry import metric_name_higher_is_better as _mhb_pick
            _direction_pick = _mhb_pick(primary_metric)
            _is_min_for_pick = True if _direction_pick is None else (not _direction_pick)
            def _better(a: float | None, b: float | None) -> bool:
                if a is None or not np.isfinite(a):
                    return False
                if b is None or not np.isfinite(b):
                    return True
                return (a < b) if _is_min_for_pick else (a > b)
            if best_model_metrics_by_target is not None:
                key = (str(target_type), str(target_name))
                model_metrics = best_model_metrics_by_target.get(key, {})
                if model_metrics:
                    # Map the dummy primary_metric to the model's analogue
                    # (e.g. val_RMSE -> RMSE_val or vice versa). Caller is
                    # expected to pass model_metrics keyed compatibly.
                    model_val = model_metrics.get(primary_metric)
                    best_model_name = model_metrics.get("model_name", "--")
            if cross_target_ensemble_metrics is not None:
                _ens_key = (str(target_type), str(target_name))
                _ens_m = cross_target_ensemble_metrics.get(_ens_key, {})
                _ens_val = _ens_m.get(primary_metric) if _ens_m else None
                if _better(_ens_val, model_val):
                    model_val = _ens_val
                    best_model_name = _ens_m.get("model_name", "CT_ENSEMBLE")

            # Wave 20 fix: registry dispatcher. The previous substring
            # whitelist missed val_MAPE (MAPE not in 'MAE' substring),
            # val_MSE (not in 'RMSE'), val_ICE, val_brier, val_KL,
            # val_perplexity -- all lower-is-better metrics whose lift
            # was silently computed as model_val/dummy_val (giving
            # "TASK_NON_TRIVIAL" verdict on degenerate models).
            from ..metrics_registry import metric_name_higher_is_better as _mhb
            _direction = _mhb(primary_metric)
            # Unknown -> default to minimize (the prior heuristic-default).
            is_minimize = True if _direction is None else (not _direction)
            lift_str = "-"
            verdict = "-"
            if model_val is not None and np.isfinite(dummy_val) and np.isfinite(model_val):
                if is_minimize and model_val > 0:
                    lift = dummy_val / model_val
                elif not is_minimize and dummy_val > 0:
                    lift = model_val / dummy_val
                else:
                    lift = float("nan")
                if np.isfinite(lift):
                    lift_str = f"{lift:.2f}x"
                    if lift >= min_lift:
                        verdict = "TASK_NON_TRIVIAL_AND_MODELS_HEALTHY"
                        n_healthy += 1
                    else:
                        verdict = "MODELS_BARELY_BEAT_TRIVIAL"
                        warn_lines.append(
                            f"[DUMMY_BASELINES] WARN BEST_MODEL_BELOW_DUMMY "
                            f"target='{target_name}' lift={lift:.2f}x -- "
                            f"investigate label encoding, target leak, "
                            f"train/test contamination."
                        )

            # TS_BEATS_TREES heuristic: strongest baseline name contains
            # 'naive' / 'seasonal' / 'rolling' / 'linear_extrap', AND
            # model_val is worse than dummy_val on a minimize metric.
            ts_strongest = any(
                tok in str(strongest).lower()
                for tok in ("naive", "seasonal", "rolling", "linear_extrap")
            )
            if (
                ts_strongest
                and model_val is not None
                and is_minimize
                and np.isfinite(model_val)
                and np.isfinite(dummy_val)
                and model_val > dummy_val
            ):
                warn_lines.append(
                    f"[DUMMY_BASELINES] WARN TS_BEATS_TREES "
                    f"target='{target_name}' -- verify val_placement='forward'; "
                    f"check for leaked-from-future feature columns."
                )

            # ALL_BASELINES_BELOW_RANDOM (binary only): every classifier
            # baseline has AUC < 0.5 -> label flip suspected.
            if str(target_type) == "binary_classification":
                aucs = [
                    row.get("val_AUC") for row in data.values()
                    if row.get("val_AUC") is not None
                    and np.isfinite(row.get("val_AUC", float("nan")))
                ]
                if aucs and all(a < 0.5 for a in aucs):
                    warn_lines.append(
                        f"[DUMMY_BASELINES] WARN ALL_BASELINES_BELOW_RANDOM "
                        f"target='{target_name}' -- check target_label_encoder "
                        f"direction; check sign of cost_function."
                    )

            _strongest_label = str(strongest)
            lines.append(
                f"{str(target_name)[:24]:<24} {_strongest_label[:28]:<28} "
                f"{primary_metric}={dummy_val:<.4f}     {str(best_model_name)[:12]:<12} "
                f"{(primary_metric + '=' + (f'{model_val:.4f}' if model_val is not None else '-'))[:22]:<22} "
                f"{lift_str:<8} {verdict}"
            )

    # PARTIAL_FAILURE WARN -- emitted once per target with failures.
    if failures_metadata:
        for target_type, by_name in failures_metadata.items():
            for target_name, err_msg in by_name.items():
                warn_lines.append(
                    f"[DUMMY_BASELINES] WARN PARTIAL_FAILURE "
                    f"target='{target_name}' ({target_type}) -- {err_msg}"
                )

    lines.extend(warn_lines)
    if best_model_metrics_by_target is not None:
        lines.append(
            f"[DUMMY_BASELINES] HEALTH: {n_healthy}/{n_total} targets -- "
            f"{'ALL_HEALTHY' if n_healthy == n_total else 'see WARN lines above'}"
        )

    return "\n".join(lines)


def format_unified_target_verdict_table(
    dummy_baselines_metadata: dict[str, Any],
    best_model_metrics_by_target: dict[tuple[str, str], dict[str, float]] | None = None,
    composite_to_raw_target_map: dict[tuple[str, str], str] | None = None,
    cross_target_ensemble_metrics: dict[tuple[str, str], dict[str, float]] | None = None,
) -> str:
    """Build a single per-original-target consolidated verdict row.

    Side-by-side: ``target | raw_best | composite_best | CT_ensemble | lift_vs_raw% | verdict``. Different from ``format_suite_end_summary`` which emits ONE row per composite spec; this groups composites under their parent raw target and exposes the head-to-head improvement.

    Inputs:
    - ``dummy_baselines_metadata``: per-target dummy data (same shape as ``format_suite_end_summary``).
    - ``best_model_metrics_by_target``: ``{(target_type, target_name): {metric: value}}`` keyed by EACH model the suite trained (raw + per-composite).
    - ``composite_to_raw_target_map``: ``{(target_type, composite_name): raw_name}``. Used to group composites under raw.
    - ``cross_target_ensemble_metrics``: ``{(target_type, raw_target_name): {metric: value}}`` for the NNLS-stack ensemble.

    Empty string when no data is available.
    """
    if not dummy_baselines_metadata or not best_model_metrics_by_target:
        return ""

    lines: list[str] = ["[DUMMY_BASELINES] UNIFIED PER-TARGET VERDICT"]
    lines.append(
        f"{'target':<24} {'raw_best':<14} {'composite_best':<16} "
        f"{'CT_ensemble':<14} {'lift_vs_raw_%':<14} verdict"
    )

    # Group rows: {(target_type, raw_name): {'raw_metric': ..., 'composite_specs': [(name, metric), ...]}}.
    composite_to_raw = composite_to_raw_target_map or {}
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for (tt, tname), m in best_model_metrics_by_target.items():
        raw_name = composite_to_raw.get((str(tt), str(tname))) or str(tname)
        slot = grouped.setdefault(
            (str(tt), raw_name),
            {"raw": None, "composites": []},
        )
        if str(tname) == raw_name:
            slot["raw"] = m
        else:
            slot["composites"].append((str(tname), m))

    for (tt, raw_name), slot in grouped.items():
        raw_metric = slot["raw"] or {}
        comp_specs = slot["composites"]
        # Prefer val_RMSE; fall back to whatever the slot has.
        raw_val = raw_metric.get("val_RMSE") or raw_metric.get("RMSE")
        best_comp = min(
            (c for c in comp_specs
             if (c[1].get("val_RMSE") or c[1].get("RMSE")) is not None),
            key=lambda c: c[1].get("val_RMSE") or c[1].get("RMSE"),
            default=None,
        )
        comp_val = (
            best_comp[1].get("val_RMSE") or best_comp[1].get("RMSE")
            if best_comp else None
        )
        ct_val = None
        if cross_target_ensemble_metrics:
            ct_entry = cross_target_ensemble_metrics.get((str(tt), raw_name))
            if ct_entry:
                ct_val = ct_entry.get("val_RMSE") or ct_entry.get("RMSE")
        lift = (
            (float(raw_val) / float(comp_val)) if (raw_val and comp_val and comp_val > 0)
            else None
        )
        verdict_parts = []
        if lift is not None and lift > 1.05:
            verdict_parts.append("COMPOSITE_WINS")
        elif lift is not None and lift < 0.95:
            verdict_parts.append("RAW_WINS")
        else:
            verdict_parts.append("TIE")
        if ct_val is not None and raw_val is not None and ct_val < raw_val * 0.95:
            verdict_parts.append("CT_ENSEMBLE_HELPS")
        verdict = " ".join(verdict_parts) or "-"

        lines.append(
            f"{raw_name[:24]:<24} "
            f"{('-' if raw_val is None else f'{raw_val:.4f}'):<14} "
            f"{('-' if comp_val is None else f'{comp_val:.4f}'):<16} "
            f"{('-' if ct_val is None else f'{ct_val:.4f}'):<14} "
            f"{('-' if lift is None else f'{(lift-1)*100:+.1f}%'):<14} "
            f"{verdict}"
        )

    return "\n".join(lines)
