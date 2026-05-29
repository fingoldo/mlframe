"""``_run_suite_end_dummy_baselines_summary`` carved out of
``_phase_composite_post`` so the parent stays under the 1k-line
monolith threshold. Re-exported from the parent's module bottom so
historical ``from ._phase_composite_post import
_run_suite_end_dummy_baselines_summary`` keeps resolving.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..dummy_baselines import format_suite_end_summary
from .utils import _entry_metric

logger = logging.getLogger("mlframe.training.core._phase_composite_post")


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
                # Registry dispatcher: substring whitelist missed MAPE / MSE / ICE / brier / KL /
                # perplexity -- those would silently route through the
                # else-branch and pick the WORST model as "best" for the
                # suite-end verdict block.
                from ..metrics_registry import metric_name_higher_is_better as _mhb
                _direction = _mhb(_metric_name)
                _is_minimize = True if _direction is None else (not _direction)
                # For composite targets prefer y-scale metrics (post-inverse, comparable to raw / y-scale dummy).
                _yscale_entries = (
                    metadata.get("composite_target_y_scale_metrics", {})
                    .get(str(_tt), {})
                    .get(_tname, [])
                )
                _best_val: float | None = None
                _best_name = "-"
                _best_split = None  # "val" or "test" -- track for tag
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
                            _best_split = "val"
                else:
                    # First pass: prefer VAL metrics (aligned with dummy's
                    # primary_metric which is val_*).
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
                            _best_split = "val"
                    # Second pass: fall back to TEST metrics if VAL was
                    # never populated for any model in this slot. The
                    # verdict block then surfaces a "(test)" tag so the
                    # operator sees this is a cross-split comparison
                    # (apples-to-oranges with the val_RMSE dummy but
                    # still strictly more informative than "-"). Prod
                    # incident: the trained models' val metrics were
                    # unpopulated in the suite run, so the verdict
                    # table showed "best_model=-" even though Ridge
                    # had TEST RMSE=11.63 right there in the log.
                    if _best_val is None:
                        for _m in _model_list:
                            _v = _entry_metric(_m, "test", _metric_name)
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
                                _best_split = "test"
                if _best_val is not None:
                    # Tag the model name with "(test fallback)" so the
                    # operator can spot val-vs-test cross-comparisons.
                    _display_name = (
                        f"{_best_name} (test fallback)"
                        if _best_split == "test" else _best_name
                    )
                    _best_metrics[(str(_tt), str(_tname))] = {
                        _pm: _best_val,
                        "model_name": _display_name,
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
        # Cross-target ensemble metrics (stamped by _phase_composite_post_xt_ensemble at the
        # val/test report site). Keyed by (target_type, original_target_name) -> {split_metric: value, model_name: ...}.
        # The verdict picker compares this against the single best model and uses whichever wins.
        _ct_ens_raw = metadata.get("cross_target_ensemble_metrics", {})
        _cross_target_ensemble_metrics: dict[tuple[str, str], dict[str, float]] = {}
        for _tt_str, _by_orig in _ct_ens_raw.items():
            for _orig_tname, _m in _by_orig.items():
                if isinstance(_m, dict) and _m:
                    _cross_target_ensemble_metrics[(str(_tt_str), str(_orig_tname))] = _m
        _summary_text = format_suite_end_summary(
            dummy_baselines_metadata=metadata.get("dummy_baselines", {}),
            failures_metadata=metadata.get("dummy_baselines_failures", {}),
            best_model_metrics_by_target=_best_metrics if _best_metrics else None,
            min_lift=dummy_baselines_config.best_model_min_lift,
            composite_to_raw_target_map=_composite_to_raw if _composite_to_raw else None,
            cross_target_ensemble_metrics=_cross_target_ensemble_metrics or None,
        )
        if _summary_text:
            logger.info(_summary_text)
    except Exception as _db_summary_err:
        logger.warning(
            "[DUMMY_BASELINES] suite-end summary failed: %s",
            _db_summary_err,
        )
