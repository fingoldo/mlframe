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

from ..baselines import format_suite_end_summary
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
        # Composite-target names per target-type, sourced from the SPEC list (always populated for
        # composites), NOT from composite_target_y_scale_metrics -- the latter is skipped when
        # skip_wrap_pass_predict=True, which left ``_is_composite`` False so the verdict fell through to
        # the composite's T-scale (residual) model metric and printed a misleading y-vs-T comparison.
        _composite_names_by_tt: dict[str, set] = {}
        for _tt_str, _by_tn in metadata.get("composite_target_specs", {}).items():
            _names: set = set()
            for _spec_list in (_by_tn or {}).values():
                for _s in _spec_list or []:
                    _nm = _s.get("name") if isinstance(_s, dict) else getattr(_s, "name", None)
                    if _nm:
                        _names.add(_nm)
            _composite_names_by_tt[str(_tt_str)] = _names
        for _tt, _by_name in metadata.get("dummy_baselines", {}).items():
            for _tname, _rep_dict in _by_name.items():
                _pm = _rep_dict.get("primary_metric")
                if not _pm or not _pm.startswith("val_"):
                    continue
                _metric_name = _pm[len("val_") :]
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
                _yscale_by_tt = metadata.get("composite_target_y_scale_metrics", {}).get(str(_tt), {})
                # A composite target HAS a key here (possibly an empty list); a raw target does not.
                _is_composite = _tname in _yscale_by_tt or _tname in _composite_names_by_tt.get(str(_tt), set())
                _yscale_entries = _yscale_by_tt.get(_tname, [])
                _best_val: float | None = None
                _best_name = "-"
                _best_split = None  # "val" or "test" -- track for tag
                _best_model = None  # the raw model picked, whose TEST metric the composite-vs-raw verdict compares on
                if _yscale_entries:
                    for _ye in _yscale_entries:
                        _split_metric = _ye.get("metrics", {}).get("val", {})
                        _v = _split_metric.get(_metric_name)
                        if _v is None or not np.isfinite(_v):
                            continue
                        if _best_val is None or (_is_minimize and _v < _best_val) or (not _is_minimize and _v > _best_val):
                            _best_val = float(_v)
                            _best_name = _ye.get("model_name") or "Composite"
                            _best_split = "val"
                    # y-scale entries may carry only TEST metrics (no finite val); fall back to those before declaring "-".
                    if _best_val is None:
                        for _ye in _yscale_entries:
                            _split_metric = _ye.get("metrics", {}).get("test", {})
                            _v = _split_metric.get(_metric_name)
                            if _v is None or not np.isfinite(_v):
                                continue
                            if _best_val is None or (_is_minimize and _v < _best_val) or (not _is_minimize and _v > _best_val):
                                _best_val = float(_v)
                                _best_name = _ye.get("model_name") or "Composite"
                                _best_split = "test"
                # When y-scale entries are absent OR carry no usable metric, fall through to the T-scale model-list metrics.
                # NOT for composite targets: their model_list metrics are on the T (residual) scale, while the dummy this
                # is compared against is y-scale -- mixing them produced a FALSE "TASK_NON_TRIVIAL_AND_MODELS_HEALTHY"
                # verdict (a residual RMSE of ~1.5 "beat" a y-scale dummy RMSE of ~13 by 9x while the model's actual
                # y-scale R^2 was -146). For a composite with no usable y-scale metric, leave best_model unset so the
                # verdict honestly shows "-" rather than an apples-to-oranges lift.
                if _best_val is None and not _is_composite:
                    # Prefer VAL metrics (aligned with the dummy's val_* primary_metric). Fall back to TEST when no model in the
                    # slot has a val metric: the verdict then tags "(test)" so the operator sees the cross-split comparison, which
                    # is still more informative than "-" (prod: val metrics were unpopulated while Ridge had TEST RMSE=11.63).
                    for _best_split in ("val", "test"):
                        _best_model, _best_val = _best_entry_on(_model_list, _best_split, _metric_name, _is_minimize)
                        if _best_model is not None:
                            _best_name = getattr(_best_model, "model_name", None) or type(getattr(_best_model, "model", _best_model)).__name__
                            break
                if _best_val is not None:
                    # Tag the model name with "(test fallback)" so the
                    # operator can spot val-vs-test cross-comparisons.
                    _display_name = f"{_best_name} (test fallback)" if _best_split == "test" else _best_name
                    _best_metrics[(str(_tt), str(_tname))] = {
                        _pm: _best_val,
                        "model_name": _display_name,
                        f"test_{_metric_name}": _entry_metric(_best_model, "test", _metric_name) if _best_model is not None else None,
                    }
        # composite -> raw target map so the verdict block uses the raw median(y_raw) constant as the trivial baseline
        # (not the inverted-T fake baseline that uses fitted alpha).
        _composite_to_raw: dict[tuple[str, str], str] = {}
        for _tt_str, _by_tname in metadata.get("composite_target_specs", {}).items():
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
        _cvr_text = format_composite_vs_raw_block(models=models, metadata=metadata, best_metrics=_best_metrics, composite_to_raw=_composite_to_raw)
        if _cvr_text:
            logger.info(_cvr_text)
    except Exception as _db_summary_err:
        logger.warning(
            "[DUMMY_BASELINES] suite-end summary failed: %s",
            _db_summary_err,
        )


def _metric_better(a: float, b: float, is_min: bool) -> bool:
    """True when metric value ``a`` beats ``b`` (lower is better when ``is_min``)."""
    return (a < b) if is_min else (a > b)


def _best_entry_on(model_list: Any, split: str, metric_name: str, is_minimize: bool) -> tuple[Any, float | None]:
    """``(entry, value)`` of the model in ``model_list`` with the best finite ``split`` metric, or ``(None, None)``."""
    best, best_v = None, None
    for m in model_list:
        v = _entry_metric(m, split, metric_name)
        if np.isfinite(v) and (best_v is None or (v < best_v if is_minimize else v > best_v)):
            best, best_v = m, float(v)
    return best, best_v


def _get_key_or_str(d: Any, key: Any) -> Any:
    """``d[key]``, else ``d[str(key)]`` (metadata round-trips through JSON stringify keys), else ``None``."""
    if not d:
        return None
    if key in d:
        return d[key]
    return d.get(str(key))


def format_composite_vs_raw_block(*, models: dict, metadata: dict, best_metrics: dict, composite_to_raw: dict) -> str:
    """Answer "did the composite (log / cbrt / residual ...) beat training on raw y?" per composite target.

    One row per composite: its best model by y-scale VAL metric (never selected on test), that metric, the raw-y trivial dummy
    and the best raw-target model of the same original target on the same val metric, the lifts vs both, and the composite's
    test metric alongside. Composite models with no y-scale metric recorded (typically ensembles, which bypass the per-model
    y-scale hook) are listed explicitly instead of being dropped.
    """
    from ..metrics_registry import metric_name_higher_is_better as _mhb

    _dummies = metadata.get("dummy_baselines", {}) or {}
    _yscale = metadata.get("composite_target_y_scale_metrics", {}) or {}
    lines: list[str] = []
    for (_tt, _comp), _raw in composite_to_raw.items():
        _raw_rep = _get_key_or_str(_dummies.get(_tt), _raw)
        _pm = (_raw_rep or {}).get("primary_metric")
        if not _pm or not _pm.startswith("val_"):
            continue
        _metric = _pm[len("val_"):]
        _dir = _mhb(_metric)
        _is_min = True if _dir is None else (not _dir)
        _rows = (_yscale.get(str(_tt)) or {}).get(_comp) or []
        _best = None
        for _r in _rows:
            _v = ((_r.get("metrics") or {}).get("val") or {}).get(_metric)
            if _v is not None and np.isfinite(_v) and (_best is None or _metric_better(float(_v), _best[0], _is_min)):
                _best = (float(_v), _r)
        _have = {_r.get("model_name") for _r in _rows if (_r.get("metrics") or {})}
        _entries = (_get_key_or_str(models, _tt) or {}).get(_comp) or []
        if not _rows and not _entries:
            # Never trained (dropped by the composite budget / honest-gain floor, which the discovery log already lists):
            # a NO_Y_SCALE_METRIC row here read as if the model had been trained and lost its metrics.
            continue
        _missing = [str(getattr(_e, "model_name", None) or type(getattr(_e, "model", _e)).__name__) for _e in _entries
                    if getattr(_e, "model_name", None) not in _have]
        _dummy_val = None
        _strongest = (_raw_rep or {}).get("strongest")
        if _strongest:
            _dummy_val = ((_raw_rep.get("data") or {}).get(_strongest) or {}).get(_pm)
        _raw_best = best_metrics.get((str(_tt), str(_raw))) or {}
        _raw_val = _raw_best.get(_pm)
        _raw_test = _raw_best.get(f"test_{_metric}")
        _raw_name = str(_raw_best.get("model_name", "-"))

        def _lift(ref, val, _is_min=_is_min):
            """Improvement factor of ``val`` over reference ``ref`` (>1 means better); ``None`` when either is missing, non-finite or non-positive."""
            if ref is None or val is None or not np.isfinite(ref) or not np.isfinite(val):
                return None
            if _is_min:
                return ref / val if val > 0 else None
            return val / ref if ref > 0 else None

        _comp_val = _best[0] if _best else None
        _comp_test = (((_best[1].get("metrics") or {}).get("test") or {}).get(_metric)) if _best else None
        _l_dummy = _lift(_dummy_val, _comp_val)
        _l_raw = _lift(_raw_val, _comp_val)
        # The verdict reads TEST for both sides: discovery kept only composites that beat raw on val, so a val verdict is
        # selection-biased toward COMPOSITE_BEATS_RAW. The val lift stays in the table for reference.
        _l_raw_test = _lift(_raw_test, _comp_test)
        if _comp_val is None and _comp_test is None:
            _verdict = "NO_Y_SCALE_METRIC"
        elif not _raw_best:
            _verdict = "NO_RAW_MODEL_TO_COMPARE"
        elif _l_raw_test is None:
            _verdict = "NO_TEST_METRIC_TO_COMPARE"
        elif _l_raw_test > 1.005:
            _verdict = "COMPOSITE_BEATS_RAW"
        elif _l_raw_test < 0.995:
            _verdict = "RAW_BEATS_COMPOSITE"
        else:
            _verdict = "TIE_WITH_RAW"
        _f = lambda v: "-" if v is None or not np.isfinite(v) else f"{v:.4f}"  # noqa: E731
        _fl = lambda v: "-" if v is None else f"{v:.3f}x"  # noqa: E731
        lines.append(
            f"{_comp[:40]:<40} {(str(_best[1].get('model_name')) if _best else '-')[:28]:<28} {_f(_comp_val):>11} {_f(_comp_test):>11} "
            f"{_f(_dummy_val):>11} {_fl(_l_dummy):>9} {_raw_name[:24]:<24} {_f(_raw_val):>11} {_fl(_l_raw):>9} {_f(_raw_test):>11} {_fl(_l_raw_test):>9} {_verdict}"
            + (f"  [no y-scale metric: {', '.join(_missing)}]" if _missing else "")
        )
    if not lines:
        return ""
    header = (
        "[DUMMY_BASELINES] COMPOSITE vs RAW (y-scale; best models picked on VAL, verdict on TEST, val lift for reference)" + chr(10)
        + f"{'composite':<40} {'best_model':<28} {'val':>11} {'test':>11} {'raw_dummy':>11} {'vs_dummy':>9} {'raw_best_model':<24} "
        f"{'raw_val':>11} {'vs_raw':>9} {'raw_test':>11} {'test_lift':>9} verdict"
    )
    return header + chr(10) + chr(10).join(lines)
