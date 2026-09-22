"""Target-drift diagnostics wiring (PSI heatmap, adversarial validation, calibration-over-time panels).

Carved out of ``diagnostics_dispatch.py``, which was over the 1k-line house limit, the same way
``_diagnostics_dispatch_extra`` was; ``diagnostics_dispatch`` re-exports every public name here. The parent's small
accounting helpers are reached through lazy trampolines, because the parent imports this module at its bottom.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger("mlframe.reporting.diagnostics_dispatch")


def _record(*args, **kwargs):
    """Trampoline to ``diagnostics_dispatch._record``."""
    from .diagnostics_dispatch import _record as _f

    return _f(*args, **kwargs)


def _record_path(*args, **kwargs):
    """Trampoline to ``diagnostics_dispatch._record_path``."""
    from .diagnostics_dispatch import _record_path as _f

    return _f(*args, **kwargs)


def _save_spec(*args, **kwargs):
    """Trampoline to ``diagnostics_dispatch._save_spec``."""
    from .diagnostics_dispatch import _save_spec as _f

    return _f(*args, **kwargs)


def _row_count(*args, **kwargs):
    """Trampoline to ``diagnostics_dispatch._row_count``."""
    from .diagnostics_dispatch import _row_count as _f

    return _f(*args, **kwargs)


def _column_names(*args, **kwargs):
    """Trampoline to ``diagnostics_dispatch._column_names``."""
    from .diagnostics_dispatch import _column_names as _f

    return _f(*args, **kwargs)


def render_calibration_drift_diagnostic(*args, **kwargs):
    """Trampoline to ``diagnostics_dispatch.render_calibration_drift_diagnostic``."""
    from .diagnostics_dispatch import render_calibration_drift_diagnostic as _f

    return _f(*args, **kwargs)


def render_target_acf_diagnostic(*args, **kwargs):
    """Trampoline to ``diagnostics_dispatch.render_target_acf_diagnostic``."""
    from .diagnostics_dispatch import render_target_acf_diagnostic as _f

    return _f(*args, **kwargs)


def _diag_max_features() -> int:
    """``diagnostics_dispatch.DIAG_MAX_FEATURES``, read at call time (the parent imports this module at its bottom)."""
    from .diagnostics_dispatch import DIAG_MAX_FEATURES

    return DIAG_MAX_FEATURES


def render_target_drift_diagnostics(
    *,
    train_frame: Any,
    test_frame: Any,
    val_frame: Any = None,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    task: str = "regression",
    plot_outputs: str = "",
    base_path: str = "",
    metrics_dict: Optional[dict] = None,
    feature_names: Optional[Sequence[str]] = None,
    metric: str = "roc_auc",
    seed: int = 0,
    calibration_drift: bool = True,
    target_acf: bool = True,
    cusum_drift: bool = True,
    adversarial_validation: bool = True,
) -> None:
    """Render the per-target temporal-drift + adversarial-validation diagnostics, each accounted.

    ``psi_heatmap`` + ``residual_vs_time`` + ``metric_over_time`` fire when ``timestamps`` cover the split (same gate as
    the temporal target audit); ``adversarial_validation`` fires when train + test (or train + val) feature frames are
    available AND ``adversarial_validation`` is True. When timestamps cover the split, ``calibration_drift``
    (classification) + ``target_acf`` also emit default-on (both cheap: O(n) warmed njit / FFT-capped). All builders
    cap their own compute, so 100GB frames stay safe (column-view histograms, 200k/side fit) -- EXCEPT
    ``adversarial_validation``, whose own LightGBM classifier fit cost scales with column count, not just row count
    (unbounded by any cap here); set the flag False for very wide frames where that cost dominates.
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if not plot_outputs or not base_path:
        return

    from mlframe.reporting.charts.drift import (
        adversarial_validation as _adversarial_validation_fn,
        metric_over_time,
        psi_heatmap,
        residual_vs_time,
    )

    has_time = timestamps is not None and len(np.asarray(timestamps)) > 0

    # Calendar encodings of the timestamp (day/weekday/hour sin-cos, ...) differ between time buckets and between an
    # earlier and a later split BY CONSTRUCTION; left in, they dominate both the PSI heatmap and the adversarial ranking.
    _calendar: list = []
    if has_time and test_frame is not None:
        try:
            from mlframe.reporting.charts._calendar_features import calendar_feature_names

            _calendar = calendar_feature_names(test_frame, np.asarray(timestamps)[: _row_count(test_frame)], feature_names)
        except Exception:
            logger.debug("calendar-feature detection failed; drift charts keep every feature.", exc_info=True)
    _all_names = list(feature_names) if feature_names is not None else None
    if _calendar:
        if _all_names is None:
            from mlframe.reporting.charts._drift_shared import _frame_columns

            _all_names = [str(n) for n in _frame_columns(test_frame, None)[1]]
        _non_calendar = [n for n in _all_names if str(n) not in set(_calendar)]
        logger.info("drift charts: excluding %d calendar feature(s) derived from the timestamp: %s", len(_calendar), ", ".join(_calendar))
    else:
        _non_calendar = _all_names

    if has_time and test_frame is not None:
        ts = np.asarray(timestamps)
        try:
            spec = _cached_psi_heatmap(psi_heatmap, test_frame, ts, _non_calendar)
            if _calendar:
                spec = _with_caption_note(spec, f"Excluded {len(_calendar)} calendar feature(s) derived from the timestamp (they differ between time buckets by construction): {', '.join(_calendar)}.")
            ok = _save_spec(spec, plot_outputs, base_path + "_psi")
            _record(charts, "psi_heatmap", ok)
            if ok:
                _record_path(charts, base_path + "_psi")
        except Exception:
            logger.exception("diagnostics_dispatch: psi_heatmap failed; continuing.")
            _record(charts, "psi_heatmap", False)

    if has_time and y_true is not None and y_pred is not None:
        ts = np.asarray(timestamps)
        yt = np.asarray(y_true).ravel()
        yp = np.asarray(y_pred).ravel()
        m = min(len(yt), len(yp), len(ts))
        if m > 0:
            if task == "regression":
                try:
                    spec = residual_vs_time(yt[:m], yp[:m], ts[:m])
                    ok = _save_spec(spec, plot_outputs, base_path + "_residual_vs_time")
                    _record(charts, "residual_vs_time", ok)
                    if ok:
                        _record_path(charts, base_path + "_residual_vs_time")
                except Exception:
                    logger.exception("diagnostics_dispatch: residual_vs_time failed; continuing.")
                    _record(charts, "residual_vs_time", False)
                # CUSUM change-point catches a SUSTAINED residual mean shift that per-bucket residual_vs_time misses.
                if cusum_drift:
                    try:
                        from mlframe.reporting.charts.drift import cusum_residual_drift

                        spec = cusum_residual_drift(yt[:m], yp[:m], ts[:m])
                        ok = _save_spec(spec, plot_outputs, base_path + "_cusum_drift")
                        _record(charts, "cusum_drift", ok)
                        if ok:
                            _record_path(charts, base_path + "_cusum_drift")
                    except Exception:
                        logger.exception("diagnostics_dispatch: cusum_drift failed; continuing.")
                        _record(charts, "cusum_drift", False)
            try:
                # Direction from the canonical metric-direction table, not a 2-item
                # allowlist that mislabeled rmse/mae/mape/log_loss/ice/ece/pinball as
                # higher-is-better and inverted the "over time" trend annotation.
                from mlframe.training.metrics_registry import metric_name_higher_is_better
                _dir = metric_name_higher_is_better(metric)
                if _dir is None:
                    # Defaulting an UNKNOWN metric to higher-is-better silently inverts this panel's trend
                    # annotation for every custom error metric, which is the common case for a custom name.
                    # Warn and pick the safer default: most bespoke metric names in this codebase are losses.
                    logger.warning(
                        "metric_over_time: optimisation direction for metric=%r is unknown; assuming "
                        "lower-is-better. Register it via mlframe.training.metrics_registry.register_metric "
                        "to silence this and get the trend annotation right.", metric,
                    )
                higher_is_better = False if _dir is None else _dir
                spec = metric_over_time(yt[:m], yp[:m], ts[:m], metric=metric, higher_is_better=higher_is_better)
                ok = _save_spec(spec, plot_outputs, base_path + "_metric_over_time")
                _record(charts, "metric_over_time", ok)
                if ok:
                    _record_path(charts, base_path + "_metric_over_time")
            except Exception:
                logger.exception("diagnostics_dispatch: metric_over_time failed; continuing.")
                _record(charts, "metric_over_time", False)

            # Calibration drift over time -- classification only (y_pred is the positive-class probability here).
            if calibration_drift and task != "regression":
                render_calibration_drift_diagnostic(
                    y_true=yt[:m], y_score=yp[:m], timestamps=ts[:m],
                    plot_outputs=plot_outputs, base_path=base_path, metrics_dict=metrics_dict,
                )
            # Target serial-dependence ACF/PACF on the time-ordered target.
            if target_acf:
                render_target_acf_diagnostic(
                    y_true=yt[:m], timestamps=ts[:m],
                    plot_outputs=plot_outputs, base_path=base_path, metrics_dict=metrics_dict,
                )

    if adversarial_validation and train_frame is not None and (test_frame is not None or val_frame is not None):
        _render_adversarial_panel(train_frame=train_frame, test_frame=test_frame, val_frame=val_frame, non_calendar=_non_calendar,
                                  calendar=_calendar, plot_outputs=plot_outputs, base_path=base_path, charts=charts, seed=seed)


def _render_adversarial_panel(*, train_frame: Any, test_frame: Any, val_frame: Any, non_calendar: Any, calendar: list,
                              plot_outputs: str, base_path: str, charts: Optional[dict], seed: int) -> None:
    """The train-vs-test separability panel, cached across targets since it depends only on the feature frames."""
    # Imported per call, not at module import: the builder's real home is the patch point tests reach for.
    from mlframe.reporting.charts.drift import adversarial_validation as _adversarial_validation_fn

    try:
        # Its own LightGBM classifier fit cost scales with COLUMN count, not just row count -- unlike
        # every other builder in this dispatcher (all row/histogram capped), this one had no bound on a
        # very wide frame at all. Capped the same way this module's OWN dense-matrix builders already
        # are (DIAG_MAX_FEATURES), by restricting feature_names before the fit rather than after: the
        # underlying frame-reader already narrows to exactly the given names, so no extra frame slicing
        # is needed. Traced to a production profile alongside the (separately fixed) PDP categorical-
        # sweep cost -- the same "cost scales with an unbounded dimension" bug class.
        _adv_names = list(non_calendar) if non_calendar is not None else _column_names(train_frame)
        if _adv_names is not None and len(_adv_names) > _diag_max_features():
            _adv_names = _adv_names[:_diag_max_features()]
        # The adversarial classifier depends only on the feature frames, not on the target: every target of a run
        # (raw and composite alike) re-fitted the same 3-fold LightGBM, ~15 s x 32 targets in one production log.
        _adv_key = _adversarial_cache_key(train_frame, test_frame, val_frame, _adv_names, seed)
        spec = _ADVERSARIAL_CACHE.get(_adv_key) if _adv_key is not None else None
        if spec is None:
            spec = _adversarial_validation_fn(
                train_frame, test_frame if test_frame is not None else val_frame,
                val_frame=val_frame if test_frame is not None else None,
                feature_names=_adv_names, seed=seed,
            )
            if _adv_key is not None:
                while len(_ADVERSARIAL_CACHE) >= 8:
                    _ADVERSARIAL_CACHE.pop(next(iter(_ADVERSARIAL_CACHE)))
                _ADVERSARIAL_CACHE[_adv_key] = spec
        if calendar:
            spec = _with_caption_note(spec, f"Excluded {len(calendar)} calendar feature(s) derived from the timestamp (an earlier and a later period differ in them by construction): {', '.join(calendar)}.")
        ok = _save_spec(spec, plot_outputs, base_path + "_adversarial")
        _record(charts, "adversarial", ok)
        if ok:
            _record_path(charts, base_path + "_adversarial")
    except Exception:
        logger.exception("diagnostics_dispatch: adversarial_validation failed; continuing.")
        _record(charts, "adversarial", False)


_ADVERSARIAL_CACHE: dict = {}

_PSI_CACHE: dict = {}
_PSI_CACHE_LOCK = threading.Lock()


def _cached_psi_heatmap(psi_heatmap, test_frame, ts, feature_names):
    """``psi_heatmap`` for this feature frame, computed once per content.

    Quantile PSI reads only the feature frame and the timestamps, never the target, so every target of a run
    recomputed an identical matrix - 14 of them in one production log, over a feature frame the suite's own
    PipelineCache reported as a single cached entry. Same content key as the adversarial panel beside it; eight
    entries, oldest evicted first.
    """
    key = _psi_cache_key(test_frame, ts, feature_names)
    if key is not None:
        with _PSI_CACHE_LOCK:
            cached = _PSI_CACHE.get(key)
        if cached is not None:
            return cached
    spec = psi_heatmap(test_frame, ts[: _row_count(test_frame)], feature_names=feature_names)
    if key is not None:
        with _PSI_CACHE_LOCK:
            while len(_PSI_CACHE) >= 8:
                _PSI_CACHE.pop(next(iter(_PSI_CACHE)))
            _PSI_CACHE[key] = spec
    return spec


def _psi_cache_key(test_frame: Any, timestamps: Any, names: Any) -> Optional[tuple]:
    """Content key for a PSI heatmap: the frame's signature, the timestamp axis and the feature set.

    The timestamps enter via length plus their endpoints rather than a full hash: the axis that reaches this builder
    is a split's own ordered timestamp column, so two calls sharing a frame signature and those three numbers are
    reading the same rows.
    """
    try:
        from mlframe.training._dataset_cache_fingerprint import compute_signature

        ts = np.asarray(timestamps)
        ts_key = (int(ts.size), str(ts[0]), str(ts[-1])) if ts.size else (0,)
        return (
            compute_signature(test_frame)[:4],
            ts_key,
            tuple(str(n) for n in names) if names is not None else None,
        )
    except Exception:
        return None


def _adversarial_cache_key(train_frame: Any, test_frame: Any, val_frame: Any, names: Any, seed: int) -> Optional[tuple]:
    """Content key for an adversarial-validation figure: frame signatures (columns, shape, row-sample hash) + features."""
    try:
        from mlframe.training._dataset_cache_fingerprint import compute_signature

        sig = tuple(compute_signature(f)[:4] if f is not None else None for f in (train_frame, test_frame, val_frame))
        return sig + (tuple(str(n) for n in names) if names is not None else None, int(seed))
    except Exception:
        return None


def _with_caption_note(spec: Any, note: str) -> Any:
    """Return ``spec`` with ``note`` appended to its caption (FigureSpec is a frozen dataclass)."""
    import dataclasses

    try:
        cap = getattr(spec, "caption", "") or ""
        return dataclasses.replace(spec, caption=(cap + " " + note).strip())
    except Exception:
        return spec
