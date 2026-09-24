"""``_run_composite_target_wrapping`` -- composite-target ensemble wrapping.

split out from
``training/core/_phase_composite_post.py`` to keep that file below the
1k-line monolith threshold. Behaviour preserved bit-for-bit; the symbol
is re-exported from ``_phase_composite_post`` so existing imports
continue to work.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: F401

import numpy as np
from ._prediction_memo import memo_predict

# dependencies needed by the moved _run_composite_target_wrapping.
from ..composite import CompositeTargetEstimator
from .._format import format_metric as _fmt, strip_shim_suffix as _strip
from ._composite_wrap_helpers import build_composite_wrapper
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

from ._composite_wrap_watchdog import (  # noqa: F401  (re-exported: tests and _phase_composite_post read them from here)
    _WATCHDOG_RELATIVE_THRESHOLD,
    _watchdog_base_columns,
    _watchdog_extract_base,
    _is_additive_in_t,
    run_wrap_watchdog,
    watchdog_sample,
)


def _emit_yscale_composite_chart(
    *,
    y_target: np.ndarray,
    y_pred: np.ndarray,
    inner_entry: Any,
    composite_name: str,
    orig_tname: str,
    target_name: str,
    plot_file: str | None,
    reporting_config: Any,
    rmse_y: float,
    mae_y: float,
    r2_y: float,
    split_name: str = "test",
    y_train_mean: float | None = None,
) -> None:
    """Emit a y-scale chart for a composite-target model on one split.

    The chart's ``targets`` and ``preds`` are both already on the raw y scale (wrapper.predict returns y-scale; y_target is
    the original raw target sliced to the split's rows). The in-training chart for a composite model is skipped (its
    metrics are on the transformed T-scale), so THIS is the model's perfplot / residuals / res_dist_and_acf: it is written
    under the model's own chart prefix (``inner_entry.plot_file``, recorded by the trainer) as ``{prefix}_{split}_perfplot``,
    i.e. exactly where and how the raw-target model's charts are named.
    """
    try:
        from ..evaluation import report_regression_model_perf
    except ImportError as e:
        logger.debug("report_regression_model_perf import failed: %s", e)
        return
    if y_target.size == 0 or y_pred.size == 0:
        return
    # Derive a chart-friendly model_name. The original entry's
    # ``model_name`` attribute (if present) carries the inner backend
    # name; fall back to the inner class name.
    _outer = getattr(inner_entry, "model", None) or inner_entry
    from mlframe.training.reporting import display_estimator_name
    _inner_class = display_estimator_name(type(getattr(_outer, "estimator_", _outer)).__name__)
    # Header stats are the split's own mean/std (y_target is the split slice), distinct from the suite's MTTR (TRAIN-split mean) -- label them as such to avoid cross-reading drift.
    _split = str(split_name) if split_name else "test"  # an empty split name labels as "test" too, on purpose
    _mttr = float(np.mean(y_target))
    _mtts = float(np.std(y_target))
    chart_model_name = f"{_inner_class} {target_name} [y-scale] {_split}_mean/{_split}_std={_mttr:.2f}/{_mtts:.2f}"
    _report_title = _split.upper()
    # Same header as the raw-target model's own charts (dates, trained-on rows, @iter, feature count): reuse the title the
    # trainer recorded for this model and swap its T-scale token (MTRESID = mean of the transformed train target) for the
    # y-scale train mean, then let the shared helper append this split's mean exactly as it does for every other chart.
    _native_title = getattr(inner_entry, "chart_model_name", None)
    if _native_title and y_train_mean is not None and np.isfinite(y_train_mean):
        from .._eval_helpers import _MTTR_RE, _append_split_rate_suffix
        from .._format import format_metric as _fmt

        _t = _MTTR_RE.sub(f"MTTR={_fmt(float(y_train_mean))}", str(_native_title), count=1)
        if _MTTR_RE.search(_t):
            _t = _t.replace(f" {target_name} ", f" {target_name} [y-scale] ", 1)
            if "[y-scale]" not in _t:
                _t = _MTTR_RE.sub(lambda m: "[y-scale] " + m.group(0), _t, count=1)
            chart_model_name = _append_split_rate_suffix(_t, split_name=_split, target=y_target)
            _details = (getattr(inner_entry, "chart_split_details", None) or {}).get(_split, "")
            _report_title = " ".join([_split.upper(), _details]).strip()
    # Preferred: the model's own chart prefix (recorded on the entry by the trainer), joined to the split exactly like the
    # regular eval path (_eval_helpers: a trailing os.sep is a directory -> join, else underscore-join). The composite model's
    # T-scale chart is never written, so there is nothing to collide with and no disambiguating suffix is needed.
    _entry_prefix = getattr(inner_entry, "plot_file", None)
    if _entry_prefix:
        import os as _os

        _plot_path = _os.path.join(_entry_prefix, _split) if _entry_prefix.endswith(_os.sep) else f"{_entry_prefix}_{_split}"
    elif plot_file:
        # Fallback for entries without a recorded prefix (e.g. ensemble pseudo-entries) when the caller supplies a base:
        # keep a per-composite, per-split suffix so it cannot overwrite a raw-target chart sharing that base.
        if "." in plot_file:
            _stem, _ext = plot_file.rsplit(".", 1)
            _plot_path = f"{_stem}_yscale_{composite_name}_{_split}.{_ext}"
        else:
            _plot_path = f"{plot_file}_yscale_{composite_name}_{_split}"
    else:
        logger.debug(
            "[CompositeTargetEstimator] no chart path for composite='%s' split=%s (entry has no plot_file, none supplied); chart skipped.",
            composite_name, _split,
        )
        return
    _plot_outputs = getattr(reporting_config, "plot_outputs", None) if reporting_config else None
    _plot_dpi = getattr(reporting_config, "plot_dpi", None) if reporting_config else None
    report_regression_model_perf(
        targets=y_target,
        preds=y_pred,
        columns=(),
        model_name=chart_model_name,
        model=None,
        report_title=_report_title,
        print_report=True,
        show_perf_chart=True,
        plot_file=_plot_path,
        plot_outputs=_plot_outputs,
        plot_dpi=_plot_dpi,
    )
    logger.info(
        "[CompositeTargetEstimator] y-scale chart emitted for "
        "composite='%s' inner=%s (MAE=%.4g RMSE=%.4g R2=%.4f, "
        "n_test=%d)",
        composite_name, _inner_class, mae_y, rmse_y, r2_y, int(y_target.size),
    )


def _record_ensemble_y_scale_metrics(*, entries: list, y_full: np.ndarray, metadata: dict, target_type: Any, composite_name: str, splits: tuple) -> None:
    """y-scale val/test metrics for ensemble entries of a composite target, recorded like a single model's.

    Ensembles carry T-scale ``val_preds`` / ``test_preds`` and no model, so they never got a y-scale metric and never
    entered the composite-vs-raw verdict ("no y-scale metric: EnsARITHM ..." for every composite in a production log).
    Their predictions are mapped to y with a member's wrapper (``predict_from_t``): members share the transform and
    its fitted params, and the wrapper supplies the base column from the split frame.
    """
    _sibling = next(
        (getattr(e, "model", None) for e in entries if callable(getattr(getattr(e, "model", None), "predict_from_t", None))),
        None,
    )
    if _sibling is None:
        return
    for _entry in entries:
        _m = getattr(_entry, "model", None)
        if _m is not None and callable(getattr(_m, "predict", None)):
            continue  # a real model: scored by the per-model hook
        _scores: dict[str, dict[str, float]] = {}
        for _split, _idx, _df in splits:
            _t = getattr(_entry, f"{_split}_preds", None)
            if _idx is None or _df is None or _t is None:
                continue
            _t = np.asarray(_t, dtype=np.float64).reshape(-1)
            _y = np.asarray(y_full, dtype=np.float64)[_idx]
            if _t.shape[0] != _y.shape[0]:
                continue
            try:
                _yp = np.asarray(_sibling.predict_from_t(_df, _t), dtype=np.float64)
            except Exception as e:
                logger.debug("ensemble y-scale mapping failed for %s: %s", getattr(_entry, "model_name", "?"), e)
                continue
            _ok = np.isfinite(_yp) & np.isfinite(_y)
            if not _ok.any():
                continue
            _d = _yp[_ok] - _y[_ok]
            _ss = float(np.sum((_y[_ok] - _y[_ok].mean()) ** 2))
            _scores[_split] = {
                "RMSE": float(np.sqrt(np.mean(_d * _d))), "MAE": float(np.mean(np.abs(_d))),
                "R2": (1.0 - float(np.sum(_d * _d)) / _ss) if _ss > 0 else float("nan"), "n_rows_finite": int(_ok.sum()),
            }
        if _scores:
            record_composite_y_scale_metrics(
                metadata=metadata, target_type=target_type, composite_name=composite_name,
                model_name=getattr(_entry, "model_name", None) or "ensemble", scores=_scores,
            )


def record_composite_y_scale_metrics(*, metadata: dict, target_type: Any, composite_name: str, model_name: Any, scores: dict) -> None:
    """Upsert one model's y-scale split metrics into ``metadata["composite_target_y_scale_metrics"][tt][composite]``.

    Keyed by ``model_name`` so a re-run of the hook for the same model replaces its row instead of duplicating it.
    """
    _rows = metadata.setdefault("composite_target_y_scale_metrics", {}).setdefault(str(target_type), {}).setdefault(composite_name, [])
    _row = {"model_name": model_name, "metrics": {k: dict(v) for k, v in scores.items()}, "source": "per_model_hook"}
    for _i, _r in enumerate(_rows):
        if isinstance(_r, dict) and model_name is not None and _r.get("model_name") == model_name:
            _rows[_i] = _row
            return
    _rows.append(_row)


def _composite_predict(wrapper: Any, df: Any) -> Any:
    """``wrapper.predict(df)``, recovering from CatBoost's polars dispatch miss the way the predict path already does.

    ``CompositeTargetEstimator.predict`` reaches the inner model directly, so it bypasses
    ``_predict_with_fallback``'s recovery. Without this, CatBoost's ``TypeError: No matching signature found`` on a
    nullable-Categorical / Enum polars frame ends the emit: a production run lost the y-scale metrics for six
    composites that way, which are exactly the numbers needed to judge whether a composite beat raw y.
    """
    try:
        return wrapper.predict(df)
    except TypeError as exc:
        if "No matching signature found" not in str(exc):
            raise
        inner = getattr(wrapper, "estimator_", wrapper)
        logger.warning(
            "[CompositeTargetEstimator] CatBoost rejected the polars frame on the y-scale predict (%s); " "converting to pandas and retrying.",
            str(exc).splitlines()[-1][:200],
        )
        from mlframe.training._predict_guards import _cb_polars_to_pandas

        return wrapper.predict(_cb_polars_to_pandas(inner, df, "predict"))


def emit_per_model_composite_y_scale_test(
    *,
    entry: Any,
    composite_spec: dict,
    orig_target_name: str,
    composite_name: str,
    target_name: str,
    y_full: np.ndarray,
    test_idx,
    test_df_pd,
    train_idx=None,
    plot_file: str | None = None,
    reporting_config: Any = None,
    val_idx=None,
    val_df=None,
    metadata: dict | None = None,
    target_type: str | None = None,
    train_df=None,
    group_column: str | None = None,
) -> None:
    """Wrap a freshly-fit composite-target inner model in
    CompositeTargetEstimator (IDEMPOTENT -- safe to call again at end-of-target)
    and emit a TEST-split y-scale chart + log a TEST-split y-scale metric line
    immediately, so composite targets get per-model feedback in the ORIGINAL
    scale right after each fit instead of only at end-of-target.

    Mutates ``entry.model`` (or ``entry`` itself) to the wrapper, matching
    what the end-of-target ``_run_composite_target_wrapping`` does. The
    downstream wrap-pass re-checks idempotency and skips already-wrapped
    entries for the wrap step; its multi-split metric block still runs for
    the comprehensive train/val/test table + watchdog. Never raises -- any
    failure is logged at WARNING and swallowed (training must not crash on
    a reporting hook).
    """
    try:
        if test_idx is None or test_df_pd is None or y_full is None:
            return
        _inner = getattr(entry, "model", None) or entry
        if _inner is None or not hasattr(_inner, "predict"):
            return
        # Idempotent: if already wrapped (e.g. recover_composite_y_scale_metrics
        # re-entry, or the per-model hook ran already), reuse the wrapper.
        if isinstance(_inner, CompositeTargetEstimator):
            _wrapper = _inner
        else:
            _y_full_arr = np.asarray(y_full)
            # y-clip envelope MUST be train-only: this wrapper persists (the
            # end-of-target pass skips already-wrapped entries for idempotency),
            # so a full-y envelope here would leak val/test range into the
            # post-inverse clip and flatter the reported TEST metrics. Slice to
            # the train rows when available; fall back to full y only when the
            # caller cannot supply train_idx (still no leak into the inner --
            # just a wider, conservative envelope).
            if train_idx is not None:
                try:
                    _y_train_arr = _y_full_arr[train_idx]
                except Exception as e:
                    logger.debug("indexing y_full by train_idx failed, using the full array: %s", e)
                    _y_train_arr = _y_full_arr
            else:
                _y_train_arr = _y_full_arr
            _wrapper = build_composite_wrapper(
                entry=entry, inner=_inner, spec=composite_spec, y_train=_y_train_arr,
                train_df=train_df, target_name=orig_target_name, group_column=group_column,
            )
            # Mutate the entry so downstream callers (and the end-of-target
            # wrap-pass idempotency check) see the wrapped form.
            if hasattr(entry, "model"):
                try:
                    entry.model = _wrapper
                except Exception as e:  # nosec B110 - non-trivial body
                    # Read-only attribute -- skip the in-place mutation; the
                    # end-of-target pass will rebuild the wrapper.
                    logger.debug("entry.model assignment failed (likely read-only): %s", e)
        _y_arr = np.asarray(y_full)
        # Inner class name for the log line, matching raw-target reports.
        _inner_for_label = _wrapper.estimator_ if hasattr(_wrapper, "estimator_") else _inner
        from mlframe.training.reporting import display_estimator_name
        _inner_cls = display_estimator_name(type(_inner_for_label).__name__)
        _entry_scores: dict[str, dict[str, float]] = {}
        # VAL + TEST, like a raw-target model's `{prefix}_val_perfplot` / `{prefix}_test_perfplot` pair.
        for _split_name, _split_idx, _split_df in (("val", val_idx, val_df), ("test", test_idx, test_df_pd)):
            if _split_idx is None or _split_df is None:
                continue
            _y_split = _y_arr[_split_idx]
            _y_pred = np.asarray(_composite_predict(_wrapper, _split_df), dtype=np.float64).reshape(-1)
            _finite = np.isfinite(_y_pred) & np.isfinite(_y_split)
            if int(_finite.sum()) == 0:
                continue
            _yt = _y_split.astype(np.float64)[_finite]
            _yp = _y_pred[_finite]
            _diff = _yp - _yt
            _rmse = float(np.sqrt(np.mean(_diff * _diff)))
            _mae = float(np.mean(np.abs(_diff)))
            _ss_tot = float(np.sum((_yt - _yt.mean()) ** 2))
            _r2 = (1.0 - float(np.sum(_diff * _diff)) / _ss_tot) if _ss_tot > 0 else float("nan")
            logger.info(
                "%s %s %s %s [y-scale, per-model immediate] MAE=%.4f RMSE=%.4f R2=%.4f n=%d",
                _split_name.upper(), _inner_cls, target_name, composite_name,
                _mae, _rmse, _r2, int(_finite.sum()),
            )
            _entry_scores[_split_name] = {"RMSE": _rmse, "MAE": _mae, "R2": _r2, "n_rows_finite": int(_finite.sum())}
            _emit_yscale_composite_chart(
                y_target=_yt, y_pred=_yp,
                inner_entry=entry,
                composite_name=composite_name,
                orig_tname=orig_target_name,
                target_name=target_name,
                plot_file=plot_file,
                reporting_config=reporting_config,
                rmse_y=_rmse, mae_y=_mae, r2_y=_r2,
                split_name=_split_name,
            )
        # The end-of-target wrap pass may skip its metric block (skip_wrap_pass_predict=True), in which case these per-model
        # numbers are the ONLY y-scale metrics the suite-end verdict can compare against raw-y models; record them.
        if metadata is not None and target_type is not None and _entry_scores:
            record_composite_y_scale_metrics(
                metadata=metadata, target_type=target_type, composite_name=composite_name,
                model_name=getattr(entry, "model_name", None) or _inner_cls, scores=_entry_scores,
            )
        # Mark the entry so the end-of-target wrap pass skips re-emitting the identical charts (same path -> overwrite + duplicate predict).
        try:
            entry._yscale_chart_emitted = True
        except Exception as e:
            logger.debug("swallowed exception in _phase_composite_wrapping.py: %s", e)
            pass
    except Exception as _err:
        logger.warning(
            "[CompositeTargetEstimator] per-model y-scale emit failed for " "composite='%s' (non-fatal): %s",
            composite_name,
            _err,
        )


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
    skip_predict: bool = False,
    enable_watchdog: bool = True,
    target_name: str | None = None,
    plot_file: str | None = None,
    reporting_config: Any = None,
    group_column: str | None = None,
) -> dict[tuple, np.ndarray]:
    """Wrap T-scale inner models in CompositeTargetEstimator so predict() returns y-scale; record y-scale RMSE/MAE/R2 per split.

    Mutates ``models`` in-place (replaces each composite-target inner with its wrapper) and writes ``metadata["composite_target_y_scale_metrics"]``.
    Returns the train-prediction cache (keyed by ``(id(wrapper), id(filtered_train_df), shape)``) so the downstream cross-target ensemble block can reuse the predictions
    without re-calling ``.predict`` on the wrapped models. Folding the frame identity into the key defends against ``id()`` recycling across GC cycles when wrappers
    or frames get freed between the wrap pass and the ensemble pass on long-lived suites.

    ``skip_predict=True`` (Pack): skip the 3-split predict() calls used to compute y-scale RMSE/MAE/R2 metrics. The wrap step (replacing each entry's inner with ``CompositeTargetEstimator``) still runs so downstream predict-path consumers see y-scale predictions; only the metric computation block is bypassed. Pack G watchdog (additive transforms: T-MAE == y-MAE) already covers the correctness check, so the y-scale metrics are redundant when watchdog is on -- skipping them saves up to ~30 predict() calls on multi-million-row frames per composite target.
    """
    _train_pred_cache: dict[tuple, np.ndarray] = {}
    _train_frame_key = (id(filtered_train_df), getattr(filtered_train_df, "shape", None))
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
                log_throttle(
                    logger, "composite_wrap_missing_original_target", logging.WARNING,
                    "[CompositeTargetEstimator] missing original target '%s' "
                    "in target_by_type for composite='%s'; skipping wrap. "
                    "Predictions will remain in T-scale.",
                    _orig_tname, _composite_name,
                )
                continue
            try:
                _y_train_for_wrap = np.asarray(_y_full)[filtered_train_idx]
            except Exception as _y_err:
                log_throttle(
                    logger, "composite_wrap_cannot_align_y_train", logging.WARNING,
                    "[CompositeTargetEstimator] cannot align y_train for '%s': %s. " "Skipping wrap.",
                    _composite_name,
                    _y_err,
                )
                continue
            if not isinstance(_entries, list):
                continue
            _n_wrapped = 0
            for _i, _entry in enumerate(_entries):
                _inner = getattr(_entry, "model", None) or _entry
                if not hasattr(_inner, "predict"):
                    continue
                # Idempotency: if the entry is ALREADY a CompositeTargetEstimator (re-entry via recover_composite_y_scale_metrics), skip wrap.
                # Double-wrap would treat y-scale predict output as if it were T-scale and invert the transform a second time, producing garbage.
                if isinstance(_inner, CompositeTargetEstimator):
                    continue
                try:
                    # Multi-base specs (linear_residual_multi / future multi-base transforms) carry extra base columns alongside the
                    # primary; the builder passes the full base_columns tuple so predict() reconstructs the (n, K) base matrix matching
                    # the K alphas in fitted_params (else it raises "base has 1 columns but fitted alphas has K entries").
                    _wrapper = build_composite_wrapper(
                        entry=_entry, inner=_inner, spec=_spec, y_train=_y_train_for_wrap,
                        train_df=filtered_train_df, target_name=_orig_tname, group_column=group_column,
                    )
                except Exception as _wrap_err:
                    log_throttle(
                        logger, "composite_wrap_failed", logging.WARNING,
                        "[CompositeTargetEstimator] wrap failed for '%s' (entry %d): %s. "
                        "Predictions will remain in T-scale.",
                        _composite_name, _i, _wrap_err,
                    )
                    continue
                # Preserve auxiliary metadata (columns, model_name, metrics) by replacing inner on entry.
                if hasattr(_entry, "model"):
                    try:
                        _entry.model = _wrapper
                    except Exception as e:
                        # Read-only attribute: replace the entry itself.
                        logger.debug("_entry.model assignment failed (likely read-only), replacing the entry instead: %s", e)
                        _entries[_i] = _wrapper
                        _n_wrapped += 1
                    else:
                        _n_wrapped += 1
                else:
                    _entries[_i] = _wrapper
                    _n_wrapped += 1
            logger.info(
                "[CompositeTargetEstimator] wrapped %d model(s) for composite " "target '%s'; predictions now y-scale.",
                _n_wrapped,
                _composite_name,
            )
            if metadata is not None:
                _y_full_ens = target_by_type.get(_tt_w, {}).get(_orig_tname)
                if _y_full_ens is not None:
                    _record_ensemble_y_scale_metrics(
                        entries=_entries, y_full=np.asarray(_y_full_ens), metadata=metadata, target_type=_tt_w,
                        composite_name=_composite_name,
                        splits=(("val", filtered_val_idx, filtered_val_df), ("test", test_idx, test_df_pd)),
                    )
            # Compute y-scale RMSE/MAE/R2 per split so composite is comparable to raw (per-target metrics were T-scale).
            # ``skip_predict``: bypass the per-split predict + metric block; wrap step above already ran so downstream
            # predict-path callers see y-scale predictions. Pack G watchdog on additive transforms (T-MAE == y-MAE) is
            # the correctness gate; the y-scale numbers here would just restate what the T-scale metrics already say.
            if skip_predict:
                logger.info(
                    "[CompositeTargetEstimator] composite='%s': wrap done, "
                    "y-scale metric block SKIPPED (skip_wrap_pass_predict=True). "
                    "T-scale metrics already in the per-target training log; "
                    "the watchdog checks a val sample.",
                    _composite_name,
                )
                _y_full_wd = target_by_type.get(_tt_w, {}).get(_orig_tname)
                if enable_watchdog and _y_full_wd is not None and filtered_val_idx is not None and filtered_val_df is not None:
                    _wd_df, _wd_y = watchdog_sample(filtered_val_df, np.asarray(_y_full_wd)[filtered_val_idx])
                    for _entry in _entries:
                        _wd_model = getattr(_entry, "model", None) or _entry
                        if callable(getattr(_wd_model, "predict", None)):
                            run_wrap_watchdog(_wd_model, _spec, _wd_df, _wd_y, composite_name=_composite_name, split_name="val")
                # Even when the heavy multi-split metric block is skipped,
                # emit a SINGLE test-split y-scale chart per composite entry
                # so the operator gets the chart the user asked for.
                # Cost: one wrapper.predict(test_df)
                # per entry (~0.1s booster, ~5s MLP). Cheap relative to the
                # full 3-split metric block (~5-15 min).
                if target_name is not None and test_idx is not None and test_df_pd is not None:
                    _y_full_chart = target_by_type.get(_tt_w, {}).get(_orig_tname)
                    if _y_full_chart is not None:
                        _y_arr_chart = np.asarray(_y_full_chart)
                        for _entry in _entries:
                            # The per-model hook already emitted this entry's test chart; skip to avoid a duplicate predict + overwrite of the same _yscale_{composite} file.
                            if getattr(_entry, "_yscale_chart_emitted", False):
                                continue
                            try:
                                _wrap_chart = getattr(_entry, "model", None) or _entry
                                if not callable(getattr(_wrap_chart, "predict", None)):
                                    # Ensemble pseudo-entries carry predictions, not a model: there is nothing to predict with
                                    # (a production log warned "'SimpleNamespace' object has no attribute 'predict'" per entry).
                                    continue
                                _y_split_chart = _y_arr_chart[test_idx]
                                _y_pred_chart = np.asarray(
                                    _wrap_chart.predict(test_df_pd),
                                    dtype=np.float64,
                                ).reshape(-1)
                                _finite_chart = np.isfinite(_y_pred_chart) & np.isfinite(_y_split_chart)
                                if _finite_chart.sum() == 0:
                                    continue
                                _y_t = _y_split_chart[_finite_chart]
                                _y_p = _y_pred_chart[_finite_chart]
                                _diff = _y_p - _y_t.astype(np.float64)
                                _rmse_c = float(np.sqrt(np.mean(_diff * _diff)))
                                _mae_c = float(np.mean(np.abs(_diff)))
                                _ss_tot_c = float(np.sum((_y_t - _y_t.mean()) ** 2))
                                _r2_c = (1.0 - float(np.sum(_diff * _diff)) / _ss_tot_c) if _ss_tot_c > 0 else float("nan")
                                _emit_yscale_composite_chart(
                                    y_target=_y_t,
                                    y_pred=_y_p,
                                    inner_entry=_entry,
                                    composite_name=_composite_name,
                                    orig_tname=_orig_tname,
                                    target_name=target_name,
                                    plot_file=plot_file,
                                    reporting_config=reporting_config,
                                    rmse_y=_rmse_c, mae_y=_mae_c, r2_y=_r2_c,
                                )
                            except Exception as _chart_err:
                                log_throttle(
                                    logger, "composite_wrap_yscale_chart_emit_failed_skip_predict", logging.WARNING,
                                    "[CompositeTargetEstimator] y-scale chart " "emit failed for composite='%s' (non-fatal): %s",
                                    _composite_name,
                                    _chart_err,
                                )
                continue
            _metrics_dict = metadata.setdefault(
                "composite_target_y_scale_metrics", {},
            ).setdefault(str(_tt_w), {}).setdefault(_composite_name, [])
            # Re-scored below per real model; ensemble rows (no model, scored by _record_ensemble_y_scale_metrics) stay.
            _ens_names = {getattr(e, "model_name", None) for e in _entries if not callable(getattr(getattr(e, "model", None), "predict", None))}
            _metrics_dict[:] = [row for row in _metrics_dict if row.get("model_name") in _ens_names]
            _y_full_metric = target_by_type.get(_tt_w, {}).get(_orig_tname)
            if _y_full_metric is None:
                continue
            _y_arr_metric = np.asarray(_y_full_metric)
            for _entry in _entries:
                _wrapper_for_score = getattr(_entry, "model", None) or _entry
                if not callable(getattr(_wrapper_for_score, "predict", None)):
                    continue  # ensemble pseudo-entry without a model (see the chart loop above)
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
                        _y_pred_wrapped = memo_predict(_wrapper_for_score, _split_df)
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
                            _pairs = ", ".join(f"({_y_pred[_i]:.3f}, {_y_split[_i]:.3f})" for _i in range(_n_dbg))
                            _outer_dbg = getattr(_entry, "model", None) or _entry
                            _inner_dbg = getattr(_outer_dbg, "base_estimator", None) or getattr(_outer_dbg, "estimator_", None) or _outer_dbg
                            logger.debug(
                                "[CompositeTargetEstimator.diag] inner=%s split=%s sample(y_hat, y_true) = %s",
                                type(_inner_dbg).__name__, _split_name, _pairs,
                            )
                        if _split_name == "train":
                            _train_pred_cache[(id(_wrapper_for_score), *_train_frame_key)] = _y_pred
                            # Inner-model key too: composite_post.py reads via ``getattr(comp, 'model', comp)`` which unwraps one level.
                            _inner_for_write = getattr(_wrapper_for_score, "model", None)
                            if _inner_for_write is not None and _inner_for_write is not _wrapper_for_score:
                                _train_pred_cache[(id(_inner_for_write), *_train_frame_key)] = _y_pred
                        _diff = _y_pred - _y_split.astype(np.float64)
                        _finite = np.isfinite(_diff)
                        if _finite.sum() == 0:
                            continue
                        # Zero-variance y => R2 undefined; emit NaN rather than 0.0 to mark the degenerate case.
                        _y_finite = _y_split.astype(np.float64)[_finite]
                        _ss_tot = float(np.sum((_y_finite - _y_finite.mean()) ** 2))
                        _ss_res = float(np.sum(_diff[_finite] * _diff[_finite]))
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
                        # emit a Y-SCALE
                        # chart for composite models on the TEST split
                        # so it is directly comparable to raw-target
                        # charts (same MTTR/MTTS units, same scatter
                        # axes). The T-scale residual chart in
                        # ``_reporting_regression`` is skipped exactly
                        # to make room for this y-scale chart.
                        if (
                            _split_name in ("val", "test")
                            and target_name is not None
                            and not getattr(_entry, "_yscale_chart_emitted", False)  # per-model hook already wrote these
                        ):
                            try:
                                _emit_yscale_composite_chart(
                                    y_target=_y_split.astype(np.float64)[_finite],
                                    y_pred=_y_pred[_finite],
                                    inner_entry=_entry,
                                    composite_name=_composite_name,
                                    orig_tname=_orig_tname,
                                    target_name=target_name,
                                    plot_file=plot_file,
                                    reporting_config=reporting_config,
                                    rmse_y=_rmse_wrapped, mae_y=_mae_wrapped, r2_y=_r2,
                                    split_name=_split_name,
                                    y_train_mean=(
                                        float(np.nanmean(_y_arr_metric[filtered_train_idx].astype(np.float64)))
                                        if filtered_train_idx is not None and len(filtered_train_idx)
                                        else None
                                    ),
                                )
                            except Exception as _chart_err:
                                log_throttle(
                                    logger, "composite_wrap_yscale_chart_emit_failed", logging.WARNING,
                                    "[CompositeTargetEstimator] y-scale chart " "emit failed for composite='%s' (non-fatal): %s",
                                    _composite_name,
                                    _chart_err,
                                )
                        # Independent-oracle watchdog (see ``_composite_wrap_watchdog``); ``enable_watchdog=False`` skips its extra predicts.
                        if enable_watchdog:
                            run_wrap_watchdog(_wrapper_for_score, _spec, _split_df, _y_split, composite_name=_composite_name, split_name=_split_name)
                    except Exception as _split_err:
                        # A composite whose predict raises would otherwise vanish from the y-scale verdict with only a DEBUG line;
                        # the model stays in metadata, but its missing split metrics must be visible.
                        log_throttle(
                            logger, "composite_yscale_split_metrics_failed", logging.WARNING,
                            "[composite y-scale metrics] split='%s' composite='%s' skipped: %s: %s",
                            _split_name, _composite_name, type(_split_err).__name__, _split_err,
                        )
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
                            f"{_split_name.upper()}=RMSE_y:{_fmt(_s['RMSE'])} " f"MAE_y:{_fmt(_s['MAE'])} " f"R2_y:{_fmt(_s.get('R2', float('nan')), 4)}"
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
                            "[CompositeTargetEstimator] composite='%s' " "model='%s' y-scale metrics (post-inverse, " "comparable to raw): %s",
                            _composite_name,
                            _mn,
                            " | ".join(_y_summary_parts),
                        )
    return _train_pred_cache
