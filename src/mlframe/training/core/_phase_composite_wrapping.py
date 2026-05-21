"""``_run_composite_target_wrapping`` -- composite-target ensemble wrapping.

Wave 100 (2026-05-21): split out from
``training/core/_phase_composite_post.py`` to keep that file below the
1k-line monolith threshold. Behaviour preserved bit-for-bit; the symbol
is re-exported from ``_phase_composite_post`` so existing imports
continue to work.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Wave 100: dependencies needed by the moved _run_composite_target_wrapping.
# _ADDITIVE_TRANSFORMS is defined inside the function body itself, not at
# module scope of any sibling -- the static analyzer flagged it as a missing
# reference but Python resolves it from function-local scope.
from ..composite import CompositeTargetEstimator, get_transform
from .._format import format_metric as _fmt, strip_shim_suffix as _strip

logger = logging.getLogger(__name__)

# Mirror parent's watchdog threshold so the moved function sees the same
# value it did pre-split. Kept in sync with _phase_composite_post.py:48.
_WATCHDOG_RELATIVE_THRESHOLD = 0.01


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
) -> dict[int, np.ndarray]:
    """Wrap T-scale inner models in CompositeTargetEstimator so predict() returns y-scale; record y-scale RMSE/MAE/R2 per split.

    Mutates ``models`` in-place (replaces each composite-target inner with its wrapper) and writes ``metadata["composite_target_y_scale_metrics"]``.
    Returns the train-prediction cache (keyed by ``id(wrapper)``) so the downstream cross-target ensemble block can reuse the predictions
    without re-calling ``.predict`` on the wrapped models.

    ``skip_predict=True`` (Pack 2026-05-18): skip the 3-split predict() calls used to compute y-scale RMSE/MAE/R2 metrics. The wrap step (replacing each entry's inner with ``CompositeTargetEstimator``) still runs so downstream predict-path consumers see y-scale predictions; only the metric computation block is bypassed. Pack G watchdog (additive transforms: T-MAE == y-MAE) already covers the correctness check, so the y-scale metrics are redundant when watchdog is on -- skipping them saves up to ~30 predict() calls on multi-million-row frames per composite target.
    """
    _train_pred_cache: dict[int, np.ndarray] = {}
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
                _inner = getattr(_entry, "model", None) or _entry
                if not hasattr(_inner, "predict"):
                    continue
                # Idempotency: if the entry is ALREADY a CompositeTargetEstimator (re-entry via recover_composite_y_scale_metrics), skip wrap.
                # Double-wrap would treat y-scale predict output as if it were T-scale and invert the transform a second time, producing garbage.
                if isinstance(_inner, CompositeTargetEstimator):
                    continue
                try:
                    # Multi-base specs (linear_residual_multi / future
                    # multi-base transforms) carry extra base columns
                    # alongside the primary. Build the full base_columns
                    # tuple so the wrapper's predict() reconstructs the
                    # (n, K) base matrix to match the K alphas saved in
                    # fitted_params. Without this the wrapper defaults to
                    # base_columns=None and _resolve_base_columns()
                    # falls back to (base_column,) -> predict raises
                    # "base has 1 columns but fitted alphas has K entries".
                    _extra = tuple(_spec.get("extra_base_columns") or ())
                    _base_columns = (
                        (_spec["base_column"], *_extra) if _extra else None
                    )
                    _wrapper = CompositeTargetEstimator.from_fitted_inner(
                        fitted_inner=_inner,
                        transform_name=_spec["transform_name"],
                        base_column=_spec["base_column"],
                        base_columns=_base_columns,
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
                # Preserve auxiliary metadata (columns, model_name, metrics) by replacing inner on entry.
                if hasattr(_entry, "model"):
                    try:
                        _entry.model = _wrapper
                    except Exception:
                        # Read-only attribute: replace the entry itself.
                        _entries[_i] = _wrapper
                else:
                    _entries[_i] = _wrapper
            logger.info(
                "[CompositeTargetEstimator] wrapped %d model(s) for composite "
                "target '%s'; predictions now y-scale.",
                len(_entries), _composite_name,
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
                    "Pack G watchdog covers correctness for additive transforms.",
                    _composite_name,
                )
                continue
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
                        _y_pred_wrapped = np.asarray(
                            _wrapper_for_score.predict(_split_df),
                            dtype=np.float64,
                        ).reshape(-1)
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
                        if _split_name == "train":
                            _train_pred_cache[id(_wrapper_for_score)] = _y_pred
                        _diff = _y_pred - _y_split.astype(np.float64)
                        _finite = np.isfinite(_diff)
                        if _finite.sum() == 0:
                            continue
                        # Zero-variance y => R2 undefined; emit NaN rather than 0.0 to mark the degenerate case.
                        _y_finite = _y_split.astype(np.float64)[_finite]
                        _ss_tot = float(np.sum(
                            (_y_finite - _y_finite.mean()) ** 2
                        ))
                        _ss_res = float(np.sum(
                            _diff[_finite] * _diff[_finite]
                        ))
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
                        # HIGH#4 2026-05-18: watchdog short-circuit when caller disabled it.
                        # The check below does an extra wrapper.predict + inner.predict
                        # per (entry, split) so a wide model zoo can pay 10s+ on 4M-row
                        # frames. Caller passes ``enable_watchdog=False`` to skip.
                        if not enable_watchdog:
                            continue
                        # Pack G runtime watchdog: detects when wrapper math
                        # silently breaks. Pre-fix this covered only additive
                        # transforms via the ``error_T == error_y`` invariant.
                        # The extended check below works for ALL transforms
                        # via the fundamental contract: ``wrapper.predict(X)``
                        # must equal ``transform.inverse(inner.predict(X),
                        # base, params)`` modulo y-clip. If they diverge, the
                        # wrapper's predict path is corrupted (entry-mutation
                        # cache stale, inner double-scaled via TTR state loss,
                        # base column mismatch).
                        #
                        # Additive transforms ALSO get the original
                        # ``error_T == error_y`` check (legacy) since that's
                        # a stricter assertion (the MAE numbers must agree,
                        # not just per-row predictions).
                        try:
                            _spec_t_name = _spec.get("transform_name") if isinstance(_spec, dict) else None
                            _ADDITIVE_TRANSFORMS = {
                                "linear_residual", "linear_residual_robust",
                                "linear_residual_multi", "linear_residual_grouped",
                                "diff", "monotonic_residual", "quantile_residual",
                                "ewma_residual",
                            }
                            # Universal predict-vs-inverse-of-inner check (works for ALL transforms including multiplicative).
                            # If wrapper.predict diverges from manually-reconstructed inverse(inner.predict, base, params), the wrapper math is broken.
                            try:
                                _wi_uni = getattr(_wrapper_for_score, "estimator_", None)
                                _bc_uni = _spec.get("base_column") if isinstance(_spec, dict) else None
                                if (_wi_uni is not None and _spec_t_name
                                        and _bc_uni and _bc_uni in _split_df):
                                    _bivar_uni = get_transform(_spec_t_name)
                                    _base_uni = np.asarray(_split_df[_bc_uni]).astype(np.float64)
                                    _t_pred_uni = np.asarray(
                                        _wi_uni.predict(_split_df), dtype=np.float64,
                                    ).reshape(-1)
                                    _y_reconstructed = _bivar_uni.inverse(
                                        _t_pred_uni, _base_uni,
                                        _spec.get("fitted_params", {}),
                                    )
                                    _ru = _y_pred - _y_reconstructed
                                    _fu = np.isfinite(_ru)
                                    if int(_fu.sum()) > 0:
                                        _max_dev = float(np.max(np.abs(_ru[_fu])))
                                        _y_scale = float(np.std(_y_split[np.isfinite(_y_split)])) or 1.0
                                        _rel = _max_dev / _y_scale
                                        # ``_WATCHDOG_RELATIVE_THRESHOLD`` (module-level) carries the rationale; tune there.
                                        if _rel > _WATCHDOG_RELATIVE_THRESHOLD:
                                            logger.warning(
                                                "[CompositeTargetEstimator.watchdog.universal] "
                                                "composite='%s' split='%s' transform=%s inner=%s: "
                                                "wrapper.predict diverges from "
                                                "transform.inverse(inner.predict, base, params) "
                                                "by max abs=%.4f (%.2f%% of y_std). Wrapper "
                                                "math broken. (additive-error invariant check "
                                                "may also fire below for additive transforms.)",
                                                _composite_name, _split_name, _spec_t_name,
                                                type(_wi_uni).__name__, _max_dev, _rel * 100.0,
                                            )
                            except Exception as _uni_err:
                                logger.debug(
                                    "[CompositeTargetEstimator.watchdog.universal] check "
                                    "failed for composite='%s' split='%s': %s",
                                    _composite_name, _split_name, _uni_err,
                                )
                            if _spec_t_name in _ADDITIVE_TRANSFORMS:
                                _wi = getattr(_wrapper_for_score, "estimator_", None)
                                _base_col = _spec.get("base_column") if isinstance(_spec, dict) else None
                                if _wi is not None and _base_col and _base_col in _split_df:
                                    _bivar = get_transform(_spec_t_name)
                                    _base_arr = np.asarray(_split_df[_base_col]).astype(np.float64)
                                    _t_true = _bivar.forward(
                                        _y_split.astype(np.float64),
                                        _base_arr,
                                        _spec.get("fitted_params", {}),
                                    )
                                    _t_pred = np.asarray(
                                        _wi.predict(_split_df), dtype=np.float64,
                                    ).reshape(-1)
                                    _dt = _t_pred - _t_true
                                    _ft = np.isfinite(_dt)
                                    if int(_ft.sum()) > 0:
                                        _mae_t = float(np.mean(np.abs(_dt[_ft])))
                                        _drel = abs(_mae_t - _mae_wrapped) / max(_mae_t, 1e-9)
                                        if _drel > 0.01:
                                            # Pack #9 diagnostic dump: when the watchdog fires we want enough info in the log to ROOT-CAUSE the divergence on the next production run without re-running. Surface first 5 (y_true, y_pred, T_true, T_pred) tuples + per-inner statistics so the operator can see WHERE the divergence enters: the wrapper math (T_pred vs T_true), the inverse path (T_pred -> y_pred), or the post-clip step.
                                            _n_dbg = min(5, int(_ft.sum()))
                                            _dbg_rows = []
                                            _ft_idx = np.flatnonzero(_ft)[:_n_dbg]
                                            for _i in _ft_idx:
                                                _dbg_rows.append(
                                                    f"y={_y_split[_i]:.4f}, "
                                                    f"y_hat={_y_pred[_i]:.4f}, "
                                                    f"T={_t_true[_i]:.4f}, "
                                                    f"T_hat={_t_pred[_i]:.4f}, "
                                                    f"base={_base_arr[_i]:.4f}"
                                                )
                                            _y_resid_sample = _y_pred[_ft_idx] - _y_split[_ft_idx]
                                            _t_resid_sample = _dt[_ft_idx]
                                            logger.warning(
                                                "[CompositeTargetEstimator.watchdog] "
                                                "composite='%s' split='%s' inner=%s: "
                                                "y-MAE=%.4f diverges from T-MAE=%.4f "
                                                "by %.1f%% (>1%%). Additive-invertible "
                                                "transform should give identical errors. "
                                                "Probable causes: (1) inner.predict NOT "
                                                "returning T-scale (TTR transformer_ "
                                                "state lost via clone/pickle), (2) "
                                                "wrapper.predict double-applies inverse, "
                                                "(3) base column at predict differs from "
                                                "fit. Diagnostic sample of first %d rows: "
                                                "%s. y-residuals first %d: %s; T-residuals "
                                                "first %d: %s.",
                                                _composite_name, _split_name,
                                                type(_wi).__name__,
                                                _mae_wrapped, _mae_t, _drel * 100.0,
                                                _n_dbg, " | ".join(_dbg_rows),
                                                _n_dbg,
                                                ", ".join(f"{v:.4f}" for v in _y_resid_sample.tolist()),
                                                _n_dbg,
                                                ", ".join(f"{v:.4f}" for v in _t_resid_sample.tolist()),
                                            )
                        except Exception as _watchdog_err:
                            logger.debug(
                                "[CompositeTargetEstimator.watchdog] check failed for "
                                "composite='%s' split='%s': %s",
                                _composite_name, _split_name, _watchdog_err,
                            )
                    except Exception as _split_err:
                        # Per-split metric block can fail on shape / predict
                        # mismatch (especially during composite-target rerank
                        # where wrapper.predict may raise on edge geometries).
                        # Log at DEBUG so per-target failures are visible in
                        # verbose logs without spamming WARN -- caller still
                        # gets the model in metadata; just missing this entry's
                        # split metrics.
                        logger.debug(
                            "[composite y-scale metrics] split='%s' composite='%s' skipped: %s",
                            _split_name, _composite_name, _split_err,
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
                            f"{_split_name.upper()}=RMSE_y:{_fmt(_s['RMSE'])} "
                            f"MAE_y:{_fmt(_s['MAE'])} "
                            f"R2_y:{_fmt(_s.get('R2', float('nan')), 4)}"
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
                            "[CompositeTargetEstimator] composite='%s' "
                            "model='%s' y-scale metrics (post-inverse, "
                            "comparable to raw): %s",
                            _composite_name, _mn,
                            " | ".join(_y_summary_parts),
                        )
    return _train_pred_cache
