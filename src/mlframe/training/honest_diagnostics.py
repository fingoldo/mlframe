"""Honest-estimator diagnostics aggregator (Wave 9 AP13).

Produces four artefacts at suite-finalize time so every run carries the
machine-checkable honest-evaluation trail demanded by the 2026-05-24
ml-best-practices critique:

  1. Bootstrap CI per top-line metric (Brier / AUC / ECE / RMSE / log-loss),
     via :func:`mlframe.evaluation.bootstrap.bootstrap_metric` so the same
     percentile-CI machinery powers every emitted estimate.
  2. Categorical PSI drift summary across train / val / test, via the existing
     :func:`mlframe.training.feature_drift_report.compute_categorical_drift_psi`.
     Surfaces silent new-category levels that destroy calibration in prod.
  3. Reliability / calibration plot via :func:`mlframe.calibration.policy.pick_best_calibrator`
     with ``emit_plot=True`` -- the same auto-pick helper consumed by
     ``post_calibrate_model``, so the report agrees with the calibrator the
     suite actually picked.
  4. Provenance disposition table via
     :func:`mlframe.training.provenance.format_provenance_table`, so a reviewer
     sees the source-split / row-count / seed each producer step touched at a
     glance.

Outputs live under ``metadata["honest_diagnostics"]`` (always populated when the
helper runs) and -- when ``ctx.data_dir + ctx.models_dir`` are set --- also as
``reports/<target>/honest_diagnostics_*.{png,csv,txt}`` on disk so an operator
can hand a single folder to a reviewer.

The aggregator is defensive: any artefact whose source is unavailable (no
test_probs, no oof, missing dep) is recorded with ``status: "skipped"`` + reason
instead of crashing the suite finalize phase.
"""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from os.path import join
from typing import Any, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _safe_arr(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        a = np.asarray(x)
        if a.size == 0:
            return None
        return a
    except Exception as exc:
        logger.debug("honest_diagnostics: _safe_arr coercion failed: %r", exc, exc_info=True)
        return None


def _is_binary_classif(y: np.ndarray) -> bool:
    if y is None:
        return False
    try:
        u = np.unique(y[np.isfinite(y) if y.dtype.kind in "fc" else slice(None)])
    except Exception:
        return False
    return u.size == 2 and set(u.tolist()).issubset({0, 1})


def _derive_seed(master_seed: int, key: str) -> int:
    """Derive a stable per-target/per-block bootstrap seed from the suite master seed.

    Hashing the (master_seed, key) pair gives each target an independent-but-reproducible seed so the whole diagnostics
    run is reproducible from the one master seed, while distinct targets don't share a single fixed-0 seed. Kept in the
    int32 range for sklearn / numpy splitter compatibility.
    """
    h = hashlib.blake2b(f"{int(master_seed)}|{key}".encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "big") % (2**31 - 1)


def _bootstrap_block(
    y_true: np.ndarray, probs: np.ndarray, preds: Optional[np.ndarray] = None, *, rng_seed: int = 0,
) -> dict[str, Any]:
    """Compute bootstrap CIs for the binary top-line metrics that apply to ``(y_true, probs)``."""
    from mlframe.evaluation.bootstrap import bootstrap_metric, bootstrap_metrics

    p = probs
    if p is not None and p.ndim == 2 and p.shape[1] >= 2:
        p_pos = p[:, 1]
    else:
        p_pos = p.ravel() if p is not None else None
    out: dict[str, Any] = {}

    if p_pos is not None and _is_binary_classif(y_true):
        metric_fns: dict = {}
        try:
            # iter295 (2026-05-26): use mlframe's numba-compiled fast metric
            # kernels instead of sklearn for bootstrap. c0028 profile attrib-
            # uted 124.96s (86pct of 145.78s wall) to run_honest_diagnostics;
            # 4 metric_fn calls x 1000 bootstrap resamples x 8 _bootstrap_block
            # invocations = 32k sklearn metric calls each ~4-5ms. The numba
            # kernels are 3.0x / 45.7x / 8.5x faster on roc_auc / brier /
            # log_loss respectively (bench at n=20k, see commit msg). Result
            # is numerically identical for the binary case (mlframe versions
            # are direct numba ports of the sklearn algorithm).
            # iter336 (2026-05-27): use ``fast_roc_auc_unstable`` (quicksort
            # instead of stable sort) inside the bootstrap loop. Bench at
            # c0083 sizes: 2.25x at n=200k, 2.75x at n=20k. Numerically
            # identical when scores are float64 continuous (the dominant
            # case from real ML model outputs); see kernel docstring for
            # the rationale and when to use the stable variant instead.
            from mlframe.metrics.core import (
                fast_brier_score_loss as _fast_brier,
                fast_log_loss as _fast_ll,
                make_bootstrap_auc_resampler,
            )

            # mlframe kernels accept (y_true, y_pred) bound on float64; the
            # bootstrap resampler hands us views of the original arrays so
            # one-time cast in the calling closure is enough (numba dispatcher
            # caches per dtype, so the first resample warms the JIT).
            # iter599: dropped the .astype on _auc -- fast_roc_auc_unstable's
            # underlying njit kernel accepts int y_true natively (``tps +=
            # y_true[i]`` widens automatically). For int labels (the common
            # bootstrap path) astype(np.float64) ALWAYS copies despite copy=
            # False (the flag only saves a copy when dtype matches). Bench
            # n=25k bootstrap-resample size: int64 1.05x, int8 1.19x,
            # float64 1.00x. Bit-equivalent.
            # KEPT .astype on _brier / _ll -- bench showed regressions on
            # float64 labels (0.77x / 0.85x) when kernel was called direct;
            # something about the dispatcher fast-path on (float64,float64)
            # outperforms the bare call. Not worth the float64 regression to
            # win on int paths.
            # Cast y_true / p_pos to float64 ONCE before the 1000-resample loop
            # (below), so the resampled ``y_true[idx]`` / ``p_pos
            # [idx]`` views handed to each metric are already float64. The old
            # per-call ``yy.astype(np.float64, copy=False)`` inside _brier/_ll
            # still COPIED on every int-label resample (copy=False only avoids a
            # copy when dtype already matches) -- ~4000 in-loop copies / block.
            # With the pre-cast the kernels receive float64 directly: bit-
            # identical (same values), and the dispatcher hits the same
            # (float64,float64) fast path the prior code wanted.
            def _brier(yy, pp):
                return float(_fast_brier(yy, pp))

            def _ll(yy, pp):
                return float(_fast_ll(yy, pp))

            metric_fns["brier"] = _brier
            metric_fns["log_loss"] = _ll
            # roc_auc via the INDEX-aware resampler: pre-argsort the base score
            # vector ONCE, then build each of the 1000 resamples' descending order
            # in O(n) (counting-gather over base ranks) instead of a fresh
            # O(n log n) argsort per resample. 1.6x-4.4x on the 1000-bootstrap
            # loop, BIT-IDENTICAL on tie-free float64 scores (dominant case);
            # tied/discrete base auto-falls back to exact argsort in the factory.
            _make_auc_resampler = make_bootstrap_auc_resampler
        except ImportError as exc:
            out["status"] = "skipped"
            out["reason"] = f"mlframe metrics import failed: {exc}"
        # ECE via the policy module's _ece_score (consistent with auto-pick).
        try:
            from mlframe.calibration.policy import _ece_score
            metric_fns["ece"] = lambda yy, pp: _ece_score(yy, pp)
        except Exception as exc:
            out["ece"] = {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
        # Bootstrap roc_auc / brier / log_loss / ece TOGETHER: they share the
        # same y_true / p_pos / seed / stratify, so one resample loop serves all
        # of them. The resample generation + ``(y_true[idx], y_pred[idx])`` slice
        # is the bootstrap's dominant cost; doing it once instead of once per
        # metric is the win. bootstrap_metrics is bit-identical to the prior
        # per-metric bootstrap_metric calls for the same random_state (identical
        # index sequence; a metric raising never advances the RNG).
        if metric_fns:
            # Pre-cast ONCE so resampled views are float64 (see _brier/_ll note):
            # removes ~4000 in-loop per-resample copies. _auc/_ece accept float64
            # natively; stratify uses the original-label y_true for class masks.
            y_true_f64 = np.ascontiguousarray(y_true, dtype=np.float64)
            p_pos_f64 = np.ascontiguousarray(p_pos, dtype=np.float64)
            # Build the index-aware AUC resampler on the same f64 base arrays the
            # loop resamples, so resampler(idx) matches y_true_f64[idx]/p_pos_f64[idx].
            _metric_fns_idx = None
            _resampler_factory = locals().get("_make_auc_resampler")
            if _resampler_factory is not None:
                _metric_fns_idx = {"roc_auc": _resampler_factory(y_true_f64, p_pos_f64)}
            try:
                cis = bootstrap_metrics(
                    y_true_f64, p_pos_f64, metric_fns,
                    n_bootstrap=1000, alpha=0.05, stratify=y_true, random_state=rng_seed,
                    metric_fns_idx=_metric_fns_idx,
                )
            except Exception as exc:
                cis = {name: {"error": f"{type(exc).__name__}: {exc}"} for name in metric_fns}
            for name, ci in cis.items():
                if "error" in ci:
                    out[name] = {"status": "skipped", "reason": ci["error"]}
                else:
                    out[name] = {"point": ci["point"], "ci_lo": ci["lo"], "ci_hi": ci["hi"]}
    elif p_pos is not None and y_true is not None:
        # Regression-ish fallback: RMSE on point-prediction-or-prob-mean.
        try:
            # iter367: route through mlframe.metrics.scoring.fast_rmse so the
            # inner bootstrap loop (1000 resamples) calls the numba single-
            # pass kernel instead of the np.asarray + element-wise difference
            # + np.mean + np.sqrt chain. 37x microbench (889us -> 24us / call
            # at n=100k); c0095 regression bootstrap saves ~10s of metric_fn
            # wall on a 9.8s bootstrap_metric tottime baseline.
            from mlframe.metrics.scoring import fast_rmse as _rmse

            ci = bootstrap_metric(y_true, p_pos, metric_fn=_rmse, n_bootstrap=1000, alpha=0.05, random_state=rng_seed)
            out["rmse"] = {"point": ci["point"], "ci_lo": ci["lo"], "ci_hi": ci["hi"]}
        except Exception as exc:
            out["rmse"] = {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
    return out


def _drift_block(ctx: Any) -> dict[str, Any]:
    """Run categorical PSI drift across the train/val/test trio on ctx."""
    train_df = getattr(ctx, "train_df", None)
    val_df = getattr(ctx, "val_df", None)
    test_df = getattr(ctx, "test_df", None)
    if train_df is None:
        return {"status": "skipped", "reason": "ctx.train_df is None"}
    try:
        from mlframe.training.feature_drift_report import compute_categorical_drift_psi
        psi = compute_categorical_drift_psi(train_df, val_df, test_df)
        return {
            "status": "ok",
            "n_categorical_features": psi.get("n_categorical_features", 0),
            "drift_candidates": psi.get("drift_candidates", []),
            "moderate_threshold": psi.get("moderate_threshold"),
            "high_threshold": psi.get("high_threshold"),
        }
    except Exception as exc:
        logger.warning("honest_diagnostics: categorical PSI drift failed: %s", exc)
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}


def _posthoc_calibrated_flag(model_entry: Any) -> Optional[bool]:
    """Read whether this model's probabilities were held-set post-hoc calibrated (True), eval-metric calibration-trained
    only (False, the tree default), or unknown (None). Stamped by ``_maybe_apply_posthoc_calibration``."""
    for obj in (model_entry, getattr(model_entry, "model", None)):
        if obj is None:
            continue
        flag = getattr(obj, "_mlframe_probs_posthoc_calibrated", None)
        if flag is not None:
            return bool(flag)
    return None


def _calibration_block(model_entry: Any, target_name: str, out_dir: Optional[str], *, rng_seed: int = 0) -> dict[str, Any]:
    """Emit reliability plot + auto-pick verdict for ``model_entry`` when OOF probs are available."""
    _posthoc = _posthoc_calibrated_flag(model_entry)
    oof = getattr(model_entry, "oof_probs", None)
    if oof is None:
        # Some entries expose ``model.oof_probs`` instead of attaching directly on the entry tuple.
        inner = getattr(model_entry, "model", None)
        oof = getattr(inner, "oof_probs", None) if inner is not None else None
    if oof is None:
        return {"status": "skipped", "reason": "no oof_probs on model entry", "probs_posthoc_calibrated": _posthoc}
    oof_arr = _safe_arr(oof)
    if oof_arr is None:
        return {"status": "skipped", "reason": "oof_probs empty / unreadable", "probs_posthoc_calibrated": _posthoc}
    # OOF target: prefer attached attribute, fall back to test_target as poor-but-consistent proxy.
    y = getattr(model_entry, "oof_target", None)
    if y is None:
        y = getattr(model_entry, "test_target", None)
    y_arr = _safe_arr(y)
    if y_arr is None or y_arr.size < 4:
        return {"status": "skipped", "reason": "oof_target absent / too small"}
    # Align row counts (oof_probs are typically train-aligned; truncate to common length).
    n = min(oof_arr.shape[0], y_arr.shape[0])
    if n < 4:
        return {"status": "skipped", "reason": f"aligned row count {n} < 4"}
    oof_arr = oof_arr[:n]
    y_arr = y_arr[:n]
    if not _is_binary_classif(y_arr):
        return {"status": "skipped", "reason": "non-binary target; calibration policy is binary-only"}
    plot_path: Optional[str] = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plot_path = join(out_dir, f"calibration_{target_name}.png")
    try:
        from mlframe.calibration.policy import pick_best_calibrator
        out = pick_best_calibrator(
            probs=None, y=None,
            oof_probs=oof_arr, oof_y=y_arr,
            n_bootstrap=500,
            random_state=rng_seed,
            emit_plot=bool(plot_path),
            plot_path=plot_path,
        )
        return {
            "status": "ok",
            "chosen": out["chosen"],
            "ece_mean": out["ece_mean"],
            "ece_ci": list(out["ece_ci"]),
            "rule": out["rule"],
            "n_oof": out["n_oof"],
            "plot_path": out["plot_path"],
            "alternatives": out["alternatives"],
            "probs_posthoc_calibrated": _posthoc,
        }
    except Exception as exc:
        logger.warning("honest_diagnostics: calibration block failed for %s: %s", target_name, exc)
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}


def _provenance_block(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Format the suite-level provenance trail into a single rendered table."""
    try:
        from mlframe.training.provenance import format_provenance_table, get_provenance
        trail = get_provenance(metadata)
        table = format_provenance_table(metadata)
        return {"status": "ok", "n_steps": len(trail), "table": table, "raw": trail}
    except Exception as exc:
        logger.warning("honest_diagnostics: provenance block failed: %s", exc)
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}


def _resolve_reports_dir(ctx: Any) -> Optional[str]:
    data_dir = getattr(ctx, "data_dir", "") or ""
    models_dir = getattr(ctx, "models_dir", "") or ""
    if not data_dir or not models_dir:
        return None
    try:
        out_dir = join(data_dir, models_dir, "honest_diagnostics")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
    except OSError as exc:
        logger.warning("honest_diagnostics: cannot create reports dir %s: %s", out_dir, exc)
        return None


def _walk_top_models(models: Any) -> list[tuple[str, str, Any]]:
    """Yield ``(target_type_str, target_name, entry)`` tuples for every model entry."""
    out: list[tuple[str, str, Any]] = []
    if not isinstance(models, dict):
        return out
    for tt, by_name in models.items():
        if not isinstance(by_name, dict):
            continue
        for tname, entries in by_name.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                out.append((str(tt), str(tname), entry))
    return out


def run_honest_diagnostics(
    ctx: Any,
    models: Any,
    metadata: Optional[dict] = None,
) -> dict[str, Any]:
    """Produce the four honest-diagnostics artefacts; stamp into ``metadata["honest_diagnostics"]``.

    Parameters
    ----------
    ctx
        TrainingContext-like object with ``train_df`` / ``val_df`` / ``test_df``
        and optional ``data_dir`` + ``models_dir`` for disk artefacts.
    models
        Suite-built ``{target_type: {target_name: [model_entry, ...]}}`` mapping.
        Each entry should expose ``test_target`` + ``test_probs`` for the bootstrap
        block and optionally ``oof_probs`` for the calibration block.
    metadata
        Suite metadata dict; the four artefacts are stamped under
        ``metadata["honest_diagnostics"]``. When ``metadata`` is None a fresh dict
        is created and returned so callers can adopt it.

    Returns
    -------
    dict
        ``metadata["honest_diagnostics"]`` payload, always containing the four
        top-level keys ``bootstrap_ci`` / ``drift_psi`` / ``calibration`` /
        ``provenance`` -- each individual entry's ``status`` reflects whether the
        artefact was emitted or skipped (with reason).
    """
    if metadata is None:
        metadata = {}

    # Suite master seed: derive every per-target bootstrap / calibration seed from it so the whole diagnostics run is
    # reproducible from one seed (previously each block used a fixed 0, so distinct targets shared the same draws).
    _split_cfg = getattr(ctx, "split_config", None)
    master_seed = int(getattr(_split_cfg, "random_seed", 42)) if _split_cfg is not None else 42

    reports_dir = _resolve_reports_dir(ctx)
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bootstrap_ci": {},
        "drift_psi": {},
        "calibration": {},
        "provenance": {},
        "reports_dir": reports_dir,
    }

    # Block 1: bootstrap CI for every top-line metric, per (target_type, target_name, model).
    for tt_str, tname, entry in _walk_top_models(models):
        key = f"{tt_str}/{tname}/{getattr(entry, 'model_name', type(getattr(entry, 'model', entry)).__name__)}"
        y_test = _safe_arr(getattr(entry, "test_target", None))
        p_test = _safe_arr(getattr(entry, "test_probs", None))
        if y_test is None or p_test is None:
            payload["bootstrap_ci"][key] = {"status": "skipped", "reason": "no test_target / test_probs"}
            continue
        try:
            payload["bootstrap_ci"][key] = _bootstrap_block(
                y_test, p_test, getattr(entry, "test_preds", None), rng_seed=_derive_seed(master_seed, key),
            )
        except Exception as exc:
            payload["bootstrap_ci"][key] = {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}

    # Block 2: categorical PSI drift across train/val/test.
    payload["drift_psi"] = _drift_block(ctx)

    # Block 3: calibration reliability + auto-pick verdict, per (target_type, target_name, model).
    for tt_str, tname, entry in _walk_top_models(models):
        key = f"{tt_str}/{tname}/{getattr(entry, 'model_name', type(getattr(entry, 'model', entry)).__name__)}"
        payload["calibration"][key] = _calibration_block(
            entry, target_name=tname, out_dir=reports_dir, rng_seed=_derive_seed(master_seed, key + "/calib"),
        )

    # Block 4: provenance disposition table.
    payload["provenance"] = _provenance_block(metadata)

    metadata["honest_diagnostics"] = payload
    return payload


__all__ = ["run_honest_diagnostics"]
