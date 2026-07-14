"""Suite-end calibration / conformal helpers carved out of ``_phase_finalize.py``.

Re-imported at the parent's module top so ``finalize_suite`` calls them unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._training_context import TrainingContext

logger = logging.getLogger(__name__)


def _auto_calibrate_on_calib_slice(ctx: "TrainingContext") -> None:
    """Auto-fit post-hoc calibrators for every per-target model that carries a disjoint calib slice.

    Active only when ``TrainingSplitConfig.calib_size > 0`` carved a calib slice (``ctx.calib_idx``) and
    the trainer stamped ``entry.calib_probs`` / ``entry.calib_target`` (base-model predict_proba on the
    carved slice + aligned labels). The slice is leakage-free: carved from train, base model fit on
    train-minus-calib, disjoint from val/test by the splitter's hard asserts. Skips models without a
    stamped calib slice (no-op), so calib_size==0 runs are unaffected.
    """
    _calib_idx = getattr(ctx, "calib_idx", None)
    if _calib_idx is None or len(_calib_idx) == 0:
        return
    from .._calibration_models import calibrate_namespace_model

    _n = 0
    for _ttype, _by_name in (ctx.models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        for _tname, _entries in _by_name.items():
            if not isinstance(_entries, list):
                continue
            for _entry in _entries:
                try:
                    if calibrate_namespace_model(_entry, target_type=_ttype):
                        _n += 1
                except Exception as _cal_err:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
                    logger.warning("[calib] auto-calibration failed for %s/%s: %s", _ttype, _tname, _cal_err)
    if _n and getattr(ctx, "verbose", 0):
        logger.info("[calib] auto-calibrated %d per-target model(s) on the disjoint calib slice.", _n)


def _isotonic_overfit_risk_check(ctx: "TrainingContext") -> None:
    """Opt-in: flag binary isotonic post-hoc calibrators (fit by ``_auto_calibrate_on_calib_slice``) that are
    tracking per-point noise rather than a genuine monotone relationship.

    Reuses the SAME ``(calib_p, calib_y)`` pair ``calibrate_namespace_model`` fit isotonic on (positive-class
    column of ``entry.calib_probs`` vs ``entry.calib_target``), so the risk report describes the calibrator
    actually shipped. Gated by ``TrainingBehaviorConfig.check_isotonic_overfit_risk``; default OFF (no extra
    fit work, no metadata key). Best-effort: a per-model failure is logged and skipped, never aborts finalize.
    """
    import numpy as _np

    from ...calibration.isotonic_risk import isotonic_overfit_risk

    _cfg = getattr(ctx, "behavior_config", None)
    if _cfg is None:
        _root = getattr(ctx, "configs", None)
        _cfg = getattr(_root, "behavior_config", None) if _root is not None else None
    if _cfg is None or not bool(getattr(_cfg, "check_isotonic_overfit_risk", False)):
        return
    _kwargs = dict(getattr(_cfg, "isotonic_risk_kwargs", None) or {})

    out: dict = {}
    for _ttype, _by_name in (ctx.models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        for _tname, _entries in _by_name.items():
            if not isinstance(_entries, list):
                continue
            for _i, _entry in enumerate(_entries):
                _e = _entry[0] if isinstance(_entry, tuple) and _entry else _entry
                _cp = getattr(_e, "calib_probs", None)
                _ct = getattr(_e, "calib_target", None)
                if _cp is None or _ct is None:
                    continue
                _cp = _np.asarray(_cp, dtype=_np.float64)
                _ct = _np.asarray(_ct.values if hasattr(_ct, "values") else _ct, dtype=_np.float64).reshape(-1)
                if _cp.ndim != 2 or _cp.shape[1] != 2 or _cp.shape[0] != _ct.shape[0] or _cp.shape[0] < 2:
                    continue  # only the binary isotonic path ``calibrate_namespace_model`` actually fits
                _pos = _cp[:, 1]
                try:
                    _rep = isotonic_overfit_risk(_pos, _ct, **_kwargs)
                except Exception as _iso_err:
                    logger.warning("[isotonic_risk] check failed for %s/%s: %s", _ttype, _tname, _iso_err)
                    continue
                _rep = {k: v for k, v in _rep.items() if k not in ("isotonic_fit", "platt_fit", "predict")}  # drop non-serializable fitted objects
                _mn = str(getattr(_e, "model_name", None) or f"model_{_i}")
                out[f"{_ttype}/{_tname}/{_mn}"] = _rep
    if out:
        ctx.metadata["isotonic_risk_report"] = out
        if getattr(ctx, "verbose", 0):
            logger.info("[isotonic_risk] stamped overfit-risk report for %d model(s).", len(out))


def _optimize_decision_threshold_on_calib_slice(ctx: "TrainingContext") -> None:
    """Opt-in: fit an optimized binary decision threshold (+ optional per-cohort thresholds / cv stability
    report) on the disjoint calib slice via ``mlframe.calibration.threshold_optimizer.optimize_decision_threshold``.

    Fit on ``entry.calib_probs``/``entry.calib_target`` (leakage-free, disjoint from test by construction),
    never on test. Stores the result into ``metadata["decision_threshold"]``. Gated by
    ``TrainingBehaviorConfig.auto_optimize_threshold``; default OFF (bit-identical no-op).
    """
    import numpy as _np
    from sklearn.metrics import balanced_accuracy_score

    from ...calibration.threshold_optimizer import optimize_decision_threshold

    _cfg = getattr(ctx, "behavior_config", None)
    if _cfg is None:
        _root = getattr(ctx, "configs", None)
        _cfg = getattr(_root, "behavior_config", None) if _root is not None else None
    if _cfg is None or not bool(getattr(_cfg, "auto_optimize_threshold", False)):
        return
    _kwargs = dict(getattr(_cfg, "threshold_optimizer_kwargs", None) or {})
    _metric_fn = _kwargs.pop("metric_fn", None) or (lambda y_true, y_pred: float(balanced_accuracy_score(y_true, y_pred)))

    out: dict = {}
    for _ttype, _by_name in (ctx.models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        for _tname, _entries in _by_name.items():
            if not isinstance(_entries, list):
                continue
            for _i, _entry in enumerate(_entries):
                _e = _entry[0] if isinstance(_entry, tuple) and _entry else _entry
                _cp = getattr(_e, "calib_probs", None)
                _ct = getattr(_e, "calib_target", None)
                if _cp is None or _ct is None:
                    continue
                _cp = _np.asarray(_cp, dtype=_np.float64)
                _ct = _np.asarray(_ct.values if hasattr(_ct, "values") else _ct, dtype=_np.float64).reshape(-1)
                if _cp.ndim != 2 or _cp.shape[1] != 2 or _cp.shape[0] != _ct.shape[0] or _cp.shape[0] < 2:
                    continue  # binary classification only
                _pos = _cp[:, 1]
                try:
                    _rep = optimize_decision_threshold(_ct, _pos, _metric_fn, **_kwargs)
                except Exception as _thr_err:
                    logger.warning("[threshold_optimizer] fit failed for %s/%s: %s", _ttype, _tname, _thr_err)
                    continue
                _rep = {k: v for k, v in _rep.items() if k not in ("thresholds", "scores")}  # drop the full per-candidate sweep; keep the compact summary
                _mn = str(getattr(_e, "model_name", None) or f"model_{_i}")
                out[f"{_ttype}/{_tname}/{_mn}"] = _rep
    if out:
        ctx.metadata["decision_threshold"] = out
        if getattr(ctx, "verbose", 0):
            logger.info("[threshold_optimizer] stamped optimized decision threshold for %d model(s).", len(out))


def _apply_confidence_shrinkage_to_regression(ctx: "TrainingContext") -> None:
    """Opt-in final prediction-shrinkage step for regression targets: pull weakly-discriminative targets'
    test/val predictions toward a neutral value, per ``mlframe.calibration.confidence_shrinkage``.

    Confidence is computed from each model's OOF preds/target (``compute_oof_confidence``); the shrinkage is
    then applied to that same model's test/val predictions in place (``apply_confidence_shrinkage``). Gated by
    ``RegressionCalibrationConfig.apply_confidence_shrinkage``; default OFF (bit-identical no-op).
    """
    import numpy as _np

    from ...calibration.confidence_shrinkage import apply_confidence_shrinkage, compute_oof_confidence

    _cfg = getattr(ctx, "regression_calibration_config", None)
    if _cfg is None:
        _root = getattr(ctx, "configs", None)
        _cfg = getattr(_root, "regression_calibration_config", None) if _root is not None else None
    if _cfg is None or not bool(getattr(_cfg, "apply_confidence_shrinkage", False)):
        return
    _kwargs = dict(getattr(_cfg, "confidence_shrinkage_kwargs", None) or {})
    _segment_ids = _kwargs.pop("segment_ids", None)

    def _arr(v):
        """Coerce a pandas Series/ndarray/None to a flat float64 array, or None if empty/missing."""
        if v is None:
            return None
        a = v.values if hasattr(v, "values") else v
        a = _np.asarray(a, dtype=_np.float64).reshape(-1)
        return a if a.size else None

    entries_by_key: dict = {}
    for _ttype, _by_name in (ctx.models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        for _tname, _entries in _by_name.items():
            if not isinstance(_entries, list):
                continue
            for _i, _entry in enumerate(_entries):
                _e = _entry[0] if isinstance(_entry, tuple) and _entry else _entry
                if getattr(_e, "test_probs", None) is not None:  # regression only
                    continue
                oof_preds = _arr(getattr(_e, "oof_preds", None))
                train_target = _arr(getattr(_e, "train_target", None))
                test_preds = _arr(getattr(_e, "test_preds", None))
                if oof_preds is None or train_target is None or test_preds is None or oof_preds.size != train_target.size:
                    continue
                _mn = str(getattr(_e, "model_name", None) or f"model_{_i}")
                entries_by_key[f"{_ttype}/{_tname}/{_mn}"] = (_e, oof_preds, train_target, test_preds)

    if not entries_by_key:
        return

    confidences: dict = {}
    preds: dict = {}
    for _key, (_e, _oof_preds, _train_target, _test_preds) in entries_by_key.items():
        try:
            confidences[_key] = compute_oof_confidence(_oof_preds, _train_target, segment_ids=_segment_ids)
        except Exception as _conf_err:
            logger.warning("[confidence_shrinkage] OOF confidence failed for %s: %s", _key, _conf_err)
            continue
        preds[_key] = _test_preds

    if not confidences:
        return

    try:
        shrunk = apply_confidence_shrinkage(preds, confidences, **_kwargs)
    except Exception as _shrink_err:
        logger.warning("[confidence_shrinkage] shrinkage failed: %s", _shrink_err)
        return

    applied: dict = {}
    for _key, _shrunk_preds in shrunk.items():
        _e = entries_by_key[_key][0]
        _e.test_preds = _shrunk_preds
        applied[_key] = {"confidence": confidences[_key]}
    if applied:
        ctx.metadata["confidence_shrinkage"] = applied
        if getattr(ctx, "verbose", 0):
            logger.info("[confidence_shrinkage] applied prediction shrinkage to %d regression model(s).", len(applied))


def _conformal_finalize_structure(ctx: "TrainingContext") -> str:
    """Infer the conformal split-structure tag from the suite's split config (best-effort, defaults iid)."""
    from .._conformal_finalize import infer_split_structure

    _sc = getattr(ctx, "split_config", None)
    if _sc is None:
        _root = getattr(ctx, "configs", None)
        _sc = getattr(_root, "split_config", None) if _root is not None else None
    if _sc is None:
        return "iid"
    return infer_split_structure(
        time_column=getattr(_sc, "time_column", None),
        cv_strategy=getattr(_sc, "cv_strategy", None),
        use_groups=bool(getattr(_sc, "use_groups", False)),
        bucket_stratify=bool(getattr(_sc, "bucket_stratify", False)),
        wholeday_splitting=bool(getattr(_sc, "wholeday_splitting", False)),
    )


def _recalibrate_regression_on_calib_slice(ctx: "TrainingContext") -> None:
    """Apply a monotone point recalibration ``g(yhat)~=E[y|yhat]`` to regression models when it measurably helps.

    Default OFF (opt-in): enabled by ``regression_calibration_config.point in {isotonic,linear}`` or the
    ``MLFRAME_REGRESSION_RECALIBRATION`` env var. ``g`` is fit on the disjoint calib slice and applied ONLY
    when a 2-fold-within-calib gain estimate beats ``min_gain`` (honest gate -- never apply a recalibration
    that doesn't help). When applied, the model is wrapped (``ship = g(base.predict)``) and the cached
    per-split predictions are re-stamped to the recalibrated values so the conformal pass (which runs after)
    scores the SHIPPED predictor. Runs before ``_conformal_on_calib_slice``.
    """
    import numpy as _np

    from .._regression_calibration import RecalibratedRegressor, cv2_recalibration_gain, fit_point_recalibrator

    _cfg = getattr(ctx, "regression_calibration_config", None)
    if _cfg is None:
        _root = getattr(ctx, "configs", None)
        _cfg = getattr(_root, "regression_calibration_config", None) if _root is not None else None
    method = str(getattr(_cfg, "point", "off")) if _cfg is not None else "off"
    _env = os.environ.get("MLFRAME_REGRESSION_RECALIBRATION", "").strip().lower()
    if _env in ("isotonic", "linear"):
        method = _env
    if method not in ("isotonic", "linear"):
        return
    min_gain = float(getattr(_cfg, "min_gain", 0.0)) if _cfg is not None else 0.0

    def _arr(v):
        """Coerce a pandas Series/ndarray/None to a flat float64 array, or None if empty/missing."""
        if v is None:
            return None
        a = v.values if hasattr(v, "values") else v
        a = _np.asarray(a, dtype=_np.float64).reshape(-1)
        return a if a.size else None

    applied: dict = {}
    for _ttype, _by_name in (ctx.models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        for _tname, _entries in _by_name.items():
            if not isinstance(_entries, list):
                continue
            for _i, _entry in enumerate(_entries):
                _e = _entry[0] if isinstance(_entry, tuple) and _entry else _entry
                if getattr(_e, "test_probs", None) is not None:  # regression only
                    continue
                cp = _arr(getattr(_e, "calib_preds", None))
                ct = _arr(getattr(_e, "calib_target", None))
                model = getattr(_e, "model", None)
                if cp is None or ct is None or model is None or cp.size != ct.size:
                    continue
                gain = cv2_recalibration_gain(cp, ct, method)
                if gain <= min_gain:
                    continue
                g = fit_point_recalibrator(cp, ct, method)
                try:
                    _e.model = RecalibratedRegressor(model, g)
                except Exception as _wrap_err:
                    logger.warning("[regression_recal] wrap failed for %s/%s: %s", _ttype, _tname, _wrap_err)
                    continue
                # Re-stamp cached preds to the SHIPPED (recalibrated) values so conformal + any later reader agree.
                for _attr in ("test_preds", "val_preds", "train_preds", "oof_preds", "calib_preds"):
                    _v = _arr(getattr(_e, _attr, None))
                    if _v is not None:
                        setattr(_e, _attr, g.transform(_v))
                _mn = str(getattr(_e, "model_name", None) or f"model_{_i}")
                applied[f"{_ttype}/{_tname}/{_mn}"] = {"method": method, "cv2_gain": float(gain), "metrics_are_pre_recalibration": True}
    if applied:
        ctx.metadata["regression_recalibration"] = applied
        if getattr(ctx, "verbose", 0):
            logger.info("[regression_recal] applied monotone recalibration to %d regression model(s).", len(applied))


def _stamp_composite_estimator_recommendation(ctx: "TrainingContext") -> None:
    """Stamp the distribution-driven composite-estimator recommendation into metadata (advisory, E3).

    Reads the analyzer's pathology list from ``metadata["target_distribution_report"]`` and records the
    matching specialised estimator (heavy-tail -> TailComposite, multi-modal -> CompositeDistribution) under
    ``metadata["composite_estimator_recommendation"]``. Advisory only -- it surfaces the dispatch decision;
    auto-applying it (swapping the trained estimator) is a follow-up. No-op when the analyzer did not run.
    """
    report = (ctx.metadata or {}).get("target_distribution_report")
    if not isinstance(report, dict):
        return
    pathologies = report.get("pathologies")
    if not pathologies:
        return
    from ..composite._estimator_dispatch import recommend_composite_estimator

    rec = recommend_composite_estimator(pathologies)
    if rec is not None:
        ctx.metadata["composite_estimator_recommendation"] = rec
        if getattr(ctx, "verbose", 0):
            logger.info("[estimator_recommendation] %s recommended (%s).", rec["estimator"], rec["reason"])


def _conformal_on_calib_slice(ctx: "TrainingContext") -> None:
    """Stamp regression conformal prediction intervals + achieved test coverage per per-target model.

    For each regression entry that carries a test split, build distribution-free intervals from the
    leakage-free calib-slice residuals (split-conformal, >=1-alpha) when present, else from OOF residuals
    (CV+/Jackknife+, >=1-2alpha); measure coverage/width/Winkler on the honest test split. Best-effort and
    additive: it only reads already-stamped arrays and writes ``metadata["conformal"]``, never mutating a
    model, so a run without calib/OOF data is a clean no-op. The compact report drops the per-row interval
    arrays to keep metadata small.
    """
    import numpy as _np

    from .._conformal_finalize import conformal_classification_report, conformal_regression_report

    _cfg = getattr(ctx, "conformal_config", None)
    if _cfg is None:
        _root = getattr(ctx, "configs", None)
        _cfg = getattr(_root, "conformal_config", None) if _root is not None else None
    if _cfg is not None and not bool(getattr(_cfg, "enabled", True)):
        return
    alphas = tuple(getattr(_cfg, "alphas", (0.1, 0.2))) if _cfg is not None else (0.1, 0.2)
    score = str(getattr(_cfg, "score", "normalized")) if _cfg is not None else "normalized"
    cls_mode = str(getattr(_cfg, "classification_mode", "sets_lac")) if _cfg is not None else "sets_lac"
    structure = _conformal_finalize_structure(ctx)

    def _arr(v):
        """Coerce a pandas Series/ndarray/None to a flat float64 array, or None if empty/missing."""
        if v is None:
            return None
        a = v.values if hasattr(v, "values") else v
        a = _np.asarray(a, dtype=_np.float64).reshape(-1)
        return a if a.size else None

    def _raw1d(v):
        """Like _arr but preserves the original dtype (needed for classification labels, not just floats)."""
        if v is None:
            return None
        a = _np.asarray(v.values if hasattr(v, "values") else v).reshape(-1)
        return a if a.size else None

    out: dict = {}
    for _ttype, _by_name in (ctx.models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        for _tname, _entries in _by_name.items():
            if not isinstance(_entries, list):
                continue
            for _i, _entry in enumerate(_entries):
                _e = _entry[0] if isinstance(_entry, tuple) and _entry else _entry
                # Classification entries carry test_probs -> conformal prediction SETS (LAC/APS) from the calib probs.
                if getattr(_e, "test_probs", None) is not None:
                    if cls_mode == "off":
                        continue
                    _cp = getattr(_e, "calib_probs", None)
                    _ct = _raw1d(getattr(_e, "calib_target", None))
                    _tt = _raw1d(getattr(_e, "test_target", None))
                    if _cp is None or _ct is None or _tt is None:
                        continue
                    _tp = _np.asarray(getattr(_e, "test_probs", None), dtype=_np.float64)
                    _cp = _np.asarray(_cp, dtype=_np.float64)
                    if _tp.ndim != 2 or _cp.ndim != 2 or _tp.shape[1] != _cp.shape[1]:
                        continue
                    _classes = getattr(getattr(_e, "model", None), "classes_", None)
                    if _classes is None:
                        _classes = _np.unique(_np.concatenate([_ct, _tt]))
                    _classes = _np.asarray(_classes)
                    if _classes.size != _tp.shape[1]:
                        continue
                    _cset_score = "aps" if "aps" in cls_mode else "lac"
                    try:
                        _rep = conformal_classification_report(
                            test_probs=_tp,
                            test_target=_tt,
                            calib_probs=_cp,
                            calib_target=_ct,
                            classes=_classes,
                            alphas=alphas,
                            score=_cset_score,
                            structure=structure,
                        )
                    except Exception as _cls_err:
                        logger.warning("[conformal] classification sets failed for %s/%s: %s", _ttype, _tname, _cls_err)
                        continue
                    _mn = str(getattr(_e, "model_name", None) or f"model_{_i}")
                    out[f"{_ttype}/{_tname}/{_mn}"] = _rep
                    continue
                y_pred_test = _arr(getattr(_e, "test_preds", None))
                y_true_test = _arr(getattr(_e, "test_target", None))
                if y_pred_test is None or y_true_test is None or y_pred_test.size != y_true_test.size:
                    continue
                calib_preds = _arr(getattr(_e, "calib_preds", None))
                calib_target = _arr(getattr(_e, "calib_target", None))
                oof_preds = _arr(getattr(_e, "oof_preds", None))
                train_target = _arr(getattr(_e, "train_target", None))
                kwargs: dict = {}
                if calib_preds is not None and calib_target is not None and calib_preds.size == calib_target.size:
                    kwargs = dict(residuals_cal=calib_target - calib_preds, y_pred_cal=calib_preds, score=score)
                elif oof_preds is not None and train_target is not None and oof_preds.size == train_target.size:
                    kwargs = dict(oof_residuals=train_target - oof_preds)
                else:
                    continue
                try:
                    rep = conformal_regression_report(
                        y_pred_test=y_pred_test,
                        y_true_test=y_true_test,
                        alphas=alphas,
                        structure=structure,
                        **kwargs,
                    )
                except Exception as _cf_err:
                    logger.warning("[conformal] report failed for %s/%s: %s", _ttype, _tname, _cf_err)
                    continue
                rep.pop("intervals", None)  # drop per-row arrays; keep the compact per-alpha coverage summary
                _mn = str(getattr(_e, "model_name", None) or f"model_{_i}")
                out[f"{_ttype}/{_tname}/{_mn}"] = rep
    if out:
        ctx.metadata["conformal"] = out
        if getattr(ctx, "verbose", 0):
            logger.info("[conformal] stamped intervals + coverage for %d regression model(s).", len(out))
