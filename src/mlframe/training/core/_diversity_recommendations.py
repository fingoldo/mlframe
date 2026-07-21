"""Wires ``mlframe.votenrank.correlation_diversity_ablation.recommend_diversity_additions`` into the
suite's per-target ensembling step as a default-ON, observational diagnostic.

``recommend_diversity_additions`` needs a genuine POOL of already-fitted candidates (OOF predictions +
individual scores) to rank -- it does not need an external model zoo. ``ens_models`` (the members
``_finalize_per_target_ensembling`` just blended via ``score_ensemble``) already IS that pool, so this
reuses it directly: no candidate models outside the suite's own fitted set are required or fetched.

Fires only when every member exposes ``.oof_preds``/``.oof_probs`` + ``.oof_target`` (the same
OOF-preferred convention ``score_gate.select_gate_source_split`` uses for the quality gate) -- OOF is the
only honest surface for a diversity-vs-accuracy tradeoff read, so this silently no-ops (not a WARN; OOF
availability is a caller config choice via ``oof_n_splits``) rather than falling back to val, which the
gate itself only does for the coarse catastrophic-outlier check, not for a reported recommendation.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("mlframe.training.core._phase_train_one_target")


def _member_oof_array(member: Any, is_classification: bool) -> Optional[np.ndarray]:
    """Return the member's 1-D OOF prediction array, or ``None`` when absent/unusable."""
    if is_classification:
        _arr = getattr(member, "oof_probs", None)
        if _arr is None:
            return None
        _arr = np.asarray(_arr, dtype=np.float64)
        if _arr.ndim == 2 and _arr.shape[1] >= 2:
            return _arr[:, 1]
        return _arr.ravel()
    _arr = getattr(member, "oof_preds", None)
    if _arr is None:
        return None
    return np.asarray(_arr, dtype=np.float64).ravel()


def _member_individual_score(member: Any, is_classification: bool, *, oof_fallback: Optional[tuple] = None) -> Optional[float]:
    """Pull a comparable single-model score from ``member.metrics['val']`` (case-insensitive key match).

    ``oof_fallback``, when given as ``(y_true, oof_pred)``, is used to derive a score from the member's
    own OOF surface instead of returning ``None`` (which kills the WHOLE diagnostic, not just this one
    member) when val metrics are unavailable -- e.g. a suite run with ``compute_valset_metrics=False``.
    The fallback keeps the SAME "higher is better" direction the val-derived score already uses: raw RMSE
    for regression (already lower-is-better, matching ``rmse``/``mae``), negated log-loss for
    classification (so it still points higher-is-better, matching ``roc_auc``/``pr_auc``).
    """
    from ._ensemble_chooser import _lookup_metric_ci

    _metrics = getattr(member, "metrics", None)
    if isinstance(_metrics, dict):
        _val = _metrics.get("val")
        if isinstance(_val, dict):
            if is_classification and 1 in _val and isinstance(_val[1], dict):
                _val = _val[1]
            _metric_names = ("roc_auc", "pr_auc") if is_classification else ("rmse", "mae")
            for _name in _metric_names:
                _score = _lookup_metric_ci(_val, _name)
                if _score is not None:
                    try:
                        _f = float(_score)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(_f):
                        return _f
    if oof_fallback is not None:
        _y_oof, _pred_oof = oof_fallback
        _loss = _classification_loss_fn(_y_oof, _pred_oof) if is_classification else _regression_loss_fn(_y_oof, _pred_oof)
        if np.isfinite(_loss):
            return -_loss if is_classification else _loss
    return None


def _classification_loss_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clipped binary log-loss, lower is better -- matches ``diversity_ablation_report``'s ``loss_fn`` contract."""
    _eps = 1e-7
    _p = np.clip(y_pred, _eps, 1.0 - _eps)
    _y = np.asarray(y_true, dtype=np.float64)
    return float(-np.mean(_y * np.log(_p) + (1.0 - _y) * np.log(1.0 - _p)))


def _regression_loss_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE, lower is better."""
    _y = np.asarray(y_true, dtype=np.float64)
    _p = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((_y - _p) ** 2)))


def compute_diversity_recommendations(
    *,
    ens_models: list,
    target_type: Any,
    behavior_config: Any,
    verbose: bool = False,
) -> Optional[list]:
    """Rank the suite's own fitted-but-not-selected members for genuine blend-additive diversity.

    Returns ``recommend_diversity_additions``'s shortlist (list of dicts, possibly empty), or ``None``
    when the diagnostic can't fire (fewer than 2 members, or any member missing OOF preds/target).
    """
    if not bool(getattr(behavior_config, "recommend_diversity_additions_in_leaderboard", True)):
        return None
    if not ens_models or len(ens_models) < 2:
        return None

    from ..configs import TargetTypes

    _is_classification = target_type in (TargetTypes.BINARY_CLASSIFICATION, TargetTypes.MULTICLASS_CLASSIFICATION)
    if target_type == TargetTypes.MULTICLASS_CLASSIFICATION:
        # oof_probs is (n, C>=3) for multiclass; the single-column proxy used elsewhere in this module
        # (and by compute_high_correlation_pairs for the binary/regression case) is not a meaningful
        # diversity signal across classes -- skip rather than silently mis-measure.
        return None

    from .._format import short_model_tag

    _oof_target = None
    _oof_preds: dict = {}
    _individual_scores: dict = {}
    for _m in ens_models:
        _y = getattr(_m, "oof_target", None)
        _arr = _member_oof_array(_m, _is_classification)
        if _y is None or _arr is None:
            return None
        if _oof_target is None:
            _oof_target = np.asarray(_y).ravel()
        elif len(_oof_target) != len(_arr):
            return None
        _score = _member_individual_score(_m, _is_classification, oof_fallback=(np.asarray(_y).ravel(), _arr))
        if _score is None:
            return None
        _tag = short_model_tag(getattr(_m, "model", _m))
        # De-dupe repeated tags (same family, different pre_pipeline) by appending an index -- keys must
        # be unique for oof_preds/individual_scores dicts.
        _name = _tag
        _dupe_i = 1
        while _name in _oof_preds:
            _dupe_i += 1
            _name = f"{_tag}#{_dupe_i}"
        _oof_preds[_name] = _arr
        _individual_scores[_name] = _score

    if _oof_target is None or len(_oof_preds) < 2:
        return None

    from mlframe.votenrank.correlation_diversity_ablation import recommend_diversity_additions

    _loss_fn = _classification_loss_fn if _is_classification else _regression_loss_fn
    _higher_is_better = _is_classification  # roc_auc/pr_auc higher-better; rmse/mae lower-better.

    try:
        _shortlist = recommend_diversity_additions(
            oof_preds=_oof_preds,
            individual_scores=_individual_scores,
            y_true=_oof_target,
            loss_fn=_loss_fn,
            correlation_threshold=float(getattr(behavior_config, "diversity_recommendation_correlation_threshold", 0.85)),
            higher_score_is_better=_higher_is_better,
            min_ablation_improvement=float(getattr(behavior_config, "diversity_recommendation_min_improvement", 0.0)),
            top_k=getattr(behavior_config, "diversity_recommendation_top_k", None),
        )
    except Exception as _div_err:
        logger.warning("diversity_recommendations computation failed: %s", _div_err)
        return None

    if verbose and _shortlist:
        logger.info("diversity recommendations: %d candidate(s) worth adding despite lower individual score", len(_shortlist))
    return _shortlist
