"""Lean full-suite metric aggregator for per-iteration capture (meta-learning / HPO-from-early-observation).

``compute_all_metrics(y_true, y_score, target_type)`` returns a flat ``dict[str, float]`` of the full metric
suite appropriate to the target type, reusing the SAME fast numba kernels the per-model report assembles -- but
WITHOUT the report's plotting / figure / string-formatting machinery, so it is cheap enough to call every boosting
round / training epoch. The keys mirror the report's headline title-metric set
(``ICE BR_DECOMP ECE CMAEW LL ROC_AUC PR_AUC KS MCC BSS``) plus the precision/recall/F1 block, so downstream
meta-learners see the same feature names the reporting layer uses.

This is a thin delegating aggregator, NOT a reimplementation: every number comes from an existing public metric
function (``fast_aucs``, ``compute_ece_and_brier_decomposition``, ``fast_calibration_metrics``, ``fast_ice_only``,
``fast_brier_score_loss``, ``fast_log_loss``, ``ks_statistic``, ``matthews_corrcoef_*``, ``brier_skill_score_*``,
``fast_classification_report``, the regression ``fast_*`` block). The binary calibration report
(``fast_calibration_report``) is the report-time aggregator, but it is plot-coupled and ~10x heavier per call than
needed here, so per-iteration capture composes the underlying kernels directly (identical numbers, no figure).

Degrades gracefully: a single-class val set, NaN scores, or an empty array yield NaN for the undefined metrics
rather than raising -- per-iteration capture must never abort a training run.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["compute_all_metrics", "CLASSIFICATION_METRIC_KEYS", "REGRESSION_METRIC_KEYS"]

# The flat metric-name surface, fixed per target-type family so meta-learners get stable feature columns across
# iterations even when a metric is NaN on a degenerate round (single-class val, all-equal scores).
_BINARY_METRIC_KEYS = (
    "ICE", "ECE", "Brier_REL", "Brier_RES", "Brier_UNC", "brier_loss", "CMAEW",
    "calibration_std", "calibration_coverage", "log_loss", "ROC_AUC", "PR_AUC", "KS", "MCC", "BSS",
    "precision", "recall", "f1", "accuracy", "balanced_accuracy",
)
_MULTICLASS_METRIC_KEYS = (
    "log_loss", "MCC", "accuracy", "balanced_accuracy",
    "macro_precision", "macro_recall", "macro_f1",
    "weighted_precision", "weighted_recall", "weighted_f1",
    "macro_ROC_AUC", "macro_PR_AUC",
)
REGRESSION_METRIC_KEYS = ("MAE", "MSE", "RMSE", "MaxError", "R2")
CLASSIFICATION_METRIC_KEYS = _BINARY_METRIC_KEYS  # back-compat alias for the common binary case


def _as_1d(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    return a


def _nan_dict(keys) -> dict:
    return {k: float("nan") for k in keys}


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray, nbins: int) -> dict:
    """Full binary metric suite from the fast kernels; NaN-safe on degenerate input."""
    from .core import (
        compute_ece_and_brier_decomposition,
        fast_aucs,
        fast_brier_score_loss,
        fast_calibration_metrics,
        fast_classification_report,
        fast_log_loss,
    )
    from .classification import (
        brier_skill_score,
        fast_ice_only,
        ks_statistic,
        matthews_corrcoef_binary,
    )

    out = _nan_dict(_BINARY_METRIC_KEYS)
    yt = _as_1d(y_true)
    ys = _as_1d(y_score)
    if ys.ndim == 2:  # (n, 2) proba matrix -> positive-class column
        ys = ys[:, -1]
    n = yt.shape[0]
    if n == 0:
        return out
    finite = np.isfinite(ys)
    if not finite.all():
        # Drop non-finite score rows so the kernels (which assume [0,1] probabilities) don't propagate NaN
        # into every metric. A round with all-NaN scores degrades to the all-NaN dict above.
        keep = finite
        yt = yt[keep]
        ys = ys[keep]
        n = yt.shape[0]
        if n == 0:
            return out
    yt_i = yt.astype(np.int64, copy=False)
    n_pos = int(yt_i.sum())
    single_class = n_pos == 0 or n_pos == n

    # Calibration block (ECE / Brier decomposition / CMAEW / ICE) is defined even on a single-class val set.
    ece, rel, res, unc, _brier_binned = compute_ece_and_brier_decomposition(yt_i.astype(np.float64), ys, nbins)
    cmaew, cstd, ccov = fast_calibration_metrics(yt_i.astype(np.float64), ys, nbins)
    brier = float(fast_brier_score_loss(yt_i, ys))
    out.update(
        ICE=float(fast_ice_only(yt_i, ys, nbins=nbins)),
        ECE=float(ece), Brier_REL=float(rel), Brier_RES=float(res), Brier_UNC=float(unc),
        brier_loss=brier, CMAEW=float(cmaew), calibration_std=float(cstd), calibration_coverage=float(ccov),
        log_loss=float(fast_log_loss(yt_i, ys)),
    )
    # Ranking metrics (ROC/PR/KS/BSS) are undefined when y_true is single-class -> leave NaN.
    if not single_class:
        roc_auc, pr_auc = fast_aucs(yt_i, ys)
        out["ROC_AUC"] = float(roc_auc)
        out["PR_AUC"] = float(pr_auc)
        out["KS"] = float(ks_statistic(yt_i, ys))
        out["BSS"] = float(brier_skill_score(yt_i, ys))

    y_pred = (ys >= 0.5).astype(np.int64)
    (_hits, _misses, accuracy, bal_acc, _supports, precisions, recalls, f1s,
     _macro, _weighted) = fast_classification_report(yt_i, y_pred, nclasses=2)
    out["accuracy"] = float(accuracy)
    out["balanced_accuracy"] = float(bal_acc)
    out["precision"] = float(precisions[-1])
    out["recall"] = float(recalls[-1])
    out["f1"] = float(f1s[-1])
    out["MCC"] = float(matthews_corrcoef_binary(yt_i, y_pred))
    return out


def _multiclass_metrics(y_true: np.ndarray, y_score: np.ndarray, n_classes: Optional[int], nbins: int) -> dict:
    """Multiclass suite: log-loss + per-class-averaged AUC + classification report aggregates."""
    from .core import fast_aucs, fast_classification_report
    from .classification import matthews_corrcoef_multiclass

    out = _nan_dict(_MULTICLASS_METRIC_KEYS)
    yt = _as_1d(y_true).astype(np.int64, copy=False)
    ys = np.asarray(y_score, dtype=np.float64)
    n = yt.shape[0]
    if n == 0 or ys.ndim != 2:
        return out
    k = ys.shape[1] if n_classes is None else int(n_classes)
    finite = np.isfinite(ys).all(axis=1)
    if not finite.all():
        yt = yt[finite]
        ys = ys[finite]
        n = yt.shape[0]
        if n == 0:
            return out

    eps = 1e-15
    row_sum = ys.sum(axis=1, keepdims=True)
    probs = np.where(row_sum > 0, ys / np.where(row_sum == 0, 1.0, row_sum), ys)
    clipped = np.clip(probs, eps, 1.0)
    valid = (yt >= 0) & (yt < k)
    if valid.any():
        ll = -np.log(clipped[np.arange(n)[valid], yt[valid]]).mean()
        out["log_loss"] = float(ll)

    y_pred = np.argmax(ys, axis=1).astype(np.int64)
    (_hits, _misses, accuracy, bal_acc, supports, _precisions, _recalls, _f1s,
     macro, weighted) = fast_classification_report(yt, y_pred, nclasses=k)
    out["accuracy"] = float(accuracy)
    out["balanced_accuracy"] = float(bal_acc)
    out["macro_precision"], out["macro_recall"], out["macro_f1"] = (float(macro[0]), float(macro[1]), float(macro[2]))
    out["weighted_precision"], out["weighted_recall"], out["weighted_f1"] = (
        float(weighted[0]), float(weighted[1]), float(weighted[2]))
    out["MCC"] = float(matthews_corrcoef_multiclass(yt, y_pred, n_classes=k))

    roc_aucs, pr_aucs = [], []
    for c in range(k):
        yc = (yt == c).astype(np.int64)
        n_pos = int(yc.sum())
        if n_pos == 0 or n_pos == yc.shape[0]:
            continue
        roc, pr = fast_aucs(yc, ys[:, c])
        if np.isfinite(roc):
            roc_aucs.append(roc)
        if np.isfinite(pr):
            pr_aucs.append(pr)
    if roc_aucs:
        out["macro_ROC_AUC"] = float(np.mean(roc_aucs))
    if pr_aucs:
        out["macro_PR_AUC"] = float(np.mean(pr_aucs))
    return out


def _multilabel_metrics(y_true: np.ndarray, y_score: np.ndarray, nbins: int) -> dict:
    """Multilabel suite: per-label binary metrics, macro-averaged across label columns."""
    yt = np.asarray(y_true)
    ys = np.asarray(y_score)
    if yt.ndim != 2 or ys.shape != yt.shape:
        return _nan_dict(_BINARY_METRIC_KEYS)
    per_label = [_binary_metrics(yt[:, j], ys[:, j], nbins) for j in range(yt.shape[1])]
    out = {}
    for key in _BINARY_METRIC_KEYS:
        vals = [d[key] for d in per_label if np.isfinite(d[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def _regression_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    from .regression import (
        fast_max_error,
        fast_mean_absolute_error,
        fast_mean_squared_error,
        fast_r2_score,
        fast_root_mean_squared_error,
    )

    out = _nan_dict(REGRESSION_METRIC_KEYS)
    yt = _as_1d(y_true).astype(np.float64, copy=False)
    ys = _as_1d(y_score).astype(np.float64, copy=False)
    if yt.shape[0] == 0:
        return out
    keep = np.isfinite(yt) & np.isfinite(ys)
    if not keep.all():
        yt, ys = yt[keep], ys[keep]
        if yt.shape[0] == 0:
            return out
    out["MAE"] = float(fast_mean_absolute_error(yt, ys))
    out["MSE"] = float(fast_mean_squared_error(yt, ys))
    out["RMSE"] = float(fast_root_mean_squared_error(yt, ys))
    out["MaxError"] = float(fast_max_error(yt, ys))
    out["R2"] = float(fast_r2_score(yt, ys))
    return out


def compute_all_metrics(
    y_true,
    y_score,
    target_type: str = "binary_classification",
    *,
    n_classes: Optional[int] = None,
    nbins: int = 10,
) -> dict:
    """Compute the full metric suite for ``target_type`` from val ``(y_true, y_score)`` -- one flat dict of floats.

    ``target_type`` accepts the ``mlframe.training.TargetTypes`` string values (or a bare ``TargetTypes`` member);
    recognised families: binary / multiclass / multilabel classification and (any) regression. Unknown / ranking
    types degrade to the regression block when ``y_score`` is 1-D, else an empty dict (never raises).

    ``y_score`` shapes: 1-D probabilities (binary), ``(n, n_classes)`` proba matrix (multiclass), ``(n, n_labels)``
    indicator-proba matrix (multilabel), 1-D predictions (regression).

    Designed to be called per boosting round / training epoch: pure delegation to the fast kernels, NaN-safe on
    degenerate rounds (single-class val, NaN scores, empty arrays).
    """
    tt = str(target_type)
    if tt.endswith("binary_classification") or tt == "binary":
        return _binary_metrics(y_true, y_score, nbins)
    if tt.endswith("multiclass_classification") or tt == "multiclass":
        return _multiclass_metrics(y_true, y_score, n_classes, nbins)
    if tt.endswith("multilabel_classification") or tt == "multilabel":
        return _multilabel_metrics(y_true, y_score, nbins)
    if "regression" in tt:
        return _regression_metrics(y_true, y_score)
    # Unknown / learning_to_rank: best-effort regression block on a 1-D score, else nothing.
    ys = np.asarray(y_score)
    if ys.ndim == 1 or (ys.ndim == 2 and ys.shape[1] == 1):
        return _regression_metrics(y_true, y_score)
    logger.debug("compute_all_metrics: unsupported target_type=%r with %dD y_score; returning empty dict.", tt, ys.ndim)
    return {}
