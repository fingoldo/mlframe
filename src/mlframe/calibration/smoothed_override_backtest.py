"""``backtest_override``: validate a confident-override source on history BEFORE wiring it into
``apply_smoothed_override`` in production.

``apply_smoothed_override`` blends a rule/lookup-derived label into a model's prediction wherever
``override_mask`` fires, at a single caller-supplied blend strength ``a``. That mask is usually built
from some per-row confidence score (e.g. lookup match quality, rule specificity) thresholded by the
caller - but nothing in the module previously helped the caller pick that threshold, or told them
whether the override source is trustworthy at all. An override that's excellent at high confidence but
noisy/wrong at low confidence (the common real-world shape: a fuzzy match degrades gracefully) would
silently hurt accuracy in the low-confidence tail if the caller thresholds too low.

``backtest_override`` takes historical ``(y_true, model_pred, override_pred, confidence)`` tuples,
buckets rows by confidence, and reports blended-vs-model-only error per bucket - so a caller can see
exactly where the override source stops paying for itself, and read off a safe confidence threshold to
feed into ``override_mask = confidence >= safe_threshold`` before calling ``apply_smoothed_override``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mlframe.calibration.smoothed_override import apply_smoothed_override


@dataclass
class ConfidenceBucket:
    """Backtest result for one confidence bucket, ascending by confidence."""

    conf_lo: float
    conf_hi: float
    n: int
    mae_model: float
    mae_blend: float

    @property
    def improvement(self) -> float:
        """Positive means blending helps in this bucket; negative means it hurts."""
        return self.mae_model - self.mae_blend


@dataclass
class OverrideBacktestResult:
    """Per-bucket backtest result for one smoothing/override coefficient ``a``."""

    a: float
    buckets: list[ConfidenceBucket] = field(default_factory=list)
    safe_threshold: float = 1.0
    mae_model_overall: float = 0.0
    mae_blend_all: float = 0.0
    mae_blend_safe: float = 0.0
    # Out-of-sample companion to ``mae_blend_safe``: the threshold is chosen on K-1 folds and the blend scored
    # on the held-out fold, averaged. NaN when the input is too small to split.
    mae_blend_safe_heldout: float = float("nan")

    def summary(self) -> str:
        """Render a human-readable multi-line report of this backtest's buckets and overall MAEs."""
        lines = [f"backtest_override(a={self.a}): safe_threshold={self.safe_threshold:.4f}"]
        lines.extend(
            f"  conf[{b.conf_lo:.3f},{b.conf_hi:.3f}] n={b.n:>6} mae_model={b.mae_model:.4f} mae_blend={b.mae_blend:.4f} improvement={b.improvement:+.4f}"
            for b in self.buckets
        )
        lines.append(
            f"  overall: mae_model={self.mae_model_overall:.4f} mae_blend_all={self.mae_blend_all:.4f} "
            f"mae_blend_safe={self.mae_blend_safe:.4f} (in-sample) mae_blend_safe_heldout={self.mae_blend_safe_heldout:.4f}"
        )
        return "\n".join(lines)


def _confidence_buckets(y_true_arr: np.ndarray, model_arr: np.ndarray, blended_all: np.ndarray, conf_arr: np.ndarray, edges: np.ndarray) -> list:
    """Per-bucket model-vs-blend MAE over the confidence quantile edges."""
    buckets: list = []
    for i in range(edges.size - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == edges.size - 2:
            bucket_mask = (conf_arr >= lo) & (conf_arr <= hi)
        else:
            bucket_mask = (conf_arr >= lo) & (conf_arr < hi)
        n = int(bucket_mask.sum())
        if n == 0:
            continue
        mae_model = float(np.mean(np.abs(y_true_arr[bucket_mask] - model_arr[bucket_mask])))
        mae_blend = float(np.mean(np.abs(y_true_arr[bucket_mask] - blended_all[bucket_mask])))
        buckets.append(ConfidenceBucket(conf_lo=float(lo), conf_hi=float(hi), n=n, mae_model=mae_model, mae_blend=mae_blend))
    return buckets


def _safe_threshold_from(buckets: list) -> float:
    """Lowest confidence above which every bucket still benefits from blending, scanning down from the top.

    A single bad low-confidence bucket does not disqualify a good high-confidence one, but the safe region must
    be contiguous from the top so a caller can express it as one ``confidence >= threshold`` mask.
    """
    safe_threshold = 1.0
    for b in reversed(buckets):
        if b.improvement > 0.0:
            safe_threshold = b.conf_lo
        else:
            break
    return safe_threshold


def _heldout_safe_mae(
    y_true_arr: np.ndarray, model_arr: np.ndarray, blended_all: np.ndarray, conf_arr: np.ndarray, quantiles: np.ndarray, n_folds: int = 5
) -> float:
    """K-fold honest version of ``mae_blend_safe``: pick the threshold on K-1 folds, score on the held-out one.

    ``mae_blend_safe`` is computed on the SAME rows the threshold was scanned from, so it is optimistically
    biased by selection. With an override source that is pure noise every bucket's improvement is positive by
    luck about half the time, and the top-down scan stops at the first non-positive bucket -- so it selects a
    region whose in-sample improvement is positive BECAUSE it was selected. The bias grows with the bucket
    count and shrinks with bucket population; on a rare-event target it easily exceeds the real effect.

    Returns NaN when the input cannot be split into at least two usable folds.
    """
    n = conf_arr.shape[0]
    if n < 2 * n_folds:
        return float("nan")
    from sklearn.model_selection import KFold

    fold_maes: list = []
    for train_idx, test_idx in KFold(n_splits=n_folds, shuffle=True, random_state=0).split(np.arange(n)):
        tr_edges = np.unique(np.quantile(conf_arr[train_idx], quantiles))
        if tr_edges.size < 2:
            continue
        tr_buckets = _confidence_buckets(y_true_arr[train_idx], model_arr[train_idx], blended_all[train_idx], conf_arr[train_idx], tr_edges)
        thr = _safe_threshold_from(tr_buckets)
        te_safe = np.where(conf_arr[test_idx] >= thr, blended_all[test_idx], model_arr[test_idx])
        fold_maes.append(float(np.mean(np.abs(y_true_arr[test_idx] - te_safe))))
    return float(np.mean(fold_maes)) if fold_maes else float("nan")


def backtest_override(
    y_true: np.ndarray,
    model_pred: np.ndarray,
    override_pred: np.ndarray,
    confidence: np.ndarray,
    a: float = 0.9,
    n_buckets: int = 5,
) -> OverrideBacktestResult:
    """Backtest a confident-override source against held-out history, bucketed by confidence.

    Parameters
    ----------
    y_true
        ``(n,)`` ground-truth labels for the historical rows.
    model_pred
        ``(n,)`` the model's own predictions on those rows.
    override_pred
        ``(n,)`` the override rule/lookup's predicted label on those rows (evaluated everywhere, not
        just where it would have fired in production - this is a backtest, so every row is "known").
    confidence
        ``(n,)`` in ``[0, 1]`` - the override source's own confidence/match-quality score per row.
    a
        Blend strength to backtest, passed straight through to ``apply_smoothed_override``.
    n_buckets
        Number of confidence quantile buckets to report (fewer buckets are used if ``confidence`` has
        too few distinct quantile edges, e.g. a mostly-constant confidence score).

    Returns
    -------
    OverrideBacktestResult
        Per-bucket MAE comparison (model-only vs blended) plus ``safe_threshold``: the lowest
        confidence value above which blending measurably beats the model alone in every bucket at or
        above it, scanning from the highest-confidence bucket downward. ``mae_blend_safe`` applies the
        blend only to rows at/above ``safe_threshold``.

        ``mae_blend_safe`` IS IN-SAMPLE: the threshold was chosen by scanning bucket improvements on these
        same rows, so the number is biased by that selection and is not what a caller who thresholds on
        ``safe_threshold`` gets in production. Read ``mae_blend_safe_heldout`` for that -- it picks the
        threshold on K-1 folds and scores the blend on the held-out fold, averaged over folds.
    """
    if n_buckets < 1:
        raise ValueError(f"backtest_override: n_buckets must be >= 1, got {n_buckets}")

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    model_arr = np.asarray(model_pred, dtype=np.float64)
    override_arr = np.asarray(override_pred, dtype=np.float64)
    conf_arr = np.asarray(confidence, dtype=np.float64)

    if not (y_true_arr.shape == model_arr.shape == override_arr.shape == conf_arr.shape):
        raise ValueError("backtest_override: y_true, model_pred, override_pred, confidence must share shape")
    if conf_arr.size == 0:
        raise ValueError("backtest_override: empty input")
    if np.any((conf_arr < 0.0) | (conf_arr > 1.0)):
        raise ValueError("backtest_override: confidence must be in [0, 1]")

    quantiles = np.linspace(0.0, 1.0, n_buckets + 1)
    edges = np.unique(np.quantile(conf_arr, quantiles))
    if edges.size < 2:
        edges = np.array([conf_arr.min(), conf_arr.max()])
        if edges[0] == edges[1]:
            edges[1] = edges[1] + 1e-12

    always_mask = np.ones(conf_arr.shape, dtype=bool)
    blended_all = apply_smoothed_override(model_arr, override_arr, always_mask, a=a)
    buckets = _confidence_buckets(y_true_arr, model_arr, blended_all, conf_arr, edges)
    safe_threshold = _safe_threshold_from(buckets)

    safe_mask = conf_arr >= safe_threshold
    blended_safe = np.where(safe_mask, blended_all, model_arr)

    return OverrideBacktestResult(
        a=a,
        buckets=buckets,
        safe_threshold=safe_threshold,
        mae_model_overall=float(np.mean(np.abs(y_true_arr - model_arr))),
        mae_blend_all=float(np.mean(np.abs(y_true_arr - blended_all))),
        mae_blend_safe=float(np.mean(np.abs(y_true_arr - blended_safe))),
        mae_blend_safe_heldout=_heldout_safe_mae(y_true_arr, model_arr, blended_all, conf_arr, quantiles),
    )


__all__ = ["backtest_override", "OverrideBacktestResult", "ConfidenceBucket"]
