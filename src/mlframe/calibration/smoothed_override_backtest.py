"""``backtest_override``: validate a confident-override source on history BEFORE wiring it into
``apply_smoothed_override`` in production.

``apply_smoothed_override`` blends a rule/lookup-derived label into a model's prediction wherever
``override_mask`` fires, at a single caller-supplied blend strength ``a``. That mask is usually built
from some per-row confidence score (e.g. lookup match quality, rule specificity) thresholded by the
caller -- but nothing in the module previously helped the caller pick that threshold, or told them
whether the override source is trustworthy at all. An override that's excellent at high confidence but
noisy/wrong at low confidence (the common real-world shape: a fuzzy match degrades gracefully) would
silently hurt accuracy in the low-confidence tail if the caller thresholds too low.

``backtest_override`` takes historical ``(y_true, model_pred, override_pred, confidence)`` tuples,
buckets rows by confidence, and reports blended-vs-model-only error per bucket -- so a caller can see
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

    def summary(self) -> str:
        """Render a human-readable multi-line report of this backtest's buckets and overall MAEs."""
        lines = [f"backtest_override(a={self.a}): safe_threshold={self.safe_threshold:.4f}"]
        lines.extend(
            f"  conf[{b.conf_lo:.3f},{b.conf_hi:.3f}] n={b.n:>6} mae_model={b.mae_model:.4f} mae_blend={b.mae_blend:.4f} improvement={b.improvement:+.4f}"
            for b in self.buckets
        )
        lines.append(f"  overall: mae_model={self.mae_model_overall:.4f} mae_blend_all={self.mae_blend_all:.4f} mae_blend_safe={self.mae_blend_safe:.4f}")
        return "\n".join(lines)


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
        just where it would have fired in production -- this is a backtest, so every row is "known").
    confidence
        ``(n,)`` in ``[0, 1]`` -- the override source's own confidence/match-quality score per row.
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
        blend only to rows at/above ``safe_threshold``, matching what a caller who thresholds on
        ``safe_threshold`` would actually get in production.
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

    buckets: list[ConfidenceBucket] = []
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

    # Scan from the highest-confidence bucket downward; the safe threshold is the lowest confidence
    # value for which every bucket at/above it still benefits from blending (a single bad low-confidence
    # bucket doesn't disqualify a good high-confidence one, but the safe region must be contiguous from
    # the top so a caller can express it as a single ``confidence >= threshold`` mask).
    safe_threshold = 1.0
    for b in reversed(buckets):
        if b.improvement > 0.0:
            safe_threshold = b.conf_lo
        else:
            break

    safe_mask = conf_arr >= safe_threshold
    blended_safe = np.where(safe_mask, blended_all, model_arr)

    return OverrideBacktestResult(
        a=a,
        buckets=buckets,
        safe_threshold=safe_threshold,
        mae_model_overall=float(np.mean(np.abs(y_true_arr - model_arr))),
        mae_blend_all=float(np.mean(np.abs(y_true_arr - blended_all))),
        mae_blend_safe=float(np.mean(np.abs(y_true_arr - blended_safe))),
    )


__all__ = ["backtest_override", "OverrideBacktestResult", "ConfidenceBucket"]
