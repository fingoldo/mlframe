"""Confidence-weighted shrinkage: pull a class/segment's predictions toward neutral when it's weakly discriminative.

A multi-output/multi-segment model (e.g. one probability column per product in a recommender) can have wildly
different discriminative power per output -- some outputs cleanly separate positive from negative rows, others
barely do better than chance. Ranking on raw probability lets a weak output's noisy scores compete on equal
footing with a strong output's genuinely informative ones. A 2nd-place Santander-recommendation team's fix:
compute each output's OOF ``confidence = mean(prediction | positive) / mean(prediction | negative)`` (how much
higher predictions run for true positives vs true negatives), then shrink weak-confidence outputs' predictions
toward a neutral value before ranking -- measurably improving MAP@K.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def compute_oof_confidence(oof_pred: np.ndarray, oof_label: np.ndarray) -> float:
    """``mean(oof_pred | label==1) / mean(oof_pred | label==0)`` for one output/segment's OOF predictions.

    Returns ``1.0`` (neutral -- no discriminative signal detectable) when either class is empty, both
    conditional means are non-positive, or the negative-class mean is zero.
    """
    oof_pred = np.asarray(oof_pred, dtype=np.float64)
    # a dot-product split (dot(pred, label) for the positive sum, total-sum minus that for the negative sum)
    # avoids materializing two boolean-mask-indexed copies of ``oof_pred`` -- ~4.3x faster at n=1M, bit-
    # identical to the mask-based computation (see bench_confidence_shrinkage.py).
    label_f = np.asarray(oof_label, dtype=np.float64)
    n_pos = float(label_f.sum())
    n_neg = float(label_f.shape[0]) - n_pos
    if n_pos <= 0.0 or n_neg <= 0.0:
        return 1.0
    pos_sum = float(np.dot(oof_pred, label_f))
    neg_sum = float(oof_pred.sum()) - pos_sum
    pos_mean = pos_sum / n_pos
    neg_mean = neg_sum / n_neg
    if neg_mean <= 0.0:
        return 1.0
    return pos_mean / neg_mean


def apply_confidence_shrinkage(
    preds: Dict[str, np.ndarray],
    confidences: Dict[str, float],
    neutral_value: float = 0.5,
    min_confidence: float = 1.0,
    max_confidence: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Shrink each output's predictions toward ``neutral_value`` proportionally to its OOF confidence.

    Parameters
    ----------
    preds
        ``{output_name: (n_samples,) predictions}``.
    confidences
        ``{output_name: confidence}`` from :func:`compute_oof_confidence` (or any equivalent measure where
        higher = more discriminative, ``1.0`` = no signal).
    neutral_value
        The value low-confidence predictions shrink toward (e.g. the overall base rate).
    min_confidence
        Confidences at or below this shrink fully to ``neutral_value`` (weight ``0.0`` on the raw prediction).
    max_confidence
        Confidences at or above this get full weight (``1.0``, no shrinkage); defaults to the max confidence
        actually observed across ``confidences`` (so the most-confident output present is never shrunk).

    Returns
    -------
    dict
        ``{output_name: shrunk_predictions}``, same shape as ``preds``.
    """
    if not confidences:
        return {name: np.asarray(p, dtype=np.float64) for name, p in preds.items()}

    hi = max_confidence if max_confidence is not None else max(confidences.values())
    span = hi - min_confidence

    shrunk: Dict[str, np.ndarray] = {}
    for name, pred in preds.items():
        pred = np.asarray(pred, dtype=np.float64)
        conf = confidences.get(name, min_confidence)
        if span <= 0:
            weight = 1.0 if conf >= hi else 0.0
        else:
            weight = float(np.clip((conf - min_confidence) / span, 0.0, 1.0))
        shrunk[name] = weight * pred + (1.0 - weight) * neutral_value
    return shrunk


__all__ = ["compute_oof_confidence", "apply_confidence_shrinkage"]
