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

from typing import Any, Dict, Optional, Union

import numpy as np


def compute_oof_confidence(
    oof_pred: np.ndarray,
    oof_label: np.ndarray,
    segment_ids: Optional[np.ndarray] = None,
) -> Union[float, Dict[Any, float]]:
    """``mean(oof_pred | label==1) / mean(oof_pred | label==0)`` for one output/segment's OOF predictions.

    Returns ``1.0`` (neutral -- no discriminative signal detectable) when either class is empty, both
    conditional means are non-positive, or the negative-class mean is zero.

    Parameters
    ----------
    segment_ids
        Optional, opt-in: a ``(n_samples,)`` array of per-row segment/group labels (e.g. region, cohort,
        traffic source). When given, a single global confidence scalar can hide the fact that a model is
        genuinely reliable for one subpopulation and unreliable for another -- a global ratio averages the
        two away. Returns ``{segment_id: confidence}`` instead, one ratio per distinct value in
        ``segment_ids``, each computed with the exact same formula restricted to that segment's rows.
        Omitting this (the default) is bit-identical to the pre-extension single-segment behavior.
    """
    if segment_ids is not None:
        segment_ids = np.asarray(segment_ids)
        oof_pred = np.asarray(oof_pred, dtype=np.float64)
        oof_label = np.asarray(oof_label)
        return {seg: _single_oof_confidence(oof_pred[segment_ids == seg], oof_label[segment_ids == seg]) for seg in np.unique(segment_ids)}

    return _single_oof_confidence(oof_pred, oof_label)


def _single_oof_confidence(oof_pred: np.ndarray, oof_label: np.ndarray) -> float:
    """Shared, non-recursive body of :func:`compute_oof_confidence`'s single-segment computation."""
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
    confidences: Dict[str, Union[float, Dict[Any, float]]],
    neutral_value: float = 0.5,
    min_confidence: float = 1.0,
    max_confidence: Optional[float] = None,
    segments: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Shrink each output's predictions toward ``neutral_value`` proportionally to its OOF confidence.

    Parameters
    ----------
    preds
        ``{output_name: (n_samples,) predictions}``.
    confidences
        ``{output_name: confidence}`` from :func:`compute_oof_confidence` (or any equivalent measure where
        higher = more discriminative, ``1.0`` = no signal). When ``segments`` is given for an output name,
        that output's entry must instead be a ``{segment_id: confidence}`` dict (as returned by
        :func:`compute_oof_confidence` called with ``segment_ids``).
    neutral_value
        The value low-confidence predictions shrink toward (e.g. the overall base rate).
    min_confidence
        Confidences at or below this shrink fully to ``neutral_value`` (weight ``0.0`` on the raw prediction).
    max_confidence
        Confidences at or above this get full weight (``1.0``, no shrinkage); defaults to the max confidence
        actually observed across ``confidences`` (so the most-confident output present is never shrunk).
    segments
        Optional, opt-in: ``{output_name: (n_samples,) segment_ids}``. A single global confidence scalar
        applies the same shrinkage strength to every row of an output even when the model is reliably
        discriminative for one subpopulation and near-random for another -- over-shrinking the strong
        segment, under-shrinking the weak one. When an output name is present here, its shrinkage weight is
        computed per-segment (from ``confidences[name]``, a ``{segment_id: confidence}`` dict) and applied
        row-by-row via each row's ``segment_ids`` value; rows whose segment is absent from that dict fall
        back to ``min_confidence`` (fully shrunk), matching the existing missing-output fallback. Omitting
        this (the default) is bit-identical to the pre-extension single-confidence-per-output behavior.

    Returns
    -------
    dict
        ``{output_name: shrunk_predictions}``, same shape as ``preds``.
    """
    if not confidences:
        return {name: np.asarray(p, dtype=np.float64) for name, p in preds.items()}

    if max_confidence is not None:
        hi = max_confidence
    else:
        # flatten so a mix of plain floats and (opt-in) per-segment dicts both contribute to the observed max.
        flat_values = [v for v in confidences.values() if not isinstance(v, dict)] + [sv for v in confidences.values() if isinstance(v, dict) for sv in v.values()]
        hi = max(flat_values) if flat_values else min_confidence
    span = hi - min_confidence

    shrunk: Dict[str, np.ndarray] = {}
    for name, pred in preds.items():
        pred = np.asarray(pred, dtype=np.float64)
        seg_arr = segments.get(name) if segments is not None else None
        conf = confidences.get(name, min_confidence)
        if seg_arr is not None and isinstance(conf, dict):
            seg_arr = np.asarray(seg_arr)
            weight_arr = np.empty(pred.shape[0], dtype=np.float64)
            for seg in np.unique(seg_arr):
                seg_conf = conf.get(seg, min_confidence)
                if span <= 0:
                    seg_weight = 1.0 if seg_conf >= hi else 0.0
                else:
                    seg_weight = float(np.clip((seg_conf - min_confidence) / span, 0.0, 1.0))
                weight_arr[seg_arr == seg] = seg_weight
            shrunk[name] = weight_arr * pred + (1.0 - weight_arr) * neutral_value
            continue
        if span <= 0:
            weight = 1.0 if conf >= hi else 0.0  # type: ignore[operator]
        else:
            weight = float(np.clip((conf - min_confidence) / span, 0.0, 1.0))  # type: ignore[operator]
        shrunk[name] = weight * pred + (1.0 - weight) * neutral_value
    return shrunk


__all__ = ["compute_oof_confidence", "apply_confidence_shrinkage"]
