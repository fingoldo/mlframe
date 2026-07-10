"""biz_value test for ``calibration.compute_oof_confidence`` / ``apply_confidence_shrinkage``.

The win: in a multi-output ranking scenario (e.g. per-product recommendation scores), outputs with genuinely
NO discriminative power can still occasionally emit spuriously high raw scores by chance, competing on equal
footing with genuinely-informative outputs and hurting top-K accuracy. Shrinking low-OOF-confidence outputs
toward a neutral value before ranking should measurably improve top-1 accuracy.
"""
from __future__ import annotations

import numpy as np

from mlframe.calibration.confidence_shrinkage import apply_confidence_shrinkage, compute_oof_confidence


def _top1_accuracy(pred_dict: dict, true_relevant: np.ndarray) -> float:
    names = list(pred_dict.keys())
    matrix = np.column_stack([pred_dict[n] for n in names])
    top1_idx = np.argmax(matrix, axis=1)
    top1_names = np.array(names)[top1_idx]
    return float((top1_names == true_relevant).mean())


def test_biz_val_confidence_shrinkage_improves_top1_ranking_accuracy():
    rng = np.random.default_rng(0)
    n_users = 2000
    true_relevant = rng.choice(["A", "B", "C"], size=n_users)

    preds, labels = {}, {}
    for name in ["A", "B", "C"]:
        label = (true_relevant == name).astype(int)
        preds[name] = np.where(label == 1, rng.uniform(0.6, 0.9, n_users), rng.uniform(0.05, 0.2, n_users))
        labels[name] = label
    for name in ["D", "E"]:
        labels[name] = np.zeros(n_users, dtype=int)  # never relevant -- genuinely zero discriminative power
        preds[name] = rng.uniform(0.0, 0.95, n_users)  # but noisy raw scores occasionally spike high

    confidences = {name: compute_oof_confidence(preds[name], labels[name]) for name in preds}
    assert confidences["D"] == 1.0 and confidences["E"] == 1.0
    assert confidences["A"] > 2.0 and confidences["B"] > 2.0 and confidences["C"] > 2.0

    shrunk = apply_confidence_shrinkage(preds, confidences, neutral_value=0.1, min_confidence=1.0)

    raw_accuracy = _top1_accuracy(preds, true_relevant)
    shrunk_accuracy = _top1_accuracy(shrunk, true_relevant)

    assert shrunk_accuracy > raw_accuracy + 0.2, (
        f"confidence shrinkage should measurably improve top-1 ranking accuracy: shrunk={shrunk_accuracy:.4f} raw={raw_accuracy:.4f}"
    )


def test_compute_oof_confidence_no_positives_returns_neutral():
    assert compute_oof_confidence(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0])) == 1.0


def test_apply_confidence_shrinkage_zero_span_uses_hard_cutoff():
    preds = {"a": np.array([0.8, 0.9]), "b": np.array([0.1, 0.2])}
    confidences = {"a": 3.0, "b": 3.0}
    shrunk = apply_confidence_shrinkage(preds, confidences, neutral_value=0.5, min_confidence=1.0)
    # equal confidence at the max -> both get full weight (no shrinkage).
    assert np.allclose(shrunk["a"], preds["a"])
    assert np.allclose(shrunk["b"], preds["b"])
