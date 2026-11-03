"""With sample weights the per-class ICE reduction weights each class by its weighted positive count."""

from __future__ import annotations

import numpy as np

from mlframe.metrics._ice_metric import compute_probabilistic_multiclass_error


def _bed():
    """Three-class scores with imbalanced classes and row weights that upweight the rarest class."""
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(600), np.ones(300), np.full(100, 2)]).astype(int)
    logits = rng.normal(size=(1000, 3)) + 2.0 * np.eye(3)[y]
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    w = np.where(y == 2, 10.0, 1.0)
    return y, p, w


def test_reduction_weights_classes_by_weighted_positive_count():
    """The reduced value equals the weighted-positive average of the per-class values, not the row-count average."""
    y, p, w = _bed()
    total, per_class = compute_probabilistic_multiclass_error(y, p, sample_weight=w, weight_by_class_npositives=True, return_per_class=True)
    wpos = {c: float(w[y == c].sum()) for c in per_class}
    expected = sum(per_class[c] * wpos[c] for c in per_class) / sum(wpos.values())
    assert abs(total - expected) < 1e-9
    unweighted_counts = {c: float((y == c).sum()) for c in per_class}
    row_count_average = sum(per_class[c] * unweighted_counts[c] for c in per_class) / sum(unweighted_counts.values())
    assert abs(expected - row_count_average) > 1e-6
