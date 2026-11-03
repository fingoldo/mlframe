"""fit_composite_similarity scores multiclass targets by a weighted class vote, so accuracy can reach 1 when the blocks separate the classes."""

from __future__ import annotations

import numpy as np

from mlframe.core.composite_similarity import fit_composite_similarity


def _block_sim(labels: np.ndarray) -> np.ndarray:
    """Similarity 1 inside a class, 0 across (a block that separates the classes perfectly)."""
    return (labels[:, None] == labels[None, :]).astype(np.float64)


def test_perfect_three_class_block_scores_accuracy_one():
    """y in {0, 1, 2}: a weighted-mean-then-round prediction can only emit 0 or 1 (accuracy <= ~2/3); the class vote gives 1.0."""
    y = np.repeat([0, 1, 2], 30)
    res = fit_composite_similarity([_block_sim(y)], y, k=5)
    assert res.classification and res.score > 0.99


def test_binary_zero_one_labels_still_work():
    """{0, 1} keeps the historical path (negative control)."""
    y = np.repeat([0, 1], 40)
    res = fit_composite_similarity([_block_sim(y)], y, k=5)
    assert res.score > 0.99
