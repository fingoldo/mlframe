"""``cv_loss`` of a multiclass target scores every fold against the full label set, also when a training fold misses a rare class."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._linear_cv_loss import cv_loss


def test_a_class_with_one_row_does_not_break_the_cv_loss():
    """Three classes, one of them with a single row: the training rows of the fold that holds it miss that class; the loss is finite instead of raising a label-set mismatch."""
    rng = np.random.default_rng(0)
    n = 600
    x = rng.standard_normal((n, 2))
    y = (x[:, 0] > 0.3).astype(np.int64)
    y[0] = 2
    loss = cv_loss(x, y, True, 5, 0)
    assert np.isfinite(loss) and loss > 0.0
