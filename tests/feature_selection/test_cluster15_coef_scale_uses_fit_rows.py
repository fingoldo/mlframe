"""Cluster 15 regression: the coefficient scale comes from the rows the model was fitted on (audit 2026-11-20, finding 15-F7).

Stability selection used to scale |coef| by its out-of-bag rows, so each bootstrap ranking used held-out variance. The scale now comes from the
training rows when they are passed, and the out-of-bag rows are used only when no training rows exist.
"""

import numpy as np

from mlframe.feature_selection.wrappers._importance_methods import _scaled_coef


def test_scale_uses_train_rows_not_eval_rows():
    """With training and out-of-bag rows of different spread, the scale follows the training rows (15-F7)."""
    coef = np.array([1.0, 1.0])
    train_rows = np.array([[0.0, 0.0], [2.0, 4.0]])  # stds (1, 2)
    eval_rows = np.array([[0.0, 0.0], [20.0, 40.0]])  # stds (10, 20)
    out = _scaled_coef(coef, eval_rows, train_rows, "max", "train")
    assert np.allclose(out, [1.0, 2.0])


def test_no_train_rows_falls_back_to_the_data_rows():
    """Control: with no training rows the data rows are used, as before (15-F7)."""
    coef = np.array([1.0, 1.0])
    eval_rows = np.array([[0.0, 0.0], [20.0, 40.0]])
    out = _scaled_coef(coef, eval_rows, None, "max", "train")
    assert np.allclose(out, [10.0, 20.0])
