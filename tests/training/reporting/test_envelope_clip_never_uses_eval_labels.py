"""Without train target statistics the envelope clip leaves predictions alone; it never bounds them by the scored labels.

The eval-derived fallback bounded each split's predictions by that split's own labels, so the reported metrics read the labels they score.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.reporting._reporting_regression import _sensors


def test_predictions_are_not_clipped_to_the_scored_labels_without_train_stats():
    """With no train statistics, an outlier prediction passes through unchanged even when the labels would bound it."""
    labels = np.linspace(0.0, 1.0, 50)
    preds = np.concatenate([labels, [1000.0]])
    targets = np.concatenate([labels, [0.5]])
    out = _sensors.apply_prediction_envelope_clip(
        preds, targets, y_train_min=None, y_train_max=None, y_train_std=None, model_name="m", report_title="t",
    )
    np.testing.assert_array_equal(out, preds)
