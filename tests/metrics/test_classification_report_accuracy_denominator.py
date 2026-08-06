"""METRICS-11 (2026-08-05 audit): ``fast_classification_report``'s ``accuracy`` must divide by
``supports.sum()`` (in-range true labels only), matching ``weighted_averages``'s denominator -- not
``len(y_true)``, which over-counts out-of-range true labels that never contribute a hit.
"""

from __future__ import annotations

import numpy as np

from mlframe.metrics.core import fast_classification_report


def test_accuracy_denominator_excludes_out_of_range_true_labels():
    """With an out-of-range true label present, accuracy must be computed over in-range labels only."""
    # 3 in-range samples, all correctly classified (accuracy should be 1.0 over the 3 in-range labels),
    # plus 1 out-of-range true label (=5, nclasses=2) that contributes neither a hit nor a support count.
    y_true = np.array([0, 1, 0, 5], dtype=np.int64)
    y_pred = np.array([0, 1, 0, 1], dtype=np.int64)

    out = fast_classification_report(y_true, y_pred, nclasses=2)
    accuracy = out[2]

    assert accuracy == 1.0, f"expected accuracy=1.0 (3/3 in-range hits), got {accuracy} -- len(y_true)=4 denominator would give 0.75"
