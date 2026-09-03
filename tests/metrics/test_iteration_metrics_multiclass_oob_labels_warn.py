"""METRICS-14 (2026-08-05 audit): ``_multiclass_metrics``'s inline log-loss must warn (with the count) when
it silently drops out-of-range ``y_true`` labels, matching the package's established pattern (e.g.
``fast_mape_mean``, ``fast_rmspe``) of surfacing a dropped-row count via a rate-limited RuntimeWarning
instead of dropping rows with no trace.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.iteration_metrics import _multiclass_metrics


def test_out_of_range_labels_emit_runtime_warning():
    """An out-of-range y_true label (>= n_classes) must trigger a RuntimeWarning naming the dropped count."""
    import mlframe.metrics.iteration_metrics as mod

    mod._MULTICLASS_LOGLOSS_OOB_WARN_SEEN.clear()
    n = 10
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, n)
    y[0] = 99  # out of range for n_classes=3
    sc = rng.random((n, 3))

    with pytest.warns(RuntimeWarning, match="out of range"):
        out = _multiclass_metrics(y, sc, n_classes=3, nbins=10)
    assert np.isfinite(out["log_loss"])
