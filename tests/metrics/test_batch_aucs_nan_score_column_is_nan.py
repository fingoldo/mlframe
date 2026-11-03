"""A score column with a NaN gets NaN AUCs on the CPU and GPU backends instead of a finite number."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics._gpu_metrics import compute_batch_aucs, is_gpu_metrics_available


def _data():
    """Binary labels and two score columns; the first holds a NaN."""
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.4).astype(np.int8)
    p = rng.random((300, 2))
    p[5, 0] = np.nan
    return y, p


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_nan_column_is_nan_clean_column_is_finite(backend):
    """The NaN column reports NaN for ROC and PR AUC, the clean column keeps its value."""
    if backend == "gpu" and not is_gpu_metrics_available():
        pytest.skip("no CUDA device")
    y, p = _data()
    roc, pr = compute_batch_aucs(y, p, force_backend=backend)
    assert np.isnan(roc[0]) and np.isnan(pr[0])
    assert np.isfinite(roc[1]) and np.isfinite(pr[1])
