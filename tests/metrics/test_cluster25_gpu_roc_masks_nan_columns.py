"""Audit 2026-11-20 finding 25-F34: the GPU multi-column ROC primitive reports NaN for a column with a NaN score.

The batch helper already masks such columns; the primitive did not, so a direct call returned a finite, order-dependent AUC where
the CPU path returns NaN. Runs only where cupy is installed, since the primitive is a cupy kernel.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.metrics._gpu_metrics import gpu_multiple_roc_auc_scores


def test_nan_score_column_gives_nan_and_clean_columns_keep_their_value():
    """Column 0 has a NaN score, so it is NaN; column 1 is clean and keeps a finite AUC."""
    y = np.array([0, 1, 0, 1, 1, 0])
    s = np.column_stack([np.array([0.1, np.nan, 0.4, 0.8, 0.6, 0.2]), np.array([0.1, 0.9, 0.4, 0.8, 0.6, 0.2])])
    out = cp.asnumpy(gpu_multiple_roc_auc_scores(y, s))
    assert np.isnan(out[0])
    assert np.isfinite(out[1])
