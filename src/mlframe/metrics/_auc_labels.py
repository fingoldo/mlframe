"""Label normalisation for the AUC entry points, kept beside ``_core_auc_brier.py`` so that module stays under 1k lines."""

from __future__ import annotations

import numpy as np


def _binary_labels_01(y_true):
    """``y_true`` as {0, 1} labels, positive = the larger of two labels (1 for {0, 1} and {-1, 1}) - sklearn's default ``pos_label``. The AUC kernels count ``tps += y`` / ``fps += 1 - y``, so {-1, 1} labels gave NaN while sklearn returned the
    real AUC on the same data. Labels already in [0, 1] (the overwhelmingly common case) pass through untouched; the
    min/max probe is O(n) against the kernel's O(n log n) sort."""
    y = np.asarray(y_true)
    if y.dtype == np.bool_ or y.size == 0:
        return y
    if y.min() < 0 or y.max() > 1:
        # sklearn takes the LARGER of two labels as positive ({1, 2}: 2; {0, 2}: 2; {-1, 1}: 1). A label set that is not exactly two values keeps ``== 1``.
        present = y[~np.isnan(y)] if y.dtype.kind == "f" else y
        values = np.unique(present)
        positive = values.max() if values.size == 2 else 1
        return np.ascontiguousarray(y == positive, dtype=np.float64)
    return y
