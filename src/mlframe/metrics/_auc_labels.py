"""Label normalisation for the AUC entry points, kept beside ``_core_auc_brier.py`` so that module stays under 1k lines."""

from __future__ import annotations

import numpy as np


def _binary_labels_01(y_true):
    """``y_true`` as {0, 1} labels, positive = 1 - the convention ``fast_roc_curve`` and sklearn's default ``pos_label``
    use. The AUC kernels count ``tps += y`` / ``fps += 1 - y``, so {-1, 1} labels gave NaN while sklearn returned the
    real AUC on the same data. Labels already in [0, 1] (the overwhelmingly common case) pass through untouched; the
    min/max probe is O(n) against the kernel's O(n log n) sort."""
    y = np.asarray(y_true)
    if y.dtype == np.bool_ or y.size == 0:
        return y
    if y.min() < 0 or y.max() > 1:
        return np.ascontiguousarray(y == 1, dtype=np.float64)
    return y
