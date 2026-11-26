"""Cluster 22 regression: the fused bootstrap bundle only handles 0/1 labels (audit 2026-11-20, finding 22-F25).

The fused AUC counts the label vector as 0/1, with no guard, so a {1, 2} or {-1, +1} encoding produced a meaningless AUC without any error. Such labels are
now not applicable (None), and the caller falls back to the generic bootstrap; genuine 0/1 labels still take the fused path.
"""

import numpy as np

from mlframe.evaluation._bootstrap_fused_binary_bundle import bootstrap_auc_brier_ll_ece_batch


def _data(n=80, seed=0):
    """Predicted probabilities and matching 0/1 labels."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p).astype(np.int64)
    return y, p


def test_non_binary_label_encodings_are_not_handled():
    """Labels in {1, 2}, {-1, +1} or with a NaN return None so the caller uses the generic path (22-F25)."""
    y, p = _data()
    assert bootstrap_auc_brier_ll_ece_batch(y + 1, p, n_bootstrap=20, random_state=0) is None
    assert bootstrap_auc_brier_ll_ece_batch(2 * y - 1, p, n_bootstrap=20, random_state=0) is None
    y_nan = y.astype(float)
    y_nan[0] = np.nan
    assert bootstrap_auc_brier_ll_ece_batch(y_nan, p, n_bootstrap=20, random_state=0) is None


def test_binary_labels_still_take_the_fused_path():
    """Genuine 0/1 labels return the four-metric result (22-F25)."""
    y, p = _data()
    out = bootstrap_auc_brier_ll_ece_batch(y, p, n_bootstrap=20, random_state=0)
    assert out is not None and set(out) == {"roc_auc", "brier", "log_loss", "ece"}
