"""Cluster 17 regression: top_k_accuracy warns when k >= K, as its docstring promises (audit 2026-11-20, finding 17-F24).

The return stays NaN; the warning is what the caller now sees. A k below the class count returns a value and no warning.
"""

import warnings

import numpy as np

from mlframe.metrics.classification._classification_extras import top_k_accuracy


def test_k_at_or_above_class_count_warns_and_returns_nan():
    """k >= K gives NaN and a UserWarning naming k and K (17-F24)."""
    y = np.array([0, 1, 2, 1])
    probs = np.full((4, 3), 1.0 / 3.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = top_k_accuracy(y, probs, k=3)
    assert np.isnan(out)
    assert any(issubclass(w.category, UserWarning) and "trivially 1.0" in str(w.message) for w in caught)


def test_k_below_class_count_is_silent_and_returns_a_value():
    """Control: k < K returns the accuracy with no warning (17-F24)."""
    y = np.array([0, 1, 2, 1])
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.3, 0.6, 0.1]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = top_k_accuracy(y, probs, k=1)
    assert np.isfinite(out)
    assert not any("trivially 1.0" in str(w.message) for w in caught)
