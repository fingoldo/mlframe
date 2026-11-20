"""Cluster 17 regressions: perfect single-class agreement has kappa 1, and non-integer labels are refused by the quantile calibration branch (audit 2026-11-20, findings 17-F13, 17-F16)."""

import numpy as np
import pytest

from mlframe.metrics.calibration._calibration_plot import _integer_labels_or_raise
from mlframe.metrics.classification._classification_extras import cohen_kappa_binary


def test_perfect_single_class_agreement_has_kappa_one():
    """All-zero truth and prediction: observed agreement is 1, so kappa is 1, not 0 (17-F16)."""
    y = np.zeros(20, dtype=np.int64)
    assert cohen_kappa_binary(y, y) == pytest.approx(1.0)


def test_quantile_branch_refuses_soft_labels():
    """A probability label used to be truncated to 0/1 by the int64 cast (17-F13)."""
    with pytest.raises(ValueError, match="integer"):
        _integer_labels_or_raise(np.array([0.2, 0.9, 1.0]))


def test_quantile_branch_accepts_integer_labels():
    """Integer-valued labels keep working."""
    out = _integer_labels_or_raise(np.array([0.0, 1.0, 1.0]))
    assert out.dtype == np.int64 and out.tolist() == [0, 1, 1]
