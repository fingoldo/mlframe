"""Constant members are maximally redundant, not independent (audit 2026-11-20, finding 22-F1)."""
import numpy as np

from mlframe.calibration._independence_check import member_residual_correlation


def test_all_constant_members_read_as_fully_redundant():
    """Every member constant: the residual-correlation summary must be 1.0 (redundant), not 0.0 (independent)."""
    p = np.full((50, 3), 0.7)
    diag = member_residual_correlation(p)
    assert diag["mean_abs_residual_correlation"] == 1.0
    assert diag["max_abs_residual_correlation"] == 1.0
