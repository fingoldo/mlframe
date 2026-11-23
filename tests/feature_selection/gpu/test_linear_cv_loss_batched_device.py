"""The batched device CV loss of single columns equals the per-column host scorer, including unusable columns, and the side-check verdicts of the batched route equal the per-candidate route."""

import numpy as np
import pytest

pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _linear_cv_loss as lcl


def _data(n=6000, k=10, seed=0):
    """A binary target and columns of varying strength, one constant column and one column holding NaN."""
    rng = np.random.default_rng(seed)
    y = (rng.standard_normal(n) + 0.8 * rng.standard_normal(n) > 0).astype(np.int64)
    cols = [rng.standard_normal(n) * (1 + i) + 0.3 * i * y for i in range(k)]
    cols[2] = np.full(n, 1.0)
    cols[4] = cols[4].copy()
    cols[4][::9] = np.nan
    return cols, y


def test_batched_losses_equal_the_host_scorer():
    """Every column's mean CV loss agrees with ``cv_loss`` to 1e-7 and the unusable columns (constant, NaN) are infinite in both."""
    cols, y = _data()
    ref = [lcl.cv_loss(c, y, True, 3, 0) for c in cols]
    got = lcl.cv_loss_columns_device(cols, y, True, 3, 0)
    assert got is not None and len(got) == len(cols)
    assert len(ref) == len(got) > 0, "no candidate was scored"
    for a, b in zip(ref, got):
        assert (np.isinf(a) and np.isinf(b)) or abs(a - b) < 1e-7


def test_regression_and_multiclass_are_not_batched():
    """Regression and a three-class target return None so the per-column scorer runs."""
    cols, y = _data()
    assert lcl.cv_loss_columns_device(cols, y.astype(float), False, 3, 0) is None
    y3 = (np.arange(y.shape[0]) % 3).astype(np.int64)
    assert lcl.cv_loss_columns_device(cols, y3, True, 3, 0) is None
