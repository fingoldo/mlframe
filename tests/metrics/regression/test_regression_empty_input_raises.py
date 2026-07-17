"""Regression test (audit3 degenerate-P1): the fast_* regression metric drop-ins claim sklearn parity but
the njit kernels did 0/0 -> a SILENT NaN on empty input (where sklearn raises ValueError), inconsistent with
the sibling fast_regression_metrics_block which guards n==0. They now raise ValueError on empty, matching
sklearn; a real (non-empty) call is unaffected.
"""

import numpy as np
import pytest

from mlframe.metrics.regression._regression_metrics import (
    fast_r2_score,
    fast_mean_absolute_error,
    fast_mean_squared_error,
    fast_max_error,
)

_FUNCS = [fast_r2_score, fast_mean_absolute_error, fast_mean_squared_error, fast_max_error]


@pytest.mark.parametrize("fn", _FUNCS, ids=[f.__name__ for f in _FUNCS])
def test_empty_input_raises_valueerror_not_silent_nan(fn):
    """Empty input raises valueerror not silent nan."""
    with pytest.raises(ValueError, match="0 sample"):
        fn(np.array([], dtype=float), np.array([], dtype=float))


@pytest.mark.parametrize("fn", _FUNCS, ids=[f.__name__ for f in _FUNCS])
def test_nonempty_input_still_computes(fn):
    """Nonempty input still computes."""
    yt = np.array([1.0, 2.0, 3.0, 4.0])
    yp = np.array([1.1, 1.9, 3.2, 3.7])
    val = np.asarray(fn(yt, yp), dtype=float)
    assert np.all(np.isfinite(val)), f"{fn.__name__} produced non-finite on a normal input"
