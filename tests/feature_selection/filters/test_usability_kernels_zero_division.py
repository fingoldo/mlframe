"""The serial usability kernels no longer raise ZeroDivisionError on an exact zero and agree with the parallel ones (tracker 2026-11-02 wave 12, NJ12-1)."""

import numpy as np

from mlframe.feature_selection.filters import _numba_utils_usability as nu
from mlframe.feature_selection.filters import _usability_njit_pool as pool


def test_serial_and_parallel_agree_when_a_column_holds_an_exact_zero():
    """Unary code 4 is 1/v; the serial kernels used Python's error model and raised on v == 0 while the parallel ones returned inf."""
    args = list(pool._make_usability_inputs({"n_rows": 2_000, "n_combos": 20}))
    args[0] = args[0].copy()
    args[0][::50] = 0.0
    args[6] = np.full_like(args[6], 4)  # every combo applies the reciprocal to the first operand
    serial = nu._pair_combo_mi_njit(*args)
    parallel = nu._pair_combo_mi_njit_parallel(*args)
    np.testing.assert_allclose(serial, parallel, atol=1e-12)
