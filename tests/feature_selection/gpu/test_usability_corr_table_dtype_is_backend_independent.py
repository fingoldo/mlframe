"""The |corr| table has the same dtype whichever backend computed it.

The CPU kernel returned float64 while the GPU kernels returned the criterion float dtype, so a fallback from GPU to CPU changed the dtype of the
table a caller got back for the same request.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import batch_pair_usability_corr_gpu as bp
from mlframe.feature_selection.filters._fe_gpu_batch._devices import crit_float_dtype


def test_cpu_result_has_the_criterion_float_dtype():
    """Forcing the CPU backend returns the table in crit_float_dtype(), as the GPU path does."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(size=n)
    operand = rng.normal(size=(n, 4))
    pair_a = np.array([0, 1, 2], dtype=np.int64)
    pair_b = np.array([1, 2, 3], dtype=np.int64)
    table, _ = bp.dispatch_batch_pair_usability_corr(y, operand, pair_a, pair_b, force_backend="cpu")
    assert table.dtype == crit_float_dtype()
