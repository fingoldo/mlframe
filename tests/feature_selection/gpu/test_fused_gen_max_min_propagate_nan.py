"""The fused GPU candidate kernel's max and min propagate NaN like the NumPy path.

``max(NaN, y)`` is NaN in ``np.maximum`` and the CPU path then scrubs it to 0. The kernel returned ``y`` instead, so the GPU and CPU candidate
matrices disagreed on every max/min column whose first operand is NaN.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover - no usable CUDA runtime
    _HAS_DEVICE = False

pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


def test_fused_candidate_matrix_matches_cpu_on_nan_operands():
    """With NaN in the first operand, the GPU candidate matrix equals the NumPy one column for column."""
    from mlframe.feature_selection.filters import _gpu_resident_shared as shared

    rng = np.random.default_rng(0)
    n = 500
    a = rng.normal(size=n)
    a[::7] = np.nan
    b = rng.normal(size=n)
    ua_cm = shared._unary_stack_cm(cp, cp.asarray(a))
    ub_cm = shared._unary_stack_cm(cp, cp.asarray(b))
    gpu = cp.asnumpy(shared._fused_generate_block(ua_cm, ub_cm, shared._COMBOS))
    cpu = shared._build_candidate_matrix(np, a, b)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-10, atol=1e-12)
