"""The GPU external-validation binner hands columns with NaN to the host binner, whose edges use finite rows only (tracker 2026-11-02 wave 13, PAR13-1)."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _gpu_resident_extval as ev

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="needs a CUDA device")


def test_nan_in_an_external_factor_returns_none_and_finite_input_returns_codes():
    """30% NaN in the external factor: the device edges would sit inside the NaN tail; the function must decline (None) rather than return different codes than the host."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=1000)
    b = rng.normal(size=1000)
    b_nan = b.copy()
    b_nan[:300] = np.nan
    ops = np.array([0], dtype=np.int64)
    assert ev.gpu_materialise_extval_codes_host(a, [b_nan], ops, 10) is None
    assert ev.gpu_materialise_extval_codes_host(a, [b], ops, 10) is not None
