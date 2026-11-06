"""The device |corr| twins keep float64 precision, so a column or target at a large offset does not read as constant."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_abs_corr_gpu import abs_corr_finite_gpu, abs_corr_zerofill_gpu


def _need_cuda():
    """Skip when no CUDA device is usable."""
    try:  # audit-exempt: except_skip_masks_call_under_test -- the guarded call is a one-element allocation that probes for a usable CUDA device; nothing under test runs in the try
        cp.zeros(1).get()
    except Exception as exc:  # no device or driver
        pytest.skip(f"no usable CUDA device: {exc}")


def test_target_at_a_large_offset_is_not_read_as_constant():
    """y = 1e7 + 0.01 * noise correlated with a: float32 y has spacing 1.0 near 1e7 and returned 0.0, the host twin returns the true |r|."""
    _need_cuda()
    rng = np.random.default_rng(0)
    a = rng.standard_normal(5000)
    y = 1e7 + 0.01 * (a + 0.3 * rng.standard_normal(5000))
    fin = np.ones(5000, dtype=bool)
    got = abs_corr_finite_gpu(cp.asarray(a), y, fin, 8)
    want = abs(float(np.corrcoef(a, y)[0, 1]))
    assert want > 0.9 and abs(got - want) < 1e-3, (got, want)


def test_zerofill_operands_at_a_large_offset_keep_their_correlation():
    """a = 1e7 + noise, b = a + small noise: the float32 device variance was ~0 and the statistic read 0.0."""
    _need_cuda()
    rng = np.random.default_rng(1)
    a = 1e7 + rng.standard_normal(5000)
    b = a + 0.1 * rng.standard_normal(5000)
    got = abs_corr_zerofill_gpu(cp.asarray(a), b)
    assert got > 0.99, got
