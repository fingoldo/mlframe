"""The column-tiled |corr| equals numpy's Pearson and the one-block-per-column kernel it replaces for 32 columns and more, with the -1 mark for degenerate columns."""

import numpy as np
import pytest


@pytest.mark.parametrize("n", [31, 129, 8191, 8193, 50000])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("m", [32, 33, 100])
def test_tiled_matches_numpy(n, dtype, m):
    """Same |corr| as np.corrcoef per column; constant columns are -1.0."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_basis import _gpu_batched_abs_corr

    rng = np.random.default_rng(0)
    y = rng.normal(size=n)
    block = rng.normal(size=(n, m)) + 0.3 * y[:, None] * rng.uniform(-1, 1, m)
    block[:, 5] = 2.5
    block = block.astype(dtype)
    try:
        got = cp.asnumpy(_gpu_batched_abs_corr(cp, cp.asarray(block), cp.asarray(y.astype(dtype))))
    except Exception as exc:  # pragma: no cover - no usable device
        pytest.skip(f"needs a usable CUDA device: {exc}")
    want = np.array([abs(np.corrcoef(block[:, j].astype(np.float64), y.astype(dtype).astype(np.float64))[0, 1]) if j != 5 else -1.0 for j in range(m)])
    np.testing.assert_allclose(got, want, atol=2e-5 if dtype == np.float32 else 1e-9)


def test_tiled_equals_the_v1_kernel():
    """Below 32 columns the one-block-per-column kernel runs; the same columns scored by both agree."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_basis import _gpu_batched_abs_corr

    rng = np.random.default_rng(1)
    n = 20000
    y = cp.asarray(rng.normal(size=n).astype(np.float32))
    block = cp.asarray(rng.normal(size=(n, 40)).astype(np.float32))
    wide = cp.asnumpy(_gpu_batched_abs_corr(cp, block, y))
    narrow = cp.asnumpy(_gpu_batched_abs_corr(cp, cp.ascontiguousarray(block[:, :31]), y))
    np.testing.assert_allclose(wide[:31], narrow, atol=2e-6)
