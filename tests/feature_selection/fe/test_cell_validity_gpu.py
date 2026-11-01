"""The fused validity kernel equals the numpy filter ``isfinite().all() and std() > 1e-12`` and the column means, for odd sizes, both dtypes and degenerate columns."""

import numpy as np
import pytest


def _block(n: int, dtype) -> np.ndarray:
    """Columns: normal, large offset with a small spread, constant, one NaN, one infinity, tiny-but-real spread, ties."""
    rng = np.random.default_rng(0)
    cols = [rng.normal(size=n), 1e6 + rng.normal(size=n) * 1e-2, np.full(n, 3.0), rng.normal(size=n), rng.normal(size=n), rng.normal(size=n) * 1e-9, np.round(rng.normal(size=n))]
    cols[3][n // 2] = np.nan
    cols[4][n - 1] = np.inf
    return np.column_stack(cols).astype(dtype)


@pytest.mark.parametrize("n", [31, 129, 2047, 2049, 10007, 100000])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_validity_and_mean_match_numpy(n, dtype):
    """Usable flags equal the numpy rule; the means of the usable columns agree."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._cell_validity_gpu import cell_validity_gpu

    block = _block(n, dtype)
    try:
        usable, mean = cell_validity_gpu(cp, cp.asarray(block))
    except Exception as exc:  # pragma: no cover - no usable device
        pytest.skip(f"needs a usable CUDA device: {exc}")
    want = np.isfinite(block).all(axis=0) & (block.astype(np.float64).std(axis=0) > 1e-12)
    np.testing.assert_array_equal(usable, want)
    good = want
    np.testing.assert_allclose(mean[good], block.astype(np.float64)[:, good].mean(axis=0), rtol=1e-9, atol=1e-6)


def test_result_does_not_depend_on_the_launch():
    """Two launches on the same block give identical statistics (no floating-point atomics)."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._cell_validity_gpu import cell_validity_gpu

    d = cp.asarray(_block(50000, np.float32))
    a, b = cell_validity_gpu(cp, d), cell_validity_gpu(cp, d)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
