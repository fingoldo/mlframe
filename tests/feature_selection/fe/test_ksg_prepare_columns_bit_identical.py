"""The njit column preparation of the batched KSG scorer is bit-identical to the numpy form it replaced, constant columns, float32 and float64 blocks included."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _ksg_cd_fast as kf


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(20000, 100), (3000, 7), (50, 3)])
def test_prepare_columns_is_bit_identical(dtype, shape):
    """Scaled columns plus noise equal the numpy reference bit for bit, with a constant column (std 0 -> 1) among them."""
    n, p = shape
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((n, p)) * np.arange(1, p + 1)).astype(dtype)
    x[:, 0] = 3.0
    draw = np.random.RandomState(7).standard_normal(size=n)
    got = kf._prepare_columns(np.ascontiguousarray(x), draw)
    want = kf._prepare_columns_numpy(x, draw)
    assert got.dtype == np.float64 and np.array_equal(got, want)


def test_batched_scores_equal_the_per_column_scores():
    """The batched call (njit preparation) and the per-column calls of the same block and seed give the same MI values."""
    rng = np.random.default_rng(1)
    n, p = 4000, 6
    x = rng.standard_normal((n, p))
    y = (x[:, 0] + 0.5 * rng.standard_normal(n) > 0).astype(np.int64)
    batch = kf.ksg_mi_discrete_target_columns(x, y, n_neighbors=3, random_state=2)
    for j in range(p):
        one = kf.ksg_mi_discrete_target_columns(x[:, [j]], y, n_neighbors=3, random_state=2)
        assert batch[j] == one[0]
