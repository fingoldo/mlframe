"""Streamed GPU MI: block planning respects VRAM, the pinned cap and the block cap; blocks match the CPU batch MI."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _streamed_gpu_mi as S

GiB = 1 << 30


DTYPES = [np.float32, np.float64]  # the suite runs in both: the float32 default and the strict float64 setting


@pytest.mark.parametrize("dtype", DTYPES)
def test_no_device_gives_zero_width(dtype):
    """With no usable device the plan is empty, and the caller must use its CPU path."""
    plan = S.plan_streamed_mi(1000, [], 8 * GiB, dtype)
    assert plan["width"] == 0 and plan["devices"] == []


@pytest.mark.parametrize("dtype", DTYPES)
def test_width_is_limited_by_the_smallest_device(dtype):
    """Two devices with different free VRAM: the block width fits the smaller one, so one width serves every device."""
    small = [(0, 2 * GiB), (1, 8 * GiB)]
    big = [(0, 8 * GiB), (1, 8 * GiB)]
    n = 1_000_000
    assert S.plan_streamed_mi(n, small, 16 * GiB, dtype)["width"] <= S.plan_streamed_mi(n, big, 16 * GiB, dtype)["width"]
    assert S.plan_streamed_mi(n, small, 16 * GiB, dtype)["devices"] == [0, 1]


@pytest.mark.parametrize("dtype", DTYPES)
def test_block_bytes_and_pinned_total_are_capped(dtype):
    """A block never exceeds the block cap, and two staging buffers per device stay inside the pinned cap."""
    plan = S.plan_streamed_mi(1_000_000, [(0, 64 * GiB)], 64 * GiB, dtype)
    assert plan["block_bytes"] <= S.BLOCK_BYTES_MAX
    assert 2 * plan["block_bytes"] <= plan["pinned_total"] <= S.PIN_CAP_MAX


@pytest.mark.parametrize("dtype", DTYPES)
def test_small_host_ram_shrinks_the_width(dtype):
    """Less free host RAM lowers the pinned cap, which shrinks the block width instead of over-allocating."""
    roomy = S.plan_streamed_mi(10_000_000, [(0, 8 * GiB)], 64 * GiB, dtype)
    tight = S.plan_streamed_mi(10_000_000, [(0, 8 * GiB)], 1 * GiB, dtype)
    assert tight["width"] <= roomy["width"]
    assert 2 * tight["width"] * 10_000_000 * np.dtype(dtype).itemsize <= tight["pinned_total"]


@pytest.mark.parametrize("dtype", DTYPES)
def test_streamed_mi_matches_cpu_batch_mi(dtype):
    """Column blocks scored on the GPU agree with the CPU batch MI (tie-free data, values to 1e-6, same ranking)."""
    pytest.importorskip("cupy")
    if S._devices() == []:
        pytest.skip("no CUDA device")
    from mlframe.feature_selection.filters.hermite_fe import plugin_mi_classif_batch_dispatch as cpu

    rng = np.random.default_rng(0)
    n, p = 200_000, 24
    dt = dtype
    X = rng.standard_normal((n, p)).astype(dt)
    y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.standard_normal(n) > 0).astype(np.int64)
    blocks = (X[:, a : a + 8] for a in range(0, p, 8))
    dt = dtype
    gpu = S.mi_classif_streamed((np.asarray(b, dtype=dt) for b in blocks), y, nbins=10, n_rows=n, dtype=dt)
    ref = np.asarray(cpu(X, y, 10))
    assert gpu.shape == ref.shape == (p,)
    assert np.allclose(gpu, ref, atol=1e-6)
    assert np.array_equal(np.argsort(-gpu, kind="stable")[:5], np.argsort(-ref, kind="stable")[:5])


@pytest.mark.parametrize("dtype", DTYPES)
def test_streamed_fill_matches_copy_path(dtype):
    """The production fill path and the copy-into-staging reference score the same blocks to identical MI values."""
    pytest.importorskip("cupy")
    if S._devices() == []:
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(1)
    n, p = 50_000, 16
    X = rng.standard_normal((n, p)).astype(dtype)
    y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.standard_normal(n) > 0).astype(np.int64)
    blocks = [X[:, a : a + 8] for a in range(0, p, 8)]
    copy_path = S.mi_classif_streamed(iter(blocks), y, nbins=10, n_rows=n, dtype=dtype)

    def fill(k, out):
        """Write column block k into the staging view the fill path hands over."""
        out[...] = blocks[k]

    fill_path = S.mi_classif_streamed_fill(fill, len(blocks), [b.shape[1] for b in blocks], y, nbins=10, n_rows=n, dtype=dtype)
    assert np.array_equal(copy_path, fill_path)
