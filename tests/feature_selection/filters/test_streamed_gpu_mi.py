"""Streamed GPU MI: block planning respects VRAM, the pinned cap and the block cap; blocks match the CPU batch MI."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _streamed_gpu_mi as S

GiB = 1 << 30


def test_no_device_gives_zero_width():
    """With no usable device the plan is empty, and the caller must use its CPU path."""
    plan = S.plan_streamed_mi(1000, [], 8 * GiB)
    assert plan["width"] == 0 and plan["devices"] == []


def test_width_is_limited_by_the_smallest_device():
    """Two devices with different free VRAM: the block width fits the smaller one, so one width serves every device."""
    small = [(0, 2 * GiB), (1, 8 * GiB)]
    big = [(0, 8 * GiB), (1, 8 * GiB)]
    n = 1_000_000
    assert S.plan_streamed_mi(n, small, 16 * GiB)["width"] <= S.plan_streamed_mi(n, big, 16 * GiB)["width"]
    assert S.plan_streamed_mi(n, small, 16 * GiB)["devices"] == [0, 1]


def test_block_bytes_and_pinned_total_are_capped():
    """A block never exceeds the block cap, and two staging buffers per device stay inside the pinned cap."""
    plan = S.plan_streamed_mi(1_000_000, [(0, 64 * GiB)], 64 * GiB)
    assert plan["block_bytes"] <= S.BLOCK_BYTES_MAX
    assert 2 * plan["block_bytes"] <= plan["pinned_total"] <= S.PIN_CAP_MAX


def test_small_host_ram_shrinks_the_width():
    """Less free host RAM lowers the pinned cap, which shrinks the block width instead of over-allocating."""
    roomy = S.plan_streamed_mi(10_000_000, [(0, 8 * GiB)], 64 * GiB)
    tight = S.plan_streamed_mi(10_000_000, [(0, 8 * GiB)], 1 * GiB)
    assert tight["width"] <= roomy["width"]
    assert 2 * tight["width"] * 10_000_000 * 4 <= tight["pinned_total"]


def test_streamed_mi_matches_cpu_batch_mi():
    """Column blocks scored on the GPU agree with the CPU batch MI (tie-free data, values to 1e-6, same ranking)."""
    pytest.importorskip("cupy")
    if S._devices() == []:
        pytest.skip("no CUDA device")
    from mlframe.feature_selection.filters.hermite_fe import plugin_mi_classif_batch_dispatch as cpu

    rng = np.random.default_rng(0)
    n, p = 200_000, 24
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.standard_normal(n) > 0).astype(np.int64)
    blocks = (X[:, a : a + 8] for a in range(0, p, 8))
    gpu = S.mi_classif_streamed(blocks, y, nbins=10, n_rows=n)
    ref = np.asarray(cpu(X.astype(np.float64), y, 10))
    assert gpu.shape == ref.shape == (p,)
    assert np.allclose(gpu, ref, atol=1e-6)
    assert np.array_equal(np.argsort(-gpu, kind="stable")[:5], np.argsort(-ref, kind="stable")[:5])
