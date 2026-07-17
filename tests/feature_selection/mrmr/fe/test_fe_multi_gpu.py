"""Heterogeneous multi-GPU FE-batcher packer + executor guard (2026-06-26).

CPU-only (no CUDA needed): the speed-weighted CP-SAT/greedy packer balances work across heterogeneous
devices and CP-SAT is never worse than greedy. CUDA-gated: the multi-GPU executor produces the IDENTICAL
MI table as the single-GPU path regardless of how columns are spread across devices (per-column MI is
assignment-invariant) -- validated by injecting two distinct device profiles onto the one physical GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_gpu_batch._devices import DeviceProfile
from mlframe.feature_selection.filters._fe_gpu_batch._packer import (
    _cpsat_pack,
    _greedy_lpt,
    pack_blocks_to_devices,
)


def _makespan(assign, works, speeds):
    """Helper that makespan."""
    load = [0.0] * len(speeds)
    for b, d in enumerate(assign):
        load[d] += works[b]
    return max(load[d] / speeds[d] for d in range(len(speeds)))


def test_pack_single_device_trivial():
    """Pack single device trivial."""
    assert pack_blocks_to_devices([5, 3, 2], [1.0]) == [0, 0, 0]
    assert pack_blocks_to_devices([], [1.0, 2.0]) == []


def test_pack_heterogeneous_speed_loads_the_faster_device_more():
    """Pack heterogeneous speed loads the faster device more."""
    works = [10] * 8
    speeds = [1.0, 3.0]  # device 1 is 3x faster
    assign = pack_blocks_to_devices(works, speeds)
    load0 = sum(works[b] for b in range(len(works)) if assign[b] == 0)
    load1 = sum(works[b] for b in range(len(works)) if assign[b] == 1)
    assert load1 > load0, f"faster device must get more work: load0={load0} load1={load1}"


def test_cpsat_not_worse_than_greedy():
    """CP-SAT makespan <= greedy on the canonical LPT-suboptimal instance (and generally)."""
    works = [3, 3, 2, 2, 2]
    speeds = [1.0, 1.0]
    greedy = _greedy_lpt(works, speeds)
    cpsat = _cpsat_pack(works, speeds)
    if cpsat is None:
        pytest.skip("ortools unavailable")
    ms_greedy = _makespan(greedy, works, speeds)
    ms_cpsat = _makespan(cpsat, works, speeds)
    assert ms_cpsat <= ms_greedy + 1e-9, f"CP-SAT {ms_cpsat} should be <= greedy {ms_greedy}"
    assert ms_cpsat == pytest.approx(6.0), f"CP-SAT must hit the optimal makespan 6, got {ms_cpsat}"
    assert ms_greedy == pytest.approx(7.0), f"this instance pins greedy's 7 (the 4/3-1/3m gap)"


def test_device_profile_speed_is_proportional():
    """Device profile speed is proportional."""
    base = dict(free_vram=1, total_vram=1, shared_per_block=49152, cc_major=6, cc_minor=1)
    slow = DeviceProfile(device=0, sm_count=6, clock_khz=1_000_000, **base)
    fast = DeviceProfile(device=1, sm_count=12, clock_khz=1_500_000, **base)
    assert fast.speed == pytest.approx(slow.speed * (12 / 6) * (1.5))


# ---------------------------------------------------------------------------
# CUDA: multi-GPU executor == single-GPU, via two injected profiles on device 0.
# ---------------------------------------------------------------------------
def _need_cuda() -> bool:
    """Need cuda."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        return False


def _fixture(seed=9, n=4000, k=50, nbins=10):
    """Helper that fixture."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(1, 5, n)
    b = rng.uniform(1, 5, n)
    cols = [a**2 / b, np.log(a) * b]
    while len(cols) < k:
        cols.append(rng.uniform(0, 1, n))
    X = np.column_stack([np.nan_to_num(c.astype(np.float64)) for c in cols])
    y = rng.integers(0, nbins, n).astype(np.int64)
    return X, y, nbins


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_multi_gpu_matches_single_gpu():
    """Two heterogeneous profiles pointing at the SAME physical device 0 -> the multi-GPU partition +
    pack + per-device-thread + reassemble must reproduce the single-GPU MI table exactly."""
    import cupy as cp
    from mlframe.feature_selection.filters._fe_gpu_batch._executor import (
        gpu_fe_batch_mi,
        multi_gpu_fe_batch_mi,
    )

    X, y, nb = _fixture()
    base = dict(device=0, free_vram=2**31, total_vram=2**32, shared_per_block=49152, cc_major=6, cc_minor=1)
    profs = [
        DeviceProfile(sm_count=6, clock_khz=1_000_000, **base),
        DeviceProfile(sm_count=12, clock_khz=1_400_000, **base),  # "faster" -> gets more columns
    ]
    single = gpu_fe_batch_mi(X, y, nb)
    multi = multi_gpu_fe_batch_mi(X, y, nb, profiles=profs)
    cp.get_default_memory_pool().free_all_blocks()
    assert np.allclose(single, multi, atol=1e-9, rtol=0), f"multi-GPU MI table diverged from single-GPU: max|d|={np.max(np.abs(single - multi)):.3e}"


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_multi_gpu_collapses_to_single_profile():
    """Multi gpu collapses to single profile."""
    import cupy as cp
    from mlframe.feature_selection.filters._fe_gpu_batch._executor import (
        gpu_fe_batch_mi,
        multi_gpu_fe_batch_mi,
    )

    X, y, nb = _fixture()
    one = [DeviceProfile(device=0, free_vram=2**31, total_vram=2**32, sm_count=6, clock_khz=1_000_000, shared_per_block=49152, cc_major=6, cc_minor=1)]
    assert np.array_equal(multi_gpu_fe_batch_mi(X, y, nb, profiles=one), gpu_fe_batch_mi(X, y, nb))
    cp.get_default_memory_pool().free_all_blocks()
