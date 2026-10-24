"""A kernel-tuning-cache sweep must not pick the same backend in every cell.

A GPU-capable kernel that the sweep sends to the CPU everywhere (or to the GPU everywhere) is either a kernel with no crossover, which has to be said and
measured, or a sweep that dropped a variant by mistake. The pair residual test hit the second case: its GPU variant differed from the CPU reference by
about 1e-8 relative against a sweep tolerance of 1e-9, the harness dropped it as divergent, and every cell chose the CPU.

Two checks:

* ``test_pair_residual_sweep_chooses_cpu_small_and_gpu_large``: a reduced sweep of that kernel on a GPU host picks the CPU for a tiny shape and the GPU for a large one.
* ``test_every_tuned_gpu_kernel_uses_more_than_one_path``: for every GPU-capable kernel registered in the package that has a measured entry in this host's
  cache, the regions contain at least two distinct choices, unless the kernel is listed in ``SINGLE_PATH_BY_MEASUREMENT`` with the measurement that justifies it.
"""

from __future__ import annotations

import pytest

# Kernels whose measured regions legitimately hold a single backend on this class of host. Each entry states what was measured; an entry without a
# measurement does not belong here.
SINGLE_PATH_BY_MEASUREMENT: dict[str, str] = {
    "cmi_batched_cpu_cuda_crossover": "GTX 1050 Ti host: all 17 measured regions chose cpu; revisit on a newer GPU (audit: KTC single-path kernels)",
    "discretize_2d_array": "GTX 1050 Ti host: all 5 measured regions chose cpu; revisit on a newer GPU (audit: KTC single-path kernels)",
}

# A tuner whose choices are variants of one backend (radix-select version, thread count, resident-only) is not a backend crossover.
BACKEND_NAMES = {"cpu", "gpu", "cupy", "njit", "njit_serial", "njit_parallel", "gpu_fused", "numba_cuda"}


def _gpu_present() -> bool:
    """Whether cupy sees at least one device."""
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # nosec B110 - no cupy or no device
        return False


needs_gpu = pytest.mark.skipif(not _gpu_present(), reason="no usable cupy device on this host")


@needs_gpu
def test_pair_residual_sweep_chooses_cpu_small_and_gpu_large(monkeypatch):
    """On a reduced grid the sweep keeps both variants and the winner changes with the shape."""
    from mlframe.feature_selection.filters import _fe_sufficient_summary_pairs as sp

    monkeypatch.setattr(sp, "_PRT_SWEEP_N_SAMPLES", [1_500, 20_000])
    monkeypatch.setattr(sp, "_PRT_SWEEP_N_PAIRS_GRID", [100, 80_000])
    regions = sp._run_pair_residual_sweep()
    choices = {(reg["n_samples_max"], reg["n_pairs_max"]): reg["backend_choice"] for reg in regions if reg["n_samples_max"] is not None}
    assert choices[(1_500, 100)] == "njit", choices
    assert choices[(20_000, 80_000)] == "gpu_fused", choices
    assert len(set(choices.values())) == 2, choices


@needs_gpu
def test_every_tuned_gpu_kernel_uses_more_than_one_path():
    """Every GPU-capable kernel with measured regions on this host sends different shapes to different backends, or is justified in the table above."""
    from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
    from pyutilz.performance.kernel_tuning.registry import discover_tuners

    registry = discover_tuners("mlframe.feature_selection.filters", warn_on_import_fail=False)
    cache = KernelTuningCache.load_or_create()
    single, untuned = {}, []
    for name, spec in sorted(registry.items()):
        if not spec.gpu_capable:
            continue
        regions = cache.get_regions(name) or []
        if not regions:
            untuned.append(name)
            continue
        choices = {str(reg.get("backend_choice")) for reg in regions if reg.get("backend_choice")}
        if not choices & BACKEND_NAMES:
            continue
        if len(choices) < 2 and name not in SINGLE_PATH_BY_MEASUREMENT:
            single[name] = sorted(choices)
    assert not single, (
        "GPU-capable kernels whose measured regions hold one backend only (check that the sweep did not drop a variant, or add the kernel to "
        f"SINGLE_PATH_BY_MEASUREMENT with the measurement): {single}; kernels with no measured entry on this host: {untuned}"
    )
