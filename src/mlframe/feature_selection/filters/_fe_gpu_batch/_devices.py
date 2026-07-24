"""Heterogeneous GPU device profiling for the FE batcher's multi-GPU packer (2026-06-26).

Enumerates the usable CUDA devices and captures, per device, the attributes the packer needs to schedule
candidate-column blocks across HETEROGENEOUS GPUs (different VRAM, SM count, shared-mem, clock -> different
throughput): free/total VRAM, SM count, clock, shared-mem-per-block, compute capability. From these it
derives a VRAM CAPACITY (hard constraint, reusing the FE VRAM governor) and a SPEED weight (for the
makespan objective). Everything here is transient (never stored on a pickled estimator). Collapses to a
single profile on a 1-GPU host, so the multi-GPU path runs unchanged with one device.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# CUDA cores per SM by compute capability -- a coarse but monotone FLOP proxy for the speed weight.
_CORES_PER_SM = {
    (3, 0): 192, (3, 5): 192, (3, 7): 192,
    (5, 0): 128, (5, 2): 128, (5, 3): 128,
    (6, 0): 64, (6, 1): 128, (6, 2): 128,
    (7, 0): 64, (7, 2): 64, (7, 5): 64,
    (8, 0): 64, (8, 6): 128, (8, 7): 128, (8, 9): 128,
    (9, 0): 128, (10, 0): 128, (12, 0): 128,
}
_DEFAULT_CORES_PER_SM = 64

# VRAM safety cushion: fraction of free VRAM held back, with an absolute floor (protects small cards where
# a percentage alone is too thin near OOM). Both env-overridable.
_FE_VRAM_CUSHION_FRAC = float(os.environ.get("MLFRAME_FE_VRAM_CUSHION_FRAC", "0.10") or 0.10)
_FE_VRAM_CUSHION_FLOOR = int(os.environ.get("MLFRAME_FE_VRAM_CUSHION_FLOOR_BYTES", str(128 * 1024 * 1024)) or (128 * 1024 * 1024))


def _cores_per_sm(cc_major: int, cc_minor: int) -> int:
    """Look up CUDA cores per SM for a compute-capability tuple, falling back to a conservative default for unlisted (future/rare) capabilities."""
    return _CORES_PER_SM.get((int(cc_major), int(cc_minor)), _DEFAULT_CORES_PER_SM)


def fe_gpu_f32_enabled() -> bool:
    """Whether the GPU FE-batch path scores in float32 (``MLFRAME_FE_VRAM_F32`` truthy). Default OFF -> f64,
    which keeps the GPU MI bit-~ (1e-9) to the CPU njit. f32 is ~2.2x faster (half H2D + f32 radix-select)
    and SELECTION-EQUIVALENT (Spearman rank 1.0, identical top-K; values drift ~5e-6 only), but NOT 1e-9
    bit-identical -- so it is opt-in, validated by the f32 selection-equivalence test."""
    return os.environ.get("MLFRAME_FE_VRAM_F32", "").strip().lower() in ("1", "true", "on", "yes")


def crit_dtype_relaxed() -> bool:
    """Whether PRECISION-CRITICAL float64 compute may relax to float32. Controlled by
    ``MLFRAME_CRIT_DTYPE_RELAXED`` (DEFAULT ON).

    The numerically-heavier FE stages that still upload / compute in float64 (the ALS prewarp seed's
    standardised columns, the fourier detrend/periodogram column splits, the quantile discretiser input matrix)
    are relaxed to float32 when this is on -- half the H2D and faster device math. It is a SEPARATE knob from
    ``MLFRAME_FE_VRAM_F32`` (which governs the FE-batch MI scoring dtype): this one governs the residual f64
    hotspots. It is applied ONLY where the FE candidate selection is unchanged by the f32 rounding (the f32
    drift is far below every FE decision margin -- validated per-stage on F2 across all distributions + the
    stage's biz suites). Set ``MLFRAME_CRIT_DTYPE_RELAXED=0`` to force the strict float64 path everywhere."""
    return os.environ.get("MLFRAME_CRIT_DTYPE_RELAXED", "1").strip().lower() not in ("0", "false", "off", "no")


def crit_float_dtype():
    """``cupy.float32`` when the precision-critical relaxation is on (default), else ``cupy.float64``. The single
    dtype source the relaxed f64 hotspots pick their upload/compute dtype from (see :func:`crit_dtype_relaxed`)."""
    import cupy as cp
    return cp.float32 if crit_dtype_relaxed() else cp.float64


@dataclass(frozen=True)
class DeviceProfile:
    """Immutable per-device capability snapshot (transient; built fresh each fit)."""
    device: int
    free_vram: int
    total_vram: int
    sm_count: int
    clock_khz: int
    shared_per_block: int
    cc_major: int
    cc_minor: int

    @property
    def speed(self) -> float:
        """Relative throughput weight = SM count * clock(kHz) * cores/SM. Monotone, unit-free; used only to
        balance the makespan across devices, never as an absolute time."""
        return float(self.sm_count) * float(self.clock_khz) * float(_cores_per_sm(self.cc_major, self.cc_minor))

    def capacity_bytes(self, n_rows: int) -> int:
        """VRAM budget for one resident candidate block on this device: free * KTC vram_fraction - cushion.
        Reuses the FE VRAM governor's per-host fraction so the budget matches the rest of the FE path."""
        from .._gpu_resident_fe import _gpu_k_chunk_vram_fraction
        frac = _gpu_k_chunk_vram_fraction(int(n_rows))
        frac = min(0.9, max(1e-3, float(frac)))
        cushion = max(_FE_VRAM_CUSHION_FLOOR, int(self.free_vram * _FE_VRAM_CUSHION_FRAC))
        return max(1, int(self.free_vram * frac) - cushion)


def _visible_device_ids() -> list[int]:
    """CUDA device indices the FE batcher may use: getDeviceCount() filtered by the global GPU off-switches
    and an optional ``MLFRAME_FE_VRAM_DEVICES=0,1`` subset. cupy already remaps CUDA_VISIBLE_DEVICES to a
    dense 0..n-1 range, so indices here are cupy-local."""
    # GPU_INFRA_C-9 fix (mrmr_audit_2026-07-22): delegate the MLFRAME_DISABLE_GPU/CUDA_VISIBLE_DEVICES=""
    # opt-out check to the shared _gpu_policy module instead of a THIRD inline reimplementation of it.
    from .._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return []
    try:
        import cupy as cp
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return []
    ids = list(range(count))
    subset = os.environ.get("MLFRAME_FE_VRAM_DEVICES", "").strip()
    if subset:
        try:
            want = {int(t) for t in subset.split(",") if t.strip() != ""}
            ids = [i for i in ids if i in want]
        except Exception:  # nosec B110 - best-effort path
            pass
    return ids


def _profile_device(device: int) -> "DeviceProfile":
    """Query one CUDA device's current free/total VRAM and static properties (SM count, clock, shared mem, compute capability) and pack them into a :class:`DeviceProfile`."""
    import cupy as cp
    with cp.cuda.Device(device):
        free, total = cp.cuda.runtime.memGetInfo()
        props = cp.cuda.runtime.getDeviceProperties(device)

    def _p(key, default):
        """Best-effort lookup of a device-properties key, falling back to ``default`` when the key is missing or the underlying value is None (older cupy/driver combos don't expose every field)."""
        try:
            v = props[key]
        except Exception:
            return default
        return v if v is not None else default

    return DeviceProfile(
        device=int(device),
        free_vram=int(free),
        total_vram=int(total),
        sm_count=int(_p("multiProcessorCount", 1) or 1),
        clock_khz=int(_p("clockRate", 1) or 1),
        shared_per_block=int(_p("sharedMemPerBlock", 48 * 1024) or (48 * 1024)),
        cc_major=int(_p("major", 6) or 6),
        cc_minor=int(_p("minor", 0) or 0),
    )


def enumerate_device_profiles() -> list[DeviceProfile]:
    """Profiles for every usable device (empty if no CUDA / disabled). One entry on a single-GPU host."""
    out = []
    for i in _visible_device_ids():
        try:
            out.append(_profile_device(i))
        except Exception:  # nosec B112 - best-effort path  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
            continue
    return out
