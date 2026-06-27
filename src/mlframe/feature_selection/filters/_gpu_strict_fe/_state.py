"""Multi-device GPU-resident FE state for the KTC-free ``MLFRAME_FE_GPU_STRICT`` path.

The residency contract this state enforces (see the plan + ``_audit.py``):
  * Operands + y are uploaded ONCE PER PARTICIPATING DEVICE (G devices -> G bulk H2D, never per-candidate).
  * The shared operand table is REPLICATED per device (every candidate needs every operand); candidate COLUMNS
    are the shard axis (assigned by ``_fe_gpu_batch/_packer.pack_blocks_to_devices`` on ``DeviceProfile.speed``).
  * Per device, the growing candidate buffer is VRAM-chunked via ``DeviceProfile.capacity_bytes`` /
    ``_gpu_resident_fe._gpu_k_chunk``; the launch config (threads, fused-vs-codes path) is chosen from the
    device's queried ``maxThreadsPerBlock`` / ``shared_per_block`` rather than hardcoded.
  * Only HOST metadata (operand names, recipes, op-specs) lives off-device; no bulk array returns mid-pipeline.

Collapses to a single entry when one (or zero) device is visible, so the multi-GPU code runs unchanged on a
1-GPU host. Module-level / instance-held device handles are NEVER pickled (see ``__getstate__``)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ResidentFEState:
    """GPU-resident, multi-device operand/candidate state for one FE step under STRICT.

    Built via :meth:`build`; holds per-device resident replicas of the operand table + y and the per-device
    launch config. The candidate generation/scoring (Phase 1+) shards columns across ``profiles`` and reads
    these resident replicas with zero per-candidate H2D."""

    profiles: list                       # list[DeviceProfile] (>=1; collapses to 1 entry single-GPU)
    op_names: list                       # host: operand column names, len == n_ops (column order is canonical)
    n_sub: int                           # subsample row count
    y_min: int                           # fit-constant label min (host scalar, hoisted once)
    n_classes: int                       # fit-constant class span (host scalar)
    f32: bool                            # f32 operand/candidate dtype (MLFRAME_FE_VRAM_F32) else f64
    # device index -> resident cupy arrays (populated by build(); NEVER pickled)
    _operands: dict = field(default_factory=dict)   # device -> (n_sub, n_ops) cupy operand table replica
    _y_codes: dict = field(default_factory=dict)     # device -> (n_sub,) cupy int64 label codes
    _y_cont: Optional[dict] = field(default_factory=dict)  # device -> (n_sub,) cupy float (corr/routing), or {}

    # ---- construction -----------------------------------------------------------------------------------
    @classmethod
    def build(cls, operands_host: np.ndarray, op_names, y_codes_host: np.ndarray,
              *, y_cont_host: Optional[np.ndarray] = None, profiles=None, f32: bool = False) -> "ResidentFEState":
        """Upload the (n_sub, n_ops) host operand table + y to EVERY participating device exactly once.

        ``operands_host`` is the canonical-order operand columns (float); ``y_codes_host`` the int label codes;
        ``y_cont_host`` the continuous y for correlation/routing (optional). ``profiles`` overrides device
        enumeration (tests). Raises on no-cupy / no-device so the caller falls back to the CPU FE path."""
        import cupy as cp

        from .._fe_gpu_batch._devices import enumerate_device_profiles

        profs = list(profiles) if profiles is not None else enumerate_device_profiles()
        if not profs:
            raise RuntimeError("ResidentFEState.build: no CUDA device profiles available")

        dt = np.float32 if f32 else np.float64
        ops_h = np.ascontiguousarray(operands_host, dtype=dt)            # (n_sub, n_ops)
        if ops_h.ndim == 1:
            ops_h = ops_h.reshape(-1, 1)
        n_sub = int(ops_h.shape[0])
        yc_h = np.ascontiguousarray(np.asarray(y_codes_host)).astype(np.int64).ravel()
        y_min = int(yc_h.min()) if yc_h.size else 0
        n_classes = (int(yc_h.max()) - y_min + 1) if yc_h.size else 1
        ycont_h = (np.ascontiguousarray(np.asarray(y_cont_host, dtype=np.float64).ravel())
                   if y_cont_host is not None else None)

        st = cls(profiles=profs, op_names=list(op_names), n_sub=n_sub, y_min=y_min, n_classes=n_classes, f32=bool(f32))
        for p in profs:
            with cp.cuda.Device(p.device):
                st._operands[p.device] = cp.asarray(ops_h)               # ONE bulk H2D per device
                st._y_codes[p.device] = cp.asarray(yc_h)
                if ycont_h is not None:
                    st._y_cont[p.device] = cp.asarray(ycont_h)
        return st

    # ---- per-device accessors ---------------------------------------------------------------------------
    def operands(self, device: int):
        """The resident (n_sub, n_ops) operand table on ``device`` (no H2D)."""
        return self._operands[device]

    def y_codes(self, device: int):
        return self._y_codes[device]

    def y_cont(self, device: int):
        return self._y_cont.get(device) if self._y_cont else None

    def device_ids(self) -> list:
        return [p.device for p in self.profiles]

    def profile(self, device: int):
        for p in self.profiles:
            if p.device == device:
                return p
        raise KeyError(device)

    def launch_config(self, device: int, ky: Optional[int] = None) -> dict:
        """Per-device kernel launch config from the QUERIED hw specs (not hardcoded). ``threads`` clamps to the
        device max; ``use_fused`` is True when the (nbins*ky) shared histogram tile fits the device's
        per-block shared memory (else the codes/global-atomic path)."""
        p = self.profile(device)
        shared = int(getattr(p, "shared_per_block", 48 * 1024))
        max_threads = int(getattr(p, "max_threads_per_block", 1024) or 1024)
        cfg = {"threads": min(256, max_threads), "shared_per_block": shared}
        if ky is not None:
            cfg["use_fused"] = (int(ky) * 4) <= shared   # one int32 histogram row per class fits shared
        return cfg

    def k_chunk(self, device: int, n_cols: int, *, working_multiple: int = 5) -> int:
        """Max candidate columns to materialize+score in one on-device batch on ``device`` so peak VRAM stays
        within the per-device budget. Reuses the FE VRAM governor (``_gpu_k_chunk``) under the device's free
        VRAM, bounded by ``DeviceProfile.capacity_bytes``."""
        from .._gpu_resident_fe import _gpu_k_chunk

        p = self.profile(device)
        free = int(getattr(p, "free_vram", 0)) or None
        bpe = 4 if self.f32 else 8
        chunk = _gpu_k_chunk(self.n_sub, free_bytes=free, bytes_per_elem=bpe,
                             working_multiple=working_multiple, max_cols=int(n_cols))
        return max(1, int(chunk))

    def free(self) -> None:
        """Release all resident device arrays (call at FE-step end). Idempotent."""
        try:
            import cupy as cp
        except Exception:
            self._operands.clear(); self._y_codes.clear(); (self._y_cont or {}).clear()
            return
        for d in list(self._operands):
            with cp.cuda.Device(d):
                self._operands.pop(d, None)
                self._y_codes.pop(d, None)
                if self._y_cont:
                    self._y_cont.pop(d, None)
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    # device handles must never be pickled (mirrors the module-level resident-cache convention).
    def __getstate__(self):
        d = dict(self.__dict__)
        d["_operands"] = {}
        d["_y_codes"] = {}
        d["_y_cont"] = {}
        return d
