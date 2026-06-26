"""GPU FE-batcher path -- the GPU half of the two separate, independently-optimised FE-scoring backends.

Phase 2 ships the single-GPU resident executor (``gpu_fe_batch_mi``): score a candidate matrix by the
edge-binned plug-in MI ON the device, VRAM-budget column-chunked via ``_gpu_resident_fe._gpu_k_chunk``,
using the SAME percentile-edge binning + plain plug-in MI as the CPU twin (``_fe_cpu_batch``) so the two
backends select identical features (``test_fe_batch_parity``). Phase 3 adds heterogeneous multi-GPU
device profiling (``_devices``) + CP-SAT VRAM packing (``_packer``) on top of this executor.
"""
from __future__ import annotations

from ._executor import gpu_fe_batch_mi

__all__ = ["gpu_fe_batch_mi"]
