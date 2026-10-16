"""Fused histogram + MI/SU noise gate (one kernel, no counts matrix) for the resident gate; see ``_cuda_fused_gate_kernel_factory``."""

from __future__ import annotations

import logging
import os

import numpy as np

from ._batch_mi_noise_gate_kernels import FUSED_GATE_MAX_BINS, _cuda_fused_gate_kernel_factory, _nb_cuda, cuda_available

logger = logging.getLogger(__name__)

_FUSED_KERNEL = None


def fused_gate_enabled() -> bool:
    """``MLFRAME_FE_GPU_FUSED_GATE=1`` turns the fused gate on. Default off: on the 4 GB dev card the fused launch (one block per (column, permutation), 2.8 M blocks per dispatch) measured 790 ms against 1130 ms for the two kernels, but the wall time of a strict-GPU xor3 fit did not move (165 s against 160 s), so the extra launches it enables for SU gates cost more than the counts matrix it saves."""
    return os.environ.get("MLFRAME_FE_GPU_FUSED_GATE", "0").strip().lower() in ("1", "true", "on", "yes")


def fused_gate_mi(d_disc, nbins_arr, d_nb, d_y, d_freq, freqs_y, n, K, K_y, P, threads, use_su):
    """``(P, K)`` MI (or SU) matrix on the host from one fused launch, or ``None`` when the fused gate does not apply (disabled, too many bins for the block,
    no CUDA, or a launch / transpose failure: the caller then runs the two-kernel gate)."""
    global _FUSED_KERNEL
    if not fused_gate_enabled() or not cuda_available() or K == 0:
        return None
    if int(np.max(nbins_arr)) > FUSED_GATE_MAX_BINS or int(threads) < int(np.max(nbins_arr)):
        return None
    sh_bytes = int(np.max(nbins_arr)) * int(K_y) * 4
    from ._batch_mi_noise_gate_kernels import _cuda_shared_mem_per_block

    if not 0 < sh_bytes <= int(_cuda_shared_mem_per_block() * 0.75):
        return None
    try:
        import cupy as cp

        from ._gpu_resident_select import transpose_codes_to_cm

        if _FUSED_KERNEL is None:
            _FUSED_KERNEL = _cuda_fused_gate_kernel_factory()
        if _FUSED_KERNEL is None:
            return None
        d_cm = transpose_codes_to_cm(cp.asarray(d_disc))
        d_out = _nb_cuda.device_array((P, K), dtype=np.float64)
        d_ref = _nb_cuda.to_device(np.ones(K, dtype=np.float64))
        fy = np.asarray(freqs_y, dtype=np.float64)
        fy = fy[fy > 0]
        h_y = float(-(fy * np.log(fy)).sum())
        _FUSED_KERNEL[(K, P), int(threads), 0, sh_bytes](d_cm, d_nb, d_y, d_freq, n, K_y, d_ref, d_out, h_y, bool(use_su))  # type: ignore[index]  # numba cuda launch syntax
        return np.asarray(d_out.copy_to_host())
    except Exception:
        logger.warning("fused noise-gate kernel failed; using the two-kernel gate", exc_info=True)
        return None
