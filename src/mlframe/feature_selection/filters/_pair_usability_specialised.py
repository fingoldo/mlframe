"""The warp usability |corr| kernel specialised on one form: the form id is a compile-time constant, so the form's branch chain folds away.

``_kernel_warp`` of ``batch_pair_usability_corr_gpu`` reads its form id from an array and evaluates ``_eval_form_nan``'s chain of comparisons on every element of both passes. The kernel is instruction-bound (its
rows are read about 6 times faster than it runs), so a launch per form with the id as a constant is faster: 0.74 s -> 0.58 s for 20000 pairs x 5 forms x 30000 rows. The arithmetic is the generic kernel's, term
for term, so the outputs are bit-identical; the generic kernel stays the default and the fallback; this one is selected by ``MLFRAME_USABILITY_CORR_KERNEL=specialised``.

bench-attempt-rejected as the default (2026-11-21, GTX 1050 Ti): it won on a small matrix (20000 pairs, 400 operands: 0.74 s -> 0.58 s) and lost at the scale of the dominant-pair scan (200000 pairs, 1000 operands = 120 MB
of rows: 5.61 s generic against 6.53 s here). The generic kernel's warps of one pair run side by side and share the operand rows in L2; five launches each stream every pair's rows from DRAM again.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

_KERNELS: dict = {}
_LOCK = threading.Lock()


def specialised_kernel(ft: Any, form_id: int) -> Any:
    """The compiled kernel of ``form_id`` at float type ``ft`` (compiled once per pair), or ``None`` without CUDA."""
    from . import batch_pair_usability_corr_gpu as base

    if not base._CUDA_AVAIL or base._nb_cuda is None:
        return None
    key = (np.dtype(ft).name, int(form_id))
    with _LOCK:
        kernel = _KERNELS.get(key)
        if kernel is None:
            kernel = _KERNELS[key] = _build(ft, int(form_id), base)
        return kernel


def _build(ft: Any, form_const: int, base: Any) -> Any:
    """Define the kernel of one form (the device function it reaches gets a stable numba id, so the on-disk cache key is the same in every process)."""
    cuda = base._nb_cuda
    warp = base._WARP_SIZE
    eval_form = base._eval_form_nan
    nan_ft = ft(np.nan)
    from ._numba_stable_keys import pin_dispatcher_uuids

    pin_dispatcher_uuids(eval_form, tag="spec-" + np.dtype(ft).name)

    @cuda.jit(cache=True)
    def _kernel_form(y, operand_matrix, pair_a, pair_b, eps, cv2, out, column):
        """One warp per pair for the compile-time form ``form_const``; the result goes to ``out[:, column]``."""
        n_rows = y.shape[0]
        global_tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        p = global_tid // warp
        lane = global_tid % warp
        if p >= pair_a.shape[0]:
            return
        a_idx = pair_a[p]
        b_idx = pair_b[p]
        cnt = 0
        sa = ft(0.0)
        sv = ft(0.0)
        i = lane
        while i < n_rows:
            yi = ft(y[i])
            x0 = ft(operand_matrix[a_idx, i])
            x1 = ft(operand_matrix[b_idx, i])
            v = eval_form(form_const, x0, x1, eps, nan_ft)
            if math.isfinite(yi) and math.isfinite(v):
                cnt += 1
                sa += yi
                sv += v
            i += warp
        offset = 16
        while offset > 0:
            cnt += cuda.shfl_down_sync(0xFFFFFFFF, cnt, offset)
            sa += cuda.shfl_down_sync(0xFFFFFFFF, sa, offset)
            sv += cuda.shfl_down_sync(0xFFFFFFFF, sv, offset)
            offset //= 2
        cnt = cuda.shfl_sync(0xFFFFFFFF, cnt, 0)
        sa = cuda.shfl_sync(0xFFFFFFFF, sa, 0)
        sv = cuda.shfl_sync(0xFFFFFFFF, sv, 0)
        if cnt < 2:
            if lane == 0:
                out[p, column] = 0.0
            return
        inv = ft(1.0) / ft(cnt)
        ma = sa * inv
        mv = sv * inv
        saa = ft(0.0)
        svv = ft(0.0)
        sav = ft(0.0)
        i = lane
        while i < n_rows:
            yi = ft(y[i])
            x0 = ft(operand_matrix[a_idx, i])
            x1 = ft(operand_matrix[b_idx, i])
            v = eval_form(form_const, x0, x1, eps, nan_ft)
            if math.isfinite(yi) and math.isfinite(v):
                da = yi - ma
                dv = v - mv
                saa += da * da
                svv += dv * dv
                sav += da * dv
            i += warp
        offset = 16
        while offset > 0:
            saa += cuda.shfl_down_sync(0xFFFFFFFF, saa, offset)
            svv += cuda.shfl_down_sync(0xFFFFFFFF, svv, offset)
            sav += cuda.shfl_down_sync(0xFFFFFFFF, sav, offset)
            offset //= 2
        if lane == 0:
            fc = ft(cnt)
            if saa <= fc * cv2 * ma * ma or svv <= fc * cv2 * mv * mv:
                out[p, column] = 0.0
                return
            den = math.sqrt(saa) * math.sqrt(svv)
            if den <= ft(0.0):
                out[p, column] = 0.0
                return
            c = sav / den
            if not math.isfinite(c):
                out[p, column] = 0.0
                return
            out[p, column] = -c if c < ft(0.0) else c

    return _kernel_form


def launch_specialised(ft: Any, y_d: Any, operand_d: Any, pair_a_d: Any, pair_b_d: Any, form_ids: np.ndarray, eps: Any, cv2: Any, out_d: Any, threads: int, warp: int) -> bool:
    """Run one launch per form into the columns of ``out_d``; ``False`` (nothing launched) when a kernel is unavailable, so the caller takes the generic kernel."""
    kernels = [specialised_kernel(ft, int(f)) for f in form_ids]
    if any(k is None for k in kernels):
        return False
    n_pairs = int(pair_a_d.shape[0])
    blocks = (n_pairs * warp + threads - 1) // threads
    for column, kernel in enumerate(kernels):
        kernel[blocks, threads](y_d, operand_d, pair_a_d, pair_b_d, eps, cv2, out_d, column)
    return True
