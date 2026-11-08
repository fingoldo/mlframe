"""Family-test codes of the modular candidate pool computed on the device from the uploaded source columns.

The gate-first modular scan codes every ``mod`` / ``sin`` / ``cos`` candidate of every (source column, period) with the quantile coder of the family max-T test.
On the host path the candidates are computed on the CPU (0.7 s for 1500 candidates at 100000 rows) and then uploaded to the device coder (600 MB). Here the
source columns are uploaded once (in double precision, as the host residue is computed) and one kernel writes the three candidates of each period straight
into the float32 rows the resident coder (``_quantile_codes_gpu.quantile_codes_resident_f32``) reads, a block of source columns at a time within the memory pool.

Same arithmetic as ``_periodic_fe._modular_all_ops_into``: ``r = v - p * floor(v / p)`` in double precision, ``sin`` / ``cos`` of ``2 pi r / p``, non-finite to 0,
stored at float32. The device ``sin`` / ``cos`` may differ from the host libm by an ulp of the double, which the float32 store almost always absorbs; a value
that rounds differently can move one row across a bin edge, which leaves the family test's verdicts (a max over 1500 candidates) unchanged in practice.

``MLFRAME_MODULAR_CODES_DEVICE=0|1`` forces the path (default: on with a usable device for a pool of at least ``_quantile_codes_gpu`` device-size elements).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for
import pandas as pd

from ._raw_module_cache import RawModuleCache
from mlframe._numba_parallel_guard import parallel_kernel_entry
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

THREADS = 256
_BYTES_PER_SOURCE_ROW = 8  # one double source value per row and source column (the float32 mode needs 4)
_BUDGET_FRACTION = 0.2

_SRC = r"""
// dtype-ok: Intentional float64, reviewed 2026-11-05: float64 on purpose, floor(v / p) of a timestamp (~1.6e9) is noise at float32 resolution 128; the host uploads float64 explicitly.
extern "C" __global__ void modular_all_ops(const SRC_T* __restrict__ S, const double* __restrict__ P, const long long n, const int n_periods, float* __restrict__ out) {
    const int src = blockIdx.y / n_periods;
    const int ip = blockIdx.y - src * n_periods;
    const double p = P[ip];
    const double k = 2.0 * 3.141592653589793 / p;
    const SRC_T* s = S + (long long)src * n;
    float* o_mod = out + ((long long)blockIdx.y * 3) * n;
    float* o_sin = o_mod + n;
    float* o_cos = o_sin + n;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const double v = (double)s[i];  // a float32 source widens exactly; the residue below is still double
        if (!isfinite(v)) { o_mod[i] = 0.0f; o_sin[i] = 0.0f; o_cos[i] = 0.0f; continue; }
        const double r = v - p * floor(v / p);
        double sn, cs;
        sincos(k * r, &sn, &cs);
        o_mod[i] = isfinite(r) ? (float)r : 0.0f;
        o_sin[i] = isfinite(sn) ? (float)sn : 0.0f;
        o_cos[i] = isfinite(cs) ? (float)cs : 0.0f;
    }
}
"""

_MODULE = RawModuleCache("#define SRC_T double\n" + _SRC)  # Intentional float64: source columns uploaded as float64
_MODULE_F32 = RawModuleCache("#define SRC_T float\n" + _SRC)  # source columns that are float32 already are uploaded as they are (half the copy and the transfer)


def _use_device(n_elements: int) -> bool:
    """Whether the device path runs: forced by ``MLFRAME_MODULAR_CODES_DEVICE``, else a float32 fit, the default copy policy and a large enough pool."""
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return False  # a run without a GPU (MLFRAME_DISABLE_GPU=1, CUDA_VISIBLE_DEVICES=-1 / "") never tries the device
    forced = os.environ.get("MLFRAME_MODULAR_CODES_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if os.environ.get("MLFRAME_MODULAR_DROP_COPIES", "0").strip().lower() in ("1", "true", "on", "yes"):
        return False
    from ._fe_usability_signal import _crit_np_dtype
    from ._quantile_codes_gpu import _device_min_elements

    return bool(np.dtype(_crit_np_dtype()) == np.float32 and n_elements >= _device_min_elements())


def _sources_per_block(cp, n_rows: int, per_source: int, source_bytes: int = _BYTES_PER_SOURCE_ROW) -> int:
    """Source columns per launch: the double source block and its float32 candidates within ``_BUDGET_FRACTION`` of what the pool can hand out (at least 1)."""
    try:
        pool = cp.get_default_memory_pool()
        free_device = int(cp.cuda.runtime.memGetInfo()[0]) + int(pool.free_bytes())
        limit = int(pool.get_limit())
        room = free_device if limit <= 0 else min(free_device, max(0, limit - int(pool.used_bytes())))
    except Exception as exc:
        log_throttle(logger, "_modular_codes_gpu.except_85", logging.WARNING, "modular device codes: free memory unknown (%s)", exc)
        return 4
    per = (source_bytes + 4 * per_source) * max(1, n_rows)
    return int(max(1, (_BUDGET_FRACTION * room) // per))


def modular_family_codes_gpu(X: pd.DataFrame, cols: Sequence[str], periods: Sequence[float], n_bins: int, family_elements: Optional[int] = None) -> "Optional[tuple[list[str], np.ndarray, np.ndarray]]":
    """``(names, codes, nx)`` of the modular candidates of ``cols`` (``modular_candidate_block`` names and order; ``family_codes_of_rows`` codes of its rows), or
    ``None`` when the device path does not apply or fails (the caller then builds the pool on the host)."""
    from ._periodic_fe import _numeric_cols, engineered_name_modular

    cols = list(_numeric_cols(X, cols))
    periods = tuple(float(p) for p in periods if float(p) > 0.0)
    n = len(X)
    per_source = 3 * len(periods)
    # ``family_elements``: the candidate count of the whole gate (every chunk of source columns) times the rows, so all chunks of one family take the same path;
    # judged per chunk, the last, smaller chunk fell below the strict-resident bound and took the host pool while the others ran here
    if not cols or not periods or n < 2 or not _use_device(int(family_elements) if family_elements is not None else len(cols) * per_source * n):
        return None
    try:
        import cupy as cp

        from ._quantile_codes_gpu import quantile_codes_resident_f32

        # float32 sources (the project dtype) go up as float32: every value widens exactly to the double the residue is computed in, so the codes are the same
        # Intentional float64: non-float32 sources go up as float64, the residue is double
        src_dtype = np.float32 if all(np.dtype(X[c].dtype) == np.float32 for c in cols) else np.float64
        kernel = (_MODULE_F32 if src_dtype == np.float32 else _MODULE).get(cp).get_function("modular_all_ops")
        p_dev = cp.asarray(np.asarray(periods, dtype=np.float64))  # Intentional float64: the residue is computed in double precision, as on the host
        codes = np.empty((len(cols) * per_source, n), dtype=np.int16)
        nx = np.empty(len(cols) * per_source, dtype=np.int64)
        step = _sources_per_block(cp, n, per_source, np.dtype(src_dtype).itemsize)

        def _host_block(s0: int) -> np.ndarray:
            """Source columns ``s0:s0+step`` as a contiguous (columns, rows) block."""
            # the residue v - p*floor(v/p) is double inside the kernel (a float32 timestamp ~1.6e9 has resolution 128); only the upload follows the column dtype
            return np.ascontiguousarray(X[cols[s0 : s0 + step]].to_numpy(dtype=src_dtype).T)

        starts = list(range(0, len(cols), step))
        prefetch = ThreadPoolExecutor(max_workers=1)  # the next block is prepared on the host while the device scores the current one
        pending = prefetch.submit(_host_block, starts[0])
        for k, s0 in enumerate(starts):
            part = cols[s0 : s0 + step]
            host = pending.result()
            if k + 1 < len(starts):
                pending = prefetch.submit(_host_block, starts[k + 1])
            src = cp.asarray(host)
            del host
            out = cp.empty((len(part) * per_source, n), dtype=cp.float32)
            threads = block_size_for(kernel, work_items=n)
            kernel((min(1024, (n + threads - 1) // threads), len(part) * len(periods)), (threads,), (src, p_dev, np.int64(n), np.int32(len(periods)), out))
            del src
            r0 = s0 * per_source
            with parallel_kernel_entry():
                quantile_codes_resident_f32(cp, out, int(n_bins), codes[r0 : r0 + out.shape[0]], nx[r0 : r0 + out.shape[0]])
            del out
        prefetch.shutdown(wait=True)
    except Exception as exc:
        logger.warning("device modular candidate codes failed (%s: %s); building the pool on the host", type(exc).__name__, exc)
        return None
    names = [engineered_name_modular(c, p, op) for c in cols for p in periods for op in ("mod", "sin", "cos")]
    return names, codes, nx


__all__ = ["modular_family_codes_gpu"]
