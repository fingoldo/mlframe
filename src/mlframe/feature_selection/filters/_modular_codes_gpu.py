"""Family-test codes of the modular candidate pool computed on the device from the uploaded source columns.

The gate-first modular scan codes every ``mod`` / ``sin`` / ``cos`` candidate of every (source column, period) with the quantile coder of the family max-T test.
On the host path the candidates are computed on the CPU (0.7 s for 1500 candidates at 100000 rows) and then uploaded to the device coder (600 MB). Here the
source columns are uploaded once (in double precision, as the host residue is computed) and one kernel writes the three candidates of each period straight
into the float32 rows the resident coder (``_quantile_codes_gpu.quantile_codes_resident_f32``) reads, a block of source columns at a time within the memory pool.

Same residue as ``_periodic_fe._modular_all_ops_into``: ``r = v - p * floor(v / p)`` in double precision (the host twin is that njit kernel). The angle ``2 pi r / p`` is formed in
double and the ``sin`` / ``cos`` are taken at the stored float32 type (non-finite to 0), which keeps the card's 1:32 FP64 pipe out of the transcendental (kernel 37 -> 17.6 ms at
16 sources x 5 periods x 400000 rows). The stored values differ from the host's by float rounding (max 2.7e-7): for continuous columns a few cells in a million land one bin over, for
a column of exactly tied values the tied group splits differently, which is arbitrary on either side; the family test's verdicts (a max over 1500 candidates) are unchanged in practice.

``MLFRAME_MODULAR_CODES_DEVICE=0|1`` forces the path (default: on with a usable device for a pool of at least ``_quantile_codes_gpu`` device-size elements).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Sequence

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for
import pandas as pd

from ._ktc_fallback import host_device_sweep, rows_fallback
from ._raw_module_cache import RawModuleCache
from mlframe._numba_parallel_guard import parallel_kernel_entry
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

THREADS = 256
_BYTES_PER_SOURCE_ROW = 8  # one double source value per row and source column (the float32 mode needs 4)
_BUDGET_FRACTION = 0.2

_SRC = r"""
// dtype-ok: Intentional float64, reviewed 2026-11-05: float64 on purpose, floor(v / p) of a timestamp (~1.6e9) is noise at float32 resolution 128; the host uploads float64 explicitly.
extern "C" __global__ void modular_all_ops(const SRC_T* __restrict__ S, const {ACC}* __restrict__ P, const long long n, const int n_periods, {REAL}* __restrict__ out) {
    const int src = blockIdx.y / n_periods;
    const int ip = blockIdx.y - src * n_periods;
    const {ACC} p = P[ip];
    const {ACC} k = 2.0 * 3.141592653589793 / p;
    const SRC_T* s = S + (long long)src * n;
    {REAL}* o_mod = out + ((long long)blockIdx.y * 3) * n;
    {REAL}* o_sin = o_mod + n;
    {REAL}* o_cos = o_sin + n;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const {ACC} v = ({ACC})s[i];  // a float32 source widens exactly; the residue below is still {ACC}
        if (!isfinite(v)) { o_mod[i] = ({REAL})0; o_sin[i] = ({REAL})0; o_cos[i] = ({REAL})0; continue; }
        const {ACC} r = v - p * floor(v / p);
        // The angle k * r lies in [0, 2 pi] and is formed in {ACC}; the sine and cosine are taken at the stored type (the candidates are stored at {REAL}), which keeps the
        // card's 1:32 FP64 pipe out of the transcendental.
        {REAL} sn, cs;
        sincos(({REAL})(k * r), &sn, &cs);
        o_mod[i] = isfinite(r) ? ({REAL})r : ({REAL})0;
        o_sin[i] = isfinite(sn) ? sn : ({REAL})0;
        o_cos[i] = isfinite(cs) ? cs : ({REAL})0;
    }
}
"""

_SRC_RENDERED = _SRC.replace("{ACC}", "double").replace(
    "{REAL}", "float"
)  # Intentional float64: the residue and the angle are computed in double; the candidate rows are stored as float32
_MODULE = RawModuleCache("#define SRC_T double\n" + _SRC_RENDERED)  # Intentional float64: source columns uploaded as float64
_MODULE_F32 = RawModuleCache(
    "#define SRC_T float\n" + _SRC_RENDERED
)  # source columns that are float32 already are uploaded as they are (half the copy and the transfer)


DEVICE_FALLBACK_ROWS = 1_500_000  # elements (sources x candidates per source x rows); the measured crossover (GTX 1050 Ti), replaced per host by the tuner once a sweep has run
_SWEEP_ROWS = [300_000, 1_500_000, 6_000_000, 24_000_000]  # elements of the pool, the "n_samples" axis of the sweep
_SWEEP_SOURCES = 8
_SWEEP_PERIODS = (7.0, 24.0, 52.0, 365.0, 12.5)
_SWEEP_BINS = 10
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """A float32 frame of ``_SWEEP_SOURCES`` columns whose pool (3 candidates per period) holds ``dims["n_samples"]`` elements."""
    n = max(64, int(dims["n_samples"]) // (_SWEEP_SOURCES * 3 * len(_SWEEP_PERIODS)))
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"c{i}": (rng.random(n) * 1000.0).astype(np.float32) for i in range(_SWEEP_SOURCES)})
    return (X,)


def _code_fractions(codes: np.ndarray) -> np.ndarray:
    """The share of cells in each bin over the whole pool. The device takes the sine and cosine at the stored type, so a handful of values (about 4 in a million) land one bin
    over from the host's double result; that moves the share of a bin by a fraction of 1e-6, while a divergent kernel moves it by orders more. Row-exact equivalence of the
    verdicts is pinned by the tests, not by this sweep."""
    out = np.zeros(64, dtype=np.float64)  # Intentional float64: comparison vector
    counts = np.bincount(np.asarray(codes).ravel().astype(np.int64, copy=False), minlength=1)[:64]
    out[: counts.size] = counts / float(max(1, np.asarray(codes).size))
    return out


def _sweep_host(X: pd.DataFrame) -> np.ndarray:
    """The host pool: candidates computed on the CPU, then coded."""
    from ._family_maxt_gate import family_codes_of_rows
    from ._periodic_fe import modular_candidate_block

    _names, values = modular_candidate_block(X, list(X.columns), list(_SWEEP_PERIODS))
    codes, _nx = family_codes_of_rows(values, _SWEEP_BINS)
    return _code_fractions(codes)


def _sweep_device(X: pd.DataFrame) -> np.ndarray:
    """The device pool; a refusal raises so the sweep does not record the host fallback as a device time."""
    got = modular_family_codes_gpu(X, list(X.columns), list(_SWEEP_PERIODS), _SWEEP_BINS, family_elements=1 << 40)
    if got is None:
        raise RuntimeError("device modular candidate codes unavailable")
    return _code_fractions(got[1])


_fallback_choice = rows_fallback(__name__)

try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="fe_modular_codes_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-6, atol=1e-6),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_modular_codes_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the element-count fallback decides
    logger.warning("kernel_tuner spec build for fe_modular_codes_backend failed: %s", e)
    _SPEC = None


def modular_device_chosen(n_elements: int) -> bool:
    """Whether a pool of ``n_elements`` takes the device: the per-host kernel-tuning cache, fallback ``DEVICE_FALLBACK_ROWS`` elements."""
    if _SPEC is None:
        return int(n_elements) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_elements))) == "device"
    except Exception as exc:
        logger.debug("fe_modular_codes_backend: KTC choose() failed, using the element-count fallback: %s", exc)
        return int(n_elements) >= DEVICE_FALLBACK_ROWS


def _use_device(n_elements: int) -> bool:
    """Whether the device path runs: forced by ``MLFRAME_MODULAR_CODES_DEVICE``, else a float32 fit, the default copy policy and a large enough pool (the strict-resident bound under the
    strict path, whose residency contract keeps bulk device-to-host traffic out of small pools; otherwise the kernel-tuning cache)."""
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return False  # a run without a GPU (MLFRAME_DISABLE_GPU=1, CUDA_VISIBLE_DEVICES=-1 / "") never tries the device
    forced = os.environ.get("MLFRAME_MODULAR_CODES_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if os.environ.get("MLFRAME_MODULAR_DROP_COPIES", "0").strip().lower() in ("1", "true", "on", "yes"):
        return False
    from ._fe_usability_signal import _crit_np_dtype
    from ._quantile_codes_gpu import _device_min_elements, STRICT_DEVICE_MIN_ELEMENTS

    if np.dtype(_crit_np_dtype()) != np.float32:
        return False
    if _device_min_elements() == STRICT_DEVICE_MIN_ELEMENTS:
        return int(n_elements) >= STRICT_DEVICE_MIN_ELEMENTS
    return modular_device_chosen(int(n_elements))


def _sources_per_block(cp, n_rows: int, per_source: int, source_bytes: int = _BYTES_PER_SOURCE_ROW) -> int:
    """Source columns per launch (one column block): the double source block and its float32 candidates within ``_BUDGET_FRACTION`` of what the pool can hand out (at least 1)."""
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


def modular_family_codes_gpu(
    X: pd.DataFrame, cols: Sequence[str], periods: Sequence[float], n_bins: int, family_elements: Optional[int] = None
) -> "Optional[tuple[list[str], np.ndarray, np.ndarray]]":
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
