"""One kernel for the orthogonal-polynomial design matrix ``B[i, k] = T_k(z[i])``.

The cupy build (:func:`_hermite_prewarp_gpu_resident._build_basis_matrix_gpu`) runs the three-term recurrence one degree at a time, every step several elementwise launches
over the full column plus a strided column write: ~60 launches for a degree-6 pair of designs, in a function the ALS seed calls dozens of times per fit. Here one thread owns
one row and runs the whole recurrence in registers.

The design is built as one column block of at most ``MAX_COLUMNS`` columns (the ALS solve reads the whole design, so there is nothing to stream; a request the VRAM cushion refuses returns
``None``). A block of threads owns a tile of consecutive rows: the recurrences run into a shared tile and the block writes it out as one contiguous run of the row-major design, so the
stores are coalesced (1.6M rows x 7 columns: 1.52 -> 0.57 ms, bit-identical output).

The kernel is compiled with ``--fmad=false`` and evaluates each recurrence step in the order the cupy expression does, so every rounding happens where cupy's separate
elementwise kernels round and the result is bit-identical to the loop it replaces.
"""

from __future__ import annotations

from pyutilz.performance.gpu_hw_launch import block_size_for

import logging
from typing import Any, Optional

from mlframe.utils.log_throttle import log_throttle

from .._gpu_raw_module import LazyRawModule

logger = logging.getLogger(__name__)

_BASIS_CODE = {"hermite": 0, "legendre": 1, "chebyshev": 2, "laguerre": 3}
MAX_COLUMNS = 40  # register budget of the unrolled recurrence
_TILE_BYTES = 32 * 1024  # shared-memory tile of one block (``threads`` rows x ``n_columns`` values)

_SRC = r"""
extern "C" __global__ void build_basis(const {REAL}* __restrict__ x, {REAL}* __restrict__ B, const long long n, const int nc, const int basis) {
    // A block owns a tile of blockDim.x consecutive rows: each thread runs its row's recurrence into a shared tile (row-major, nc per row), then the block writes the tile out as one
    // contiguous run of the (n, nc) matrix, so a warp stores consecutive words instead of one row of nc words per thread.
    extern __shared__ unsigned char sh_raw[];
    {REAL}* tile = ({REAL}*)sh_raw;
    for (long long base = (long long)blockIdx.x * blockDim.x; base < n; base += (long long)gridDim.x * blockDim.x) {
        const long long i = base + threadIdx.x;
        const bool live = i < n;
        const {REAL} xi = live ? x[i] : ({REAL})0;
        {REAL}* row = tile + (long long)threadIdx.x * nc;
        {REAL} p2 = ({REAL})1.0;  // T_{k-2}
        row[0] = p2;
        {REAL} p1 = (basis == 3) ? (({REAL})1.0 - xi) : xi;  // T_{k-1}
        if (nc > 1) row[1] = p1;
        const {REAL} two_x = ({REAL})2.0 * xi;
        for (int k = 2; k < nc; ++k) {
            {REAL} v;
            if (basis == 0) {
                v = xi * p1 - ({REAL})(k - 1) * p2;
            } else if (basis == 1) {
                const {REAL} t1 = ({REAL})(2 * k - 1) * xi;
                const {REAL} t2 = t1 * p1;
                const {REAL} t3 = ({REAL})(k - 1) * p2;
                v = (t2 - t3) / ({REAL})k;
            } else if (basis == 2) {
                v = two_x * p1 - p2;
            } else {
                const {REAL} t1 = ({REAL})(2 * k - 1) - xi;
                const {REAL} t2 = t1 * p1;
                const {REAL} t3 = ({REAL})(k - 1) * p2;
                v = (t2 - t3) / ({REAL})k;
            }
            row[k] = v;
            p2 = p1;
            p1 = v;
        }
        __syncthreads();
        const long long rows_here = (n - base < (long long)blockDim.x) ? (n - base) : (long long)blockDim.x;
        const long long count = rows_here * nc;
        for (long long e = threadIdx.x; e < count; e += blockDim.x) B[base * nc + e] = tile[e];
        __syncthreads();
    }
}
"""


_OPTIONS = ("-std=c++14", "--fmad=false")  # --fmad=false: every operation rounds where the separate cupy kernels round
_KERNELS_BY_REAL = {"double": LazyRawModule(_SRC.replace("{REAL}", "double"), _OPTIONS), "float": LazyRawModule(_SRC.replace("{REAL}", "float"), _OPTIONS)}


def build_basis_fused(cp: Any, basis: str, x: Any, n_columns: int) -> Optional[Any]:
    """``(n, n_columns)`` design of the named basis for the resident column ``x``, or ``None`` when the fused kernel does not cover the request (unknown basis, too wide).

    The design takes the dtype of ``x`` and the recurrence runs in that dtype, rounding where the cupy loop it replaces rounds."""
    code = _BASIS_CODE.get(basis)
    if code is None or n_columns < 1 or n_columns > MAX_COLUMNS:
        return None
    n = int(x.shape[0])
    from .._fe_gpu_vram import fe_gpu_has_vram_cushion

    if not fe_gpu_has_vram_cushion(n * n_columns * x.dtype.itemsize):
        return None
    B = cp.empty((n, n_columns), dtype=x.dtype)
    build = _KERNELS_BY_REAL["float" if x.dtype.itemsize == 4 else "double"].get(cp).get_function("build_basis")
    threads = block_size_for(build, work_items=n, dyn_smem=0)
    while threads > 32 and threads * n_columns * x.dtype.itemsize > _TILE_BYTES:
        threads //= 2  # the shared tile holds one row of ``n_columns`` values per thread
    blocks = max(1, min(4096, (n + threads - 1) // threads))
    build((blocks,), (threads,), (x, B, cp.int64(n), cp.int32(n_columns), cp.int32(code)), shared_mem=threads * n_columns * x.dtype.itemsize)
    return B


DEVICE_FALLBACK_ROWS = 0  # the single kernel won at every measured size (20000 to 1.6M rows, 8 to 13x over the loop); the tuner replaces this per host
_SWEEP_ROWS = [5_000, 100_000, 1_000_000]
_SWEEP_COLUMNS = 7
_SWEEP_BASES = ("hermite", "laguerre")
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """A resident float column of ``dims["n_samples"]`` rows in the critical dtype (the design is built from a resident column, so the sweep times the build alone)."""
    import cupy as cp
    import numpy as np

    from .._fe_gpu_batch._devices import crit_float_dtype

    return (cp.asarray(np.random.default_rng(0).standard_normal(int(dims["n_samples"]))).astype(crit_float_dtype()),)


def _designs(x: Any, build: Any) -> Any:
    """Both bases at degree ``_SWEEP_COLUMNS - 1`` stacked into one host array, so the comparison covers the recurrences with and without the Laguerre start."""
    import cupy as cp

    out = cp.concatenate([build(b, x) for b in _SWEEP_BASES], axis=1)
    return cp.asnumpy(out)


def _sweep_loop(x: Any) -> Any:
    """The cupy-op-per-degree loop, for the sweep."""
    import cupy as cp

    from ._hermite_prewarp_gpu_resident import _build_basis_matrix_loop

    return _designs(x, lambda b, v: _build_basis_matrix_loop(cp, b, v, _SWEEP_COLUMNS))


def _sweep_fused(x: Any) -> Any:
    """The single kernel, for the sweep."""
    import cupy as cp

    return _designs(x, lambda b, v: build_basis_fused(cp, b, v, _SWEEP_COLUMNS))


def _fallback_choice(n_samples: int) -> str:
    """Pre-sweep fallback: the single kernel from ``DEVICE_FALLBACK_ROWS`` rows."""
    return "fused" if int(n_samples) >= DEVICE_FALLBACK_ROWS else "loop"


def _run_sweep() -> list:
    """Time the loop and the kernel per row count (both on a resident column); the kernel must be bit-equal to the loop."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return list(
        sweep_backend_grid({"loop": _sweep_loop, "fused": _sweep_fused}, {"n_samples": _SWEEP_ROWS}, _sweep_inputs, reference="loop", repeats=3, equiv_rtol=1e-12, equiv_atol=1e-12)
    )


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="fe_basis_build_backend",
        variant_fns=(),  # the two builds are timed by the sweep callable
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_basis_build_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the single kernel is used
    logger.warning("kernel_tuner spec build for fe_basis_build_backend failed: %s", e)
    _SPEC = None


def basis_fused_chosen(n_rows: int) -> bool:
    """Whether a design of ``n_rows`` rows is built by the single kernel (the per-host kernel-tuning cache; ``MLFRAME_FE_BASIS_FUSED=0`` still forces the loop in the caller)."""
    if _SPEC is None:
        return True
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) != "loop"
    except Exception as exc:
        log_throttle(
            logger, "fe_basis_build_backend_choose_failed", logging.WARNING,
            "fe_basis_build_backend: KTC choose() failed (%s: %s), using the single kernel for this call", type(exc).__name__, exc,
        )
        return True
