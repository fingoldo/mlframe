"""Pair joint MI against K permuted targets in ONE launch: the inner loop of the order-2 maxT permutation-null floor.

The floor shuffles the target K times and records, per shuffle, the maximum plug-in joint MI ``I((x_a, x_b); y_perm)`` over the whole candidate pool. A pair's joint X-code ``x_a * nb_b + x_b`` and its
marginal counts do not depend on the shuffle, so one block builds them once and then replays the ``(joint, y_perm)`` contingency table for each of the K targets in shared memory.

The pool of a conjunction fit holds 31878 pairs; against 25 shuffles of 15000 rows that is 1.2e10 histogram updates. The CPU kernel with the hoisted encoding takes 8.5 s; the pair-MI CUDA kernels
that exist take 3.8 s PER SHUFFLE on this shape because they read the row-major matrix with a stride of one row per thread (no coalescing) and know one target at a time. Here the data is uploaded
column-major as bytes (consecutive threads read consecutive rows of a column), the K targets are a ``(K, n)`` byte matrix, and the result is the ``(K, n_pairs)`` MI matrix.

The MI is the same ``sum jf * log(jf / (px * py))`` over the occupied cells the CPU kernel evaluates; the counts are exact integers and the final sums add in a fixed order, so a run is reproducible,
and it agrees with the CPU kernel to the last bits of a double (about 1e-17 absolute). A pair whose joint cardinality exceeds the cap gets zeros, as on the CPU.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_THREADS = 256
#: Dynamic shared memory the histogram may use (the 48 KB every device grants, less the static warp sums and the driver's reserve).
_SHARED_BUDGET_BYTES = 44 * 1024
_KERNEL: Any = None
_KERNEL_LOCK = threading.Lock()


class DeviceKernelDeclinedError(RuntimeError):
    """Raised by a floor that was asked for the device kernel only when the kernel does not apply (no CUDA, the shape does not fit, the host prefers the CPU kernel)."""

_SRC = r"""
extern "C" __global__
void pair_mi_perm(const unsigned char* __restrict__ cols, const long long* __restrict__ pair_a, const long long* __restrict__ pair_b,
                  const int* __restrict__ nbins, const unsigned char* __restrict__ yperm, const double* __restrict__ freqs_y,
                  const long long n, const int n_pairs, const int K, const int ky, const long long cap, const double inv_n,
                  double* __restrict__ mi_out) {
    extern __shared__ int sh[];
    __shared__ double warp_sums[32];
    const int p = blockIdx.x;
    const int tid = threadIdx.x, nt = blockDim.x;
    const long long a = pair_a[p], b = pair_b[p];
    const long long nb_a = nbins[a], nb_b = nbins[b];
    const bool bad = (nb_a <= 0 || nb_b <= 0 || ky <= 0 || nb_b > cap / ky || nb_a > cap / (nb_b * ky));
    if (bad) {
        for (int k = tid; k < K; k += nt) mi_out[(long long)k * n_pairs + p] = 0.0;
        return;
    }
    const int joint = (int)(nb_a * nb_b);
    const int cells = joint * ky;
    int* hist = sh;
    int* fx = sh + cells;
    const unsigned char* ca = cols + a * n;
    const unsigned char* cb = cols + b * n;
    const int nbb = (int)nb_b;
    for (int i = tid; i < joint; i += nt) fx[i] = 0;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) atomicAdd(&fx[(int)ca[i] * nbb + (int)cb[i]], 1);
    __syncthreads();
    for (int k = 0; k < K; ++k) {
        for (int i = tid; i < cells; i += nt) hist[i] = 0;
        __syncthreads();
        const unsigned char* yk = yperm + (long long)k * n;
        for (long long i = tid; i < n; i += nt) atomicAdd(&hist[((int)ca[i] * nbb + (int)cb[i]) * ky + (int)yk[i]], 1);
        __syncthreads();
        double loc = 0.0;
        for (int idx = tid; idx < cells; idx += nt) {
            const int jc = hist[idx];
            if (jc == 0) continue;
            const int ci = idx / ky;
            const int cj = idx - ci * ky;
            const int fxc = fx[ci];
            if (fxc == 0) continue;
            const double py = freqs_y[cj];
            if (!(py > 0.0)) continue;
            const double px = (double)fxc * inv_n;
            const double jf = (double)jc * inv_n;
            loc += jf * log(jf / (px * py));
        }
        for (int off = 16; off > 0; off >>= 1) loc += __shfl_down_sync(0xffffffffu, loc, off);
        if ((tid & 31) == 0) warp_sums[tid >> 5] = loc;
        __syncthreads();
        if (tid == 0) {
            double total = 0.0;
            for (int w = 0; w < (nt >> 5); ++w) total += warp_sums[w];
            mi_out[(long long)k * n_pairs + p] = total;
        }
        __syncthreads();
    }
}
"""


def _kernel(cp: Any) -> Any:
    """The compiled kernel, built once per process."""
    global _KERNEL
    with _KERNEL_LOCK:
        if _KERNEL is None:
            _KERNEL = cp.RawKernel(_SRC, "pair_mi_perm")
        return _KERNEL


def pair_mi_perm_matrix_cuda(
    factors_data: np.ndarray, pair_a: np.ndarray, pair_b: np.ndarray, nbins: np.ndarray, y_perms: np.ndarray, freqs_y: np.ndarray, cap: int
) -> Optional[np.ndarray]:
    """``(K, n_pairs)`` plug-in joint MI of every pair against each row of ``y_perms``, or ``None`` when the device path does not apply.

    ``None`` means "use the CPU kernel": no CUDA, codes that do not fit a byte, a histogram that does not fit shared memory, or a pool that does not fit free VRAM. Pairs the CPU kernel zeroes (joint
    cardinality above ``cap``) are zeroed here too; every other pair must have codes below its ``nbins``.
    """
    try:
        import cupy as cp
    except ImportError:
        return None
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return None
    n, n_features = int(factors_data.shape[0]), int(factors_data.shape[1])
    n_pairs, K, ky = int(pair_a.shape[0]), int(y_perms.shape[0]), int(freqs_y.shape[0])
    if n == 0 or n_pairs == 0 or K == 0 or ky < 1 or ky > 255 or int(y_perms.shape[1]) != n:
        return None
    nb = np.asarray(nbins).astype(np.int64)
    a = np.ascontiguousarray(pair_a, dtype=np.int64)
    b = np.ascontiguousarray(pair_b, dtype=np.int64)
    valid = (nb[a] > 0) & (nb[b] > 0) & (nb[b] <= cap // ky) & (nb[a] <= cap // np.maximum(nb[b] * ky, 1))
    if not bool(valid.any()):
        return np.zeros((K, n_pairs), dtype=np.float64)
    max_joint = int((nb[a][valid] * nb[b][valid]).max())
    if (max_joint * ky + max_joint) * 4 > _SHARED_BUDGET_BYTES:
        return None
    if int(factors_data.min()) < 0 or int(factors_data.max()) > 255 or int(y_perms.min()) < 0 or int(y_perms.max()) >= ky:
        return None
    used = np.unique(np.concatenate([a[valid], b[valid]]))
    col_max = factors_data[:, used].max(axis=0).astype(np.int64)
    if bool((col_max >= nb[used]).any()):
        return None
    need = n_features * n + K * n + K * n_pairs * 8 + n_pairs * 16 + n_features * 4 + ky * 8
    try:
        free = int(cp.cuda.runtime.memGetInfo()[0])
        if need * 2 > free:
            return None
        cols = cp.asarray(np.ascontiguousarray(factors_data.T, dtype=np.uint8))
        d_a, d_b = cp.asarray(a), cp.asarray(b)
        d_nb = cp.asarray(nb.astype(np.int32))
        d_y = cp.asarray(np.ascontiguousarray(y_perms, dtype=np.uint8))
        d_fy = cp.asarray(np.ascontiguousarray(freqs_y, dtype=np.float64))
        out = cp.empty((K, n_pairs), dtype=cp.float64)
        shared = (max_joint * ky + max_joint) * 4
        _kernel(cp)((n_pairs,), (_THREADS,), (cols, d_a, d_b, d_nb, d_y, d_fy, np.int64(n), np.int32(n_pairs), np.int32(K), np.int32(ky), np.int64(cap), np.float64(1.0 / n), out), shared_mem=shared)
        return cp.asnumpy(out)
    except Exception as exc:  # nosec B110 - a device fault or an allocation failure falls back to the exact CPU kernel
        logger.warning("pair_mi_perm_matrix_cuda failed (%s: %s); using the CPU kernel", type(exc).__name__, exc)
        return None


# ---- backend choice through the kernel tuning cache --------------------------------------------------------------------------------------------------------------------------------------------
# The floor's CPU kernel and this one cross over where the CPU work (about n * n_pairs * K / 1.4e9 s on this host) stops hiding the upload and the launch. The per-host crossover is measured by
# the sweep below and stored in the cache; the fallback applies verbatim while the host has no entry. Measured here (GTX 1050 Ti, K=25, 10 bins): GPU already ahead at n=4000 x 1770 pairs
# (0.06 s vs 0.18 s) and at 15000 x 780 (0.05 s vs 0.23 s); the CPU kernel wins only for pools whose whole CPU time is a few tens of milliseconds.
_FLOOR_SWEEP_N_SAMPLES = [4000, 15000]
_FLOOR_SWEEP_N_PAIRS = [300, 2000, 8000, 32000]
_FLOOR_SWEEP_PERMUTATIONS = 25
_FLOOR_FALLBACK_MIN_WORK = 2_000_000  # n_samples * n_pairs from which the device kernel is the fallback choice
_FLOOR_SALT = 1


def _floor_fallback_choice(n_samples: int, n_pairs: int) -> str:
    """The backend while the host has no tuning-cache entry: the device kernel from ``_FLOOR_FALLBACK_MIN_WORK`` rows x pairs, the CPU kernel below it."""
    return "cuda" if int(n_samples) * int(n_pairs) >= _FLOOR_FALLBACK_MIN_WORK else "cpu"


def _floor_sweep_inputs(dims: dict) -> tuple:
    """Synthetic ``(factors_data, pair_a, pair_b, nbins, y_perms, freqs_y, cap)`` of ``dims['n_samples']`` rows and ``dims['n_pairs']`` pairs for the backend sweep."""
    from .info_theory.shared import joint_cardinality_cap

    rng = np.random.default_rng(0)
    n, n_pairs, n_cols, n_bins, ky = int(dims["n_samples"]), int(dims["n_pairs"]), 96, 10, 10
    data = rng.integers(0, n_bins, (n, n_cols)).astype(np.int8)
    pair_a = rng.integers(0, n_cols, n_pairs).astype(np.int64)
    pair_b = ((pair_a + 1 + rng.integers(0, n_cols - 1, n_pairs)) % n_cols).astype(np.int64)
    y = rng.integers(0, ky, n).astype(np.int64)
    freqs = np.bincount(y, minlength=ky).astype(np.float64) / n
    y_perms = np.stack([rng.permutation(y) for _ in range(_FLOOR_SWEEP_PERMUTATIONS)])
    return data, pair_a, pair_b, np.full(n_cols, n_bins, dtype=np.int64), y_perms, freqs, joint_cardinality_cap()


def _run_floor_sweep() -> list:
    """Sweep ``cpu`` (the permutation-batched njit kernel) against ``cuda`` (this kernel) over rows x pairs; fastest equivalent backend per cell."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    from .info_theory import batch_pair_mi_perm_batched

    def _cuda(*args: Any) -> np.ndarray:
        """The device kernel, raising when it declines so the sweep leaves that cell to the CPU."""
        out = pair_mi_perm_matrix_cuda(*args)
        if out is None:
            raise RuntimeError("the device kernel declined this shape")
        return out

    return sweep_backend_grid(
        {"cpu": lambda *a: batch_pair_mi_perm_batched(*a), "cuda": _cuda},
        {"n_samples": _FLOOR_SWEEP_N_SAMPLES, "n_pairs": _FLOOR_SWEEP_N_PAIRS},
        _floor_sweep_inputs,
        reference="cpu",
        repeats=3,
        equiv_rtol=1e-9,
        equiv_atol=1e-13,
    )


def _register_floor_spec() -> Any:
    """Register the backend choice with the kernel tuning registry (discovered by ``mlframe-tune-kernels``)."""
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    from .info_theory import batch_pair_mi_perm_batched

    return kernel_tuner(
        kernel_name="pair_mi_perm_floor",
        variant_fns=(batch_pair_mi_perm_batched,),
        tuner=_run_floor_sweep,
        axes={"n_samples": list(_FLOOR_SWEEP_N_SAMPLES), "n_pairs": list(_FLOOR_SWEEP_N_PAIRS)},
        fallback=_floor_fallback_choice,
        gpu_capable=True,
        salt=_FLOOR_SALT,
        cli_label="pair_mi_perm_floor",
    )


_FLOOR_SPEC = _register_floor_spec()


def floor_backend_choice(n_samples: int, n_pairs: int) -> str:
    """``"cuda"`` or ``"cpu"`` for a floor over ``n_samples`` rows and ``n_pairs`` pairs, from the host's tuning-cache entry (or the fallback)."""
    try:
        return str(_FLOOR_SPEC.choose(n_samples=int(n_samples), n_pairs=int(n_pairs)))
    except Exception as exc:  # nosec B110 - a cache that cannot be read leaves the fallback in charge
        logger.debug("pair_mi_perm_floor choose() failed (%s); using the fallback", exc)
        return _floor_fallback_choice(n_samples, n_pairs)
