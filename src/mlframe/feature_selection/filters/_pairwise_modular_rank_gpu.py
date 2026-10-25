"""GPU ranking of the modular-scan pair and triple candidates (``_pairwise_modular_fe._residue_rank_scores`` / ``_residue_rank_scores3`` on the device).

One CUDA block per candidate: the block loops over the pair operators (sum / diff / prod) or the triple sum and over the moduli, builds the
(residue, class) count matrix in shared memory with integer atomics and reduces the plug-in MI ``H(R) + H(Y) - H(R, Y)`` across its threads. The
estimate is the same as the CPU kernels', so the ranking only differs by the float summation order. ``None`` is returned when the shared-memory
count matrix does not fit or cupy is unavailable, and the caller falls back to the CPU kernels.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_BLOCK = 128
_SHARED_LIMIT_BYTES = 48 * 1024

_KERNEL_SRC = r"""
extern "C" __global__ void residue_rank(
    const long long* __restrict__ cols, const int n,
    const int* __restrict__ ia, const int* __restrict__ ib, const int* __restrict__ ic,
    const int mode, const int* __restrict__ mods, const int n_mods,
    const int* __restrict__ ycls, const int n_cls, const int max_k, const double hy,
    double* __restrict__ scores)
{
    extern __shared__ int sh[];
    __shared__ double red[128];
    int* joint = sh;
    int* marg = sh + max_k * n_cls;
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    const long long* ca = cols + (long long)ia[q] * n;
    const long long* cb = cols + (long long)ib[q] * n;
    const long long* cc = (mode == 1) ? cols + (long long)ic[q] * n : ca;
    const int n_ops = (mode == 1) ? 1 : 3;
    double best = 0.0;
    for (int op = 0; op < n_ops; ++op) {
        for (int mi = 0; mi < n_mods; ++mi) {
            const int k = mods[mi];
            const int cells = k * n_cls;
            for (int i = tid; i < cells; i += blockDim.x) joint[i] = 0;
            for (int i = tid; i < k; i += blockDim.x) marg[i] = 0;
            __syncthreads();
            for (int t = tid; t < n; t += blockDim.x) {
                long long v;
                if (mode == 1) v = ca[t] + cb[t] + cc[t];
                else if (op == 0) v = ca[t] + cb[t];
                else if (op == 1) v = ca[t] - cb[t];
                else v = ca[t] * cb[t];
                int r = (int)(((v % k) + k) % k);
                atomicAdd(&joint[r * n_cls + ycls[t]], 1);
                atomicAdd(&marg[r], 1);
            }
            __syncthreads();
            double local = 0.0;
            for (int r = tid; r < k; r += blockDim.x) {
                if (marg[r] > 0) { double pr = (double)marg[r] / n; local -= pr * log(pr); }
            }
            for (int j = tid; j < cells; j += blockDim.x) {
                if (joint[j] > 0) { double pj = (double)joint[j] / n; local += pj * log(pj); }
            }
            red[tid] = local;
            __syncthreads();
            for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
                if (tid < s) red[tid] += red[tid + s];
                __syncthreads();
            }
            if (tid == 0) {
                const double mi_val = hy + red[0];
                if (mi_val > best) best = mi_val;
            }
            __syncthreads();
        }
    }
    if (tid == 0) scores[q] = best;
}
"""

_kernel = None


def _get_kernel():
    """Compile the ranking kernel once per process."""
    global _kernel
    if _kernel is None:
        import cupy as cp

        _kernel = cp.RawKernel(_KERNEL_SRC, "residue_rank")
    return _kernel


def residue_rank_scores_gpu(
    cols: np.ndarray, ia: np.ndarray, ib: np.ndarray, ic: Optional[np.ndarray], mods: np.ndarray, ycls: np.ndarray, n_cls: int
) -> Optional[np.ndarray]:
    """Best plug-in residue MI per candidate on the device: pairs ``(ia, ib)`` over the three pair operators, or triples ``(ia, ib, ic)`` over ``sum3``.

    ``cols`` is the (p, n) int64 matrix, ``ycls`` the 0-based class codes. ``None`` when cupy is unavailable, the count matrix exceeds the shared
    memory or any step fails (the caller then runs the CPU kernel).
    """
    try:
        import cupy as cp

        n = int(ycls.shape[0])
        max_k = int(np.max(mods))
        shared = (max_k * int(n_cls) + max_k) * 4
        if shared > _SHARED_LIMIT_BYTES or ia.shape[0] == 0:
            return None
        n_ops = 1 if ic is not None else 3
        cols_d = cp.asarray(np.ascontiguousarray(cols, dtype=np.int64))
        ia_d = cp.asarray(ia.astype(np.int32))
        ib_d = cp.asarray(ib.astype(np.int32))
        ic_d = cp.asarray(ic.astype(np.int32)) if ic is not None else ia_d
        mods_d = cp.asarray(mods.astype(np.int32))
        y_d = cp.asarray(ycls.astype(np.int32))
        counts = np.bincount(ycls, minlength=int(n_cls)).astype(np.float64)
        nz = counts[counts > 0] / n
        hy = float(-np.sum(nz * np.log(nz)))
        scores = cp.zeros(ia.shape[0], dtype=cp.float64)
        _get_kernel()(
            (int(ia.shape[0]),),
            (_BLOCK,),
            (cols_d, np.int32(n), ia_d, ib_d, ic_d, np.int32(0 if n_ops == 3 else 1), mods_d, np.int32(mods.shape[0]), y_d, np.int32(n_cls), np.int32(max_k), np.float64(hy), scores),
            shared_mem=shared,
        )
        return cp.asnumpy(scores)
    except Exception as e:
        logger.warning("modular rank GPU kernel unavailable (%s), using the CPU kernel", e)
        return None


def _gpu_available() -> bool:
    """Whether the ranking kernel can run here: the project GPU policy allows the GPU and cupy sees a device."""
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return False
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # nosec B110 - no cupy or no device: the CPU kernels serve
        return False


# Measured on the GTX 1050 Ti host (synchronized, 1000 rows, 17 moduli): the CPU kernel costs about 50 microseconds per scored operator-candidate and the GPU kernel about
# 4 microseconds. In a fresh process the first call of 1140 triples takes 0.66 s on the CPU (numba cache load) and 0.17 s on the GPU (cupy's on-disk kernel cache), and
# the warm calls take 0.041 s and 0.006 s, so the crossover lies below the smallest cell of the first sweep; the grid now starts lower so the cache measures both sides.
_RANK_GPU_MIN_WORK = 1_000
_RANK_SWEEP_WORK_GRID = [100, 1_000, 5_000, 20_000, 80_000, 250_000]
_RANK_SALT = 1


def _rank_fallback_choice(n_work: int, **_axes) -> str:
    """Pre-sweep heuristic: the GPU from ``_RANK_GPU_MIN_WORK`` scored operator-candidates."""
    return "gpu" if int(n_work) >= _RANK_GPU_MIN_WORK else "cpu"


def _make_rank_inputs(dims: dict) -> tuple:
    """Synthetic int64 columns, class codes and ``dims['n_work']`` random triples for the CPU-vs-GPU sweep."""
    n_work = int(dims["n_work"])
    rng = np.random.default_rng(0)
    p, n = 64, 1000
    cols = rng.integers(0, 30, size=(p, n)).astype(np.int64)
    ycls = rng.integers(0, 2, size=n).astype(np.int64)
    idx = rng.integers(0, p, size=(n_work, 3)).astype(np.int64)
    mods = np.array([2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 23, 24, 32], dtype=np.int64)
    return cols, idx[:, 0].copy(), idx[:, 1].copy(), idx[:, 2].copy(), mods, ycls


def _run_rank_sweep() -> list:
    """Grid sweep over the scored work -> the fastest equivalent backend per cell (the njit kernel or the CUDA kernel)."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    from ._pairwise_modular_fe import _residue_rank_scores3

    def cpu(cols, ia, ib, ic, mods, ycls):
        """CPU variant: the njit triple kernel."""
        return _residue_rank_scores3(cols, ia, ib, ic, mods, ycls, int(ycls.max()) + 1)

    def gpu(cols, ia, ib, ic, mods, ycls):
        """GPU variant: the CUDA kernel, falling back to the CPU one if it is unavailable (then the sweep records equal timings)."""
        out = residue_rank_scores_gpu(cols, ia, ib, ic, mods, ycls, int(ycls.max()) + 1)
        return out if out is not None else cpu(cols, ia, ib, ic, mods, ycls)

    variants = {"cpu": cpu}
    if _gpu_available():
        variants["gpu"] = gpu
    return list(sweep_backend_grid(variants, {"n_work": _RANK_SWEEP_WORK_GRID}, _make_rank_inputs, reference="cpu", repeats=3, equiv_rtol=1e-9, equiv_atol=1e-12))


from pyutilz.performance.kernel_tuning.registry import kernel_tuner

_RANK_SPEC = kernel_tuner(
    kernel_name="modular_rank",
    variant_fns=(residue_rank_scores_gpu,),
    tuner=_run_rank_sweep,
    axes={"n_work": list(_RANK_SWEEP_WORK_GRID)},
    fallback=_rank_fallback_choice,
    gpu_capable=True,
    salt=_RANK_SALT,
    cli_label="modular_rank",
)


def rank_backend_choice(n_work: int) -> str:
    """Per-host backend (``"cpu"`` or ``"gpu"``) for ranking ``n_work`` operator-candidates through the kernel-tuning cache, the heuristic as the fallback."""
    if not _gpu_available():
        return "cpu"
    try:
        choice = _RANK_SPEC.choose(n_work=int(n_work))
    except Exception:  # nosec B110 - a broken cache must not break the scan; the heuristic decides
        choice = _rank_fallback_choice(n_work)
    return "gpu" if choice in ("gpu", "cuda", "cupy") else "cpu"
