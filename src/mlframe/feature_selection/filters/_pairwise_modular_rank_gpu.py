"""GPU ranking of the modular-scan pair and triple candidates (``_pairwise_modular_fe._residue_rank_scores`` / ``_residue_rank_scores3`` on the device).

One CUDA block per candidate: the block loops over the pair operators (sum / diff / prod) or the triple sum and over the moduli, builds the
(residue, class) count matrix in shared memory with integer atomics and reduces the plug-in MI ``H(R) + H(Y) - H(R, Y)`` across its threads. The
estimate is the same as the CPU kernels', so the ranking only differs by the float summation order. ``None`` is returned when the shared-memory
count matrix does not fit or cupy is unavailable, and the caller falls back to the CPU kernels.
"""

from __future__ import annotations

import logging
from typing import Optional, cast

import numpy as np

logger = logging.getLogger(__name__)

_BLOCK = 128
_SHARED_LIMIT_BYTES = 48 * 1024

_KERNEL_TEMPLATE = r"""
extern "C" __BOUNDS__ __global__ void residue_rank(
    const __COLT__* __restrict__ cols, const int n,
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
    const __COLT__* ca = cols + (long long)ia[q] * n;
    const __COLT__* cb = cols + (long long)ib[q] * n;
    const __COLT__* cc = (mode == 1) ? cols + (long long)ic[q] * n : ca;
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
                __COLT__ v;
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
__REDUCE__
        }
    }
    if (tid == 0) scores[q] = best;
}
"""

# Version 1: shared-memory tree reduction, about ten block barriers per (operator, modulus). Kept as the reference version.
_REDUCE_V1 = """            red[tid] = local;
            __syncthreads();
            for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
                if (tid < s) red[tid] += red[tid + s];
                __syncthreads();
            }
            if (tid == 0) {
                const double mi_val = hy + red[0];
                if (mi_val > best) best = mi_val;
            }
            __syncthreads();"""

# Version 2: warp-shuffle reduction, one barrier after the four warp sums are written (the pair kernel's v2 measured the same stall drop).
_REDUCE_V2 = """            for (int o = 16; o > 0; o >>= 1) local += __shfl_down_sync(0xffffffffu, local, o);
            if ((tid & 31) == 0) red[tid >> 5] = local;
            __syncthreads();
            if (tid == 0) {
                const double mi_val = hy + red[0] + red[1] + red[2] + red[3];
                if (mi_val > best) best = mi_val;
            }"""

# Version 4: version 2's reduction with fewer live registers - 32-bit row offsets instead of three 64-bit column pointers, and the running best kept in shared memory
# by thread 0 instead of in every thread's registers - so the kernel can meet the 32-register cap of full occupancy on Pascal without spilling to local memory.
_KERNEL_V4_TEMPLATE = r"""
extern "C" __BOUNDS__ __global__ void residue_rank(
    const __COLT__* __restrict__ cols, const int n,
    const int* __restrict__ ia, const int* __restrict__ ib, const int* __restrict__ ic,
    const int mode, const int* __restrict__ mods, const int n_mods,
    const int* __restrict__ ycls, const int n_cls, const int max_k, const double hy,
    double* __restrict__ scores)
{
    extern __shared__ int sh[];
    __shared__ double red[4];
    __shared__ double best_sh;
    int* joint = sh;
    int* marg = sh + max_k * n_cls;
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    const int oa = ia[q] * n;  // width-ok: a row offset below p * n; the host refuses inputs whose element count reaches 2**31
    const int ob = ib[q] * n;  // width-ok: same bound
    const int oc = (mode == 1) ? ic[q] * n : oa;  // width-ok: same bound
    const int n_ops = (mode == 1) ? 1 : 3;
    if (tid == 0) best_sh = 0.0;
    for (int op = 0; op < n_ops; ++op) {
        for (int mi = 0; mi < n_mods; ++mi) {
            const int k = mods[mi];
            const int cells = k * n_cls;
            for (int i = tid; i < cells; i += blockDim.x) joint[i] = 0;
            for (int i = tid; i < k; i += blockDim.x) marg[i] = 0;
            __syncthreads();
            for (int t = tid; t < n; t += blockDim.x) {
                __COLT__ v;
                if (mode == 1) v = cols[oa + t] + cols[ob + t] + cols[oc + t];
                else if (op == 0) v = cols[oa + t] + cols[ob + t];
                else if (op == 1) v = cols[oa + t] - cols[ob + t];
                else v = cols[oa + t] * cols[ob + t];
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
            for (int o = 16; o > 0; o >>= 1) local += __shfl_down_sync(0xffffffffu, local, o);
            if ((tid & 31) == 0) red[tid >> 5] = local;
            __syncthreads();
            if (tid == 0) {
                const double mi_val = hy + red[0] + red[1] + red[2] + red[3];
                if (mi_val > best_sh) best_sh = mi_val;
            }
        }
    }
    if (tid == 0) scores[q] = best_sh;
}
"""

# Version 6: version 4 with one private (residue, class) count matrix per warp (four copies), so the atomics of a warp no longer collide with the other three warps' and
# the copies are summed when the entropies are formed. The target is the shared-memory atomic contention on a handful of cells (shared efficiency 0.3 to 4 percent).
_KERNEL_V6_TEMPLATE = _KERNEL_V4_TEMPLATE.replace(
    "    int* joint = sh;\n    int* marg = sh + max_k * n_cls;\n",
    "    const int sj = max_k * n_cls;\n    const int sm = max_k;\n    int* joint = sh;\n    int* marg = sh + 4 * sj;\n    const int wp = threadIdx.x >> 5;\n",
).replace(
    "            for (int i = tid; i < cells; i += blockDim.x) joint[i] = 0;\n            for (int i = tid; i < k; i += blockDim.x) marg[i] = 0;\n",
    "            for (int i = tid; i < 4 * cells; i += blockDim.x) joint[(i / cells) * sj + (i % cells)] = 0;\n            for (int i = tid; i < 4 * k; i += blockDim.x) marg[(i / k) * sm + (i % k)] = 0;\n",
).replace(
    "                atomicAdd(&joint[r * n_cls + ycls[t]], 1);\n                atomicAdd(&marg[r], 1);\n",
    "                atomicAdd(&joint[wp * sj + r * n_cls + ycls[t]], 1);\n                atomicAdd(&marg[wp * sm + r], 1);\n",
).replace(
    "                if (marg[r] > 0) { double pr = (double)marg[r] / n; local -= pr * log(pr); }\n",
    "                const int mr = marg[r] + marg[sm + r] + marg[2 * sm + r] + marg[3 * sm + r];\n                if (mr > 0) { double pr = (double)mr / n; local -= pr * log(pr); }\n",
).replace(
    "                if (joint[j] > 0) { double pj = (double)joint[j] / n; local += pj * log(pj); }\n",
    "                const int cj = joint[j] + joint[sj + j] + joint[2 * sj + j] + joint[3 * sj + j];\n                if (cj > 0) { double pj = (double)cj / n; local += pj * log(pj); }\n",
)
# Version 7: no division in the row loop. The residues of every column at every modulus are computed once into an int8 table (p, n_mods, n); (a + b) mod k is then
# ra + rb with one conditional subtraction, a - b one conditional addition, a + b + c two, and a * b a lookup in a k x k table built once per modulus in shared memory.
# The result is the same as the division form because a residue of a sum, difference or product is the same operation on the residues.
_KERNEL_V7_TEMPLATE = r"""
extern "C" __BOUNDS__ __global__ void residue_rank(
    const signed char* __restrict__ R, const int n, const int n_mods,
    const int* __restrict__ ia, const int* __restrict__ ib, const int* __restrict__ ic,
    const int mode, const int* __restrict__ mods,
    const int* __restrict__ ycls, const int n_cls, const int max_k, const double hy,
    double* __restrict__ scores)
{
    extern __shared__ int sh[];
    __shared__ double red[4];
    __shared__ double best_sh;
    int* joint = sh;
    int* marg = sh + max_k * n_cls;
    unsigned char* ptab = (unsigned char*)(marg + max_k);
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_ops = (mode == 1) ? 1 : 3;
    if (tid == 0) best_sh = 0.0;
    for (int op = 0; op < n_ops; ++op) {
        for (int mi = 0; mi < n_mods; ++mi) {
            const int k = mods[mi];
            const int cells = k * n_cls;
            for (int i = tid; i < cells; i += blockDim.x) joint[i] = 0;
            for (int i = tid; i < k; i += blockDim.x) marg[i] = 0;
            if (mode == 0 && op == 2) {
                for (int i = tid; i < k * k; i += blockDim.x) ptab[i] = (unsigned char)(((i / k) * (i % k)) % k);
            }
            __syncthreads();
            const signed char* pa = R + ((long long)ia[q] * n_mods + mi) * n;
            const signed char* pb = R + ((long long)ib[q] * n_mods + mi) * n;
            const signed char* pc = (mode == 1) ? R + ((long long)ic[q] * n_mods + mi) * n : pa;
            for (int t = tid; t < n; t += blockDim.x) {
                const int ra = pa[t];
                const int rb = pb[t];
                int r;
                if (mode == 1) { r = ra + rb + (int)pc[t]; if (r >= 2 * k) r -= 2 * k; else if (r >= k) r -= k; }
                else if (op == 0) { r = ra + rb; if (r >= k) r -= k; }
                else if (op == 1) { r = ra - rb; if (r < 0) r += k; }
                else r = ptab[ra * k + rb];
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
            for (int o = 16; o > 0; o >>= 1) local += __shfl_down_sync(0xffffffffu, local, o);
            if ((tid & 31) == 0) red[tid >> 5] = local;
            __syncthreads();
            if (tid == 0) {
                const double mi_val = hy + red[0] + red[1] + red[2] + red[3];
                if (mi_val > best_sh) best_sh = mi_val;
            }
        }
    }
    if (tid == 0) scores[q] = best_sh;
}
"""

# Version 8: version 7 with one shared atomic per row instead of two. The residue marginal is not counted; it is the sum of the residue's row of the (residue, class)
# counts, formed when the entropies are computed.
_KERNEL_V8_TEMPLATE = _KERNEL_V7_TEMPLATE.replace(
    "                atomicAdd(&marg[r], 1);\n",
    "",
).replace(
    "                if (marg[r] > 0) { double pr = (double)marg[r] / n; local -= pr * log(pr); }\n",
    "                int mr = 0;\n                for (int y = 0; y < n_cls; ++y) mr += joint[r * n_cls + y];\n                if (mr > 0) { double pr = (double)mr / n; local -= pr * log(pr); }\n",
)
_KERNEL_COPIES = {6: 4}  # versions that keep several count-matrix copies in shared memory
_TABLE_VERSIONS = {7, 8}  # versions that read the int8 residue table instead of the columns
_TABLE_MAX_BYTES = 1 << 30  # the residue table is refused above this size and the caller falls back to the CPU kernel

_KERNEL_SOURCES = {
    1: _KERNEL_TEMPLATE.replace("__BOUNDS__", "").replace("__REDUCE__", _REDUCE_V1),
    2: _KERNEL_TEMPLATE.replace("__BOUNDS__", "").replace("__REDUCE__", _REDUCE_V2),
    # Version 3: version 2 capped at 32 registers per thread (__launch_bounds__(128, 16)) so a Pascal SM holds 16 blocks, i.e. full occupancy.
    3: _KERNEL_TEMPLATE.replace("__BOUNDS__", "__launch_bounds__(128, 16)").replace("__REDUCE__", _REDUCE_V2),
    4: _KERNEL_V4_TEMPLATE.replace("__BOUNDS__", "__launch_bounds__(128, 16)"),
    5: _KERNEL_V4_TEMPLATE.replace("__BOUNDS__", ""),  # version 4 without the cap: shows the registers the leaner body needs by itself
    6: _KERNEL_V6_TEMPLATE.replace("__BOUNDS__", "__launch_bounds__(128, 16)"),
    7: _KERNEL_V7_TEMPLATE.replace("__BOUNDS__", "__launch_bounds__(128, 16)"),
    8: _KERNEL_V8_TEMPLATE.replace("__BOUNDS__", "__launch_bounds__(128, 16)"),
}
_DEFAULT_VERSION = 8
_kernels: dict = {}


def _kernel_version() -> int:
    """The kernel version to run: ``MLFRAME_MODULAR_RANK_KERNEL`` (1, 2 or 3) when set, else the measured default."""
    import os

    try:
        v = int(os.environ.get("MLFRAME_MODULAR_RANK_KERNEL", _DEFAULT_VERSION))
    except ValueError:
        v = _DEFAULT_VERSION
    return v if v in _KERNEL_SOURCES else _DEFAULT_VERSION


_INT32_PRODUCT_LIMIT = 2**31 - 1


def _get_kernel(version: Optional[int] = None, wide: bool = True):
    """Compile the ranking kernel of ``version`` once per process; ``wide`` keeps the values in 64 bits, else they are 32-bit (cheaper modulo)."""
    v = _kernel_version() if version is None else version
    key = (v, wide)
    if key not in _kernels:
        import cupy as cp

        src = _KERNEL_SOURCES[v].replace("__COLT__", "long long" if wide else "int")
        _kernels[key] = cp.RawKernel(src, "residue_rank")
    return _kernels[key]


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
        version = _kernel_version()
        shared = _KERNEL_COPIES.get(version, 1) * (max_k * int(n_cls) + max_k) * 4
        if version in _TABLE_VERSIONS:
            shared += ((max_k * max_k + 3) // 4) * 4  # the k x k product table
        if shared > _SHARED_LIMIT_BYTES or ia.shape[0] == 0 or cols.size >= 2**31:  # the v4 kernel addresses rows with 32-bit offsets
            return None
        if version in _TABLE_VERSIONS and cols.shape[0] * mods.shape[0] * n > _TABLE_MAX_BYTES:
            return None
        n_ops = 1 if ic is not None else 3
        # 32-bit values when no pair product or triple sum can overflow an int: the modulo in the inner loop is then a 32-bit operation.
        biggest = int(np.max(np.abs(cols))) if cols.size else 0
        wide = biggest * biggest > _INT32_PRODUCT_LIMIT or 3 * biggest > _INT32_PRODUCT_LIMIT
        cols_d = cp.asarray(np.ascontiguousarray(cols, dtype=np.int64 if wide else np.int32))
        ia_d = cp.asarray(ia.astype(np.int32))
        ib_d = cp.asarray(ib.astype(np.int32))
        ic_d = cp.asarray(ic.astype(np.int32)) if ic is not None else ia_d
        mods_d = cp.asarray(mods.astype(np.int32))
        y_d = cp.asarray(ycls.astype(np.int32))
        counts = np.bincount(ycls, minlength=int(n_cls)).astype(np.float64)
        nz = counts[counts > 0] / n
        hy = float(-np.sum(nz * np.log(nz)))
        scores = cp.zeros(ia.shape[0], dtype=cp.float64)
        if version in _TABLE_VERSIONS:
            table = cp.empty((cols.shape[0], mods.shape[0], n), dtype=cp.int8)
            for mi in range(mods.shape[0]):
                table[:, mi, :] = cp.mod(cols_d, int(mods[mi])).astype(cp.int8)  # floored residue, 0 <= r < k
            _get_kernel(version, wide)(
                (int(ia.shape[0]),),
                (_BLOCK,),
                (table, np.int32(n), np.int32(mods.shape[0]), ia_d, ib_d, ic_d, np.int32(0 if n_ops == 3 else 1), mods_d, y_d, np.int32(n_cls), np.int32(max_k), np.float64(hy), scores),
                shared_mem=shared,
            )
            return cast(np.ndarray, cp.asnumpy(scores))
        _get_kernel(None, wide)(
            (int(ia.shape[0]),),
            (_BLOCK,),
            (cols_d, np.int32(n), ia_d, ib_d, ic_d, np.int32(0 if n_ops == 3 else 1), mods_d, np.int32(mods.shape[0]), y_d, np.int32(n_cls), np.int32(max_k), np.float64(hy), scores),
            shared_mem=shared,
        )
        return cast(np.ndarray, cp.asnumpy(scores))
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
