"""Device twin of ``_cat_pair_joint_njit.pair_joint_mi``: the plug-in MI of the joint of two categorical code columns with the binned target, for many pairs.

One block per pair: the rows fill a shared ``(radix_i * radix_j, k_y)`` count table (integer atomics, so the counts do not depend on thread order), then the threads sum ``p * log(p * n / (Px * Py / n))`` over the cells,
the same expression as ``_plug_in_mi_njit``; the joint key is ``code_i * radix_j + code_j`` and the plug-in estimate does not depend on how the cells are numbered, so the host's first-appearance renumbering is not needed.
A pair whose table does not fit the shared memory (``_MAX_CELLS``) keeps the host value. Pairs go to the device in blocks, and each block uploads only the code columns its pairs reference (``run_column_blocks`` halves the
block on an out-of-memory error). Host or device is chosen per call from the kernel-tuning entry ``cat_pair_joint_backend`` by the work ``n * pairs``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback

logger = logging.getLogger(__name__)

__all__ = ["pair_joint_mi_gpu", "pair_joint_mi_dispatch", "pair_device_chosen"]

_MAX_CELLS = 8192  # joint cells (radix_i * radix_j * k_y) that fit the shared count table (32 KB)
_THREADS = 256

_SRC = render(
    r"""
extern "C" __global__ __launch_bounds__(256)
void cat_pair_joint_mi(const int* __restrict__ codes, const int* __restrict__ radix, const int* __restrict__ first, const int* __restrict__ second,
                       const int* __restrict__ y, const {ACC}* __restrict__ py, const long long n, const int k_y, {ACC}* __restrict__ out) {
    extern __shared__ int tab[];                         // counts[cells * k_y]
    __shared__ {ACC} red[32];
    const int p = blockIdx.x;
    const int a = first[p], b = second[p];
    const int rb = radix[b];
    const int cells = radix[a] * rb;
    const int total = cells * k_y;
    for (int t = threadIdx.x; t < total; t += blockDim.x) tab[t] = 0;
    __syncthreads();
    const int* ca = codes + (long long)a * n;
    const int* cb = codes + (long long)b * n;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) atomicAdd(&tab[(ca[i] * rb + cb[i]) * k_y + y[i]], 1);
    __syncthreads();
    const {ACC} nf = ({ACC})n;
    {ACC} mi = 0.0;
    for (int c = threadIdx.x; c < cells; c += blockDim.x) {
        int px_i = 0;
        for (int j = 0; j < k_y; ++j) px_i += tab[c * k_y + j];
        if (px_i <= 0) continue;
        const {ACC} px = ({ACC})px_i;
        for (int j = 0; j < k_y; ++j) {
            const {ACC} v = ({ACC})tab[c * k_y + j];
            if (v <= 0.0 || py[j] <= 0.0) continue;
            const {ACC} pr = v / nf;
            mi += pr * log(pr * nf / (px * py[j] / nf));
        }
    }
    for (int o = 16; o > 0; o >>= 1) mi += __shfl_down_sync(0xffffffffu, mi, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = mi;
    __syncthreads();
    if (threadIdx.x == 0) {
        {ACC} total_mi = 0.0;
        for (int w = 0; w < (blockDim.x >> 5); ++w) total_mi += red[w];
        out[p] = total_mi;
    }
}
""",
)

_KERNEL: Any = None


def _kernel(cp: Any) -> Any:
    """The compiled kernel (one per process)."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(_SRC, "cat_pair_joint_mi")
    return _KERNEL


def pair_joint_mi_gpu(codes: np.ndarray, radix: np.ndarray, first: np.ndarray, second: np.ndarray, y_b: np.ndarray, k_y: int) -> np.ndarray:
    """Device twin of ``pair_joint_mi``; entries of pairs whose table exceeds the shared memory are NaN (the caller fills them from the host). Raises on a device fault."""
    import cupy as cp

    from ._device_blocks import first_block, run_column_blocks

    n_pairs = int(first.shape[0])
    out = np.full(n_pairs, np.nan)  # Intentional float64: MI statistic vector compared to the host at full precision
    n = int(codes.shape[1])
    if n_pairs == 0 or n == 0:
        return out
    fits = np.nonzero(radix[first] * radix[second] * int(k_y) <= _MAX_CELLS)[0]
    if fits.size == 0:
        return out
    y_d = cp.asarray(np.ascontiguousarray(y_b, dtype=np.int32))
    py = np.bincount(y_b, minlength=int(k_y)).astype(np.float64)  # Intentional float64: class counts of the MI statistic
    py_d = cp.asarray(py)
    kernel = _kernel(cp)

    def block_work(lo: int, hi: int) -> int:
        """Score the pairs ``fits[lo:hi]``, uploading only the code columns they reference."""
        sel = fits[lo:hi]
        used = np.unique(np.concatenate([first[sel], second[sel]]))
        remap = np.zeros(codes.shape[0], dtype=np.int32)
        remap[used] = np.arange(used.shape[0], dtype=np.int32)
        codes_d = cp.asarray(np.ascontiguousarray(codes[used], dtype=np.int32))
        radix_d = cp.asarray(np.ascontiguousarray(radix[used], dtype=np.int32))
        f_d = cp.asarray(remap[first[sel]])
        s_d = cp.asarray(remap[second[sel]])
        res = cp.empty(sel.shape[0], dtype=cp.float64)  # Intentional float64: MI statistic output
        shared = int((radix[first[sel]] * radix[second[sel]] * int(k_y)).max()) * 4
        kernel((int(sel.shape[0]),), (_THREADS,), (codes_d, radix_d, f_d, s_d, y_d, py_d, np.int64(n), np.int32(k_y), res), shared_mem=shared)  # launch-ok: one block per pair is the design
        out[sel] = cp.asnumpy(res)
        return int(sel.shape[0])

    block = first_block(cp, n, int(fits.size), 8, 1)
    run_column_blocks(cp, int(fits.size), block, block_work)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DEVICE_FALLBACK_ROWS = 3_000_000  # rows x pairs from which the device scores the pairs before a sweep has run (crossover between 40 and 160 pairs of 20000 rows)
_SWEEP_N = 20_000
_SWEEP_ROWS = [_SWEEP_N * k for k in (10, 40, 160, 640)]
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """Six categorical columns of 12 levels, ``dims["n_samples"] / 20000`` pairs over them, a 3-class target that depends on the first two."""
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 12, (6, _SWEEP_N)).astype(np.int64)
    radix = codes.max(axis=1) + 1
    n_pairs = max(1, int(dims["n_samples"]) // _SWEEP_N)
    first = rng.integers(0, 6, n_pairs).astype(np.int64)
    second = ((first + 1 + rng.integers(0, 5, n_pairs)) % 6).astype(np.int64)
    y = ((codes[0] + codes[1]) % 3).astype(np.int64)
    return (codes, radix, first, second, y, 3)


def _sweep_host(codes: np.ndarray, radix: np.ndarray, first: np.ndarray, second: np.ndarray, y: np.ndarray, k_y: int) -> np.ndarray:
    """The numba pass."""
    from ._cat_pair_joint_njit import pair_joint_mi

    return np.asarray(pair_joint_mi(codes, radix, first, second, y, k_y))


def _sweep_device(codes: np.ndarray, radix: np.ndarray, first: np.ndarray, second: np.ndarray, y: np.ndarray, k_y: int) -> np.ndarray:
    """The device kernel; every pair of the sweep fits, so a failed launch raises and is never recorded as a host time."""
    return pair_joint_mi_gpu(codes, radix, first, second, y, k_y)


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="cat_pair_joint_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-9, atol=1e-9),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__),
        gpu_capable=True,
        salt=_SALT,
        cli_label="cat_pair_joint_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for cat_pair_joint_backend failed: %s", e)
    _SPEC = None


def pair_device_chosen(n: int, n_pairs: int) -> bool:
    """Whether ``n_pairs`` pairs over ``n`` rows are scored on the device: the per-host cache by the work ``n * pairs``, else the ``DEVICE_FALLBACK_ROWS`` floor."""
    work = int(n) * int(n_pairs)
    if _SPEC is None:
        return work >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=work)) == "device"
    except Exception as exc:
        logger.debug("cat_pair_joint_backend: KTC choose() failed, using the floor: %s", exc)
        return work >= DEVICE_FALLBACK_ROWS


def pair_joint_mi_dispatch(codes: np.ndarray, radix: np.ndarray, first: np.ndarray, second: np.ndarray, y_b: np.ndarray, k_y: int) -> np.ndarray:
    """``pair_joint_mi`` on the device when the tuned choice says so and a device is usable (pairs with a table too big for shared memory, and every pair after a device fault, from the numba pass), else the numba pass."""
    from ._cat_pair_joint_njit import pair_joint_mi

    if first.shape[0] and pair_device_chosen(codes.shape[1], first.shape[0]):
        try:
            from ._gpu_policy import cuda_available_for_run

            if cuda_available_for_run():
                out = pair_joint_mi_gpu(codes, radix, first, second, y_b, k_y)
                todo = np.nonzero(np.isnan(out))[0]
                if todo.size:
                    out[todo] = pair_joint_mi(codes, radix, first[todo], second[todo], y_b, k_y)
                return out
        except Exception as exc:  # a device fault must not lose the stage
            from mlframe.utils.log_throttle import log_throttle

            log_throttle(logger, "cat_pair_joint.device_failed", logging.WARNING, "categorical pair joint MI device path failed (%s: %s); numba pass", type(exc).__name__, exc)
    return np.asarray(pair_joint_mi(codes, radix, first, second, y_b, k_y))
