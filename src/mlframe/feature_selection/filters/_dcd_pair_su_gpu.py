"""Device twin of the pair joint entropies ``H(X_a, X_b)`` of ``_dcd_pair_su_batch`` (``_batch_joint_entropy_pairs`` / ``_batch_joint_entropy_columns``).

One block per pair: the rows fill a shared ``nb_a * nb_b`` integer histogram (integer atomics, so thread order does not change the counts), the threads add ``p ln p`` over the occupied cells and one warp-shuffle block
reduction gives ``-sum``. The host sums the cells in ascending order, the device in a different order, so a value agrees to about one ULP (the same selection-equivalence the multi-pair numba kernel documents).
A pair whose histogram does not fit the shared memory (``_MAX_CELLS``) returns NaN and keeps the host value. Pairs go to the device in blocks, and each block uploads only the code columns its pairs reference as
contiguous rows (a column block of the code matrix, halved on an out-of-memory error by ``run_column_blocks``). Host or device is chosen per call from the kernel-tuning entry ``dcd_pair_entropy_backend`` by rows x pairs.
"""

from __future__ import annotations

import logging
from typing import Any

import threading
import weakref

import numpy as np

from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback

logger = logging.getLogger(__name__)

__all__ = ["joint_entropies_gpu", "joint_entropies_dispatch", "entropy_device_chosen"]

_MAX_CELLS = 8192  # histogram cells (nb_a * nb_b) that fit the shared table
_THREADS = 256

_SRC = render(
    r"""
extern "C" __global__ __launch_bounds__(256)
void pair_joint_entropy(const int* __restrict__ cols, const int* __restrict__ a_slot, const int* __restrict__ b_slot, const int* __restrict__ nb_a_arr,
                        const int* __restrict__ nb_b_arr, const long long n, {ACC}* __restrict__ out) {
    extern __shared__ int hist[];
    __shared__ {ACC} red[32];
    const int p = blockIdx.x;
    const int nb_a = nb_a_arr[p];
    const int size = nb_a * nb_b_arr[p];
    for (int c = threadIdx.x; c < size; c += blockDim.x) hist[c] = 0;
    __syncthreads();
    const int* ca = cols + (long long)a_slot[p] * n;
    const int* cb = cols + (long long)b_slot[p] * n;
    for (long long r = threadIdx.x; r < n; r += blockDim.x) atomicAdd(&hist[ca[r] + cb[r] * nb_a], 1);
    __syncthreads();
    {ACC} h = 0.0;
    for (int c = threadIdx.x; c < size; c += blockDim.x) {
        const int cnt = hist[c];
        if (cnt != 0) {
            const {ACC} pr = ({ACC})cnt / ({ACC})n;
            h += log(pr) * pr;
        }
    }
    for (int o = 16; o > 0; o >>= 1) h += __shfl_down_sync(0xffffffffu, h, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = h;
    __syncthreads();
    if (threadIdx.x == 0) {
        {ACC} total = 0.0;
        for (int w = 0; w < (blockDim.x >> 5); ++w) total += red[w];
        out[p] = -total;
    }
}
""",
)

_KERNEL: Any = None


def _kernel(cp: Any) -> Any:
    """The compiled kernel (one per process)."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(_SRC, "pair_joint_entropy")
    return _KERNEL


# The code matrix of the pair sweep is round-constant: it is uploaded once (int16, 2 bytes a code) and the columns of a block are gathered on the device, instead of a host transposed copy of
# the block columns converted to int32 and uploaded per block (about 19 ms of host work and 20 MB of upload per block at 100000 rows). Two entries; the key is the array object (checked through a
# weak reference, so a recycled id cannot answer for another matrix), its shape, dtype and data address.
_RESIDENT_FD: list = []  # [(weakref, key, device matrix)], newest last
_RESIDENT_FD_MAX = 2
_RESIDENT_FD_LOCK = threading.Lock()


def _resident_codes(cp: Any, fd: np.ndarray) -> "Any | None":
    """The ``(n, p)`` code matrix ``fd`` as a resident int16 array, uploaded once per matrix; ``None`` when the codes do not fit int16 (the per-block host path then runs)."""
    if fd.ndim != 2 or fd.size == 0:
        return None
    key = (id(fd), tuple(fd.shape), str(fd.dtype), int(fd.ctypes.data))
    with _RESIDENT_FD_LOCK:
        for ref, k, dev in _RESIDENT_FD:
            if k == key and ref() is fd:
                return dev
    if int(fd.min()) < 0 or int(fd.max()) > 32767:
        return None
    dev = cp.asarray(np.ascontiguousarray(fd, dtype=np.int16))
    with _RESIDENT_FD_LOCK:
        _RESIDENT_FD.append((weakref.ref(fd), key, dev))
        del _RESIDENT_FD[:-_RESIDENT_FD_MAX]
    return dev


def joint_entropies_gpu(fd: np.ndarray, a_arr: np.ndarray, b_arr: np.ndarray, nb_arr: np.ndarray) -> np.ndarray:
    """``H(X_a, X_b)`` per pair from the ``(n, p)`` code matrix ``fd``; NaN for a pair over the shared-memory limit. Raises on a device fault."""
    import cupy as cp

    from ._device_blocks import first_block, run_column_blocks

    k = int(a_arr.shape[0])
    n = int(fd.shape[0])
    out = np.full(k, np.nan)  # Intentional float64: entropy statistic compared to the host at full precision
    nb_a_all = nb_arr[a_arr].astype(np.int64)
    nb_b_all = nb_arr[b_arr].astype(np.int64)
    fits = np.nonzero(nb_a_all * nb_b_all <= _MAX_CELLS)[0]
    if k == 0 or n == 0 or fits.size == 0:
        return out
    kernel = _kernel(cp)
    fd_dev = _resident_codes(cp, fd)

    def block_work(lo: int, hi: int) -> int:
        """Entropies of the pairs ``fits[lo:hi]``, over the columns they reference as contiguous int32 rows (gathered on the device from the resident matrix, or uploaded when it does not fit int16)."""
        sel = fits[lo:hi]
        used, inverse = np.unique(np.concatenate([a_arr[sel], b_arr[sel]]), return_inverse=True)
        m = int(sel.shape[0])
        if fd_dev is not None:
            cols_d = cp.ascontiguousarray(fd_dev[:, cp.asarray(used)].T).astype(cp.int32)
        else:
            cols_d = cp.asarray(np.ascontiguousarray(fd[:, used].T, dtype=np.int32))
        a_d = cp.asarray(inverse[:m].astype(np.int32))
        b_d = cp.asarray(inverse[m:].astype(np.int32))
        na_d = cp.asarray(nb_a_all[sel].astype(np.int32))
        nb_d = cp.asarray(nb_b_all[sel].astype(np.int32))
        res = cp.empty(m, dtype=cp.float64)  # Intentional float64: entropy output
        shared = int((nb_a_all[sel] * nb_b_all[sel]).max()) * 4
        kernel((m,), (_THREADS,), (cols_d, a_d, b_d, na_d, nb_d, np.int64(n), res), shared_mem=shared)  # launch-ok: one block per pair is the design
        out[sel] = cp.asnumpy(res)
        return m

    block = first_block(cp, n, int(fits.size), 8, 1)
    run_column_blocks(cp, int(fits.size), block, block_work)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DEVICE_FALLBACK_ROWS = 8_000_000  # rows x pairs from which the device takes the joint entropies before a sweep has run (between 160 and 640 pairs of 20000 rows)
_SWEEP_N = 20_000
_SWEEP_ROWS = [_SWEEP_N * k for k in (10, 40, 160, 640)]
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``(fd, a, b, nb)``: a 20-column, 12-level code matrix over 20000 rows and ``dims["n_samples"] / 20000`` pairs of it."""
    rng = np.random.default_rng(0)
    fd = rng.integers(0, 12, (_SWEEP_N, 20)).astype(np.int32)
    k = max(1, int(dims["n_samples"]) // _SWEEP_N)
    a = rng.integers(0, 20, k).astype(np.int64)
    b = ((a + 1 + rng.integers(0, 19, k)) % 20).astype(np.int64)
    return (fd, a, b, np.full(20, 12, dtype=np.int64))


def _sweep_host(fd: np.ndarray, a: np.ndarray, b: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """The numba pair kernel."""
    from ._dcd_pair_su_batch import _joint_entropies_host

    return np.asarray(_joint_entropies_host(fd, a, b, nb))


def _sweep_device(fd: np.ndarray, a: np.ndarray, b: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """The device kernel; every sweep pair fits, so a failed launch raises and is never recorded as a host time."""
    return joint_entropies_gpu(fd, a, b, nb)


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="dcd_pair_entropy_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-9, atol=1e-12),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__),
        gpu_capable=True,
        salt=_SALT,
        cli_label="dcd_pair_entropy_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for dcd_pair_entropy_backend failed: %s", e)
    _SPEC = None


def entropy_device_chosen(n: int, k: int) -> bool:
    """Whether ``k`` pairs over ``n`` rows take their joint entropies on the device: the per-host cache by rows x pairs, else the ``DEVICE_FALLBACK_ROWS`` floor."""
    work = int(n) * int(k)
    if _SPEC is None:
        return work >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=work)) == "device"
    except Exception as exc:
        logger.debug("dcd_pair_entropy_backend: KTC choose() failed, using the floor: %s", exc)
        return work >= DEVICE_FALLBACK_ROWS


def joint_entropies_dispatch(fd: np.ndarray, a_arr: np.ndarray, b_arr: np.ndarray, nb_arr: np.ndarray, host: Any) -> np.ndarray:
    """The joint entropies of the pairs: on the device when the tuned choice says so and a device is usable (pairs over the shared-memory limit, and every pair after a device fault, from ``host``), else ``host``."""
    if a_arr.shape[0] and entropy_device_chosen(fd.shape[0], a_arr.shape[0]):
        try:
            from ._gpu_policy import cuda_available_for_run

            if cuda_available_for_run():
                out = joint_entropies_gpu(fd, a_arr, b_arr, nb_arr)
                todo = np.nonzero(np.isnan(out))[0]
                if todo.size:
                    out[todo] = host(fd, a_arr[todo], b_arr[todo], nb_arr)
                return out
        except Exception as exc:  # a device fault must not lose the stage
            from mlframe.utils.log_throttle import log_throttle

            log_throttle(logger, "dcd_pair_entropy.device_failed", logging.WARNING, "pair joint entropy device path failed (%s: %s); numba kernel", type(exc).__name__, exc)
    return np.asarray(host(fd, a_arr, b_arr, nb_arr))
