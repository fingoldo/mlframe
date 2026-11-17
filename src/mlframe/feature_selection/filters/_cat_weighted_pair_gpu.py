"""Device twin of ``_cat_target_encoding_and_weighted._pair_search_kernel_weighted_njit``: the weighted joint MI of categorical column pairs with the class codes.

One block per pair: the rows add their sample weight to a shared ``(merged cell, class)`` table, then the threads sum ``jw / W * log(jw * W / (marg_m * marg_y))`` over the cells exactly as the host kernel does.
The weights are added as 64-bit integers at a power-of-two scale chosen so that the total stays below 2^62 (``scale = 2^floor(61 - log2(W))``), so the tables do not depend on the order in which threads add and
a run gives the same bits as the next; the rounding of each weight to the scale is below 2^-62 of the total weight, far under the double rounding of the host sums. A pair whose table does not fit the shared
memory (``_MAX_CELLS``) returns NaN and the caller computes it with the host kernel. Pairs go to the device in blocks, each block uploading only the code columns it references (column blocks of the code matrix,
halved on an out-of-memory error). Host or device is chosen per call from the kernel-tuning entry ``cat_weighted_pair_backend`` by rows x pairs.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback

logger = logging.getLogger(__name__)

__all__ = ["weighted_pair_search_gpu", "weighted_pair_search_dispatch", "weighted_device_chosen"]

_MAX_CELLS = 4096  # (merged cell, class) entries of 8 bytes that fit the shared table
_THREADS = 256

_SRC = render(
    r"""
extern "C" __global__ __launch_bounds__(256)
void cat_weighted_pair(const int* __restrict__ codes, const int* __restrict__ nbins, const int* __restrict__ first, const int* __restrict__ second, const int* __restrict__ y,
                       const long long* __restrict__ wfix, const {ACC}* __restrict__ marg_y_w, const long long n, const int ky, const {ACC} inv_scale, const {ACC} total_w,
                       {ACC}* __restrict__ mi_out, int* __restrict__ uniq_out) {
    extern __shared__ unsigned long long jt[];            // joint[merged * ky + class] as fixed-point weights
    __shared__ {ACC} red[32];
    __shared__ int red_u[32];
    const int p = blockIdx.x;
    const int a = first[p], b = second[p];
    const int nba = nbins[a];
    const int merged = nba * nbins[b];
    const int size = merged * ky;
    for (int t = threadIdx.x; t < size; t += blockDim.x) jt[t] = 0ULL;
    __syncthreads();
    const int* ca = codes + (long long)a * n;
    const int* cb = codes + (long long)b * n;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) atomicAdd(&jt[(ca[i] + cb[i] * nba) * ky + y[i]], (unsigned long long)wfix[i]);
    __syncthreads();
    {ACC} mi = 0.0;
    int uniq = 0;
    for (int m = threadIdx.x; m < merged; m += blockDim.x) {
        unsigned long long mm_fix = 0ULL;
        for (int c = 0; c < ky; ++c) mm_fix += jt[m * ky + c];
        const {ACC} mm = ({ACC})mm_fix * inv_scale;
        if (mm <= 0.0) continue;
        uniq += 1;
        for (int c = 0; c < ky; ++c) {
            const {ACC} jw = ({ACC})jt[m * ky + c] * inv_scale;
            if (jw <= 0.0) continue;
            const {ACC} my = marg_y_w[c];
            if (my <= 0.0) continue;
            mi += (jw / total_w) * log(jw * total_w / (mm * my));
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        mi += __shfl_down_sync(0xffffffffu, mi, o);
        uniq += __shfl_down_sync(0xffffffffu, uniq, o);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { red[warp] = mi; red_u[warp] = uniq; }
    __syncthreads();
    if (threadIdx.x == 0) {
        {ACC} tot = 0.0;
        int tu = 0;
        for (int w = 0; w < (blockDim.x >> 5); ++w) { tot += red[w]; tu += red_u[w]; }
        mi_out[p] = tot;
        uniq_out[p] = tu;
    }
}
""",
)

_KERNEL: Any = None


def _kernel(cp: Any) -> Any:
    """The compiled kernel (one per process)."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(_SRC, "cat_weighted_pair")
    return _KERNEL


def weighted_pair_search_gpu(
    factors_data: np.ndarray, pairs_a: np.ndarray, pairs_b: np.ndarray, marginal_mi: np.ndarray, nbins: np.ndarray, classes_y: np.ndarray, weights: np.ndarray
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """``(joint_mi, ii, n_uniq, fits)`` of the pairs, as the host weighted kernel returns them; ``fits`` marks the pairs the device scored (the others are zero and left to the host). Raises on a device fault."""
    import cupy as cp

    from ._device_blocks import first_block, run_column_blocks

    n_pairs = int(pairs_a.shape[0])
    n = int(factors_data.shape[0])
    joint = np.zeros(n_pairs)  # Intentional float64: pairwise MI accumulator output compared to the host's
    uniq = np.zeros(n_pairs, dtype=np.int64)
    ky = int(classes_y.max()) + 1 if classes_y.size else 1
    w = np.asarray(weights, dtype=np.float64)  # Intentional float64: sample weights are summed and scaled in double
    total_w = float(w.sum())
    fits_all = (nbins[pairs_a].astype(np.int64) * nbins[pairs_b].astype(np.int64) * ky <= _MAX_CELLS) & (total_w > 0.0)
    fits_idx = np.nonzero(fits_all)[0]
    if n_pairs == 0 or n == 0 or fits_idx.size == 0:
        return joint, joint.copy(), uniq, fits_all
    scale = 2.0 ** math.floor(61.0 - math.log2(total_w))
    wfix = np.rint(w * scale).astype(np.int64)
    wfix_d = cp.asarray(wfix)
    y_d = cp.asarray(np.ascontiguousarray(classes_y, dtype=np.int32))
    marg_y = np.zeros(ky)  # Intentional float64: class weights of the MI statistic
    np.add.at(marg_y, classes_y.astype(np.int64), w)
    marg_y_d = cp.asarray(marg_y)
    kernel = _kernel(cp)

    def block_work(lo: int, hi: int) -> int:
        """Score the pairs ``fits_idx[lo:hi]``, uploading only the code columns they reference."""
        sel = fits_idx[lo:hi]
        used = np.unique(np.concatenate([pairs_a[sel], pairs_b[sel]]))
        remap = np.zeros(factors_data.shape[1], dtype=np.int32)
        remap[used] = np.arange(used.shape[0], dtype=np.int32)
        codes_d = cp.asarray(np.ascontiguousarray(factors_data[:, used].T, dtype=np.int32))
        nb_d = cp.asarray(nbins[used].astype(np.int32))
        a_d = cp.asarray(remap[pairs_a[sel]])
        b_d = cp.asarray(remap[pairs_b[sel]])
        mi_d = cp.empty(sel.shape[0], dtype=cp.float64)  # Intentional float64: MI statistic output
        uq_d = cp.empty(sel.shape[0], dtype=cp.int32)
        shared = int((nbins[pairs_a[sel]].astype(np.int64) * nbins[pairs_b[sel]].astype(np.int64) * ky).max()) * 8
        kernel((int(sel.shape[0]),), (_THREADS,), (codes_d, nb_d, a_d, b_d, y_d, wfix_d, marg_y_d, np.int64(n), np.int32(ky), np.float64(1.0 / scale), np.float64(total_w), mi_d, uq_d), shared_mem=shared)  # launch-ok: one block per pair is the design
        joint[sel] = cp.asnumpy(mi_d)
        uniq[sel] = cp.asnumpy(uq_d)
        return int(sel.shape[0])

    block = first_block(cp, n, int(fits_idx.size), 4, 2)
    run_column_blocks(cp, int(fits_idx.size), block, block_work)
    ii = joint - marginal_mi[pairs_a] - marginal_mi[pairs_b]
    return joint, ii, uniq, fits_all


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DEVICE_FALLBACK_ROWS = 2_000_000  # rows x pairs from which the device scores the pairs before a sweep has run (crossover between 40 and 160 pairs of 20000 rows)
_SWEEP_N = 20_000
_SWEEP_ROWS = [_SWEEP_N * k for k in (10, 40, 160, 640)]
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """Six categorical columns of 12 levels over 20000 rows, ``dims["n_samples"] / 20000`` pairs over them, a 3-class target tied to two columns and positive sample weights."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 12, (_SWEEP_N, 6)).astype(np.int32)
    k = max(1, int(dims["n_samples"]) // _SWEEP_N)
    a = rng.integers(0, 6, k).astype(np.int64)
    b = ((a + 1 + rng.integers(0, 5, k)) % 6).astype(np.int64)
    y = ((data[:, 0] + data[:, 1]) % 3).astype(np.int64)
    w = rng.random(_SWEEP_N) + 0.5
    return (data, a, b, np.zeros(6), np.full(6, 12, dtype=np.int64), y, w)


def _sweep_host(*args: Any) -> np.ndarray:
    """The numba weighted kernel."""
    from ._cat_target_encoding_and_weighted import _pair_search_kernel_weighted_njit

    data, a, b, marg, nb, y, w = args
    return np.asarray(_pair_search_kernel_weighted_njit(data, a, b, marg, nb, y, w, np.int32)[0])


def _sweep_device(*args: Any) -> np.ndarray:
    """The device kernel; every sweep pair fits, so a failed launch raises and is never recorded as a host time."""
    data, a, b, marg, nb, y, w = args
    return weighted_pair_search_gpu(data, a, b, marg, nb, y, w)[0]


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="cat_weighted_pair_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-9, atol=1e-12),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__),
        gpu_capable=True,
        salt=_SALT,
        cli_label="cat_weighted_pair_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for cat_weighted_pair_backend failed: %s", e)
    _SPEC = None


def weighted_device_chosen(n: int, n_pairs: int) -> bool:
    """Whether ``n_pairs`` pairs over ``n`` rows are scored on the device: the per-host cache by rows x pairs, else the ``DEVICE_FALLBACK_ROWS`` floor."""
    work = int(n) * int(n_pairs)
    if _SPEC is None:
        return work >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=work)) == "device"
    except Exception as exc:
        logger.debug("cat_weighted_pair_backend: KTC choose() failed, using the floor: %s", exc)
        return work >= DEVICE_FALLBACK_ROWS


def weighted_pair_search_dispatch(
    host: Any, factors_data: np.ndarray, pairs_a: np.ndarray, pairs_b: np.ndarray, marginal_mi: np.ndarray, nbins: np.ndarray, classes_y: np.ndarray, weights: np.ndarray, dtype: Any
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """``(joint_mi, ii, n_uniq)`` of the weighted pair search: on the device when the tuned choice says so and a device is usable (pairs over the table limit, and everything after a device fault, from ``host``), else ``host``."""
    if pairs_a.shape[0] and weighted_device_chosen(factors_data.shape[0], pairs_a.shape[0]):
        try:
            from ._gpu_policy import cuda_available_for_run

            if cuda_available_for_run():
                joint, ii, uniq, fits = weighted_pair_search_gpu(factors_data, pairs_a, pairs_b, marginal_mi, nbins, classes_y, weights)
                rest = np.nonzero(~fits)[0]
                if rest.size:
                    from mlframe.utils.log_throttle import log_throttle

                    log_throttle(
                        logger, "cat_weighted_pair.table_too_big", logging.INFO,
                        "weighted categorical pair search: %d of %d pairs exceed the %d-entry shared table (or have no positive weight) and run on the numba kernel",
                        int(rest.size), int(pairs_a.shape[0]), _MAX_CELLS,
                    )
                    j2, i2, u2 = host(factors_data, pairs_a[rest], pairs_b[rest], marginal_mi, nbins, classes_y, weights, dtype)
                    joint[rest], ii[rest], uniq[rest] = j2, i2, u2
                return joint, ii, uniq
        except Exception as exc:  # a device fault must not lose the stage
            from mlframe.utils.log_throttle import log_throttle

            log_throttle(logger, "cat_weighted_pair.device_failed", logging.WARNING, "weighted categorical pair search device path failed (%s: %s); numba kernel", type(exc).__name__, exc)
    result: "tuple[np.ndarray, np.ndarray, np.ndarray]" = host(factors_data, pairs_a, pairs_b, marginal_mi, nbins, classes_y, weights, dtype)
    return result
