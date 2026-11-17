"""Device twin of ``_pair_residual_screen._score_pairs``: the held-out t statistic of the residual reduction by the pair-cell means, for every column pair.

For a pair ``(a, b)`` the host shrinks the mean residual of each ``N_BINS x N_BINS`` cell estimated on the even rows toward 0 and applies it to the odd rows; the evidence is the paired reduction
``d = e^2 - (e - m_cell)^2`` and its t statistic. One block per pair does the same: the even rows add their residual (as a 2^-k fixed-point integer, so the cell sums do not depend on the order of the threads) and
their count to a shared table, the odd rows give the sum of ``d`` and then, with the mean in hand, the sum of ``(d - mean)^2`` (the host uses a Welford running mean: the same quantity to rounding). The bin codes go up once as
bytes (``N_BINS`` is small), the residual and the mask once; pairs go to the device in blocks of pairs (a column block of the pair list, halved on an out-of-memory error by ``run_column_blocks``). Host or device is chosen per call
from the kernel-tuning entry ``pair_residual_backend`` by rows x pairs.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback

logger = logging.getLogger(__name__)

__all__ = ["score_pairs_gpu", "score_pairs_dispatch", "pair_device_chosen"]

_THREADS = 256
_MAX_CELLS = 2048  # n_bins * n_bins cells held in the shared table (two 8-byte arrays)

_SRC = render(
    r"""
extern "C" __global__ __launch_bounds__(256)
void pair_residual_t(const signed char* __restrict__ Q, const {ACC}* __restrict__ resid, const int* __restrict__ pair_a, const int* __restrict__ pair_b,
                     const long long n, const long long n_even, const int n_bins, const {ACC} shrink, const {ACC} fix, {ACC}* __restrict__ t_out) {
    extern __shared__ unsigned long long sh[];            // fixed-point sum[cells], count[cells], then cell mean[cells] and 32 reduction slots ({ACC})
    const int cells = n_bins * n_bins;
    unsigned long long* s = sh;
    unsigned long long* c = sh + cells;
    {ACC}* cell_mean = ({ACC}*)(sh + 2 * cells);
    {ACC}* red = cell_mean + cells;
    __shared__ long long cnt_sh[32];
    __shared__ {ACC} mean_sh;
    __shared__ long long n_odd;
    const int p = blockIdx.x;
    const signed char* qa = Q + (long long)pair_a[p] * n;
    const signed char* qb = Q + (long long)pair_b[p] * n;
    for (int t = threadIdx.x; t < 2 * cells; t += blockDim.x) sh[t] = 0ULL;
    __syncthreads();
    // rows arrive with the even rows first (the host reorders them), so each pass is a plain loop without a mask load or a divergent branch
    for (long long i = threadIdx.x; i < n_even; i += blockDim.x) {
        const int cell = (int)qa[i] * n_bins + (int)qb[i];
        atomicAdd(&s[cell], (unsigned long long)llrint(resid[i] * fix));
        atomicAdd(&c[cell], 1ULL);
    }
    __syncthreads();
    for (int cell = threadIdx.x; cell < cells; cell += blockDim.x) cell_mean[cell] = ({ACC})(long long)s[cell] / fix / (({ACC})c[cell] + shrink);  // once per cell, not once per row
    __syncthreads();
    {ACC} sum = 0.0;
    long long cnt = 0;
    for (long long i = n_even + threadIdx.x; i < n; i += blockDim.x) {
        const int cell = (int)qa[i] * n_bins + (int)qb[i];
        const {ACC} mc = cell_mean[cell];
        const {ACC} e = resid[i];
        sum += e * e - (e - mc) * (e - mc);
        cnt += 1;
    }
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, o);
        cnt += __shfl_down_sync(0xffffffffu, cnt, o);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = (blockDim.x + 31) >> 5;
    if (lane == 0) { red[warp] = sum; cnt_sh[warp] = cnt; }
    __syncthreads();
    if (threadIdx.x == 0) {
        {ACC} total = 0.0; long long k = 0;
        for (int w = 0; w < nwarps; ++w) { total += red[w]; k += cnt_sh[w]; }
        n_odd = k;
        mean_sh = k > 0 ? total / ({ACC})k : 0.0;
    }
    __syncthreads();
    const {ACC} mean = mean_sh;
    {ACC} m2 = 0.0;
    for (long long i = n_even + threadIdx.x; i < n; i += blockDim.x) {
        const int cell = (int)qa[i] * n_bins + (int)qb[i];
        const {ACC} mc = cell_mean[cell];
        const {ACC} e = resid[i];
        const {ACC} dd = (e * e - (e - mc) * (e - mc)) - mean;
        m2 += dd * dd;
    }
    for (int o = 16; o > 0; o >>= 1) m2 += __shfl_down_sync(0xffffffffu, m2, o);
    __syncthreads();
    if (lane == 0) red[warp] = m2;
    __syncthreads();
    if (threadIdx.x == 0) {
        if (n_odd > 1) {
            {ACC} tot2 = 0.0;
            for (int w = 0; w < nwarps; ++w) tot2 += red[w];
            const {ACC} var = fmax(tot2 / ({ACC})n_odd, 0.0);
            const {ACC} se = sqrt(var / ({ACC})n_odd);
            t_out[p] = se > 0.0 ? mean / se : mean * 1e300;
        } else {
            t_out[p] = 0.0;
        }
    }
}
""",
)

_KERNEL: Any = None


def _kernel(cp: Any) -> Any:
    """The compiled kernel (one per process)."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(_SRC, "pair_residual_t")
    return _KERNEL


def score_pairs_gpu(Q: np.ndarray, resid: np.ndarray, even: np.ndarray, pairs: np.ndarray, n_bins: int, shrink: float) -> np.ndarray:
    """The ``t`` statistic of every row of ``pairs`` (indices into the rows of the bin-code matrix ``Q``); the device twin of ``_score_pairs``. Raises when the table does not fit or on a device fault."""
    import cupy as cp

    from ._device_blocks import first_block, run_column_blocks

    n_pairs = int(pairs.shape[0])
    n = int(Q.shape[1])
    out = np.zeros(n_pairs)  # Intentional float64: the t statistic compared to the host's at full precision
    if n_pairs == 0 or n == 0:
        return out
    if int(n_bins) * int(n_bins) > _MAX_CELLS or int(n_bins) > 127:
        raise ValueError("pair table does not fit the shared memory")
    ev = np.asarray(even, dtype=bool)
    perm = np.concatenate([np.nonzero(ev)[0], np.nonzero(~ev)[0]])  # even rows first: no mask in the kernel
    n_even = int(ev.sum())
    perm_d = cp.asarray(perm)
    r = np.ascontiguousarray(np.asarray(resid)[perm], dtype=np.float64)  # Intentional float64: the residual statistic is formed and summed in double, as the host does
    m = float(np.max(np.abs(r))) * n if r.size else 1.0
    fix = 2.0 ** math.floor(61.0 - math.log2(max(m, 1e-300)))
    r_d = cp.asarray(r)
    kernel = _kernel(cp)
    shared = (3 * int(n_bins) * int(n_bins) + 32) * 8

    def block_work(lo: int, hi: int) -> int:
        """Score the pairs ``lo:hi``, uploading only the code rows they reference as bytes."""
        sel = pairs[lo:hi]
        used = np.unique(sel)
        remap = np.zeros(Q.shape[0], dtype=np.int32)
        remap[used] = np.arange(used.shape[0], dtype=np.int32)
        q_d = cp.take(cp.asarray(np.ascontiguousarray(Q[used], dtype=np.int8)), perm_d, axis=1)  # bytes up, then the even-first order is applied on the device
        a_d = cp.asarray(remap[sel[:, 0]])
        b_d = cp.asarray(remap[sel[:, 1]])
        t_d = cp.empty(hi - lo, dtype=cp.float64)  # Intentional float64: statistic output
        kernel((hi - lo,), (_THREADS,), (q_d, r_d, a_d, b_d, np.int64(n), np.int64(n_even), np.int32(n_bins), np.float64(shrink), np.float64(fix), t_d), shared_mem=shared)  # launch-ok: one block per pair is the design
        out[lo:hi] = cp.asnumpy(t_d)
        return hi - lo

    block = first_block(cp, n, n_pairs, 1, 1)
    run_column_blocks(cp, n_pairs, max(block, 1), block_work)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DEVICE_FALLBACK_ROWS = 4_000_000  # rows x pairs from which the device scores the pairs before a sweep has run (crossover between 160 and 780 pairs of 20000 rows)
_SWEEP_N = 20_000
_SWEEP_ROWS = [_SWEEP_N * k for k in (10, 40, 160, 780)]
_SWEEP_BINS = 8
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """40 columns of 8 bins over 20000 rows, ``dims["n_samples"] / 20000`` pairs over them, a residual carrying an interaction of the first two columns."""
    rng = np.random.default_rng(0)
    Q = rng.integers(0, _SWEEP_BINS, (40, _SWEEP_N)).astype(np.int64)
    resid = rng.standard_normal(_SWEEP_N) + 0.5 * (((Q[0] + Q[1]) % 2) - 0.5)
    even = np.arange(_SWEEP_N) % 2 == 0
    all_pairs = np.array([(a, b) for a in range(40) for b in range(a + 1, 40)], dtype=np.int64)
    k = max(1, min(int(dims["n_samples"]) // _SWEEP_N, all_pairs.shape[0]))
    return (Q, resid, even, all_pairs[:k], _SWEEP_BINS, 3.0)


def _sweep_host(*args: Any) -> np.ndarray:
    """The numba pass."""
    from ._pair_residual_screen import _score_pairs

    return np.asarray(_score_pairs(*args))


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="pair_residual_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, score_pairs_gpu, rtol=1e-6, atol=1e-6),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__),
        gpu_capable=True,
        salt=_SALT,
        cli_label="pair_residual_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for pair_residual_backend failed: %s", e)
    _SPEC = None


from ._ktc_fallback import ktc_device_chooser as _ktc_device_chooser

pair_device_chosen = _ktc_device_chooser(__name__, "pair_residual_backend", pairs=True)


def score_pairs_dispatch(host: Any, Q: np.ndarray, resid: np.ndarray, even: np.ndarray, pairs: np.ndarray, n_bins: int, shrink: float) -> np.ndarray:
    """The pair ``t`` statistics: on the device when the tuned choice says so and a device is usable, else ``host`` (the numba pass); a device fault falls back to ``host`` with a throttled warning."""
    if pairs.shape[0] and pair_device_chosen(Q.shape[1], pairs.shape[0]):
        try:
            from ._gpu_policy import cuda_available_for_run

            if cuda_available_for_run():
                return score_pairs_gpu(Q, resid, even, pairs, n_bins, shrink)
        except Exception as exc:  # a device fault must not lose the screen
            from mlframe.utils.log_throttle import log_throttle

            log_throttle(logger, "pair_residual.device_failed", logging.WARNING, "pair residual screen device path failed (%s: %s); numba pass", type(exc).__name__, exc)
    result: np.ndarray = np.asarray(host(Q, resid, even, pairs, n_bins, shrink))
    return result
