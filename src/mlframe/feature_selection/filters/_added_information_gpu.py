"""Device twin of the G statistic of ``_added_information_gate.added_information_p`` for many candidates that share one source and one target.

``G = 2 * sum n_scy * (ln n_scy + ln n_s - ln n_sc - ln n_sy)`` over the occupied cells of the (source, candidate, target) table. One block per candidate: the rows fill a shared integer count table
(integer atomics, so thread order does not matter), the marginals ``n_sc``, ``n_sy``, ``n_s`` are summed from it and the threads add the cell terms; the block reduction is the only floating-point order.
A candidate whose table does not fit the shared memory (``_MAX_CELLS``) returns NaN and the caller computes it on the host. Candidates go to the device in blocks sized from the room under the pool
(``_device_blocks.run_column_blocks``: a column block of candidate codes at a time, halved on OOM); the source and target codes are uploaded once. The p-value (degrees of freedom, sparsity rule,
chi-square tail) stays the host code of ``added_information_p`` so both paths apply the same decisions. Host or device is chosen per call from the kernel-tuning entry ``added_information_backend`` by rows x candidates.
"""

from __future__ import annotations

import threading

import logging
from typing import Any

import numpy as np
from mlframe.utils.array_shape import length_of

from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback

logger = logging.getLogger(__name__)

__all__ = ["g_statistics_gpu", "added_information_p_many", "added_device_chosen"]

_MAX_CELLS = 4096  # (source, candidate, target) cells that fit the shared count table together with the marginals
_THREADS = 256

_SRC = render(
    r"""
extern "C" __global__ __launch_bounds__(256)
void added_information_g(const {CT}* __restrict__ cand, const int* __restrict__ bcs, const int* __restrict__ s, const int* __restrict__ t,
                         const long long n, const int bs, const int by, {ACC}* __restrict__ g_out) {
    extern __shared__ int tab[];                          // joint[bs * bc * by], n_sc[bs * bc], n_sy[bs * by], n_s[bs]
    __shared__ {ACC} red[32];
    const int k = blockIdx.x;
    const int bc = bcs[k];
    const int cells = bs * bc * by;
    int* n_sc = tab + cells;
    int* n_sy = n_sc + bs * bc;
    int* n_s = n_sy + bs * by;
    const int total = cells + bs * bc + bs * by + bs;
    for (int q = threadIdx.x; q < total; q += blockDim.x) tab[q] = 0;
    __syncthreads();
    const {CT}* c = cand + (long long)k * n;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        const int sc = s[i] * bc + c[i];
        atomicAdd(&tab[sc * by + t[i]], 1);
        atomicAdd(&n_sc[sc], 1);
        atomicAdd(&n_sy[s[i] * by + t[i]], 1);
        atomicAdd(&n_s[s[i]], 1);
    }
    __syncthreads();
    {ACC} acc = 0.0;
    for (int q = threadIdx.x; q < cells; q += blockDim.x) {
        const int v = tab[q];
        if (v <= 0) continue;
        const int sc = q / by, y = q % by, si = sc / bc;
        acc += ({ACC})v * (log(({ACC})v) + log(({ACC})n_s[si]) - log(({ACC})n_sc[sc]) - log(({ACC})n_sy[si * by + y]));
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        {ACC} total_g = 0.0;
        for (int w = 0; w < (blockDim.x >> 5); ++w) total_g += red[w];
        g_out[k] = 2.0 * total_g;
    }
}
""",
)

_KERNELS: dict = {}
_KERNELS_LOCK = threading.RLock()  # serialises the get-or-build of the cache above (one build per key even with concurrent fits)


def _kernel(cp: Any, narrow: bool) -> Any:
    """The compiled kernel; candidate codes are read as bytes when every candidate has at most 127 levels, else as 32-bit integers (one build per code type)."""
    with _KERNELS_LOCK:
        if narrow not in _KERNELS:
            _KERNELS[narrow] = cp.RawKernel(_SRC.replace("{CT}", "signed char" if narrow else "int"), "added_information_g")
        return _KERNELS[narrow]


def g_statistics_gpu(cands: np.ndarray, bcs: np.ndarray, s: np.ndarray, bs: int, t: np.ndarray, by: int) -> np.ndarray:
    """``G`` per row of ``cands`` (dense candidate codes, ``(k, n)``, ``bcs[j]`` levels in row ``j``) against the dense source ``s`` (``bs`` levels) and target ``t`` (``by`` levels); NaN for a row over the table limit. Raises on a device fault."""
    import cupy as cp

    from ._device_blocks import first_block, run_column_blocks

    k, n = cands.shape
    out = np.full(k, np.nan)  # Intentional float64: the G statistic compared to the host at full precision
    fits = np.nonzero(bs * bcs * by <= _MAX_CELLS)[0]
    all_fit = bool(fits.size == k)
    if k == 0 or n == 0 or fits.size == 0:
        return out
    s_d = cp.asarray(np.ascontiguousarray(s, dtype=np.int32))
    t_d = cp.asarray(np.ascontiguousarray(t, dtype=np.int32))
    narrow = bool(bcs.max() <= 127)  # dense codes are below the level count: a byte holds them and the upload is a quarter of the 32-bit one
    kernel = _kernel(cp, narrow)

    def block_work(lo: int, hi: int) -> int:
        """G of the candidates ``fits[lo:hi]`` (a column block of the candidate matrix)."""
        sel = fits[lo:hi]
        rows = cands[lo:hi] if all_fit else cands[sel]  # a contiguous slice (no copy) when every candidate fits the table
        c_d = cp.asarray(np.ascontiguousarray(rows, dtype=np.int8 if narrow else np.int32))
        b_d = cp.asarray(np.ascontiguousarray(bcs[sel], dtype=np.int32))
        res = cp.empty(sel.shape[0], dtype=cp.float64)  # Intentional float64: G output
        cells = bs * int(bcs[sel].max()) * by
        shared = (cells + bs * int(bcs[sel].max()) + bs * by + bs) * 4
        kernel((int(sel.shape[0]),), (_THREADS,), (c_d, b_d, s_d, t_d, np.int64(n), np.int32(bs), np.int32(by), res), shared_mem=shared)  # launch-ok: one block per candidate is the design
        out[sel] = cp.asnumpy(res)
        return int(sel.shape[0])

    block = first_block(cp, n, int(fits.size), 1 if narrow else 4, 2)
    run_column_blocks(cp, int(fits.size), block, block_work)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DEVICE_FALLBACK_ROWS = 500_000  # rows x candidates from which the device takes the G statistics before a sweep has run (16 candidates of 30000 rows)
_SWEEP_N = 30_000
_SWEEP_ROWS = [_SWEEP_N * k for k in (4, 16, 64, 256)]
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``(cands, bcs, s, bs, t, by)``: ``dims["n_samples"] / 30000`` candidates of 10 levels over 30000 rows, a 4-level source, a binary target tied to the first candidate."""
    rng = np.random.default_rng(0)
    k = max(1, int(dims["n_samples"]) // _SWEEP_N)
    cands = rng.integers(0, 10, (k, _SWEEP_N)).astype(np.int64)
    s = rng.integers(0, 4, _SWEEP_N).astype(np.int64)
    t = ((cands[0] + s) % 2).astype(np.int64)
    return (cands, np.full(k, 10, dtype=np.int64), s, 4, t, 2)


def _g_host(cands: np.ndarray, bcs: np.ndarray, s: np.ndarray, bs: int, t: np.ndarray, by: int) -> np.ndarray:
    """``G`` per candidate with the numpy expression of ``added_information_p`` (the host reference)."""
    out = np.empty(cands.shape[0])
    for j in range(cands.shape[0]):
        bc = int(bcs[j])
        joint = np.bincount((s * bc + cands[j]) * by + t, minlength=bs * bc * by).reshape(bs, bc, by).astype(np.float64)  # Intentional float64: counts and their logarithms in the G statistic
        n_sc, n_sy, n_s = joint.sum(axis=2), joint.sum(axis=1), joint.sum(axis=(1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            term = joint * (np.log(joint) + np.log(n_s)[:, None, None] - np.log(n_sc)[:, :, None] - np.log(n_sy)[:, None, :])
        out[j] = 2.0 * float(np.nansum(term[joint > 0]))
    return out


def _sweep_host(*args: Any) -> np.ndarray:
    """The host reference."""
    return _g_host(*args)


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="added_information_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, g_statistics_gpu, rtol=1e-9, atol=1e-9),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__),
        gpu_capable=True,
        salt=_SALT,
        cli_label="added_information_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for added_information_backend failed: %s", e)
    _SPEC = None


def added_device_chosen(n: int, k: int) -> bool:
    """Whether ``k`` candidates over ``n`` rows take their G statistics on the device: the per-host cache by rows x candidates, else the ``DEVICE_FALLBACK_ROWS`` floor."""
    work = int(n) * int(k)
    if _SPEC is None:
        return work >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=work)) == "device"
    except Exception as exc:
        logger.debug("added_information_backend: KTC choose() failed, using the floor: %s", exc)
        return work >= DEVICE_FALLBACK_ROWS


def added_information_p_many(candidates: "list[np.ndarray]", source: "np.ndarray | None", y: np.ndarray, *, y_dense: Any = None, source_dense: Any = None) -> "list[float]":
    """``added_information_p`` of each candidate against the same source and target: the G statistics on the device when the tuned choice says so and a device is usable (candidates over the table limit, and every
    candidate after a device fault, on the host), else the per-candidate host path."""
    from ._added_information_gate import _MIN_EXPECTED_CELL, _dense_codes, added_information_p
    from ._analytic_mi_null import _chi2_sf

    k = len(candidates)
    if k == 0:
        return []
    n = int(length_of(y))
    dense = [_dense_codes(c) for c in candidates]
    t, by = y_dense if y_dense is not None else _dense_codes(y)
    if source is None:
        s, bs = np.zeros(n, dtype=np.int64), 1
    else:
        s, bs = source_dense if source_dense is not None else _dense_codes(source)
    p: "list[float]" = [float("nan")] * k
    todo = list(range(k))
    if k > 1 and n > 0 and added_device_chosen(n, k):
        try:
            from ._gpu_policy import cuda_available_for_run

            if cuda_available_for_run():
                bcs = np.array([d[1] for d in dense], dtype=np.int64)
                code_type = np.int8 if int(bcs.max()) <= 127 else np.int64  # the matrix is built at the width the kernel reads: no 64-bit copy of it
                cand_mat = np.empty((k, n), dtype=code_type)
                for j, d in enumerate(dense):
                    cand_mat[j] = d[0]
                g = g_statistics_gpu(cand_mat, bcs, s, bs, t, by)
                todo = []
                for j in range(k):
                    bc = int(bcs[j])
                    df = bs * (bc - 1) * (by - 1)
                    if np.isnan(g[j]):
                        if n > 0 and df > 0 and n / float(bs * bc * by) < _MIN_EXPECTED_CELL:
                            p[j] = float("nan")  # too sparse to test, as on the host
                        else:
                            todo.append(j)
                    elif n <= 0 or df <= 0:
                        p[j] = 1.0
                    elif n / float(bs * bc * by) < _MIN_EXPECTED_CELL:
                        p[j] = float("nan")
                    elif g[j] <= 0.0:
                        p[j] = 1.0
                    else:
                        p[j] = float(min(1.0, max(0.0, _chi2_sf(float(g[j]), df))))
        except Exception as exc:  # a device fault must not lose the gate
            from mlframe.utils.log_throttle import log_throttle

            log_throttle(logger, "added_information.device_failed", logging.WARNING, "added-information device path failed (%s: %s); host path", type(exc).__name__, exc)
            todo = list(range(k))
    for j in todo:
        p[j] = added_information_p(candidates[j], source, y, y_dense=(t, by), source_dense=(s, bs))
    return p
