"""Device twin of the pair loop of the RelaxMRMR interaction term (``_relaxmrmr_pair_loop._pair_interaction_terms_njit``).

For every pair ``(i, j)`` of already-selected features the host forms the composite code ``z = c_i * K_j + c_j`` and needs two Miller-Madow quantities: ``I(x; z | y)`` and ``I(x; z)``. One block per pair
fills one shared ``(x, z, y)`` integer count table (integer atomics, so the counts do not depend on the order of the threads), derives every marginal table from it in shared memory, and the threads sum the same terms
as ``_cmi_mm_njit`` and ``_mi_mm_njit`` do (``(v / n) * log(...)`` over the occupied cells, then the Miller-Madow debit from the occupied counts); the block reduces the two sums and the five counts. The host
combines them with the per-feature terms exactly as the numba loop does. A pair whose table does not fit the shared memory (``_MAX_CELLS`` entries in all) is left to the host kernel. The selected-feature code
rows go up once as bytes (a column block of the selection), pairs in one launch; the host or device choice is the kernel-tuning entry ``relaxmrmr_pair_backend`` by rows x pairs.
"""

from __future__ import annotations

import threading

import logging
from typing import Any, cast

import numpy as np

from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback
from ._raw_module_cache import code_width_kernel

logger = logging.getLogger(__name__)

__all__ = ["pair_terms_gpu", "pair_terms_dispatch", "pair_device_chosen"]

_MAX_CELLS = 9000  # shared integer entries per block: the (x, z, y) table plus its six marginal tables
_THREADS = 256

_SRC = render(
    r"""
extern "C" __global__ __launch_bounds__(256)
void relaxmrmr_pair(const {CT}* __restrict__ sel, const {CT}* __restrict__ x, const {CT}* __restrict__ y, const int* __restrict__ pair_i, const int* __restrict__ pair_j,
                    const int* __restrict__ k_sel, const long long n, const int kx, const int ky, {ACC}* __restrict__ cmi_out, {ACC}* __restrict__ mi_out) {
    extern __shared__ int sh[];
    __shared__ {ACC} red[64];
    __shared__ int cnt_red[5][32];
    const int p = blockIdx.x;
    const int i = pair_i[p], j = pair_j[p];
    const int kj = k_sel[j];
    const int kz = k_sel[i] * kj;
    int* tab = sh;                                        // [x][z][y]
    int* pxy = tab + kx * kz * ky;                        // n(x, y)
    int* pzy = pxy + kx * ky;                             // n(z, y)
    int* pxz = pzy + kz * ky;                             // n(x, z)
    int* px = pxz + kx * kz;                              // n(x)
    int* pz = px + kx;                                    // n(z)
    int* py = pz + kz;                                    // n(y)
    const int total = kx * kz * ky + kx * ky + kz * ky + kx * kz + kx + kz + ky;
    for (int t = threadIdx.x; t < total; t += blockDim.x) sh[t] = 0;
    __syncthreads();
    const {CT}* ci = sel + (long long)i * n;
    const {CT}* cj = sel + (long long)j * n;
    for (long long r = threadIdx.x; r < n; r += blockDim.x) {
        const int z = (int)ci[r] * kj + (int)cj[r];
        atomicAdd(&tab[((int)x[r] * kz + z) * ky + (int)y[r]], 1);
    }
    __syncthreads();
    for (int c = threadIdx.x; c < kx * kz * ky; c += blockDim.x) {
        const int v = tab[c];
        if (v > 0) {
            const int yy = c % ky, z = (c / ky) % kz, xx = c / (ky * kz);
            atomicAdd(&pxy[xx * ky + yy], v);
            atomicAdd(&pzy[z * ky + yy], v);
            atomicAdd(&pxz[xx * kz + z], v);
            atomicAdd(&px[xx], v);
            atomicAdd(&pz[z], v);
            atomicAdd(&py[yy], v);
        }
    }
    __syncthreads();
    const {ACC} nf = ({ACC})n;
    {ACC} cmi = 0.0, mi = 0.0;
    int m_xyz = 0, m_xz = 0, m_yz = 0, m_ab = 0;
    for (int c = threadIdx.x; c < kx * kz * ky; c += blockDim.x) {
        const int v = tab[c];
        if (v > 0) {
            const int yy = c % ky, z = (c / ky) % kz, xx = c / (ky * kz);
            m_xyz += 1;
            cmi += (({ACC})v / nf) * log((({ACC})v * ({ACC})py[yy]) / (({ACC})pxy[xx * ky + yy] * ({ACC})pzy[z * ky + yy]));
        }
    }
    for (int c = threadIdx.x; c < kx * ky; c += blockDim.x) if (pxy[c] > 0) m_xz += 1;      // occupied (x, y) cells: the "xz" table of the host call
    for (int c = threadIdx.x; c < kz * ky; c += blockDim.x) if (pzy[c] > 0) m_yz += 1;      // occupied (z, y) cells: the "yz" table
    for (int c = threadIdx.x; c < kx * kz; c += blockDim.x) {
        const int v = pxz[c];
        if (v > 0) {
            m_ab += 1;
            mi += (({ACC})v / nf) * log(({ACC})v * nf / (({ACC})px[c / kz] * ({ACC})pz[c % kz]));
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        cmi += __shfl_down_sync(0xffffffffu, cmi, o);
        mi += __shfl_down_sync(0xffffffffu, mi, o);
        m_xyz += __shfl_down_sync(0xffffffffu, m_xyz, o);
        m_xz += __shfl_down_sync(0xffffffffu, m_xz, o);
        m_yz += __shfl_down_sync(0xffffffffu, m_yz, o);
        m_ab += __shfl_down_sync(0xffffffffu, m_ab, o);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) {
        red[warp] = cmi;
        red[32 + warp] = mi;
        cnt_red[0][warp] = m_xyz; cnt_red[1][warp] = m_xz; cnt_red[2][warp] = m_yz; cnt_red[3][warp] = m_ab;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        {ACC} c_tot = 0.0, m_tot = 0.0;
        int a = 0, b = 0, c = 0, d = 0;
        for (int w = 0; w < (blockDim.x >> 5); ++w) { c_tot += red[w]; m_tot += red[32 + w]; a += cnt_red[0][w]; b += cnt_red[1][w]; c += cnt_red[2][w]; d += cnt_red[3][w]; }
        int m_z = 0, m_x = 0, m_zz = 0;
        for (int t = 0; t < ky; ++t) if (py[t] > 0) m_z += 1;
        for (int t = 0; t < kx; ++t) if (px[t] > 0) m_x += 1;
        for (int t = 0; t < kz; ++t) if (pz[t] > 0) m_zz += 1;
        cmi_out[p] = c_tot - (({ACC})(a - b - c + m_z)) / (2.0 * nf);
        mi_out[p] = m_tot - (({ACC})(d - m_x - m_zz + 1)) / (2.0 * nf);
    }
}
""",
)

_KERNELS: dict = {}
_KERNELS_LOCK = threading.RLock()  # serialises the get-or-build of the cache above (one build per key even with concurrent fits)


def _kernel(cp: Any, narrow: bool) -> Any:
    """The compiled kernel, one per code type: bytes when every cardinality is at most 127, else 32-bit integers."""
    with _KERNELS_LOCK:
        if narrow not in _KERNELS:
            _KERNELS[narrow] = cp.RawKernel(_SRC.replace("{CT}", "signed char" if narrow else "int"), "relaxmrmr_pair")
        return _KERNELS[narrow]


_kernel_factory = code_width_kernel(_kernel)


def kernel(cp: Any, real: Any) -> Any:
    """``relaxmrmr_pair`` for byte or 32-bit codes (``real`` is the code dtype); the register-pressure registry compiles it through this ``kernel(cp, real)`` factory."""
    return _kernel_factory(cp, real)


def _cells(k_x: int, k_y: int, k_sel: np.ndarray, pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """Shared integer entries one block needs for each pair."""
    kz = k_sel[pi].astype(np.int64) * k_sel[pj].astype(np.int64)
    return cast(np.ndarray, k_x * kz * k_y + k_x * k_y + kz * k_y + k_x * kz + k_x + kz + k_y)


def pair_terms_gpu(
    x: np.ndarray, y: np.ndarray, sel_mat: np.ndarray, k_sel: np.ndarray, k_x: int, k_y: int, pair_i: np.ndarray, pair_j: np.ndarray
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """``(cmi_ij, mi_xzz, fits)`` per pair: ``I(x; z | y)`` and ``I(x; z)`` (Miller-Madow, as ``_cmi_mm_njit`` / ``_mi_mm_njit``) of the composite code of the pair, and which pairs the device scored. Raises on a device fault."""
    import cupy as cp

    n_pairs = int(pair_i.shape[0])
    n = int(x.shape[0])
    cmi = np.zeros(n_pairs)  # Intentional float64: MI statistics compared to the host's at full precision
    mi = np.zeros(n_pairs)  # Intentional float64: MI statistics compared to the host's at full precision
    kk = np.asarray(k_sel, dtype=np.int64)
    fits = _cells(int(k_x), int(k_y), kk, pair_i, pair_j) <= _MAX_CELLS
    idx = np.nonzero(fits)[0]
    if n_pairs == 0 or n == 0 or idx.size == 0:
        return cmi, mi, fits
    narrow = bool(max(int(kk.max()), int(k_x), int(k_y)) <= 127)
    ct = np.int8 if narrow else np.int32
    sel_d = cp.asarray(np.ascontiguousarray(sel_mat, dtype=ct))
    x_d = cp.asarray(np.ascontiguousarray(x, dtype=ct))
    y_d = cp.asarray(np.ascontiguousarray(y, dtype=ct))
    ks_d = cp.asarray(kk.astype(np.int32))
    cmi_d = cp.empty(idx.size, dtype=cp.float64)  # Intentional float64: statistic output
    mi_d = cp.empty(idx.size, dtype=cp.float64)  # Intentional float64: statistic output
    shared = int(_cells(int(k_x), int(k_y), kk, pair_i[idx], pair_j[idx]).max()) * 4
    _kernel(cp, narrow)(  # launch-ok: one block per pair is the design
        (int(idx.size),),
        (_THREADS,),
        (sel_d, x_d, y_d, cp.asarray(pair_i[idx].astype(np.int32)), cp.asarray(pair_j[idx].astype(np.int32)), ks_d, np.int64(n), np.int32(k_x), np.int32(k_y), cmi_d, mi_d),
        shared_mem=shared,
    )
    cmi[idx] = cp.asnumpy(cmi_d)
    mi[idx] = cp.asnumpy(mi_d)
    return cmi, mi, fits


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DEVICE_FALLBACK_ROWS = 2_000_000  # rows x pairs from which the device scores the pairs before a sweep has run
_SWEEP_N = 20_000
_SWEEP_ROWS = [_SWEEP_N * k for k in (10, 45, 190, 435)]
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """A selection of ``s`` features of 6 levels, with ``s`` chosen so that C(s, 2) is about ``dims["n_samples"] / 20000``, over 20000 rows; a 6-level candidate and a 3-class target."""
    rng = np.random.default_rng(0)
    want = max(1, int(dims["n_samples"]) // _SWEEP_N)
    s = 2
    while (s + 1) * s // 2 <= want:
        s += 1
    sel = rng.integers(0, 6, (s, _SWEEP_N)).astype(np.int64)
    x = rng.integers(0, 6, _SWEEP_N).astype(np.int64)
    y = ((x + sel[0]) % 3).astype(np.int64)
    pi, pj = np.triu_indices(s, k=1)
    return (x, y, sel, np.full(s, 6, dtype=np.int64), 6, 3, pi.astype(np.int64), pj.astype(np.int64))


def _sweep_host(x: np.ndarray, y: np.ndarray, sel: np.ndarray, k_sel: np.ndarray, kx: int, ky: int, pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """The production numba ``prange`` pair loop, per-pair term ``mi_xzz - cmi_ij`` (the per-feature terms set to zero, no pair skipped)."""
    import numba

    from ._relaxmrmr_pair_loop import _pair_interaction_terms_njit

    zeros = np.zeros(sel.shape[0])
    scratch = np.empty((max(1, int(numba.get_num_threads())), sel.shape[1]), dtype=np.int64)
    return cast(np.ndarray, np.asarray(_pair_interaction_terms_njit(x, y, sel, k_sel, np.int64(kx), np.int64(ky), zeros, zeros, 0.0, scratch, pi, pj)))


def _sweep_device(x: np.ndarray, y: np.ndarray, sel: np.ndarray, k_sel: np.ndarray, kx: int, ky: int, pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """The device kernel; every sweep pair fits, so a failed launch raises and is never recorded as a host time."""
    cmi, mi, _ = pair_terms_gpu(x, y, sel, k_sel, kx, ky, pi, pj)
    return cast(np.ndarray, mi - cmi)


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="relaxmrmr_pair_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-9, atol=1e-12),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__),
        gpu_capable=True,
        salt=_SALT,
        cli_label="relaxmrmr_pair_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for relaxmrmr_pair_backend failed: %s", e)
    _SPEC = None


from ._ktc_fallback import ktc_device_chooser as _ktc_device_chooser

pair_device_chosen = _ktc_device_chooser(__name__, "relaxmrmr_pair_backend", pairs=True)


def pair_terms_dispatch(
    x: np.ndarray, y: np.ndarray, sel_mat: np.ndarray, k_sel: np.ndarray, k_x: int, k_y: int, pair_i: np.ndarray, pair_j: np.ndarray
) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | None":
    """``(cmi_ij, mi_xzz, fits)`` from the device when the tuned choice says so and a device is usable, else ``None`` (the numba loop runs); a device fault returns ``None`` with a throttled warning."""
    if pair_i.shape[0] and pair_device_chosen(int(x.shape[0]), int(pair_i.shape[0])):
        try:
            from ._gpu_policy import cuda_available_for_run

            if cuda_available_for_run():
                return pair_terms_gpu(x, y, sel_mat, k_sel, k_x, k_y, pair_i, pair_j)
        except Exception as exc:  # a device fault must not lose the greedy round
            from mlframe.utils.log_throttle import log_throttle

            log_throttle(logger, "relaxmrmr_pair.device_failed", logging.WARNING, "RelaxMRMR pair loop device path failed (%s: %s); numba loop", type(exc).__name__, exc)
    return None
