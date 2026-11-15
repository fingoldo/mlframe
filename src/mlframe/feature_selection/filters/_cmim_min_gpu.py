"""CMIM scores of a block of candidates against every reference column: a numba kernel, a fused CUDA kernel and the per-host choice between them.

``_orthogonal_cmim_fe.score_features_by_cmim`` scores each engineered candidate ``x`` by ``min over the references z of CMI(x; y | z)`` (Fleuret 2004), skipping the references of the candidate's own
source. The host loop factorised ``x z``, ``x y z`` per (candidate, reference) pair with ``np.unique``; the device path made one ``batched_cmi_gpu`` call per reference (about forty launches sets per call).

For codes in small ranges one joint table does it all. With ``c_xyz`` the counts of a (candidate, reference) pair and ``c_xz``, ``c_yz``, ``c_z`` their marginals,
``CMI = sum_{c_xyz > 0} (c_xyz / n) ln(c_xyz c_z / (c_xz c_yz))`` and the Miller-Madow bias is ``(k_xyz + k_z - k_xz - k_yz) / (2n)`` over the occupied cells, the same estimator as
``_cmi_from_binned_with_cached_z`` (``H_xz + H_yz - H_z - H_xyz - bias``, clipped at 0), summed in another order (agreement to about 1e-15).

Two backends of one function: ``_scores_njit`` (``prange`` over candidates, early exit at a zero) and ``cmim_min_scores`` of the device (``cmim_min_blocks``: ONE launch for the whole block, a CUDA block per
candidate walking the references, the joint table in shared memory). ``cmim_min_on_device`` reads the per-host kernel-tuning cache (``cmim_min_backend``) with a row-count fallback;
``MLFRAME_CMIM_MIN_DEVICE=0|1`` forces a side. Codes whose joint table would not fit (``MAX_CELLS``) are declined (``None``) and the caller keeps the older paths.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

import numpy as np
from numba import njit, prange

from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback
from ._raw_module_cache import RawModuleCache

logger = logging.getLogger(__name__)

MAX_CELLS = 6144  # int32 cells of the joint table of one pair held in shared memory (24 KB)

_SRC = render(
    r"""
extern "C" __global__ void cmim_min_blocks(const {XT}* __restrict__ x, const int* __restrict__ y, const int* __restrict__ zs, const int* __restrict__ kzs,
                                            const unsigned char* __restrict__ allowed, const long long n, const int n_cand, const int n_ref, const int kx, const int ky, const int kz_max,
                                            {ACC}* __restrict__ out) {
    // one block per candidate: for every allowed reference the joint table (x, y, z) is counted in shared memory, its marginals are taken from it, and the CMI is summed over the cells
    extern __shared__ int sh[];
    const int cells = kx * ky * kz_max;
    int* cxyz = sh;                       // cells
    int* cxz = cxyz + cells;              // kx * kz_max
    int* cyz = cxz + kx * kz_max;         // ky * kz_max
    int* cz = cyz + ky * kz_max;          // kz_max
    __shared__ {ACC} red[32];
    __shared__ int s_done;
    const long long cand = blockIdx.x;
    const {XT}* xc = x + cand * n;
    {ACC} best = 1e300;
    bool any = false;
    if (threadIdx.x == 0) s_done = 0;
    __syncthreads();
    for (int j = 0; j < n_ref; ++j) {
        if (s_done) break;   // a zero is the floor of the clipped values: the minimum cannot drop below it
        if (!allowed[(long long)j * n_cand + cand]) continue;
        const int* zr = zs + (long long)j * n;
        const int kz = kzs[j];
        for (int c = threadIdx.x; c < cells + kx * kz_max + ky * kz_max + kz_max; c += blockDim.x) sh[c] = 0;
        __syncthreads();
        for (long long i = threadIdx.x; i < n; i += blockDim.x) atomicAdd(&cxyz[((int)xc[i] * ky + y[i]) * kz + zr[i]], 1);
        __syncthreads();
        for (int c = threadIdx.x; c < kx * ky * kz; c += blockDim.x) {
            const int v = cxyz[c];
            if (v) {
                const int z = c % kz, xy = c / kz, yy = xy % ky, xx = xy / ky;
                atomicAdd(&cxz[xx * kz + z], v);
                atomicAdd(&cyz[yy * kz + z], v);
                atomicAdd(&cz[z], v);
            }
        }
        __syncthreads();
        {ACC} s = 0;
        int k_xyz = 0, k_xz = 0, k_yz = 0, k_z = 0;
        for (int c = threadIdx.x; c < kx * ky * kz; c += blockDim.x) {
            const int v = cxyz[c];
            if (v) {
                const int z = c % kz, xy = c / kz, yy = xy % ky, xx = xy / ky;
                s += ({ACC})v * log(({ACC})v * ({ACC})cz[z] / (({ACC})cxz[xx * kz + z] * ({ACC})cyz[yy * kz + z]));
                ++k_xyz;
            }
        }
        for (int c = threadIdx.x; c < kx * kz; c += blockDim.x) k_xz += (cxz[c] > 0);
        for (int c = threadIdx.x; c < ky * kz; c += blockDim.x) k_yz += (cyz[c] > 0);
        for (int c = threadIdx.x; c < kz; c += blockDim.x) k_z += (cz[c] > 0);
        for (int o = 16; o > 0; o >>= 1) {
            s += __shfl_down_sync(0xffffffffu, s, o);
            k_xyz += __shfl_down_sync(0xffffffffu, k_xyz, o);
            k_xz += __shfl_down_sync(0xffffffffu, k_xz, o);
            k_yz += __shfl_down_sync(0xffffffffu, k_yz, o);
            k_z += __shfl_down_sync(0xffffffffu, k_z, o);
        }
        __shared__ int kw[32][4];
        if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = s; kw[threadIdx.x >> 5][0] = k_xyz; kw[threadIdx.x >> 5][1] = k_xz; kw[threadIdx.x >> 5][2] = k_yz; kw[threadIdx.x >> 5][3] = k_z; }
        __syncthreads();
        if (threadIdx.x == 0) {
            {ACC} tot = 0;
            int a = 0, b = 0, d = 0, e = 0;
            for (int w = 0; w < (int)(blockDim.x >> 5); ++w) { tot += red[w]; a += kw[w][0]; b += kw[w][1]; d += kw[w][2]; e += kw[w][3]; }
            const {ACC} cmi = tot / ({ACC})n - ({ACC})(a + e - b - d) / (2.0 * ({ACC})n);
            const {ACC} clipped = cmi > 0 ? cmi : 0;
            if (clipped < best) best = clipped;
            any = true;
            if (best <= 0) s_done = 1;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[cand] = any ? best : 1e300;
}
"""
)
_MODULES = {
    "short": RawModuleCache(_SRC.replace("{XT}", "short")),
    "unsigned char": RawModuleCache(_SRC.replace("{XT}", "unsigned char")),
}


def kernel(cp: Any, code_type: Any) -> Any:
    """``cmim_min_blocks`` for candidate codes of numpy dtype ``code_type`` (uint8 or int16); compile point of :func:`_scores_device` and of the register-pressure registry."""
    key = "unsigned char" if np.dtype(code_type) == np.uint8 else "short"
    return _MODULES[key].get(cp).get_function("cmim_min_blocks")


@njit(cache=True, nogil=True)
def _pair_cmi(x_row: np.ndarray, y: np.ndarray, z_row: np.ndarray, kx: int, ky: int, kz: int, cxyz: np.ndarray, cxz: np.ndarray, cyz: np.ndarray, cz: np.ndarray) -> float:
    """Miller-Madow CMI of one (candidate, reference) pair, clipped at 0, from the joint table ``cxyz`` and the marginals (scratch arrays of the caller, cleared here)."""
    n = x_row.shape[0]
    cxyz[: kx * ky * kz] = 0
    cxz[: kx * kz] = 0
    cyz[: ky * kz] = 0
    cz[:kz] = 0
    for i in range(n):
        cxyz[(x_row[i] * ky + y[i]) * kz + z_row[i]] += 1
    k_xyz = 0
    for xx in range(kx):
        for yy in range(ky):
            for z in range(kz):
                v = cxyz[(xx * ky + yy) * kz + z]
                if v:
                    cxz[xx * kz + z] += v
                    cyz[yy * kz + z] += v
                    cz[z] += v
                    k_xyz += 1
    s = 0.0
    for xx in range(kx):
        for yy in range(ky):
            for z in range(kz):
                v = cxyz[(xx * ky + yy) * kz + z]
                if v:
                    s += v * np.log(v * np.float64(cz[z]) / (np.float64(cxz[xx * kz + z]) * np.float64(cyz[yy * kz + z])))
    k_xz = np.count_nonzero(cxz[: kx * kz])
    k_yz = np.count_nonzero(cyz[: ky * kz])
    k_z = np.count_nonzero(cz[:kz])
    cmi = s / n - (k_xyz + k_z - k_xz - k_yz) / (2.0 * n)
    return cmi if cmi > 0.0 else 0.0


@njit(cache=True, parallel=True, nogil=True)
def _scores_njit(x: np.ndarray, y: np.ndarray, zs: np.ndarray, kzs: np.ndarray, allowed: np.ndarray, kx: int, ky: int) -> np.ndarray:
    """``(K,)`` minima over the allowed references of the Miller-Madow CMI clipped at 0 (``1e300`` for a candidate with no allowed reference); ``x`` is ``(K, n)``, ``zs`` ``(S, n)``, ``allowed`` ``(S, K)``."""
    k = x.shape[0]
    s_count = zs.shape[0]
    kz_max = 1
    for j in range(s_count):
        if kzs[j] > kz_max:
            kz_max = kzs[j]
    out = np.empty(k, dtype=np.float64)
    for c in prange(k):
        cxyz = np.zeros(kx * ky * kz_max, dtype=np.int32)
        cxz = np.zeros(kx * kz_max, dtype=np.int32)
        cyz = np.zeros(ky * kz_max, dtype=np.int32)
        cz = np.zeros(kz_max, dtype=np.int32)
        best = 1e300
        for j in range(s_count):
            if not allowed[j, c]:
                continue
            cmi = _pair_cmi(x[c], y, zs[j], kx, ky, kzs[j], cxyz, cxz, cyz, cz)
            if cmi < best:
                best = cmi
            if best <= 0.0:
                break
        out[c] = best
    return out


def _prepare(x: np.ndarray, y: np.ndarray, zs: Sequence[np.ndarray], n_bins: int) -> Optional[tuple]:
    """The packed inputs ``(x, y, zs, kzs, kx, ky)`` or ``None`` when the codes are out of range or the joint table would exceed ``MAX_CELLS``."""
    if x.ndim != 2 or not len(zs) or not x.shape[0]:
        return None
    y_i = np.ascontiguousarray(y, dtype=np.int32)
    z_mat = np.ascontiguousarray(np.stack([np.asarray(z) for z in zs]), dtype=np.int32)
    if int(x.min()) < 0 or int(x.max()) >= int(n_bins) or int(y_i.min()) < 0 or int(z_mat.min()) < 0:
        return None
    kx, ky = int(n_bins), int(y_i.max()) + 1
    kzs = z_mat.max(axis=1).astype(np.int32) + 1
    if kx * ky * int(kzs.max()) > MAX_CELLS:
        return None
    return np.ascontiguousarray(x), y_i, z_mat, kzs, kx, ky


def _scores_host(x: np.ndarray, y: np.ndarray, zs: Sequence[np.ndarray], allowed: np.ndarray, n_bins: int) -> Optional[np.ndarray]:
    """Host form: the parallel numba kernel."""
    from mlframe._numba_parallel_guard import parallel_kernel_entry

    packed = _prepare(x, y, zs, n_bins)
    if packed is None:
        return None
    xp, y_i, z_mat, kzs, kx, ky = packed
    with parallel_kernel_entry():
        out: np.ndarray = _scores_njit(xp, y_i, z_mat, kzs, np.ascontiguousarray(allowed, dtype=np.bool_), kx, ky)
    return out


def _scores_device(x: np.ndarray, y: np.ndarray, zs: Sequence[np.ndarray], allowed: np.ndarray, n_bins: int) -> Optional[np.ndarray]:
    """Device form: one launch for the block, or ``None`` (no cupy, codes out of range, a device fault)."""
    try:
        import cupy as cp
        from pyutilz.performance.gpu_hw_launch import block_size_for

        packed = _prepare(x, y, zs, n_bins)
        if packed is None:
            return None
        xp, y_i, z_mat, kzs, kx, ky = packed
        k, n = xp.shape
        s_count = int(z_mat.shape[0])
        code_dtype = np.uint8 if int(n_bins) <= 255 else np.int16
        kz_max = int(kzs.max())
        shared = 4 * (kx * ky * kz_max + kx * kz_max + ky * kz_max + kz_max)
        out = cp.empty(k, dtype=cp.float64)
        ker = kernel(cp, code_dtype)
        ker(
            (k,), (block_size_for(ker, dyn_smem=shared, work_items=n),),
            (cp.asarray(xp.astype(code_dtype, copy=False)), cp.asarray(y_i), cp.asarray(z_mat), cp.asarray(kzs), cp.asarray(np.ascontiguousarray(allowed, dtype=np.uint8)),
             np.int64(n), np.int32(k), np.int32(s_count), np.int32(kx), np.int32(ky), np.int32(kz_max), out),
            shared_mem=shared,
        )
        return np.asarray(cp.asnumpy(out), dtype=np.float64)
    except Exception as exc:
        logger.debug("cmim min: the device form is unavailable (%s: %s)", type(exc).__name__, exc)
        return None


DEVICE_FALLBACK_ROWS = 20_000
_SWEEP_ROWS = [5_000, 20_000, 50_000, 200_000]
_SWEEP_CANDIDATES = 96
_SWEEP_REFS = 24
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``(x, y, zs, allowed, n_bins)``: ``_SWEEP_CANDIDATES`` binned candidates against ``_SWEEP_REFS`` references over ``dims["n_samples"]`` rows."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    n_bins = 10
    y = (rng.random(n) < 0.4).astype(np.int64)
    # every candidate carries the target (a binary signal on the high or low half of the bins), so no pair reaches a zero early: the sweep times the full table work, the case the kernels exist for
    x = (rng.integers(0, n_bins // 2, size=(_SWEEP_CANDIDATES, n)) + (n_bins // 2) * np.where(rng.random((_SWEEP_CANDIDATES, n)) < 0.85, y[None, :], 1 - y[None, :])).astype(np.int16)
    zs = [rng.integers(0, n_bins, size=n) for _ in range(_SWEEP_REFS)]
    allowed = rng.random((_SWEEP_REFS, _SWEEP_CANDIDATES)) < 0.95
    return (x, y, zs, allowed, n_bins)


def _sweep_host(x: np.ndarray, y: np.ndarray, zs: Sequence[np.ndarray], allowed: np.ndarray, n_bins: int) -> np.ndarray:
    """The host form on the sweep inputs."""
    got = _scores_host(x, y, zs, allowed, n_bins)
    if got is None:
        raise RuntimeError("host cmim min declined")
    return got


def _sweep_device(x: np.ndarray, y: np.ndarray, zs: Sequence[np.ndarray], allowed: np.ndarray, n_bins: int) -> np.ndarray:
    """The device form on the sweep inputs."""
    got = _scores_device(x, y, zs, allowed, n_bins)
    if got is None:
        raise RuntimeError("device cmim min declined")
    return got


_fallback_choice = rows_fallback(__name__)

try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="cmim_min_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-9, atol=1e-9),
        axes={"n_rows": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="cmim_min_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the row-count fallback decides
    logger.warning("kernel_tuner spec build for cmim_min_backend failed: %s", e)
    _SPEC = None


def cmim_min_on_device(n_rows: int) -> bool:
    """Whether a block of candidates of ``n_rows`` rows is scored on the device: ``MLFRAME_CMIM_MIN_DEVICE=0|1`` forces a side, else the per-host kernel-tuning cache, else the row mark."""
    forced = os.environ.get("MLFRAME_CMIM_MIN_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_rows=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("cmim_min_backend: KTC choose() failed, using the row-count fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


#: Host bytes of one candidate block (int16 codes of its candidates).
_BLOCK_BYTES = 256 * 1024 * 1024


def candidate_block_width(n_rows: int) -> int:
    """Candidates per column block: as many as ``_BLOCK_BYTES`` holds at ``n_rows`` rows of int16 codes (at least one)."""
    return max(1, _BLOCK_BYTES // max(1, 2 * int(n_rows)))


def cmim_min_scores(x: np.ndarray, y: np.ndarray, zs: Sequence[np.ndarray], allowed: np.ndarray, n_bins: int) -> Optional[np.ndarray]:
    """``(K,)`` CMIM scores of the binned candidates ``x`` ``(K, n)`` against the references ``zs`` (``allowed`` ``(S, K)`` says which reference counts for which candidate): the minimum over the allowed
    references of the Miller-Madow CMI clipped at 0, ``1e300`` for a candidate with no allowed reference. The backend is the one the kernel-tuning cache picks; the candidates go through in column blocks of
    :func:`candidate_block_width`. ``None`` when the codes are out of range or the joint table is too large (the caller keeps its older paths)."""
    n = int(x.shape[1]) if x.ndim == 2 else 0
    width = candidate_block_width(n)
    on_device = cmim_min_on_device(n)
    parts = []
    for a in range(0, int(x.shape[0]), width):
        block, mask = x[a : a + width], allowed[:, a : a + width]
        got = _scores_device(block, y, zs, mask, n_bins) if on_device else None
        if got is None:
            got = _scores_host(block, y, zs, mask, n_bins)
        if got is None:
            return None
        parts.append(got)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
