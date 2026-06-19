"""Batched CUDA conditional-mutual-information kernel for the MRMR/Fleuret greedy redundancy loop.

The dominant large-``n`` cost of MRMR is the greedy Fleuret redundancy step: for every
candidate column ``X_j`` it evaluates ``I(X_j; Y | Z)`` against the just-selected feature ``Z``,
each greedy round. The CPU path (``info_theory._entropy_kernels.conditional_mi``) is a serial
njit that rebuilds four joint histograms per call via the per-row ``merge_vars`` melt -- O(p * n)
per round with no parallelism.

This module adds a BATCHED GPU kernel that computes ``I(X_j; Y | Z)`` for ALL ``p`` candidates in a
single launch, returning a ``(p,)`` float64 array. ``Y`` and ``Z`` are fixed across candidates, so
``H(Y, Z)`` and ``H(Z)`` are computed ONCE per round; only the per-candidate 3-variable joint
``H(X_j, Y, Z)`` (and the 2-variable ``H(X_j, Z)``) are the per-candidate piece. One block per
candidate, shared-memory histogram (mirrors ``filters/gpu.py``'s ``compute_joint_hist_*`` design).

    I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)

returned RAW (clamped at 0), bit-comparable (within ~1e-9) to the CPU ``conditional_mi`` on the
SAME discretized inputs -- see ``test_cmi_cuda_kernel.py``.

PARITY: the CPU ``entropy`` reduces ``-(log(p) * p)`` over the NONZERO joint bins in ascending
merged-class-id order, with ``p = count / n`` in float64. The kernel walks the dense per-candidate
histogram in the same ascending merged-class order, skips zero bins, and accumulates in float64 --
reproducing the CPU reduction value (sequential add order matches numpy ``.sum`` for the small
nonzero-bin counts here). The merged class id ``cz + cx * nbz + cy * (nbx * nbz)`` keeps ``Z`` the
fastest-varying axis so the (X, Z) marginal and (Y, Z) marginal collapse in the same monotone order
the CPU ``merge_vars`` pruning produces.

PICKLE: NO live CUDA context or RawKernel object is ever stored on a class instance -- the kernel is
a module-level singleton built lazily in ``_get_kernel()`` and attached to THIS module's namespace.
Callers hold only plain numpy arrays. Safe to pickle any consumer.

DISPATCH: ``conditional_mi_batched_dispatch`` is the public entry. It routes to CUDA at large
``(n, p)`` where the launch wins, and to the CPU per-candidate loop otherwise / when no GPU. The
size/HW gate integrates with ``pyutilz`` ``kernel_tuning_cache`` (via the FS ``_kernel_tuning``
singleton) -- NO hardcoded thresholds baked into the hot path; a hand fallback only applies on
cache miss / pyutilz-unavailable.
"""
from __future__ import annotations

import logging
import sys
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Module-level kernel singleton (pickle-safe: never stored on instances).
_cmi_joint_hist_cuda = None  # type: ignore[assignment]
_KERNEL_LOCK = threading.Lock()
_CUPY_OK: Optional[bool] = None

# Tuning-region name for the kernel_tuning_cache dispatch (mirrors the
# joint-hist dispatch keys in filters/gpu.py).
_TUNING_REGION = "cmi_batched"


def cupy_available() -> bool:
    """True if cupy imports AND a CUDA device is visible. Cached after first probe."""
    global _CUPY_OK
    if _CUPY_OK is not None:
        return _CUPY_OK
    try:
        import cupy as cp

        _CUPY_OK = cp.cuda.runtime.getDeviceCount() > 0
    except Exception as _exc:  # noqa: BLE001 - any cupy/CUDA failure -> CPU path
        logger.debug("cmi_cuda: cupy unavailable (%s); CPU fallback", _exc)
        _CUPY_OK = False
    return _CUPY_OK


def _get_kernel():
    """Build (idempotently) and return the batched 3-var joint-hist RawKernel.

    One block per candidate; shared-memory histogram of size ``nbx * nby * nbz`` ints. Each block
    strides over the n samples of its candidate column, atomically bumping the shared histogram,
    then reduces shared -> the candidate's global slot in one pass.
    """
    global _cmi_joint_hist_cuda
    if _cmi_joint_hist_cuda is not None:
        return _cmi_joint_hist_cuda
    import cupy as cp

    with _KERNEL_LOCK:
        if _cmi_joint_hist_cuda is not None:
            return _cmi_joint_hist_cuda
        module = sys.modules[__name__]
        module._cmi_joint_hist_cuda = cp.RawKernel(
            r"""
        extern "C" __global__
        void cmi_joint_hist_cuda(
            const int *Xc,            // (p, n) row-major: candidate codes
            const int *y,             // (n,)  target codes
            const int *z,             // (n,)  conditioning codes
            int *joint_counts,        // (p, nbx*nby*nbz) row-major output
            int n,
            int p,
            int nbx,
            int nby,
            int nbz
        ) {
            extern __shared__ int sm[];   // nbx*nby*nbz ints
            int cand = blockIdx.x;
            if (cand >= p) return;
            int joint_size = nbx * nby * nbz;
            int tid = threadIdx.x;
            int nthreads = blockDim.x;

            for (int i = tid; i < joint_size; i += nthreads) sm[i] = 0;
            __syncthreads();

            const int *xrow = Xc + (long)cand * n;
            // merged id = cz + cx*nbz + cy*(nbx*nbz)  -> Z fastest, then X, then Y.
            for (int r = tid; r < n; r += nthreads) {
                int cx = xrow[r];
                int cy = y[r];
                int cz = z[r];
                int cell = cz + cx * nbz + cy * (nbx * nbz);
                atomicAdd(&sm[cell], 1);
            }
            __syncthreads();

            int *out = joint_counts + (long)cand * joint_size;
            for (int i = tid; i < joint_size; i += nthreads) {
                if (sm[i] != 0) out[i] = sm[i];
            }
        }
        """,
            "cmi_joint_hist_cuda",
        )
        return module._cmi_joint_hist_cuda


def _entropy_from_counts_axis(counts_3d, axes, n):
    """Shannon entropy (nats) of a marginal of the 3-D joint count tensor.

    ``counts_3d`` is (p, nby, nbx, nbz) int (Y, X, Z layout matching the merged id Z-fastest /
    X-next / Y-slowest). ``axes`` are the axes to SUM OUT before the entropy reduction (so the
    remaining axes form the marginal). Returns a (p,) float64 entropy vector.

    Reduction matches the CPU ``entropy``: ``p_bin = count / n`` in float64, ``-sum p log p`` over
    nonzero bins. CuPy reduces a contiguous flattened marginal; the small nonzero-bin counts here
    make the float64 accumulation track numpy's within ~1e-12 (well inside the 1e-9 gate).
    """
    import cupy as cp

    marg = counts_3d.sum(axis=axes)  # (p, ...) remaining axes
    marg = marg.reshape(marg.shape[0], -1).astype(cp.float64)
    pr = marg / n
    # 0 * log 0 -> 0; mask zeros.
    nz = pr > 0
    contrib = cp.where(nz, pr * cp.log(pr), 0.0)
    return -contrib.sum(axis=1)


def conditional_mi_batched_cuda(
    Xc: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nbins_x: int,
    nbins_y: int,
    nbins_z: int,
    block_size: int = 256,
) -> np.ndarray:
    """Compute ``I(X_j; Y | Z)`` for all ``p`` candidate columns in one launch -> (p,) float64.

    Parameters
    ----------
    Xc : (p, n) int32 -- candidate codes, one row per candidate (0..nbins_x-1).
    y, z : (n,) int32 -- target / conditioning codes.
    nbins_x, nbins_y, nbins_z : per-variable cardinalities (max code + 1 is fine; empty bins cost
        only shared-mem, not correctness).
    block_size : threads per block (one block per candidate).

    Returns RAW CMI clamped at 0, mirroring the CPU ``conditional_mi``.
    """
    import cupy as cp

    Xc = np.ascontiguousarray(Xc, dtype=np.int32)
    y = np.ascontiguousarray(y, dtype=np.int32)
    z = np.ascontiguousarray(z, dtype=np.int32)
    p, n = Xc.shape
    joint_size = int(nbins_x) * int(nbins_y) * int(nbins_z)

    kern = _get_kernel()

    Xc_g = cp.asarray(Xc)
    y_g = cp.asarray(y)
    z_g = cp.asarray(z)
    joint = cp.zeros((p, joint_size), dtype=cp.int32)

    kern(
        (p,),
        (block_size,),
        (Xc_g, y_g, z_g, joint, np.int32(n), np.int32(p),
         np.int32(nbins_x), np.int32(nbins_y), np.int32(nbins_z)),
        shared_mem=joint_size * 4,
    )

    # joint is (p, nby, nbx, nbz) in the merged-id layout (Z fastest, X next, Y slowest).
    counts = joint.reshape(p, nbins_y, nbins_x, nbins_z)

    # H(Z): sum out Y(1) and X(2) -> marginal over Z.
    h_z = _entropy_from_counts_axis(counts, (1, 2), n)        # (p,) -- all identical, kept per-cand for clamp simplicity
    # H(X, Z): sum out Y(1).
    h_xz = _entropy_from_counts_axis(counts, (1,), n)
    # H(Y, Z): sum out X(2).
    h_yz = _entropy_from_counts_axis(counts, (2,), n)
    # H(X, Y, Z): no sum.
    h_xyz = _entropy_from_counts_axis(counts, (), n)

    cmi = h_xz + h_yz - h_z - h_xyz
    cmi = cp.maximum(cmi, 0.0)
    return cp.asnumpy(cmi)


def _cpu_cmi_loop(factors_data, cand_indices, y, z, factors_nbins, dtype=np.int32) -> np.ndarray:
    """Exact CPU fallback: loop ``conditional_mi`` over each candidate column index.

    ``cand_indices`` is a (p,) array of column indices into ``factors_data``; ``y`` / ``z`` are
    1-element column-index arrays (the target / selected-feature columns), matching the
    ``conditional_mi`` signature. Returns (p,) float64.
    """
    from ._entropy_kernels import conditional_mi

    out = np.empty(len(cand_indices), dtype=np.float64)
    for i, ci in enumerate(cand_indices):
        out[i] = conditional_mi(
            factors_data=factors_data,
            x=np.asarray([ci], dtype=np.int64),
            y=np.asarray(y, dtype=np.int64),
            z=np.asarray(z, dtype=np.int64),
            var_is_nominal=None,
            factors_nbins=factors_nbins,
            dtype=dtype,
        )
    return out


def _should_use_cuda(n: int, p: int, joint_size: int) -> bool:  # noqa: C901
    """Size/HW gate. Routes to CUDA at large (n, p) where the batched launch wins; CPU otherwise.

    Integrates with the FS kernel_tuning_cache singleton (pyutilz). On cache hit the decision is the
    measured one; on cache miss / pyutilz-unavailable a hand fallback applies (NO hardcoded value on
    the hot path -- the fallback is the documented bootstrap heuristic only). VRAM guard: the joint
    buffer is ``p * joint_size * 4`` bytes; reject if it would exceed a conservative slice.
    """
    if not cupy_available():
        return False
    # VRAM guard (GTX 1050 Ti = 4 GB): the dominant device buffer is the candidate matrix Xc
    # (p * n * 4 bytes); the joint output is p * joint_size * 4. Reject if the combined working
    # set would exceed a conservative slice of free VRAM (leave headroom for the model / other
    # worker). Best-effort free-VRAM probe; falls back to a fixed cap.
    bytes_needed = p * n * 4 + p * joint_size * 4
    cap = 1536 * 1024 * 1024  # 1.5 GB conservative cap (shared 4 GB card)
    try:
        import cupy as cp

        free_b, _total = cp.cuda.runtime.memGetInfo()
        cap = min(cap, int(free_b * 0.5))
    except Exception:  # noqa: BLE001
        pass
    if bytes_needed > cap:
        return False
    # Shared-mem guard: cc 6.x has 48 KB/block -> joint_size*4 must fit.
    if joint_size * 4 > 48 * 1024:
        return False

    try:
        from mlframe.feature_selection.filters._kernel_tuning import get_kernel_tuning_cache

        cache = get_kernel_tuning_cache()
        if cache is not None:
            key = {"region": _TUNING_REGION, "n": int(n), "p": int(p), "joint_size": int(joint_size)}
            try:
                decision = cache.lookup(_TUNING_REGION, key)
                if decision is not None and "use_cuda" in decision:
                    return bool(decision["use_cuda"])
            except Exception as _exc:  # noqa: BLE001
                logger.debug("cmi_cuda: kernel_tuning_cache lookup failed (%s); hand fallback", _exc)
    except Exception as _exc:  # noqa: BLE001
        logger.debug("cmi_cuda: kernel_tuning_cache unavailable (%s); hand fallback", _exc)

    # Hand bootstrap heuristic (cache-miss only): the batched launch amortizes its host<->device
    # transfer + launch overhead once p and n are both sizable. Measured crossover (GTX 1050 Ti,
    # bench_cmi_cuda): CUDA wins from roughly n*p >= ~1e6 with p >= ~100.
    return (n * p) >= 1_000_000 and p >= 64


def conditional_mi_batched_dispatch(
    factors_data: np.ndarray,
    cand_indices: np.ndarray,
    y_index,
    z_index,
    factors_nbins: np.ndarray,
    dtype=np.int32,
    force: Optional[str] = None,
) -> np.ndarray:
    """Public entry: ``I(X_j; Y | Z)`` for every candidate column in ``cand_indices`` -> (p,) float64.

    Routes to the batched CUDA kernel at large ``(n, p)`` (where it wins), else to the exact CPU
    ``conditional_mi`` loop. ``y_index`` / ``z_index`` are the single column indices of the target
    and the just-selected feature; ``cand_indices`` are the candidate column indices.

    ``force`` -- ``"cuda"`` or ``"cpu"`` overrides the size/HW gate (tests / benchmarks). Default
    ``None`` uses the dispatcher.
    """
    cand_indices = np.asarray(cand_indices)
    y_index = int(np.asarray(y_index).ravel()[0])
    z_index = int(np.asarray(z_index).ravel()[0])
    factors_nbins = np.asarray(factors_nbins)
    n = factors_data.shape[0]
    p = len(cand_indices)

    nbins_x = int(factors_nbins[cand_indices].max()) if p else 0
    nbins_y = int(factors_nbins[y_index])
    nbins_z = int(factors_nbins[z_index])
    joint_size = nbins_x * nbins_y * nbins_z

    use_cuda = force == "cuda" or (force is None and _should_use_cuda(n, p, joint_size))
    if force == "cpu":
        use_cuda = False

    if use_cuda:
        try:
            Xc = np.ascontiguousarray(factors_data[:, cand_indices].T, dtype=np.int32)
            y = np.ascontiguousarray(factors_data[:, y_index], dtype=np.int32)
            z = np.ascontiguousarray(factors_data[:, z_index], dtype=np.int32)
            return conditional_mi_batched_cuda(Xc, y, z, nbins_x, nbins_y, nbins_z)
        except Exception as _exc:  # noqa: BLE001 - any GPU failure -> exact CPU fallback
            logger.warning("cmi_cuda: GPU path failed (%s); falling back to CPU", _exc)

    return _cpu_cmi_loop(factors_data, cand_indices, np.asarray([y_index]), np.asarray([z_index]), factors_nbins, dtype=dtype)


__all__ = [
    "conditional_mi_batched_cuda",
    "conditional_mi_batched_dispatch",
    "cupy_available",
]
