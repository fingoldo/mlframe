"""Fleuret conditional recheck of a WINDOW of screen candidates in one device launch.

``confirm_candidate`` (``_confirm_predictor``) rechecks every candidate that passed the marginal bootstrap with ``get_fleuret_criteria_confidence`` / ``fleuret_confidence_prange`` (or, with several
workers and a larger budget, the joblib twin ``get_fleuret_criteria_confidence_parallel``): for each permutation ``p`` it shuffles the ``y`` column (and the ``x`` columns with ``extra_x_shuffling``)
by an inline LCG Fisher-Yates seeded from ``(base_seed, p)`` alone, takes, for every conditioning set ``Z`` (each combination of the selected variables up to the interactions order), the
conditional MI ``I(X; Y | Z) = H(XZ) + H(YZ) - H(Z) - H(XYZ)`` of the shuffled columns (floored at zero), raises it to ``nexisting + 1`` and keeps the minimum over ``Z``; the permutation "fails" when that
minimum reaches the candidate's gain, and the recheck stops at ``max_failed`` failures. At 100000 rows that was 178 calls of 25 ms of thread time (4.45 s) in the 100000 x 100
all-families fit, the largest piece of ``screen_predictors``; each call runs on its own host thread with about 3 permutations, which leaves the cores idle between short blocks.

Every permutation starts from the ORIGINAL columns with its own seed, so the ``candidates x permutations`` shuffles are independent: the host draws them with the same LCG
(``_fleuret_shuffle_col_lcg``, njit prange over the jobs), and ONE launch of ``fleuret_cmi_blocks`` (a block per ``(job, Z)``: one joint table over the rows in shared memory, the marginals as sums of
its cells, the entropy sums by a fixed-order tree) returns nine sufficient statistics of every ``(job, Z)``: ``sum p ln p`` of the XYZ, XZ, YZ, Z and Y tables and the occupied-cell counts of the first
four. Every statistic the host criterion can take is a function of those nine numbers, so the same launch serves all of them: the plug-in conditional MI, its Miller-Madow variant (the bias
``(k_xyz + k_z - k_xz - k_yz) / (2 n)``), the conditional symmetric uncertainty (Miller-Madow entropies) and the JMIM joint MI ``I({X, Z}; Y) = H(XZ) + H(Y) - H(XYZ)``. The failure counts are then read off in
permutation order exactly as the host kernels do (per worker segment for the joblib route), so ``(nfailed, nchecked)`` equals the host value.

Scope (anything else returns ``None`` and the host kernel runs): one ``y`` column, the ``fleuret`` relevance and redundancy, a joint XYZ table within :data:`MAX_SHARED_CELLS`, codes inside their
declared bin counts. The host sums the entropies in numpy order and the kernel in a tree: the statistics differ by about 1e-16, which moves a verdict only for a tie of the two floats.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Optional, Sequence

import numpy as np
from numba import njit, prange

from ._internals import LARGE_CONST, MAX_CONFIRMATION_CAND_NBINS, NMAX_NONPARALLEL_ITERS
from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback
from ._numba_utils import _fleuret_shuffle_col_lcg
from ._raw_module_cache import RawModuleCache

logger = logging.getLogger(__name__)

MAX_SHARED_CELLS = 9_000  # int32 cells of the joint table of one block (36 KB; the 10 KB of reduction scratch and the launch share the 48 KB a block may hold)
MAX_Z_SETS = 64  # conditioning sets of one launch (the grid's second axis)
N_STATS = 9
_THREADS = 256
_STATISTICS = ("cmi", "mm", "su", "jmim")

# One block per (job, Z). The rows are walked coalesced (consecutive threads, consecutive rows of the (jobs, n) shuffled x / y codes and of the (n_z, n) Z codes). ONE joint table xyz (nx ny nz) is filled
# by integer shared atomics, one atomic per row (the first version filled xyz, xz, yz and z: four atomics a row into tables of ten to a hundred cells, nvprof: shared-memory efficiency 0.27 percent,
# 465 load and 563 store transactions per request, 436 us a launch; one atomic: 69 and 125 transactions, 329 us); the marginals xz, yz, z and y are SUMS over the cells of xyz, which are the same integers.
# The sums are accumulated per thread over strided cells and reduced by a fixed tree, so a launch gives the same value every time.
# out[(job * n_z + zi) * 9 + k]: S_xyz, S_xz, S_yz, S_z, S_y (S = sum p ln p over occupied cells, p = count / n), then the occupied-cell counts k_xyz, k_xz, k_yz, k_z.
_SRC = render(
    r"""
__device__ __forceinline__ void tree_reduce({ACC} (*red)[256], const int nvals) {
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) for (int v = 0; v < nvals; ++v) red[v][threadIdx.x] += red[v][threadIdx.x + s];
        __syncthreads();
    }
}

extern "C" __global__ void fleuret_cmi_blocks(const {CODE}* __restrict__ xs, const {CODE}* __restrict__ ys, const {CODE}* __restrict__ zs, const int* __restrict__ nbx,
                                              const int* __restrict__ nbz, const int ny, const long long n, const int perms_per_cand, const int n_z, {ACC}* __restrict__ out) {
    extern __shared__ int sh[];
    __shared__ {ACC} red[5][256];
    const int job = blockIdx.x, zi = blockIdx.y;
    const int nx = nbx[job / perms_per_cand], nz = nbz[zi];
    const int cells = nx * ny * nz;
    for (int c = threadIdx.x; c < cells; c += blockDim.x) sh[c] = 0;
    __syncthreads();
    const {CODE}* xj = xs + (long long)job * n;
    const {CODE}* yj = ys + (long long)job * n;
    const {CODE}* zc = zs + (long long)zi * n;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) atomicAdd(&sh[((int)xj[i] * ny + (int)yj[i]) * nz + (int)zc[i]], 1);
    __syncthreads();
    {ACC} s_xyz = 0.0, s_xz = 0.0, s_yz = 0.0, s_z = 0.0, s_y = 0.0;
    int k_xyz = 0, k_xz = 0, k_yz = 0, k_z = 0;
    const {ACC} inv_n = 1.0 / (({ACC})n);
    for (int c = threadIdx.x; c < cells; c += blockDim.x) {
        const int cnt = sh[c];
        if (cnt > 0) { const {ACC} p = ({ACC})cnt * inv_n; s_xyz += p * log(p); ++k_xyz; }
    }
    for (int c = threadIdx.x; c < nx * nz; c += blockDim.x) {
        const int x = c / nz, z = c - x * nz;
        int cnt = 0;
        for (int y = 0; y < ny; ++y) cnt += sh[(x * ny + y) * nz + z];
        if (cnt > 0) { const {ACC} p = ({ACC})cnt * inv_n; s_xz += p * log(p); ++k_xz; }
    }
    for (int c = threadIdx.x; c < ny * nz; c += blockDim.x) {
        const int y = c / nz, z = c - y * nz;
        int cnt = 0;
        for (int x = 0; x < nx; ++x) cnt += sh[(x * ny + y) * nz + z];
        if (cnt > 0) { const {ACC} p = ({ACC})cnt * inv_n; s_yz += p * log(p); ++k_yz; }
    }
    for (int z = threadIdx.x; z < nz; z += blockDim.x) {
        int cnt = 0;
        for (int xy = 0; xy < nx * ny; ++xy) cnt += sh[xy * nz + z];
        if (cnt > 0) { const {ACC} p = ({ACC})cnt * inv_n; s_z += p * log(p); ++k_z; }
    }
    for (int y = threadIdx.x; y < ny; y += blockDim.x) {
        int cnt = 0;
        for (int xz = 0; xz < nx * nz; ++xz) { const int x = xz / nz, z = xz - x * nz; cnt += sh[(x * ny + y) * nz + z]; }
        if (cnt > 0) { const {ACC} p = ({ACC})cnt * inv_n; s_y += p * log(p); }
    }
    red[0][threadIdx.x] = s_xyz; red[1][threadIdx.x] = s_xz; red[2][threadIdx.x] = s_yz; red[3][threadIdx.x] = s_z; red[4][threadIdx.x] = s_y;
    __syncthreads();
    tree_reduce(red, 5);
    {ACC}* o = out + ((long long)job * n_z + zi) * 9;
    if (threadIdx.x == 0) { o[0] = red[0][0]; o[1] = red[1][0]; o[2] = red[2][0]; o[3] = red[3][0]; o[4] = red[4][0]; }
    __syncthreads();
    red[0][threadIdx.x] = ({ACC})k_xyz; red[1][threadIdx.x] = ({ACC})k_xz; red[2][threadIdx.x] = ({ACC})k_yz; red[3][threadIdx.x] = ({ACC})k_z;
    __syncthreads();
    tree_reduce(red, 4);
    if (threadIdx.x == 0) { o[5] = red[0][0]; o[6] = red[1][0]; o[7] = red[2][0]; o[8] = red[3][0]; }
}
""",
).replace("{CODE}", "unsigned char")
_MODULES = {
    "unsigned char": RawModuleCache(_SRC),
    "short": RawModuleCache(_SRC.replace("unsigned char", "short")),
    "int": RawModuleCache(_SRC.replace("unsigned char", "int")),
}


def kernel(cp: Any, real: Any) -> Any:
    """``fleuret_cmi_blocks`` for shuffled-column codes of dtype ``real`` (uint8, int16 or int32); compile point of :func:`fleuret_confirm_batch` and of the register-pressure registry."""
    name = {np.dtype(np.uint8): "unsigned char", np.dtype(np.int16): "short"}.get(np.dtype(real), "int")
    return _MODULES[name].get(cp).get_function("fleuret_cmi_blocks")


@njit(parallel=True, nogil=True, cache=True)
def _shuffled_jobs(y_col, x_cols, mult, seeds, n_perm, perm_offset, extra_x_shuffling, out_y, out_x):  # pragma: no cover - numba body
    """Fill ``out_y[c * n_perm + p]`` with the ``y`` column of candidate ``c`` shuffled as permutation ``perm_offset + p`` of ``get_fleuret_criteria_confidence`` does, and ``out_x[...]`` with the joint
    code ``sum_q x_q * mult[c, q]`` of the candidate's ``x`` columns (each shuffled in turn when ``extra_x_shuffling``, the LCG state carried on from the ``y`` shuffle as the host does)."""
    k = x_cols.shape[0]
    kx = x_cols.shape[1]
    n = x_cols.shape[2]
    for j in prange(k * n_perm):
        c = j // n_perm
        p = j % n_perm
        state = np.uint64(seeds[c]) * np.uint64(2654435761) + np.uint64(perm_offset + p + 1)
        out_y[j, :] = y_col
        state = _fleuret_shuffle_col_lcg(out_y[j], state)
        if kx == 1:  # the common case: one column, shuffled in place in its output row (its joint code is the column itself)
            out_x[j, :] = x_cols[c, 0]
            if extra_x_shuffling:
                state = _fleuret_shuffle_col_lcg(out_x[j], state)
        else:
            tmp = x_cols[c].copy()
            if extra_x_shuffling:
                for q in range(kx):
                    state = _fleuret_shuffle_col_lcg(tmp[q], state)
            for i in range(n):
                code = np.int64(0)
                for q in range(kx):
                    code += np.int64(tmp[q, i]) * mult[c, q]
                out_x[j, i] = code


def _code_dtype(max_code: int) -> Any:
    """The narrowest of uint8 / int16 / int32 that holds codes up to ``max_code``."""
    return np.uint8 if max_code < 256 else (np.int16 if max_code < 32767 else np.int32)


def _joint_code(block: np.ndarray, pos: dict, nb: np.ndarray, cols: Sequence[int], dt: Any) -> np.ndarray:
    """The joint code of the columns ``cols`` of ``block`` (mixed radix over their bin counts), in the dtype ``dt``."""
    code = np.zeros(block.shape[1], dtype=np.int64)
    mult = 1
    for c in cols:
        code += block[pos[c]].astype(np.int64) * mult
        mult *= int(nb[c])
    return code.astype(dt)


def _counts_from_gains(job_gain: np.ndarray, n_cand: int, n_perm: int, gains: np.ndarray, max_failed: int, segments: Sequence[tuple]) -> list:
    """``(nfailed, nchecked)`` per candidate, the host kernels' rule: permutations in order within each ``(start, length)`` segment (one segment for the serial kernels, one per worker for the joblib
    route), a failure when the job's gain reaches the candidate's gain, a stop at ``max_failed`` failures within the segment; the counts of the segments are summed."""
    out = []
    for c in range(n_cand):
        nfailed_total = nchecked_total = 0
        for start, length in segments:
            nfailed, nchecked = 0, length
            for p in range(start, start + length):
                if job_gain[c * n_perm + p] >= gains[c]:
                    nfailed += 1
                    if nfailed >= max_failed:
                        nchecked = p - start + 1
                        break
            nfailed_total += nfailed
            nchecked_total += nchecked
        out.append((nfailed_total, nchecked_total))
    return out


def _h_mm(s: np.ndarray, k: np.ndarray, n: int) -> np.ndarray:
    """Miller-Madow entropy from ``s = sum p ln p`` and the occupied-cell count ``k``: the plug-in entropy plus ``(k - 1) / (2 n)`` for more than one occupied cell."""
    return np.where(k > 1, -s + (k - 1) / (2.0 * n), -s)


def _term(stats: np.ndarray, n: int, statistic: str, nexisting: np.ndarray, jmim_discount_only: bool) -> np.ndarray:
    """The per-``(job, Z)`` additional-knowledge term of ``evaluate_gain`` from the nine statistics ``stats[..., 9]``: plug-in conditional MI floored at zero, its Miller-Madow variant,
    the conditional symmetric uncertainty or the JMIM joint MI, then the ``nexisting + 1`` power (JMIM: unless ``jmim_discount_only`` and the power would raise it)."""
    s_xyz, s_xz, s_yz, s_z, s_y = (stats[..., i] for i in range(5))
    k_xyz, k_xz, k_yz, k_z = (stats[..., i] for i in range(5, 9))
    ne = nexisting[:, None]
    if statistic == "jmim":
        ak = s_xyz - s_xz - s_y
        disc = np.where(ne > 0, ak ** (ne + 1), ak)
        return np.where(bool(jmim_discount_only) & (disc > ak) & (ne > 0), ak, disc)
    if statistic == "su":
        h_xz, h_yz, h_z, h_xyz = _h_mm(s_xz, k_xz, n), _h_mm(s_yz, k_yz, n), _h_mm(s_z, k_z, n), _h_mm(s_xyz, k_xyz, n)
        cmi = h_xz + h_yz - h_z - h_xyz
        denom = h_xz + h_yz - 2.0 * h_z
        ak = np.where((denom <= 1e-12) | (cmi < 0.0), 0.0, np.minimum(2.0 * np.maximum(cmi, 0.0) / np.where(denom > 1e-12, denom, 1.0), 1.0))
    else:
        ak = np.maximum(-s_xz - s_yz + s_z + s_xyz, 0.0)
        if statistic == "mm":
            ak = np.maximum(ak - (k_xyz + k_z - k_xz - k_yz) / (2.0 * n), 0.0)
    return np.where(ne > 0, ak ** (ne + 1), ak)


def fleuret_confirm_batch(
    data_copy: np.ndarray,
    factors_nbins: np.ndarray,
    x_sets: Sequence[Sequence[int]],
    y_var: int,
    selected_vars: Sequence[int],
    npermutations: int,
    bootstrapped_gains: Sequence[float],
    max_failed: int,
    nexisting: Sequence[int],
    extra_x_shuffling: bool,
    base_seeds: Sequence[int],
    max_confirmation_cand_nbins: int = MAX_CONFIRMATION_CAND_NBINS,
    perm_offset: int = 0,
    max_order: int = 1,
    statistic: str = "cmi",
    jmim_discount_only: bool = False,
    worker_loads: Optional[Sequence[int]] = None,
) -> Optional[list]:
    """``[(nfailed, nchecked), ...]`` for the candidates ``x_sets`` (tuples of one or more columns, all of the same length) against the target column ``y_var`` given ``selected_vars`` (their
    combinations up to ``max_order`` as conditioning sets), each equal to what ``get_fleuret_criteria_confidence`` returns for that candidate (its own ``bootstrapped_gain``, ``nexisting`` and seed),
    computed in one launch.

    ``statistic`` is ``cmi`` (the default criterion), ``mm`` (Miller-Madow), ``su`` or ``jmim``; ``worker_loads`` (the permutation split of the joblib route) makes the stop at ``max_failed`` apply per worker
    segment as ``get_fleuret_criteria_confidence_parallel`` does. ``None`` when the device path does not apply (no card, no selected variable, a joint table over :data:`MAX_SHARED_CELLS`, too many
    conditioning sets, a code outside its bin count, candidates of different lengths, an unknown statistic) or faulted: the caller then runs the host kernel per candidate."""
    n_cand, n_perm = len(x_sets), int(npermutations)
    if n_cand == 0 or n_perm < 1 or len(selected_vars) == 0 or statistic not in _STATISTICS or int(max_order) < 1 or len({len(xs) for xs in x_sets}) != 1:
        return None
    try:
        import cupy as cp
        from pyutilz.performance.gpu_hw_launch import block_size_for

        nb = np.asarray(factors_nbins)
        ny = int(nb[y_var])
        sel = [int(z) for z in selected_vars]
        z_sets = [c for o in range(1, min(int(max_order), len(sel)) + 1) for c in itertools.combinations(sel, o)]
        if len(z_sets) > MAX_Z_SETS:
            return None
        # a conditioning set whose summed bin counts exceed the confirmation cap contributes the constant 0.0 to the minimum (as ``evaluate_gain`` does); the others are scored on the device
        scored = [i for i, zs in enumerate(z_sets) if int(sum(int(nb[c]) for c in zs)) <= int(max_confirmation_cand_nbins)]
        n = int(data_copy.shape[0])
        if n < 2:
            return None
        nx_tot = np.asarray([int(np.prod([int(nb[int(c)]) for c in xs])) for xs in x_sets], dtype=np.int64)
        nz_tot = np.asarray([int(np.prod([int(nb[c]) for c in z_sets[i]])) for i in scored], dtype=np.int64)
        if scored and int(nx_tot.max() * ny * nz_tot.max()) > MAX_SHARED_CELLS:
            return None
        # one gather of every column the launch reads into a contiguous (columns, n) block; the code ranges are checked on it with two reductions, and int8 codes are read as uint8 without a copy
        cols_needed = sorted({int(c) for xs in x_sets for c in xs} | {int(y_var)} | {c for i in scored for c in z_sets[i]})
        pos = {c: i for i, c in enumerate(cols_needed)}
        block = np.ascontiguousarray(np.asarray(data_copy)[:, cols_needed].T)
        limits = np.asarray([int(nb[c]) for c in cols_needed])
        if bool((block.max(axis=1) >= limits).any()) or bool((block.min(axis=1) < 0).any()):
            return None  # a code outside its own declared bin count: the host kernel owns that error
        dt = _code_dtype(int(max([ny, int(nx_tot.max())] + ([int(nz_tot.max())] if scored else []))))
        block = block.view(np.uint8) if (block.dtype == np.int8 and dt is np.uint8) else block.astype(dt, copy=False)
        kx = len(x_sets[0])
        x_cols = np.ascontiguousarray(np.stack([block[[pos[int(c)] for c in xs]] for xs in x_sets]))
        mult = np.ones((n_cand, kx), dtype=np.int64)
        for ci, xs in enumerate(x_sets):
            for q in range(1, kx):
                mult[ci, q] = mult[ci, q - 1] * int(nb[int(xs[q - 1])])
        y_col = block[pos[int(y_var)]]
        out_y = np.empty((n_cand * n_perm, n), dtype=dt)
        out_x = np.empty((n_cand * n_perm, n), dtype=dt)
        _shuffled_jobs(y_col, x_cols, mult, np.asarray(base_seeds, dtype=np.uint64), n_perm, int(perm_offset), bool(extra_x_shuffling), out_y, out_x)
        n_jobs = n_cand * n_perm
        gains_arr = np.asarray(bootstrapped_gains, dtype=np.float64)  # Intentional float64: compared with the statistics
        ne = np.repeat(np.asarray(nexisting, dtype=np.int64), n_perm)
        gains = np.full(n_jobs, LARGE_CONST, dtype=np.float64)  # Intentional float64: same start value and arithmetic as evaluate_gain
        if scored:
            from ._fe_resident_operands import resident_operand

            z_code = np.stack([block[pos[z_sets[i][0]]] if len(z_sets[i]) == 1 else _joint_code(block, pos, nb, z_sets[i], dt) for i in scored])
            z_dev = resident_operand(z_code, ("fleuret_z",), dtype=dt)  # the conditioning codes stay on the card while the selection is the same
            nb_dev = cp.asarray(np.concatenate([nx_tot, nz_tot]).astype(np.int32))
            kern = kernel(cp, dt)
            cells = int(nx_tot.max() * ny * nz_tot.max())
            threads = min(_THREADS, block_size_for(kern, dyn_smem=4 * cells, work_items=n))
            threads = 1 << (int(threads).bit_length() - 1)  # a power of two for the reduction tree
            d_out = cp.empty((n_jobs, len(scored), N_STATS), dtype=cp.float64)
            kern(
                (n_jobs, len(scored)),
                (threads,),
                (cp.asarray(out_x), cp.asarray(out_y), z_dev, nb_dev[:n_cand], nb_dev[n_cand:], np.int32(ny), np.int64(n), np.int32(n_perm), np.int32(len(scored)), d_out),
                shared_mem=4 * cells,
            )  # launch-ok: one launch scores every (candidate x permutation, conditioning set) pair
            terms = _term(cp.asnumpy(d_out), n, statistic, ne, jmim_discount_only)
            gains = np.minimum(gains, terms.min(axis=1))
        if len(scored) < len(z_sets):
            gains = np.minimum(gains, 0.0)  # an over-cap conditioning set contributes 0.0
        segments = [(0, n_perm)]
        if worker_loads is not None:
            offs = np.concatenate(([0], np.cumsum(worker_loads)[:-1]))
            segments = [(int(o), int(w)) for o, w in zip(offs, worker_loads)]
        return _counts_from_gains(gains, n_cand, n_perm, gains_arr, int(max_failed), segments)
    except Exception as exc:
        logger.debug("device Fleuret confirmation failed (%s: %s); host kernel", type(exc).__name__, exc)
        return None


DEVICE_FALLBACK_ROWS = 20_000  # the row mark used until a sweep has run on this host
_SWEEP_ROWS = [5_000, 20_000, 100_000, 400_000]
_SWEEP_COLS = 12
_SWEEP_NBINS = 10
_SWEEP_WINDOW = 4
_SWEEP_PERMS = 3
_SWEEP_SELECTED = 5
_SALT = 2


def _sweep_inputs(dims: dict) -> tuple:
    """``(data, nbins, candidate columns, target column, selected columns)``: a code frame of ``dims["n_rows"]`` rows whose target depends on the first two columns."""
    rng = np.random.default_rng(0)
    n = int(dims["n_rows"])
    cols = [rng.integers(0, _SWEEP_NBINS, n) for _ in range(_SWEEP_COLS)]
    y = (cols[0] + cols[1] + rng.integers(0, 2, n)) % 3
    data = np.column_stack([*cols, y]).astype(np.int8)
    nbins = np.asarray([_SWEEP_NBINS] * _SWEEP_COLS + [3], dtype=np.int64)
    return (data, nbins, list(range(_SWEEP_SELECTED, _SWEEP_SELECTED + _SWEEP_WINDOW)), _SWEEP_COLS, list(range(_SWEEP_SELECTED)))


def _sweep_host(data: np.ndarray, nbins: np.ndarray, xs: list, y_col: int, selected: list) -> np.ndarray:
    """The host kernel per candidate (the production serial kernel on the compact inputs)."""
    from ._confirm_predictor_helpers import _compact_confidence_inputs
    from ._numba_utils import get_fleuret_criteria_confidence

    out = []
    for i, x in enumerate(xs):
        cd, cnb, cx, cy, csel, cent = _compact_confidence_inputs(data, nbins, (x,), (y_col,), selected, None)
        out.append(
            get_fleuret_criteria_confidence(
                data_copy=cd, factors_nbins=cnb, x=cx, y=cy, selected_vars=csel, bootstrapped_gain=0.001, npermutations=_SWEEP_PERMS, max_failed=3, nexisting=1,
                mrmr_relevance_algo="fleuret", mrmr_redundancy_algo="fleuret", max_veteranes_interactions_order=1, cached_cond_MIs=cent[1], entropy_cache=cent[0],
                extra_x_shuffling=True, base_seed=np.uint64(1000 + i),
            )
        )
    return np.asarray(out, dtype=np.float64)  # Intentional float64: counts compared exactly


def _sweep_device(data: np.ndarray, nbins: np.ndarray, xs: list, y_col: int, selected: list) -> np.ndarray:
    """The device window."""
    got = fleuret_confirm_batch(data, nbins, [(x,) for x in xs], y_col, selected, _SWEEP_PERMS, [0.001] * len(xs), 3, [1] * len(xs), True, [1000 + i for i in range(len(xs))])
    if got is None:
        raise RuntimeError("device Fleuret confirmation declined")
    return np.asarray(got, dtype=np.float64)  # Intentional float64: counts compared exactly


_fallback_choice = rows_fallback(__name__)

try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="fleuret_confirm_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=0.0, atol=0.0),
        axes={"n_rows": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fleuret_confirm_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the row-count fallback decides
    logger.warning("kernel_tuner spec build for fleuret_confirm_backend failed: %s", e)
    _SPEC = None


def fleuret_confirm_on_device(n_rows: int) -> bool:
    """Whether a window of candidates of ``n_rows`` rows is rechecked on the device: the per-host kernel-tuning cache, else the row mark."""
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_rows=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("fleuret_confirm_backend: KTC choose() failed, using the row-count fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


def fleuret_base_seed(random_seed: Any, n_selected: int, X: Any) -> int:
    """The seed of the Fleuret recheck of candidate ``X`` when ``n_selected`` variables are selected: the candidate's identity folded into ``random_seed`` (one definition for the host and the device path)."""
    return int(((int(random_seed or 0) * 2654435761) + n_selected + 1 + hash(X)) & 0xFFFFFFFFFFFFFFFF)


def _window_store(ctx: Any) -> dict:
    """The per-screen dict of precomputed rechecks, ``{(candidate, selected tuple): (nfailed, nchecked)}``, created on first use."""
    return ctx.__dict__.setdefault("fleuret_batch", {})


def take_precomputed_recheck(ctx: Any, X: tuple) -> Optional[tuple]:
    """The device counts of candidate ``X`` for the current selected variables, removed from the store, or ``None``."""
    store = ctx.__dict__.get("fleuret_batch")
    return store.pop((X, tuple(ctx.selected_vars)), None) if store else None


def _settings_supported(ctx: Any) -> bool:
    """Whether the screen runs the ``fleuret`` criterion on one target column with a recheck to do (so a device recheck can stand in for the host kernels)."""
    if ctx.use_simple_mode or not ctx.full_npermutations or not ctx.selected_vars or ctx.data_copy is None or len(ctx.y) != 1:
        return False
    return (ctx.mrmr_relevance_algo, ctx.mrmr_redundancy_algo) == ("fleuret", "fleuret")


def _nexisting(X: tuple, sel: tuple) -> int:
    """How many columns of ``X`` are already selected (the exponent base of the redundancy term)."""
    return int(sum(1 for c in X if c in sel))


def precompute_window(ctx: Any, candidates: Sequence[tuple], gains: Sequence[float]) -> None:
    """Recheck the ``candidates`` of the prefetch window in one device launch per column count and store their counts for :func:`take_precomputed_recheck`.

    Does nothing (the host kernel then runs per candidate) when the settings are not the Fleuret criterion, when the tuner prefers the host for this row count, when fewer than two candidates need it,
    or when the device declines. The window's results are speculative in the way the prefetch itself is: a candidate the walk never reaches, or one the marginal bootstrap rejects, costs a share of one launch.
    The prefetch threads run the serial kernels with the default statistic, so this is the plug-in criterion; the joblib route is :func:`device_parallel_recheck`."""
    if not _settings_supported(ctx):
        return
    from ._numba_utils import count_cand_nbins

    if ctx.n_workers and ctx.n_workers > 1 and ctx.full_npermutations > NMAX_NONPARALLEL_ITERS:
        return
    if not fleuret_confirm_on_device(int(ctx.data_copy.shape[0])):
        return
    sel = tuple(ctx.selected_vars)
    store = _window_store(ctx)
    for key in [k for k in store if k[1] != sel]:
        del store[key]  # rechecks of an earlier selection are stale
    todo = [(x, g) for x, g in zip(candidates, gains) if (x, sel) not in store and count_cand_nbins(x, ctx.factors_nbins) <= ctx.max_confirmation_cand_nbins]
    for kx in sorted({len(x) for x, _ in todo}):
        group = [(x, g) for x, g in todo if len(x) == kx]
        if len(group) < 2:
            continue
        got = fleuret_confirm_batch(
            ctx.data_copy, ctx.factors_nbins, [tuple(int(c) for c in x) for x, _ in group], int(ctx.y[0]), list(sel), int(ctx.full_npermutations), [float(g) for _, g in group],
            int(ctx.max_failed), [_nexisting(x, sel) for x, _ in group], bool(ctx.extra_x_shuffling), [fleuret_base_seed(ctx.random_seed, len(sel), x) for x, _ in group],
            max_order=int(ctx.max_veteranes_interactions_order),
        )
        if got is not None:
            for (x, _), counts in zip(group, got):
                store[(x, sel)] = counts


def device_parallel_recheck(ctx: Any, X: tuple, next_best_gain: float, base_seed: int, statistic: str, jmim_discount_only: bool) -> Optional[tuple]:
    """``(nfailed, nchecked)`` of the joblib route (``get_fleuret_criteria_confidence_parallel``) for candidate ``X`` from one device launch, with the permutations split over the workers exactly as
    that route splits them (the stop at ``max_failed`` applies per worker), or ``None`` for the host route. ``statistic`` is the scoring toggle the route forwards (``cmi``, ``mm``, ``su``, ``jmim``)."""
    if not _settings_supported(ctx) or not fleuret_confirm_on_device(int(ctx.data_copy.shape[0])):
        return None
    from ._numba_utils import distribute_permutations

    sel = tuple(ctx.selected_vars)
    loads = [int(w) for w in distribute_permutations(npermutations=int(ctx.full_npermutations), n_workers=int(ctx.n_workers))]
    got = fleuret_confirm_batch(
        ctx.data_copy, ctx.factors_nbins, [tuple(int(c) for c in X)], int(ctx.y[0]), list(sel), int(ctx.full_npermutations), [float(next_best_gain)], int(ctx.max_failed),
        [_nexisting(X, sel)], bool(ctx.extra_x_shuffling), [int(base_seed)], max_order=int(ctx.max_veteranes_interactions_order), statistic=statistic,
        jmim_discount_only=bool(jmim_discount_only), worker_loads=loads,
    )
    return got[0] if got is not None else None


__all__ = [
    "fleuret_confirm_batch", "fleuret_confirm_on_device", "fleuret_base_seed", "precompute_window", "take_precomputed_recheck", "device_parallel_recheck", "kernel", "MAX_SHARED_CELLS",
]
