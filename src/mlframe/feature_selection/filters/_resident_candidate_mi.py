"""MANDATE-2 (2026-06-23): resident GPU candidate-GENERATION + MI for the host-numpy FE-gate MI path.

The #1 mlframe CPU-compute kernel in the F2 MRMR fit is ``_plugin_mi_classif_batch_njit`` (cProfile tottime
~2.94s of a 34.7s warm F2 100k fit, 157 calls), reached via ``_orth_mi_backends._mi_classif_batch`` from the
conditional-gate FE (``best_existing_op_mi`` / ``_gate_grid_mi``), the pairwise-modular FE, and the unified
FE gate. Its candidate matrices are built on the HOST with numpy (``u*v``, ``u-v``, ``u/(|v|+eps)``, ``u+v``,
``column_stack``, ``stack.max/min/sum``) -- the candidates were NEVER on the GPU, so unlike the FE-PAIR MI path
(already end-to-end resident) there was no device handoff to extend. This module ports that candidate
GENERATION to cupy and feeds the ALREADY-RESIDENT plug-in MI (``_plugin_mi_classif_batch_cuda_resident`` in
``_hermite_fe_mi``) with NO host round-trip -- the candidate columns are built, binned, and MI-scored entirely
on the device.

PER-OP COVERAGE: every operator ``best_existing_op_mi`` uses has a bit-identical cupy twin --
``u*v`` / ``u-v`` / ``u+v`` / ``u/(|v|+eps)`` (elementwise, IEEE-identical to numpy), and the
``stack.max/min/sum(axis=1)`` row reductions (cupy reductions match numpy to fp64 round-off). There is NO
scipy.special / transcendental op in this path, so the WHOLE candidate set ports; a future op with no
bit-identical GPU twin would stay on the per-op CPU fallback.

GATE: this resident path engages ONLY where a per-host KTC crossover (sibling ``_resident_candidate_mi_ktc``,
keyed on (n, k)) has MEASURED it faster than the host njit batch-MI. On the dev GTX 1050 Ti the F2 calls are
overwhelmingly sub-crossover (k<=18 dominate; the resident GPU MI crossover is ~k>=100 @ n=100k), so the gate
keeps small-k on CPU here (correct) and selects the resident path for large-k / stronger GPUs / larger p.

EQUIVALENCE: the resident MI uses percentile-edge equi-frequency binning (vs the njit rank-based binning),
selection-equivalent (not bit-identical at ties) -- the SAME approved trade the FE-PAIR resident path already
ships (Spearman 1.0, argmax match; MRMR selection-equivalence tests pass). On a no-cupy / CPU-only host the
gate returns ``None`` and the caller stays on the exact njit path -- byte-for-byte unchanged.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


_BUILD_OPCANDS_SRC = r"""
extern "C" __global__
void build_op_cands(const double* __restrict__ M, const int* __restrict__ pi, const int* __restrict__ pj,
                    const long long n, const int m, const int npairs, const int has_sum,
                    const int k, double* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)k;
    if (t >= total) return;
    int c = (int)(t % (long long)k);
    long long row = t / (long long)k;
    const double* Mr = M + row * (long long)m;
    double val;
    if (c < m) {
        val = Mr[c];                                   // raw column
    } else if (c < m + 4 * npairs) {
        int q = c - m; int p = q >> 2; int o = q & 3;
        double u = Mr[pi[p]], v = Mr[pj[p]];
        val = o == 0 ? u * v : (o == 1 ? u - v : (o == 2 ? u / (fabs(v) + 1e-6) : u + v));
    } else {
        int r = c - m - 4 * npairs;
        if (r <= 1) {                                  // row max / min over the m operands
            double acc = Mr[0];
            for (int q = 1; q < m; ++q) { double w = Mr[q]; acc = (r == 0) ? (w > acc ? w : acc) : (w < acc ? w : acc); }
            val = acc;
        } else {                                        // row sum (has_sum)
            double acc = 0.0;
            for (int q = 0; q < m; ++q) acc += Mr[q];
            val = acc;
        }
    }
    out[row * (long long)k + c] = val;
}
"""
_BUILD_OPCANDS_KERNEL = None


def _get_build_opcands_kernel(cp):
    global _BUILD_OPCANDS_KERNEL
    if _BUILD_OPCANDS_KERNEL is None:
        _BUILD_OPCANDS_KERNEL = cp.RawKernel(_BUILD_OPCANDS_SRC, "build_op_cands")
    return _BUILD_OPCANDS_KERNEL


def _build_best_existing_op_candidates_gpu(cols_arr_gpu: list, cp):
    """Build the ``best_existing_op_mi`` candidate columns ON THE DEVICE from resident operand columns.

    Mirrors the host numpy generation in ``_conditional_gate_fe.best_existing_op_mi`` op-for-op (raw columns
    + pairwise product / diff / ratio / sum + row max / min + full row sum when >=3 operands). Returns an
    (n, k) cupy float64 matrix -- NO host transfer. Column ORDER is identical to the host path so the
    per-column MI maps 1:1.

    FUSED (launch-reduction): one op-table RawKernel builds ALL candidate columns (raw + 4 pairwise ops +
    row max/min/sum) in ONE launch instead of ~4*pairs + reductions cupy elementwise ops. Same f64 ops in
    the same column order -> bit-identical; falls back to the cupy build on any kernel error.

    bench-attempt-rejected (2026-06-26): host-stacking the operand columns (np.column_stack + one H2D) to drop
    the device cp.stack launch was selection-IDENTICAL by construction (same f64 matrix) yet flipped
    test_gpu_cpu_mi_selection_equivalence[reg_mixed] -- the host column_stack of the operand views does not
    reproduce the exact device cp.stack byte layout/order this razor-edge case depends on. Kept the device
    cp.stack."""
    m = len(cols_arr_gpu)
    try:
        n = int(cols_arr_gpu[0].shape[0])
        M = cp.ascontiguousarray(cp.stack(cols_arr_gpu, axis=1).astype(cp.float64, copy=False))  # (n, m)
        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
        npairs = len(pairs)
        has_sum = 1 if m >= 3 else 0
        k = m + 4 * npairs + 2 + has_sum
        pi = cp.asarray(np.asarray([p[0] for p in pairs] or [0], dtype=np.int32))
        pj = cp.asarray(np.asarray([p[1] for p in pairs] or [0], dtype=np.int32))
        out = cp.empty((n, k), dtype=cp.float64)
        total = n * k
        threads = 256
        _get_build_opcands_kernel(cp)(((total + threads - 1) // threads,), (threads,),
                                      (M, pi, pj, np.int64(n), np.int32(m), np.int32(npairs),
                                       np.int32(has_sum), np.int32(k), out))
        return out
    except Exception:
        cands = list(cols_arr_gpu)
        for i in range(m):
            for j in range(i + 1, m):
                u, v = cols_arr_gpu[i], cols_arr_gpu[j]
                cands.append(u * v); cands.append(u - v); cands.append(u / (cp.abs(v) + 1e-6)); cands.append(u + v)
        stk = cp.stack(cols_arr_gpu, axis=1)
        cands.append(stk.max(axis=1)); cands.append(stk.min(axis=1))
        if m >= 3:
            cands.append(stk.sum(axis=1))
        return cp.stack(cands, axis=1)


def best_existing_op_mi_resident(
    arrs: dict, names: Sequence[str], yi: np.ndarray, nbins: int,
    *, y_gpu: object = None, y_min: object = None, n_classes: object = None,
    rank_binning: bool = False,
) -> Optional[float]:
    """Resident-GPU twin of ``_conditional_gate_fe.best_existing_op_mi``: build the candidate columns on the
    device + score MI via the resident plug-in kernel, NO host round-trip. Returns the max MI (float), or
    ``None`` if cupy is unavailable / the build fails (caller then takes the exact njit path).

    ``y_gpu`` / ``y_min`` / ``n_classes`` may be passed pre-computed (y is a fit-constant) to skip the
    per-call label H2D + min/max reduction.

    ``rank_binning`` (default False): when True, score the candidates with the RANK (argsort equi-frequency)
    resident kernel instead of the percentile-EDGE one, so the gate's resident baseline MI byte-matches the
    CPU njit rank MI on tied operator outputs (the gate's heavily-tied gate_mask). Returns ``None`` if the
    rank kernel is unavailable so the caller falls back to the edge path / host njit."""
    try:
        import cupy as cp
    except Exception:
        return None
    try:
        from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

        names = list(names)
        # The base operand columns in ``arrs`` are FIT-CONSTANTS re-uploaded on every gate call (H2D
        # instrumentation: this listcomp 80x / 140 MB on a 250k F2 strict fit). Read them from a resident
        # device cache keyed on the parent-array identity + content fingerprint so each is uploaded ONCE per
        # fit (selection-equivalent: same f64 values, just not re-uploaded). The candidate MATRIX built from
        # them below is device-born + transient and is NOT cached.
        from ._fe_resident_operands import resident_operand
        cols_arr_gpu = [resident_operand(arrs[c], ("op", c), dtype=cp.float64) for c in names]
        mat_gpu = _build_best_existing_op_candidates_gpu(cols_arr_gpu, cp)
        if y_gpu is None:
            # y is a fit-constant -> resident cache (instrumentation: 30x re-upload here when y_gpu is None).
            y_gpu = resident_operand(yi, "y", dtype=np.int64)
        if rank_binning:
            from ._gpu_resident_rank_bin import plugin_mi_classif_batch_rank_cuda_resident
            mis = plugin_mi_classif_batch_rank_cuda_resident(
                mat_gpu, y_gpu, nbins, y_min=y_min, n_classes=n_classes,
            )
            if mis is None:
                return None
        else:
            mis = _plugin_mi_classif_batch_cuda_resident(
                mat_gpu, y_gpu, nbins, y_min=y_min, n_classes=n_classes, relax_binning=True,
            )
        return float(np.max(mis))
    except Exception as _exc:  # noqa: BLE001
        logger.debug("best_existing_op_mi_resident: GPU path failed (%s); host fallback", _exc)
        return None


def gate_grid_mi_resident(
    specs: Sequence[tuple], yi: np.ndarray, nbins: int,
    *, rank_binning: bool = False, y_gpu: object = None, y_min: object = None, n_classes: object = None,
) -> Optional[np.ndarray]:
    """Resident-GPU twin of ``_conditional_gate_fe._gate_grid_mi`` that builds the tau-grid candidate columns
    ON THE DEVICE (no host-built matrix uploaded) and scores per-column MI with the resident plug-in kernel.

    DEVICE-BORN TAU-GRID (2026-06-29): the host gate-grid path materialises an ``(n, sum_k)`` float64 matrix on
    the HOST then ``cp.asarray``-uploads it at ``_orth_mi_backends.py:311`` -- the dominant H2D of a GPU-strict
    F2 fit (~2.8 GB / 65% on a 300k fit). Here each combo's tau-grid block is built directly on the device from
    the RESIDENT operand columns (uploaded once per fit via the operand cache), so ONLY the small operand
    columns cross H2D, never the (much larger) candidate matrix.

    ``specs`` is a list of per-combo build descriptors, each one of:
        ("mask",   ctup, (cv_arr, av_arr),        taus)  -> column j = (cv > taus[j]) * av
        ("select", ctup, (cv_arr, av_arr, bv_arr), taus) -> column j = where(cv > taus[j], av, bv)
    where ``cv_arr`` / ``av_arr`` / ``bv_arr`` are HOST float64 operand columns and ``ctup`` is the operand
    column-name tuple ((a, c) for mask = operands (cv=c, av=a); (a, b, c) for select = operands (cv=c, av=a,
    bv=b)) used as the STABLE resident-operand cache key so each recurring column uploads ONCE per fit (not once
    per spec). ``taus`` is a host 1-D float64 array of thresholds. The returned (sum_k,) host MI vector
    concatenates each combo's per-tau MIs in spec order -- IDENTICAL layout to the host ``_gate_grid_mi`` of the
    concatenated blocks, so the caller's per-combo argmax slicing is unchanged.

    ESTIMATOR CONSISTENCY: ``rank_binning`` is threaded through verbatim so the resident batch bins with the
    SAME estimator (RANK vs percentile-EDGE) the per-triple / host path would have used -- no EDGE<->RANK switch
    that could shift selection (the reg_mixed failure mode). Each column is binned INDEPENDENTLY, so the MI is
    per-column bit-identical to the host estimator's per-column MI.

    Returns the host (sum_k,) float64 MI vector, or ``None`` on any cupy failure / no-cupy (caller falls back to
    the exact host ``_gate_grid_mi``)."""
    try:
        import cupy as cp
    except Exception:
        return None
    try:
        from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident
        from ._fe_resident_operands import resident_operand

        if not specs:
            return np.zeros(0, dtype=np.float64)
        blocks = []  # device (n, k_combo) matrices, built op-for-op like the host tau-grid
        for (mode, ctup, cols, taus) in specs:
            taus_g = cp.asarray(np.ascontiguousarray(np.asarray(taus, dtype=np.float64)))  # (k,)
            if mode == "mask":
                cv, av = cols
                a_name, c_name = ctup  # mask ctup = (a, c); operands = (cv=c, av=a)
                cv_g = resident_operand(cv, ("gate_op", c_name), dtype=cp.float64)
                av_g = resident_operand(av, ("gate_op", a_name), dtype=cp.float64)
                # column j = (cv > taus[j]) * av ; broadcast (n,1) > (k,) -> (n,k), * (n,1)
                blk = (cv_g[:, None] > taus_g[None, :]).astype(cp.float64) * av_g[:, None]
            elif mode == "select":
                cv, av, bv = cols
                a_name, b_name, c_name = ctup  # select ctup = (a, b, c); operands = (cv=c, av=a, bv=b)
                cv_g = resident_operand(cv, ("gate_op", c_name), dtype=cp.float64)
                av_g = resident_operand(av, ("gate_op", a_name), dtype=cp.float64)
                bv_g = resident_operand(bv, ("gate_op", b_name), dtype=cp.float64)
                # column j = where(cv > taus[j], av, bv)
                mask = cv_g[:, None] > taus_g[None, :]
                blk = cp.where(mask, av_g[:, None], bv_g[:, None])
            else:
                return None  # unknown mode -> host fallback
            blocks.append(cp.ascontiguousarray(blk.astype(cp.float64, copy=False)))
        mat_gpu = cp.ascontiguousarray(cp.concatenate(blocks, axis=1)) if len(blocks) > 1 else blocks[0]
        if y_gpu is None:
            y_gpu = resident_operand(yi, "y", dtype=np.int64)
        if rank_binning:
            from ._gpu_resident_rank_bin import plugin_mi_classif_batch_rank_cuda_resident
            mis = plugin_mi_classif_batch_rank_cuda_resident(
                mat_gpu, y_gpu, nbins, y_min=y_min, n_classes=n_classes,
            )
            if mis is None:
                return None
        else:
            mis = _plugin_mi_classif_batch_cuda_resident(
                mat_gpu, y_gpu, nbins, y_min=y_min, n_classes=n_classes, relax_binning=True,
            )
        return np.asarray(mis, dtype=np.float64)
    except Exception as _exc:  # noqa: BLE001
        logger.debug("gate_grid_mi_resident: GPU path failed (%s); host fallback", _exc)
        return None
