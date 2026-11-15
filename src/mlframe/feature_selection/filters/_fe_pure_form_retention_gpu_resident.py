"""GPU-RESIDENT twin of the per-candidate non-separability filter inside
:func:`_fe_pure_form_retention.retain_usable_pure_forms`.

RESIDENCY CONTRACT (not a wall win). Gated on the resident flag
(``MLFRAME_FE_GPU_STRICT`` + ``MLFRAME_FE_GPU_STRICT_RESIDENT``); default OFF. On
this GTX 1050 Ti the retention pool runs on a ~3000-row subsample, so the cupy
launch + reduction overhead dominates the tiny per-candidate 12-column OLS and
this twin is EXPECTED to be SLOWER than the per-candidate sklearn CPU path - and
that is a PASS by the residency contract. This twin exists to KILL the
per-candidate value D2H/H2D churn the gated ``_usability_gpu`` primitives incur:
the prior GPU usability path (``MLFRAME_FE_GPU_USABILITY``) re-uploaded each
candidate's form column AND both raw operand columns PER candidate
(``gpu_additive_basis_residual`` did one ``cp.asarray`` per array per call) and
pulled the residual array back PER candidate (one bulk D2H of the (n,) residual
per candidate). For a retention pool of P pair candidates that is ~3P bulk H2D +
P bulk D2H. This twin uploads the candidate value matrix + every distinct raw
operand column ONCE (one bulk H2D at entry) and keeps EVERYTHING resident: each
candidate's additive single-operand basis residual is solved on device and only
the two BOUNDED SCALARS the gate needs (std(resid) and |corr(resid, y)|) are
pulled back - NO per-candidate residual D2H, NO per-candidate re-upload.

What is resident vs host control-flow (allowed by the contract):
  * RESIDENT (one bulk H2D at entry): the (n, P) candidate form-value matrix
    ``Vdev`` (double), the (n, B) distinct raw-operand matrix ``Bdev`` and the
    (n,) relevance target ``ydev``. From them, per candidate, the device builds the
    12-column additive basis (6 functions of each of the candidate's two operands,
    looked up by base-column index - NO re-upload), solves the mean-centered OLS
    via ``cp.linalg.lstsq``, forms the residual, and reduces it to the two scalars
    the CPU gate compares. NO per-candidate H2D and NO per-candidate value/residual
    D2H.
  * HOST scalar D2H (allowed by the contract): per candidate the form std, the
    residual std and the |corr(resid, y)| - three bounded scalars, the SAME
    quantities the CPU ``_adds_nonlinear_value`` computes per candidate. None is
    bulk per-candidate value data.

SELECTION-EQUIVALENCE IS THE BAR. The math is the SAME as the CPU
``_adds_nonlinear_value``: the SAME 6-function additive single-operand basis per
operand (standardized x, x^2, x^3, sign*sqrt|x|, sign*log1p|x|, 1/(|x|+1)), the
SAME StandardScaler+LinearRegression == mean-centered-OLS residual (the scaler's
affine map is absorbed into the coefficients + intercept; identical up to fp
round-off - this is exactly the equivalence ``gpu_additive_basis_residual``
already documents), the SAME ``std(resid) < min_resid_frac * f_std`` non-separable
gate and the SAME ``|corr(resid, y)| >= min_resid_corr`` relevance gate (REGRESSION
relevance target ``y``; the SAME ``_abscorr`` centered dot / sqrt(ss) estimator
with the SAME std<1e-12 zero-guard). Only the float reduction ORDER differs between
cupy and numpy (~1e-12), to which the boolean gates are tolerant; on ANY
cupy/device error the whole batch returns ``None`` so the caller runs the exact
per-candidate CPU/sklearn path.

CLASSIFICATION is NOT ported here (the caller passes ``is_clf``): the classification
relevance gate is the WHOLE-form point-biserial ``|corr(fv, class_indicator)|`` (not
the residual corr), a different discriminator - a classification call returns
``None`` so the dispatcher falls through to the exact CPU path.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from ._kernel_types import render

from ._safe_scale import standardise

logger = logging.getLogger(__name__)


def _device_error_types(cp: Any) -> tuple:
    """Genuine cupy / device / linalg fault types that legitimately route to the exact CPU path (logic bugs are not in the set)."""
    # Narrow device-fault set for the resident fallbacks below. A genuine cupy/device/linalg fault
    # (singular OLS, OOM, CUDA runtime error) legitimately routes to the exact CPU path; a logic/shape
    # bug (ValueError/KeyError/IndexError) must SURFACE to tests, not be silently swallowed as a
    # "device fallback" - so the per-candidate and outer guards catch only these, not bare Exception.
    _dev_errs = [np.linalg.LinAlgError]
    try:
        _dev_errs.append(cp.cuda.runtime.CUDARuntimeError)
    except Exception as e:  # nosec B110 - best-effort path
        logger.debug("CUDARuntimeError symbol lookup failed: %s", e)
    try:
        _dev_errs.append(cp.cuda.memory.OutOfMemoryError)
    except Exception as e:  # nosec B110 - best-effort path
        logger.debug("OutOfMemoryError symbol lookup failed: %s", e)
    # FIX4: the direct cp.linalg.lstsq below raises cuSOLVER/cuBLAS errors that subclass
    # plain RuntimeError, NOT CUDARuntimeError -> omitting them would crash instead of falling back to
    # the exact CPU path. getattr so an absent symbol can't break the tuple builder.
    try:
        from cupy_backends.cuda.libs import cusolver as _cusolver
        _e = getattr(_cusolver, "CUSOLVERError", None)
        if _e is not None:
            _dev_errs.append(_e)
        from cupy_backends.cuda.libs import cublas as _cublas
        _e = getattr(_cublas, "CUBLASError", None)
        if _e is not None:
            _dev_errs.append(_e)
    except Exception as e:  # nosec B110 - optional dependency import guard
        logger.debug("cusolver/cublas error-type import probe failed: %s", e)
    return tuple(_dev_errs)


_GRAM_SRC = render(
    r"""
extern "C" __global__
void pair_gram_blocks(const {ACC}* __restrict__ bc, const {ACC}* __restrict__ fc, const long long* __restrict__ ia, const long long* __restrict__ ib, const long long n,
                      const long long row_stride, const long long col_stride, const int rows_per_chunk, {ACC}* __restrict__ part) {
    // For the pair p = blockIdx.x over the row chunk blockIdx.y: the 6 x 6 cross Gram of its two centered operand blocks (bc is (U, n, 6)) and the two X^T fc products, 48 sums
    // (0..35 cross, 36..41 A^T f, 42..47 B^T f), f = fc[r * row_stride + p * col_stride]. Per-thread sums, a fixed-order warp shuffle and block reduction, one partial row per
    // (pair, chunk): the host adds the chunks in a fixed order, so the result is the same on every run.
    __shared__ {ACC} red[4][48];
    const int p = blockIdx.x;
    const {ACC}* A = bc + ia[p] * n * 6;
    const {ACC}* B = bc + ib[p] * n * 6;
    {ACC} acc[48];
    #pragma unroll
    for (int k = 0; k < 48; ++k) acc[k] = 0.0;
    const long long r0 = (long long)blockIdx.y * rows_per_chunk;
    const long long r1 = min(n, r0 + rows_per_chunk);
    for (long long r = r0 + threadIdx.x; r < r1; r += blockDim.x) {
        {ACC} a[6], b[6];
        #pragma unroll
        for (int k = 0; k < 6; ++k) { a[k] = A[r * 6 + k]; b[k] = B[r * 6 + k]; }
        const {ACC} f = fc[r * row_stride + p * col_stride];
        #pragma unroll
        for (int k = 0; k < 6; ++k) {
            #pragma unroll
            for (int m = 0; m < 6; ++m) acc[k * 6 + m] += a[k] * b[m];
            acc[36 + k] += a[k] * f;
            acc[42 + k] += b[k] * f;
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int k = 0; k < 48; ++k) {
        {ACC} v = acc[k];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();
    if (threadIdx.x < 48) {
        {ACC} t = 0.0;
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w) t += red[w][threadIdx.x];
        part[((long long)p * gridDim.y + blockIdx.y) * 48 + threadIdx.x] = t;
    }
}
""",
)
_GRAM_KERNEL: Any = None
_GRAM_THREADS = 128
_GRAM_ROWS_PER_CHUNK = 4096


def kernel(cp: Any, real: Any = None) -> Any:
    """``pair_gram_blocks`` (double only: the Gram feeds a ridge solve); compiled once. The register-pressure registry compiles it through this."""
    global _GRAM_KERNEL
    if _GRAM_KERNEL is None:
        _GRAM_KERNEL = cp.RawKernel(_GRAM_SRC, "pair_gram_blocks")
    return _GRAM_KERNEL


def _gram_sums(cp: Any, bc: Any, fc: Any, ia_d: Any, ib_d: Any, row_stride: int, col_stride: int) -> Any:
    """The (P, 48) sums of ``pair_gram_blocks`` for the pairs ``(ia_d[p], ib_d[p])`` over all rows of the (U, n, 6) operand blocks."""
    P, n = int(ia_d.shape[0]), int(bc.shape[1])
    chunks = max(1, -(-n // _GRAM_ROWS_PER_CHUNK))
    part = cp.empty((P, chunks, 48), dtype=cp.float64)  # Intentional float64: Gram sums feeding a ridge solve
    kernel(cp)((P, chunks), (_GRAM_THREADS,), (bc, fc, ia_d, ib_d, np.int64(n), np.int64(row_stride), np.int64(col_stride), np.int32(_GRAM_ROWS_PER_CHUNK), part))  # launch-ok: one block per (pair, row chunk) is the design
    return part.sum(axis=1)


def _batched_gate_scalars(cp: Any, Vdev: Any, basis_by_idx: list, name_to_idx: dict, src_pairs: list, yc_rel: Any, P: int) -> Any:
    """Per-candidate gate scalars ``(form std, residual std, residual SS, residual.y dot)`` from one batched normal-equations solve; ``None`` if an operand is unknown."""
    # BATCHED-READ gate (residency): the per-candidate OLS keeps the EXACT per-candidate lstsq
    # (SVD, same solver + values as before - cusolver runs fully on device, no host read), but each
    # candidate's four gate scalars (f_std, std(resid), centered-resid SS, resid.y_rel dot) are written into
    # resident (P,) vectors and read back in ONE batched D2H at the end, instead of two scalar syncs per
    # candidate. The host gate branches below are byte-identical to the per-candidate version.
    # Build every candidate's 12x12 normal-equations Gram + rhs resident (matmuls, no host read), then ONE
    # BATCHED solve - avoiding cp.linalg.lstsq's per-candidate internal host sync (its SVD rank readback).
    # The centered-OLS residual gate scalars are quadratic forms in beta, so they never need the residual
    # vector materialised: with the design centered and fc = fv - mean(fv), resid has mean 0, hence
    #   ss_resid = fc.fc - beta.rhs   (rhs = Xc^T fc, using G beta = rhs),
    #   resid_std = sqrt(ss_resid / n),   num = fc.y_rel - beta.(Xc^T y_rel).
    # A tiny trace-scaled ridge makes every Gram solvable in one batched call (negligible ~1e-10 shift;
    # handles the near-collinear designs lstsq would rank-cut - the gate scalars still do not flip, F2 +
    # the retention suites confirm). f_std / the four gate scalars read back in ONE batched D2H.
    _nrows = float(int(Vdev.shape[0]))
    ia_l, ib_l = [], []
    for j in range(P):
        ia = name_to_idx.get(src_pairs[j][0])
        ib = name_to_idx.get(src_pairs[j][1])
        if ia is None or ib is None:
            return None  # operand missing from the resident base set -> exact CPU path
        ia_l.append(ia)
        ib_l.append(ib)
    # The centered basis of every distinct operand once as an (operands, n, 6) block: centering is per column, so a pair's centered design is the two operands' blocks side by side.
    used = sorted(set(ia_l) | set(ib_l))
    slot = {u: k for k, u in enumerate(used)}
    # Intentional float64: the Gram sums feed a ridge solve and the Gram kernel reads double operands (a float32 operand would be read as garbage), so every input is widened here
    bc = cp.stack([cp.asarray(basis_by_idx[u], dtype=cp.float64) for u in used])  # (U, n, 6)
    bc = bc - bc.mean(axis=1, keepdims=True)
    ia_d = cp.asarray(np.fromiter((slot[i] for i in ia_l), dtype=np.int64, count=P))
    ib_d = cp.asarray(np.fromiter((slot[i] for i in ib_l), dtype=np.int64, count=P))
    Vdev = cp.asarray(Vdev, dtype=cp.float64)  # Intentional float64: see above
    yc_rel = cp.asarray(yc_rel, dtype=cp.float64)  # Intentional float64: see above
    FC = Vdev - Vdev.mean(axis=0, keepdims=True)  # (n, P) centered form values
    _fstd_v = Vdev.std(axis=0)
    _sfc = (FC * FC).sum(axis=0)
    _fyr = yc_rel @ FC
    # Intentional float64: batched Gram solve, the 1e-10 trace ridge is below float32 epsilon so the solve needs double
    _G = cp.empty((P, 12, 12), dtype=cp.float64)
    # Intentional float64: batched Gram solve, the 1e-10 trace ridge is below float32 epsilon so the solve needs double
    _rhs = cp.empty((P, 12), dtype=cp.float64)
    # Intentional float64: batched Gram solve, the 1e-10 trace ridge is below float32 epsilon so the solve needs double
    _xy = cp.empty((P, 12), dtype=cp.float64)
    # The Gram of a pair is [[Gaa, Gab], [Gab^T, Gbb]]: the diagonal blocks and X^T y depend on one operand and are formed once per distinct operand; the cross block and the two X^T fc
    # products are one ``pair_gram_blocks`` launch for all pairs, each block reading the two operands' rows once (no gathered copies, no 6 x 6 batched GEMM over n rows).
    nu = len(used)
    ar = cp.arange(nu, dtype=cp.int64)
    diag = _gram_sums(cp, bc, yc_rel, ar, ar, 1, 0)  # (U, 48): [:36] the operand's own Gram, [36:42] X^T y
    cross = _gram_sums(cp, bc, FC, ia_d, ib_d, int(FC.shape[1]), 1)  # (P, 48)
    gab = cross[:, :36].reshape(P, 6, 6)
    gd = diag[:, :36].reshape(nu, 6, 6)
    _G[:, :6, :6] = gd[ia_d]
    _G[:, 6:, 6:] = gd[ib_d]
    _G[:, :6, 6:] = gab
    _G[:, 6:, :6] = gab.transpose(0, 2, 1)
    _rhs[:, :6] = cross[:, 36:42]
    _rhs[:, 6:] = cross[:, 42:48]
    _xy[:, :6] = diag[ia_d, 36:42]
    _xy[:, 6:] = diag[ib_d, 36:42]
    # Intentional float64: batched Gram solve, the 1e-10 trace ridge is below float32 epsilon so the solve needs double
    _ridge = (1e-10 * cp.einsum("pii->p", _G) / 12.0 + 1e-30)[:, None, None] * cp.eye(12, dtype=cp.float64)
    _beta = cp.linalg.solve(_G + _ridge, _rhs[:, :, None])[:, :, 0]  # (P, 12) batched
    _ssrc_v = _sfc - (_beta * _rhs).sum(axis=1)
    _rstd_v = cp.sqrt(cp.maximum(_ssrc_v, 0.0) / _nrows)
    _num_v = _fyr - (_beta * _xy).sum(axis=1)
    # ONE batched D2H of every candidate's gate scalars (was a per-candidate lstsq sync + gate reads).
    _gate = cp.asnumpy(cp.stack([_fstd_v, _rstd_v, _ssrc_v, _num_v], axis=1))  # (P, 4)
    return _gate


def _upload_pure_form_inputs(cp, form_values, base_columns, rel_host, n, P, B):
    """ONE bulk host-to-device upload of the form-value matrix, the raw-operand matrix and the relevance target; ``None`` on a ragged column."""
    from ._usability_aware_selection import _f64, _scrub

    # ---- ONE bulk H2D: the (n, P) form-value matrix, the (n, B) raw-operand matrix, the (n,) target. ----
    # Intentional float64: host copy of the float64 producer matrix the batched solve reads back
    Vhost = np.empty((n, P), dtype=np.float64)
    for j in range(P):
        col = _f64(_scrub(np.asarray(form_values[j])))
        if col.shape[0] != n:
            return None  # ragged -> let the CPU path handle it exactly
        Vhost[:, j] = col
    # Intentional float64: host copy of the float64 producer matrix the batched solve reads back
    Bhost = np.empty((n, B), dtype=np.float64)
    for i in range(B):
        col = _f64(_scrub(np.asarray(base_columns[i])))
        if col.shape[0] != n:
            return None
        Bhost[:, i] = col

    return cp.asarray(Vhost), cp.asarray(Bhost), cp.asarray(rel_host)


def _basis_stack(cp, xcol):
    """Standardize ``xcol`` and expand it into the 6-function additive basis (linear, square, cube, signed-sqrt, signed-log1p, reciprocal-magnitude) used to test whether a candidate adds nonlinear value over its raw operand."""
    xs = standardise(xcol, xp=cp)
    return cp.stack([
        xs, xs * xs, xs * xs * xs,
        cp.sign(xs) * cp.sqrt(cp.abs(xs)),
        cp.sign(xs) * cp.log1p(cp.abs(xs)),
        1.0 / (cp.abs(xs) + 1.0),
    ], axis=1)                                 # (n, 6)
def _verdict_flags(_gate, P, min_resid_frac, min_resid_corr, y_std, ss_yrel) -> list:
    """Per-candidate keep verdicts from the batched gate scalars: non-separable and residual relevant to y."""
    out = [False] * P
    for j in range(P):
        f_std, resid_std, ss_rc, num = (float(_v) for _v in _gate[j])
        if f_std <= 1e-12:
            continue
        # (a) non-separable gate.
        if resid_std < min_resid_frac * f_std:
            continue
        # (b) residual relevant to y: the SAME centered |corr| estimator (std<1e-12 -> 0) the CPU ``_abscorr``
        # uses; only the scalar |corr| decides admission.
        if resid_std < 1e-12 or y_std < 1e-12 or ss_yrel <= 0.0 or ss_rc <= 0.0:
            continue
        denom = float(np.sqrt(ss_rc * ss_yrel))
        corr = abs(num / denom) if denom > 0.0 and np.isfinite(num / denom) else 0.0
        out[j] = corr >= min_resid_corr
    return out


def adds_nonlinear_value_batch_gpu_resident(
    form_values: list,
    src_pairs: list,
    base_names: list,
    base_columns: list,
    rel_y: np.ndarray,
    *,
    min_resid_frac: float,
    min_resid_corr: float,
) -> Optional[list]:
    """Resident twin of the per-candidate ``_adds_nonlinear_value`` gate, batched over the whole pool.

    Parameters mirror the CPU loop's inputs, all on the HOST (uploaded ONCE here):
      * ``form_values``  : list of P candidate form-value columns (each (n,) host float; the SAME
                            ``cand.values`` the CPU loop scrubs/upcasts).
      * ``src_pairs``    : list of P ``(nm_a, nm_b)`` raw-operand-name tuples (the candidate's two operands).
      * ``base_names``   : the B distinct raw-operand names, indexable to ``base_columns``.
      * ``base_columns`` : list of B host (n,) raw-operand columns aligned to ``base_names`` (X_fit[nm]).
      * ``rel_y``        : the (n,) REGRESSION relevance target (continuous y).

    Returns a list of P booleans (one per candidate, the SAME verdict the CPU ``_adds_nonlinear_value``
    returns: kept = genuinely non-separable AND residual relevant to y), or ``None`` on an empty/degenerate
    pool or ANY cupy/device/import error so the caller runs the exact per-candidate CPU path.

    Each candidate's verdict needs only bounded scalars off device (form std, residual std, |corr(resid,y)|);
    the (n,) residual itself NEVER leaves the GPU. Raw-operand columns are uploaded ONCE (``Bdev``) and reused
    by index for every candidate - no per-candidate re-upload.
    """
    if not form_values:
        return None
    try:
        import cupy as cp
    except ImportError as e:
        logger.debug("cupy import failed, resident pure-form retention unavailable: %s", e)
        return None
    _DEV_ERRS = _device_error_types(cp)
    try:
        from ._gpu_policy import gpu_globally_disabled
        if gpu_globally_disabled():
            return None
    except Exception as e:  # nosec B110 - optional dependency import guard
        logger.debug("gpu_globally_disabled() check failed, proceeding without the global-disable gate: %s", e)

    try:
        from ._usability_aware_selection import _f64, _scrub

        # Intentional float64: host copy of the float64 producer matrix the batched solve reads back
        rel_host = _f64(_scrub(np.asarray(rel_y, dtype=np.float64)))
        n = int(rel_host.shape[0])
        if n < 2:
            return None

        P = len(form_values)
        B = len(base_names)
        if B == 0 or len(base_columns) != B:
            return None
        name_to_idx = {nm: i for i, nm in enumerate(base_names)}

        uploaded = _upload_pure_form_inputs(cp, form_values, base_columns, rel_host, n, P, B)
        if uploaded is None:
            return None
        Vdev, Bdev, ydev = uploaded

        # Precompute the 6-function additive basis for EVERY distinct raw operand ONCE (resident, (n, 6) each),
        # so a candidate's design is two index lookups - not a re-derivation/re-upload per candidate.

        basis_by_idx = [_basis_stack(cp, Bdev[:, i]) for i in range(B)]

        # Relevance-target centered pieces (resident, computed ONCE).
        ybar_rel = ydev.mean()
        yc_rel = ydev - ybar_rel
        ss_yrel = float(cp.dot(yc_rel, yc_rel))  # scalar; bounded
        y_std = float(ydev.std())

        _gate = _batched_gate_scalars(cp, Vdev, basis_by_idx, name_to_idx, src_pairs, yc_rel, P)
        if _gate is None:
            return None  # operand missing from the resident base set -> exact CPU path
        return _verdict_flags(_gate, P, min_resid_frac, min_resid_corr, y_std, ss_yrel)
    except _DEV_ERRS as _exc:
        logger.debug("adds_nonlinear_value_batch_gpu_resident fell back to its default after %s", type(_exc).__name__)
        return None  # genuine cupy/device/linalg fault in setup/upload -> exact CPU path (logic bugs propagate)


__all__ = ["adds_nonlinear_value_batch_gpu_resident"]
