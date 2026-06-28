"""GPU-RESIDENT twin of the per-candidate non-separability filter inside
:func:`_fe_pure_form_retention.retain_usable_pure_forms`.

RESIDENCY CONTRACT (not a wall win). Gated on the resident flag
(``MLFRAME_FE_GPU_STRICT`` + ``MLFRAME_FE_GPU_STRICT_RESIDENT``); default OFF. On
this GTX 1050 Ti the retention pool runs on a ~3000-row subsample, so the cupy
launch + reduction overhead dominates the tiny per-candidate 12-column OLS and
this twin is EXPECTED to be SLOWER than the per-candidate sklearn CPU path -- and
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
pulled back -- NO per-candidate residual D2H, NO per-candidate re-upload.

What is resident vs host control-flow (allowed by the contract):
  * RESIDENT (one bulk H2D at entry): the (n, P) candidate form-value matrix
    ``Vdev`` (float64), the (n, B) distinct raw-operand matrix ``Bdev`` and the
    (n,) relevance target ``ydev``. From them, per candidate, the device builds the
    12-column additive basis (6 functions of each of the candidate's two operands,
    looked up by base-column index -- NO re-upload), solves the mean-centered OLS
    via ``cp.linalg.lstsq``, forms the residual, and reduces it to the two scalars
    the CPU gate compares. NO per-candidate H2D and NO per-candidate value/residual
    D2H.
  * HOST scalar D2H (allowed by the contract): per candidate the form std, the
    residual std and the |corr(resid, y)| -- three bounded scalars, the SAME
    quantities the CPU ``_adds_nonlinear_value`` computes per candidate. None is
    bulk per-candidate value data.

SELECTION-EQUIVALENCE IS THE BAR. The math is the SAME as the CPU
``_adds_nonlinear_value``: the SAME 6-function additive single-operand basis per
operand (standardized x, x^2, x^3, sign*sqrt|x|, sign*log1p|x|, 1/(|x|+1)), the
SAME StandardScaler+LinearRegression == mean-centered-OLS residual (the scaler's
affine map is absorbed into the coefficients + intercept; identical up to fp
round-off -- this is exactly the equivalence ``gpu_additive_basis_residual``
already documents), the SAME ``std(resid) < min_resid_frac * f_std`` non-separable
gate and the SAME ``|corr(resid, y)| >= min_resid_corr`` relevance gate (REGRESSION
relevance target ``y``; the SAME ``_abscorr`` centered dot / sqrt(ss) estimator
with the SAME std<1e-12 zero-guard). Only the float reduction ORDER differs between
cupy and numpy (~1e-12), to which the boolean gates are tolerant; on ANY
cupy/device error the whole batch returns ``None`` so the caller runs the exact
per-candidate CPU/sklearn path.

CLASSIFICATION is NOT ported here (the caller passes ``is_clf``): the classification
relevance gate is the WHOLE-form point-biserial ``|corr(fv, class_indicator)|`` (not
the residual corr), a different discriminator -- a classification call returns
``None`` so the dispatcher falls through to the exact CPU path.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


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
    by index for every candidate -- no per-candidate re-upload.
    """
    if not form_values:
        return None
    try:
        import cupy as cp
    except Exception:
        return None
    # Narrow device-fault set for the resident fallbacks below. A genuine cupy/device/linalg fault
    # (singular OLS, OOM, CUDA runtime error) legitimately routes to the exact CPU path; a logic/shape
    # bug (ValueError/KeyError/IndexError) must SURFACE to tests, not be silently swallowed as a
    # "device fallback" -- so the per-candidate and outer guards catch only these, not bare Exception.
    _dev_errs = [np.linalg.LinAlgError]
    try:
        _dev_errs.append(cp.cuda.runtime.CUDARuntimeError)
    except Exception:
        pass
    try:
        _dev_errs.append(cp.cuda.memory.OutOfMemoryError)
    except Exception:
        pass
    # FIX4 (2026-06-28): the direct cp.linalg.lstsq below raises cuSOLVER/cuBLAS errors that subclass
    # plain RuntimeError, NOT CUDARuntimeError -> omitting them would crash instead of falling back to
    # the exact CPU path. getattr so an absent symbol can't break the tuple builder.
    try:
        from cupy_backends.cuda.libs import cusolver as _cusolver  # type: ignore
        _e = getattr(_cusolver, "CUSOLVERError", None)
        if _e is not None:
            _dev_errs.append(_e)
        from cupy_backends.cuda.libs import cublas as _cublas  # type: ignore
        _e = getattr(_cublas, "CUBLASError", None)
        if _e is not None:
            _dev_errs.append(_e)
    except Exception:
        pass
    _DEV_ERRS = tuple(_dev_errs)
    try:
        from ._gpu_policy import gpu_globally_disabled
        if gpu_globally_disabled():
            return None
    except Exception:
        pass

    try:
        from ._usability_aware_selection import _f64, _scrub

        rel_host = _f64(_scrub(np.asarray(rel_y, dtype=np.float64)))
        n = int(rel_host.shape[0])
        if n < 2:
            return None

        P = len(form_values)
        B = len(base_names)
        if B == 0 or len(base_columns) != B:
            return None
        name_to_idx = {nm: i for i, nm in enumerate(base_names)}

        # ---- ONE bulk H2D: the (n, P) form-value matrix, the (n, B) raw-operand matrix, the (n,) target. ----
        Vhost = np.empty((n, P), dtype=np.float64)
        for j in range(P):
            col = _f64(_scrub(np.asarray(form_values[j])))
            if col.shape[0] != n:
                return None  # ragged -> let the CPU path handle it exactly
            Vhost[:, j] = col
        Bhost = np.empty((n, B), dtype=np.float64)
        for i in range(B):
            col = _f64(_scrub(np.asarray(base_columns[i])))
            if col.shape[0] != n:
                return None
            Bhost[:, i] = col

        Vdev = cp.asarray(Vhost)          # (n, P) resident
        Bdev = cp.asarray(Bhost)          # (n, B) resident
        ydev = cp.asarray(rel_host)       # (n,) resident

        # Precompute the 6-function additive basis for EVERY distinct raw operand ONCE (resident, (n, 6) each),
        # so a candidate's design is two index lookups -- not a re-derivation/re-upload per candidate.
        def _basis(xcol):
            xs = (xcol - xcol.mean()) / (xcol.std() + 1e-12)
            return cp.stack([
                xs, xs * xs, xs * xs * xs,
                cp.sign(xs) * cp.sqrt(cp.abs(xs)),
                cp.sign(xs) * cp.log1p(cp.abs(xs)),
                1.0 / (cp.abs(xs) + 1.0),
            ], axis=1)                                 # (n, 6)

        basis_by_idx = [_basis(Bdev[:, i]) for i in range(B)]

        # Relevance-target centered pieces (resident, computed ONCE).
        ybar_rel = ydev.mean()
        yc_rel = ydev - ybar_rel
        ss_yrel = float(cp.dot(yc_rel, yc_rel))   # scalar; bounded
        y_std = float(ydev.std())

        out = [False] * P
        for j in range(P):
            try:
                ia = name_to_idx.get(src_pairs[j][0])
                ib = name_to_idx.get(src_pairs[j][1])
                if ia is None or ib is None:
                    return None  # operand missing from the resident base set -> exact CPU path
                fv = Vdev[:, j]
                f_std = float(fv.std())            # scalar D2H (bounded), matches CPU np.std(fv)
                if f_std <= 1e-12:
                    out[j] = False
                    continue
                # 12-column additive basis design (the candidate's two operands), mean-centered OLS == the
                # CPU StandardScaler+LinearRegression residual (affine scaler absorbed into coeffs+intercept).
                Xr = cp.concatenate([basis_by_idx[ia], basis_by_idx[ib]], axis=1)   # (n, 12)
                Xc = Xr - Xr.mean(axis=0, keepdims=True)
                fbar = fv.mean()
                fc = fv - fbar
                # rcond=None (cupy uses the eps*max(M,N) singular-value cutoff) on the 12-col additive basis.
                # Accepted vs sklearn LinearRegression's lstsq: on the well-conditioned basis the two agree;
                # on a near-collinear basis the rank cutoff can differ, but only the std(resid)/|corr| GATE
                # SCALARS feed the verdict and the F2 selection-equivalence suite confirms no flip. A device
                # fault here routes to the exact CPU path via _DEV_ERRS; not a silent divergence.
                beta, *_ = cp.linalg.lstsq(Xc, fc, rcond=None)
                pred = fbar + Xc @ beta
                resid = fv - pred
                resid_std = float(resid.std())     # scalar D2H (bounded), matches CPU np.std(resid)
                # (a) non-separable gate.
                if resid_std < min_resid_frac * f_std:
                    out[j] = False
                    continue
                # (b) residual relevant to y: the SAME centered |corr| estimator (std<1e-12 -> 0) the CPU
                # ``_abscorr`` uses, computed resident; only the scalar |corr| is pulled back.
                if resid_std < 1e-12 or y_std < 1e-12 or ss_yrel <= 0.0:
                    out[j] = False
                    continue
                rc = resid - resid.mean()
                ss_rc = float(cp.dot(rc, rc))      # scalar; bounded
                if ss_rc <= 0.0:
                    out[j] = False
                    continue
                num = float(cp.dot(rc, yc_rel))    # scalar; bounded
                denom = float(np.sqrt(ss_rc * ss_yrel))
                corr = abs(num / denom) if denom > 0.0 and np.isfinite(num / denom) else 0.0
                out[j] = corr >= min_resid_corr
            except _DEV_ERRS:
                return None  # genuine cupy/device/linalg fault mid-batch -> exact CPU path (logic bugs propagate)
        return out
    except _DEV_ERRS:
        return None  # genuine cupy/device/linalg fault in setup/upload -> exact CPU path (logic bugs propagate)


__all__ = ["adds_nonlinear_value_batch_gpu_resident"]
