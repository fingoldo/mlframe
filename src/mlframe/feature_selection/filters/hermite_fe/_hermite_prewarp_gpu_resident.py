"""GPU-RESIDENT twin of the rank-1 ALS warm start
(:func:`_hermite_prewarp.warm_start_als_seed` and its inner ``_als_solve``).

RESIDENCY CONTRACT (not a wall win). Gated on the resident flag
(``MLFRAME_FE_GPU_STRICT`` + ``MLFRAME_FE_GPU_STRICT_RESIDENT``); default OFF.
On this GTX 1050 Ti the ALS operand is small and the sweep is STRICTLY SEQUENTIAL
(1 init + iters*2 dependent normal-equation solves on a tiny ``(degree+1)`` system;
g depends on cb, f on ca, next cb on f), so the GPU twin is EXPECTED to be slower
than the CPU normal-eq path -- and that is a PASS by the residency contract. The
CPU bench-note in ``warm_start_als_seed`` documents why it stays CPU for the WALL;
this twin exists for RESIDENCY COMPLETENESS so that, under the resident flag, the
design columns (``B_a``/``B_b``) and the target stay resident on the device and the
per-iteration linear algebra (the alternating ``solve(AtA, At b)`` + factor matvecs)
runs on cupy with NO per-iteration bulk (n-scaled) H2D/D2H of the operands.

DEVICE-BORN DESIGN (2026-06-30, H2D collapse). The two rank-1 ALS design matrices
``B_a``/``B_b`` (n x degree+1) are the only bulk n-scaled operands and, at 300k rows
over the per-operand-pair sweeps, the host->device upload of the prebuilt matrices
(``cp.asarray(B_a)`` / ``cp.asarray(B_b)``) was ~374MB aggregate. Those matrices are
a DETERMINISTIC closed-form function of the small standardised column ``z`` and the
basis recurrence, so they are now BUILT ON DEVICE from ``z`` via
:func:`_build_basis_matrix_gpu` -- the cupy mirror of the host
``_build_basis_*``/``build_basis_matrix`` recurrences. The caller uploads ONLY the
tiny standardised columns ``z_a``/``z_b`` (n,) and the centred target ``yc`` (n,);
the (n x degree+1) designs never touch the H2D path. (n,) << (n x degree+1) so the
bulk transfer collapses by the design-matrix width factor (~degree+1).

What is resident vs host control-flow (allowed by the contract):
  * RESIDENT: ``B_a``/``B_b`` (n x degree+1, BUILT ON DEVICE from resident
    ``z_a``/``z_b``), ``yc`` (n,), the alternating factors ``f``/``g`` (n,), every
    ``AtA``/``Atb`` normal-equation system and its solution coefficient vector --
    one bulk H2D of the two standardised columns + target at entry, then no
    n-scaled transfer until the two small coefficient vectors return.
  * HOST scalar D2H (bounded, O(iters)): the ``std(yc)``/``std(f)``/``std(g)``
    stabiliser scalars and the final ``isfinite`` guard. These are tiny (a single
    float, or a (degree+1)-length coefficient vector far below ``BULK_BYTES``).

Selection-equivalence: the returned ``(coef_a, coef_b)`` mirror the CPU normal-eq
math IDENTICALLY (same init = plain 1-D fit ``solve(Bb^T Bb, Bb^T yc)``, same
``g_norm = g/(std(g)+1e-12)`` / ``f_norm`` stabilisers, same iteration count, same
``solve(AtA, Atb)`` with an ``lstsq`` fallback on a singular system). The
device-built basis matrix mirrors the host recurrence EXACTLY (same recurrence,
same column order) and agrees to ~1e-13 (only FP reduction order differs); the
resulting coefficients agree to ~1e-12. The downstream optimiser warm-starts from
these coefficients and the feature selection is unchanged; cupy vs numpy
reduction-order differs only at ~1e-13, far below any decision boundary.

Any cupy / device error raises, so the caller falls back to the CPU path and the
default (flag-off) path stays byte-identical and a GPU fault never breaks a fit.
"""
from __future__ import annotations

import numpy as np


def _build_basis_matrix_gpu(cp, basis: str, z_gpu, max_degree: int):
    """Build ``B[i, k] = T_k(z[i])`` for k=0..max_degree DIRECTLY ON DEVICE -- the
    cupy mirror of the host ``_build_basis_*`` njit recurrences (hermite_fe
    ``_build_basis_hermite``/``_legendre``/``_chebyshev``/``_laguerre``) and the
    public ``build_basis_matrix`` dispatcher.

    ``z_gpu`` is a resident ``(n,)`` cupy array (the standardised column). Returns a
    resident ``(n, max_degree+1)`` cupy array. The recurrence, the seed columns
    (``B[:,0]=1`` and the basis-specific ``B[:,1]``) and the column order MATCH the
    host build EXACTLY, so the design is selection-equivalent to ~1e-13 (only the
    FP reduction order of the cupy vector ops differs from the scalar njit loop).

    Raises ``KeyError`` on an unsupported basis (matching ``build_basis_matrix``)."""
    # The recurrence runs in FLOAT64 regardless of the input dtype -- NOT a lazy hardcode: the fast-growing
    # Laguerre/Hermite polynomials (and high-degree Chebyshev) build L_k / He_k from large-magnitude alternating
    # terms, so evaluating the recurrence in float32 suffers catastrophic cancellation (verified: ~10% relative
    # error in the deg-4 Laguerre coefficients vs f64). The design is device-BUILT (not H2D), so its dtype does
    # not affect the H2D residency win -- only the recurrence stability -- so it stays f64. (The relaxed prewarp
    # path still halves the H2D by uploading the standardised COLUMN za/zb as f32; only this on-device recurrence
    # keeps f64.) A f32 input column is widened here.
    x = cp.ascontiguousarray(cp.asarray(z_gpu, dtype=cp.float64)).reshape(-1)
    n = x.shape[0]
    nc = int(max_degree) + 1
    B = cp.empty((n, nc), dtype=cp.float64)
    B[:, 0] = 1.0
    if nc <= 1:
        return B

    if basis == "hermite":
        # He recurrence: He_k = x*He_{k-1} - (k-1)*He_{k-2}; He_1 = x.
        B[:, 1] = x
        for k in range(2, nc):
            B[:, k] = x * B[:, k - 1] - (k - 1) * B[:, k - 2]
    elif basis == "legendre":
        # Bonnet: P_k = ((2k-1)*x*P_{k-1} - (k-1)*P_{k-2}) / k; P_1 = x.
        B[:, 1] = x
        for k in range(2, nc):
            B[:, k] = ((2 * k - 1) * x * B[:, k - 1] - (k - 1) * B[:, k - 2]) / k
    elif basis == "chebyshev":
        # T_{k+1} = 2x*T_k - T_{k-1}; T_1 = x.
        two_x = 2.0 * x
        B[:, 1] = x
        for k in range(2, nc):
            B[:, k] = two_x * B[:, k - 1] - B[:, k - 2]
    elif basis == "laguerre":
        # L_k = ((2k-1-x)*L_{k-1} - (k-1)*L_{k-2}) / k; L_1 = 1 - x.
        B[:, 1] = 1.0 - x
        for k in range(2, nc):
            B[:, k] = ((2 * k - 1 - x) * B[:, k - 1] - (k - 1) * B[:, k - 2]) / k
    else:
        raise KeyError(
            f"_build_basis_matrix_gpu: basis {basis!r} not in "
            f"['chebyshev', 'hermite', 'laguerre', 'legendre']; "
            f"factory-based bases must use the per-call eval_func path."
        )
    return B


def _als_solve_gpu(cp, A, b):
    """Resident normal-equations solve ``solve(AtA, At b)`` with an exact ``lstsq``
    fallback on a singular ``AtA`` -- GPU twin of the CPU inner ``_als_solve``.
    ``A`` (n x d) and ``b`` (n,) are resident (f64 -- the design recurrence keeps f64 for stability); the
    returned coefficient (d,) stays resident. Mirrors the CPU least-norm / normal-eq equivalence for a
    full-column-rank system (the orthogonal-poly basis scaled by g_norm/f_norm stays well-conditioned), falling
    back to SVD lstsq bit-for-bit as the CPU path does."""
    AtA = A.T @ A
    try:
        return cp.linalg.solve(AtA, A.T @ b)
    except Exception:
        coef = cp.linalg.lstsq(A, b, rcond=None)[0]
        return coef


def _als_sweep_gpu(cp, Ba, Bb, yc, iters):
    """Resident alternating sweep shared by both entry points. ``Ba``/``Bb``/``yc``
    are resident; returns the two host coefficient vectors (or ``(None, None)``)."""
    # Initialise g(b) from a plain 1-D least-squares fit on the b-basis (resident).
    cb = _als_solve_gpu(cp, Bb, yc)
    g = Bb @ cb
    ca = None
    for _ in range(max(1, int(iters))):
        # cp.std(...) kept as a device 0-dim scalar (no float()): it is only a broadcast divisor, so the host
        # roundtrip was pure waste -- the divide stays fully resident and the result is bit-identical.
        g_norm = g / (cp.std(g) + 1e-12)
        ca = _als_solve_gpu(cp, Ba * g_norm[:, None], yc)
        f = Ba @ ca
        f_norm = f / (cp.std(f) + 1e-12)
        cb = _als_solve_gpu(cp, Bb * f_norm[:, None], yc)
        g = Bb @ cb
    if ca is None:
        return None, None
    # Bring the two tiny coefficient vectors back (degree+1 floats each, far below BULK_BYTES).
    ca_h = cp.asnumpy(ca)
    cb_h = cp.asnumpy(cb)
    if not (np.all(np.isfinite(ca_h)) and np.all(np.isfinite(cb_h))):
        return None, None
    return np.ascontiguousarray(ca_h, dtype=np.float64), np.ascontiguousarray(cb_h, dtype=np.float64)


def warm_start_als_seed_gpu(B_a: np.ndarray, B_b: np.ndarray, y: np.ndarray,
                            *, iters: int = 3) -> tuple:
    """GPU-resident rank-1 ALS warm start from PREBUILT host design matrices.

    Legacy entry retained for callers that already hold ``B_a``/``B_b`` (the
    per-degree optimiser warm-start and the OOS auto-escalation paths). Returns the
    SAME ``(coef_a, coef_b)`` as the CPU :func:`_hermite_prewarp.warm_start_als_seed`
    (selection-equivalent, equal to ~1e-13). Raises on any cupy/device error so the
    caller falls back to the CPU normal-eq path.

    Prefer :func:`warm_start_als_seed_gpu_from_z` (device-born design, no n-scaled
    H2D of the matrices) when the caller can supply the standardised columns.

    The host preamble (centring ``y``, the no-variance guard) mirrors the CPU path
    EXACTLY using numpy so the resident sweep operates on the byte-identical centred
    target. Only AFTER that guard are the two design matrices + the centred target
    uploaded ONCE to the device; the alternating sweep is then fully resident."""
    import cupy as cp

    yc_h = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    yc_h = yc_h - yc_h.mean()
    if float(np.std(yc_h)) < 1e-12:
        return None, None

    # ---- ONE bulk H2D of the two designs + target; everything below stays resident ----
    Ba = cp.asarray(np.ascontiguousarray(np.asarray(B_a, dtype=np.float64)))
    Bb = cp.asarray(np.ascontiguousarray(np.asarray(B_b, dtype=np.float64)))
    yc = cp.asarray(yc_h)

    try:
        return _als_sweep_gpu(cp, Ba, Bb, yc, iters)
    except (np.linalg.LinAlgError, ValueError):
        return None, None


def warm_start_als_seed_gpu_from_z(z_a: np.ndarray, z_b: np.ndarray, y: np.ndarray,
                                   *, basis: str, max_degree: int,
                                   iters: int = 3) -> tuple:
    """GPU-resident rank-1 ALS warm start with a DEVICE-BORN design (H2D collapse).

    Identical math / return contract to :func:`warm_start_als_seed_gpu` and to the
    CPU :func:`_hermite_prewarp.warm_start_als_seed`, but the (n x degree+1) design
    matrices ``B_a``/``B_b`` are BUILT ON DEVICE from the resident standardised
    columns ``z_a``/``z_b`` via :func:`_build_basis_matrix_gpu` instead of being
    uploaded prebuilt. Only the small ``z_a``/``z_b`` (n,) and centred target ``yc``
    (n,) cross the H2D path, collapsing the ~374MB design upload at 300k.

    The device basis matrix mirrors the host ``build_basis_matrix`` recurrence
    EXACTLY (same recurrence + column order), so the design agrees to ~1e-13 and the
    coefficients to ~1e-12 -- selection-equivalent. Raises on any cupy/device error
    so the caller falls back to the CPU normal-eq path."""
    import cupy as cp

    yc_h = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    yc_h = yc_h - yc_h.mean()
    if float(np.std(yc_h)) < 1e-12:
        return None, None

    # ---- ONE bulk H2D of the two SMALL standardised columns + target ----
    # za/zb are the per-pair standardised columns -> distinct, a genuine upload each. yc_h is the centred fit
    # target, identical across every pair the ALS seed scans, so it re-uploaded once per pair (H2D audit: 32x).
    # Route it through the content-keyed resident cache so the target uploads ONCE; read-only in _als_sweep_gpu
    # (only _als_solve reads it, never reassigns/mutates) -> selection-equivalent.
    from .._fe_resident_operands import resident_operand
    # Under MLFRAME_CRIT_DTYPE_RELAXED (default ON) the two standardised columns + the centred target UPLOAD as
    # float32 (half the H2D -- the residency win). The on-device basis recurrence + the least-squares sweep then
    # keep FLOAT64 (see _build_basis_matrix_gpu): the fast-growing Laguerre/Hermite polynomials cancel
    # catastrophically in f32, so the design must be f64 -- and since it is device-BUILT, its dtype does not
    # touch the H2D. So only the za/zb/yc VALUES are f32-rounded (~1e-6); the coefficients shift within the
    # f32-input condition bound, a smooth non-tie-sensitive CMA-ES seed, so the FE selection is unchanged
    # (validated on F2 across distributions + the hermite biz/e2e suites). MLFRAME_CRIT_DTYPE_RELAXED=0 restores
    # the strict f64 upload. yc rides the content-keyed cache; f64 basis @ f32 yc promotes to f64 in the solve.
    try:
        from .._fe_gpu_batch._devices import crit_float_dtype
        _zdt = crit_float_dtype()
    except Exception:
        _zdt = cp.float64
    za = cp.asarray(np.ascontiguousarray(np.asarray(z_a, dtype=_zdt)).reshape(-1))
    zb = cp.asarray(np.ascontiguousarray(np.asarray(z_b, dtype=_zdt)).reshape(-1))
    yc = resident_operand(yc_h, "hermite_prewarp_yc", dtype=_zdt)

    deg = max(1, int(max_degree))
    try:
        # ---- DEVICE-BORN designs: the n x (degree+1) matrices never touch H2D ----
        Ba = _build_basis_matrix_gpu(cp, str(basis), za, deg)
        Bb = _build_basis_matrix_gpu(cp, str(basis), zb, deg)
        return _als_sweep_gpu(cp, Ba, Bb, yc, iters)
    except (np.linalg.LinAlgError, ValueError):
        return None, None
