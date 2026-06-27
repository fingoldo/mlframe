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

What is resident vs host control-flow (allowed by the contract):
  * RESIDENT: ``B_a``/``B_b`` (n x degree+1), ``yc`` (n,), the alternating factors
    ``f``/``g`` (n,), every ``AtA``/``Atb`` normal-equation system and its solution
    coefficient vector -- one bulk H2D of the two design matrices + target at entry,
    then no n-scaled transfer until the two small coefficient vectors return.
  * HOST scalar D2H (bounded, O(iters)): the ``std(yc)``/``std(f)``/``std(g)``
    stabiliser scalars and the final ``isfinite`` guard. These are tiny (a single
    float, or a (degree+1)-length coefficient vector far below ``BULK_BYTES``).

Selection-equivalence: the returned ``(coef_a, coef_b)`` mirror the CPU normal-eq
math IDENTICALLY (same init = plain 1-D fit ``solve(Bb^T Bb, Bb^T yc)``, same
``g_norm = g/(std(g)+1e-12)`` / ``f_norm`` stabilisers, same iteration count, same
``solve(AtA, Atb)`` with an ``lstsq`` fallback on a singular system). The downstream
optimiser warm-starts from these coefficients and the feature selection is unchanged;
cupy vs numpy reduction-order differs only at ~1e-13, far below any decision boundary.

Any cupy / device error raises, so the caller falls back to the CPU path and the
default (flag-off) path stays byte-identical and a GPU fault never breaks a fit.
"""
from __future__ import annotations

import numpy as np


def _als_solve_gpu(cp, A, b):
    """Resident normal-equations solve ``solve(AtA, At b)`` with an exact ``lstsq``
    fallback on a singular ``AtA`` -- GPU twin of the CPU inner ``_als_solve``.
    ``A`` (n x d) and ``b`` (n,) are resident; the returned coefficient (d,) stays
    resident. Mirrors the CPU least-norm / normal-eq equivalence for a full-column-
    rank system (the orthogonal-poly basis scaled by g_norm/f_norm stays well-
    conditioned), falling back to SVD lstsq bit-for-bit as the CPU path does."""
    AtA = A.T @ A
    try:
        return cp.linalg.solve(AtA, A.T @ b)
    except Exception:
        coef = cp.linalg.lstsq(A, b, rcond=None)[0]
        return coef


def warm_start_als_seed_gpu(B_a: np.ndarray, B_b: np.ndarray, y: np.ndarray,
                            *, iters: int = 3) -> tuple:
    """GPU-resident rank-1 ALS warm start. Returns the SAME ``(coef_a, coef_b)`` as
    the CPU :func:`_hermite_prewarp.warm_start_als_seed` (selection-equivalent, equal
    to ~1e-13). Raises on any cupy/device error so the caller falls back to the CPU
    normal-eq path.

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
        # Initialise g(b) from a plain 1-D least-squares fit on the b-basis (resident).
        cb = _als_solve_gpu(cp, Bb, yc)
        g = Bb @ cb
        ca = None
        for _ in range(max(1, int(iters))):
            g_norm = g / (float(cp.std(g)) + 1e-12)            # scalar D2H (std), resident vector op
            ca = _als_solve_gpu(cp, Ba * g_norm[:, None], yc)
            f = Ba @ ca
            f_norm = f / (float(cp.std(f)) + 1e-12)            # scalar D2H (std), resident vector op
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
    except (np.linalg.LinAlgError, ValueError):
        return None, None
