"""BATCHED, born-on-device GPU path for the wavelet leg-rank MI (replatform step 1, 2026-06-25).

PARALLEL implementation -- this module is SELF-CONTAINED and imported by NOTHING in the production path
yet; it exists so the batched device design can be built + validated against the per-leg CPU path
(``_wavelet_basis_fe._binned_mi`` / ``_select_wavelet_legs``) WITHOUT touching/breaking the primary path.
Once ``test_wavelet_batched_mi_parity`` pins selection-equivalence it gets wired under MLFRAME_FE_GPU_STRICT.

WHY batched: the per-leg path calls ``_binned_mi`` ~5x per leg x dozens of legs x ``n_perm`` shuffles =
thousands of tiny calls, each a separate cp.unique (5-10 cub launches) + H2D -> 118k launches / 5.8k H2D
in one fit (nsys-measured, launch-bound). The batched primitive below scores ALL K code-columns against y
in ONE device workload: a single ``cp.bincount`` over the flat index ``k*(Kx*Ky) + cx*Ky + cy`` yields the
(K, Kx, Ky) joint histogram for every column at once -> MI(col_k; y) for all k with NO per-column launch
and ONE H2D of the (n, K) code matrix. This is the same "ONE batched workload" discipline as
``pooled_gain_floor_perms_cupy``.

MI is partition-based, so value-order codes are fine -> selection-equivalent to the CPU plug-in MI
(fp reduction order ~1e-15), per the FE-perf bar (selection, not bit-identity).
"""
from __future__ import annotations

import numpy as np


def batched_binned_mi_gpu(code_cols: np.ndarray, y_codes: np.ndarray, kx_per_col=None, ky: int = 0):
    """Plug-in MI(col_k; y) in nats for EVERY column of ``code_cols`` in ONE device workload.

    Parameters
    ----------
    code_cols : (n, K) int array  -- each column is a non-negative integer bin code (the leg/joint codes).
    y_codes   : (n,) int array    -- non-negative class codes for the target.
    kx_per_col: optional (K,) ints -- per-column cardinality; default = per-column max+1.
    ky        : optional int       -- target cardinality; default = y_codes.max()+1.

    Returns a host (K,) float64 array of MI values. Uses ONE ``cp.bincount`` over the padded flat joint
    index (k, cx, cy) so there is no per-column kernel launch. Raises on cupy error (caller falls back).
    """
    import cupy as cp

    C = cp.asarray(np.ascontiguousarray(code_cols).astype(np.int64))
    if C.ndim == 1:
        C = C[:, None]
    n, K = int(C.shape[0]), int(C.shape[1])
    y = cp.asarray(np.ascontiguousarray(y_codes).astype(np.int64).ravel())
    Ky = int(ky) if ky > 0 else int(y.max()) + 1
    # Per-column cardinality -> a single padded Kx so the flat index layout is uniform across columns.
    if kx_per_col is not None:
        Kx = int(np.max(np.asarray(kx_per_col)))
    else:
        Kx = int(C.max()) + 1
    Kx = max(Kx, 1)
    inv_n = 1.0 / float(n)
    # flat = k*(Kx*Ky) + cx*Ky + cy ; ONE bincount -> (K, Kx, Ky) joint counts for all columns at once.
    col_off = (cp.arange(K, dtype=cp.int64) * (Kx * Ky))[None, :]          # (1, K)
    flat = C * Ky + y[:, None] + col_off                                  # (n, K)
    counts = cp.bincount(flat.ravel(), minlength=K * Kx * Ky).astype(cp.float64).reshape(K, Kx, Ky)
    pij = counts * inv_n                                                  # joint p(x,y) per column
    pi = pij.sum(axis=2)                                                  # (K, Kx) p(x)
    pj = pij.sum(axis=1)                                                  # (K, Ky) p(y)
    # MI = sum pij * log(pij / (pi*pj)) over occupied cells.
    denom = pi[:, :, None] * pj[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = cp.where((pij > 0) & (denom > 0), pij / denom, 1.0)
        mi = cp.sum(cp.where(pij > 0, pij * cp.log(ratio), 0.0), axis=(1, 2))
    return cp.asnumpy(cp.maximum(mi, 0.0))
