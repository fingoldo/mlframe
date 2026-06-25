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


def batched_binned_mi_gpu(code_cols: np.ndarray, y_codes: np.ndarray, kx_per_col: np.ndarray | None = None, ky: int = 0) -> np.ndarray:
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


def _dense_leg_codes(leg_sub: np.ndarray) -> "tuple[np.ndarray, int]":
    """Densify a Haar-leg subset ({-1,0,+1}-valued) to bin codes. Since MI is partition-based, the exact
    code labels are irrelevant -- only the partition matters -- so map ``leg -> leg + 1`` ({0,1,2}) with a
    fixed cardinality 3 instead of ``searchsorted(unique(leg), leg)``. This drops the per-leg host
    ``np.unique`` sort + ``searchsorted`` (dozens of legs/fit); an absent value just leaves an empty bin
    that contributes 0 to MI, so the result is selection-equivalent to the value-rank coding _binned_mi
    uses (pinned by test_wavelet_batched_mi_parity)."""
    codes = (np.asarray(leg_sub) + 1).astype(np.int64)
    return codes, 3


def select_wavelet_legs_batched(x: np.ndarray, y: np.ndarray, lo: float, span: float, *, max_scale: int, max_legs: int, scale_sigma: float) -> list:
    """BATCHED born-on-device twin of ``_wavelet_basis_fe._select_wavelet_legs``: returns the SAME admitted
    ``(j, k)`` legs, but scores every candidate leg's train + held-out MI in TWO batched device workloads
    (one per split) instead of ~2 per-leg ``_binned_mi`` calls. Selection-equivalent (plug-in MI is
    partition-based). Reuses the primary module's leg builder + gate constants read-only (no mutation)."""
    from ._wavelet_basis_fe import (
        _WAVELET_MIN_HALF_ROWS,
        _WAVELET_MIN_HELDOUT_MI,
        _WAVELET_MIN_ROWS,
        _bin_y_codes,
        _dyadic_haar_leg,
    )

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    n = x.size
    if n != y.size or n < _WAVELET_MIN_ROWS:
        return []
    if span <= 1e-12 or float(np.std(x)) < 1e-12:
        return []
    z = np.clip((x - lo) / span, 0.0, 1.0)
    idx = np.arange(n)
    va = (idx % 3) == 0
    tr = ~va
    if int(tr.sum()) < 64 or int(va.sum()) < 32:
        return []
    yb_tr = _bin_y_codes(y[tr])
    yb_va = _bin_y_codes(y[va])

    metas: list[tuple] = []          # (j, k)
    tr_cols: list[np.ndarray] = []
    va_cols: list[np.ndarray] = []
    tr_kx: list[int] = []
    va_kx: list[int] = []
    for j in range(int(max_scale) + 1):
        for k in range(2 ** j):
            leg = _dyadic_haar_leg(z, j, k)
            if int(np.count_nonzero(leg > 0)) < _WAVELET_MIN_HALF_ROWS or int(np.count_nonzero(leg < 0)) < _WAVELET_MIN_HALF_ROWS:
                continue
            c_tr, k_tr = _dense_leg_codes(leg[tr])
            c_va, k_va = _dense_leg_codes(leg[va])
            metas.append((int(j), int(k)))
            tr_cols.append(c_tr)
            va_cols.append(c_va)
            tr_kx.append(k_tr)
            va_kx.append(k_va)
    if not metas:
        return []

    tr_mat = np.stack(tr_cols, axis=1)   # (n_tr, K)
    va_mat = np.stack(va_cols, axis=1)   # (n_va, K)
    ky_tr = int(np.asarray(yb_tr).max()) + 1
    ky_va = int(np.asarray(yb_va).max()) + 1
    mi_tr = batched_binned_mi_gpu(tr_mat, np.asarray(yb_tr), kx_per_col=tr_kx, ky=ky_tr)
    mi_va = batched_binned_mi_gpu(va_mat, np.asarray(yb_va), kx_per_col=va_kx, ky=ky_va)

    heldout = np.asarray(mi_va, dtype=np.float64)
    if heldout.size >= 4:
        med = float(np.median(heldout))
        mad = float(np.median(np.abs(heldout - med)))
        floor = med + scale_sigma * 1.4826 * mad
    else:
        floor = 0.0
    floor = max(floor, _WAVELET_MIN_HELDOUT_MI)
    cand = [(float(mi_tr[i]), float(mi_va[i]), metas[i][0], metas[i][1]) for i in range(len(metas))]
    admitted = [c for c in cand if c[1] >= floor]
    if not admitted:
        return []
    admitted.sort(key=lambda c: c[0], reverse=True)
    return [(int(c[2]), int(c[3])) for c in admitted[: int(max_legs)]]
