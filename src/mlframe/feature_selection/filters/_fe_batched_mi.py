"""Batched born-on-device MI / CMI for the FE candidate pool (replatform step 3, 2026-06-25).

PARALLEL / gated module -- imported by nothing in production yet; it lets the batched device CMI be
built + parity-validated against the per-call CPU path (``_mi_greedy_cmi_fe._cmi_from_binned``) WITHOUT
touching the primary path, then wired under MLFRAME_FE_GPU_STRICT.

The per-call cp.unique CMI ports (cmi_from_binned / fixed_yz) are launch-bound: nsys measured 118k kernel
launches + 5.8k H2D in one F2 1M fit because CMI is called per-candidate-per-permutation, each a separate
cp.unique (5-10 cub launches) + H2D. ``batched_cmi_gpu`` scores CMI(x_k; y | z) for EVERY candidate column
in ONE workload: the y/z-only terms (H(Z), H(Y,Z)) are computed once, then TWO cp.bincount passes over the
flat (k, cx, c*) joint index give H(X_k,Z) and H(X_k,Y,Z) for all k at once -> no per-candidate launch,
one H2D of the (n,K) candidate code matrix. Same Miller-Madow plug-in CMI as the CPU path; partition-based
-> value-order codes are fine -> selection-equivalent (fp reduction order ~1e-15).
"""
from __future__ import annotations

import numpy as np


def _rows_entropy_and_k(counts, inv_n):
    """Per-row (per-candidate) plug-in entropy + occupied-cell count from a (K, M) device count matrix."""
    import cupy as cp

    p = counts * inv_n
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -cp.sum(cp.where(counts > 0, p * cp.log(cp.where(counts > 0, p, 1.0)), 0.0), axis=1)
    k = cp.sum(counts > 0, axis=1)
    return h, k


def batched_quantile_bin_gpu(x_cols, nbins: int):
    """Born-on-device equi-frequency binning of an (n,K) float matrix -> (n,K) int codes, RESIDENT on GPU.

    Device twin of ``_mi_greedy_cmi_fe._quantile_bin`` applied per column. ONE batched ``cp.percentile``
    (axis=0) computes all K columns' edges in a single device sort -- replacing K host ``np.quantile``
    (introselect partition) calls -- then per-column ``cp.searchsorted`` on the (deduped) inner edges. The
    per-column ``cp.unique`` is over only (nbins+1) edge values (negligible), so the n-sized sort stays
    batched. Returns a cupy int64 (n,K) array (kept on device to feed ``batched_cmi_gpu`` without a code
    H2D). Selection-equivalent to the host binning: same equi-frequency partition (value-edge, rank-based).
    Assumes all-finite input (the production nan-scrubbed case); the caller falls back to the host path
    otherwise.
    """
    import cupy as cp

    Xd = x_cols if isinstance(x_cols, cp.ndarray) else cp.asarray(np.ascontiguousarray(np.asarray(x_cols, dtype=np.float64)))
    if Xd.ndim == 1:
        Xd = Xd[:, None]
    Xd = Xd.astype(cp.float64, copy=False)
    n, K = int(Xd.shape[0]), int(Xd.shape[1])
    qs = cp.linspace(0.0, 100.0, nbins + 1)
    edges_all = cp.percentile(Xd, qs, axis=0)        # (nbins+1, K) -- one batched device sort
    codes = cp.zeros((n, K), dtype=cp.int64)
    for k in range(K):
        edges = cp.unique(edges_all[:, k])           # dedupe equi-freq edges (mirror np.unique on host)
        if int(edges.size) <= 2:
            if int(edges.size) == 2:
                codes[:, k] = (Xd[:, k] >= edges[1]).astype(cp.int64)
            continue
        codes[:, k] = cp.searchsorted(edges[1:-1], Xd[:, k], side="right").astype(cp.int64)
    return codes


def batched_cmi_gpu(x_cols, y: np.ndarray, z=None) -> np.ndarray:
    """Miller-Madow plug-in CMI(x_k; y | z) in nats for EVERY column of ``x_cols``, in ONE device workload.

    ``x_cols`` (n,K) int codes -- a host ndarray OR an already-resident cupy array (born-on-device codes
    from ``batched_quantile_bin_gpu``, no code H2D); ``y`` (n,) int codes; ``z`` (n,) int codes or None
    (marginal MI). Returns a host (K,) float64 array. Matches ``_mi_greedy_cmi_fe._cmi_from_binned`` per
    column (selection-equivalent).
    """
    import cupy as cp

    if isinstance(x_cols, cp.ndarray):
        X = x_cols.astype(cp.int64, copy=False)
    else:
        X = cp.asarray(np.ascontiguousarray(x_cols).astype(np.int64))
    if X.ndim == 1:
        X = X[:, None]
    n, K = int(X.shape[0]), int(X.shape[1])
    dy = cp.asarray(np.ascontiguousarray(y).astype(np.int64).ravel())
    nf = float(max(1, n))
    inv_n = 1.0 / nf
    Kx = int(X.max()) + 1 if X.size else 1
    col_off = (cp.arange(K, dtype=cp.int64))[None, :]

    if z is None or (hasattr(z, "size") and np.asarray(z).size == 0):
        # Marginal MI(x_k; y), MM-corrected:  H(x)+H(y)-H(x,y) - (k_x+k_y-k_xy-1)/2n
        Ky = int(dy.max()) + 1 if dy.size else 1
        cnt_xy = cp.bincount((X * Ky + dy[:, None] + col_off * (Kx * Ky)).ravel(),
                             minlength=K * Kx * Ky).astype(cp.float64).reshape(K, Kx * Ky)
        h_xy, k_xy = _rows_entropy_and_k(cnt_xy, inv_n)
        cnt_x = cp.bincount((X + col_off * Kx).ravel(), minlength=K * Kx).astype(cp.float64).reshape(K, Kx)
        h_x, k_x = _rows_entropy_and_k(cnt_x, inv_n)
        yc = cp.bincount(dy, minlength=Ky).astype(cp.float64)
        py = yc[yc > 0] * inv_n
        h_y = float(-(py * cp.log(py)).sum())
        k_y = int((yc > 0).sum())
        mi = h_x + h_y - h_xy
        bias = (k_x + k_y - k_xy - 1) / (2.0 * nf)
        return cp.asnumpy(cp.maximum(mi - bias, 0.0))

    dz = cp.asarray(np.ascontiguousarray(z).astype(np.int64).ravel())
    Kz = int(dz.max()) + 1 if dz.size else 1
    # shared y/z terms (column-invariant)
    zc = cp.bincount(dz, minlength=Kz).astype(cp.float64)
    pz = zc[zc > 0] * inv_n
    h_z = float(-(pz * cp.log(pz)).sum())
    k_z = int((zc > 0).sum())
    yz = dy * Kz + dz                      # dense (y,z) code
    yzc = cp.bincount(yz, minlength=int(yz.max()) + 1).astype(cp.float64)
    pyz = yzc[yzc > 0] * inv_n
    h_yz = float(-(pyz * cp.log(pyz)).sum())
    k_yz = int((yzc > 0).sum())
    Kyz = int(yz.max()) + 1

    # H(x_k, z) for all k: flat = k*(Kx*Kz) + x*Kz + z
    cnt_xz = cp.bincount((X * Kz + dz[:, None] + col_off * (Kx * Kz)).ravel(),
                         minlength=K * Kx * Kz).astype(cp.float64).reshape(K, Kx * Kz)
    h_xz, k_xz = _rows_entropy_and_k(cnt_xz, inv_n)
    # H(x_k, y, z) for all k: flat = k*(Kx*Kyz) + x*Kyz + yz
    cnt_xyz = cp.bincount((X * Kyz + yz[:, None] + col_off * (Kx * Kyz)).ravel(),
                          minlength=K * Kx * Kyz).astype(cp.float64).reshape(K, Kx * Kyz)
    h_xyz, k_xyz = _rows_entropy_and_k(cnt_xyz, inv_n)

    cmi = h_xz + h_yz - h_z - h_xyz
    bias = (k_xyz + k_z - k_xz - k_yz) / (2.0 * nf)
    return cp.asnumpy(cp.maximum(cmi - bias, 0.0))
