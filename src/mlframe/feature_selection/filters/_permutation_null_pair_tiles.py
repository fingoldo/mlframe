"""Stages of ``pooled_pair_permutation_null_joint_mi_floor_cupy``."""

from __future__ import annotations

# --- imports (managed) ---
# --- end imports ---


def _pooled_pair_null_perm_tiles(nperm, perm_chunk, y_perms, cp, n_pairs, per_pair_extent, cls_x, n_classes_y, inv_n, h_x, h_y, d_mm, d_maxes):
    """Stage of ``pooled_pair_permutation_null_joint_mi_floor_cupy``."""
    # Flat joint index per (perm, pair, sample): the pad-safe layout mirrors the order-1 resident floor.
    # base_pair[pair] = pair * per_pair_extent gives each pair its own contiguous histogram block; within
    # a pair the cell is cls_x * n_classes_y + y_perm (the exact 1-D collapse batch_pair_mi_cupy uses).
    # Per-perm slabs are offset by (perm_local * n_pairs * per_pair_extent). Padding cells (joint code in
    # [joint_card[pair], max_joint)) cannot occur (x < joint_card by construction) -> trailing zero-count
    # cells per (perm, pair) block contribute 0 to the entropy, exactly like the njit ``if c>0`` guard.
    base_pair = (cp.arange(n_pairs, dtype=cp.int64) * per_pair_extent)[None, :, None]  # (1, n_pairs, 1)
    for p0 in range(0, nperm, perm_chunk):
        p1 = min(p0 + perm_chunk, nperm)
        pb = p1 - p0
        yp = y_perms[p0:p1]  # (pb, n)
        base_perm = (cp.arange(pb, dtype=cp.int64) * (n_pairs * per_pair_extent))[:, None, None]  # (pb,1,1)
        joint = cls_x[None, :, :] * n_classes_y + yp[:, None, :]  # (pb, n_pairs, n) cell within pair block
        flat = (base_perm + base_pair + joint).ravel()  # (pb*n_pairs*n,)
        counts = cp.bincount(flat, minlength=pb * n_pairs * per_pair_extent)[: pb * n_pairs * per_pair_extent]
        del joint, flat
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        counts = counts.reshape(pb * n_pairs, per_pair_extent).astype(cp.float64)
        # Plug-in joint entropy H(x, y_perm) per (perm, pair): -sum p log p, p = count / n.
        p_xy = counts * inv_n
        nz = p_xy > 0
        h_xy = -cp.sum(cp.where(nz, p_xy * cp.log(cp.where(nz, p_xy, 1.0)), 0.0), axis=1)
        h_xy = h_xy.reshape(pb, n_pairs)  # (pb, n_pairs)
        mi = (h_x.T + h_y) - h_xy  # (pb, n_pairs)  broadcast h_x (1,n_pairs)
        if d_mm is not None:
            mi = mi - d_mm.T  # subtract per-pair MM bias (1, n_pairs)
        d_maxes[p0:p1] = cp.max(mi, axis=1)
        del counts, p_xy, h_xy, mi
