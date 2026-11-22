"""Stages of ``pooled_pair_permutation_null_joint_mi_floor_cupy``."""

from __future__ import annotations

# --- imports (managed) ---
# --- end imports ---


def _pooled_pair_null_perm_tiles(nperm, perm_chunk, y_perms, cp, n_pairs, per_pair_extent, cls_x, n_classes_y, inv_n, h_x, h_y, d_mm, d_maxes, pair_chunk=None):
    """Stage of ``pooled_pair_permutation_null_joint_mi_floor_cupy``.

    ``pair_chunk``: pairs per block of a permutation tile (default every pair at once). When one permutation's histogram of all pairs (``n_pairs * per_pair_extent`` cells) is itself larger than the tile budget, the
    pairs are taken in blocks and the per-permutation maximum is carried across them (the maximum over pairs is the maximum of the block maxima, so the result is the same)."""
    # Flat joint index per (perm, pair, sample): the pad-safe layout mirrors the order-1 resident floor.
    # base_pair[pair] = pair * per_pair_extent gives each pair its own contiguous histogram block; within
    # a pair the cell is cls_x * n_classes_y + y_perm (the exact 1-D collapse batch_pair_mi_cupy uses).
    # Per-perm slabs are offset by (perm_local * n_pairs * per_pair_extent). Padding cells (joint code in
    # [joint_card[pair], max_joint)) cannot occur (x < joint_card by construction) -> trailing zero-count
    # cells per (perm, pair) block contribute 0 to the entropy, exactly like the njit ``if c>0`` guard.
    block = n_pairs if not pair_chunk else max(1, min(int(pair_chunk), n_pairs))
    for p0 in range(0, nperm, perm_chunk):
        p1 = min(p0 + perm_chunk, nperm)
        pb = p1 - p0
        yp = y_perms[p0:p1]  # (pb, n)
        best = None
        for q0 in range(0, n_pairs, block):
            q1 = min(q0 + block, n_pairs)
            nq = q1 - q0
            base_pair = (cp.arange(nq, dtype=cp.int64) * per_pair_extent)[None, :, None]  # (1, nq, 1)
            base_perm = (cp.arange(pb, dtype=cp.int64) * (nq * per_pair_extent))[:, None, None]  # (pb,1,1)
            joint = cls_x[None, q0:q1, :] * n_classes_y + yp[:, None, :]  # (pb, nq, n) cell within pair block
            flat = (base_perm + base_pair + joint).ravel()  # (pb*nq*n,)
            counts = cp.bincount(flat, minlength=pb * nq * per_pair_extent)[: pb * nq * per_pair_extent]
            del joint, flat
            # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
            counts = counts.reshape(pb * nq, per_pair_extent).astype(cp.float64)
            # Plug-in joint entropy H(x, y_perm) per (perm, pair): -sum p log p, p = count / n.
            p_xy = counts * inv_n
            nz = p_xy > 0
            h_xy = -cp.sum(cp.where(nz, p_xy * cp.log(cp.where(nz, p_xy, 1.0)), 0.0), axis=1)
            h_xy = h_xy.reshape(pb, nq)  # (pb, nq)
            mi = (h_x[q0:q1].T + h_y) - h_xy  # (pb, nq)  broadcast h_x (1, nq)
            if d_mm is not None:
                mi = mi - d_mm[q0:q1].T  # subtract per-pair MM bias (1, nq)
            top = cp.max(mi, axis=1)
            best = top if best is None else cp.maximum(best, top)
            del counts, p_xy, h_xy, mi
        d_maxes[p0:p1] = best
