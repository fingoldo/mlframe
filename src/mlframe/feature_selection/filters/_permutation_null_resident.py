"""Resident-GPU twin of the maxT permutation-null pooled gain floor (iter16, 2026-06-23).

``_permutation_null._pooled_gain_floor_perms_njit`` is a ``(nperm x ncand x n)`` histogram-MI loop: for
every target shuffle and every candidate it rebuilds the joint H(x, y_perm) contingency table and tracks
the per-shuffle MAX corrected marginal MI over the pool. The X-codes, the per-candidate marginal entropy
``h_x`` and the Miller-Madow bias are permutation-INVARIANT (precomputed once on host, cheap); only the
joint entropy changes per shuffle. This is the textbook resident pattern -- ONE batched workload, no
per-pair launches (the trap that sank iter13's per-pair ``score_pair_combos``): upload the invariant
operands ONCE, run the whole (perm x cand) histogram + MI + per-shuffle-max reduction on the device, and
D2H only the (nperm,) per-shuffle maxes.

GATE: engages only where a per-host KTC crossover (``_permutation_null_resident_ktc.permnull_use_resident``)
MEASURED the resident path faster than the njit kernel; otherwise the caller stays on the exact njit. On a
small card (GTX 1050 Ti, 6 SMs) the crossover favours the resident path only at LARGE (nperm * ncand * n) --
the production tabular F2 floor (ncand<=9, nperm=200, n=100k) may stay CPU there, which is correct.

SELECTION-EQUIVALENCE: the per-cell ``-p*log(p)`` accumulation differs from the njit kernel ONLY in FP
reduction order (the njit loops cells in code-ascending order; cupy's segmented reduction sums in a
hardware-defined order). Both are exact plug-in entropies of the SAME integer contingency table, so the
per-shuffle max corrected MI matches to fp64 round-off (~1e-15), far below the floor's selection scale
(``np.quantile`` of the maxes). The host caller owns the final ``np.quantile`` so the floor value is
computed identically.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Working-set budget so the largest intermediate -- the (chunk_cand * chunk_perm * n) int64 joint-code
# array -- stays well inside a small (4GB) card. We size the (cand, perm) tile so that
# ``cc * pb * n * 8`` bytes is bounded; below ~96M cells -> ~768MB for the joint plus a same-size bincount
# input, comfortably under 4GB with headroom for the operand uploads. Candidates and perms are BOTH tiled.
_PERMNULL_TILE_CELLS = 24_000_000  # cc * pb * n upper bound (int64 cells); ~192MB per (joint) buffer


def pooled_gain_floor_perms_cupy(scaled_flat: np.ndarray, offsets: np.ndarray, joint_card: np.ndarray, h_x: np.ndarray, mm_bias: np.ndarray, h_y: float, y_perms: Any, inv_n: float) -> np.ndarray:
    """Resident cupy twin of :func:`_permutation_null._pooled_gain_floor_perms_njit`. Same signature; returns
    a host ``(nperm,)`` float64 array of the per-shuffle MAX corrected marginal MI over the candidate pool.

    All heavy arrays are uploaded ONCE and the whole (perm x cand) histogram + MI + per-shuffle-max runs on
    the device. ``joint_card`` may be ragged across candidates, so candidates are processed in chunks of a
    fixed working size; within a chunk every candidate's joint table is padded to the chunk's max cardinality
    (empty padding cells contribute 0 to the entropy, exactly like the njit ``if c>0`` guard)."""
    import cupy as cp
    import cupyx

    nperm = int(y_perms.shape[0])
    n = int(y_perms.shape[1])
    ncand = int(offsets.shape[0] - 1)
    if nperm == 0 or ncand == 0:
        return np.zeros(nperm, dtype=np.float64)

    d_scaled = cp.asarray(scaled_flat)  # (ncand*n,) int (>=0 joint base codes)
    d_yperms = cp.asarray(y_perms)  # (nperm, n) int (>=0 class codes)
    d_off = cp.asarray(offsets, dtype=cp.int64)  # (ncand+1,) int64 flat segment offsets
    jc_host = np.asarray(joint_card, dtype=np.int64)  # (ncand,) per-candidate joint cardinality (host)
    d_hx = cp.asarray(h_x, dtype=cp.float64)  # (ncand,) marginal H(x)
    d_mm = cp.asarray(mm_bias, dtype=cp.float64)  # (ncand,) Miller-Madow bias
    h_y = float(h_y)
    inv_n = float(inv_n)

    # Per-shuffle running max corrected MI over the pool (starts at 0.0 -> the njit ``best=0.0`` floor; MI is
    # clamped implicitly by the running max never dropping below 0, matching ``if mi>best``).
    d_best = cp.zeros(nperm, dtype=cp.float64)

    # Tile both candidates and perms so the (cc * pb * n) joint-code intermediate fits a small card.
    cand_chunk = max(1, min(ncand, _PERMNULL_TILE_CELLS // max(1, n)))
    for c0 in range(0, ncand, cand_chunk):
        c1 = min(c0 + cand_chunk, ncand)
        cc = c1 - c0
        # max_jc drives host-side shape/arange sizing, so read it from the HOST joint_card slice -- a device
        # ``cp.max(...).item()`` here was a tiny per-chunk D2H scalar pull (latency, not bandwidth).
        max_jc = int(jc_host[c0:c1].max()) if c1 > c0 else 0
        if max_jc <= 0:
            continue
        # Gather this chunk's X-codes as (cc, n). Segments are contiguous length-n runs in scaled_flat
        # (offsets[j] = j*n in the order-1 caller, but read via offsets to stay faithful to any layout).
        starts = d_off[c0:c1]
        row_idx = starts[:, None] + cp.arange(n, dtype=cp.int64)[None, :]
        x_codes = d_scaled[row_idx].astype(cp.int64)  # (cc, n)
        d_hx_c = d_hx[c0:c1][:, None]
        d_mm_c = d_mm[c0:c1][:, None]

        perm_chunk = max(1, _PERMNULL_TILE_CELLS // max(1, cc * n))
        for p0 in range(0, nperm, perm_chunk):
            p1 = min(p0 + perm_chunk, nperm)
            pb = p1 - p0
            yp = d_yperms[p0:p1]  # (pb, n)
            # Joint flat index for the batched bincount: ((cand_local*pb + perm_local)*max_jc) + (x+y_perm).
            # Padding cells (code >= jc[cand]) cannot occur: x in [0, nbins_x*nbins_y) and y in [0, nbins_y)
            # so x+y < jc by construction; padding to max_jc only leaves trailing zero-count cells per (cand,
            # perm) slab, which contribute 0 to the entropy exactly like the njit ``if c>0`` short-circuit.
            joint = (x_codes[:, None, :] + yp[None, :, :]).astype(cp.int64)  # (cc, pb, n)
            base = (cp.arange(cc, dtype=cp.int64)[:, None] * pb + cp.arange(pb, dtype=cp.int64)[None, :]) * max_jc  # (cc, pb)
            flat = (base[:, :, None] + joint).ravel()  # (cc*pb*n,)
            # COALESCE (2026-06-24): cp.bincount internally pulls the input max (+ a sizing scalar) to host on
            # EVERY call -- 2 tiny per-(cand,perm)-tile D2H scalar pulls that, summed over the floor's tile loop,
            # are the canonical-fit's dominant tiny-D2H source (~2400 of ~2810 sub-4KB .get()s, F2 100k). The
            # output extent is KNOWN here (cc*pb*max_jc, every flat code < it by construction), so scatter_add
            # into a pre-zeroed buffer needs no max-pull and is sync-free (0 tiny D2H). int32 counts hold the
            # per-cell totals (bounded by n) exactly; BIT-IDENTICAL integer histogram to bincount (verified
            # array_equal). The downstream entropy upcasts to float64 just as the bincount path did.
            counts_i = cp.zeros(cc * pb * max_jc, dtype=cp.int32)
            cupyx.scatter_add(counts_i, flat, cp.int32(1))

            # Plug-in joint entropy H(x, y_perm) per (cand, perm): -sum p*log(p), p=count/n, 0-count -> 0.
            # The per-(cand,perm)-row entropy over the max_jc joint cells is exactly the fused rows-entropy
            # kernel (one block per row, shared-mem reduction) -> ONE launch, replacing the counts*inv_n +
            # where + log + where + sum chain. Same float64 plug-in entropy -> selection-equivalent.
            from ._fe_batched_mi import _rows_entropy_and_k

            h_xy_flat, _ = _rows_entropy_and_k(counts_i.reshape(cc * pb, max_jc), inv_n)  # (cc*pb,)
            h_xy = h_xy_flat.reshape(cc, pb)  # (cc, pb)
            mi = (d_hx_c + h_y) - h_xy - d_mm_c  # (cc, pb)
            d_best[p0:p1] = cp.maximum(d_best[p0:p1], cp.max(mi, axis=0))
            del joint, flat, counts_i, h_xy_flat, h_xy, mi

    return np.asarray(cp.asnumpy(d_best))


def gen_target_shuffles_cupy(y_codes: np.ndarray, nperm: int, dtype: type, random_seed: Optional[int]) -> object:
    """Generate the ``(nperm, n)`` target-shuffle matrix ON the device (argsort of random keys) and return it
    as a CUPY array, fed DIRECTLY into :func:`pooled_gain_floor_perms_cupy` (whose ``cp.asarray`` is a no-op on
    an already-device array). So the permutation matrix is BORN on the GPU: no host Fisher-Yates generation and
    no H2D upload of the ``(nperm, n)`` matrix (only the small ``(n,)`` target codes go up once). This is the
    residency-fair shuffle-gen the campaign left undone: on a small-VRAM card the ``(nperm, n)`` key+order
    buffers OOM at large n, so it is KTC-gated (``shufflegen_use_gpu``) and only fires where a per-host sweep
    measured it faster -- a big-VRAM host -- with the host njit/numpy gen as the default + fallback.

    Each row is a uniform permutation of ``y_codes`` (argsort of i.i.d. keys), a DIFFERENT stream than the
    host backends -> the floor is statistically equivalent, not byte-identical (same contract as the existing
    njit-gen crossover). Returns ``None`` on any cupy error / OOM so the caller falls back to the host gen."""
    try:
        import cupy as cp

        n = int(y_codes.shape[0])
        nperm = int(nperm)
        seed = 0x9E3779B9 if random_seed is None else (int(random_seed) & 0x7FFFFFFF)
        _rng = cp.random.default_rng(seed)
        d_y = cp.asarray(np.ascontiguousarray(y_codes).astype(dtype, copy=False))  # one tiny (n,) upload
        keys = _rng.random((nperm, n), dtype=cp.float32)  # (nperm, n) i.i.d. sort keys, on device
        order = cp.argsort(keys, axis=1)  # per-row permutation indices
        del keys
        out = d_y[order]  # (nperm, n) gathered shuffles, on device
        del order
        return out
    except Exception:
        return None
