"""Stages of ``score_pair_combos_table_resident``."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
from mlframe.feature_selection.filters._usability_pool_resident import _write_combo_chunk_mi
# --- end imports ---


def _score_one_pair_combos(operands, p, resident_operand, f32, n, cp, nu, _gpu_apply_unary, ua_codes_l, ub_codes_l, nc, combo_chunk, col_major, cp_dt, fused_gen, threads, ua_idx_d, ub_idx_d, bop_d, nbins, ky_w, h_y, k_y, d_y, d_qs, mi_table_d):
    """Stage of ``score_pair_combos_table_resident``."""
    x1, x2 = operands[p]
    # Content-hash-keyed resident cache (cProfile-driven): a raw operand column
    # commonly recurs across MANY pairs (var A paired with B, C, D, ... all share A) - was
    # re-uploaded from host on every occurrence despite unchanged content. resident_operand
    # (already used pervasively elsewhere in the FE-resident stack) uploads each DISTINCT
    # content ONCE per fit and returns the cached device buffer on every repeat. Read-only
    # inputs -> sharing the buffer across pairs is safe; values are byte-identical either way.
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    d_x1 = resident_operand(x1, "usability_pool_operand_f32" if f32 else "usability_pool_operand", dtype=np.float32 if f32 else np.float64)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    d_x2 = resident_operand(x2, "usability_pool_operand_f32" if f32 else "usability_pool_operand", dtype=np.float32 if f32 else np.float64)
    # The operands are host arrays that were just uploaded: their nanmin shifts are a numpy reduction of a column already in memory, instead of two
    # device nanmin kernels plus a blocking read per pair (cupy's nanmin was ~15 ms a call here, almost all of it waiting on the queued work).
    xmin_a, xmin_b = (float(np.nanmin(x1)), float(np.nanmin(x2))) if n else (0.0, 0.0)
    # (nu, n) column-major stacks: each operand's nu unaries applied ONCE (reused across all combos).
    ua_stack = cp.ascontiguousarray(cp.stack([_gpu_apply_unary(d_x1, ua_codes_l[ia], xmin_a) for ia in range(nu)]))
    ub_stack = cp.ascontiguousarray(cp.stack([_gpu_apply_unary(d_x2, ub_codes_l[ib], xmin_b) for ib in range(nu)]))
    base = p * nc
    for c0 in range(0, nc, combo_chunk):
        c1 = min(c0 + combo_chunk, nc)
        kk = c1 - c0
        cand = cp.empty((kk, n) if col_major else (n, kk), dtype=cp_dt)  # (kk, n) column-major or (n, kk) row-major
        total = n * kk
        fused_gen(((total + threads - 1) // threads,), (threads,),
                  (ua_stack, ub_stack, ua_idx_d[c0:c1], ub_idx_d[c0:c1], bop_d[c0:c1],
                   np.int64(n), np.int32(kk), cand))
        # cp.var (two-pass, one fused reduction) replaces the explicit mean + (cand*cand).mean - mean^2,
        # which allocated a full (n,kk) ``cand*cand`` temp and ran two reduction passes. Measured 200k/178:
        # 9.98ms vs 13.80ms (-3.8ms/chunk) and the std<=1e-9 sentinel mask (``> 1e-18``) is BIT-IDENTICAL
        # (verified ((cand*cand).mean-mean^2 > 1e-18) == (cp.var > 1e-18) over the full block).
        # Near-constant mask, relative to each combo's largest |value| exactly as the njit table kernel does (an absolute 1e-18
        # variance floor rejected genuinely tiny-scale combos); D2H deferred (see below).
        from mlframe.feature_selection.filters._usability_njit_pool import _USABILITY_DEGENERATE_REL_TOL

        _axis = 1 if col_major else 0
        live_d = cand.var(axis=_axis) > (_USABILITY_DEGENERATE_REL_TOL * cp.abs(cand).max(axis=_axis)) ** 2
        _write_combo_chunk_mi(cand, live_d, col_major, nbins, ky_w, h_y, k_y, d_y, d_qs, n, mi_table_d, base + c0, base + c1)
        del cand
    del d_x1, d_x2, ua_stack, ub_stack
