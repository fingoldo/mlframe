"""GPU-RESIDENT batched pair-combo MI TABLE for the usability candidate pool (iter17, 2026-06-23).

THE TRAP (iter13/iter16, documented in _usability_njit_pool.py:737-749): forcing the per-PAIR
``score_pair_combos`` onto its cupy twin (``_pair_combo_mi_cupy``) is a 3x LOSS end-to-end -- F2 100k
wall 34.8s -> 102.5s. Each pair enumerates a SMALL ``nu x nu x nb`` combo grid, so per-pair invocation
pays a fresh operand H2D + many tiny launches that swamp the ~1.0s ``_pair_combo_mi_njit_table_parallel``
CPU kernel.

THE RESIDENT FIX (this module): the pair-combo MI TABLE computation (a pure function of each pair's two
operand columns + the shared y) is cleanly SEPARABLE from the per-pair retention/diversity BOOKKEEPING in
``build_usability_candidate_pool`` -- the bookkeeping loop only reads ``mis[j]`` per pair. So we compute the
FULL ``(npairs, nc)`` table in ONE resident pass: upload the shared y ONCE, upload each pair's two operands,
build the combo columns on device, accumulate them into a VRAM-BOUNDED chunk buffer, and bin + MI-score each
full chunk in one batched call -- D2H ONLY the table. The retention loop then consumes its per-pair slice
UNCHANGED -> byte-identical retain/drop/diversity decisions and order. (The full ``(npairs*nc, n)`` stack
would overflow a small card -- npairs=8 x nc=1734 x n=50k x 8B = 5.5GB > a 4GB GTX 1050 Ti -- so the chunk
buffer is sized to a fraction of FREE VRAM; the launch-amortisation win over the per-PAIR twin is preserved
because operand H2D + per-pair unary transforms are still done ONCE per pair, scoring over large batches.)

BIT-FAITHFUL: this reuses the SAME bit-faithful GPU primitives the per-pair ``_pair_combo_mi_cupy`` twin
uses (``_gpu_apply_unary`` / ``_gpu_apply_binary`` / ``_gpu_quantile_bin_codes`` / ``_gpu_marginal_mi``),
which the section docstring records as bit-faithful to the njit table kernel (~6e-15). It is NOT the
percentile-edge resident plug-in MI (that path is only selection-equivalent at ties) -- selection here is
ULP-sensitive (the MI table drives the diversity-filtered retain/drop set), so we keep the rank-based
binning that matches njit.

GATE: engages ONLY where a per-host KTC crossover (sibling ``_usability_pool_resident_ktc``, keyed on
(n_rows, total_combos)) has MEASURED the resident batched table faster than the per-pair njit table kernel.
On a no-cupy / CPU host the gate returns ``None`` and the caller takes the exact njit per-pair path --
byte-for-byte unchanged.

NOT WIRED INTO ``build_usability_candidate_pool`` (bench-noted there) for TWO independent reasons measured
2026-06-23 on the dev GTX 1050 Ti:

  (1) NOT SELECTION-EQUIVALENT (the blocking reason). The table is bit-FAITHFUL to ~6e-15, but the downstream
      STABLE MI-sort + greedy ``_abscorr`` diversity filter in the pool builder is ULP-SENSITIVE at MI ties:
      a 6e-15 reassociation in ``_gpu_marginal_mi`` vs the njit reduction flips the tie ORDER, changing which
      of two near-equal-MI combos is retained. Verified: a 125-form structured pool had ~6 retained forms
      DIFFER (e.g. ``mul(invsquared(a),neg(b))`` -> ``mul(invsquared(a),identity(b))``). The pool path must
      stay byte-identical, so the resident MI cannot feed it without a BIT-EXACT (njit-reduction-order) MI.

  (2) IT LOSES ON THIS CARD anyway: n=100k npairs=4 nc=1734/pair CUDA-event interleaved-min A/B = 29.6s
      resident vs 14.7s CPU njit -> 0.50x. The separation + launch-amortisation IS achieved (operand H2D +
      unary transforms ONCE per pair; batched scoring) so this is NOT the per-pair-launch trap of iter13/16;
      the new bottleneck is the bit-faithful bin+MI itself -- ``_gpu_quantile_bin_codes``/``_gpu_marginal_mi``
      run a per-row device->host scalar sync (~14k tiny syncs) the 6-SM card cannot hide (HW-bound regime).

NEEDS-X to ship: a BIT-EXACT GPU MI matching the njit reduction order (to make selection byte-identical) AND
a row-vectorised sync-free bin+MI kernel AND a card where it wins. The KTC gate (``_usability_pool_resident_ktc``)
+ this kernel are kept (feedback_keep_all_kernel_versions) so the work is ready to re-bench / re-wire once a
bit-exact vectorised MI exists; they are intentionally NOT called from the selection path today.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _build_combo_index_arrays(nu: int, nb: int):
    """The flat ``for ia: for ib: for ibn`` op-index enumeration shared by every pair (the combo grid is
    pair-INVARIANT -- only the operand columns change). Returns (ua_idx, ub_idx, bn_idx) int64 arrays of
    length ``nu*nu*nb`` in the SAME order as ``score_pair_combos`` / the Python retention loop."""
    nc = nu * nu * nb
    ua_idx = np.empty(nc, dtype=np.int64)
    ub_idx = np.empty(nc, dtype=np.int64)
    bn_idx = np.empty(nc, dtype=np.int64)
    j = 0
    for ia in range(nu):
        for ib in range(nu):
            for ibn in range(nb):
                ua_idx[j] = ia
                ub_idx[j] = ib
                bn_idx[j] = ibn
                j += 1
    return ua_idx, ub_idx, bn_idx


def score_pair_combos_table_resident(
    operands: Sequence[tuple],
    y_codes: np.ndarray,
    y_terms: tuple,
    nbins: int,
    ua_codes: np.ndarray,
    ub_codes: np.ndarray,
    bn_codes: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute the FULL ``(npairs, nc)`` pair-combo MI table for ALL pairs in ONE resident GPU pass.

    ``operands`` is a sequence of ``(x1, x2)`` float64 operand-column pairs (host arrays); ``y_codes`` the
    shared label codes; ``y_terms`` is ``(y_i, h_y, k_y)`` from ``precompute_marginal_y_terms``;
    ``ua_codes``/``ub_codes`` the per-unary op-codes (preset order), ``bn_codes`` the per-binary op-codes.

    Returns a host ``(npairs, nc)`` float64 array (row ``p`` == ``score_pair_combos`` for pair ``p``,
    sentinel ``-1.0`` for an ``std<=1e-9`` combo), or ``None`` if cupy is unavailable / a device error
    occurs (the caller then takes the exact per-pair njit path). BIT-FAITHFUL to the njit table kernel.

    Resident strategy: y is uploaded ONCE (fit-constant). For each pair we build its ``nc`` combo columns
    on the device (reusing the per-operand distinct unary transforms), accumulate them into one big
    ``(npairs*nc, n)`` candidate matrix, then bin + MI-score the WHOLE stack in one batched call -- so the
    only per-pair host->device traffic is the two operand columns (unavoidable), NOT a fresh kernel launch
    grid per pair. Only the ``(npairs, nc)`` table comes back to the host."""
    try:
        import cupy as cp
    except Exception:
        return None
    try:
        from ._gpu_policy import gpu_globally_disabled
        if gpu_globally_disabled():
            return None
        from ._usability_njit_pool import (
            _gpu_apply_unary, _gpu_apply_binary, _gpu_quantile_bin_codes, _gpu_marginal_mi,
        )

        npairs = len(operands)
        nu = len(ua_codes)
        nb = len(bn_codes)
        nc = nu * nu * nb
        if npairs == 0 or nc == 0:
            return np.empty((npairs, nc), dtype=np.float64)

        _, h_y, k_y = y_terms
        h_y = float(h_y)
        k_y = int(k_y)
        n = int(np.asarray(operands[0][0]).shape[0])
        qs = np.linspace(0.0, 1.0, int(nbins) + 1)
        d_qs = cp.asarray(qs, dtype=cp.float64)
        d_y = cp.asarray(np.ascontiguousarray(y_codes, dtype=np.int64))
        ky_w = int(d_y.max()) + 1 if n else 1   # y-cardinality (histogram width) for the fused MM kernel
        # SYNC-FREE fused bin+hist+MM-MI (2026-06-26): the resident table's per-flush scoring used
        # _gpu_quantile_bin_codes + _gpu_marginal_mi, whose per-ROW cp.bincount + .max() device->host sync
        # (~one per combo column) made the whole resident table a 0.5x LOSS on a 6-SM card. The fused
        # radix-edges + mi_mm_from_values kernels score the whole flush chunk in ONE launch each (no per-row
        # sync), MM-correct and partition-identical to _gpu_marginal_mi (maxdiff ~1e-15, rank identical).
        from ._fe_batched_mi import binned_mm_mi_from_values_gpu
        from ._gpu_resident_select import _radix_select_interior_edges

        ua_codes_l = [int(c) for c in ua_codes]
        ub_codes_l = [int(c) for c in ub_codes]
        bn_codes_l = [int(c) for c in bn_codes]
        ua_idx, ub_idx, bn_idx = _build_combo_index_arrays(nu, nb)
        from ._gpu_resident_fe import _get_fused_gen_kernel

        # FULLY-FUSED MATRIX-NATIVE (2026-06-26): generate AND score every combo on the device with NO
        # per-combo work. Per pair the nu distinct unaries of each operand are applied ONCE into a (nu, n)
        # column-major stack; then for each VRAM-bounded combo chunk ONE fused_gen launch builds the (n, kk)
        # candidate block (binary(unary_a, unary_b) by op-code, NaN/inf scrubbed) and ONE radix-edges +
        # mi_mm_from_values pair scores its MM-MI. So the per-pair cost is ~(nu unary applies + n_chunks*(gen
        # + edges + MI)) launches, NOT the nc per-combo elementwise + per-row-sync of the old path. MM-MI is
        # partition-identical to the njit table (~1e-15). Combo enumeration order (ua_idx/ub_idx/bn_idx) is the
        # SAME for ub/x2 -> the flat output index maps 1:1 to the caller's for ua: for ub: for bn loop.
        out = np.empty(npairs * nc, dtype=np.float64)
        ua_idx_d = cp.asarray(np.asarray(ua_idx, dtype=np.int32))
        ub_idx_d = cp.asarray(np.asarray(ub_idx, dtype=np.int32))
        bop_d = cp.asarray(np.asarray([bn_codes_l[int(bn_idx[j])] for j in range(nc)], dtype=np.int32))
        fused_gen = _get_fused_gen_kernel()
        try:
            free_b = int(cp.cuda.runtime.memGetInfo()[0])
        except Exception:
            free_b = 512 * 1024 * 1024
        # cand block (n, kk) f64 + radix/MI working (~3x): combo_chunk = ~25% free VRAM / (n*8*3), clamped.
        combo_chunk = int(max(1, min(nc, (free_b // 4) // (max(1, n * 8) * 3))))
        threads = 256

        for p in range(npairs):
            x1, x2 = operands[p]
            d_x1 = cp.asarray(np.ascontiguousarray(x1, dtype=np.float64))
            d_x2 = cp.asarray(np.ascontiguousarray(x2, dtype=np.float64))
            xmin_a = float(cp.asnumpy(cp.nanmin(d_x1))) if n else 0.0
            xmin_b = float(cp.asnumpy(cp.nanmin(d_x2))) if n else 0.0
            # (nu, n) column-major stacks: each operand's nu unaries applied ONCE (reused across all combos).
            ua_stack = cp.ascontiguousarray(cp.stack([_gpu_apply_unary(d_x1, ua_codes_l[ia], xmin_a) for ia in range(nu)]))
            ub_stack = cp.ascontiguousarray(cp.stack([_gpu_apply_unary(d_x2, ub_codes_l[ib], xmin_b) for ib in range(nu)]))
            base = p * nc
            for c0 in range(0, nc, combo_chunk):
                c1 = min(c0 + combo_chunk, nc)
                kk = c1 - c0
                cand = cp.empty((n, kk), dtype=cp.float64)          # (n, kk) row-major
                total = n * kk
                fused_gen(((total + threads - 1) // threads,), (threads,),
                          (ua_stack, ub_stack, ua_idx_d[c0:c1], ub_idx_d[c0:c1], bop_d[c0:c1],
                           np.int64(n), np.int32(kk), cand))
                mean = cand.mean(axis=0)
                var = (cand * cand).mean(axis=0) - mean * mean
                live = cp.asnumpy(var > 1e-18)
                mi_h = None
                try:
                    interior = _radix_select_interior_edges(cand, int(nbins))
                    if interior is not None:
                        mi_h = binned_mm_mi_from_values_gpu(cand, interior, d_y, int(nbins), ky_w, h_y, k_y, codes_trusted=True)   # d_y dense 0-based fit-constant (FIX1)
                except Exception:
                    mi_h = None
                if mi_h is None:                                    # per-row sync fallback (bit-faithful)
                    codes, kx = _gpu_quantile_bin_codes(cp.ascontiguousarray(cand.T), d_qs)
                    mi_h = cp.asnumpy(_gpu_marginal_mi(codes, kx, d_y, h_y, k_y, n))
                out[base + c0:base + c1] = np.where(live, mi_h, -1.0)
                del cand
            del d_x1, d_x2, ua_stack, ub_stack
        return out.reshape(npairs, nc)
    except Exception as _exc:  # noqa: BLE001
        logger.debug("score_pair_combos_table_resident: GPU path failed (%s); host fallback", _exc)
        return None
