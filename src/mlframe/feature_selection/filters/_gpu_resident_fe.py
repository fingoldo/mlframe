"""GPU-resident FE candidate generation + MI (prototype, gated, un-wired).

The terminal phase of the matrix-native FE replatform, and the only part with genuine NEW value: the
reason the GPU LOST the MI dispatch (see _hermite_fe_mi / the 2026-06-19 perf series) was the per-call
H2D upload + many tiny kernels -- ~700ms/call of pure overhead the on-device compute (~10-36ms) was
dwarfed by. The fix is to keep the data RESIDENT: upload the raw operands ONCE, generate the whole
unary x binary candidate grid ON the GPU (cupy elementwise), and score the entire grid in ONE big-k
batch-MI call. No per-candidate transfer, one large kernel -- exactly the regime the contention-aware
sweep showed the GPU winning (n=100k k>=100: cuda < njit).

GATED behind ``MLFRAME_FE_GPU_RESIDENT`` and imported by nothing in the production FE path: this is a
validated prototype proving the approach (correct MI vs the CPU path + faster at large n), not yet the
production recipe-integrated generator. It mirrors the MINIMAL unary/binary preset (enough to express
a**2/b and log(c)*sin(d)); the full catalog + recipe replay is the follow-up once the win is locked.

Non-pure op handling: ``smart_log`` shifts by the FULL-column nanmin (computed once on-device here, the
same anchor the CPU recipe freezes), ``div`` reproduces the exact ``y==0 -> eps`` branch -- so the
on-device candidate equals the CPU one to fp round-off.

BENCH (GTX 1050 Ti, K=384 minimal-preset candidates per pair, median of 3, warm; vs numpy-gen + njit
batch MI). Keeping data resident flips the GPU from the old 3x LOSER (per-call H2D path) to a WINNER
that SCALES -- and the VRAM-bounded K-chunk (``_gpu_k_chunk``, mirroring the CPU RAM governor on-device)
removes the large-n cliff entirely, with the on-device MI matching the CPU path to fp round-off (argmax + values):
  * n=20k   : CPU 287ms   / GPU 379ms  -> 0.76x  (small n: GPU launch dominates -> dispatcher routes CPU)
  * n=100k  : CPU 1771ms  / GPU 854ms  -> 2.07x  (k_chunk=141)
  * n=300k  : CPU 7013ms  / GPU 2046ms -> 3.43x  (k_chunk=47 -- was a 0.12x cliff before chunking)
  * n=1M    : CPU 29731ms / GPU 6424ms -> 4.63x  (k_chunk=14; the win GROWS with n)
``pair_candidate_mi_dispatch`` routes >= _GPU_RESIDENT_MIN_N (50k) to the chunked GPU path, CPU below.

KERNEL-TUNING INVESTIGATION (2026-06-19, GTX 1050 Ti, the on-device MI = cupy, not numba.cuda). The
on-device MI breaks down (n=1M, k=14) as: ``cp.argsort`` quantile-binning = 161ms (69%), histogram+MI
math = 73ms (31%). So the FULL O(n log n) sort cupy uses for equi-frequency binning is the dominant
cost and the obvious tuning target. Two sort-free replacements were prototyped + measured:
  * equi-WIDTH sub-histogram -> CDF -> quantile edges (no sort): 4.36x faster binning (43 vs 189ms) BUT
    BREAKS on heavy-tailed FE candidates (a**2/b's outliers stretch the range so ~all mass collapses to
    bin 0): bin-code agreement 98.6%, but candidate-MI Spearman only 0.88, MI maxdiff ~1.9, argmax flips
    -> REJECTED (would change selection).
  * monotone tail-compressed (``sign(x)*log1p|x|``, rank-preserving so quantiles are invariant) THEN
    equi-width sub-hist: Spearman 0.9993, MI maxdiff 0.029 -- statistically excellent and still 4.36x
    faster binning. But the TOP candidate is a near-tie among EQUIVALENT a**2/b spellings, and a 0.03 MI
    perturbation reorders that tie -> argmax flips -> NOT bit-exact -> unsafe as a drop-in for the
    exact-result contract. Good as an opt-in APPROXIMATE fast mode only.
The exact+fast path (NEXT): prescreen all candidates with the sort-free MI, then re-score only the
top-K (margin-guarded) with the exact ``cp.argsort`` MI -- exact winner at a fraction of the sort cost.
Numbers recorded so this is not re-derived blind; the exact ``cp.argsort`` path stays the default.

bench-attempt-rejected (2026-06-20, re-test under the RELAXED selection-equivalence bar -- user: same
features selected is enough, bit-identity not required): the tail-compressed sort-free binning IS faster
(K=48 heavy-tail: 100k 1.47x, 1M 2.04x; 99.5% code-agree, Spearman 0.99992) BUT FLIPS the END-TO-END FE
selection at n=100k -- the ~0.5% of codes differing at quantile edges perturb the noise-gate/MI ranking
past the compound-recovery threshold, yielding 9 fragmented features instead of the single fused
``add(sqr(a)/b, log(c)*sin(d))`` compound (the clean-compound gate goes RED). f32 sort keys preserve the
selection (100% code-agree) but are NOT faster here (f64->f32 cast overhead: 0.77x@100k, 1.12x@1M). So
there is NO selection-safe binning speedup on this GPU (GTX 1050 Ti) even with bit-identity waived -- the
exact cp.percentile sort is at the bandwidth floor; beating it needs a rank-EXACT sort-free kernel
(radix-rank, roadmap #2), not an approximate quantile. cp.percentile stays the default.
"""
from __future__ import annotations

import os

import numpy as np

# REVIEW ROADMAP (2026-06-19 multi-agent critique; items dispositioned FUTURE -- captured so they are
# not re-derived). PERF (ranked payoff/effort, all must stay bit-exact -> validate maxdiff 0 + argmax on
# HEAVY-TAILED a**2/b candidates, not just uniform): (1) f32 sort keys for binning (~1.5-1.8x on the 69%
# sort; exact only with row-index tie-break, else gate to prescreen); (2) radix-rank RawKernel replacing
# cp.argsort + the uncoalesced scatter (exact, removes the (n,K) int64 sort_idx); (3 + 5) DONE 2026-06-20:
# fused SHARED-MEM atomic histogram + grand fusion -- ``fused_gen_bin_hist`` (one block per candidate)
# re-generates each candidate value, bins it against the EXACT cp.percentile edges, and accumulates the
# (P1, nbins, K_y) joint histogram in shared-mem atomics in ONE kernel per chunk, so the (n,K) float
# matrix, the (n,K) int codes, the (n,K) D2H disc AND the noise-gate's (n,K) d_base / (rows*n*K) flat
# index are NEVER materialised (Option F1 recompute-not-store + #3 atomic hist). BIT-IDENTICAL (maxdiff 0,
# argmax match) -> selection EXACT; measured GTX 1050 Ti K=384 nperm=25: 100k 2.16x +3.0x less peak GPU
# mem, 300k 2.15x +2.75x, 1M 3.39x +2.26x. ``grand_fused_pair_mi_fused`` (default ON via
# MLFRAME_FE_GPU_GRAND_FUSION); the non-fused body is the exact fallback when the shared hist overflows
# the per-block limit.
#   PROTOTYPE-ONLY (2026-06-20, measured): grand-fusion CANNOT be wired into the PRODUCTION canonical fit.
#   Production candidate-MI runs through _pairs_chunks._compute_one_fe_chunk -> gpu_materialise_discretize_
#   codes_host -> _dispatch_batch_mi_with_noise_gate (NOT gpu_pairs_fe_mi, which returns None on the
#   canonical fixture). That path REQUIRES the (n,K) float candidate matrix on host -- the downstream
#   survivor / usability-corr / ext-val / multi-emit stages read the CONTINUOUS columns out of the chunk
#   buffer -- so a recompute-not-store fused kernel that skips materialisation would leave the buffer
#   uninitialised (garbage survivors). Grand-fusion's whole payoff (never materialise (n,K)) is thus
#   incompatible with production. AND the production e2e bottleneck is NOT the MI counting the fused kernel
#   collapses -- profiled n=100k/79s: cp.percentile's SORT in _gpu_resident_discretize_codes = 12.9s
#   (binning), MI counting already on GPU + cheap. So the only production lever is roadmap #2 (rank-EXACT
#   sort-free edges, e.g. radix-SELECT of the nbins-1 order statistics + cp.percentile's linear interp ->
#   exact edges without a full O(n log n) sort), which must still EMIT the float out_cand for downstream.
# (4) multi-stream across k-chunks (lowers the GPU crossover n) stays FUTURE. Route
# block size (currently threads=256) + the VRAM 0.25/5x constants + any f32 threshold
# through pyutilz kernel_tuning_cache; keep cp.argsort as the exact fallback when adding a radix _v2.
# ARCHITECTURE (wiring, before flipping any default): the production FE speaks STRUCTURED, preset-stamped,
# gate-filtered, replayable EngineeredRecipe -- this path speaks flat (name, MI). Wire it as a candidate-MI
# PROVIDER feeding the EXISTING gates (noise-gate/SU/external-validation/prevalence), emitting structured
# (ua,ub,bop) triples + real src column names + active presets (reuse fe_tuple->get_new_feature_name->
# EngineeredRecipe; never re-parse the string). Drive the op set from create_*_transformations(preset) +
# the gpu_compatible_unary_names allowlist with CPU fallback for unsupported/NON-pure ops (smart_log
# anchor must be the frozen full-column value, not the subsample min). Replace the hardcoded
# _GPU_RESIDENT_MIN_N with a CONTENTION-AWARE kernel_tuning_cache sweep (mirror _run_sweep_mi_classif_
# dispatch). Collapse MLFRAME_FE_MATRIX_P0 + MLFRAME_FE_GPU_RESIDENT into one backend selector (gpu=>
# matrix). Add: pickle/clone test (no cupy/FeatureMatrix/RawKernel reachable from estimator state),
# combo-order-vs-registry meta-test, and a 3-impl op-parity test (registry vs _unary/_binary_apply vs the
# CUDA switch -- _safe_div is the single spec; its 2026-06-13 heavy-tail fix lives in ONE place).

# Minimal-preset op NAMES (kept in sync with feature_engineering.create_*_transformations "minimal").
_MINIMAL_UNARY = ("identity", "neg", "abs", "sqr", "reciproc", "sqrt", "log", "sin")
_MINIMAL_BINARY = ("mul", "add", "sub", "div", "max", "min")


# RESIDENCY REPLATFORM MAP (2026-06-21, measured -- the production chunk path, NOT this prototype).
# Goal: kill the FE chunk's bulk float-buffer D2H. ``gpu_materialise_discretize_codes_host`` D2Hs the
# full (n,K) float candidate buffer via ``out_cand`` -- measured 6.7 GB total across the canonical
# n=100k fit (15 calls), a large slice of the ~10.5s ``cupy.get`` wall. The codes (for MI) are produced
# RESIDENT and don't need that buffer; only a HANDFUL of host reads do.
#   * Final survivors ALREADY recompute from raw via ``_rebuild_full_survivor_col`` (the subsample path,
#     _pairs_core.py:2218) -- they do NOT read the float buffer.
#   * The bulk float buffer feeds only the INTERMEDIATE subsample scoring reads in check_prospective_fe_pairs:
#     best-config (~1625/1749), multi-emit (~2126/2137), MI-replay (~1499). A few columns per pair.
# So the win = KEEP the chunk-batch GPU-codes path, pass ``out_cand=None`` (skip the 6.7 GB D2H), and
# route those few intermediate reads through the SAME validated recompute helper (``_config_by_i`` /
# ``_rebuild_full_survivor_col``, already bit-identical). Gate it on the GPU-fused path actually running
# (else the CPU binning still needs the buffer) and report ``float_deferred`` back from the chunk so the
# caller knows to recompute vs read the buffer.
# MEASURED CONSTRAINT: forcing the pure recompute path (no buffer, no chunk-batch) is 3x SLOWER
# (217s vs 69s) -- chunk-batch is essential, so the replatform must COMBINE chunk-batch + deferred-float
# + per-read recompute, NOT replace chunk-batch with recompute. Implement gated + pin-validated.
# TURNKEY IMPLEMENTATION (exact sites in _pairs_core.check_prospective_fe_pairs):
#   1. Gate ``MLFRAME_FE_GPU_DEFER_FLOAT`` (default off). Active only when the chunk used the GPU FUSED
#      codes path (_gpu_disc_2d produced) -- else the CPU binning still needs the float buffer.
#   2. When deferred: pass ``out_cand=None`` to gpu_materialise_discretize_codes_host (skip the float D2H),
#      and after the chunk set ``final_transformed_vals = None`` for the read phase + populate
#      ``_config_by_i`` from the chunk's _batch_candidates (the (a_key,b_key,bin_name) per buffer column).
#   3. Add the SAME ``elif _config_by_i is not None`` recompute fallback that _config_corr (~1500-1506)
#      already has to the THREE direct buffer reads that lack it: best-config (~1625), winner-vals (~1749),
#      multi-emit (~2126/2137). The recompute is binary_transformations[bn](pa,pb)+nan_to_num -- bit-
#      identical to the buffer value. (1499 already has the fallback.) Survivor packing (~2218) already
#      recomputes via _rebuild_full_survivor_col on the subsample path, so it needs no change.
#   4. Validate: the 3 clean-compound pins + _compound_gate at n=10k/100k with the flag ON must match OFF.
# BENCH-REJECTED (2026-06-21): the float-D2H deferral was IMPLEMENTED (gated scaffold) and measured --
# clean paired A/B (6 cold fits, quiet box) = OFF 160.0s vs ON 162.6s = 0.98x, a WASH. The 6.7 GB float
# D2H OVERLAPS the GPU/CPU compute (async), so skipping it does not cut wall -- cProfile's cupy.get time
# is overlapped WAIT, not serial transfer. This is the THIRD GPU-data-movement wash after radix coalescing
# (7x kernel, 0x wall) and noise-gate launch-batching (1.00x): the canonical FE fit is COMPUTE-bound, not
# transfer/residency-bound. So residency/round-trip reduction is a dead lever here; reverted (commit after
# 72d1c364). Real wall levers are COMPUTE reduction (subsample scale-down -- 64s@30k -> 36s@8k but plateaus
# at a ~30s FE-component floor; selection holds on the canonical formula) or cutting FE work, NOT transfers.


def fe_gpu_resident_enabled() -> bool:
    """Whether the GPU-resident FE prototype is active. OFF unless ``MLFRAME_FE_GPU_RESIDENT`` truthy."""
    return os.environ.get("MLFRAME_FE_GPU_RESIDENT", "").strip().lower() in ("1", "true", "on", "yes")


def _cuda_present() -> bool:
    """Best-effort CUDA-availability probe for the resident-codes default. Mirrors
    ``batch_mi_noise_gate_gpu._CUDA_AVAIL`` (pyutilz ``is_cuda_available`` -> numba.cuda fallback) without
    importing the GPU twin at module-import time. Any failure -> False (CPU path, never a regression)."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        return bool(is_cuda_available())
    except Exception:
        try:
            from numba import cuda as _c
            return bool(getattr(_c, "is_available", lambda: False)())
        except Exception:
            return False


def fe_gpu_resident_codes_enabled() -> bool:
    """Whether the GPU-RESIDENT-CODES handoff is active. DEFAULT ON when CUDA is present (env opt-out
    ``MLFRAME_FE_GPU_RESIDENT_CODES=0``); explicit truthy forces it on even without a detected device.

    When ON, ``gpu_materialise_discretize_codes_host`` keeps the on-device int codes RESIDENT (a single
    (n, K) cupy array in the narrow code dtype) and stashes them in a module-level handoff (keyed on the
    identity of the host codes array it returns). The noise-gate dispatch then feeds those device codes
    STRAIGHT into ``batch_mi_with_noise_gate_cuda_resident`` -- skipping the H2D re-upload of the codes the
    resident gate would otherwise pay (the codes were just produced on the GPU, D2H'd, and re-H2D'd: a
    pointless GPU->host->GPU round-trip of the (n, K) int codes). The host codes are STILL produced, so the
    CPU dispatch / analytic gate / GPU-opt-out / SU / any-failure branches are byte-for-byte unchanged and
    the round-trip is skipped ONLY when the resident CUDA gate is the chosen consumer (so this is
    selection-equivalent -- the device codes are the EXACT bytes the host ``out`` carries). Gate:
    ``MLFRAME_FE_GPU_RESIDENT_CODES`` (0/false/off -> off; 1/true/on/yes -> on; UNSET -> on iff CUDA
    present). Honours the CUDA opt-out conventions (``CUDA_VISIBLE_DEVICES=""`` / ``MLFRAME_DISABLE_GPU=1``)
    by short-circuiting to OFF -- the resident gate is skipped on those runs anyway, so never stash."""
    _v = os.environ.get("MLFRAME_FE_GPU_RESIDENT_CODES", "").strip().lower()
    if _v in ("0", "false", "off", "no"):
        return False
    if _v in ("1", "true", "on", "yes"):
        return True
    # UNSET: default ON when a CUDA device is usable. Respect the documented GPU opt-outs so a no-GPU run
    # never pays the resident-copy alloc / stash (the dispatch routes to CPU there regardless).
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if (_cvd is not None and _cvd.strip() == "") or os.environ.get("MLFRAME_DISABLE_GPU", "") == "1":
        return False
    return _cuda_present()


def fe_gpu_resident_basis_mi_enabled() -> bool:
    """Whether the matrix-native RESIDENT orth-FE basis-MI path is active (Piece 3). DEFAULT OFF.

    When ON, ``hybrid_orth_mi_fe`` builds the univariate orth-basis candidate matrix ON the device
    (BATCHED ``_gpu_evaluate_basis_matrix``) and scores its plug-in MI with
    ``_plugin_mi_classif_batch_cuda_resident`` -- with NO per-call H2D (the dispatcher's 2x trap).

    DEFAULT ON when CUDA is present (2026-06-21). Validated: selection-EQUIVALENT (it swaps the njit RANK
    binning for the GPU equi-frequency-edge binning in the basis ranking -- the orth-basis recovery pins
    test_layer21/22, the canonical single_compound, and the full biz-value hybrid_orth suite all pass with
    it on: 385 passed) AND FASTER (clean canonical 100k wall 34.8s -> 30.7s, ~12%; the batched build also
    clears the p200 high-feature perf budget). Opt out with ``MLFRAME_FE_GPU_RESIDENT_BASIS_MI=0``. Any GPU
    failure / unported basis falls back to the host path per-call (never a correctness regression)."""
    _v = os.environ.get("MLFRAME_FE_GPU_RESIDENT_BASIS_MI", "").strip().lower()
    if _v in ("0", "false", "off", "no"):
        return False
    if _v in ("1", "true", "on", "yes"):
        return True
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if (_cvd is not None and _cvd.strip() == "") or os.environ.get("MLFRAME_DISABLE_GPU", "") == "1":
        return False
    return _cuda_present()


def fe_gpu_routing_enabled() -> bool:
    """Whether the orth-FE basis ROUTING (basis_route_by_signal) runs on the device. DEFAULT ON.

    When ON, ``_gpu_build_and_score_univariate`` picks each source column's orth-basis on the GPU (batched
    eval of all candidate bases x degrees on the RESIDENT operand matrix -- reused, no second H2D -- + a
    batched |Pearson corr| vs the resident continuous y, argmax per column) instead of the per-column host
    ``basis_route_by_signal``. Opt out with ``MLFRAME_FE_GPU_ROUTING=0``.

    Routing is SELECTION-BEARING (the chosen basis is baked into the EngineeredRecipe) and the GPU basis eval
    is parity-<1e-6, NOT bit-identical, so a near-tie between two bases can in principle flip the argmax. Two
    things make it safe to default on: (1) only the corr VALUES come from the GPU -- the argmax/tie logic is
    byte-identical to the host router, so a flip needs a genuine sub-epsilon corr tie where the two bases are
    equally usable by construction; (2) a per-column host fallback covers any GPU failure / degenerate column
    (never a correctness regression).

    VALIDATED 2026-06-22 (quiet GTX 1050 Ti): SELECTION-EQUIVALENT -- test_gpu_routing_parity matches the
    host router on every clear-margin column across 3 seeds (only sub-1e-16 chebyshev/legendre numerical ties
    diverge), and the full selection-bearing suite passes with routing ON (test_mrmr_feature_engineering +
    layer21/22 orth-recovery + canonical single_compound + hybrid_orth biz = 112 passed). WALL: measured on
    the RESIDENT operand matrix (the real residency scenario -- M is already on-device for the basis-MI
    build), routing is host 229ms vs GPU 214ms = 1.07x (30k x 24 cols, 24/24 matching); the wiring also
    reuses that single upload, removing a redundant (n, n_cand) H2D the prior per-column path implied. (An
    earlier A/B wrongly charged the GPU an H2D that residency mode does not pay -- corrected here.)"""
    _v = os.environ.get("MLFRAME_FE_GPU_ROUTING", "").strip().lower()
    if _v in ("0", "false", "off", "no"):
        return False
    if _v in ("1", "true", "on", "yes"):
        return True
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if (_cvd is not None and _cvd.strip() == "") or os.environ.get("MLFRAME_DISABLE_GPU", "") == "1":
        return False
    return _cuda_present()


def fe_gpu_defer_host_codes_enabled() -> bool:
    """Whether the DEFERRED host-codes D2H is active. DEFAULT ON whenever the resident-codes handoff is on
    (the device codes already exist resident, so skipping the eager host D2H of those SAME codes is free
    upside -- the canonical fit's single largest D2H, 1691 MB / 160 transfers @100k, paid only on the host
    paths that actually read it, which is NONE when the resident gate consumes the device copy). Opt-out
    ``MLFRAME_FE_GPU_DEFER_HOST_CODES=0`` forces the eager fill back (host ``out`` filled per block).

    SELECTION-EQUIVALENT: when a host consumer (analytic gate / CPU njit / non-resident GPU) reads the
    codes, the dispatch first calls ``ensure_host_codes_filled`` which D2Hs the resident device codes into
    ``out`` -- the EXACT bytes the eager fill produced -> identical codes -> identical MI/selection. The
    only behavioural change is WHEN the D2H happens (lazily, on demand) vs eagerly (always)."""
    _v = os.environ.get("MLFRAME_FE_GPU_DEFER_HOST_CODES", "").strip().lower()
    if _v in ("0", "false", "off", "no"):
        return False
    if _v in ("1", "true", "on", "yes"):
        return True
    return fe_gpu_resident_codes_enabled()  # default ON iff the resident-codes handoff is on


# RESIDENT-CODES HANDOFF (2026-06-21, gated). The FE chunk path calls ``gpu_materialise_discretize_codes_
# host`` (producer) and ``_dispatch_batch_mi_with_noise_gate`` (consumer) as SEPARATE steps -- the producer
# returns the host int codes (``disc_2d``) which the consumer's resident-CUDA gate re-uploads. To pass the
# ON-DEVICE codes straight through WITHOUT threading a new argument through the chunk path (which is owned
# elsewhere), the producer stashes the resident device codes here keyed on ``id()`` of the EXACT host array
# it returns; the dispatch retrieves them by that same identity (the host ``disc_2d`` flows unchanged from
# producer to dispatch in the same synchronous call, so the id is stable + alive). Shape/dtype are also
# matched as a guard. Module-level singleton (NOT on an estimator instance) -> never reachable from pickled
# state. Cleared after consumption / on any mismatch so stale device memory is not pinned.
_RESIDENT_CODES_HANDOFF: tuple | None = None  # (id(host_codes), device_codes cupy array, shape, dtype) | None


# DEFERRED HOST-CODES FILL (2026-06-21). When the resident-codes path is on, the (n, K) host int codes are
# the SINGLE largest D2H of the canonical fit (audit n=100k: _gpu_resident_fe.py codes-fill = 1691 MB / 160
# transfers, the whole D2H budget) -- yet they are REDUNDANT whenever the resident-CUDA noise gate is the
# consumer (it reads the device codes IN PLACE; host ``disc_2d`` is only touched for ``.shape``). The
# producer can therefore DEFER the codes D2H: return an UNFILLED host buffer + keep the device codes, and
# materialise the host buffer LAZILY only if a host-reading consumer (the analytic gate / CPU njit kernel /
# non-resident GPU path) actually needs it. The dispatch calls ``ensure_host_codes_filled`` before any
# host-codes read, and ``take_resident_codes`` before the resident path -- so the 1691 MB D2H is paid ONLY
# on the host paths that read it (none at the canonical n=100k fit) and SKIPPED when the resident gate
# consumes the device codes. BIT-IDENTICAL: the host buffer, if ever filled, is ``device_codes.get()`` --
# the exact bytes the eager D2H produced -> selection unchanged. Keyed on ``id(out)`` like the handoff (same
# host array flows producer -> dispatch synchronously). Module-level singleton -> never on pickled state.
_DEFERRED_HOST_FILL: tuple | None = None  # (id(host_codes), host_codes ndarray, device_codes, shape, dtype, filled[list[bool]]) | None


def _stash_resident_codes(host_codes, device_codes) -> None:
    """Record the resident device codes for the host array ``host_codes`` (keyed on its id)."""
    global _RESIDENT_CODES_HANDOFF
    _RESIDENT_CODES_HANDOFF = (id(host_codes), device_codes, tuple(host_codes.shape), np.dtype(host_codes.dtype))


def _stash_deferred_host_fill(host_codes, device_codes) -> None:
    """Register ``host_codes`` (an UNFILLED host buffer) for a lazy D2H fill from ``device_codes`` -- used
    when the producer skips the eager codes D2H because the resident gate will likely consume the device
    copy. ``ensure_host_codes_filled`` performs the fill on demand; ``clear_resident_codes_handoff`` drops
    the record so device memory is not pinned past the dispatch."""
    global _DEFERRED_HOST_FILL
    _DEFERRED_HOST_FILL = (id(host_codes), host_codes, device_codes, tuple(host_codes.shape), np.dtype(host_codes.dtype), [False])


def ensure_host_codes_filled(host_codes) -> None:
    """If ``host_codes`` was returned UNFILLED with a deferred device->host fill registered for it (same id,
    shape, dtype), D2H the device codes into it NOW (once; idempotent). No-op when there is no deferred fill
    for this array (it was filled eagerly, or is an unrelated array). Called by the dispatch on every
    host-codes-reading path (analytic gate / CPU njit kernel / non-resident GPU path) so the codes D2H is
    paid exactly when -- and only when -- a host consumer reads them. Bit-identical to the eager fill."""
    h = _DEFERRED_HOST_FILL
    if h is None:
        return
    _id, host, dev, shape, dtype, filled = h
    if _id != id(host_codes) or tuple(host_codes.shape) != shape or np.dtype(host_codes.dtype) != dtype:
        return
    if filled[0]:
        return
    # D2H the resident device codes into the caller's host buffer (the exact bytes the eager path produced).
    dev.get(out=host)
    filled[0] = True


def take_resident_codes(host_codes):
    """Return the resident device codes stashed for ``host_codes`` iff they match it (same id, shape,
    dtype); else None. The resident-CUDA gate consumes the returned device codes IN PLACE (no host read).
    The handoff is NOT cleared here (so a later host-reading branch in the SAME dispatch -- e.g. the
    analytic gate that runs after this pop -- can still trigger the deferred host fill from the same device
    codes); the dispatch clears everything via ``clear_resident_codes_handoff`` in its finally."""
    h = _RESIDENT_CODES_HANDOFF
    if h is None:
        return None
    _id, dev, shape, dtype = h
    if _id == id(host_codes) and tuple(host_codes.shape) == shape and np.dtype(host_codes.dtype) == dtype:
        return dev
    return None


def clear_resident_codes_handoff() -> None:
    """Drop the resident-codes handoff + the deferred host-fill record so device memory is not pinned past
    the dispatch that produced it, and a stale entry can never satisfy a later, unrelated dispatch. Called
    by ``_dispatch_batch_mi_with_noise_gate`` in a finally after it has decided the consumer."""
    global _RESIDENT_CODES_HANDOFF, _DEFERRED_HOST_FILL
    _RESIDENT_CODES_HANDOFF = None
    _DEFERRED_HOST_FILL = None


# GPU-RESIDENT BINNING DTYPE (2026-06-20). The FE candidate buffer is ALREADY float32 (the njit/CUDA
# materialise writes float32; the chunk buffer is float32). The original GPU-resident discretize path
# UP-CAST it to float64 before cp.percentile + cp.searchsorted, so the bandwidth-bound full sort moved
# 2x the bytes for NO accuracy gain over the f32 source. Binning NATIVELY in float32 removes the cast
# AND halves the bytes the sort/percentile + searchsorted scan, and (measured this run) PRESERVES the
# clean-compound FE SELECTION (the acceptance bar -- same features, not bit-identical edges; f32-vs-f64
# code agreement ~100%). So float32 is the DEFAULT binning dtype; the exact f64 path stays one env flip
# away (``MLFRAME_FE_GPU_BINNING_DTYPE=float64``) as the bit-identical fallback if a future host/data
# combination ever destabilises selection. The prior "f32 was SLOWER" result (docstring, 2026-06-20)
# measured an f64->f32 CAST overhead on an f64-source path; here the source is f32 so there is no cast.
# Implementation: _gpu_resident_discretize_codes bins in the INPUT's native dtype, so the float32 host
# FE-buffer paths (gpu_discretize_codes_host / gpu_materialise_discretize_codes_host) bin in float32 with
# NO up-cast, while the float64 grand-fused MI path (gpu_resident_pair_candidate_mi / grand_fused_pair_mi,
# fed by _fused_generate_block -> float64) stays float64 (bit-identical). MLFRAME_FE_GPU_BINNING_DTYPE
# (float32|float64) force-overrides the working dtype host-wide if a future host/data combo needs it.


def _xp_special(xp):
    """The scipy.special-compatible module for array module ``xp``: ``scipy.special`` for numpy,
    ``cupyx.scipy.special`` for cupy. Used for the GPU full-catalog special functions (erf / gammaln /
    beta / binom are present in cupyx; dawsn / agm are NOT -- see _GPU_UNAVAILABLE_* below)."""
    if xp is np:
        import scipy.special as _s
        return _s
    import cupyx.scipy.special as _s
    return _s


# GPU-ONLY catalog (2026-06-21): user directive -- EVERYTHING runs on the GPU, NO CPU fallback. The six
# ops that lack a cupy/cupyx equivalent or need cross-row / data-dependent / emath handling are
# TEMPORARILY DISABLED in BOTH catalogs (here AND feature_engineering.create_*_transformations) so the
# GPU grid == the CPU grid exactly:
#   unary : grad1, grad2 (cross-row np.gradient -- wrong under row-blocked generation), dawsn (no cupyx)
#   binary: agm (no cupyx), logn (np.emath complex-base), binom (deferred with the others)
# TODO(restore): re-enable each behind a real GPU kernel (grad via full-column cp.gradient; dawsn/agm via
# custom kernels; logn via log/log on the shifted operands) once the GPU residency path is validated.
# Mirrors feature_engineering's enabled catalog EXACTLY so the GPU candidate == the CPU one to fp
# round-off (selection-equivalent -- the FE perf bar).
_FULL_UNARY = (
    "identity", "neg", "abs", "sqr", "reciproc", "sqrt", "log", "sin",          # minimal
    "sign", "rint", "qubed", "invsquared", "invqubed", "cbrt", "invcbrt", "invsqrt", "exp",  # medium
    # GPU-DISABLED(restore): "grad1", "grad2",  (cross-row np.gradient)
    "sinc", "cos", "tan", "arcsin", "arccos", "arctan",                         # maximal trig
    "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "erf", "gammaln",  # maximal hyp+special
    # GPU-DISABLED(restore): "dawsn",  (no cupyx)
)
_FULL_BINARY = (
    "mul", "add", "sub", "div", "max", "min",                                   # minimal
    "abs_diff", "hypot", "signed", "ratio_abs",                                 # medium
    "logaddexp", "pow", "heaviside", "greater", "less", "equal", "beta",        # maximal
    # GPU-DISABLED(restore): "agm" (no cupyx), "logn" (np.emath), "binom"
)


def _unary_apply(xp, name, x):
    """Apply unary ``name`` to ``x`` using array module ``xp`` (numpy or cupy). Semantics mirror
    ``feature_engineering.create_unary_transformations`` EXACTLY (full maximal catalog), incl.
    smart_log's full-column nanmin shift. Raises ValueError for an op with no ``xp`` equivalent (the
    caller excludes it from the GPU grid -- see _GPU_UNAVAILABLE_UNARY)."""
    # --- minimal ---
    if name == "identity":
        return x
    if name == "neg":
        return -x
    if name == "abs":
        return xp.abs(x)
    if name == "sqr":
        return xp.power(x, 2)
    if name == "reciproc":
        return xp.power(x, -1.0)
    if name == "sqrt":
        return xp.sqrt(xp.abs(x))
    if name == "log":
        x_min = xp.nanmin(x)
        # smart_log: shift only when the column reaches <=0 (anchor frozen over the FULL column).
        return xp.log(x) if float(x_min) > 0 else xp.log(x + (1e-5 - x_min))
    if name == "sin":
        return xp.sin(x)
    # --- medium ---
    if name == "sign":
        return xp.sign(x)
    if name == "rint":
        return xp.rint(x)
    if name == "qubed":
        return xp.power(x, 3)
    if name == "invsquared":
        return xp.power(x, -2)
    if name == "invqubed":
        return xp.power(x, -3)
    if name == "cbrt":
        return xp.cbrt(x)
    if name == "invcbrt":
        return xp.power(x, -1.0 / 3.0)
    if name == "invsqrt":
        return xp.power(x, -1.0 / 2.0)
    if name == "exp":
        return xp.exp(x)
    # --- maximal: trig / inverse-trig / hyperbolic / inverse-hyperbolic (all native to numpy AND cupy) ---
    if name in ("sinc", "cos", "tan", "arcsin", "arccos", "arctan",
                "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh"):
        return getattr(xp, name)(x)
    # --- maximal: special functions (scipy.special / cupyx.scipy.special) ---
    if name == "erf":
        return _xp_special(xp).erf(x)
    if name == "gammaln":
        return _xp_special(xp).gammaln(x)
    # GPU-DISABLED(restore): uncomment these + the matching _FULL_UNARY entries + the
    # feature_engineering.py catalog lines to re-enable. grad1/grad2 need a FULL-column gradient (never a
    # row-block); dawsn needs a cupyx/custom kernel (absent on this stack).
    # if name == "grad1":
    #     return xp.gradient(x)
    # if name == "grad2":
    #     return xp.gradient(x, edge_order=2)
    # if name == "dawsn":
    #     return _xp_special(xp).dawsn(x)
    raise ValueError(f"unknown unary {name!r}")


# --- GPU port of the per-operand PRE-WARP apply (phase R1, 2026-06-21) ----------------------------------
# Mirrors hermite_fe.apply_operand_prewarp so the operand-table mirror can BUILD a prewarp operand column on
# the device (from the resident raw input + the tiny stored spec) instead of COPYING the host-computed column
# (the 1.68 MB non-plain H2D floor). The preprocess (z-score / min-max / shift, all elementwise + a clip) and
# the Clenshaw polynomial recurrences below replicate NUMPY's chebval/legval/hermeval(He)/lagval to fp
# round-off (same Clenshaw algorithm + float64 op order). SELECTION-EQUIVALENCE NOTE (P2-2): the production
# HOST path (polyeval_dispatch -> njit) evaluates these three (cheb/leg/herme) by a FORWARD recurrence, which
# differs from numpy/GPU-Clenshaw by ~1e-12 at degree>=3 (laguerre is forward on both, so it is bit-consistent
# across device+host). So a candidate MI-RANKED on GPU-Clenshaw values and later REPLAYED via host-forward
# differs by ~1e-12 -- far below any selection threshold (selection is decided on the consistent GPU values
# within the resident path; test_gpu_basis_column_parity pins the host<->GPU bound). Unifying both onto one
# recurrence is a FUTURE kernel change, unneeded at the default max_degree. Any unsupported basis / failure
# RAISES so the caller falls back to the host copy (never a correctness regression). fourier_adaptive
# (escalation only -- not in the main operand table) is ported too for completeness.

def _cheb_clenshaw_gpu(cp, x, c):
    if len(c) == 1:
        c0 = c[0]; c1 = 0.0
    elif len(c) == 2:
        c0 = c[0]; c1 = c[1]
    else:
        x2 = 2.0 * x
        c0 = c[-2]; c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1 * x2
    return c0 + c1 * x


def _leg_clenshaw_gpu(cp, x, c):
    if len(c) == 1:
        return c[0] + 0.0 * x
    if len(c) == 2:
        c0 = c[0]; c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]; c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * x * (2 * nd - 1)) / nd
    return c0 + c1 * x


def _herme_clenshaw_gpu(cp, x, c):
    if len(c) == 1:
        return c[0] + 0.0 * x
    if len(c) == 2:
        c0 = c[0]; c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]; c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - c1 * (nd - 1)
            c1 = tmp + c1 * x
    return c0 + c1 * x


def _lag_clenshaw_gpu(cp, x, c):
    # FORWARD recurrence ``out = sum_k c[k] L_k`` matching the host _lagval_njit EXACTLY
    # (L_0=1, L_1=1-x, L_k = ((2k-1-x)L_{k-1} - (k-1)L_{k-2})/k). The prior Clenshaw-style
    # recurrence here was WRONG for Laguerre (verified: L_2(0) gave -0.5 vs the correct 1) --
    # it was never exercised because the canonical prewarp uses the chebyshev basis, so no pin
    # caught it; the matrix-native basis parity test (test_gpu_basis_column_parity) surfaced it.
    nc = len(c)
    if nc == 0:
        return cp.zeros_like(x)
    out = cp.full(x.shape, c[0], dtype=x.dtype)
    if nc == 1:
        return out
    p_prev = cp.ones_like(x)        # L_0
    p_curr = 1.0 - x               # L_1
    out = out + c[1] * p_curr
    for k in range(2, nc):
        p_next = ((2 * k - 1 - x) * p_curr - (k - 1) * p_prev) / k
        out = out + c[k] * p_next
        p_prev = p_curr
        p_curr = p_next
    return out


_PREWARP_CLENSHAW_GPU = {
    "chebyshev": _cheb_clenshaw_gpu,
    "legendre": _leg_clenshaw_gpu,
    "hermite": _herme_clenshaw_gpu,
    "laguerre": _lag_clenshaw_gpu,
}

# --- GPU port of the orth-FE basis-column evaluation (matrix-native, Piece 2, 2026-06-21) -------------
# Faithful cupy mirror of _orthogonal_univariate_fe._evaluate_basis_column (no-aux, no-replay path):
# the robust heavy-tail axis detection (_hermite_robust._detect_heavy_tail_numpy/_robust_scale/
# _robust_lo_hi) + the per-basis preprocess (z-score / min-max / shift, robust + plain branches) + the
# one-hot Clenshaw eval (the _*_clenshaw_gpu above). Lets the orth-FE candidate matrix be built ON the
# device (operands resident) so it feeds _plugin_mi_classif_batch_cuda_resident with NO H2D -- removing
# the np.median (robust axis) + argsort/reduce (plug-in MI) of the CPU tail. Constants mirror
# _hermite_robust (K=3, OUTER_K=10, GAP=3, MAX_FRAC=0.20). cp.median/percentile match np to fp round-off;
# parity is asserted by test_gpu_basis_column_parity (uniform/gaussian/heavytail/skewed x 4 bases).
# These MIRROR hermite_fe._hermite_robust._ROBUST_AXIS_* (NOT imported: the GPU module avoids a top-level
# hermite_fe import -- the cycle the in-function lazy imports work around). Keep in sync; the drift is
# guarded by tests/feature_selection/gpu/test_gpu_cpu_robust_constants_in_sync.py.
_GPU_ROBUST_AXIS_K = 3.0
_GPU_ROBUST_AXIS_OUTER_K = 10.0
_GPU_ROBUST_AXIS_GAP = 3.0
_GPU_ROBUST_AXIS_MAX_FRAC = 0.20


def _gpu_robust_scale(cp, xf, med):
    """cupy mirror of _hermite_robust._robust_scale: 1.4826*MAD, IQR/1.349 fallback, 0.0 if degenerate."""
    mad = float(cp.median(cp.abs(xf - med)))
    scale = 1.4826 * mad
    if scale > 1e-12:
        return scale
    q25, q75 = (float(v) for v in cp.percentile(xf, cp.asarray([25.0, 75.0])))
    iqr_scale = (q75 - q25) / 1.349
    return iqr_scale if iqr_scale > 1e-12 else 0.0


def _gpu_detect_heavy_tail(cp, xf):
    """cupy mirror of _hermite_robust._detect_heavy_tail_numpy (the n>=3000 oracle path)."""
    if xf.size < 8:
        return False
    med = float(cp.median(xf))
    scale = _gpu_robust_scale(cp, xf, med)
    if scale <= 1e-12:
        return False
    dev = cp.abs(xf - med)
    thr = _GPU_ROBUST_AXIS_OUTER_K * scale
    outer_mask = dev > thr
    n_outer = int(cp.count_nonzero(outer_mask))
    if n_outer == 0 or n_outer > _GPU_ROBUST_AXIS_MAX_FRAC * xf.size:
        return False
    bulk_edge = float(dev[~outer_mask].max())
    outer_min = float(dev[outer_mask].min())
    return (outer_min / max(bulk_edge, 1e-12)) >= _GPU_ROBUST_AXIS_GAP


def _gpu_robust_lo_hi(cp, x, xf, med):
    scale = _gpu_robust_scale(cp, xf, med)
    if scale <= 1e-12:
        return float(cp.min(xf)), float(cp.max(xf))
    return med - _GPU_ROBUST_AXIS_K * scale, med + _GPU_ROBUST_AXIS_K * scale


def _gpu_basis_preprocess(cp, x, basis, *, robust: bool):
    """cupy mirror of the per-basis preprocess fit (_preprocess_zscore/minmax/shift). Returns z (device).
    ``robust`` is the resolved (_robust_axis_enabled() AND _gpu_detect_heavy_tail) decision from the caller."""
    xf = x[cp.isfinite(x)]
    if basis == "hermite":  # z-score
        if robust:
            center = float(cp.median(xf))
            lo, hi = _gpu_robust_lo_hi(cp, x, xf, center)
            std = (hi - lo) / 6.0
            std = std if std > 1e-12 else (float(cp.std(xf)) + 1e-12)
            return cp.clip((x - center) / std, -6.0, 6.0)
        mean = float(cp.mean(x)); std = float(cp.std(x)) + 1e-12
        return (x - mean) / std
    if basis in ("legendre", "chebyshev"):  # min-max -> [-1, 1]
        if robust:
            med = float(cp.median(xf))
            lo, hi = _gpu_robust_lo_hi(cp, x, xf, med)
            span = hi - lo + 1e-12
            return cp.clip(2.0 * (x - lo) / span - 1.0, -1.0, 1.0)
        lo = float(cp.min(x)); hi = float(cp.max(x)); span = hi - lo + 1e-12
        return 2.0 * (x - lo) / span - 1.0
    if basis == "laguerre":  # shift to >= 0
        if robust:
            med = float(cp.median(xf))
            lo, hi = _gpu_robust_lo_hi(cp, x, xf, med)
            upper = hi - lo
            return cp.clip(x - lo + 1e-9, 0.0, upper + 1e-9)
        lo = float(cp.min(x))
        return x - lo + 1e-9
    raise ValueError(f"basis {basis!r} not GPU-ported")


def _gpu_evaluate_basis_column(cp, x, basis, degree, *, robust_axis: bool):
    """Device port of _evaluate_basis_column (no-aux / no-replay path). ``x`` is an (n,) cupy float64
    operand; returns the (n,) cupy float64 basis-column values. ``robust_axis`` = _robust_axis_enabled()
    (host env, passed in to avoid a per-call import). Heavy-tail is detected on-device. Raises for an
    unported basis so the caller falls back to the host _evaluate_basis_column (never a correctness loss)."""
    xf64 = x.astype(cp.float64)
    use_robust = bool(robust_axis) and _gpu_detect_heavy_tail(cp, xf64[cp.isfinite(xf64)])
    z = cp.ascontiguousarray(_gpu_basis_preprocess(cp, xf64, basis, robust=use_robust), dtype=cp.float64)
    clen = _PREWARP_CLENSHAW_GPU.get(basis)
    if clen is None:
        raise ValueError(f"basis {basis!r} not GPU-ported")
    coef = [0.0] * (int(degree) + 1)
    coef[int(degree)] = 1.0
    return clen(cp, z, coef)


# --- BATCHED device basis build (matrix-native Piece 3b, 2026-06-21) ---------------------------------
# Vectorized port of the per-column _gpu_evaluate_basis_column: process ALL columns of a (basis, robust)
# group in ONE preprocess + ONE Clenshaw call per degree (axis=0 stats over the (n, g) submatrix),
# killing the per-column cupy launch overhead that made the per-column loop perf-lose at high feature
# count (p200). Bit-equivalent to the per-column path (same math, just vectorised); guarded by
# test_gpu_basis_column_parity's batched leg.

def _gpu_robust_scale_batched(cp, M, med):
    """Per-column (axis=0) robust scale over finite (n, g) M: 1.4826*MAD, IQR/1.349 fallback, 0 degenerate."""
    mad = cp.median(cp.abs(M - med), axis=0)
    scale = 1.4826 * mad
    q = cp.percentile(M, cp.asarray([25.0, 75.0]), axis=0)  # (2, g)
    iqr = (q[1] - q[0]) / 1.349
    return cp.where(scale > 1e-12, scale, cp.where(iqr > 1e-12, iqr, 0.0))


def _gpu_robust_lo_hi_batched(cp, M, med, scale):
    deg = scale <= 1e-12
    lo = cp.where(deg, M.min(axis=0), med - _GPU_ROBUST_AXIS_K * scale)
    hi = cp.where(deg, M.max(axis=0), med + _GPU_ROBUST_AXIS_K * scale)
    return lo, hi


def _gpu_detect_heavy_tail_batched(cp, M):
    """Vectorized per-column heavy-tail over a FINITE (n, K) matrix (caller skips non-finite cols).
    Returns (K,) cupy bool, matching _gpu_detect_heavy_tail per column."""
    n = M.shape[0]
    med = cp.median(M, axis=0)
    scale = _gpu_robust_scale_batched(cp, M, med)
    dev = cp.abs(M - med)
    outer = dev > (_GPU_ROBUST_AXIS_OUTER_K * scale)
    n_outer = outer.sum(axis=0)
    bulk_edge = cp.where(outer, -cp.inf, dev).max(axis=0)
    outer_min = cp.where(outer, dev, cp.inf).min(axis=0)
    gap_ok = (outer_min / cp.maximum(bulk_edge, 1e-12)) >= _GPU_ROBUST_AXIS_GAP
    return (
        (scale > 1e-12) & (n_outer > 0)
        & (n_outer <= _GPU_ROBUST_AXIS_MAX_FRAC * n) & gap_ok
    )


def _gpu_basis_preprocess_batched(cp, M, basis, *, robust):
    """Vectorized per-basis preprocess over (n, g) M (all g columns share basis + robust). Returns Z (n, g)."""
    if basis == "hermite":  # z-score
        if robust:
            center = cp.median(M, axis=0)
            scale = _gpu_robust_scale_batched(cp, M, center)
            lo, hi = _gpu_robust_lo_hi_batched(cp, M, center, scale)
            std = (hi - lo) / 6.0
            std = cp.where(std > 1e-12, std, M.std(axis=0) + 1e-12)
            return cp.clip((M - center) / std, -6.0, 6.0)
        mean = M.mean(axis=0); std = M.std(axis=0) + 1e-12
        return (M - mean) / std
    if basis in ("legendre", "chebyshev"):  # min-max -> [-1, 1]
        if robust:
            med = cp.median(M, axis=0)
            scale = _gpu_robust_scale_batched(cp, M, med)
            lo, hi = _gpu_robust_lo_hi_batched(cp, M, med, scale)
            span = hi - lo + 1e-12
            return cp.clip(2.0 * (M - lo) / span - 1.0, -1.0, 1.0)
        lo = M.min(axis=0); hi = M.max(axis=0); span = hi - lo + 1e-12
        return 2.0 * (M - lo) / span - 1.0
    if basis == "laguerre":  # shift -> >= 0
        if robust:
            med = cp.median(M, axis=0)
            scale = _gpu_robust_scale_batched(cp, M, med)
            lo, hi = _gpu_robust_lo_hi_batched(cp, M, med, scale)
            upper = hi - lo
            return cp.clip(M - lo + 1e-9, 0.0, upper + 1e-9)
        lo = M.min(axis=0)
        return M - lo + 1e-9
    raise ValueError(f"basis {basis!r} not GPU-ported")


def _gpu_evaluate_basis_matrix(cp, M, bases, degrees, *, robust_axis):
    """BATCHED device build. ``M`` is a finite (n, K) cupy operand matrix; ``bases`` a per-column basis
    list; ``degrees`` the degree sequence. Groups columns by (basis, robust-decision) and runs the
    preprocess + one-hot Clenshaw VECTORISED per group/degree. Returns ``(cand_matrix (n, total), meta)``
    where ``meta`` is a list of ``(col_idx, basis, degree)`` aligned with the candidate columns (any column
    whose basis is not GPU-ported is dropped -- the caller host-fallbacks those). ``(None, [])`` if empty."""
    n, K = M.shape
    if robust_axis:
        heavy_host = cp.asnumpy(_gpu_detect_heavy_tail_batched(cp, M))
    else:
        heavy_host = np.zeros(K, dtype=bool)
    groups: dict = {}
    for ci in range(K):
        groups.setdefault((bases[ci], bool(heavy_host[ci])), []).append(ci)
    cand_blocks: list = []
    meta: list = []
    for (basis, robust), idx in groups.items():
        clen = _PREWARP_CLENSHAW_GPU.get(basis)
        if clen is None:
            continue
        Mg = M[:, idx]
        Zg = _gpu_basis_preprocess_batched(cp, Mg, basis, robust=robust)
        for d in degrees:
            coef = [0.0] * (int(d) + 1)
            coef[int(d)] = 1.0
            cand_blocks.append(clen(cp, Zg, coef))   # (n, len(idx))
            meta.extend((ci, basis, int(d)) for ci in idx)
    if not cand_blocks:
        return None, []
    return cp.ascontiguousarray(cp.concatenate(cand_blocks, axis=1)), meta


def _gpu_batched_abs_corr(cp, cand, y_cont):
    """|Pearson corr| of every column of ``cand`` (n, m) with the (n,) continuous ``y_cont``, on device.
    Degenerate columns (std <= 1e-12) or non-finite results -> -1.0 (so the host argmax skips them, mirroring
    the host router's ``if std(v) < 1e-12: continue``). Same Pearson definition as np.corrcoef(v, y)[0,1]."""
    yc = y_cont - y_cont.mean()
    yn = cp.sqrt((yc * yc).sum())
    cc = cand - cand.mean(axis=0)
    cn = cp.sqrt((cc * cc).sum(axis=0))            # (m,)
    num = (cc * yc[:, None]).sum(axis=0)           # (m,)
    denom = cn * yn
    corr = cp.where(denom > 1e-300, num / denom, 0.0)
    corr = cp.abs(corr)
    finite_ok = cp.isfinite(corr) & (cn > 1e-12)
    return cp.where(finite_ok, corr, -1.0)


def _gpu_route_bases_batched(cp, M, y_cont, candidate_bases, degrees, *, robust_axis):
    """Device port of the no-aux ``basis_route_by_signal`` for ALL columns of finite (n, K) ``M`` at once.
    For each candidate basis it evaluates every column x degree on the device (reusing
    ``_gpu_evaluate_basis_matrix``), computes the |corr| vs ``y_cont``, then runs the EXACT host argmax
    (per-column ``bcorr = max over degrees`` with degenerate-skip, then first-basis-wins argmax over
    ``candidate_bases``). Returns a length-K list of chosen basis names, or ``None`` at a column index where
    no basis produced a usable expansion (caller host-fallbacks that column to basis_route_by_moments).

    Only the corr VALUES come from the GPU (parity-<1e-6); the argmax/tie logic is byte-identical to the host
    router (same loop order, same strict ``>``, same ``bcorr`` init 0.0), so a routing divergence can only
    arise from a genuine <1e-6 near-tie between two bases -- exactly the case the opt-in default guards."""
    K = int(M.shape[1])
    # corr_by_basis[basis] = (K,) host array of bcorr (max over degrees, degenerate-skipped), init 0.0
    corr_by_basis: dict = {}
    for basis in candidate_bases:
        bcorr = np.zeros(K, dtype=np.float64)
        cand, meta = _gpu_evaluate_basis_matrix(cp, M, [basis] * K, list(degrees), robust_axis=robust_axis)
        if cand is not None:
            ac = cp.asnumpy(_gpu_batched_abs_corr(cp, cand, y_cont))   # (len(meta),)
            for j, (ci, _b, _d) in enumerate(meta):
                if ac[j] > bcorr[ci]:
                    bcorr[ci] = ac[j]
        corr_by_basis[basis] = bcorr
    chosen: list = []
    for ci in range(K):
        best_corr = -1.0
        best_basis = None
        for basis in candidate_bases:        # candidate_bases order -> first basis wins a tie (host parity)
            bc = float(corr_by_basis[basis][ci])
            if bc > best_corr:
                best_corr = bc
                best_basis = basis
        chosen.append(best_basis)
    return chosen


def _gpu_apply_prewarp(cp, x, spec):
    """Device port of ``hermite_fe.apply_operand_prewarp``. ``x`` is a device array (native float dtype);
    returns a device float64 column. Raises for any basis/path not ported so the caller falls back to the
    host copy."""
    basis = str(spec["basis"])
    xf = x.astype(cp.float64)
    if basis == "fourier_adaptive":
        pp = dict(spec["preprocess"])
        if str(pp.get("arg", "linear")) == "quadratic":
            z = (xf - float(pp["mean"])) / max(float(pp["std"]), 1e-12)
            u = cp.sign(z) * (z * z)
            axis = (u - float(pp["lo"])) / max(float(pp["span"]), 1e-12)
        else:
            axis = (xf - float(pp["lo"])) / max(float(pp["span"]), 1e-12)
        coef = np.asarray(spec["coef"], dtype=np.float64).reshape(-1)
        out = cp.zeros_like(axis)
        for i, f in enumerate(pp["freqs"]):
            if 2 * i + 1 >= coef.size:
                break
            ang = 2.0 * np.pi * float(f) * axis
            out = out + float(coef[2 * i]) * cp.sin(ang) + float(coef[2 * i + 1]) * cp.cos(ang)
        return out
    clen = _PREWARP_CLENSHAW_GPU.get(basis)
    if clen is None:
        raise ValueError(f"prewarp basis {basis!r} not GPU-ported")
    pp = dict(spec["preprocess"])
    if basis in ("legendre", "chebyshev"):      # _apply_minmax
        span = pp["hi"] - pp["lo"] + 1e-12
        z = 2 * (xf - pp["lo"]) / span - 1
        clip = pp.get("clip")
        if clip is not None:
            z = cp.clip(z, -float(clip), float(clip))
    elif basis == "hermite":                    # _apply_zscore
        z = (xf - pp["mean"]) / max(pp["std"], 1e-12)
        clip = pp.get("clip")
        if clip is not None:
            z = cp.clip(z, -float(clip), float(clip))
    else:                                        # laguerre: _apply_shift
        z = xf - pp["lo"] + 1e-9
        clip = pp.get("clip")
        if clip is not None:
            z = cp.clip(z, 0.0, float(clip) + 1e-9)
    coef = [float(v) for v in np.asarray(spec["coef"], dtype=np.float64).reshape(-1)]
    return clen(cp, z, coef)


def _binary_apply(xp, name, x, y):
    """Apply binary ``name`` to ``(x, y)`` mirroring ``feature_engineering.create_binary_transformations``
    EXACTLY (full catalog minus the temporarily-disabled agm/logn/binom -- see _FULL_BINARY), incl. safe
    div's y==0 branch. Raises ValueError for an unknown/disabled op."""
    # --- minimal ---
    if name == "mul":
        return x * y
    if name == "add":
        return x + y
    if name == "sub":
        return x - y
    if name == "div":
        safe_y = xp.where(y == 0.0, 1e-9, y)
        return x / safe_y
    if name == "max":
        return xp.maximum(x, y)
    if name == "min":
        return xp.minimum(x, y)
    # --- medium ---
    if name == "abs_diff":
        return xp.abs(x - y)
    if name == "hypot":
        return xp.hypot(x, y)
    if name == "signed":            # sign(x)*|y| -- non-symmetrical
        return xp.sign(x) * xp.abs(y)
    if name == "ratio_abs":         # x/(|y|+1) -- non-symmetrical
        return x / (xp.abs(y) + 1.0)
    # --- maximal ---
    if name == "logaddexp":
        return xp.logaddexp(x, y)
    if name == "pow":               # non-symmetrical; negative^frac -> nan, scrubbed downstream (matches np.power)
        return xp.power(x, y)
    if name == "heaviside":
        return xp.heaviside(x, y)
    if name == "greater":
        return (x > y).astype(int)
    if name == "less":
        return (x < y).astype(int)
    if name == "equal":
        return (x == y).astype(int)
    if name == "beta":
        return _xp_special(xp).beta(x, y)
    # GPU-DISABLED(restore): uncomment these + the matching _FULL_BINARY entries + the
    # feature_engineering.py catalog lines to re-enable. agm needs a cupyx/custom kernel (absent here);
    # logn is np.emath.logn (complex base) -- a real GPU form is log(y-ymin+0.1)/log(x-xmin+0.1); binom
    # is deferred with the batch.
    # if name == "agm":
    #     return _xp_special(xp).agm(x, y)
    # if name == "logn":
    #     xs = x - xp.min(x) + 0.1
    #     ys = y - xp.min(y) + 0.1
    #     return xp.log(ys) / xp.log(xs)
    # if name == "binom":
    #     return _xp_special(xp).binom(x, y)
    raise ValueError(f"unknown binary {name!r}")


def _candidate_names(a_label: str = "a", b_label: str = "b") -> list[str]:
    return [
        f"{bop}({ua}({a_label}),{ub}({b_label}))"
        for ua in _MINIMAL_UNARY for ub in _MINIMAL_UNARY for bop in _MINIMAL_BINARY
    ]


# (ua, ub, bop) combo order, matching _candidate_names / _build_candidate_matrix column order.
_COMBOS = [(ua, ub, bop) for ua in _MINIMAL_UNARY for ub in _MINIMAL_UNARY for bop in _MINIMAL_BINARY]

_UNARY_IDX = {u: i for i, u in enumerate(_MINIMAL_UNARY)}
_BINOP_CODE = {"mul": 0, "add": 1, "sub": 2, "div": 3, "max": 4, "min": 5}

# FUSED-GENERATION CUDA RawKernel: one launch computes the WHOLE (n, K) candidate block from the cached
# post-unary columns, replacing the Python loop of ~K separate cupy binary ops + nan_to_num + temporaries.
# Each thread owns one (row, candidate) cell: gather its two operand columns by op-code index, apply the
# binary op (safe-div mirrors the CPU y==0 -> eps branch), scrub non-finite to 0, write row-major (n,K).
_FUSED_GEN_SRC = r"""
extern "C" __global__
void fused_gen(const double* __restrict__ ua, const double* __restrict__ ub,
               const int* __restrict__ ua_idx, const int* __restrict__ ub_idx,
               const int* __restrict__ bop, const long long n, const int K,
               double* __restrict__ out) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (tid >= total) return;
    int c = (int)(tid % (long long)K);
    long long i = tid / (long long)K;
    double x = ua[(long long)ua_idx[c] * n + i];
    double y = ub[(long long)ub_idx[c] * n + i];
    double v;
    switch (bop[c]) {
        case 0: v = x * y; break;
        case 1: v = x + y; break;
        case 2: v = x - y; break;
        case 3: v = x / ((y == 0.0) ? 1e-9 : y); break;
        case 4: v = (x > y) ? x : y; break;
        case 5: v = (x < y) ? x : y; break;
        default: v = 0.0;
    }
    if (isnan(v) || isinf(v)) v = 0.0;
    out[i * (long long)K + c] = v;
}
"""
_FUSED_GEN_KERNEL = None  # module-level singleton (lazy-compiled; never on an instance -> pickle-safe)


def _get_fused_gen_kernel():
    global _FUSED_GEN_KERNEL
    if _FUSED_GEN_KERNEL is None:
        import cupy as cp
        _FUSED_GEN_KERNEL = cp.RawKernel(_FUSED_GEN_SRC, "fused_gen")
    return _FUSED_GEN_KERNEL


# GRAND-FUSION kernel (roadmap #5 + #3, 2026-06-20). One launch computes the joint histograms for ALL
# candidate columns against the original-y AND every shuffled-y, WITHOUT ever materialising the (n,K)
# float candidate matrix, the (n,K) int codes, the (n,K) ``d_base`` index, or the (rows*n*K) flat index
# the cupy noise-gate builds. Each thread owns one (row, candidate) cell: it RE-generates the candidate
# value from the resident post-unary caches (the cheap elementwise gen -- recomputed instead of stored,
# Option F1), bins it ONCE via an upper-bound binary search over that candidate's PRE-COMPUTED exact
# ``cp.percentile`` interior edges (so the binning math is IDENTICAL to ``_gpu_resident_discretize_codes``
# -> selection stays EXACT), then atomic-adds 1 into the joint-histogram slot of EVERY y-vector (original
# + permutations) at ``[p, off_k + code*K_y + y_p[row]]`` (Option F3 atomic histogram). Recomputing gen
# + the bin code ONCE per cell and reusing it across all P1 y-vectors is the key: the per-cell work is
# paid once, only the P1 atomic adds scale with the permutation count.
#
# Edges layout: ``edges`` is (K, nbins-1) row-major -- the INTERIOR percentile edges ``bin_edges[1:-1]``
# for candidate c, exactly what ``cp.searchsorted(edges, val, side='right')`` consumes. The in-kernel
# search reproduces ``side='right'`` (strictly-greater upper bound) so the code equals the cupy path.
# y-vectors layout: ``y_all`` is (P1, n) int32 (row 0 = original y, rows 1.. = the Fisher-Yates shuffles
# built by the SAME host LCG the noise-gate uses) -> the output ``counts`` is (P1, total_size) int64,
# byte-identical to what ``batch_mi_with_noise_gate_cupy`` produces, so it feeds the SAME bit-exact
# ``_mi_from_counts_cpu`` + ``_gate_from_mi`` reduction unchanged.
_FUSED_GEN_BIN_HIST_SRC = r"""
extern "C" __global__
void fused_gen_bin_hist(const double* __restrict__ ua, const double* __restrict__ ub,
                        const int* __restrict__ ua_idx, const int* __restrict__ ub_idx,
                        const int* __restrict__ bop,
                        const double* __restrict__ edges,   // (K, nbins-1) interior edges
                        const long long* __restrict__ col_off,  // (K,) histogram offset of column c
                        const int* __restrict__ y_all,      // (P1, n) int32 y-vectors
                        const long long n, const int K, const int nbins, const int K_y,
                        const int P1, const long long total_size,
                        long long* __restrict__ counts) {   // (P1, total_size) int64, host-zeroed
    // ONE BLOCK PER CANDIDATE COLUMN (roadmap #3 fused SHARED-MEM atomic histogram). The whole column's
    // (P1, nbins, K_y) joint histogram lives in shared memory as int32 (counts <= n < 2^31). Each thread
    // strides over rows: regenerates the candidate value ONCE, bins it ONCE, then does P1 SHARED-MEM
    // atomics (orders of magnitude cheaper than the global-mem atomics the naive version paid) -- the
    // recompute-instead-of-store of Option F1 fused with the shared-atomic histogram of #3. The shared
    // tile is flushed to global counts once at the end. ``hist_size = P1 * nbins * K_y`` int32 elements
    // must fit shared memory; the host launches this path only when it does (else the global-atomic
    // fallback kernel). col_off[c] = c * nbins * K_y, so the column's global slot is contiguous.
    extern __shared__ int sh[];   // (P1, nbins, K_y) int32, dynamic shared
    int c = blockIdx.x;
    if (c >= K) return;
    int hist_size = P1 * nbins * K_y;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    for (int s = tid; s < hist_size; s += nthreads) sh[s] = 0;
    __syncthreads();
    int ne = nbins - 1;
    const double* ec = edges + (long long)c * ne;
    int ua_c = ua_idx[c], ub_c = ub_idx[c], bop_c = bop[c];
    int nbky = nbins * K_y;
    for (long long i = tid; i < n; i += nthreads) {
        double x = ua[(long long)ua_c * n + i];
        double y = ub[(long long)ub_c * n + i];
        double v;
        switch (bop_c) {
            case 0: v = x * y; break;
            case 1: v = x + y; break;
            case 2: v = x - y; break;
            case 3: v = x / ((y == 0.0) ? 1e-9 : y); break;
            case 4: v = (x > y) ? x : y; break;
            case 5: v = (x < y) ? x : y; break;
            default: v = 0.0;
        }
        if (isnan(v) || isinf(v)) v = 0.0;
        // bin: searchsorted(edges[c], v, side='right')
        int lo = 0, hi = ne;
        while (lo < hi) { int mid = (lo + hi) >> 1; if (ec[mid] > v) hi = mid; else lo = mid + 1; }
        int slot = lo * K_y;
        for (int p = 0; p < P1; ++p) {
            int yp = y_all[(long long)p * n + i];
            atomicAdd(&sh[p * nbky + slot + yp], 1);
        }
    }
    __syncthreads();
    // flush shared int32 histogram to the global int64 counts (one column block per y-vector).
    for (int s = tid; s < hist_size; s += nthreads) {
        int p = s / nbky;
        int rem = s - p * nbky;
        counts[(long long)p * total_size + col_off[c] + rem] = (long long)sh[s];
    }
}
"""
_FUSED_GEN_BIN_HIST_KERNEL = None  # module-level singleton (lazy-compiled; pickle-safe)


def _get_fused_gen_bin_hist_kernel():
    global _FUSED_GEN_BIN_HIST_KERNEL
    if _FUSED_GEN_BIN_HIST_KERNEL is None:
        import cupy as cp
        _FUSED_GEN_BIN_HIST_KERNEL = cp.RawKernel(_FUSED_GEN_BIN_HIST_SRC, "fused_gen_bin_hist")
    return _FUSED_GEN_BIN_HIST_KERNEL


def fe_gpu_grand_fusion_enabled() -> bool:
    """Whether the grand-fusion histogram path (never materialise (n,K)) is active. ON unless
    ``MLFRAME_FE_GPU_GRAND_FUSION`` is explicitly falsy (it is the selection-safe data-movement win;
    the non-fused ``grand_fused_pair_mi`` path stays one env flip away as the exact fallback)."""
    return os.environ.get("MLFRAME_FE_GPU_GRAND_FUSION", "1").strip().lower() in ("1", "true", "on", "yes")


_COMBO_IDX_CACHE: dict = {}  # block-tuple -> (ua_idx, ub_idx, bop) device int32; module-level -> pickle-safe
_QLEVELS_CACHE: dict = {}    # (nbins, work-dtype) -> cp.linspace(0,100,nbins+1) device vector; read-only, shared


def _quantile_levels_dev(cp, nbins: int, work):
    """Cached cp.linspace(0,100,nbins+1, dtype=work) percentile-level vector (depends only on (nbins, work),
    rebuilt every chunk otherwise). Read-only -- cp.percentile never mutates it -- so sharing is safe and
    the returned array is byte-identical to the inline linspace (deterministic in its args)."""
    key = (int(nbins), work)
    hit = _QLEVELS_CACHE.get(key)
    if hit is None:
        hit = cp.linspace(0.0, 100.0, int(nbins) + 1, dtype=work)
        _QLEVELS_CACHE[key] = hit
    return hit


def _fused_generate_block(ua_cm, ub_cm, combos_block):
    """Generate the (n, len(combos_block)) candidate matrix for ``combos_block`` in ONE kernel launch.

    ``ua_cm`` / ``ub_cm`` are the (U, n) C-CONTIGUOUS post-unary caches for operands a / b, where
    U = len(_MINIMAL_UNARY) (row u = _UNARY_IDX[name]). This layout lets the kernel address column u
    via ``ua[u*n + i]``; the caller builds them ONCE and reuses across chunks. Returns a row-major
    (n, K) cupy float64 matrix, bit-equal to the cupy elementwise path (same ops, same safe-div, same
    nan_to_num -- validated maxdiff 0)."""
    import cupy as cp

    # Pin the operand-plane row count to the unary set: the kernel does NO bounds check on ua_idx, so a
    # silent row/index drift would be an out-of-bounds device read. Assert it can't.
    assert ua_cm.shape[0] == len(_MINIMAL_UNARY) == ub_cm.shape[0], (ua_cm.shape, ub_cm.shape)
    n = int(ua_cm.shape[1])
    K = len(combos_block)
    # combos_block is always a slice of the module constant _COMBOS, so the int32 index trio depends only on
    # the block contents (identical across every pair at a fixed k_chunk). Cache the device vectors keyed on
    # the block tuple to drop the per-chunk-per-pair list-comp + tiny-H2D allocs (cupy._core.core.array).
    _ck = tuple(combos_block)
    _cc = _COMBO_IDX_CACHE.get(_ck)
    if _cc is None:
        ua_idx = cp.asarray(np.asarray([_UNARY_IDX[ua] for ua, _, _ in combos_block], dtype=np.int32))
        ub_idx = cp.asarray(np.asarray([_UNARY_IDX[ub] for _, ub, _ in combos_block], dtype=np.int32))
        bop = cp.asarray(np.asarray([_BINOP_CODE[bp] for _, _, bp in combos_block], dtype=np.int32))
        _COMBO_IDX_CACHE[_ck] = (ua_idx, ub_idx, bop)
    else:
        ua_idx, ub_idx, bop = _cc
    out = cp.empty((n, K), dtype=cp.float64)
    total = n * K
    threads = 256
    blocks = (total + threads - 1) // threads
    _get_fused_gen_kernel()((blocks,), (threads,), (ua_cm, ub_cm, ua_idx, ub_idx, bop, np.int64(n), np.int32(K), out))
    return out


def _unary_stack_cm(xp, x):
    """(U, n) C-contiguous stack of the minimal unary transforms of ``x`` (U=len(_MINIMAL_UNARY), row u = _UNARY_IDX[name])."""
    return xp.ascontiguousarray(xp.stack([_unary_apply(xp, u, x) for u in _MINIMAL_UNARY], axis=0))

# Per-element GPU working-set multiple for the cupy plug-in MI: the (n, k) cand f64 + argsort int64 +
# X_binned int64 + flat int64 coexist, so budget ~5x the bare cand bytes. Conservative -> avoids the
# n=300k VRAM cliff (measured: unchunked (300k,384) thrashed the 4GB card to 60s).
_GPU_MI_WORKING_MULTIPLE = 5
# Below this n the GPU launch/transfer dominates and the CPU njit grid wins (bench: 20k -> 0.76x,
# 100k -> 1.79x); the dispatcher routes < this to CPU. Provisional crossover; a later sweep can tune it.
_GPU_RESIDENT_MIN_N = 50_000


def _gpu_k_chunk(n: int, *, free_bytes: int | None = None,
                 bytes_per_elem: int = 8, working_multiple: int | None = None,
                 max_cols: int | None = None) -> int:
    """Max candidate columns to materialise+score in ONE on-device batch so the working set stays within
    a fraction of free VRAM -- bounds peak GPU memory like the CPU RAM governor, removing the large-n cliff.

    ``bytes_per_elem`` / ``working_multiple`` describe the per-column resident footprint of the CALLER's
    path. The defaults (8 bytes x ``_GPU_MI_WORKING_MULTIPLE``) match the f64 MI PROTOTYPE
    (gpu_resident_pair_candidate_mi: cand f64 + argsort/X_binned/flat int64 coexisting). The production
    CODES path (gpu_materialise_discretize_codes_host / gpu_discretize_codes_host) only keeps f32 cand +
    f32 transpose + int32 codes + narrow out alive (~4 bytes x ~4), so passing bytes_per_elem=4 +
    a small working_multiple gives it a ~3x WIDER sub-chunk = ~3x FEWER radix/bin/materialise launches
    (the launch+sync+GPU-idle overhead that dominates wall) -- still VRAM-governed by the same 0.25*free
    fraction, and per-column-independent so the codes are bit-identical regardless of chunk boundaries.
    ``max_cols`` caps to the caller's column count (defaults to len(_COMBOS))."""
    import cupy as cp

    if free_bytes is None:
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
    budget = max(1, int(free_bytes * 0.25))
    wm = _GPU_MI_WORKING_MULTIPLE if working_multiple is None else int(working_multiple)
    per_col = max(1, int(n) * int(bytes_per_elem) * wm)
    cap = len(_COMBOS) if max_cols is None else int(max_cols)
    return int(min(cap, max(1, budget // per_col)))


def _build_candidate_matrix(xp, a, b):
    """Generate the full minimal unary x unary x binary candidate grid for operands ``a``, ``b`` as one
    contiguous ``(n, K)`` matrix in array module ``xp``. Non-finite cells -> 0 (the FE scrub). With ``xp``
    = cupy and ``a``/``b`` already device-resident, the WHOLE grid is built on the GPU with no transfer."""
    ua_cache = {u: _unary_apply(xp, u, a) for u in _MINIMAL_UNARY}
    ub_cache = {u: _unary_apply(xp, u, b) for u in _MINIMAL_UNARY}
    n = a.shape[0]
    K = len(_MINIMAL_UNARY) * len(_MINIMAL_UNARY) * len(_MINIMAL_BINARY)
    out = xp.empty((n, K), dtype=xp.float64)
    j = 0
    for ua in _MINIMAL_UNARY:
        for ub in _MINIMAL_UNARY:
            for bop in _MINIMAL_BINARY:
                col = _binary_apply(xp, bop, ua_cache[ua], ub_cache[ub])
                out[:, j] = xp.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
                j += 1
    return out


def cpu_pair_candidate_mi(a: np.ndarray, b: np.ndarray, y_codes: np.ndarray, *, nbins: int = 20):
    """Reference CPU path: build the grid in numpy + score with the production njit batch MI. Returns
    ``(names, mi)`` -- the baseline the GPU-resident path must match (ranking + values to fp round-off)."""
    from .hermite_fe import _plugin_mi_classif_batch_njit

    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    cand = np.ascontiguousarray(_build_candidate_matrix(np, a, b))
    mi = _plugin_mi_classif_batch_njit(cand, np.ascontiguousarray(y_codes, dtype=np.int64), nbins)
    return _candidate_names(), np.asarray(mi, dtype=np.float64)


def gpu_resident_pair_candidate_mi(a: np.ndarray, b: np.ndarray, y_codes: np.ndarray, *, nbins: int = 20):
    """GPU-RESIDENT path: upload ``a``, ``b``, ``y`` ONCE, build the whole candidate grid on the device,
    and score it in ONE big-k batch-MI call -- the array never round-trips per candidate. Returns
    ``(names, mi)`` with ``mi`` brought back to host (the only D2H, a (K,) vector). Raises if cupy is
    unavailable (callers gate on :func:`fe_gpu_resident_enabled` + availability)."""
    import cupy as cp

    from . import hermite_fe as _hf  # noqa: F401 -- full-init the parent first so the direct
    # ``_hermite_fe_mi`` import below can't trip the _ensure_cuda_kernels back-import cycle.
    from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

    # bench-attempt-rejected (2026-06-20, "eliminate ALL f64 in the GPU FE MI path" on the GTX 1050 Ti):
    # generating + scoring this whole chain in float32 (operands/fused-gen/binning/MI all f32) gave only
    # 1.09x @100k / 1.14x @300k AND FLIPPED the winner at n=300k (div(identity(a),sqrt(b)) ->
    # mul(sqr(a),reciproc(b)) -- both spell a**2/b, a near-tie cluster the f32 round-off reorders), i.e.
    # selection instability for a small win. The f64 1/32 penalty does NOT live in the MI math (math-only
    # f32 = 1.00x; see _hermite_fe_mi._plugin_mi_classif_batch_cuda note) but in the BINNING sort, which is
    # captured separately via MLFRAME_FE_GPU_BINNING_DTYPE. So this path stays f64. (proto
    # D:/Temp/_bench_resident_f32.py)
    a_gpu = cp.asarray(a, dtype=cp.float64)   # the ONE H2D of the raw operands
    b_gpu = cp.asarray(b, dtype=cp.float64)
    n = int(a_gpu.shape[0])
    y_i64 = np.ascontiguousarray(y_codes, dtype=np.int64)
    # (U, n) unary caches (U=len(_MINIMAL_UNARY)) stay resident + reused across every chunk; the FUSED kernel generates
    # each candidate CHUNK in ONE launch (vs a Python loop of ~K cupy binary ops + nan_to_num + temps --
    # bit-equal, ~15x faster generation). Only the chunk matrix is bounded, so peak VRAM is governed.
    ua_cm = _unary_stack_cm(cp, a_gpu)
    ub_cm = _unary_stack_cm(cp, b_gpu)
    y_gpu = cp.asarray(y_i64)   # upload y ONCE, reused across chunks (the host wrapper re-H2D'd it per chunk)
    # y's min/max are a fit-constant -> compute ONCE here (one cp.min/max + one scalar D2H) and pass into the
    # resident MI for every chunk, instead of the resident MI recomputing them per chunk (nsys 2026-06-22:
    # that per-chunk recompute was the #1 source of the cp.max reductions + tiny D2H syncs). Bit-identical.
    _ymm = cp.asnumpy(cp.stack((cp.min(y_gpu), cp.max(y_gpu))))
    _ymin = int(_ymm[0]); _ncls = int(_ymm[1]) - _ymin + 1
    k_chunk = _gpu_k_chunk(n)
    mi_parts: list[np.ndarray] = []
    for start in range(0, len(_COMBOS), k_chunk):
        block = _COMBOS[start:start + k_chunk]
        cand = _fused_generate_block(ua_cm, ub_cm, block)   # one-launch fused generation
        # Both operands are already device-resident (cand from the fused kernel, y_gpu uploaded once), so the
        # H2D-free resident MI scores the chunk with no per-chunk transfer (bit-identical to the host-input
        # variant -- test_resident_batch_cuda_matches_host_input pins maxdiff 0). y_min/n_classes hoisted.
        mi_parts.append(np.asarray(
            _plugin_mi_classif_batch_cuda_resident(cand, y_gpu, nbins, y_min=_ymin, n_classes=_ncls),
            dtype=np.float64))
        del cand
    return _candidate_names(), np.concatenate(mi_parts) if mi_parts else np.empty(0)


def _sortfree_mi_gpu(cand_gpu, y_i64, nbins, *, sub: int = 4096):
    """On-device APPROXIMATE plug-in MI for an (n, k) cupy candidate block, with NO sort: bin via a
    monotone tail-compressed (``sign*log1p``, rank-preserving so equi-frequency quantiles are invariant)
    equi-width sub-histogram -> CDF -> quantile edges, then the same joint-histogram MI. Spearman ~0.999
    vs the exact argsort MI but ~4.4x faster on the binning step -- used only to PRESCREEN candidates."""
    import math

    import cupy as cp

    n, k = cand_gpu.shape
    Xt = cp.sign(cand_gpu) * cp.log1p(cp.abs(cand_gpu))   # monotone -> preserves ranks/quantiles
    mn = Xt.min(axis=0); mx = Xt.max(axis=0)
    rng = cp.where(mx > mn, mx - mn, 1.0)
    sb = cp.minimum(((Xt - mn) / rng * sub).astype(cp.int32), sub - 1)
    hist = cp.bincount((cp.arange(k)[None, :] * sub + sb).ravel(), minlength=k * sub).reshape(k, sub).astype(cp.float64)
    cdf = cp.cumsum(hist, axis=1)
    targets = cp.arange(1, nbins) / nbins * n
    yg = cp.asarray(y_i64); ymin = int(yg.min()); yg = yg - ymin; nc = int(yg.max()) + 1
    Xb = cp.empty((n, k), dtype=cp.int64)
    for j in range(k):
        e = cp.searchsorted(cdf[j], targets, side="left")
        Xb[:, j] = cp.searchsorted(e.astype(cp.int32), sb[:, j], side="right")
    flat = ((cp.arange(k)[None, :] * nbins + Xb) * nc + yg[:, None]).ravel()
    h = cp.bincount(flat, minlength=k * nbins * nc).reshape(k, nbins, nc).astype(cp.float64)
    hx = h.sum(2); hy = h.sum(1); ln = math.log(n); m = h > 0
    term = (h / n) * (cp.log(cp.where(m, h, 1.0)) + ln
                      - cp.log(cp.where(hx > 0, hx, 1.0))[:, :, None]
                      - cp.log(cp.where(hy > 0, hy, 1.0))[:, None, :])
    return cp.maximum(cp.where(m, term, 0.0).sum(axis=(1, 2)), 0.0)


def gpu_resident_pair_candidate_mi_fast(a, b, y_codes, *, nbins: int = 20, refine_k: int = 48):
    """APPROXIMATE-with-exact-head GPU pair MI (opt-in, NOT the default): prescreen ALL candidates with
    the sort-free MI (no O(n log n) sort), then re-score only the top ``refine_k`` with the EXACT argsort
    MI. The true winner is EMPIRICALLY preserved (validated 6/6 seeds @100k) because it sits in the
    high-approx-MI tie cluster of equivalent a**2/b spellings, normally well within top-K -- but this is
    NOT guaranteed: if the prescreen mis-ranks the true winner below ``refine_k`` the returned argmax is
    wrong. So this is an approximate fast mode; the exact contract uses ``gpu_resident_pair_candidate_mi``
    (the dispatcher's default). Returns ``(names, mi)``: top-K entries carry EXACT MI, the tail carries
    the (Spearman ~0.999) approx MI."""
    import cupy as cp

    from . import hermite_fe as _hf  # noqa: F401 -- full-init parent before the _hermite_fe_mi import
    from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda

    a_gpu = cp.asarray(a, dtype=cp.float64)
    b_gpu = cp.asarray(b, dtype=cp.float64)
    n = int(a_gpu.shape[0])
    y_i64 = np.ascontiguousarray(y_codes, dtype=np.int64)
    ua_cache = {u: _unary_apply(cp, u, a_gpu) for u in _MINIMAL_UNARY}
    ub_cache = {u: _unary_apply(cp, u, b_gpu) for u in _MINIMAL_UNARY}

    def _col(idx):
        ua, ub, bop = _COMBOS[idx]
        return cp.nan_to_num(_binary_apply(cp, bop, ua_cache[ua], ub_cache[ub]), nan=0.0, posinf=0.0, neginf=0.0)

    # PRESCREEN: sort-free approx MI over all candidates, VRAM-chunked.
    k_chunk = _gpu_k_chunk(n)
    approx = np.empty(len(_COMBOS), dtype=np.float64)
    for start in range(0, len(_COMBOS), k_chunk):
        idxs = range(start, min(start + k_chunk, len(_COMBOS)))
        block = cp.empty((n, len(idxs)), dtype=cp.float64)
        for jj, idx in enumerate(idxs):
            block[:, jj] = _col(idx)
        approx[start:start + len(idxs)] = cp.asnumpy(_sortfree_mi_gpu(block, y_i64, nbins))
        del block
    # REFINE: exact argsort MI on the top-refine_k by approx MI.
    k = min(int(refine_k), len(_COMBOS))
    top = np.argsort(approx)[-k:]
    refine_mat = cp.empty((n, k), dtype=cp.float64)
    for jj, idx in enumerate(top):
        refine_mat[:, jj] = _col(int(idx))
    exact_top = np.asarray(_plugin_mi_classif_batch_cuda(refine_mat, y_i64, nbins), dtype=np.float64)
    mi = approx.copy()
    mi[top] = exact_top   # exact MI for the head, approx for the cheap tail
    return _candidate_names(), mi
    # MEASURED (GTX 1050 Ti, K=384, refine_k=48): exact winner preserved 6/6 seeds @ n=100k, but the
    # end-to-end speedup over the pure-exact GPU path is only ~1.16x @100k / ~1.06x @1M -- NOT the ~2x the
    # MI-only argsort=69% microbench implied. Once candidate GENERATION + the histogram MATH (paid over
    # all 384 in the prescreen) are counted, trimming only the argsort to the top-48 saves less than
    # argsort's in-kernel share. Real + exact, but modest; the bigger lever is cutting generation/math
    # (or a fused sort-free EXACT kernel), not just the sort. Kept as a validated option, not the default.


# RANK-EXACT SORT-FREE QUANTILE EDGES via RADIX-SELECT (roadmap #2, 2026-06-20). cp.percentile bins each
# column with a FULL O(n log n) sort (profiled n=100k/79s: the cp.percentile SORT in this function = 12.9s,
# the #1 production GPU cost) but it only needs the nbins-1 INTERIOR quantile EDGES -- i.e. the ~2*(nbins-1)
# bracketing ORDER STATISTICS per column, not a full ordering. This kernel extracts exactly those order
# statistics with a byte-digit RADIX-SELECT: one block per column, R<=2*(nbins-1) target ranks resolved
# TOGETHER in 8 (float64) / 4 (float32) histogram passes over the column. Each pass reads the column once,
# bins each row's current byte-digit into its rank-window's 256-bucket SHARED-MEM histogram (a row maps to
# exactly ONE active window, found by matching its fixed high-byte prefix), then advances every rank's
# prefix to the bucket holding its target rank. After all passes each rank's exact order-statistic VALUE
# is recovered from the converged key. The float key is the standard order-preserving IEEE transform
# (flip sign bit for positives, all bits for negatives) so the byte order == the float order EXACTLY ->
# the recovered values are BIT-IDENTICAL to the sorted column at those ranks (verified maxdiff 0 on the
# order stats; the codes through cp.searchsorted match cp.percentile maxdiff 0 across all columns).
#
# WHY THIS WINS (the prior estimate said ~8-11 passes ~= sort bandwidth so it may NOT win -- DISPROVEN for
# THIS cupy: cp.percentile uses a comparison MERGE-sort over (value,index) zip-iterators, NOT a radix sort;
# nvprof n=300k K=384: DeviceMergeSort{Merge,BlockSort,Partition} = 65.6% of binning, far above the linear
# bandwidth floor. The 8-pass radix-select read floor measured 16-17x faster than that sort.) MEASURED
# GTX 1050 Ti, R=38, heavy-tailed a**2/b candidates, CUDA-event A/B vs cp.percentile, BIT-IDENTICAL codes
# (maxdiff 0 all columns): f64  100k 1.17x / 300k 1.19x / 1M 2.06x;  f32  100k 2.38x / 300k 2.30x / 1M
# 3.67x. The win GROWS with n (O(n) select vs O(n log n) sort) -- exactly the large-n*K regime the GPU
# binning engages (the auto-router keeps small n on the CPU). The per-row inner window-match loop (R<=40)
# keeps the real time above the bare bandwidth floor; a sorted-prefix binary search is the obvious next
# lever (FUTURE) but the current kernel already beats the sort at every measured size.
#
# EXACTNESS / fallback: the order statistics are exact; the cupy 'linear' interpolation is reproduced in
# float64 EXACTLY (idx=q*(N-1); w=idx-floor(idx); w<0.5 ? below+diff*w : above-diff*(1-w); diff in f64 over
# the (f32-promoted-to-f64 or native-f64) order stats) so the edges -> codes equal cp.percentile bit-for-
# bit. cp.percentile stays the gated exact fallback: MLFRAME_FE_GPU_RADIX_EDGES=0 forces it, and ANY kernel
# failure (compile / launch / shared-mem overflow) falls back to cp.percentile inside this function. The
# shared histogram is R*256 uint32 (R<=40 -> <=40KB < the 48KB default) plus a few small per-rank arrays;
# the host gates the radix path off (-> cp.percentile) if that ever exceeds the device limit.
_RADIX_SELECT_F64_SRC = r"""
#define MAXR 64
extern "C" __global__
void radix_select_f64(const double* __restrict__ data, const long long n, const int K,
                      const long long* __restrict__ ranks, const int R, double* __restrict__ out){
    int col=blockIdx.x, tid=threadIdx.x, nt=blockDim.x;
    extern __shared__ unsigned int sh[];          // W*256 histogram (counts <= n < 2^31 -> uint32 ok)
    __shared__ unsigned long long prefix[MAXR];   // per-rank running key prefix (high bytes fixed)
    __shared__ unsigned long long below[MAXR];    // count strictly below each rank's window
    __shared__ unsigned long long wpref[MAXR];    // distinct active window prefixes (masked)
    __shared__ int rank2w[MAXR];                  // rank -> window index
    __shared__ int W;
    if(tid<R){prefix[tid]=0ULL;below[tid]=0ULL;}
    __syncthreads();
    for(int byte=7;byte>=0;--byte){
        int shift=byte*8;
        unsigned long long hmask=(byte==7)?0ULL:(0xFFFFFFFFFFFFFFFFULL<<((byte+1)*8));
        if(tid==0){int w=0;for(int r=0;r<R;++r){unsigned long long p=prefix[r]&hmask;int f=-1;
            for(int q=0;q<w;++q)if(wpref[q]==p){f=q;break;} if(f<0){wpref[w]=p;rank2w[r]=w;w++;}else rank2w[r]=f;} W=w;}
        __syncthreads();
        int Wl=W;
        for(int s=tid;s<Wl*256;s+=nt)sh[s]=0u;
        __syncthreads();
        for(long long i=tid;i<n;i+=nt){
            double d=data[(long long)col*n+i];unsigned long long u;memcpy(&u,&d,8);  // COLUMN-MAJOR: coalesced
            u=(u&0x8000000000000000ULL)?~u:(u|0x8000000000000000ULL);
            unsigned long long pm=u&hmask;int win=-1;
            for(int q=0;q<Wl;++q)if(wpref[q]==pm){win=q;break;}
            if(win>=0){int dig=(int)((u>>shift)&0xFFULL);atomicAdd(&sh[win*256+dig],1u);}
        }
        __syncthreads();
        if(tid==0){for(int r=0;r<R;++r){int w=rank2w[r];unsigned long long acc=below[r];int chosen=0;long long want=ranks[r];
            for(int b=0;b<256;++b){unsigned long long c=sh[w*256+b];if(acc+c>(unsigned long long)want){chosen=b;break;}acc+=c;}
            below[r]=acc;prefix[r]=(prefix[r]&hmask)|((unsigned long long)chosen<<shift);}}
        __syncthreads();
    }
    if(tid<R){unsigned long long u=prefix[tid];u=(u&0x8000000000000000ULL)?(u&0x7FFFFFFFFFFFFFFFULL):~u;
        double d;memcpy(&d,&u,8);out[tid*K+col]=d;}
}
"""
_RADIX_SELECT_F32_SRC = r"""
#define MAXR 64
extern "C" __global__
void radix_select_f32(const float* __restrict__ data, const long long n, const int K,
                      const long long* __restrict__ ranks, const int R, float* __restrict__ out){
    int col=blockIdx.x, tid=threadIdx.x, nt=blockDim.x;
    extern __shared__ unsigned int sh[];
    __shared__ unsigned int prefix[MAXR];
    __shared__ unsigned long long below[MAXR];
    __shared__ unsigned int wpref[MAXR];
    __shared__ int rank2w[MAXR];
    __shared__ int W;
    if(tid<R){prefix[tid]=0u;below[tid]=0ULL;}
    __syncthreads();
    for(int byte=3;byte>=0;--byte){
        int shift=byte*8;
        unsigned int hmask=(byte==3)?0u:(0xFFFFFFFFu<<((byte+1)*8));
        if(tid==0){int w=0;for(int r=0;r<R;++r){unsigned int p=prefix[r]&hmask;int f=-1;
            for(int q=0;q<w;++q)if(wpref[q]==p){f=q;break;} if(f<0){wpref[w]=p;rank2w[r]=w;w++;}else rank2w[r]=f;} W=w;}
        __syncthreads();
        int Wl=W;
        // bench-attempt-rejected (2026-06-21, elevated nvprof): the per-window stride 256 is bank-aligned
        // (shared_store_transactions_per_request ~6.1), but padding to 257 to de-conflict was SLOWER
        // (264->324ms) -- the kernel is WARP-DIVERGENCE-bound (warp_execution_efficiency ~42% from the
        // per-thread window search), not bank-conflict-bound, and the extra shared bytes cut occupancy.
        for(int s=tid;s<Wl*256;s+=nt)sh[s]=0u;
        __syncthreads();
        for(long long i=tid;i<n;i+=nt){
            float d=data[(long long)col*n+i];unsigned int u;memcpy(&u,&d,4);  // COLUMN-MAJOR: coalesced
            u=(u&0x80000000u)?~u:(u|0x80000000u);
            unsigned int pm=u&hmask;int win=-1;
            for(int q=0;q<Wl;++q)if(wpref[q]==pm){win=q;break;}
            if(win>=0){int dig=(int)((u>>shift)&0xFFu);atomicAdd(&sh[win*256+dig],1u);}
        }
        __syncthreads();
        if(tid==0){for(int r=0;r<R;++r){int w=rank2w[r];unsigned long long acc=below[r];int chosen=0;long long want=ranks[r];
            for(int b=0;b<256;++b){unsigned long long c=sh[w*256+b];if(acc+c>(unsigned long long)want){chosen=b;break;}acc+=c;}
            below[r]=acc;prefix[r]=(prefix[r]&hmask)|((unsigned int)chosen<<shift);}}
        __syncthreads();
    }
    if(tid<R){unsigned int u=prefix[tid];u=(u&0x80000000u)?(u&0x7FFFFFFFu):~u;float d;memcpy(&d,&u,4);out[tid*K+col]=d;}
}
"""
_RADIX_SELECT_F64_KERNEL = None  # module-level singletons (lazy-compiled; never on an instance -> pickle-safe)
_RADIX_SELECT_F32_KERNEL = None
_RADIX_SELECT_THREADS = 512  # measured sweet spot on GTX 1050 Ti (256/512/1024); FUTURE: kernel_tuning_cache
_RADIX_SELECT_MAXR = 64      # must match MAXR in the kernel sources


def _get_radix_select_kernel(is_f32: bool):
    global _RADIX_SELECT_F64_KERNEL, _RADIX_SELECT_F32_KERNEL
    import cupy as cp
    if is_f32:
        if _RADIX_SELECT_F32_KERNEL is None:
            _RADIX_SELECT_F32_KERNEL = cp.RawKernel(_RADIX_SELECT_F32_SRC, "radix_select_f32")
        return _RADIX_SELECT_F32_KERNEL
    if _RADIX_SELECT_F64_KERNEL is None:
        _RADIX_SELECT_F64_KERNEL = cp.RawKernel(_RADIX_SELECT_F64_SRC, "radix_select_f64")
    return _RADIX_SELECT_F64_KERNEL


def fe_gpu_radix_edges_enabled() -> bool:
    """Whether the rank-EXACT sort-free radix-select quantile edges replace cp.percentile's full sort.
    ON unless ``MLFRAME_FE_GPU_RADIX_EDGES`` is explicitly falsy (it is bit-identical to cp.percentile in
    the produced codes -- verified maxdiff 0 -- and faster, the win growing with n; cp.percentile stays
    the gated exact fallback one env flip away and the automatic fallback on any kernel failure)."""
    return os.environ.get("MLFRAME_FE_GPU_RADIX_EDGES", "1").strip().lower() in ("1", "true", "on", "yes")


# (n, nbins) fully determine the radix interp gather-indices (bi/ai) and weight (w) -- they are derived
# only from np.linspace(0,100,nbins+1) and n, NOT from the candidate data -- so they are identical for every
# chunk/pair of a fit. Cache the (tiny, (nbins-1,)) device vectors keyed on (n, nbins) to drop the per-chunk
# tiny-H2D allocs (the cupy._core.core.array hotspot). Module-level dict -> not part of the MRMR instance
# pickle (mirrors the other resident-kernel singletons in this module). (n, nbins) take <=2-3 values per fit.
_RADIX_INTERP_CACHE: dict = {}


def _radix_select_interior_edges(cand_gpu, nbins: int):
    """Return the (nbins-1, K) INTERIOR quantile edges of the resident (n, K) cupy ``cand_gpu`` via the
    sort-free radix-select kernel + cupy's exact 'linear' interpolation (reproduced in float64). The edges
    are BIT-IDENTICAL (in the resulting codes) to ``cp.percentile(cand, linspace(0,100,nbins+1))[1:-1]``.
    Returns ``None`` if the radix path is inapplicable (R over the kernel cap, shared-mem over the device
    limit) so the caller uses the cp.percentile fallback. ``cand_gpu`` must be C-contiguous (n, K)."""
    import cupy as cp

    n, K = cand_gpu.shape
    is_f32 = cand_gpu.dtype == cp.float32
    # cupy 'linear' positions for the nbins-1 interior quantiles (q in (0,1)), float64 throughout.
    qfr = np.linspace(0.0, 100.0, int(nbins) + 1)[1:-1] / 100.0   # (nbins-1,) fractions
    idx = qfr * (n - 1)
    bel = np.floor(idx).astype(np.int64)
    abv = np.minimum(bel + 1, n - 1)
    uniq = np.unique(np.concatenate([bel, abv]))                   # the order-statistic ranks to extract
    R = int(uniq.size)
    if R > _RADIX_SELECT_MAXR:
        return None
    # shared-mem budget: R*256 uint32 histogram (host gate vs the device's per-block shared limit).
    shmem = R * 256 * 4
    try:
        dev = cp.cuda.Device()
        sh_limit = int(dev.attributes.get("MaxSharedMemoryPerBlock", 48 * 1024))
    except Exception:
        sh_limit = 48 * 1024
    if shmem > sh_limit:
        return None
    ranks_g = cp.asarray(uniq, dtype=cp.int64)
    osv = cp.empty((R, K), dtype=cand_gpu.dtype)
    ker = _get_radix_select_kernel(is_f32)
    # COLUMN-MAJOR input (nvprof-driven, 2026-06-20): one block/column previously read data[i*K+col] from
    # the (n,K) row-major buffer -> stride-K, gld_efficiency 12.5% (1/8 coalesced) on the dominant n-loop
    # (4 byte-passes x n reads). Transpose to (K,n) C-order so consecutive threads read consecutive memory
    # (data[col*n+i]) -- one transpose pass buys ~8x coalescing across the 4 passes. Values unchanged ->
    # bit-identical order statistics. (The bin_codes step still uses the original (n,K) cand_gpu.)
    data_cm = cp.ascontiguousarray(cand_gpu.T)   # (K, n) C-order = column-major
    ker((K,), (_RADIX_SELECT_THREADS,),
        (data_cm, np.int64(n), np.int32(K), ranks_g, np.int32(R), osv), shared_mem=shmem)
    # cupy 'linear' interpolation, in float64 over the (f32-promoted or native-f64) order stats -- exactly
    # the cupy_percentile_weightnening elementwise kernel (U=float64; below/above promoted; w in float64).
    _ik = (int(n), int(nbins))
    _ic = _RADIX_INTERP_CACHE.get(_ik)
    if _ic is None:
        pos = {int(r): i for i, r in enumerate(uniq)}
        bi = cp.asarray(np.asarray([pos[int(b)] for b in bel], dtype=np.int64))
        ai = cp.asarray(np.asarray([pos[int(a)] for a in abv], dtype=np.int64))
        w = cp.asarray(np.ascontiguousarray(idx - bel))   # float64 weight_above = idx - floor(idx)
        _RADIX_INTERP_CACHE[_ik] = (bi, ai, w)
    else:
        bi, ai, w = _ic
    # Fancy-indexed gathers are already fresh allocations (never aliased to osv); only cast when osv is f32.
    # On the f64 binning path .astype(f64) was a no-op copy (alloc + cast launch x2 per chunk) -- skip it.
    _ab = osv[bi, :]
    _aa = osv[ai, :]
    ab = _ab if _ab.dtype == cp.float64 else _ab.astype(cp.float64)
    aa = _aa if _aa.dtype == cp.float64 else _aa.astype(cp.float64)
    diff = aa - ab
    edges = cp.where(w[:, None] < 0.5, ab + diff * w[:, None], aa - diff * (1.0 - w)[:, None])
    return edges                          # (nbins-1, K) float64


# FUSED PER-COLUMN BINNING (2026-06-20, nvprof-driven). The per-column ``for j in range(K): out[:,j] =
# cp.searchsorted(edges[:,j], col, 'right')`` loop fired K separate searchsorted launches PLUS K int64->
# int32 cast-copies (searchsorted returns int64, ``out`` is int32). nvprof on the n=100k/300k binning path:
# cupy_copy__int64_int32 = 19.2% of GPU time (2304 calls) + cupy_searchsorted_kernel = 11.7% (2304 calls)
# -- ~31% of GPU time in launch overhead + a needless dtype cast. This ONE kernel bins the whole (n,K)
# matrix: each thread takes one element, binary-searches its column's nbins-1 interior edges (upper_bound
# = count of edges <= value = EXACTLY cp.searchsorted(.., side='right')), and writes the int32 code
# directly -- coalesced cand/out (row-major) + coalesced strided edge reads (consecutive threads = adjacent
# columns). BIT-IDENTICAL to the per-column searchsorted; one launch, no int64 intermediate.
# Edges are ALWAYS float64 (cp.percentile and the radix-select both produce f64 edges). The value is
# promoted to double for the compare -- EXACTLY what cp.searchsorted(f64_edges, f32_value) does (it
# upcasts the value to the edges' dtype). Comparing in the value's f32 instead would 1-off at boundaries
# (a downcast of the f64 edge loses precision) -- the bug that broke bit-identity on the first cut.
_BIN_CODES_SRC = r"""
extern "C" __global__
void bin_codes_TYPENAME(const TYPE* __restrict__ cand, const double* __restrict__ edges,
                        const long long n, const int K, const int ne, int* __restrict__ out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (idx >= total) return;
    int col = (int)(idx % (long long)K);
    double v = (double)cand[idx];
    int lo = 0, hi = ne;                       // upper_bound over this column's interior edges
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (edges[(long long)mid * K + col] <= v) lo = mid + 1; else hi = mid;
    }
    out[idx] = lo;                             // = #(edges <= v) = searchsorted(.., 'right')
}
"""

_BIN_CODES_KERNELS: dict = {}


def _get_bin_codes_kernel(dtype):
    """Lazy-compiled (pickle-safe, module-level cache) fused binning RawKernel for f32 / f64."""
    import cupy as cp

    key = "f64" if dtype == cp.float64 else "f32"
    k = _BIN_CODES_KERNELS.get(key)
    if k is None:
        ctype = "double" if key == "f64" else "float"
        src = _BIN_CODES_SRC.replace("TYPENAME", key).replace("TYPE", ctype)
        k = cp.RawKernel(src, "bin_codes_" + key)
        _BIN_CODES_KERNELS[key] = k
    return k


def _searchsorted_codes(cand_gpu, interior_edges):
    """Bin (n,K) ``cand_gpu`` against per-column ascending ``interior_edges`` (ne,K) -> int32 (n,K) codes,
    code = #(interior edges <= value) (== per-column cp.searchsorted side='right'). One fused kernel
    launch (no K searchsorted launches, no int64->int32 cast). Falls back to the per-column loop on any
    kernel failure -- bit-identical either way."""
    import cupy as cp

    n, K = cand_gpu.shape
    try:
        # cand_gpu is already C-contiguous f32 on the production path (RawKernel cp.empty output / cp.asarray
        # of a C-contiguous host slice); cp.ascontiguousarray would still alloc+copy the whole (n,K) matrix
        # (the nvprof cupy_copy__float32_float32 hotspot, 19.7%). The kernel only needs C-order memory -> reuse
        # the buffer when already contiguous (bit-identical bytes); a strided view still gets the safety copy.
        cand_c = cand_gpu if cand_gpu.flags.c_contiguous else cp.ascontiguousarray(cand_gpu)
        edges_c = cp.ascontiguousarray(interior_edges, dtype=cp.float64)  # edges f64 (match cp.searchsorted promotion)
        ne = int(edges_c.shape[0])
        out = cp.empty((n, K), dtype=cp.int32)
        total = n * K
        threads = 256
        blocks = (total + threads - 1) // threads
        _get_bin_codes_kernel(cand_c.dtype)(
            (blocks,), (threads,),
            (cand_c, edges_c, np.int64(n), np.int32(K), np.int32(ne), out),
        )
        return out
    except Exception:
        import logging
        logging.getLogger(__name__).debug("fused bin-codes kernel failed; per-column searchsorted fallback", exc_info=True)
        out = cp.empty((n, K), dtype=cp.int32)
        ec = cp.ascontiguousarray(interior_edges)
        for j in range(K):
            out[:, j] = cp.searchsorted(ec[:, j], cand_gpu[:, j], side="right")
        return out


def _gpu_resident_discretize_codes(cand_gpu, nbins: int):
    """Quantile-bin a RESIDENT (n, K) cupy candidate matrix to ordinal codes ON the GPU. Mirrors
    ``discretize_2d_array_cuda`` -- ``cp.percentile(.., linspace(0,100,nbins+1), axis=0)`` for per-column
    edges + per-column ``cp.searchsorted(edges[1:-1], col, side='right')`` -- but keeps the input + output
    on-device (no H2D of the big candidate matrix, no D2H of codes here), so it chains gen -> discretize ->
    noise-gate without round-trips. Returns a cupy int32 (n, K) codes array (resident).

    DTYPE: the percentile + searchsorted run in the INPUT's native dtype by default -- so the float32 FE
    candidate buffer stays float32 (no up-cast; float32 halves the bandwidth of the dominant full sort
    cp.percentile does and preserves the FE selection, the acceptance bar) while the float64 grand-fused
    MI path stays float64 (bit-identical). ``MLFRAME_FE_GPU_BINNING_DTYPE=float64`` forces the exact f64
    path host-wide (bit-identical to the CPU ``discretize_2d_quantile_batch``, whose ``np.percentile``
    upcasts float32 to float64); ``=float32`` forces f32."""
    import cupy as cp

    # Bin in the input's NATIVE dtype by default (the float32 FE candidate buffer stays float32 -- no
    # up-cast, half the sort bandwidth; the float64 grand-fused MI path stays float64 -- bit-identical).
    # MLFRAME_FE_GPU_BINNING_DTYPE forces a specific working dtype (float64 = the exact CPU-parity fallback).
    forced = os.environ.get("MLFRAME_FE_GPU_BINNING_DTYPE", "").strip().lower()
    if forced in ("float64", "f64", "double"):
        work = cp.float64
    elif forced in ("float32", "f32", "single"):
        work = cp.float32
    else:
        work = cand_gpu.dtype
    if cand_gpu.dtype != work:
        cand_gpu = cand_gpu.astype(work, copy=False)
    n, K = cand_gpu.shape

    # RANK-EXACT SORT-FREE EDGES (roadmap #2): extract just the nbins-1 interior quantile edges via the
    # radix-select kernel instead of cp.percentile's full sort. Bit-identical codes (verified maxdiff 0),
    # faster (win grows with n). Returns None -> cp.percentile fallback (R over cap / shared-mem over the
    # device limit); any kernel exception also falls back. cp.percentile's interior edges are bin_edges[1:-1].
    if fe_gpu_radix_edges_enabled() and n > 0:
        try:
            # Already C-contiguous here (see _searchsorted_codes note); _radix_select_interior_edges does its
            # OWN coalescing transpose internally and only needs C-order input, so skip the redundant full
            # (n,K) f32 copy when contiguous (bit-identical edges -> codes). The KEEP transpose stays inside it.
            _cand_c = cand_gpu if cand_gpu.flags.c_contiguous else cp.ascontiguousarray(cand_gpu)
            interior = _radix_select_interior_edges(_cand_c, int(nbins))
        except Exception:
            import logging
            logging.getLogger(__name__).debug("radix-select edges failed; cp.percentile fallback", exc_info=True)
            interior = None
        if interior is not None:
            return _searchsorted_codes(cand_gpu, interior)

    qs = _quantile_levels_dev(cp, nbins, work)
    if K == 1:
        # CUPY BUG GUARD: cp.percentile(X, axis=0) returns WRONG edges for a single-column (n, 1) array
        # (verified maxdiff ~23 vs numpy; multi-column is exact). A K==1 chunk occurs whenever the last
        # candidate block holds one column, which would silently corrupt that column's codes (breaking the
        # discretize bit-identity). Ravel to 1D where cp.percentile is correct, then restore the shape.
        bin_edges = cp.percentile(cand_gpu.ravel(), qs).reshape(-1, 1)  # (nbins+1, 1)
    else:
        bin_edges = cp.percentile(cand_gpu, qs, axis=0)  # (nbins+1, K)
    return _searchsorted_codes(cand_gpu, bin_edges[1:-1])


# CHUNK-MATERIALISE CUDA RawKernel (2026-06-20). The FE chunk path's #1 CPU hotspot is
# ``_materialise_chunk_njit`` -- it builds the (n, K) float32 candidate matrix by gathering strided
# operand columns ``tv[r, ai]`` / ``tv[r, bi]`` out of a row-major operand table and applying the
# binary op-code table (mlframe.feature_selection.filters._feature_engineering_pairs._pairs_materialise
# ._NJIT_BINARY_OP_CODES). It is MEMORY-BANDWIDTH bound on those gathers, not compute. This kernel does
# the IDENTICAL work on the GPU: each thread owns one (row, candidate) cell, gathers its two operand
# columns by op-code index, applies the binary op, scrubs non-finite -> 0, and writes float32 row-major.
#
# BIT-IDENTICAL to ``_materialise_chunk_njit``: operands are read as float32 (the ``tv`` dtype); mul/add/
# sub/abs_diff are plain float32 ops; max/min/signed propagate NaN exactly (``a+b`` when either is NaN);
# div (op 3) and ratio_abs (op 8) are FLOAT64-PROMOTED then cast back to float32 (matching the njit
# kernel's ``np.float32(np.float64(a)/...)`` -- numba/numpy promote the float64 ``1e-9`` / ``1.0``
# literals); the final nan_to_num(nan=0, +-inf=0) is the same predicate. The op-code numbering is the
# njit table: 0=mul 1=add 2=sub 3=div 4=max 5=min 6=abs_diff 7=signed 8=ratio_abs. ``tv`` is the
# (n, n_operands) row-major float32 operand table; the kernel addresses operand column ``c`` of row
# ``i`` via ``tv[i*n_operands + c]`` (so NO transpose is needed -- it mirrors the njit ``tv[r, ai]``).
_FE_MATERIALISE_SRC = r"""
extern "C" __global__
void fe_materialise(const float* __restrict__ tv,
                    const long long* __restrict__ a_cols,
                    const long long* __restrict__ b_cols,
                    const signed char* __restrict__ ops,
                    const long long n, const long long n_operands, const int K,
                    float* __restrict__ out) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (tid >= total) return;
    int k = (int)(tid % (long long)K);
    long long i = tid / (long long)K;
    long long ai = a_cols[k];
    long long bi = b_cols[k];
    float a = tv[i * n_operands + ai];
    float b = tv[i * n_operands + bi];
    int op = (int)ops[k];
    float v;
    if (op == 0) {            // mul
        v = a * b;
    } else if (op == 1) {     // add
        v = a + b;
    } else if (op == 2) {     // sub
        v = a - b;
    } else if (op == 3) {     // div = _safe_div (2026-06-13 form): exact x/y for y!=0, eps floor only on exact-zero
        v = (float)((double)a / ((b == 0.0f) ? 1e-9 : (double)b));
    } else if (op == 4) {     // max = np.maximum (nan-propagating)
        if (a != a || b != b) v = a + b; else v = (a > b) ? a : b;
    } else if (op == 5) {     // min = np.minimum (nan-propagating)
        if (a != a || b != b) v = a + b; else v = (a < b) ? a : b;
    } else if (op == 6) {     // abs_diff = |a - b|
        v = fabsf(a - b);
    } else if (op == 7) {     // signed = sign(a)*|b| (nan-propagating)
        if (a != a || b != b) {
            v = a + b;
        } else {
            float sgn = (a == 0.0f) ? 0.0f : ((a > 0.0f) ? 1.0f : -1.0f);
            v = sgn * fabsf(b);
        }
    } else {                  // op == 8: ratio_abs = float64-promoted a/(|b|+1)
        v = (float)((double)a / ((double)fabsf(b) + 1.0));
    }
    // np.nan_to_num(nan=0, posinf=0, neginf=0)
    if (isnan(v) || isinf(v)) v = 0.0f;
    out[i * (long long)K + k] = v;
}
"""
_FE_MATERIALISE_KERNEL = None  # module-level singleton (lazy-compiled; never on an instance -> pickle-safe)


def _get_fe_materialise_kernel():
    global _FE_MATERIALISE_KERNEL
    if _FE_MATERIALISE_KERNEL is None:
        import cupy as cp
        _FE_MATERIALISE_KERNEL = cp.RawKernel(_FE_MATERIALISE_SRC, "fe_materialise")
    return _FE_MATERIALISE_KERNEL


def _fe_materialise_block_gpu(tv_gpu, a_cols_block, b_cols_block, ops_block):
    """Generate the (n, len(ops_block)) float32 candidate matrix for the given column blocks in ONE kernel
    launch, RESIDENT on the GPU. ``tv_gpu`` is the (n, n_operands) row-major float32 operand table already
    on the device. ``a_cols_block`` / ``b_cols_block`` (int64) / ``ops_block`` (int8) are host or device
    arrays of length K. Returns a row-major (n, K) cupy float32 matrix, BIT-EQUAL to
    ``_materialise_chunk_njit`` (same float32 ops, same float64-promoted div/ratio_abs, same nan_to_num)."""
    import cupy as cp

    n = int(tv_gpu.shape[0])
    n_operands = int(tv_gpu.shape[1])
    K = int(len(ops_block))
    a_g = cp.asarray(a_cols_block, dtype=cp.int64)
    b_g = cp.asarray(b_cols_block, dtype=cp.int64)
    ops_g = cp.asarray(ops_block, dtype=cp.int8)
    out = cp.empty((n, K), dtype=cp.float32)
    total = n * K
    threads = 256
    blocks = (total + threads - 1) // threads
    _get_fe_materialise_kernel()(
        (blocks,), (threads,),
        (tv_gpu, a_g, b_g, ops_g, np.int64(n), np.int64(n_operands), np.int32(K), out),
    )
    return out


# PINNED D2H STAGING for the out_cand float buffer (2026-06-21, nvprof+paired-microbench driven).
# The downstream survivor/usability reads need the (n,K) float candidate matrix on host, so out_cand is
# unavoidable -- but ``cp.asnumpy(cand)`` copies into the caller's PAGEABLE buffer, which makes cupy stage
# the D2H through an internal pinned bounce buffer at PAGEABLE PCIe bandwidth (the #1 production wall:
# cProfile cupy.get = 9.07s, 321 blocking syncs). DMA'ing the chunk into a PERSISTENT PINNED host buffer
# first, then a plain host->host memcpy into the caller's pageable slice, runs the device transfer at full
# pinned bandwidth. MEASURED GTX 1050 Ti, (100k, blk=1200) f32 = 480MB: the device D2H 143ms->75ms (1.9x);
# end-to-end into a pageable slice incl. the added host memcpy 209ms->130ms (1.6x); the whole materialise+
# bin+codes call (K=1200) 696ms->~575ms with the float path on. The buffer is a module-level singleton
# (never on an instance -> pickle-safe), grown on demand and reused across the 15 canonical chunks.
# bench-attempt-rejected (2026-06-21, prior): DEFERRING the float D2H entirely (out_cand=None + downstream
# recompute) was a 0.98x fit-level WASH because removing an overlapped transfer cuts no wall -- but here we
# do NOT remove it, we make the SAME bytes move faster (pinned DMA), which DOES cut the blocking-sync wall.
_PINNED_D2H_BUF = None  # cupy pinned-memory allocation (raw); reused/grown across chunks


def _pinned_view(n_bytes: int, shape, dtype):
    """A pinned-host numpy view of at least ``n_bytes``, reshaped to ``shape`` (``dtype``). Reuses a
    module-level pinned allocation, growing it on demand. Lets ``cupy.ndarray.get(out=...)`` DMA at full
    pinned PCIe bandwidth instead of cp.asnumpy's pageable bounce-buffer path. Module-level (not on an
    estimator instance) -> never reachable from pickled state."""
    global _PINNED_D2H_BUF
    import cupy as cp

    if _PINNED_D2H_BUF is None or _PINNED_D2H_BUF.mem.size < n_bytes:
        _PINNED_D2H_BUF = cp.cuda.alloc_pinned_memory(int(n_bytes))
    count = int(np.prod(shape))
    return np.frombuffer(_PINNED_D2H_BUF, dtype=dtype, count=count).reshape(shape)


# Operand-table H2D cache (2026-06-21): the FE step's operand table ``transformed_vars`` is the SAME
# array object across all ~15 chunks of a step, but was re-uploaded to the GPU per chunk (and again per
# survivor re-materialise). Cache the device copy by WEAKREF IDENTITY of the host array: reuse while the
# same object is alive (across the step's chunks), re-upload when the step swaps in a new operand table
# (the weakref breaks). NOT keyed on id() -- id reuse after free would false-hit on a different table.
# Pickle-safe (module-global, never on an instance). The data is identical -> candidates/codes/MI/
# selection bit-identical; this only moves the H2D from per-chunk to once-per-step.
_OPERAND_TABLE_CACHE: dict = {"ref": None, "gpu": None}


# GPU-RESIDENT OPERAND TABLE (2026-06-21, phase 1 of the 100%-GPU-resident MRMR FE rewrite, gated).
# The operand table ``transformed_vars`` (n, n_operands) float32 is built on the CPU in
# ``check_prospective_fe_pairs`` (one column per (var, unary)), then ``_resident_operand_table`` H2Ds it to
# the device ONCE per step. Phase 1 removes even that single H2D by building the device mirror's columns ON
# the GPU directly from the resident raw operand inputs (via ``_unary_apply`` -- the same math as the CPU
# ``unary_transformations``), so the materialise consumes a DEVICE array with NO host->device transfer of
# the bulk operand bytes. The CPU ``transformed_vars`` is STILL built (the pair-search inner loops /
# discretize read it on the host -- those move to the GPU in later phases); phase 1 only kills the
# materialise H2D. Operand transforms that are NOT plain GPU unaries (prewarp / gate_med / hermite-poly --
# fitted/special, no straightforward cupy form) are built on the CPU and copied into the resident mirror (a
# few columns); the bulk plain-unary columns are GPU-built. The PREBUILT mirror is registered here by
# weakref-identity of the host ``transformed_vars`` so ``_resident_operand_table`` returns it WITHOUT the
# H2D. Module-global -> never reachable from pickled estimator state. Gated OFF by default
# (``MLFRAME_FE_GPU_RESIDENT_OPERANDS``) until proven 11-green; the CPU / no-CUDA path is unchanged.
_PREBUILT_OPERAND_TABLE: dict = {"ref": None, "gpu": None}


def fe_gpu_resident_operands_enabled() -> bool:
    """Whether the GPU-RESIDENT operand-table build (phase 1) is active. DEFAULT ON (opt-out
    ``MLFRAME_FE_GPU_RESIDENT_OPERANDS=0``). When on (and CUDA present -- the caller guards this and
    falls back on any failure) the operand table's bulk plain-unary columns are produced ON the GPU and
    the materialise consumes the device array with no H2D re-upload; the CPU / no-CUDA path is byte-for-
    byte unchanged (operand table H2D'd as before)."""
    return os.environ.get("MLFRAME_FE_GPU_RESIDENT_OPERANDS", "1").strip().lower() not in ("0", "false", "no", "off")


def register_prebuilt_operand_table(transformed_vars, device_table) -> None:
    """Register a GPU-RESIDENT device mirror ``device_table`` for the host operand table ``transformed_vars``
    (keyed on the host array's weakref identity). ``_resident_operand_table`` then returns ``device_table``
    for that exact host object WITHOUT re-uploading. Pass ``device_table=None`` to clear. The device array
    MUST be a row-major (n, n_operands) C-contiguous float32 cupy array matching ``transformed_vars``'s
    shape (the layout ``_fe_materialise_block_gpu``'s kernel addresses); a mismatch is ignored at lookup."""
    import weakref
    c = _PREBUILT_OPERAND_TABLE
    if device_table is None:
        c["ref"] = None
        c["gpu"] = None
        return
    try:
        c["ref"] = weakref.ref(transformed_vars)
        c["gpu"] = device_table
    except TypeError:
        c["ref"] = None
        c["gpu"] = None


def _prebuilt_operand_table(transformed_vars):
    """The registered GPU-resident device mirror for ``transformed_vars`` iff it matches the host array by
    weakref identity AND shape (n, n_operands); else None. Shape guard so a stale/mismatched mirror can
    never feed the materialise kernel a wrong-width table (out-of-bounds operand-column reads)."""
    c = _PREBUILT_OPERAND_TABLE
    ref = c["ref"]
    if ref is None or c["gpu"] is None:
        return None
    if ref() is not transformed_vars:
        return None
    g = c["gpu"]
    if tuple(g.shape) != tuple(transformed_vars.shape):
        return None
    return g


def _resident_operand_table(cp, transformed_vars):
    """Device (n, n_operands) float32 copy of ``transformed_vars``. When a GPU-RESIDENT mirror was prebuilt
    for this exact host object (phase 1, ``register_prebuilt_operand_table``) it is returned WITH NO H2D --
    the bulk operand bytes were produced on the device. Otherwise the host array is uploaded once per
    distinct object (weakref-identity cache) and reused across a step's chunks; falls back to a plain
    upload if the array is not weakref-able."""
    import weakref
    pre = _prebuilt_operand_table(transformed_vars)
    if pre is not None:
        return pre
    c = _OPERAND_TABLE_CACHE
    ref = c["ref"]
    if ref is not None and ref() is transformed_vars and c["gpu"] is not None:
        return c["gpu"]
    g = cp.asarray(np.ascontiguousarray(transformed_vars, dtype=np.float32))
    try:
        c["ref"] = weakref.ref(transformed_vars)
        c["gpu"] = g  # drops the prior step's device table (refcount -> freed)
    except TypeError:
        c["ref"] = None
        c["gpu"] = None
    return g


def build_resident_operand_table(transformed_vars, col_specs, *, fallback_unaries=()):
    """Build a GPU-RESIDENT (n, n_operands) row-major float32 cupy mirror of the host operand table
    ``transformed_vars``, producing the bulk PLAIN-UNARY columns ON the GPU (via ``_unary_apply`` -- the
    same math the CPU ``unary_transformations`` applied) and COPYING the rest (prewarp / gate_med /
    hermite-poly / any name in ``fallback_unaries`` / any GPU-unbuildable column) from the host array.

    ``col_specs`` is a list aligned with the operand-table columns: each entry is ``(col_idx, raw_vals,
    unary_name)`` where ``raw_vals`` is the host float64 raw operand input the CPU applied ``unary_name`` to
    (or ``None`` for a column with no GPU recipe -> copied from the host). A column is GPU-built iff
    ``raw_vals is not None``, ``unary_name`` is a known plain unary (``_unary_apply`` accepts it, not in
    ``fallback_unaries``). The unary is applied on the GPU in float64 (the dtype the CPU ``tr_func``
    received) then cast to float32 (mirroring the CPU's compute-in-f64-then-store-f32) so the GPU column
    matches the host column to fp round-off. Any per-column GPU failure falls that column back to the host
    copy (never a correctness regression). Returns the device array (already row-major C-contiguous f32);
    the caller registers it via ``register_prebuilt_operand_table``."""
    import cupy as cp

    n, n_operands = transformed_vars.shape
    fb = set(fallback_unaries)
    # Allocate the device mirror WITHOUT uploading the host table: the residency win is precisely that the
    # bulk operand bytes never make the host->device trip. We H2D ONLY the small per-operand RAW inputs (n
    # floats each, cached so each distinct raw operand is uploaded ONCE -- they recur across a var's unaries)
    # and GPU-build the plain columns from them; the FEW non-plain / failed columns are copied from the host
    # one column at a time. Columns with no spec (the unused tail, if any) are zero-filled -- they are never
    # read by the materialise (operand indices are always < the used width), so their content is irrelevant.
    g = cp.zeros((n, n_operands), dtype=cp.float32)
    # ONE-TRANSFER (phase R0, 2026-06-21): batch the DISTINCT raw operands referenced by the GPU-buildable
    # specs into per-dtype host matrices and upload each in ONE H2D, instead of one cp.asarray per distinct
    # raw. Each raw keeps its NATIVE float dtype (we group BY dtype) so the unary still applies in the exact
    # dtype the CPU ``tr_func`` saw -> the GPU column matches the host column to fp round-off (the invariant
    # the per-operand path enforced). Values are byte-identical; only the H2D packaging changes. Per-dtype
    # grouping means uniform-dtype fits (the common case: all-pandas f64 -> 14 distinct raws) collapse to ONE
    # upload. The device column is a strided VIEW into the group matrix -- _unary_apply is elementwise, so the
    # result equals the contiguous-input result bit-for-bit. Any group/build failure falls that column back to
    # the host copy below (never a correctness regression).
    _raw_slot: dict = {}   # id(raw_vals) -> (dtype_key, slot_in_group)
    _groups: dict = {}     # dtype_key -> list[host column in native float dtype]
    for _spec_t in col_specs:
        col_idx, raw_vals, unary_name = _spec_t[0], _spec_t[1], _spec_t[2]
        if raw_vals is not None and unary_name not in fb:
            _rk = id(raw_vals)
            if _rk not in _raw_slot:
                _rv = np.ascontiguousarray(raw_vals)
                if not np.issubdtype(_rv.dtype, np.floating):
                    _rv = _rv.astype(np.float64)  # CPU tr_func on a non-float would also promote
                _dk = _rv.dtype.str
                grp = _groups.setdefault(_dk, [])
                _raw_slot[_rk] = (_dk, len(grp))
                grp.append(_rv)
    _dev_groups: dict = {}  # dtype_key -> device (n, m) array (ONE H2D per dtype group)
    for _dk, cols in _groups.items():
        try:
            _host = (np.ascontiguousarray(np.column_stack(cols)) if len(cols) > 1
                     else np.ascontiguousarray(cols[0]).reshape(-1, 1))
            _dev_groups[_dk] = cp.asarray(_host)
        except Exception:
            _dev_groups[_dk] = None
    n_gpu = 0
    n_cpu = 0
    for _spec_t in col_specs:
        col_idx, raw_vals, unary_name = _spec_t[0], _spec_t[1], _spec_t[2]
        _payload = _spec_t[3] if len(_spec_t) > 3 else None  # R1: prewarp GPU-apply payload (or None)
        gpu_built = False
        if raw_vals is not None and unary_name not in fb:
            try:
                _dk, _slot = _raw_slot[id(raw_vals)]
                _dev = _dev_groups.get(_dk)
                if _dev is not None:
                    x = _dev[:, _slot]   # native-dtype device view of this raw operand (no per-operand H2D)
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        if _payload is not None and _payload.get("kind") == "prewarp":
                            # R1: APPLY the prewarp on the device (preprocess + Clenshaw) from the raw + spec,
                            # mirroring hermite_fe.apply_operand_prewarp -- no host-column H2D. _gpu_apply_prewarp
                            # raises for any unported basis -> falls to the host copy below (bit-exact).
                            col = _gpu_apply_prewarp(cp, x, _payload["spec"])
                        else:
                            col = _unary_apply(cp, unary_name, x)
                    # nan_to_num is NOT applied here: the CPU operand table stores the raw unary output
                    # (un-scrubbed) too -- the materialise kernel scrubs NaN/inf inline -> bit-equal.
                    g[:, col_idx] = col.astype(cp.float32)
                    gpu_built = True
            except Exception:
                gpu_built = False
        if not gpu_built:
            # Non-plain (prewarp / gate_med / poly) or failed: copy just THIS column from the host (a single
            # (n,) f32 H2D, not the whole table) so the device column equals the CPU bytes exactly.
            g[:, col_idx] = cp.asarray(np.ascontiguousarray(transformed_vars[:, col_idx], dtype=np.float32))
            n_cpu += 1
        else:
            n_gpu += 1
    return cp.ascontiguousarray(g), n_gpu, n_cpu


def gpu_materialise_discretize_codes_host(
    transformed_vars: np.ndarray, a_cols: np.ndarray, b_cols: np.ndarray, op_codes: np.ndarray,
    nbins: int, *, dtype=np.int8, out_cand: np.ndarray | None = None,
) -> np.ndarray:
    """GPU fast path for the FE chunk's MATERIALISE + BINNING. Uploads the operand table
    ``transformed_vars`` (n, n_operands) float32 ONCE, then for each VRAM-bounded column block: generates
    the float32 candidate matrix on the GPU (``_fe_materialise_block_gpu`` -- bit-equal to
    ``_materialise_chunk_njit``) and quantile-bins it RESIDENT (``_gpu_resident_discretize_codes``,
    bit-equal to ``discretize_2d_quantile_batch``). Returns the (n, K) ``dtype`` codes (BIT-IDENTICAL to
    the CPU njit-materialise -> ``gpu_discretize_codes_host`` pipeline, verified maxdiff 0).

    The candidate matrix is generated + binned RESIDENT (the int codes are the only mandatory D2H). But the
    downstream FE survivor / usability / ext-val stages read the CONTINUOUS candidate columns out of the
    chunk buffer, so the caller passes ``out_cand`` (the ``chunk_buffer[:, :K]`` float32 view) to receive
    the materialised float candidate matrix as well -- this replaces the CPU njit materialise with the GPU
    one (the bandwidth-bound strided-gather op the GPU is good at) while keeping the buffer the rest of the
    pipeline expects. Pass ``out_cand=None`` to skip the float D2H (codes-only, when no downstream
    continuous read is needed). Inputs are finite by construction (the kernel scrubs NaN/inf inline)."""
    import cupy as cp

    # Drop any stale handoff/deferred-fill from a PRIOR chunk before producing this one (releases that
    # chunk's pinned device codes; each chunk's dispatch should already have consumed/cleared its own).
    clear_resident_codes_handoff()
    tv = np.ascontiguousarray(transformed_vars, dtype=np.float32)
    a_cols = np.ascontiguousarray(a_cols, dtype=np.int64)
    b_cols = np.ascontiguousarray(b_cols, dtype=np.int64)
    op_codes = np.ascontiguousarray(op_codes, dtype=np.int8)
    n = int(tv.shape[0])
    K = int(a_cols.shape[0])
    # Operand table H2D cached per-step by weakref identity (same transformed_vars across the step's
    # chunks -> uploaded ONCE, not per chunk). Pass the ORIGINAL array so the weakref tracks it.
    tv_gpu = _resident_operand_table(cp, transformed_vars)
    out = np.empty((n, K), dtype=dtype)
    # RESIDENT-CODES HANDOFF (gated, default OFF): keep the on-device int codes in ONE (n, K) resident
    # cupy array so the noise-gate's resident-CUDA path can consume them DIRECTLY -- skipping the codes'
    # GPU->host (here) ->GPU (the gate's H2D) round-trip. The host ``out`` is STILL filled (the CPU /
    # analytic / opt-out / SU / any-failure dispatch branches need it and it is the safe fallback), so this
    # only ADDS a resident copy when the gate is on; the round-trip is skipped only when the resident gate
    # is the actual consumer (it matches ``out`` by identity via the module handoff).
    _resident_codes_on = fe_gpu_resident_codes_enabled()
    dev_codes = cp.empty((n, K), dtype=cp.dtype(np.dtype(dtype))) if _resident_codes_on else None
    # DEFER the host-codes D2H when the resident handoff is on: the host ``out`` is filled LAZILY (only if a
    # host consumer reads it -- see ensure_host_codes_filled) instead of eagerly per block. This skips the
    # (n, K) codes D2H (the canonical fit's single largest D2H) whenever the resident gate consumes the
    # device codes. Needs dev_codes (the resident copy) to fill from, so it is only active with it.
    _defer_host_codes = bool(dev_codes is not None and fe_gpu_defer_host_codes_enabled())
    # CODES path footprint is f32 (cand + transpose + int32 codes + narrow out), ~4B x ~4 working copies --
    # NOT the f64 MI prototype's 8x5. Budget for that so the VRAM sub-chunk is ~3x wider -> ~3x fewer
    # radix/bin/materialise launches (cuts the launch+sync+GPU-idle overhead). working_multiple=6 keeps a
    # safe margin over the honest ~4 on the 4GB card; still 0.25*free VRAM-governed; per-column-independent
    # so codes are bit-identical regardless of chunk boundary.
    k_chunk = _gpu_k_chunk(n, bytes_per_elem=4, working_multiple=6, max_cols=K)
    for start in range(0, K, k_chunk):
        stop = min(start + k_chunk, K)
        cand = _fe_materialise_block_gpu(
            tv_gpu, a_cols[start:stop], b_cols[start:stop], op_codes[start:stop]
        )  # resident (n, blk) float32 -- bit-equal to _materialise_chunk_njit
        if out_cand is not None:
            # Float candidate D2H for the downstream survivor/usability reads. Stage through a PERSISTENT
            # PINNED host buffer (full PCIe bandwidth) then host->host memcpy into the caller's pageable
            # slice -- 1.6x faster than cp.asnumpy's pageable bounce-buffer path even WITH the added memcpy
            # (see _pinned_view note). Bit-identical bytes. Falls back to cp.asnumpy on any pinned-alloc
            # failure (e.g. host pinned-memory exhaustion) so it can never regress correctness.
            try:
                hv = _pinned_view(cand.nbytes, cand.shape, cand.dtype)
                cand.get(out=hv)
                out_cand[:, start:stop] = hv
            except Exception:
                import logging
                logging.getLogger(__name__).debug("pinned D2H staging failed; cp.asnumpy fallback", exc_info=True)
                out_cand[:, start:stop] = cp.asnumpy(cand)
        # Bin the candidate RESIDENT at its native float32 (the FE buffer dtype) -- no f64 up-cast: the
        # cand already IS float32 (bit-equal to _materialise_chunk_njit), so binning in f32 removes a needless
        # cast AND halves the bandwidth-bound percentile sort, while preserving the FE selection. The exact
        # f64 fallback (bit-identical to the CPU pipeline) is one env flip away (MLFRAME_FE_GPU_BINNING_DTYPE
        # =float64). _gpu_resident_discretize_codes applies the working dtype internally.
        codes_gpu = _gpu_resident_discretize_codes(cand, int(nbins))
        # Cast int32 codes -> target narrow ``dtype`` (int8/int16) ON the GPU before the D2H so the
        # transfer moves 1/4 (int8) the bytes of the int32 codes AND skips the host-side astype copy.
        # bench (GTX 1050 Ti, n=100k K=384): int32-D2H+host-cast 170ms -> gpu-cast+D2H 25ms = 6.7x on
        # the codes export, BIT-IDENTICAL. The narrow dtype is the FE code dtype (nbins<=255 -> int8),
        # so the on-device cast cannot overflow.
        _cd = np.dtype(dtype)
        codes_out = codes_gpu.astype(cp.dtype(_cd), copy=False) if codes_gpu.dtype != _cd else codes_gpu
        if dev_codes is not None:
            # Keep this block's narrow codes RESIDENT (the EXACT bytes we D2H below). Bit-identical to the
            # host ``out`` slice -> selection-equivalent when the resident gate consumes the device copy.
            dev_codes[:, start:stop] = codes_out
        if not _defer_host_codes:
            # Eager host fill (deferral off, or no resident copy): D2H this block's codes into ``out`` now.
            out[:, start:stop] = cp.asnumpy(codes_out)
        del cand, codes_gpu, codes_out
    if dev_codes is not None:
        # Stash by the returned host array's identity so the dispatch can pick the device codes up without
        # the chunk path threading a new argument (see _RESIDENT_CODES_HANDOFF). Any consumer that is NOT
        # the resident CUDA gate simply ignores it + reads ``out`` (host) as before.
        _stash_resident_codes(out, dev_codes)
    if _defer_host_codes:
        # ``out`` is UNFILLED -- register the lazy device->host fill so a host-reading consumer (analytic /
        # CPU / non-resident GPU) can materialise it on demand via ensure_host_codes_filled. The eager
        # per-block D2H above was skipped; the resident gate reads the device codes directly (no host read).
        _stash_deferred_host_fill(out, dev_codes)
    return out


def gpu_discretize_codes_host(cand: np.ndarray, nbins: int, *, dtype=np.int8) -> np.ndarray:
    """Quantile-bin a host (n, K) float candidate matrix to ordinal codes via the GPU, returning a host
    ``(n, K)`` array of ``dtype``. The FE candidate buffer is ALREADY float32, so the matrix is kept at
    its native dtype (NO f64 up-cast) and binned in float32 (the input's native dtype) -- removing a
    needless cast AND halving the bandwidth-bound cp.percentile sort, while preserving the FE selection
    (the acceptance bar; f32-vs-f64 codes agree ~100%). Set ``MLFRAME_FE_GPU_BINNING_DTYPE=float64`` for
    the bit-identical fallback matching the CPU ``discretize_2d_quantile_batch`` (np.percentile upcasts
    float32 to float64). Feeding the result into the UNCHANGED ``_dispatch_batch_mi_with_noise_gate``
    keeps the FE selection equivalent -- this only moves the binning (CPU partition+searchsorted, the
    dominant per-pair cost at large n) onto the GPU. Inputs are assumed finite (caller scrubs NaN/inf).

    VRAM-chunked over columns so a wide candidate block never over-allocates device memory."""
    import cupy as cp

    cand = np.ascontiguousarray(cand)  # keep native dtype (float32 FE buffer) -- no f64 up-cast
    n, K = cand.shape
    out = np.empty((n, K), dtype=dtype)
    clear_resident_codes_handoff()  # drop any stale prior-chunk handoff before producing this one
    # RESIDENT-CODES HANDOFF (gated, default ON when CUDA present): this is the SECOND codes leg -- the
    # binning-only path the canonical FE chunk takes when the candidate buffer is materialised on the CPU
    # (the default minimal preset's numpy-fallback materialise) then binned on the GPU. It produces the
    # SAME on-device int codes as the fused materialise path, so keep them RESIDENT (one (n, K) cupy array
    # in the narrow code dtype) and stash them by the returned host array's identity -- the noise-gate
    # dispatch then consumes the device codes IN PLACE, skipping the codes' GPU->host (here) ->GPU (the
    # gate's H2D) round-trip. The host ``out`` is STILL filled (the CPU / analytic / opt-out / any-failure
    # branches read it, and it is the safe fallback), so this only ADDS a resident copy when the gate is on;
    # the round-trip is skipped only when the resident CUDA gate is the actual consumer (it matches ``out``
    # by identity). Bit-identical to the host codes -> selection unchanged.
    _resident_codes_on = fe_gpu_resident_codes_enabled()
    dev_codes = cp.empty((n, K), dtype=cp.dtype(np.dtype(dtype))) if _resident_codes_on else None
    # NOTE: this binning-only leg does NOT defer the host-codes D2H. Its documented contract is to RETURN a
    # filled host (n, K) codes array (direct callers -- gpu_pairs_fe_mi's analytic path + the bit-identity
    # tests -- read the return directly, not through the dispatch's ensure_host_codes_filled), and at the
    # canonical fit it is OFF the hot path (the fused gpu_materialise_discretize_codes_host leg, which DOES
    # defer, produces ~all the codes D2H). Keeping it eager keeps the contract intact for ~0 measured cost.
    # f32 codes-path footprint (see gpu_materialise_discretize_codes_host) -> wider VRAM sub-chunk, ~3x
    # fewer bin/edge launches; per-column-independent -> bit-identical codes.
    k_chunk = _gpu_k_chunk(n, bytes_per_elem=4, working_multiple=6, max_cols=K)
    for start in range(0, K, k_chunk):
        block = cand[:, start:start + k_chunk]
        stop = start + block.shape[1]
        codes_gpu = _gpu_resident_discretize_codes(cp.asarray(block), int(nbins))
        # Narrow int32->dtype ON the GPU before D2H (1/4 the bytes for int8, no host astype copy) --
        # same 6.7x codes-export win as gpu_materialise_discretize_codes_host, BIT-IDENTICAL.
        _cd = np.dtype(dtype)
        codes_out = codes_gpu.astype(cp.dtype(_cd), copy=False) if codes_gpu.dtype != _cd else codes_gpu
        if dev_codes is not None:
            # Keep this block's narrow codes RESIDENT (the EXACT bytes we D2H below) for the gate consumer.
            dev_codes[:, start:stop] = codes_out
        out[:, start:stop] = cp.asnumpy(codes_out)
        del codes_gpu, codes_out
    if dev_codes is not None:
        _stash_resident_codes(out, dev_codes)
    return out


def gpu_pairs_fe_mi(cand: np.ndarray, quantization_nbins: int, classes_y: np.ndarray,
                    classes_y_safe: np.ndarray, freqs_y: np.ndarray, npermutations: int,
                    min_nonzero_confidence: float, use_su: bool):
    """Full GPU path for the FE pair-search candidate MI, for the ANALYTIC large-n branch only.

    Returns ``fe_mi[K]`` BIT-IDENTICAL to the production ``_dispatch_batch_mi_with_noise_gate`` analytic
    path, or ``None`` when that branch does not apply (SU-normalised, npermutations<=0, analytic
    disabled / inapplicable) so the caller falls back to the CPU dispatcher. Selection is preserved by
    construction:
      * GPU quantile binning == CPU ``discretize_2d_quantile_batch`` (verified maxdiff 0), and
      * the GPU observed-MI (npermutations=0) == the CPU kernel's observed MI (verified maxdiff 0;
        the GPU twin does only integer counting, entropy stays on the bit-exact CPU path),
    so feeding them through the SAME ``analytic_batch_noise_gate`` (chi2 keep/reject on the observed MI
    + per-column occupied-bin df) yields identical gated MI. Moves BOTH the binning and the observed-MI
    counting -- the dominant large-n per-pair cost -- onto the GPU. Any failure returns None (-> CPU)."""
    n, K = int(cand.shape[0]), int(cand.shape[1])
    if bool(use_su) or int(npermutations) <= 0:
        return None  # SU has no chi2 analytic form; npermutations<=0 is already the cheap CPU path
    try:
        from ._analytic_mi_null import (
            analytic_batch_noise_gate, analytic_null_applicable, analytic_null_enabled,
        )
        by = int(np.unique(np.asarray(classes_y)).size)
        if not (analytic_null_enabled() and analytic_null_applicable(n, int(quantization_nbins), by)):
            return None  # sparse / small-n -> the asymptotic is unreliable; CPU permutation path
        from .batch_mi_noise_gate_gpu import dispatch_batch_mi_with_noise_gate_gpu

        codes = gpu_discretize_codes_host(cand, int(quantization_nbins), dtype=np.int8)  # bit-identical binning
        # (gpu_discretize_codes_host returns FILLED host codes -- this binning-only leg does not defer the
        # D2H, so the analytic dispatch below reads them directly without an ensure_host_codes_filled call.)
        fnb = np.full(K, int(quantization_nbins), dtype=np.int64)
        yc = np.ascontiguousarray(classes_y, dtype=np.int64)
        observed = None
        for _fb in ("cupy", "cuda"):
            observed = dispatch_batch_mi_with_noise_gate_gpu(
                disc_2d=codes, factors_nbins=fnb, classes_y=yc,
                classes_y_safe=np.ascontiguousarray(classes_y_safe), freqs_y=np.ascontiguousarray(freqs_y, dtype=np.float64),
                npermutations=0, base_seed=np.uint64(0), min_nonzero_confidence=float(min_nonzero_confidence),
                use_su=False, dtype=np.int32, force_backend=_fb,
            )
            if observed is not None:
                break
        if observed is None:
            return None  # no GPU backend -> CPU dispatcher
        observed = observed[0] if isinstance(observed, tuple) else observed
        # The analytic keep/reject is cheap CPU post-processing on the K-length observed MI (+ per-column
        # occupied-bin df from the codes), identical to what _dispatch_batch_mi_with_noise_gate runs.
        return analytic_batch_noise_gate(codes, np.asarray(observed, dtype=np.float64), yc, n,
                                         float(min_nonzero_confidence))
    except Exception:
        # Surface the cause (don't silently degrade to CPU forever): a real logic/shape/numeric bug in
        # the GPU path would otherwise be invisible -- the exact "GPU never helped" failure mode.
        import logging
        logging.getLogger(__name__).debug("gpu_pairs_fe_mi failed; CPU fallback", exc_info=True)
        return None


def _fe_gpu_pairs_mi_fallback_choice(n_rows: int, n_cols: int) -> str:
    """Pre-sweep crossover for the FE pair-MI GPU path: GPU only when the work size ``n_rows * n_cols``
    is large enough to amortise the per-pair H2D of the candidate matrix; CPU otherwise. Conservative so
    a small/mid fit stays on the CPU (never a regression) until the per-host sweep refines it. Env
    override ``MLFRAME_FE_GPU_DISCRETIZE_MIN_NK`` (default 2e6 ~= n=100k x K=20)."""
    try:
        min_nk = int(os.environ.get("MLFRAME_FE_GPU_DISCRETIZE_MIN_NK", "2000000"))
    except ValueError:
        min_nk = 2_000_000
    return "gpu" if int(n_rows) * int(n_cols) >= min_nk else "cpu"


def _make_fe_gpu_pairs_inputs(dims: dict) -> tuple:
    """Synthetic (cand_matrix, nbins, classes_y, freqs_y) for the crossover sweep -- an a**2/b pair so
    the analytic branch engages (n >= analytic_null_min_n)."""
    n = int(dims["n_rows"])
    rng = np.random.default_rng(0)
    a = rng.uniform(1.0, 5.0, n); b = rng.uniform(1.0, 5.0, n)
    cand = np.ascontiguousarray(_build_candidate_matrix(np, a, b)).astype(np.float32)
    np.nan_to_num(cand, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y = a ** 2 / b
    edges = np.quantile(y, np.linspace(0, 1, 21)[1:-1])
    yc = np.searchsorted(edges, y).astype(np.int64)
    fy = np.bincount(yc, minlength=int(yc.max()) + 1).astype(np.float64) / n
    return (cand, 20, yc, fy)


def _run_fe_gpu_pairs_mi_sweep() -> list:
    """Per-host CPU-vs-GPU crossover sweep for the FE pair-MI path -> backend_choice regions keyed on
    n_rows. Both variants take the SAME (cand, nbins, yc, fy) and pay their own discretize (+ the GPU
    H2D), so the timing is realistic; the GPU variant is bit-identical (verified) so equivalence holds at
    a tight tol. Skips silently (-> []) when CUDA is unavailable."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        if not is_cuda_available():
            return []
    except Exception:
        return []
    from pyutilz.dev.benchmarking import sweep_backend_grid
    from .discretization import discretize_2d_quantile_batch
    from .info_theory import batch_mi_with_noise_gate
    from ._feature_engineering_pairs._pairs_dispatch import _dispatch_batch_mi_with_noise_gate

    def _cpu(cand, nbins, yc, fy):
        disc = discretize_2d_quantile_batch(cand, n_bins=nbins, dtype=np.int8, assume_finite=True)
        return _dispatch_batch_mi_with_noise_gate(
            disc_2d=disc, quantization_nbins=nbins, classes_y=yc, classes_y_safe=yc, freqs_y=fy,
            npermutations=3, min_nonzero_confidence=0.0, use_su=False, batch_mi_kernel=batch_mi_with_noise_gate,
        )

    def _gpu(cand, nbins, yc, fy):
        return gpu_pairs_fe_mi(cand, nbins, yc, yc, fy, 3, 0.0, False)

    return sweep_backend_grid(
        {"cpu": _cpu, "gpu": _gpu},
        {"n_rows": [50_000, 100_000, 300_000]},  # GPU path engages only at n >= analytic_null_min_n
        _make_fe_gpu_pairs_inputs,
        reference="cpu", repeats=3, equiv_rtol=1e-9, equiv_atol=1e-12,
    )


def _fe_gpu_pairs_mi_code_version():
    try:
        from pyutilz.performance.kernel_tuning.code_versioning import compute_code_version
        return compute_code_version(gpu_pairs_fe_mi, gpu_discretize_codes_host, _gpu_resident_discretize_codes)
    except Exception:
        return None


def fe_gpu_pairs_mi_backend_choice(n_rows: int, n_cols: int) -> str:
    """Per-host 'gpu' or 'cpu' for the FE pair-MI path via the shared get_or_tune orchestrator
    (per-host cache, code-version checked, background sweep, measurement-backed fallback). Never blocks
    the fit: async_sweep tunes off the hot path; the conservative fallback routes meanwhile."""
    try:
        # Under an explicit max_runtime_mins budget, skip the (blocking-on-first-use, CUDA-detected-regardless-of-
        # CUDA_VISIBLE_DEVICES) CPU-vs-GPU crossover sweep -- it runs the CPU+GPU variants at n up to 300k (tens of
        # seconds) and would blow a tiny budget. Route via the measurement-backed fallback instead; the sweep still runs
        # on a normal no-budget fit, so per-host tuning is unaffected.
        from ._fe_deadline import fe_budget_active
        if fe_budget_active():
            return _fe_gpu_pairs_mi_fallback_choice(n_rows, n_cols)
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
        res = KernelTuningCache.load_or_create().get_or_tune(
            "fe_gpu_pairs_mi",
            dims={"n_rows": int(n_rows)},
            tuner=_run_fe_gpu_pairs_mi_sweep,
            axes=["n_rows"],
            fallback={"backend_choice": _fe_gpu_pairs_mi_fallback_choice(n_rows, n_cols)},
            code_version=_fe_gpu_pairs_mi_code_version(),
            async_sweep=True,
        )
        bc = res if isinstance(res, str) else str((res or {}).get("backend_choice", "cpu"))
        return bc if bc in ("cpu", "gpu") else "cpu"
    except Exception:
        return _fe_gpu_pairs_mi_fallback_choice(n_rows, n_cols)


def ensure_fe_gpu_pairs_mi_tuning(force: bool = False):
    """Force-run + persist the FE pair-MI CPU-vs-GPU crossover sweep for this host (CLI refresh hook)."""
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
        cache = KernelTuningCache.load_or_create()
        if not force:
            existing = cache.get_regions("fe_gpu_pairs_mi")
            if existing:
                return existing
        regions = _run_fe_gpu_pairs_mi_sweep()
        if regions:
            cache.update("fe_gpu_pairs_mi", axes=["n_rows"], regions=regions,
                         code_version=_fe_gpu_pairs_mi_code_version())
        return regions
    except Exception:
        return None


def grand_fused_pair_mi(
    a, b, y_codes, classes_y_safe, freqs_y, *,
    nbins: int = 20, npermutations: int = 25, min_nonzero_confidence: float = 0.0, use_su: bool = False,
):
    """GRAND FUSION: GPU fused-generate candidates -> RESIDENT GPU discretize -> the EXISTING bit-identical
    GPU noise-gate (``batch_mi_noise_gate_gpu``). Returns the SAME noise-gated fe_mi[K] the production
    pair-search computes, but with generation+discretization+noise-gate all on the GPU. Only the small
    int8 disc crosses to host for the existing noise-gate (which does its own resident permutation
    counting). Bit-identical: GPU discretize == CPU discretize (verified maxdiff 0) and the GPU noise-gate
    is the production twin. VRAM-chunked. Returns ``(names, fe_mi)``.

    MEASURED (GTX 1050 Ti, K=384, nperm=25, BIT-IDENTICAL to the production ``_dispatch_batch_mi_with_
    noise_gate`` -- bit=True, argmax match):
      * vs the PRODUCTION dispatch (its analytic large-n gate): n=50k 3169->717ms 4.42x; n=200k
        13589->2948ms 4.61x. This is the honest, fair speedup.
      * vs a forced CPU PERMUTATION gate (which production AVOIDS at large n via the analytic shortcut):
        n=200k 53902->2753ms ~19.6x -- do NOT quote this as the production win; it is the permutation-path
        ceiling, shown only to locate where the time goes (the noise-gate dominates at K=384).
    The default chooser routes the gate to CPU on this host (a tuner mis-calibration: the GPU gate is
    ~15x faster on the permutation path); grand_fused forces cupy/cuda so the gate runs on the GPU.

    GRAND FUSION (2026-06-20, default ON via ``MLFRAME_FE_GPU_GRAND_FUSION``): when enabled this delegates
    to :func:`grand_fused_pair_mi_fused`, which NEVER materialises the (n,K) float candidate matrix, the
    (n,K) int codes, the (n,K) D2H disc, nor the noise-gate's (n,K) ``d_base`` / (rows*n*K) flat index --
    it fuses gen+bin+joint-histogram into ONE shared-mem-atomic RawKernel per chunk (recompute-not-store,
    Option F1 + roadmap #3). MEASURED (GTX 1050 Ti, K=384, nperm=25, BIT-IDENTICAL -- maxdiff 0, argmax
    match, vs this non-fused path): n=100k 2.16x + 3.0x less peak GPU mem; n=300k 2.15x + 2.75x; n=1M
    3.39x + 2.26x. Selection is EXACT (same percentile edges; only the data movement changes). Falls back
    to this exact non-fused body if the shared histogram (P1*nbins*K_y int32) exceeds the device's per-block
    shared-mem limit or any GPU error occurs."""
    if fe_gpu_grand_fusion_enabled():
        try:
            return grand_fused_pair_mi_fused(
                a, b, y_codes, classes_y_safe, freqs_y, nbins=nbins, npermutations=npermutations,
                min_nonzero_confidence=min_nonzero_confidence, use_su=use_su,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).info("grand-fusion fused path unavailable (%s); non-fused fallback", e)

    import cupy as cp

    from . import hermite_fe as _hf  # noqa: F401 -- full-init parent before the GPU MI import cycle
    from .batch_mi_noise_gate_gpu import dispatch_batch_mi_with_noise_gate_gpu

    a_gpu = cp.asarray(a, dtype=cp.float64)
    b_gpu = cp.asarray(b, dtype=cp.float64)
    n = int(a_gpu.shape[0])
    ua_cm = _unary_stack_cm(cp, a_gpu)
    ub_cm = _unary_stack_cm(cp, b_gpu)
    y_i64 = np.ascontiguousarray(y_codes, dtype=np.int64)
    csafe = np.ascontiguousarray(classes_y_safe)
    fy = np.ascontiguousarray(freqs_y, dtype=np.float64)
    k_chunk = _gpu_k_chunk(n)
    parts: list[np.ndarray] = []
    for start in range(0, len(_COMBOS), k_chunk):
        block = _COMBOS[start:start + k_chunk]
        cand = _fused_generate_block(ua_cm, ub_cm, block)        # GPU gen (resident)
        disc_host = cp.asnumpy(_gpu_resident_discretize_codes(cand, nbins).astype(cp.int8))  # GPU disc -> small D2H
        del cand
        fnb = np.full(len(block), int(nbins), dtype=np.int64)
        # FORCE the GPU noise-gate: measured 15x faster than CPU here (K=384 n=200k: cupy 3093ms vs
        # CPU njit 46176ms -- the noise-gate is the REAL bottleneck, not gen/discretize). The default
        # chooser (_batch_mi_noise_gate_backend_choice) picks CPU on this host -- a tuner mis-calibration
        # (it under-rates the GPU gate, same class as the MI-dispatch issue); force cupy/cuda so the
        # grand-fused path actually runs the gate on the GPU. Falls back to CPU only if no GPU backend.
        out = None
        for _fb in ("cupy", "cuda"):
            out = dispatch_batch_mi_with_noise_gate_gpu(
                disc_2d=disc_host, factors_nbins=fnb, classes_y=y_i64, classes_y_safe=csafe, freqs_y=fy,
                npermutations=int(npermutations), base_seed=np.uint64(0),
                min_nonzero_confidence=float(min_nonzero_confidence), use_su=bool(use_su),
                dtype=np.int32, force_backend=_fb,
            )
            if out is not None:
                break
        if out is None:  # GPU noise-gate unavailable -> the always-correct CPU kernel on the same disc
            from .info_theory import batch_mi_with_noise_gate as _cpu_gate
            fe_mi = _cpu_gate(
                disc_2d=disc_host, factors_nbins=fnb, classes_y=y_i64, classes_y_safe=csafe, freqs_y=fy,
                npermutations=int(npermutations), base_seed=np.uint64(0),
                min_nonzero_confidence=float(min_nonzero_confidence), use_su=bool(use_su),
                dtype=np.int32, classes_dtype=np.int16,
            )
        else:
            fe_mi = out[0] if isinstance(out, tuple) else out
        parts.append(np.asarray(fe_mi, dtype=np.float64))
    return _candidate_names(), np.concatenate(parts) if parts else np.empty(0)


def _grand_fusion_block_counts(ua_cm, ub_cm, block, edges_int, y_all_dev, nbins, K_y, total_size):
    """Run the fused gen+bin+histogram kernel for one candidate ``block``, returning the (P1, total_size)
    int64 joint-count matrix on HOST. ``edges_int`` is the (blk, nbins-1) interior-edge matrix (the exact
    ``cp.percentile`` edges for this block), ``y_all_dev`` is the (P1, n) int32 device y-vectors. The
    candidate float matrix is NEVER stored: each cell is regenerated + binned + atomic-histogrammed inline.

    ``col_off[c] = c * nbins * K_y`` (uniform nbins per FE candidate); ``total_size = blk * nbins * K_y``."""
    import cupy as cp

    n = int(ua_cm.shape[1])
    K = int(len(block))
    # Same fit-invariant index trio as _fused_generate_block (block is a slice of the module constant
    # _COMBOS); reuse the shared cache to drop the per-chunk-per-pair list-comps + tiny H2D.
    _ck = tuple(block)
    _cc = _COMBO_IDX_CACHE.get(_ck)
    if _cc is None:
        ua_idx = cp.asarray(np.asarray([_UNARY_IDX[ua] for ua, _, _ in block], dtype=np.int32))
        ub_idx = cp.asarray(np.asarray([_UNARY_IDX[ub] for _, ub, _ in block], dtype=np.int32))
        bop = cp.asarray(np.asarray([_BINOP_CODE[bp] for _, _, bp in block], dtype=np.int32))
        _COMBO_IDX_CACHE[_ck] = (ua_idx, ub_idx, bop)
    else:
        ua_idx, ub_idx, bop = _cc
    col_off = cp.arange(K, dtype=cp.int64) * (int(nbins) * int(K_y))
    P1 = int(y_all_dev.shape[0])
    counts = cp.zeros((P1, int(total_size)), dtype=cp.int64)
    # ONE BLOCK PER CANDIDATE: shared-mem histogram is (P1, nbins, K_y) int32. Check it fits this device's
    # per-block shared-memory limit; if not, the caller must fall back (the host gates on this).
    hist_bytes = P1 * int(nbins) * int(K_y) * 4
    threads = 256
    _get_fused_gen_bin_hist_kernel()(
        (K,), (threads,),
        (ua_cm, ub_cm, ua_idx, ub_idx, bop, edges_int, col_off, y_all_dev,
         np.int64(n), np.int32(K), np.int32(int(nbins)), np.int32(int(K_y)),
         np.int32(P1), np.int64(int(total_size)), counts),
        shared_mem=hist_bytes,
    )
    out = cp.asnumpy(counts)
    del counts, ua_idx, ub_idx, bop, col_off
    return out


def grand_fused_pair_mi_fused(
    a, b, y_codes, classes_y_safe, freqs_y, *,
    nbins: int = 20, npermutations: int = 25, min_nonzero_confidence: float = 0.0, use_su: bool = False,
):
    """GRAND-FUSION (never materialise (n,K)): the fully-fused twin of :func:`grand_fused_pair_mi`.

    Collapses gen -> discretize -> noise-gate-counting into ONE histogram kernel per chunk. Pass 1 (per
    chunk, VRAM-governed) generates the (n, blk) candidate floats ONLY long enough to take the EXACT
    ``cp.percentile`` interior edges, then DISCARDS them (the edges -- (blk, nbins-1) -- are the only
    survivor). Pass 2 launches :func:`_grand_fusion_block_counts`: each (row, candidate) thread RE-generates
    its value, bins it against those exact edges (identical math -> identical codes -> SAME selection), and
    atomic-adds into the (P1, total_size) joint histogram for the original-y + every shuffled-y at once.
    The MI/SU + the noise-gate rejection are reduced from those integer counts on the bit-exact CPU path
    (``_mi_from_counts_cpu`` + ``_gate_from_mi``) -- so the returned fe_mi is BIT-IDENTICAL to
    ``grand_fused_pair_mi`` / the production gate, while the (n,K) float matrix, the (n,K) int codes, the
    (n,K) D2H disc, and the noise-gate's (n,K) ``d_base`` + (rows*n*K) flat index are ALL eliminated.
    Returns ``(names, fe_mi)``. Raises if cupy is unavailable (caller gates)."""
    import cupy as cp

    from .batch_mi_noise_gate_gpu import (
        _build_shuffle_matrix, _gate_from_mi, _mi_from_counts_cpu, _mi_columns_from_counts_cpu,
    )

    a_gpu = cp.asarray(a, dtype=cp.float64)
    b_gpu = cp.asarray(b, dtype=cp.float64)
    n = int(a_gpu.shape[0])
    ua_cm = _unary_stack_cm(cp, a_gpu)
    ub_cm = _unary_stack_cm(cp, b_gpu)

    # y-vectors: row 0 = original y, rows 1.. = the Fisher-Yates shuffles (SAME host LCG the noise-gate
    # uses -> bit-identical permutation stream). Uploaded ONCE as (P1, n) int32 and shared by every chunk.
    y_orig = np.ascontiguousarray(y_codes, dtype=np.int64).reshape(1, n)
    K_y = int(np.asarray(freqs_y).shape[0])
    fy = np.ascontiguousarray(freqs_y, dtype=np.float64)
    nperm = int(npermutations) if npermutations and npermutations > 0 else 0
    if nperm > 0:
        shuf = _build_shuffle_matrix(np.asarray(classes_y_safe), np.uint64(0), nperm)
        y_all_host = np.empty((nperm + 1, n), dtype=np.int64)
        y_all_host[0, :] = y_orig[0, :]
        y_all_host[1:, :] = shuf.astype(np.int64)
    else:
        y_all_host = y_orig
    P1 = int(y_all_host.shape[0])
    y_all_dev = cp.asarray(np.ascontiguousarray(y_all_host, dtype=np.int32))

    # The shared-mem histogram is (P1, nbins, K_y) int32 per block; it must fit this device's per-block
    # shared-memory limit. If not (very high nperm / many y-classes / large nbins), raise so the caller
    # falls back to the non-fused exact path -- correctness over fusion. Reserve a little headroom.
    hist_bytes = P1 * int(nbins) * K_y * 4
    try:
        sm_limit = int(cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["sharedMemPerBlock"])
    except Exception:
        sm_limit = 48 * 1024
    if hist_bytes > sm_limit - 256:
        raise RuntimeError(
            f"grand-fusion shared histogram {hist_bytes}B exceeds device limit {sm_limit}B "
            f"(P1={P1}, nbins={nbins}, K_y={K_y}); caller falls back to the non-fused path"
        )

    # Binning working dtype: mirror _gpu_resident_discretize_codes (native f64 here -> bit-identical edges
    # to the non-fused grand-fusion path; MLFRAME_FE_GPU_BINNING_DTYPE=float32 forces f32 percentile).
    forced = os.environ.get("MLFRAME_FE_GPU_BINNING_DTYPE", "").strip().lower()
    work = cp.float32 if forced in ("float32", "f32", "single") else cp.float64
    qs = _quantile_levels_dev(cp, nbins, work)

    k_chunk = _gpu_k_chunk(n)
    original_mi_parts: list[np.ndarray] = []
    perm_mi_parts: list[list[np.ndarray]] = [[] for _ in range(nperm)]
    for start in range(0, len(_COMBOS), k_chunk):
        block = _COMBOS[start:start + k_chunk]
        blk = len(block)
        # PASS 1: transient generate -> exact percentile edges -> discard the float matrix.
        cand = _fused_generate_block(ua_cm, ub_cm, block)            # (n, blk) f64, transient
        if cand.dtype != work:
            cand = cand.astype(work, copy=False)
        if blk == 1:
            # cupy single-column percentile bug guard (mirror _gpu_resident_discretize_codes).
            bin_edges = cp.percentile(cand.ravel(), qs).reshape(-1, 1)  # (nbins+1, 1)
        else:
            bin_edges = cp.percentile(cand, qs, axis=0)             # (nbins+1, blk)
        del cand                                                    # (n,blk) float GONE before the hist pass
        # interior edges, transposed to (blk, nbins-1) row-major f64 for the kernel's per-candidate scan.
        edges_int = cp.ascontiguousarray(bin_edges[1:-1, :].T.astype(cp.float64))
        del bin_edges
        total_size = blk * int(nbins) * K_y
        counts = _grand_fusion_block_counts(ua_cm, ub_cm, block, edges_int, y_all_dev, nbins, K_y, total_size)
        del edges_int
        # CPU bit-exact reduction (identical to batch_mi_with_noise_gate_cupy). Use the BATCHED njit
        # _mi_columns_from_counts_cpu (one compiled call over all blk columns) instead of the per-(perm,k)
        # Python->njit dispatch -- bit-identical (same _mi_from_counts_cpu body per column) but removes the
        # blk*(nperm+1) Python-call overhead. ref_mi reproduces the perm-skip: all-positive for the original
        # row (compute every column), then `om` for each perm row (compute only where om>0; else stays 0).
        nb_ky = int(nbins) * K_y
        _col_off = np.arange(blk, dtype=np.int64) * nb_ky
        _nbins_arr = np.full(blk, int(nbins), dtype=np.int64)
        _all_pos = np.ones(blk, dtype=np.float64)
        om = _mi_columns_from_counts_cpu(
            np.ascontiguousarray(counts[0]), _col_off, _nbins_arr, K_y, fy, n, bool(use_su), _all_pos,
        )
        original_mi_parts.append(om)
        for p in range(nperm):
            mp = _mi_columns_from_counts_cpu(
                np.ascontiguousarray(counts[p + 1]), _col_off, _nbins_arr, K_y, fy, n, bool(use_su), om,
            )
            perm_mi_parts[p].append(mp)
        del counts
    original_mi = np.concatenate(original_mi_parts) if original_mi_parts else np.empty(0)
    perm_mis = [np.concatenate(perm_mi_parts[p]) for p in range(nperm)] if original_mi_parts else []
    fe_mi = _gate_from_mi(original_mi, perm_mis, nperm, float(min_nonzero_confidence))
    return _candidate_names(), fe_mi


def _log_shift_anchor(operand_vals: np.ndarray, unary_name: str):
    """Frozen smart_log shift for a ``log`` side -- ``(1e-5 - nanmin)`` if the FULL column reaches <=0,
    else 0.0 (mirrors _step_core._ls_anchor exactly so replay is byte-identical). None for non-log."""
    if unary_name != "log":
        return None
    mn = float(np.nanmin(np.asarray(operand_vals, dtype=np.float64)))
    return (1e-5 - mn) if mn <= 0 else 0.0


def gpu_resident_pair_recipes(
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    y_codes: np.ndarray,
    *,
    src_a_name: str,
    src_b_name: str,
    cols_names,
    unary_preset: str = "minimal",
    binary_preset: str = "minimal",
    quantization_nbins=None,
    quantization_method=None,
    quantization_dtype=np.float32,
    top_k: int = 1,
    nbins: int = 20,
):
    """Score a pair's candidate grid on the GPU and return the top-``top_k`` as STRUCTURED, replayable
    ``EngineeredRecipe`` objects -- the bridge from this path's flat (name, MI) to what production FE
    consumes. For each winner it emits, via the SAME builders the CPU path uses
    (``get_new_feature_name`` + ``build_unary_binary_recipe``): the canonical name, the structured
    (src column names, unary names, binary name), the active presets (frozen for replay-stable semantics),
    the quantization params with fit-time edges PINNED (``fit_values_for_edges`` -> leak-free transform),
    and the frozen ``log_shift`` anchor for any ``log`` side. So the GPU result is a first-class recipe
    that ``transform()`` replays bit-identically on raw inputs -- not a string to be re-parsed.

    Returns a list of ``(name, EngineeredRecipe, mi)`` sorted by descending MI. The MI uses the exact
    GPU-resident path (``gpu_resident_pair_candidate_mi``); the recipe fields are built on CPU (cheap for
    top_k winners). Combo order is ``_COMBOS`` (kept in sync with the minimal preset)."""
    from .engineered_recipes import build_unary_binary_recipe
    from .feature_engineering import get_new_feature_name

    # Route through the dispatcher so this works on ANY backend (GPU-resident in the sweet spot, CPU
    # otherwise) -- recipe emission is backend-agnostic.
    names, mi = pair_candidate_mi_dispatch(a_vals, b_vals, y_codes, nbins=nbins)
    a64 = np.ascontiguousarray(a_vals, dtype=np.float64)
    b64 = np.ascontiguousarray(b_vals, dtype=np.float64)
    cols = list(cols_names)
    idx_a = cols.index(src_a_name)
    idx_b = cols.index(src_b_name)
    out = []
    for ci in np.argsort(mi)[::-1][: int(top_k)]:
        ua, ub, bop = _COMBOS[int(ci)]
        fe_tuple = (((idx_a, ua), (idx_b, ub)), bop, 0)
        name = get_new_feature_name(fe_tuple, cols)
        # Continuous fit-time engineered column (for edge pinning) -- identical op chain as the GPU path.
        fit_vals = _binary_apply(np, bop, _unary_apply(np, ua, a64), _unary_apply(np, ub, b64))
        fit_vals = np.nan_to_num(np.asarray(fit_vals, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        recipe = build_unary_binary_recipe(
            name=name,
            src_a_name=src_a_name, src_b_name=src_b_name,
            unary_a_name=ua, unary_b_name=ub, binary_name=bop,
            unary_preset=unary_preset, binary_preset=binary_preset,
            quantization_nbins=quantization_nbins,
            quantization_method=quantization_method,
            quantization_dtype=quantization_dtype,
            fit_values_for_edges=fit_vals,
            log_shift_a=_log_shift_anchor(a64, ua),
            log_shift_b=_log_shift_anchor(b64, ub),
        )
        out.append((name, recipe, float(mi[int(ci)])))
    return out


def pair_candidate_mi_dispatch(a: np.ndarray, b: np.ndarray, y_codes: np.ndarray, *, nbins: int = 20):
    """Route a pair's candidate-MI to the measured-fastest backend: GPU-resident in the sweet spot
    (cupy present, n >= the crossover, VRAM-chunked so it can't thrash), CPU njit otherwise. Returns
    ``(names, mi)`` identical in shape/order to both paths. The default FE pipeline does NOT call this
    yet (gated prototype); it is the dispatcher the production wiring will use."""
    n = int(np.asarray(a).shape[0])
    # Per-host crossover via the shared kernel_tuning_cache (mirrors the FE pair-MI path) rather than the
    # hardcoded 50k: the per-host CPU-vs-GPU sweep decides, and _GPU_RESIDENT_MIN_N is only the source-code
    # FALLBACK (inside _fe_gpu_pairs_mi_fallback_choice) when the cache is cold / lookup fails. Honours the
    # project rule against hardcoded GPU thresholds; the 50k stays as the conservative cold-start default.
    try:
        _use_gpu = fe_gpu_pairs_mi_backend_choice(n, len(_COMBOS)) == "gpu"
    except Exception:
        _use_gpu = n >= _GPU_RESIDENT_MIN_N
    if _use_gpu:
        try:
            import cupy  # noqa: F401

            return gpu_resident_pair_candidate_mi(a, b, y_codes, nbins=nbins)
        except Exception as e:
            # Log (don't silently swallow) -- a GPU OOM/driver error degrading to a slow CPU fallback
            # would otherwise look like "GPU never helped". A chunk-shrink-retry before CPU fallback is
            # a future refinement (the VRAM governor already bounds chunks, so OOM should be rare).
            import logging
            logging.getLogger(__name__).warning("GPU-resident pair MI failed (%s); CPU fallback.", e)
    return cpu_pair_candidate_mi(a, b, y_codes, nbins=nbins)
