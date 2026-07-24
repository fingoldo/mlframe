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
from collections import OrderedDict

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
# (4) multi-stream across k-chunks (lowers the GPU crossover n) stays FUTURE. The VRAM K-chunk fraction is
# now KTC-tuned (G3, _gpu_resident_k_chunk_ktc); block size (threads=256) + any f32 threshold + the hardcoded
# _GPU_RESIDENT_MIN_N still want a contention-aware sweep (mirror _run_sweep_mi_classif_dispatch); keep
# cp.argsort as the exact fallback when adding a radix _v2.
# ARCHITECTURE (wiring, before flipping any default): wire this flat (name, MI) path as a candidate-MI
# PROVIDER feeding the EXISTING gates, emitting structured (ua,ub,bop) triples + real src names + presets
# (reuse fe_tuple->get_new_feature_name->EngineeredRecipe; never re-parse the string); drive the op set from
# create_*_transformations(preset) + the gpu_compatible_unary_names allowlist with CPU fallback for non-pure
# ops; collapse MLFRAME_FE_MATRIX_P0 + MLFRAME_FE_GPU_RESIDENT into one selector; add pickle/clone + op-parity
# (registry vs _unary/_binary_apply vs CUDA switch -- _safe_div is the single spec) meta-tests.

# Minimal-preset op NAMES (kept in sync with feature_engineering.create_*_transformations "minimal").
_MINIMAL_UNARY = ("identity", "neg", "abs", "sqr", "reciproc", "sqrt", "log", "sin")
_MINIMAL_BINARY = ("mul", "add", "sub", "div", "max", "min")


# RESIDENCY REPLATFORM MAP (2026-06-21, measured -- the production chunk path, NOT this prototype).
# Goal: kill the FE chunk's bulk float-buffer D2H. ``gpu_materialise_discretize_codes_host`` D2Hs the
# full (n,K) float candidate buffer via ``out_cand`` -- measured 6.7 GB total across the canonical
# n=100k fit (15 calls), a large slice of the ~10.5s ``cupy.get`` wall. The codes (for MI) are produced
# RESIDENT and don't need that buffer; only a HANDFUL of host reads do.
#   * Final survivors ALREADY recompute from raw via ``_rebuild_full_survivor_col`` (subsample path,
#     _pairs_core.py:2218); the buffer feeds only the INTERMEDIATE subsample scoring reads in
#     check_prospective_fe_pairs (best-config ~1625/1749, multi-emit ~2126/2137, MI-replay ~1499).
# Proposed win = KEEP the chunk-batch GPU-codes path, pass ``out_cand=None`` (skip the 6.7 GB D2H), route
# the few intermediate reads through the validated recompute helper (_config_by_i / _rebuild_full_survivor_-
# col, bit-identical), gated on the GPU-fused path. MEASURED CONSTRAINT: pure recompute (no buffer, no
# chunk-batch) is 3x SLOWER (217s vs 69s), so chunk-batch is essential. (Turnkey deferral scaffold -- gate
# MLFRAME_FE_GPU_DEFER_FLOAT, out_cand=None, _config_by_i recompute at the 3 buffer reads in
# check_prospective_fe_pairs -- was implemented then bench-rejected below; see git 72d1c364.)
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


def _gpu_min_free_vram_mb() -> int:
    """Minimum FREE device VRAM (MB) for the FE GPU paths to default ON. Below this the (n, K) FE matrices cannot
    fit and the GPU work stalls/OOM-falls-back while the (parallelized) CPU cores idle -- so default to CPU instead.
    Env-tunable ``MLFRAME_FE_GPU_MIN_FREE_VRAM_MB`` (default 1024). 0 disables the check (legacy: present-only gate)."""
    try:
        return int(os.environ.get("MLFRAME_FE_GPU_MIN_FREE_VRAM_MB", "1024"))
    except (ValueError, TypeError):
        return 1024


def _gpu_has_free_vram() -> bool:
    """True iff the current CUDA device has at least ``_gpu_min_free_vram_mb()`` MB FREE. On a small/shared card whose
    VRAM is mostly consumed (e.g. a 4GB desktop GPU eaten by the compositor/browsers) this is False, so the FE GPU
    default-on gates route to CPU rather than starving on ~hundreds of MB. A query failure -> True (defer to the
    existing per-path OOM fallbacks). Skipped when the threshold is 0."""
    _min_mb = _gpu_min_free_vram_mb()
    if _min_mb <= 0:
        return True
    try:
        import cupy as cp

        free_b, _total = cp.cuda.runtime.memGetInfo()
        return bool(free_b >= _min_mb * 1024 * 1024)
    except Exception:
        return True


def _env_gpu_default_on(name: str) -> bool:
    """Shared GPU opt-out gate: explicit env ``<name>`` = 0/1 (off/on) wins; otherwise default ON when a CUDA device
    is usable AND has enough FREE VRAM -- False if CUDA_VISIBLE_DEVICES='' or MLFRAME_DISABLE_GPU=1, no CUDA present,
    or the device is nearly full (a tiny / desktop-shared card). Single source for the fe_gpu_*_enabled default-on
    gates. Explicit ``=1`` still forces GPU past the free-VRAM check (user override / big-GPU host)."""
    _v = os.environ.get(name, "").strip().lower()
    if _v in ("0", "false", "off", "no"):
        return False
    if _v in ("1", "true", "on", "yes"):
        return True
    # GPU_INFRA_B-9 fix: delegate the CUDA_VISIBLE_DEVICES=""/MLFRAME_DISABLE_GPU=1
    # opt-out check to the shared _gpu_policy module instead of reimplementing it inline, so a future change
    # to the shared policy's semantics reaches this cluster's default-on gates too.
    from ._gpu_policy import gpu_globally_disabled
    if gpu_globally_disabled():
        return False
    return _cuda_present() and _gpu_has_free_vram()


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
    return _env_gpu_default_on("MLFRAME_FE_GPU_RESIDENT_CODES")


def fe_gpu_resident_basis_mi_enabled() -> bool:
    """Whether the matrix-native RESIDENT orth-FE basis-MI path is active (Piece 3).

    X_EFFICIENCY_ARCHITECTURE-6 fix: this line used to say "DEFAULT OFF", directly
    contradicting the "DEFAULT ON when CUDA is present" statement two paragraphs below for the SAME flag --
    the real behavior (confirmed by ``_env_gpu_default_on``) is DEFAULT ON when a usable CUDA device is
    present, DEFAULT OFF otherwise (env-overridable either way).

    When ON, ``hybrid_orth_mi_fe`` builds the univariate orth-basis candidate matrix ON the device
    (BATCHED ``_gpu_evaluate_basis_matrix``) and scores its plug-in MI with
    ``_plugin_mi_classif_batch_cuda_resident`` -- with NO per-call H2D (the dispatcher's 2x trap).

    DEFAULT ON when CUDA is present (2026-06-21). Validated: selection-EQUIVALENT (it swaps the njit RANK
    binning for the GPU equi-frequency-edge binning in the basis ranking -- the orth-basis recovery pins
    test_layer21/22, the canonical single_compound, and the full biz-value hybrid_orth suite all pass with
    it on: 385 passed) AND FASTER (clean canonical 100k wall 34.8s -> 30.7s, ~12%; the batched build also
    clears the p200 high-feature perf budget). Opt out with ``MLFRAME_FE_GPU_RESIDENT_BASIS_MI=0``. Any GPU
    failure / unported basis falls back to the host path per-call (never a correctness regression)."""
    return _env_gpu_default_on("MLFRAME_FE_GPU_RESIDENT_BASIS_MI")


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
    return _env_gpu_default_on("MLFRAME_FE_GPU_ROUTING")


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
# Per-host-array handoff (was a single slot, which two concurrent fits clobbered: producer B's stash
# overwrote producer A's, so A's consumer read B's device codes or missed entirely). Keyed by id() of the
# EXACT host array the producer returns (stable + alive across the synchronous producer->consumer call);
# shape/dtype guard at lookup. Bounded FIFO so interleaved fits don't grow it; entries cleared after the
# dispatch consumes them (clear_resident_codes_handoff). Module-level -> never reachable from pickled state.
_RESIDENT_CODES_HANDOFF: "OrderedDict[int, tuple]" = OrderedDict()  # id(host) -> (device_codes, shape, dtype)
_RESIDENT_CODES_HANDOFF_MAX = 8


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
# Per-host-array deferred fill (was a single slot; see _RESIDENT_CODES_HANDOFF). Keyed by id(host_codes).
_DEFERRED_HOST_FILL: "OrderedDict[int, tuple]" = OrderedDict()  # id(host) -> (host_codes, device_codes, shape, dtype, filled[list[bool]])
_DEFERRED_HOST_FILL_MAX = 8


def _stash_resident_codes(host_codes, device_codes) -> None:
    """Record the resident device codes for the host array ``host_codes`` (keyed on its id)."""
    c = _RESIDENT_CODES_HANDOFF
    c[id(host_codes)] = (device_codes, tuple(host_codes.shape), np.dtype(host_codes.dtype))
    c.move_to_end(id(host_codes))
    while len(c) > _RESIDENT_CODES_HANDOFF_MAX:
        c.popitem(last=False)


def _stash_deferred_host_fill(host_codes, device_codes) -> None:
    """Register ``host_codes`` (an UNFILLED host buffer) for a lazy D2H fill from ``device_codes`` -- used
    when the producer skips the eager codes D2H because the resident gate will likely consume the device
    copy. ``ensure_host_codes_filled`` performs the fill on demand; ``clear_resident_codes_handoff`` drops
    the record so device memory is not pinned past the dispatch."""
    c = _DEFERRED_HOST_FILL
    c[id(host_codes)] = (host_codes, device_codes, tuple(host_codes.shape), np.dtype(host_codes.dtype), [False])
    c.move_to_end(id(host_codes))
    while len(c) > _DEFERRED_HOST_FILL_MAX:
        c.popitem(last=False)


def ensure_host_codes_filled(host_codes) -> None:
    """If ``host_codes`` was returned UNFILLED with a deferred device->host fill registered for it (same id,
    shape, dtype), D2H the device codes into it NOW (once; idempotent). No-op when there is no deferred fill
    for this array (it was filled eagerly, or is an unrelated array). Called by the dispatch on every
    host-codes-reading path (analytic gate / CPU njit kernel / non-resident GPU path) so the codes D2H is
    paid exactly when -- and only when -- a host consumer reads them. Bit-identical to the eager fill."""
    h = _DEFERRED_HOST_FILL.get(id(host_codes))
    if h is None:
        return
    host, dev, shape, dtype, filled = h
    if tuple(host_codes.shape) != shape or np.dtype(host_codes.dtype) != dtype:
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
    h = _RESIDENT_CODES_HANDOFF.get(id(host_codes))
    if h is None:
        return None
    dev, shape, dtype = h
    if tuple(host_codes.shape) == shape and np.dtype(host_codes.dtype) == dtype:
        return dev
    return None


def clear_resident_codes_handoff(host_codes: np.ndarray | None = None) -> None:
    """Drop the resident-codes handoff + the deferred host-fill record so device memory is not pinned past
    the dispatch that produced it, and a stale entry can never satisfy a later, unrelated dispatch. Called
    by ``_dispatch_batch_mi_with_noise_gate`` in a finally after it has decided the consumer.

    When ``host_codes`` is given, drops ONLY that host array's entries (so a concurrent fit's in-flight
    handoff is not wiped); when None, clears all (legacy whole-cache reset)."""
    if host_codes is not None:
        _RESIDENT_CODES_HANDOFF.pop(id(host_codes), None)
        _DEFERRED_HOST_FILL.pop(id(host_codes), None)
        return
    _RESIDENT_CODES_HANDOFF.clear()
    _DEFERRED_HOST_FILL.clear()


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
    if name in ("sinc", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh"):
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


_PREWARP_TRANSFORM_SRC = r"""
extern "C" __global__
void prewarp_transform(const double* __restrict__ x, const int code, const double lo, const double span,
                       const double mean, const double std_safe, const double clip_lo, const double clip_hi,
                       const int has_clip, const long long n, double* __restrict__ out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double v = x[i], z;
    if (code == 1)       z = 2.0 * (v - lo) / span - 1.0;   // legendre/chebyshev min-max
    else if (code == 0)  z = (v - mean) / std_safe;          // hermite z-score
    else                 z = v - lo + 1e-9;                  // laguerre shift
    if (has_clip) z = z < clip_lo ? clip_lo : (z > clip_hi ? clip_hi : z);
    out[i] = z;
}
"""
_PREWARP_TRANSFORM_KERNEL = None
_PREWARP_CODE = {"hermite": 0, "legendre": 1, "chebyshev": 1, "laguerre": 2}


def _get_prewarp_transform_kernel(cp):
    """Lazily compiles and caches the fused prewarp min-max/z-score/shift+clip ``RawKernel`` (module-level singleton; pickle-safe)."""
    global _PREWARP_TRANSFORM_KERNEL
    if _PREWARP_TRANSFORM_KERNEL is None:
        _PREWARP_TRANSFORM_KERNEL = cp.RawKernel(_PREWARP_TRANSFORM_SRC, "prewarp_transform")
    return _PREWARP_TRANSFORM_KERNEL


def _gpu_apply_prewarp(cp, x, spec):
    """Device port of ``hermite_fe.apply_operand_prewarp``. ``x`` is a device array (native float dtype);
    returns a device float64 column. Raises for any basis/path not ported so the caller falls back to the
    host copy."""
    basis = str(spec["basis"])
    xf = x.astype(cp.float64, copy=False)
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
    clen = _grb._PREWARP_CLENSHAW_GPU.get(basis)  # dict carved to _gpu_resident_basis (Tier E 2026-06-22); _grb bound at module bottom, resolved at call time
    if clen is None:
        raise ValueError(f"prewarp basis {basis!r} not GPU-ported")
    pp = dict(spec["preprocess"])
    # FUSED transform: the per-column min-max / z-score / shift + clip done in ONE kernel (scalar params) vs
    # the cupy sub/mul/div/clip chain. Same f64 ops -> bit-identical; cupy chain fallback on any kernel error.
    code = _PREWARP_CODE[basis]
    clip = pp.get("clip")
    has_clip = clip is not None
    if basis in ("legendre", "chebyshev"):      # _apply_minmax
        lo = float(pp["lo"]); span = float(pp["hi"]) - float(pp["lo"]) + 1e-12; mean = 0.0; std_safe = 1.0
        clip_lo = -float(clip) if clip is not None else 0.0; clip_hi = float(clip) if clip is not None else 0.0
    elif basis == "hermite":                    # _apply_zscore
        lo = 0.0; span = 1.0; mean = float(pp["mean"]); std_safe = max(float(pp["std"]), 1e-12)
        clip_lo = -float(clip) if clip is not None else 0.0; clip_hi = float(clip) if clip is not None else 0.0
    else:                                        # laguerre: _apply_shift
        lo = float(pp["lo"]); span = 1.0; mean = 0.0; std_safe = 1.0
        clip_lo = 0.0; clip_hi = (float(clip) + 1e-9) if clip is not None else 0.0
    try:
        xfc = cp.ascontiguousarray(xf)
        nrow = int(xfc.size)
        z = cp.empty(nrow, dtype=cp.float64)
        threads = 256
        _get_prewarp_transform_kernel(cp)(((nrow + threads - 1) // threads,), (threads,),
            (xfc, np.int32(code), np.float64(lo), np.float64(span), np.float64(mean), np.float64(std_safe),
             np.float64(clip_lo), np.float64(clip_hi), np.int32(1 if has_clip else 0), np.int64(nrow), z))
    except Exception:
        if basis in ("legendre", "chebyshev"):
            z = 2 * (xf - pp["lo"]) / (pp["hi"] - pp["lo"] + 1e-12) - 1
            if clip is not None:
                z = cp.clip(z, -float(clip), float(clip))
        elif basis == "hermite":
            z = (xf - pp["mean"]) / max(pp["std"], 1e-12)
            if clip is not None:
                z = cp.clip(z, -float(clip), float(clip))
        else:
            z = xf - pp["lo"] + 1e-9
            if clip is not None:
                z = cp.clip(z, 0.0, float(clip) + 1e-9)
    coef_list = [float(v) for v in np.asarray(spec["coef"], dtype=np.float64).reshape(-1)]
    return clen(cp, z, coef_list)


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
    if name == "signed":  # sign(x)*|y| -- non-symmetrical
        return xp.sign(x) * xp.abs(y)
    if name == "ratio_abs":  # x/(|y|+1) -- non-symmetrical
        return x / (xp.abs(y) + 1.0)
    # --- maximal ---
    if name == "logaddexp":
        return xp.logaddexp(x, y)
    if name == "pow":  # non-symmetrical; negative^frac -> nan, scrubbed downstream (matches np.power)
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
    """Builds the human-readable ``bop(ua(a),ub(b))`` candidate name strings in the same nested-loop order as ``_COMBOS``/the candidate matrix columns."""
    return [f"{bop}({ua}({a_label}),{ub}({b_label}))" for ua in _MINIMAL_UNARY for ub in _MINIMAL_UNARY for bop in _MINIMAL_BINARY]


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

# COLUMN-MAJOR twin of fused_gen (ARCHITECTURAL FUSE, 2026-07-03, nsys-driven): writes the SAME per-cell
# value ``v`` for candidate ``c`` at row ``i`` into a (K, n) C-order buffer (``out[c*n + i]``) instead of the
# (n, K) row-major slot (``out[i*K + c]``). The caller allocates ``out`` as (K, n), so a downstream that
# previously produced a column-major view of the (n,K) matrix (``_transpose_to_cm`` for the coalesced radix
# edges, or ``cp.percentile(cand, axis=0)`` sorting the strided axis) reads the (K, n) buffer DIRECTLY -- the
# transpose kernel / strided percentile-sort is GONE. Bit-identical values: out_cm[c, i] == out_rm[i, c] == v
# (same operand gathers, same safe-div, same non-finite scrub); only the STORE address changes. The thread
# mapping is flipped too (i = tid % n, c = tid / n) so consecutive threads own consecutive rows of one
# column -> the operand reads ``ua[ua_idx[c]*n + i]`` AND the ``out[c*n + i]`` write are BOTH coalesced.
_FUSED_GEN_CM_SRC = r"""
extern "C" __global__
void fused_gen_cm(const double* __restrict__ ua, const double* __restrict__ ub,
                  const int* __restrict__ ua_idx, const int* __restrict__ ub_idx,
                  const int* __restrict__ bop, const long long n, const int K,
                  double* __restrict__ out) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (tid >= total) return;
    long long i = tid % n;
    int c = (int)(tid / n);
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
    out[(long long)c * n + i] = v;   // COLUMN-MAJOR (K, n): out_cm[c, i] == out_rm[i, c], c*n+i < K*n < 2^63
}
"""
_FUSED_GEN_CM_KERNEL = None  # module-level singleton (lazy-compiled; pickle-safe twin of _FUSED_GEN_KERNEL)


def _get_fused_gen_kernel():
    """Lazily compiles and caches the row-major fused-generation ``RawKernel`` (one launch fills the whole (n,K) candidate matrix)."""
    global _FUSED_GEN_KERNEL
    if _FUSED_GEN_KERNEL is None:
        import cupy as cp
        _FUSED_GEN_KERNEL = cp.RawKernel(_FUSED_GEN_SRC, "fused_gen")
    return _FUSED_GEN_KERNEL


def _get_fused_gen_cm_kernel():
    """Lazily compiles and caches the column-major twin of ``_get_fused_gen_kernel`` (writes a (K,n) buffer for coalesced downstream reads)."""
    global _FUSED_GEN_CM_KERNEL
    if _FUSED_GEN_CM_KERNEL is None:
        import cupy as cp
        _FUSED_GEN_CM_KERNEL = cp.RawKernel(_FUSED_GEN_CM_SRC, "fused_gen_cm")
    return _FUSED_GEN_CM_KERNEL


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
    """Lazily compiles and caches the grand-fusion RawKernel that generates, bins, and joint-histograms every candidate against all y-vectors in one launch, without ever materialising the (n,K) candidate matrix."""
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
_QLEVELS_CACHE: dict = {}  # (nbins, work-dtype) -> cp.linspace(0,100,nbins+1) device vector; read-only, shared


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


def _fused_generate_block(ua_cm, ub_cm, combos_block, *, column_major: bool = False):
    """Generate the (n, len(combos_block)) candidate matrix for ``combos_block`` in ONE kernel launch.

    ``ua_cm`` / ``ub_cm`` are the (U, n) C-CONTIGUOUS post-unary caches for operands a / b, where
    U = len(_MINIMAL_UNARY) (row u = _UNARY_IDX[name]). This layout lets the kernel address column u
    via ``ua[u*n + i]``; the caller builds them ONCE and reuses across chunks. Returns a row-major
    (n, K) cupy float64 matrix, bit-equal to the cupy elementwise path (same ops, same safe-div, same
    nan_to_num -- validated maxdiff 0).

    ``column_major=True`` runs the COLUMN-MAJOR twin kernel instead: same per-cell value, written into a
    (K, n) C-order buffer (``out_cm[c, i] == out_rm[i, c]``). Callers that only need a column-major view of
    the candidate values (e.g. the exact per-candidate percentile edges) take this to SKIP a downstream
    transpose / strided-axis sort -- BIT-IDENTICAL values, only the layout differs. Consumers that read the
    matrix as (n, K) (the resident MI / discretize kernels) MUST keep the default row-major output."""
    import cupy as cp

    # Pin the operand-plane row count to the unary set: the kernel does NO bounds check on ua_idx, so a
    # silent row/index drift would be an out-of-bounds device read. Assert it can't.
    assert ua_cm.shape[0] == len(_MINIMAL_UNARY) == ub_cm.shape[0], (ua_cm.shape, ub_cm.shape)  # nosec B101 - internal invariant check in src/mlframe/feature_selection/filters, not reachable with untrusted input
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
    total = n * K
    threads = 256
    blocks = (total + threads - 1) // threads
    if column_major:
        out = cp.empty((K, n), dtype=cp.float64)  # (K, n) C-order == column-major over the (n, K) matrix
        _get_fused_gen_cm_kernel()((blocks,), (threads,), (ua_cm, ub_cm, ua_idx, ub_idx, bop, np.int64(n), np.int32(K), out))
        return out
    out = cp.empty((n, K), dtype=cp.float64)
    _get_fused_gen_kernel()((blocks,), (threads,), (ua_cm, ub_cm, ua_idx, ub_idx, bop, np.int64(n), np.int32(K), out))
    return out


def _unary_stack_cm(xp, x):
    """(U, n) C-contiguous stack of the minimal unary transforms of ``x`` (U=len(_MINIMAL_UNARY), row u = _UNARY_IDX[name]).

    Writes each unary DIRECTLY into a preallocated (U, n) buffer instead of ``xp.stack([...])`` (nsys-driven,
    2026-07-03): stack materialises ALL U unary temporaries in the list SIMULTANEOUSLY, then copies each into
    its output -> peak = U operand-plane temporaries + the output. The per-row assignment holds only ONE unary
    temporary at a time (peak = 1 + the output), dropping the stack copy AND the U-fold transient VRAM. The
    plane is C-contiguous by construction (``xp.empty`` default order), so the prior ``ascontiguousarray`` is a
    no-op removed. Row values are byte-identical to the stack path (same ``_unary_apply`` output, copied into
    row u). ``x`` is a float64 device operand here (both callers pass ``cp.asarray(.., float64)``); every
    minimal unary preserves that dtype, so ``plane`` dtype = ``x.dtype`` matches what stack would infer."""
    plane = xp.empty((len(_MINIMAL_UNARY), x.shape[0]), dtype=x.dtype)
    for u in _MINIMAL_UNARY:
        plane[_UNARY_IDX[u]] = _unary_apply(xp, u, x)  # copies the fresh unary result into its row slice
    return plane

# Per-element GPU working-set multiple for the cupy plug-in MI: the (n, k) cand f64 + argsort int64 +
# X_binned int64 + flat int64 coexist, so budget ~5x the bare cand bytes. Conservative -> avoids the
# n=300k VRAM cliff (measured: unchunked (300k,384) thrashed the 4GB card to 60s).
_GPU_MI_WORKING_MULTIPLE = 5

# G3 (2026-06-22): K-chunk width is governed by what FRACTION of free VRAM the working set may occupy.
# WIDER fraction = fewer/larger chunks = fewer kernel launches+reductions (the nsys launch/sync overhead),
# bounded by the card not thrashing. 0.25 is the conservative source-code default; per
# feedback_use_kernel_tuning_cache_for_gpu the live fraction is looked up per-host from the
# kernel_tuning_cache (sweep+spec in the _gpu_resident_k_chunk_ktc sibling, re-exported below; carved for
# the <1k-LOC budget). Chunk width is per-column-INDEPENDENT, so the candidate MI is selection-equivalent
# regardless of the chunk boundary (the sweep ranks fractions by WALL only).
_GPU_K_CHUNK_VRAM_FRACTION_DEFAULT = 0.25
_GPU_K_CHUNK_VRAM_FRACTIONS = (0.25, 0.40, 0.55, 0.70)  # discrete grid the per-host sweep ranks


# Below this n the GPU launch/transfer dominates and the CPU njit grid wins (bench: 20k -> 0.76x,
# 100k -> 1.79x); the dispatcher routes < this to CPU. Provisional crossover; a later sweep can tune it.
_GPU_RESIDENT_MIN_N = 50_000


def _gpu_k_chunk_vram_fraction(n: int) -> float:
    """Per-host VRAM fraction for :func:`_gpu_k_chunk` (lazy KTC resolver in sibling); safe default on miss."""
    try:
        from ._gpu_resident_k_chunk_ktc import gpu_k_chunk_vram_fraction as _resolve
        return _resolve(n)
    except Exception:
        return _GPU_K_CHUNK_VRAM_FRACTION_DEFAULT


def _gpu_k_chunk(n: int, *, free_bytes: int | None = None,
                 bytes_per_elem: int = 8, working_multiple: int | None = None,
                 max_cols: int | None = None, vram_fraction: float | None = None) -> int:
    """Max candidate columns to score in ONE on-device batch so the working set stays within a FRACTION of
    free VRAM -- bounds peak GPU memory like the CPU RAM governor, removing the large-n cliff.

    ``bytes_per_elem`` / ``working_multiple`` describe the caller's per-column resident footprint; defaults
    (8 x ``_GPU_MI_WORKING_MULTIPLE``) match the f64 MI prototype, the CODES path passes 4 x ~4 for a ~3x
    wider sub-chunk = ~3x fewer launches. Per-column-INDEPENDENT, so codes/MI are bit-identical regardless of
    chunk boundaries. ``max_cols`` caps to the caller's column count (defaults len(_COMBOS)). G3: the VRAM
    fraction is per-host KTC-tuned; an explicit ``vram_fraction`` (sweep probe) overrides; clamp to (0, 0.9]
    so a stale cache entry never zeros the budget nor exceeds 90% of free VRAM."""
    import cupy as cp

    if free_bytes is None:
        # DEVICE-free ALONE is wrong when cupy's mempool holds the card (2026-07-02, gap-analysis root cause):
        # cudaMemGetInfo reports ~0 device-free once the resident FE caches + operand tables have grown the
        # pool to fill VRAM, yet the pool holds hundreds of MB of REUSABLE free blocks that this batch's
        # (n, k_chunk) working set can allocate WITHOUT an OS request. Sizing on device-free alone collapsed
        # k_chunk to 1 -> the materialise+bin loop ran ONE candidate per launch (measured 11,664 per-candidate
        # launches + their host syncs = the dominant GPU-idle source). Add the pool's free capacity
        # (total - used) so the budget reflects what can actually be allocated -> the loop batches again.
        _dev_free, _dev_total = cp.cuda.runtime.memGetInfo()
        try:
            _mp = cp.get_default_memory_pool()
            _pool_free = int(_mp.total_bytes()) - int(_mp.used_bytes())
        except Exception:
            _pool_free = 0
        # Cap the pool-free contribution at a fraction of TOTAL VRAM so one batch's working set stays bounded
        # (the full pool_free would size a multi-GB (n, k_chunk) allocation that churns the pool on this 4 GB
        # card without shortening the host-bound wall). Enough to collapse the per-candidate loop to a handful
        # of block launches, not enough to thrash. Adaptive (no hardcoded byte threshold).
        free_bytes = int(_dev_free) + min(max(0, _pool_free), int(_dev_total * 0.25))
    frac = _gpu_k_chunk_vram_fraction(n) if vram_fraction is None else float(vram_fraction)
    frac = min(0.9, max(1e-3, frac))
    budget = max(1, int(free_bytes * frac))
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
    a_gpu = cp.asarray(a, dtype=cp.float64)  # the ONE H2D of the raw operands
    b_gpu = cp.asarray(b, dtype=cp.float64)
    n = int(a_gpu.shape[0])
    y_i64 = np.ascontiguousarray(y_codes, dtype=np.int64)
    # (U, n) unary caches (U=len(_MINIMAL_UNARY)) stay resident + reused across every chunk; the FUSED kernel generates
    # each candidate CHUNK in ONE launch (vs a Python loop of ~K cupy binary ops + nan_to_num + temps --
    # bit-equal, ~15x faster generation). Only the chunk matrix is bounded, so peak VRAM is governed.
    ua_cm = _unary_stack_cm(cp, a_gpu)
    ub_cm = _unary_stack_cm(cp, b_gpu)
    y_gpu = cp.asarray(y_i64)  # upload y ONCE, reused across chunks (the host wrapper re-H2D'd it per chunk)
    # y's min/max are a fit-constant -> compute ONCE here (one cp.min/max + one scalar D2H) and pass into the
    # resident MI for every chunk, instead of the resident MI recomputing them per chunk (nsys 2026-06-22:
    # that per-chunk recompute was the #1 source of the cp.max reductions + tiny D2H syncs). Bit-identical.
    _ymm = cp.asnumpy(cp.stack((cp.min(y_gpu), cp.max(y_gpu))))
    _ymin = int(_ymm[0]); _ncls = int(_ymm[1]) - _ymin + 1
    k_chunk = _gpu_k_chunk(n)
    mi_parts: list[np.ndarray] = []
    for start in range(0, len(_COMBOS), k_chunk):
        block = _COMBOS[start : start + k_chunk]
        cand = _fused_generate_block(ua_cm, ub_cm, block)  # one-launch fused generation
        # Both operands are already device-resident (cand from the fused kernel, y_gpu uploaded once), so the
        # H2D-free resident MI scores the chunk with no per-chunk transfer (bit-identical to the host-input
        # variant -- test_resident_batch_cuda_matches_host_input pins maxdiff 0). y_min/n_classes hoisted.
        mi_parts.append(
            np.asarray(_plugin_mi_classif_batch_cuda_resident(cand, y_gpu, nbins, y_min=_ymin, n_classes=_ncls, relax_binning=True), dtype=np.float64)
        )
        del cand
    return _candidate_names(), np.concatenate(mi_parts) if mi_parts else np.empty(0)


# G3 (2026-06-22): the per-fraction PROBE + sweep + kernel_tuner registration live in the
# ``_gpu_resident_k_chunk_ktc`` sibling (LOC budget), re-exported at the bottom of this module.

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
        """Regenerates candidate column ``idx`` from the cached post-unary operand columns, scrubbing non-finite values to 0."""
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
        approx[start : start + len(idxs)] = cp.asnumpy(_sortfree_mi_gpu(block, y_i64, nbins))
        del block
    # REFINE: exact argsort MI on the top-refine_k by approx MI.
    k = min(int(refine_k), len(_COMBOS))
    top = np.argsort(approx)[-k:]
    refine_mat = cp.empty((n, k), dtype=cp.float64)
    for jj, idx in enumerate(top):
        refine_mat[:, jj] = _col(int(idx))
    exact_top = np.asarray(_plugin_mi_classif_batch_cuda(refine_mat, y_i64, nbins), dtype=np.float64)
    mi = approx.copy()
    mi[top] = exact_top  # exact MI for the head, approx for the cheap tail
    return _candidate_names(), mi
    # MEASURED (GTX 1050 Ti, K=384, refine_k=48): exact winner preserved 6/6 seeds @ n=100k, but the
    # end-to-end speedup over the pure-exact GPU path is only ~1.16x @100k / ~1.06x @1M -- NOT the ~2x the
    # MI-only argsort=69% microbench implied. Once candidate GENERATION + the histogram MATH (paid over
    # all 384 in the prescreen) are counted, trimming only the argsort to the top-48 saves less than
    # argsort's in-kernel share. Real + exact, but modest; the bigger lever is cutting generation/math
    # (or a fused sort-free EXACT kernel), not just the sort. Kept as a validated option, not the default.


# --- Tier E carve re-exports (2026-06-22): prewarp/orth-basis + grand-fusion block -> _gpu_resident_basis.py,
# residency-buffer + radix-select block -> _gpu_resident_select.py (carved VERBATIM under the 1k ceiling).
# Rebind EVERY moved name (public AND underscore-private) into this namespace so all existing
# ``from .._gpu_resident_fe import X`` paths still resolve byte-for-byte (production code + tests import
# several private names by path). At the BOTTOM so the siblings' top-level back-imports resolve.
from . import _gpu_resident_basis as _grb
from . import _gpu_resident_select as _grs
from . import _gpu_resident_k_chunk_ktc as _grk
for _m in (_grb, _grs, _grk):
    for _n in dir(_m):
        if not _n.startswith("__") and _n not in globals():
            globals()[_n] = getattr(_m, _n)
del _m, _n
