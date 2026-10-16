"""GPU-resident FE candidate generation + MI (prototype, gated, un-wired).

The terminal phase of the matrix-native FE replatform, and the only part with genuine NEW value: the
reason the GPU LOST the MI dispatch (see _hermite_fe_mi / the 2026-06-19 perf series) was the per-call
H2D upload + many tiny kernels - ~700ms/call of pure overhead the on-device compute (~10-36ms) was
dwarfed by. The fix is to keep the data RESIDENT: upload the raw operands ONCE, generate the whole
unary x binary candidate grid ON the GPU (cupy elementwise), and score the entire grid in ONE big-k
batch-MI call. No per-candidate transfer, one large kernel - exactly the regime the contention-aware
sweep showed the GPU winning (n=100k k>=100: cuda < njit).

GATED behind ``MLFRAME_FE_GPU_RESIDENT`` and imported by nothing in the production FE path: this is a
validated prototype proving the approach (correct MI vs the CPU path + faster at large n), not yet the
production recipe-integrated generator. It mirrors the MINIMAL unary/binary preset (enough to express
a**2/b and log(c)*sin(d)); the full catalog + recipe replay is the follow-up once the win is locked.

Non-pure op handling: ``smart_log`` shifts by the FULL-column nanmin (computed once on-device here, the
same anchor the CPU recipe freezes), ``div`` reproduces the exact ``y==0 -> eps`` branch - so the
on-device candidate equals the CPU one to fp round-off.

BENCH (GTX 1050 Ti, K=384 minimal-preset candidates per pair, median of 3, warm; vs numpy-gen + njit
batch MI). Keeping data resident flips the GPU from the old 3x LOSER (per-call H2D path) to a WINNER
that SCALES - and the VRAM-bounded K-chunk (``_gpu_k_chunk``, mirroring the CPU RAM governor on-device)
removes the large-n cliff entirely, with the on-device MI matching the CPU path to fp round-off (argmax + values):
  * n=20k   : CPU 287ms   / GPU 379ms  -> 0.76x  (small n: GPU launch dominates -> dispatcher routes CPU)
  * n=100k  : CPU 1771ms  / GPU 854ms  -> 2.07x  (k_chunk=141)
  * n=300k  : CPU 7013ms  / GPU 2046ms -> 3.43x  (k_chunk=47 - was a 0.12x cliff before chunking)
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
    equi-width sub-hist: Spearman 0.9993, MI maxdiff 0.029 - statistically excellent and still 4.36x
    faster binning. But the TOP candidate is a near-tie among EQUIVALENT a**2/b spellings, and a 0.03 MI
    perturbation reorders that tie -> argmax flips -> NOT bit-exact -> unsafe as a drop-in for the
    exact-result contract. Good as an opt-in APPROXIMATE fast mode only.
The exact+fast path (NEXT): prescreen all candidates with the sort-free MI, then re-score only the
top-K (margin-guarded) with the exact ``cp.argsort`` MI - exact winner at a fraction of the sort cost.
Numbers recorded so this is not re-derived blind; the exact ``cp.argsort`` path stays the default.

bench-attempt-rejected (2026-06-20, re-test under the RELAXED selection-equivalence bar - user: same
features selected is enough, bit-identity not required): the tail-compressed sort-free binning IS faster
(K=48 heavy-tail: 100k 1.47x, 1M 2.04x; 99.5% code-agree, Spearman 0.99992) BUT FLIPS the END-TO-END FE
selection at n=100k - the ~0.5% of codes differing at quantile edges perturb the noise-gate/MI ranking
past the compound-recovery threshold, yielding 9 fragmented features instead of the single fused
``add(sqr(a)/b, log(c)*sin(d))`` compound (the clean-compound gate goes RED). f32 sort keys preserve the
selection (100% code-agree) but are NOT faster here (f64->f32 cast overhead: 0.77x@100k, 1.12x@1M). So
there is NO selection-safe binning speedup on this GPU (GTX 1050 Ti) even with bit-identity waived - the
exact cp.percentile sort is at the bandwidth floor; beating it needs a rank-EXACT sort-free kernel
(radix-rank, roadmap #2), not an approximate quantile. cp.percentile stays the default.
"""

from __future__ import annotations

from typing import Any

import logging
import os

import numpy as np

from mlframe.feature_selection.filters._safe_scale import guarded_scale


def _guarded_span(pp) -> float:
    """The stored min-max span, guarded against the bounds' own magnitude so a tiny-but-real range is not swamped by an absolute pad."""
    lo, hi = float(pp["lo"]), float(pp["hi"])
    return float(guarded_scale(hi - lo, max(abs(lo), abs(hi))))


logger = logging.getLogger(__name__)

# REVIEW ROADMAP (multi-agent critique; items dispositioned FUTURE - captured so they are
# not re-derived). PERF (ranked payoff/effort, all must stay bit-exact -> validate maxdiff 0 + argmax on
# HEAVY-TAILED a**2/b candidates, not just uniform): (1) f32 sort keys for binning (~1.5-1.8x on the 69%
# sort; exact only with row-index tie-break, else gate to prescreen); (2) radix-rank RawKernel replacing
# cp.argsort + the uncoalesced scatter (exact, removes the (n,K) int64 sort_idx); (3 + 5) DONE:
# fused SHARED-MEM atomic histogram + grand fusion - ``fused_gen_bin_hist`` (one block per candidate)
# re-generates each candidate value, bins it against the EXACT cp.percentile edges, and accumulates the
# (P1, nbins, K_y) joint histogram in shared-mem atomics in ONE kernel per chunk, so the (n,K) float
# matrix, the (n,K) int codes, the (n,K) D2H disc AND the noise-gate's (n,K) d_base / (rows*n*K) flat
# index are NEVER materialised (Option F1 recompute-not-store + #3 atomic hist). BIT-IDENTICAL (maxdiff 0,
# argmax match) -> selection EXACT; measured GTX 1050 Ti K=384 nperm=25: 100k 2.16x +3.0x less peak GPU
# mem, 300k 2.15x +2.75x, 1M 3.39x +2.26x. ``grand_fused_pair_mi_fused`` (default ON via
# MLFRAME_FE_GPU_GRAND_FUSION); the non-fused body is the exact fallback when the shared hist overflows
# the per-block limit.
# PROTOTYPE-ONLY (measured): grand-fusion CANNOT be wired into the PRODUCTION canonical fit.
#   Production candidate-MI runs through _pairs_chunks._compute_one_fe_chunk -> gpu_materialise_discretize_
#   codes_host -> _dispatch_batch_mi_with_noise_gate (NOT gpu_pairs_fe_mi, which returns None on the
#   canonical fixture). That path REQUIRES the (n,K) float candidate matrix on host - the downstream
#   survivor / usability-corr / ext-val / multi-emit stages read the CONTINUOUS columns out of the chunk
#   buffer - so a recompute-not-store fused kernel that skips materialisation would leave the buffer
#   uninitialised (garbage survivors). Grand-fusion's whole payoff (never materialise (n,K)) is thus
#   incompatible with production. AND the production e2e bottleneck is NOT the MI counting the fused kernel
#   collapses - profiled n=100k/79s: cp.percentile's SORT in _gpu_resident_discretize_codes = 12.9s
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
# (registry vs _unary/_binary_apply vs CUDA switch - _safe_div is the single spec) meta-tests.

# Minimal-preset op NAMES (kept in sync with feature_engineering.create_*_transformations "minimal").
from ._gpu_resident_shared import (  # noqa: F401 - moved to the leaf module; re-exported here
    _BINOP_CODE,
    _COMBOS,
    _COMBO_IDX_CACHE,
    _DEFERRED_HOST_FILL,
    _DEFERRED_HOST_FILL_LOCK,
    _DEFERRED_HOST_FILL_MAX,
    _GPU_K_CHUNK_VRAM_FRACTIONS,
    _GPU_K_CHUNK_VRAM_FRACTION_DEFAULT,
    _GPU_MI_WORKING_MULTIPLE,
    _GPU_RESIDENT_FE_CACHE_LOCK,
    _GPU_RESIDENT_MIN_N,
    _MINIMAL_BINARY,
    _MINIMAL_UNARY,
    _QLEVELS_CACHE,
    _RESIDENT_CODES_HANDOFF,
    _RESIDENT_CODES_HANDOFF_MAX,
    _UNARY_IDX,
    _cuda_present,
    _env_gpu_default_on,
    _fill_deferred,
    _gpu_has_free_vram,
    _gpu_k_chunk,
    _gpu_k_chunk_vram_fraction,
    _gpu_min_free_vram_mb,
    _quantile_levels_dev,
    _stash_deferred_host_fill,
    _stash_resident_codes,
    _unary_apply,
    _xp_special,
    fe_gpu_defer_host_codes_enabled,
    fe_gpu_resident_codes_enabled,
    _FUSED_GEN_BIN_HIST_KERNEL,
    _FUSED_GEN_BIN_HIST_SRC,
    _FUSED_GEN_CM_F32_KERNEL,
    _FUSED_GEN_CM_F32_SRC,
    _FUSED_GEN_CM_KERNEL,
    _FUSED_GEN_CM_SRC,
    _FUSED_GEN_KERNEL,
    _FUSED_GEN_SRC,
    _binary_apply,
    _build_candidate_matrix,
    _candidate_names,
    _fused_generate_block,
    _get_fused_gen_bin_hist_kernel,
    _get_fused_gen_cm_kernel,
    _get_fused_gen_kernel,
    _unary_stack_cm,
    cpu_pair_candidate_mi,
    fe_gpu_grand_fusion_enabled,
)

# RESIDENCY REPLATFORM MAP (measured - the production chunk path, NOT this prototype).
# Goal: kill the FE chunk's bulk float-buffer D2H. ``gpu_materialise_discretize_codes_host`` D2Hs the
# full (n,K) float candidate buffer via ``out_cand`` - measured 6.7 GB total across the canonical
# n=100k fit (15 calls), a large slice of the ~10.5s ``cupy.get`` wall. The codes (for MI) are produced
# RESIDENT and don't need that buffer; only a HANDFUL of host reads do.
#   * Final survivors ALREADY recompute from raw via ``_rebuild_full_survivor_col`` (subsample path,
#     in _pairs_core.py); the buffer feeds only the INTERMEDIATE subsample scoring reads in
#     check_prospective_fe_pairs (best-config ~1625/1749, multi-emit ~2126/2137, MI-replay ~1499).
# Proposed win = KEEP the chunk-batch GPU-codes path, pass ``out_cand=None`` (skip the 6.7 GB D2H), route
# the few intermediate reads through the validated recompute helper (_config_by_i / _rebuild_full_survivor_-
# col, bit-identical), gated on the GPU-fused path. MEASURED CONSTRAINT: pure recompute (no buffer, no
# chunk-batch) is 3x SLOWER (217s vs 69s), so chunk-batch is essential. (Turnkey deferral scaffold - gate
# MLFRAME_FE_GPU_DEFER_FLOAT, out_cand=None, _config_by_i recompute at the 3 buffer reads in
# check_prospective_fe_pairs - was implemented then bench-rejected below; see git 72d1c364.)
# BENCH-REJECTED (2026-06-21): the float-D2H deferral was IMPLEMENTED (gated scaffold) and measured -
# clean paired A/B (6 cold fits, quiet box) = OFF 160.0s vs ON 162.6s = 0.98x, a WASH. The 6.7 GB float
# D2H OVERLAPS the GPU/CPU compute (async), so skipping it does not cut wall - cProfile's cupy.get time
# is overlapped WAIT, not serial transfer. This is the THIRD GPU-data-movement wash after radix coalescing
# (7x kernel, 0x wall) and noise-gate launch-batching (1.00x): the canonical FE fit is COMPUTE-bound, not
# transfer/residency-bound. So residency/round-trip reduction is a dead lever here; reverted (commit after
# 72d1c364). Real wall levers are COMPUTE reduction (subsample scale-down - 64s@30k -> 36s@8k but plateaus
# at a ~30s FE-component floor; selection holds on the canonical formula) or cutting FE work, NOT transfers.


def fe_gpu_resident_enabled() -> bool:
    """Whether the GPU-resident FE prototype is active. OFF unless ``MLFRAME_FE_GPU_RESIDENT`` truthy."""
    return os.environ.get("MLFRAME_FE_GPU_RESIDENT", "").strip().lower() in ("1", "true", "on", "yes")


def fe_gpu_resident_basis_mi_enabled() -> bool:
    """Whether the matrix-native RESIDENT orth-FE basis-MI path is active (Piece 3).

    This line used to say "DEFAULT OFF", directly
    contradicting the "DEFAULT ON when CUDA is present" statement two paragraphs below for the SAME flag -
    the real behavior (confirmed by ``_env_gpu_default_on``) is DEFAULT ON when a usable CUDA device is
    present, DEFAULT OFF otherwise (env-overridable either way).

    When ON, ``hybrid_orth_mi_fe`` builds the univariate orth-basis candidate matrix ON the device
    (BATCHED ``_gpu_evaluate_basis_matrix``) and scores its plug-in MI with
    ``_plugin_mi_classif_batch_cuda_resident`` - with NO per-call H2D (the dispatcher's 2x trap).

    DEFAULT ON when CUDA is present. Validated: selection-EQUIVALENT (it swaps the njit RANK
    binning for the GPU equi-frequency-edge binning in the basis ranking - the orth-basis recovery pins
    test_layer21/22, the canonical single_compound, and the full biz-value hybrid_orth suite all pass with
    it on: 385 passed) AND FASTER (clean canonical 100k wall 34.8s -> 30.7s, ~12%; the batched build also
    clears the p200 high-feature perf budget). Opt out with ``MLFRAME_FE_GPU_RESIDENT_BASIS_MI=0``. Any GPU
    failure / unported basis falls back to the host path per-call (never a correctness regression)."""
    return _env_gpu_default_on("MLFRAME_FE_GPU_RESIDENT_BASIS_MI")


def fe_gpu_routing_enabled() -> bool:
    """Whether the orth-FE basis ROUTING (basis_route_by_signal) runs on the device. DEFAULT ON.

    When ON, ``_gpu_build_and_score_univariate`` picks each source column's orth-basis on the GPU (batched
    eval of all candidate bases x degrees on the RESIDENT operand matrix - reused, no second H2D - + a
    batched |Pearson corr| vs the resident continuous y, argmax per column) instead of the per-column host
    ``basis_route_by_signal``. Opt out with ``MLFRAME_FE_GPU_ROUTING=0``.

    Routing is SELECTION-BEARING (the chosen basis is baked into the EngineeredRecipe) and the GPU basis eval
    is parity-<1e-6, NOT bit-identical, so a near-tie between two bases can in principle flip the argmax. Two
    things make it safe to default on: (1) only the corr VALUES come from the GPU - the argmax/tie logic is
    byte-identical to the host router, so a flip needs a genuine sub-epsilon corr tie where the two bases are
    equally usable by construction; (2) a per-column host fallback covers any GPU failure / degenerate column
    (never a correctness regression).

    VALIDATED 2026-06-22 (quiet GTX 1050 Ti): SELECTION-EQUIVALENT - test_gpu_routing_parity matches the
    host router on every clear-margin column across 3 seeds (only sub-1e-16 chebyshev/legendre numerical ties
    diverge), and the full selection-bearing suite passes with routing ON (test_mrmr_feature_engineering +
    layer21/22 orth-recovery + canonical single_compound + hybrid_orth biz = 112 passed). WALL: measured on
    the RESIDENT operand matrix (the real residency scenario - M is already on-device for the basis-MI
    build), routing is host 229ms vs GPU 214ms = 1.07x (30k x 24 cols, 24/24 matching); the wiring also
    reuses that single upload, removing a redundant (n, n_cand) H2D the prior per-column path implied. (An
    earlier A/B wrongly charged the GPU an H2D that residency mode does not pay - corrected here.)"""
    return _env_gpu_default_on("MLFRAME_FE_GPU_ROUTING")


# RESIDENT-CODES HANDOFF (gated). The FE chunk path calls ``gpu_materialise_discretize_codes_
# host`` (producer) and ``_dispatch_batch_mi_with_noise_gate`` (consumer) as SEPARATE steps - the producer
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

# DEFERRED HOST-CODES FILL. When the resident-codes path is on, the (n, K) host int codes are
# the SINGLE largest D2H of the canonical fit (audit n=100k: _gpu_resident_fe.py codes-fill = 1691 MB / 160
# transfers, the whole D2H budget) - yet they are REDUNDANT whenever the resident-CUDA noise gate is the
# consumer (it reads the device codes IN PLACE; host ``disc_2d`` is only touched for ``.shape``). The
# producer can therefore DEFER the codes D2H: return an UNFILLED host buffer + keep the device codes, and
# materialise the host buffer LAZILY only if a host-reading consumer (the analytic gate / CPU njit kernel /
# non-resident GPU path) actually needs it. The dispatch calls ``ensure_host_codes_filled`` before any
# host-codes read, and ``take_resident_codes`` before the resident path - so the 1691 MB D2H is paid ONLY
# on the host paths that read it (none at the canonical n=100k fit) and SKIPPED when the resident gate
# consumes the device codes. BIT-IDENTICAL: the host buffer, if ever filled, is ``device_codes.get()`` -
# the exact bytes the eager D2H produced -> selection unchanged. Keyed on ``id(out)`` like the handoff (same
# host array flows producer -> dispatch synchronously). Module-level singleton -> never on pickled state.
# Per-host-array deferred fill (was a single slot; see _RESIDENT_CODES_HANDOFF). Keyed by id(host_codes).
# The FE pair scoring binds codes on several threads at once, each of which then waits on the numba kernel lock, so
# more than _DEFERRED_HOST_FILL_MAX unfilled buffers can be in flight. Evicting one used to drop its record, after which
# ensure_host_codes_filled was a no-op and the CPU kernel read an np.empty buffer: garbage codes indexing out of bounds
# (heap corruption, access violations on Windows) or, when it did not crash, MI computed on garbage. Eviction now fills
# the buffer first, and the registry is guarded because several threads stash, fill and clear at once.


def ensure_host_codes_filled(host_codes: np.ndarray) -> None:
    """If ``host_codes`` was returned UNFILLED with a deferred device->host fill registered for it (same id,
    shape, dtype), D2H the device codes into it NOW (once; idempotent). No-op when there is no deferred fill
    for this array (it was filled eagerly, or is an unrelated array). Called by the dispatch on every
    host-codes-reading path (analytic gate / CPU njit kernel / non-resident GPU path) so the codes D2H is
    paid exactly when - and only when - a host consumer reads them. Bit-identical to the eager fill."""
    with _DEFERRED_HOST_FILL_LOCK:
        h = _DEFERRED_HOST_FILL.get(id(host_codes))
        if h is None:
            return
        host_ref, _dev, shape, dtype, _filled = h
        host = host_ref()
        if host is not host_codes or tuple(host_codes.shape) != shape or np.dtype(host_codes.dtype) != dtype:
            return
        # D2H the resident device codes into the caller's host buffer (the exact bytes the eager path produced).
        _fill_deferred(h)


def take_resident_codes(host_codes: Any) -> Any:
    """Return the resident device codes stashed for ``host_codes`` iff they match it (same id, shape,
    dtype); else None. The resident-CUDA gate consumes the returned device codes IN PLACE (no host read).
    The handoff is NOT cleared here (so a later host-reading branch in the SAME dispatch - e.g. the
    analytic gate that runs after this pop - can still trigger the deferred host fill from the same device
    codes); the dispatch clears everything via ``clear_resident_codes_handoff`` in its finally."""
    with _DEFERRED_HOST_FILL_LOCK:
        h = _RESIDENT_CODES_HANDOFF.get(id(host_codes))
    if h is None:
        return None
    dev, shape, dtype, host_ref = h
    # id() is recycled once the producer's array dies: only the very array that was stashed may claim its codes.
    if host_ref() is host_codes and tuple(host_codes.shape) == shape and np.dtype(host_codes.dtype) == dtype:
        return dev
    return None


def clear_resident_codes_handoff(host_codes: np.ndarray | None = None) -> None:
    """Drop the resident-codes handoff + the deferred host-fill record so device memory is not pinned past
    the dispatch that produced it, and a stale entry can never satisfy a later, unrelated dispatch. Called
    by ``_dispatch_batch_mi_with_noise_gate`` in a finally after it has decided the consumer.

    When ``host_codes`` is given, drops ONLY that host array's entries (so a concurrent fit's in-flight
    handoff is not wiped); when None, clears all (legacy whole-cache reset)."""
    with _DEFERRED_HOST_FILL_LOCK:
        if host_codes is not None:
            _RESIDENT_CODES_HANDOFF.pop(id(host_codes), None)
            _DEFERRED_HOST_FILL.pop(id(host_codes), None)
            return
        _RESIDENT_CODES_HANDOFF.clear()
        _DEFERRED_HOST_FILL.clear()


# GPU-RESIDENT BINNING DTYPE. The FE candidate buffer is ALREADY float32 (the njit/CUDA
# materialise writes float32; the chunk buffer is float32). The original GPU-resident discretize path
# UP-CAST it to float64 before cp.percentile + cp.searchsorted, so the bandwidth-bound full sort moved
# 2x the bytes for NO accuracy gain over the f32 source. Binning NATIVELY in float32 removes the cast
# AND halves the bytes the sort/percentile + searchsorted scan, and (measured this run) PRESERVES the
# clean-compound FE SELECTION (the acceptance bar - same features, not bit-identical edges; f32-vs-f64
# code agreement ~100%). So float32 is the DEFAULT binning dtype; the exact f64 path stays one env flip
# away (``MLFRAME_FE_GPU_BINNING_DTYPE=float64``) as the bit-identical fallback if a future host/data
# combination ever destabilises selection. The prior "f32 was SLOWER" result (docstring, 2026-06-20)
# measured an f64->f32 CAST overhead on an f64-source path; here the source is f32 so there is no cast.
# Implementation: _gpu_resident_discretize_codes bins in the INPUT's native dtype, so the float32 host
# FE-buffer paths (gpu_discretize_codes_host / gpu_materialise_discretize_codes_host) bin in float32 with
# NO up-cast, while the float64 grand-fused MI path (gpu_resident_pair_candidate_mi / grand_fused_pair_mi,
# fed by _fused_generate_block -> float64) stays float64 (bit-identical). MLFRAME_FE_GPU_BINNING_DTYPE
# (float32|float64) force-overrides the working dtype host-wide if a future host/data combo needs it.


# GPU-ONLY catalog: user directive - EVERYTHING runs on the GPU, NO CPU fallback. The six
# ops that lack a cupy/cupyx equivalent or need cross-row / data-dependent / emath handling are
# TEMPORARILY DISABLED in BOTH catalogs (here AND feature_engineering.create_*_transformations) so the
# GPU grid == the CPU grid exactly:
#   unary : grad1, grad2 (cross-row np.gradient - wrong under row-blocked generation), dawsn (no cupyx)
#   binary: agm (no cupyx), logn (np.emath complex-base), binom (deferred with the others)
# TODO(restore): re-enable each behind a real GPU kernel (grad via full-column cp.gradient; dawsn/agm via
# custom kernels; logn via log/log on the shifted operands) once the GPU residency path is validated.
# Mirrors feature_engineering's enabled catalog EXACTLY so the GPU candidate == the CPU one to fp
# round-off (selection-equivalent - the FE perf bar).
_FULL_UNARY = (
    "identity",
    "neg",
    "abs",
    "sqr",
    "reciproc",
    "sqrt",
    "log",
    "sin",  # minimal
    "sign",
    "rint",
    "qubed",
    "invsquared",
    "invqubed",
    "cbrt",
    "invcbrt",
    "invsqrt",
    "exp",  # medium
    # GPU-DISABLED(restore): "grad1", "grad2",  (cross-row np.gradient)
    "sinc",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",  # maximal trig
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "erf",
    "gammaln",  # maximal hyp+special
    # GPU-DISABLED(restore): "dawsn",  (no cupyx)
)
_FULL_BINARY = (
    "mul",
    "add",
    "sub",
    "div",
    "max",
    "min",  # minimal
    "abs_diff",
    "hypot",
    "signed",
    "ratio_abs",  # medium
    "logaddexp",
    "pow",
    "heaviside",
    "greater",
    "less",
    "equal",
    "beta",  # maximal
    # GPU-DISABLED(restore): "agm" (no cupyx), "logn" (np.emath), "binom"
)


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
            z = (xf - float(pp["mean"])) / float(pp["std"])
            u = cp.sign(z) * (z * z)
            axis = (u - float(pp["lo"])) / float(pp["span"])
        else:
            axis = (xf - float(pp["lo"])) / float(pp["span"])
        coef = np.asarray(spec["coef"], dtype=np.float64).reshape(-1)
        out = cp.zeros_like(axis)
        for i, f in enumerate(pp["freqs"]):
            if 2 * i + 1 >= coef.size:
                break
            ang = 2.0 * np.pi * float(f) * axis
            out = out + float(coef[2 * i]) * cp.sin(ang) + float(coef[2 * i + 1]) * cp.cos(ang)
        return out
    clen = _grb._PREWARP_CLENSHAW_GPU.get(basis)  # dict carved to _gpu_resident_basis (Tier E); _grb bound at module bottom, resolved at call time
    if clen is None:
        raise ValueError(f"prewarp basis {basis!r} not GPU-ported")
    pp = dict(spec["preprocess"])
    # FUSED transform: the per-column min-max / z-score / shift + clip done in ONE kernel (scalar params) vs
    # the cupy sub/mul/div/clip chain. Same f64 ops -> bit-identical; cupy chain fallback on any kernel error.
    code = _PREWARP_CODE[basis]
    clip = pp.get("clip")
    has_clip = clip is not None
    if basis in ("legendre", "chebyshev"):  # _apply_minmax
        lo = float(pp["lo"])
        span = _guarded_span(pp)
        mean = 0.0
        std_safe = 1.0
        clip_lo = -float(clip) if clip is not None else 0.0
        clip_hi = float(clip) if clip is not None else 0.0
    elif basis == "hermite":  # _apply_zscore
        lo = 0.0
        span = 1.0
        mean = float(pp["mean"])
        std_safe = float(pp["std"])
        clip_lo = -float(clip) if clip is not None else 0.0
        clip_hi = float(clip) if clip is not None else 0.0
    else:  # laguerre: _apply_shift
        lo = float(pp["lo"])
        span = 1.0
        mean = 0.0
        std_safe = 1.0
        clip_lo = 0.0
        clip_hi = (float(clip) + 1e-9) if clip is not None else 0.0
    try:
        xfc = cp.ascontiguousarray(xf)
        nrow = int(xfc.size)
        z = cp.empty(nrow, dtype=cp.float64)
        threads = 256
        _get_prewarp_transform_kernel(cp)(
            ((nrow + threads - 1) // threads,),
            (threads,),
            (
                xfc,
                np.int32(code),
                np.float64(lo),
                np.float64(span),
                np.float64(mean),
                np.float64(std_safe),
                np.float64(clip_lo),
                np.float64(clip_hi),
                np.int32(1 if has_clip else 0),
                np.int64(nrow),
                z,
            ),
        )
    except Exception as e:
        logger.warning("fused prewarp-transform kernel failed, falling back to the exact cupy chain: %s", e)
        if basis in ("legendre", "chebyshev"):
            z = 2 * (xf - pp["lo"]) / _guarded_span(pp) - 1
            if clip is not None:
                z = cp.clip(z, -float(clip), float(clip))
        elif basis == "hermite":
            z = (xf - pp["mean"]) / pp["std"]
            if clip is not None:
                z = cp.clip(z, -float(clip), float(clip))
        else:
            z = xf - pp["lo"] + 1e-9
            if clip is not None:
                z = cp.clip(z, 0.0, float(clip) + 1e-9)
    coef_list = [float(v) for v in np.asarray(spec["coef"], dtype=np.float64).reshape(-1)]
    return clen(cp, z, coef_list)


# (ua, ub, bop) combo order, matching _candidate_names / _build_candidate_matrix column order.


# FUSED-GENERATION CUDA RawKernel: one launch computes the WHOLE (n, K) candidate block from the cached
# post-unary columns, replacing the Python loop of ~K separate cupy binary ops + nan_to_num + temporaries.
# Each thread owns one (row, candidate) cell: gather its two operand columns by op-code index, apply the
# binary op (safe-div mirrors the CPU y==0 -> eps branch), scrub non-finite to 0, write row-major (n,K).

# COLUMN-MAJOR twin of fused_gen (ARCHITECTURAL FUSE,, nsys-driven): writes the SAME per-cell
# value ``v`` for candidate ``c`` at row ``i`` into a (K, n) C-order buffer (``out[c*n + i]``) instead of the
# (n, K) row-major slot (``out[i*K + c]``). The caller allocates ``out`` as (K, n), so a downstream that
# previously produced a column-major view of the (n,K) matrix (``_transpose_to_cm`` for the coalesced radix
# edges, or ``cp.percentile(cand, axis=0)`` sorting the strided axis) reads the (K, n) buffer DIRECTLY - the
# transpose kernel / strided percentile-sort is GONE. Bit-identical values: out_cm[c, i] == out_rm[i, c] == v
# (same operand gathers, same safe-div, same non-finite scrub); only the STORE address changes. The thread
# mapping is flipped too (i = tid % n, c = tid / n) so consecutive threads own consecutive rows of one
# column -> the operand reads ``ua[ua_idx[c]*n + i]`` AND the ``out[c*n + i]`` write are BOTH coalesced.
# float32 twin of the column-major generator: operands, arithmetic and the candidate block in float32 (half the memory traffic of the block, and consumer GPUs run float32 arithmetic far faster than
# float64). Same operator switch, same safe division and the same non-finite scrub; a value that overflows float32 is scrubbed to 0, as the float32 feature scrub does everywhere else.


# GRAND-FUSION kernel (roadmap #5 + #3). One launch computes the joint histograms for ALL
# candidate columns against the original-y AND every shuffled-y, WITHOUT ever materialising the (n,K)
# float candidate matrix, the (n,K) int codes, the (n,K) ``d_base`` index, or the (rows*n*K) flat index
# the cupy noise-gate builds. Each thread owns one (row, candidate) cell: it RE-generates the candidate
# value from the resident post-unary caches (the cheap elementwise gen - recomputed instead of stored,
# Option F1), bins it ONCE via an upper-bound binary search over that candidate's PRE-COMPUTED exact
# ``cp.percentile`` interior edges (so the binning math is IDENTICAL to ``_gpu_resident_discretize_codes``
# -> selection stays EXACT), then atomic-adds 1 into the joint-histogram slot of EVERY y-vector (original
# + permutations) at ``[p, off_k + code*K_y + y_p[row]]`` (Option F3 atomic histogram). Recomputing gen
# + the bin code ONCE per cell and reusing it across all P1 y-vectors is the key: the per-cell work is
# paid once, only the P1 atomic adds scale with the permutation count.
#
# Edges layout: ``edges`` is (K, nbins-1) row-major - the INTERIOR percentile edges ``bin_edges[1:-1]``
# for candidate c, exactly what ``cp.searchsorted(edges, val, side='right')`` consumes. The in-kernel
# search reproduces ``side='right'`` (strictly-greater upper bound) so the code equals the cupy path.
# y-vectors layout: ``y_all`` is (P1, n) int32 (row 0 = original y, rows 1.. = the Fisher-Yates shuffles
# built by the SAME host LCG the noise-gate uses) -> the output ``counts`` is (P1, total_size) int64,
# byte-identical to what ``batch_mi_with_noise_gate_cupy`` produces, so it feeds the SAME bit-exact
# ``_mi_from_counts_cpu`` + ``_gate_from_mi`` reduction unchanged.


# Both caches can be filled from more than one thread under joblib backend="threading"; the lock covers
# the whole get-or-compute-or-insert sequence so two threads racing on the same key don't both pay the
# compute and one's result silently overwrite the other's (harmless here since both are deterministic
# and content-identical, but the race still wastes a device alloc/kernel launch every time it happens).


# Per-element GPU working-set multiple for the cupy plug-in MI: the (n, k) cand f64 + argsort int64 +
# X_binned int64 + flat int64 coexist, so budget ~5x the bare cand bytes. Conservative -> avoids the
# n=300k VRAM cliff (measured: unchunked (300k,384) thrashed the 4GB card to 60s).

# G3: K-chunk width is governed by what FRACTION of free VRAM the working set may occupy.
# WIDER fraction = fewer/larger chunks = fewer kernel launches+reductions (the nsys launch/sync overhead),
# bounded by the card not thrashing. 0.25 is the conservative source-code default; per
# feedback_use_kernel_tuning_cache_for_gpu the live fraction is looked up per-host from the
# kernel_tuning_cache (sweep+spec in the _gpu_resident_k_chunk_ktc sibling, re-exported below; carved for
# the <1k-LOC budget). Chunk width is per-column-INDEPENDENT, so the candidate MI is selection-equivalent
# regardless of the chunk boundary (the sweep ranks fractions by WALL only).

# Below this n the GPU launch/transfer dominates and the CPU njit grid wins (bench: 20k -> 0.76x,
# 100k -> 1.79x); the dispatcher routes < this to CPU. Provisional crossover; a later sweep can tune it.


def gpu_resident_pair_candidate_mi(a: np.ndarray, b: np.ndarray, y_codes: np.ndarray, *, nbins: int = 20) -> tuple[list[str], np.ndarray]:
    """GPU-RESIDENT path: upload ``a``, ``b``, ``y`` ONCE, build the whole candidate grid on the device,
    and score it in ONE big-k batch-MI call - the array never round-trips per candidate. Returns
    ``(names, mi)`` with ``mi`` brought back to host (the only D2H, a (K,) vector). Raises if cupy is
    unavailable (callers gate on :func:`fe_gpu_resident_enabled` + availability)."""
    import cupy as cp

    from . import hermite_fe as _hf  # noqa: F401 - full-init the parent first so the direct

    # ``_hermite_fe_mi`` import below can't trip the _ensure_cuda_kernels back-import cycle.
    from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

    # bench-attempt-rejected (2026-06-20, "eliminate ALL f64 in the GPU FE MI path" on the GTX 1050 Ti):
    # generating + scoring this whole chain in float32 (operands/fused-gen/binning/MI all f32) gave only
    # 1.09x @100k / 1.14x @300k AND FLIPPED the winner at n=300k (div(identity(a),sqrt(b)) ->
    # mul(sqr(a),reciproc(b)) - both spell a**2/b, a near-tie cluster the f32 round-off reorders), i.e.
    # selection instability for a small win. The f64 1/32 penalty does NOT live in the MI math (math-only
    # f32 = 1.00x; see _hermite_fe_mi._plugin_mi_classif_batch_cuda note) but in the BINNING sort, which is
    # captured separately via MLFRAME_FE_GPU_BINNING_DTYPE. So this path stays f64. (proto
    # D:/Temp/_bench_resident_f32.py)
    a_gpu = cp.asarray(a, dtype=cp.float64)  # the ONE H2D of the raw operands
    b_gpu = cp.asarray(b, dtype=cp.float64)
    n = int(a_gpu.shape[0])
    y_i64 = np.ascontiguousarray(y_codes, dtype=np.int64)
    # (U, n) unary caches (U=len(_MINIMAL_UNARY)) stay resident + reused across every chunk; the FUSED kernel generates
    # each candidate CHUNK in ONE launch (vs a Python loop of ~K cupy binary ops + nan_to_num + temps -
    # bit-equal, ~15x faster generation). Only the chunk matrix is bounded, so peak VRAM is governed.
    ua_cm = _unary_stack_cm(cp, a_gpu)
    ub_cm = _unary_stack_cm(cp, b_gpu)
    y_gpu = cp.asarray(y_i64)  # upload y ONCE, reused across chunks (the host wrapper re-H2D'd it per chunk)
    # y's min/max are a fit-constant -> compute ONCE here (one cp.min/max + one scalar D2H) and pass into the
    # resident MI for every chunk, instead of the resident MI recomputing them per chunk (nsys:
    # that per-chunk recompute was the #1 source of the cp.max reductions + tiny D2H syncs). Bit-identical.
    _ymm = cp.asnumpy(cp.stack((cp.min(y_gpu), cp.max(y_gpu))))
    _ymin = int(_ymm[0])
    _ncls = int(_ymm[1]) - _ymin + 1
    k_chunk = _gpu_k_chunk(n)
    mi_parts: list[np.ndarray] = []
    for start in range(0, len(_COMBOS), k_chunk):
        block = _COMBOS[start : start + k_chunk]
        cand = _fused_generate_block(ua_cm, ub_cm, block)  # one-launch fused generation
        # Both operands are already device-resident (cand from the fused kernel, y_gpu uploaded once), so the
        # H2D-free resident MI scores the chunk with no per-chunk transfer (bit-identical to the host-input
        # variant - test_resident_batch_cuda_matches_host_input pins maxdiff 0). y_min/n_classes hoisted.
        mi_parts.append(
            np.asarray(_plugin_mi_classif_batch_cuda_resident(cand, y_gpu, nbins, y_min=_ymin, n_classes=_ncls, relax_binning=True), dtype=np.float64)
        )
        del cand
    return _candidate_names(), np.concatenate(mi_parts) if mi_parts else np.empty(0)


# G3: the per-fraction PROBE + sweep + kernel_tuner registration live in the
# ``_gpu_resident_k_chunk_ktc`` sibling (LOC budget), re-exported at the bottom of this module.

# MEASURED (GTX 1050 Ti, K=384, refine_k=48): exact winner preserved 6/6 seeds @ n=100k, but the
# end-to-end speedup over the pure-exact GPU path is only ~1.16x @100k / ~1.06x @1M - NOT the ~2x the
# MI-only argsort=69% microbench implied. Once candidate GENERATION + the histogram MATH (paid over
# all 384 in the prescreen) are counted, trimming only the argsort to the top-48 saves less than
# argsort's in-kernel share. Real + exact, but modest; the bigger lever is cutting generation/math
# (or a fused sort-free EXACT kernel), not just the sort. Kept as a validated option, not the default.

# --- Tier E carve re-exports: prewarp/orth-basis block -> _gpu_resident_basis.py, its own further carve
# (analytic pair-MI + grand-fusion + dispatch) -> _gpu_resident_pair_mi.py, residency-buffer + radix-select
# block -> _gpu_resident_select.py (all carved VERBATIM under the 1k ceiling). Rebind EVERY moved name
# (public AND underscore-private) into this namespace so all existing ``from .._gpu_resident_fe import X``
# paths still resolve byte-for-byte (production code + tests import several private names by path). At the
# BOTTOM so the siblings' top-level back-imports resolve.
from . import _gpu_resident_basis as _grb
from . import _gpu_resident_pair_mi as _gpm
from . import _gpu_resident_select as _grs
from . import _gpu_resident_k_chunk_ktc as _grk

for _m in (_grb, _gpm, _grs, _grk):
    for _n in dir(_m):
        if not _n.startswith("__") and _n not in globals():
            globals()[_n] = getattr(_m, _n)
del _m, _n

# The loop above copies a sibling's names at the moment it runs. When a sibling is imported FIRST (``_gpu_resident_select`` -> its tail imports
# ``_gpu_resident_discretize`` -> that back-imports this module), this module's tail runs while the sibling is only partially initialised, and the names the sibling
# re-exports from its own carved files are not there yet - ``from ._gpu_resident_fe import gpu_discretize_codes_host`` then failed depending on which module a process
# happened to import first. The module-level ``__getattr__`` below resolves such a name from the siblings once they have finished loading (and caches it).
_REEXPORT_SIBLINGS = (
    "_gpu_resident_basis",
    "_gpu_resident_pair_mi",
    "_gpu_resident_select",
    "_gpu_resident_k_chunk_ktc",
    "_gpu_resident_materialise",
    "_gpu_resident_discretize",
    "_gpu_resident_select_kernels",
)


def __getattr__(name: str):
    """Late lookup of a re-exported name in the carved sibling modules (PEP 562), for the partial-initialisation import orders."""
    if name.startswith("__"):
        raise AttributeError(name)
    import sys as _sys

    for _sib in _REEXPORT_SIBLINGS:
        _mod = _sys.modules.get(f"{__package__}.{_sib}")
        if _mod is not None and name in vars(_mod):
            _value = vars(_mod)[name]
            globals()[name] = _value
            return _value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
