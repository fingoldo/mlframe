# MRMR FE perf notes (agentic optimization /loop)

Profiling findings to guide the optimization loop. NOTE: prior memory says mlframe perf is mature
(metric kernels / preprocessing / io are bench-tuned; small-n grinding yields no wins) -- so treat
these as CANDIDATES to verify, not assumed wins. Always bench before/after; document no-wins.

## cProfile: MRMR().fit on CASE2 n=20000 (2026-06-13, tottime, mlframe-side)

| tottime | ncalls | function | note |
|--------:|-------:|----------|------|
| 3.88s | 18 | `discretization/_kernels.py:_searchsorted_2d_right_njit_parallel` | njit_parallel; **18 calls** -- one per fit/FE step? possible redundant re-discretisation |
| 3.44s | 18 | `discretization/_kernels.py:_quantile_edges_2d_njit` | same 18× cadence -- edges recomputed each step |
| 2.51s | 2 | `_feature_engineering_pairs/_pairs_materialise.py:_materialise_chunk` | FE pair materialisation |
| 2.34s(cum) | 34 | `_orthogonal_univariate_fe/_orth_extra_basis_fe.py:_detect_fourier` | Fourier-basis detection in orth FE |
| 0.57s | 1564 | `_orth_extra_basis_fe.py:_corr_sq_centered` | many small calls |

Total fit ~50s (n=20k, includes FE + escalation + orth basis).

### VERDICT (2026-06-13, investigated): both top hotspots are INTRINSIC -> NO perf win

- `discretize_2d_quantile_batch` (the 18x calls): each call discretises a FRESH chunk of MATERIALISED
  FE-candidate columns (`chunk_buffer[:, :col]` in `_pairs_chunks.py:252`), NOT a re-discretisation of
  the same matrix. Distinct data per call -> not redundant; the kernel is already njit_parallel. REJECTED.
- `_detect_fourier_freqs_for_col` (34 calls): per-candidate-column FFT frequency detection for the
  adaptive-Fourier FE generator (`_orth_extra_basis_fe.py:702/757`) -- intrinsic per-column work. REJECTED.

Consistent with the standing finding that mlframe FS/FE perf is MATURE (kernels njit-tuned; small-n
yields no wins). The optimization loop should focus on FE QUALITY / correctness, not perf re-grinding,
unless profiling an UNCONTENDED large-n run surfaces a genuinely redundant call.

### (superseded) original lead candidate
The 18x `_searchsorted_2d_right` + `_quantile_edges_2d` calls (7.3s combined, ~15% of fit) suggest the
discretised matrix / quantile edges are RECOMPUTED per FE step rather than cached+extended when FE
appends columns. If the existing columns' edges are stable across steps, caching them and only
discretising the NEWLY-appended engineered columns would cut this. VERIFY first: confirm the 18 calls
are on overlapping column sets (redundant) vs genuinely new data each time; bench the cache; ensure
bit-identical selection (edges must be frozen identically). Mature-perf caveat: may already be
intentional (FE changes nbins/strategy per step) -- check before assuming redundancy.

---

## 2026-06-16 -- 400k / fe_max_steps=3 cProfile (WALL=585s, clean composite selected)

CPU hotspots (cumtime / tottime):
* **conditional-gate FE = #1 lever (~31% of wall, ~180s+):** cheap_conditional_gate_scan
  (_conditional_gate_fe.py:351, 62s TOTTIME / 181s cumtime) + _gate_grid_mi (:201, 80s)
  + best_existing_op_mi (:143, 46s) + _baseline (:409, 37s). PRIME speed target at scale.
* orth-family MI batch _mi_classif_batch (945 calls, 166s cumtime) -- MI estimation, harder.
* binned_numeric_agg (93s; per_cell_stats_bincount 21s tottime).
* MDLP supervised binning _mdlp_best_split_njit (34s tottime).
TODO (/loop): optimize the conditional-gate scan -- the grid-MI sweep dominates; check whether
the per-(tau,op) MI grid recomputes redundant baselines / can prune the op x threshold grid early.

## 2026-06-16 -- large-n MEMORY: all FE candidates materialized at once

1M x fe_max_steps=3 OOMs a 16GB box: the ~46 FE families EACH append their candidate columns into one
growing `X` frame, so the discretised screening matrix holds (n, raw + EVERY family's candidates)
simultaneously. Mitigations landed: free dead per-family concat-frames before categorize_dataset
(28dc86d4); Haar-wavelet legs float32 (5e8ec8c6). FURTHER: orth-poly working arrays kept float64
(orthogonality); a genuinely chunked/streamed candidate screen (preserving zero-marginal SYNERGY
candidates -- can't pre-discard by marginal MI) is the deeper fix if 1M must fit.

## 2026-06-16 -- polars input gets ZERO feature engineering

All FE families guard `isinstance(X, pd.DataFrame)` and SKIP (with a warning) on polars input -- only
raw MI screening runs, no engineered features. TODO: auto-convert polars->pandas at fit entry when FE
is enabled (fit AND transform consistently) so polars users get full FE (one materialization; the FE
needs pandas anyway -- strictly better than silently skipping FE).

## Scaling profile 2026-06-16 (100k/400k/1M) -- the REAL large-n hotspot is the MI permutation null

bench_scaling (full MRMR.fit, cProfile, fe_max_steps=2) settled two things measured, not guessed:

* **FE candidate CONSTRUCTION is NOT the bottleneck and its share SHRINKS with n:** 12.2% (100k) ->
  3.9% (400k). Absolute construction grew ~linearly (30s->93s) while MI exploded ~17x (130s->2188s).
  Micro-bench (microbench_fe.py, 1M rows): njit/prange beats numpy only ~1.0-1.3x on arithmetic
  (memory-bound) and ~2.5x on transcendentals (log/sin) -- so even a big construction speedup buys a
  few % of wall. CONCLUSION: do NOT re-platform FE construction to numba FOR SPEED (polars/OOM are
  separate, legitimate reasons). This confirms the older `_conditional_gate_fe.py:420-424` bench
  (construction ~0ms of the MI-bound scan) holds AND strengthens at scale.

* **The dominant large-n cost is the MI PERMUTATION NULL:** at 400k the cupy `argsort` permutation
  generator (gpu.py:mi_direct_gpu_batched, and the CPU prange twin) was ~72% of the fit -- thousands
  of O(n) shuffles across the FE scan. (1M OOM'd: peak 9.87GB at 400k on a 15.9GB box; the argsort
  blowup is amplified by the dev GTX 1050 Ti -- ~30x faster on a strong GPU -- but it stays the top
  hotspot by rank regardless.) FIXED 2026-06-16 (commit cda55bcb): the analytic large-n null
  (Miller-Madow null mean + G-test p, `_analytic_mi_null.py`) replaces the shuffles at n>=50k,
  measured 24-35x on the null computation, identical MI/null + decision-equivalent p. NEXT large-n
  lever if more is needed: the joint-hist / binning path (`_binned_numeric_agg_fe`, MDLP), not
  construction.

## Analytic-null equivalence + minimum-permutations study (2026-06-16)

Two questions settled empirically (D:/Temp/equiv_study.py, minperm2.py; pinned by
test_analytic_mi_null.py::test_permutation_converges_to_analytic):

* **The analytic null IS the permutation null's nperm->infinity limit, not an approximation.**
  Both estimate the same independence null; the permutation is a Monte-Carlo estimate (error
  ~1/sqrt(nperm)). At n=80k the permutation null_mean converges 0.00049 (32 perms) -> 0.00051
  (= analytic) and p converges 0.25 -> 0.36 (= analytic, |dp| -> 0.0002); for genuine signal p=0 at
  every nperm. So the analytic value is MORE accurate than any finite-nperm run (zero MC noise).

* **Minimum permutations for a stable keep/reject DECISION: ~32.** Panel of 24 candidates
  (strong/weak/borderline/noise) at n=80k, keep-set vs the analytic ground truth: nperm<=8 admits
  1-3 Monte-Carlo noise false-positives; nperm=16 matches but only 2/3 seeds; **nperm=32 matches the
  reference on 3/3 seeds** and is rock-stable above. This independently validates the existing
  ``_NULL_MEAN_MIN_PERMS=32`` floor as the minimum safe permutation budget for the n<50k path (below
  the analytic threshold). Above 50k the analytic null (= the nperm->inf limit) supersedes it.

## 2026-06-17 -- large-n FE OOM RESOLVED: 1M fits a 16GB box

A 1M-row fe_max_steps=2 fit went from OOM (projected ~21GB) to COMPLETING at ~10.1GB peak on a
16GB box (WALL 1176s), recovering the genuine a**2/b + log(c)*sin(d) structure. Three layered cuts,
each measured on this box (bench_fe_peak_memory + D:/Temp/oom_1m.py):

* **float32 discretization input** (commit 64923354, `MLFRAME_DISCRETIZE_FLOAT32=1`, OPT-IN): the
  ~21GB->~10GB step. categorize_dataset copied ALL numeric cols into one float64 array (a 2nd
  full-frame copy); quantile/MDLP edges don't need float64. Selection IDENTICAL float32-vs-float64 on
  the canonical 60k fit. Kept opt-in (binning is edge-sensitive; accuracy-first default = float64).
* **Fourier-detection subsample cap** (commit 76081a89, `MLFRAME_FOURIER_DETECT_MAX_N`, default 200k,
  DEFAULT-ON): the coarse (grid x n) sin/cos planes OOM'd next; detection is a heuristic, a uniform
  row-subsample preserves the dominant frequencies. n<=cap untouched.
* **FDR maxT null int32 codes** (commit 76081a89, `MLFRAME_FDR_NULL_INT32`, DEFAULT-ON): the uncaught
  OOM driver -- pooled_permutation_null_gain_floor's (n_cand x n) scaled_flat + (nperm x n) y_perms
  were int64; the values are joint codes < nbins_x*nbins_y, so int32 is BIT-IDENTICAL (A/B confirmed)
  and halves both GB-scale pools.

To run 1M today: set `MLFRAME_DISCRETIZE_FLOAT32=1` (the other two are default-on). On a normally-free
16GB box the ~10.1GB peak fits; on a loaded box free RAM must exceed the peak (a stale 11h-hung pytest
eating 2GB commit had to be reclaimed during this work). DEEPER future cut (not needed for 1M now):
float32 on the engineered candidate frame itself (the float64 X base), which needs the same
binning-safety validation as the discretization lever.

## 2026-06-17 -- 1M SPEED root cause (precise) + re-platform target

After the analytic-null + GPU-off-switch fixes, a 300k fit is 695s (1M ~20min), dominated by:
* orth-FE MI scoring `_plugin_mi_classif_batch_njit` 233s -- called 1171x in small per-family/per-gate
  batches, so its `parallel=True` prange barely engages.
* joblib internal sleep-poll 121s -- the FE pair search (`_run_fe_step` -> `parallel_run` ->
  `compute_pairs_mis`) parallelizes over pair-CHUNKS via joblib `backend="threading"` (threading is
  deliberate: loky would pickle the large X per worker -> memory blowup at large n). But
  `compute_pairs_mis` runs a PYTHON per-pair loop calling `mi_direct` -> GIL-bound glue -> threading
  does NOT parallelize it (1 core / ~12% CPU), and the main thread sleep-polls joblib (~0.01s x
  thousands of small tasks).
* conditional-gate-scan 51s + binned-agg 38s + binning sorts (partition/searchsorted/reduce) ~67s.

RE-PLATFORM TARGET (the only path to true multi-core here): replace the Python-per-pair / per-family MI
loops with ONE batched nogil-numba kernel over ALL pairs/candidates at once (no per-pair Python glue,
no joblib layer), so a single prange uses every core WITHOUT the process-pickle memory cost that forced
threading. Plus: batch the orth-FE MI scoring across families (one big (n, total_cands) call), and
float32 the candidate frame (generation-stage RAM). This is the chosen full FE re-platform (todo #3);
design at MRMR_FE_NUMBA_REPLATFORM_DESIGN.md, plan staged for per-phase bit-parity validation.

## 2026-06-17 -- FULL speed anatomy + path to the <=20s goal (15 cols)

Measured at 300k x 15 cols (all recover the genuine a**2/b + log(c)*sin(d) structure):
  exhaustive (default)            695s
  fe_fast_search=True (200k sub)  297s
  fe_fast_search + 25k subsample  214s   <- subsample levers give ~3.3x, QUALITY PRESERVED

The existing `fe_fast_search` + `*_subsample_n` knobs (detection rank-stable; recipe replays full-n,
so bit-safe for OUTPUT) are the cheap 3.3x. The remaining ~214s is EVENLY DISTRIBUTED across the FE
families (no single hotspot once subsampled):
  binned_numeric_agg per_cell_stats_bincount 38s (+50s cumtime) -- NOT subsampled
  orth-FE MI _plugin_mi_classif_batch_njit 30s (was 233s; subsample helped)
  joblib threading sleep-poll 22s
  MDLP _mdlp_best_split_njit 10s + partition/sort/searchsorted ~20s
  orth-poly _power_centered 8s + lstsq 7s + _coarse_basis 4s

PATH TO <=20s (the chosen full re-platform -- needs ALL of these, no single knob suffices):
  1. Subsample EVERY family's detection (binned_agg + orth-poly + mdlp are still full-n) -- extend the
     fe_fast_search subsample wiring uniformly (each bit-safe via recipe-replay, like the Fourier cap).
  2. Batch ALL candidate-MI into ONE njit-parallel / cupy call over all cores+GPU (microbench: batching
     1-col calls -> one call = 3.4x; bit-identical since per-column MI is independent).
  3. Pre-bin candidates ONCE -> O(n) bincount MI (drop the per-candidate argsort: partition/sort ~20s).
  4. Replace joblib-threading (GIL + sleep-poll) with the direct parallel njit / a process pool with
     shared-memory X (no per-worker pickle).
  5. float32 candidate frame (bandwidth + the generation-stage RAM).
This is a multi-family architectural rewrite of shared MI/FE infra; execute phased with per-step
bit-parity (selection-identity) validation.

## 2026-06-21 -- cProfile @ canonical 100k + the GPU-RESIDENCY verdict (why piecemeal ports backfire)

Driven by the "100% GPU / one transfer" goal (CPU touched once = the initial data copy). Two findings.

### (A) H2D residency floor is ~5 MB and mostly NECESSARY -- do NOT chase it with kernel ports
Instrumented H2D (cp.asarray/array/set byte counting) on the warm canonical 100k fit
(y = a**2/b + f/5 + log(c)*sin(d)): total **14 MB** =
  8.0 MB  input discretization (the legitimate one-time data upload)
  3.36 MB raw operands (14 distinct n-subsample float64, uploaded ONCE each)
  1.68 MB non-plain operand cols (prewarp / gate_med / poly -- host-FIT coefficients)
  ~1 MB   (a_col,b_col,op_code) chunk metadata
Phase-1's resident operand table already replaced the full ~5 MB table re-upload-per-chunk (14x) with
one pass. The whole operand table is only ~5 MB (n_subsample=30k x ~42 operands x 4B), so the residual
is at the architectural floor. Eliminating it needs prewarp/gate_med/poly APPLIED on-GPU (host-fit
coeffs uploaded, tiny) + raw operands sourced from a resident input slice -- both ZERO wall, ULP-risky
against the single_compound pin. See the rejected residency-override note in
_feature_engineering_pairs/_pairs_chunks.py (forcing fused-on-sub-crossover chunks was a no-op / latent
H2D loss).

### (B) cProfile @ 100k (warm; total 56s), mlframe-side tottime
| tottime | cum | calls | function |
|--------:|----:|------:|----------|
| 5.57s | -- | 1252 | cupy._core.core.array (GPU alloc/H2D) |
| 4.42s | -- | 1382 | numba cuda safe_cuda_api_call (launch/sync) |
| 4.07s | -- | 23574 | llvmlite ffi (njit exec/compile) |
| 3.91s | -- | 167 | hermite_fe `_plugin_mi_classif_batch_njit` |
| 1.84s | -- | 42 | orth-FE `_coarse_basis_njit` |
| 1.70s | 7.38s | 160 | `_radix_select_interior_edges` (GPU; tot=per-call Py orchestration) |
| 1.67s | -- | 924 | orth-FE `_power_centered_fused_par_njit` |
| 1.42s+1.15s+0.98s | | | numpy partition / sort / argsort (per-candidate quantile) |
| -- | 17.76s | 2 | `check_prospective_fe_pairs` (whole FE pair search) |
| -- | 5.19s | 42 | `_detect_fourier_freqs_for_col` (prewarp escalation; REJECTED at canonical) |

### VERDICT: piecemeal kernel->GPU ports are COUNTERPRODUCTIVE for residency
The remaining CPU compute (njit FE kernels) runs on HOST-generated feature columns. Moving any one to
GPU re-uploads its operands, so it ADDS H2D and fights "one transfer". Worked example: routing
`fe_baselines.score_pair_baselines` batch-MI (the 167-call / 3.9s `_plugin_mi_classif_batch_njit`) to
the existing `_plugin_mi_classif_batch_cuda` would upload ~480 MB (167 x 30k x 12 x 8B) AND is NOT
bit-identical (equi-frequency-edge vs rank binning at ties -> selection-risky for the trivial-feature
winner). Net: wrong direction on BOTH H2D and bit-parity.

Therefore literal "100% GPU" == the FULL matrix-native RESIDENT FE pipeline (generate candidate columns
ON the GPU from the resident input -> bin on GPU -> MI on GPU -> selection reads device), so nothing
round-trips. That is the multi-family rewrite already scoped in the 2026-06-17 "path to <=20s" entry
above + MRMR_FE_NUMBA_REPLATFORM_DESIGN.md / the breezy-wishing-gem plan. Wall is compute-bound (~45-56s,
no single dominant lever), so the rewrite is justified by the residency PRINCIPLE, not a wall win.

## 2026-06-21 (cont) -- RESIDENT-FE rewrite R0/R1 shipped: operand table 100% device-built

Executing the "100% GPU / one transfer" rewrite (the verdict above), phased + pin-validated:

- **R0** (commit 817f27f2): build_resident_operand_table uploaded each DISTINCT raw operand separately
  (14 cp.asarray at the canonical fit). Now batches them into per-dtype host matrices, ONE H2D each, and
  builds every GPU column from a strided device VIEW. Native dtype preserved (grouped BY dtype) so the
  unary applies in the exact dtype the CPU saw -> bit-parity invariant held. Result: operand raws are ONE
  transfer; H2D calls 953->941, bytes unchanged (repackaged), 11/11 pins green.
- **R1** (commit 9a915027): the per-operand PRE-WARP columns were the last non-plain operands COPIED from
  the host (~1.68 MB / ~14 cols). Ported hermite_fe.apply_operand_prewarp to the device (_gpu_apply_prewarp:
  cupy preprocess + Clenshaw chebyshev/legendre/hermite-He/laguerre mirroring numpy's float64 op order, +
  fourier_adaptive). col_specs carries {kind:"prewarp", spec}; builder GPU-APPLIES from resident raw + spec
  (host-copy fallback for unported bases). Result: operand table now "144 GPU-built, 0 host-copied" (was
  ~14 host-copied) -> the 1.68 MB non-plain floor ELIMINATED. 11/11 pins green INCLUDING the prewarp-ULP
  tripwire single_compound -> cupy Clenshaw matches host closely enough that prewarp still loses to the
  clean library form. CPU/no-CUDA path untouched. Zero wall (compute-bound).

NET after R0+R1: the FE operand table -- the heart of candidate generation -- is constructed 100% on the
device from one per-dtype raw upload, nothing host-copied. The residual H2D is now genuinely "initial data":
  ~8 MB  baseline discretization of the FULL input (discretize_2d_array_cuda -- the real data copy)
  ~3.36 MB FE operand-raws = the 30k SUBSAMPLE (one upload, R0)
  ~1 MB  (a_col,b_col,op_code) chunk metadata (host-computed control flow)

### R2-R4 assessment (the deeper, lower-value remainder)
- **R2** (literal one transfer): merge the 8 MB baseline-input copy and the 3.36 MB FE-subsample copy into
  ONE upload + GPU row-slice the subsample. These are genuinely DISTINCT arrays (full 100k incl. y-derived
  cols vs 30k feature subsample), uploaded by different modules (the discretization util vs the FE builder)
  with different lifetimes; unifying needs a cross-module resident-input handle threaded through MRMR.fit.
  Deep plumbing for 3.36 MB ZERO-wall of distinct data. Both copies are "initial data" in the user's sense.
- **R3/R4** (pair-search readers / greedy CMI / selection): these are about COMPUTE residency, not H2D --
  the host control flow now reads the already-resident operand table; the heavy MI is already on GPU
  (resident noise gate). Moving the symbolic enumeration/selection itself to the device is a large rewrite
  with no transfer or wall payoff (compute-bound, branchy, GPU-hostile). Principle-only.

## 2026-06-21 (cont) -- FE DECISIONS aligned onto the subsample (the real lever for "99% GPU cProfile")

KEY INSIGHT (user): the FE families were deciding on DIFFERENT data -- the pair-search decides on the
~30k subsample (replays winners full-n), but the orthogonal-FE family decided on the FULL 100k. cProfile
confirmed `_plugin_mi_classif_batch_njit` running at shape (100000, k). That is both a ~3.3x perf waste
AND a methodological inconsistency, AND the reason the CPU njit kernels dominated cProfile (blocking the
"99% attribution to GPU-calling code" goal). The dispatcher deliberately keeps the orth-FE MI on njit
(per-call H2D makes cuda 3x slower end-to-end) -- so the win is NOT to force that MI onto the GPU, it is
to stop feeding it 3.3x too many rows.

FIX (this session): thread `subsample_n=fe_check_pairs_subsample_n` + `random_seed` so each FE family
DECIDES on the same subsample and REPLAYS winners at full n (bit-safe output). Shipped:
  * orth-FE univariate  (hybrid_orth_mi_fe_with_recipes)            commit 5b26795c
  * orth-FE extra-basis (hybrid_orth_extra_basis_fe_with_recipes)   commit d0606b13  <- the Fourier
    periodogram detector (the ~5.2s cumulative hotspot) now detects on the subsample, replays via
    apply_recipe at full n.
RESULT (clean cProfile @ canonical 100k, seed 777): wall 56s -> 46s; `_plugin_mi_classif_batch_njit`,
`_coarse_basis_njit` (1.84->0.85s), `_power_centered_fused_par_njit` (1.67->0.89s), `_detect_fourier`
all dropped OFF the tottime top. Top is now GPU-calling: cupy.core.array 4.7s + numba-cuda
safe_cuda_api 3.2s + cupy.astype 0.9s (~8.8s), then numba-exec (llvmlite) 2.8s, then the residual CPU
candidate quantile-binning njit (sort 1.1 + partition 1.08 + argsort 0.88 + ufunc.reduce 1.15 ~= 4.2s).
11/11 FE pins green throughout (incl. single_compound); selection bit-equivalent.

REMAINING /loop PROGRAM toward "cProfile 99% on GPU-calling code + nvprof metrics optimized":
1. Subsample the rest of the audit's Class-A full-n FE-decision sites (all output-safe via recipe replay):
   orth pair (A1, hybrid_orth_mi_pair_fe_with_recipes -- needs the param added), triplet/quadruplet,
   adaptive-arity/degree, routing, diff-basis, cluster, ksg(O(n^2)!), copula, jmim, tc, cmim, auto/ensemble
   scorers; binned_agg edge+ranking (A23); run_fe_auto_escalation specs (A24); MDLP edges (A25). Many are
   inactive on the canonical default scorer; ksg/copula/dcor matter under non-default routing.
2. UNIFY the subsample DRAW: pair-search uses `_fe_subsample.stratified_subsample_idx` (stratified); the
   new orth-FE fixes use plain `rng.choice` -- so the families still decide on DIFFERENT rows. Route ALL
   through one shared helper + seed so every family decides on the IDENTICAL subsample.
3. Move the residual candidate quantile-binning njit (sort/partition/argsort, ~4s, numba-internal) onto
   the GPU (radix-select path exists) -- H2D is no longer a constraint (compute-on-GPU is the goal).
4. Re-profile to confirm 99% GPU-calling attribution; then nvprof-tune the resident kernels.

## 2026-06-21 (cont) -- subsample cleanup: closed-form FE families done; OOF families excluded

Shipped fe_decide_on_subsample (_mrmr_fit_impl/_helpers.py) + wired the orth pair / triplet /
quadruplet families (commit fd04e619), on top of the inline univariate (5b26795c) + extra-basis
(d0606b13) fixes. The whole CLOSED-FORM orthogonal family now DECIDES on the shared FE subsample
(fe_check_pairs_subsample_n + random_seed) and replays winners at full n. 11/11 FE pins green.

CORRECTNESS BOUNDARY (important): the wrapper rebuilds output by REPLAYING recipes (= transform-time
path). That equals the fit-time column ONLY for CLOSED-FORM families (pure functions of x: orth-poly /
Fourier / spline basis). It is NOT valid for OUT-OF-FOLD / data-dependent encoders:
  * binned_numeric_agg (default ON) = k-fold OOF target-stat encoding -> replaying its recipe uses
    full-train cell stats = the LEAKY transform value, not the OOF fit column. Must subsample only its
    pair/edge DECISION and keep the OOF stats at full n (per-family change, NOT the wrapper). LEFT FULL-N.
  * MDLP edge selection (nbins_strategy default) -> edges-then-searchsorted; separate integration. LEFT.
  * run_fe_auto_escalation (active on failed pairs; closed-form warp specs) -> wrappable in principle but
    has a custom return (candidate dicts, not (X,scores,recipes)); needs a small adapter. PENDING.
  * Alternate scorers (ksg/copula/jmim/tc/cmim/routing/diff/cluster/adaptive/auto/ensemble) are
    closed-form (rank orth-basis columns) -> wrappable, but NON-DEFAULT (fe_hybrid_orth_default_scorer)
    so untested by the canonical pins; wrap + add a non-default-scorer selection test before shipping.

STATUS vs the 99%-GPU-cProfile goal: the orth-FE njit kernels (_detect_fourier 5.2s, _coarse_basis,
_power_centered, _plugin_mi_classif_batch_njit) are now OFF the cProfile top (wall 56->46s). Remaining
CPU is the pair-search candidate quantile-binning njit (sort/partition/argsort ~4s) -> the GPU ops-
residency port (the ORIGINAL main task), to resume after the subsample cleanup.

SEED/STRATIFY note (still open): the inline univariate/extra-basis + the wrapper use plain rng.choice;
the pair-search uses _fe_subsample.stratified_subsample_idx. Unify onto one shared stratified helper so
every family decides on the IDENTICAL rows.

## 2026-06-21 (cont) -- post-subsample cProfile: GPU-dominated; remaining CPU is a scattered long tail

After orth-FE is ON by default + hybrid/escalation DECISIONS subsample (commits d76929d9 / 851e9efd),
cProfile @ canonical 100k (warm, ~52s) is GPU-dominated and ALL orth-FE njit kernels (_detect_fourier,
_coarse_basis, _power_centered, _plugin_mi_classif_batch_njit) are off the top:
  GPU-calling : cupy._core.array 5.8s + numba-cuda safe_cuda_api 3.6s + cupy.astype 1.0s (~10.4s)
  numba exec  : llvmlite ffi 4.3s (MIXED -- njit + numba-cuda kernel LAUNCHES)
  CPU (real tottime via pstats.print_callers, NOT cumtime/dispatch):
    np.median  ~0.6s  <- hermite_fe/_hermite_robust.py:172 (+ :193 _robust_scale MAD), x1020 calls:
                         the robust heavy-tail axis detection during prewarp/orth-basis fitting.
    argsort    ~0.35s <- plugin-MI quantile binning (np.argsort).
    ufunc.reduce ~1.4s <- scattered numpy sums/means in the MI / stats math.
So the remaining clear CPU is ~2.4s of ~52s (~5%), small + scattered -- NOT a single binnable hotspot.

REJECTED (bench-attempt, 2026-06-21): forcing the orth-FE MI backend to GPU (MLFRAME_MI_BACKEND=cuda)
to push attribution onto the device -> WALL 52s -> 105s (2x SLOWER, the per-call H2D penalty the
dispatcher's ground-truth note already records) AND the partition/sort/argsort DID NOT drop (they are
np.median / plugin-argsort, not the _mi_classif_batch binning). So the naive route-to-GPU is wrong on
BOTH wall and attribution -- confirms (again) that the only profitable GPU path here is RESIDENT
candidates (matrix-native), not backend-switching.

PATH TO LITERAL 99% GPU-attribution (long tail, each H2D-risky / fine-grained):
  * batch the per-column robust heavy-tail stats (np.median/MAD over the candidate matrix in ONE pass,
    optionally on GPU) instead of 1020 per-column calls;
  * GPU-resident plug-in-MI binning (the matrix-native rewrite -- candidates built + binned + scored on
    device, the only thing that removes the argsort/reduce without the 2x H2D regression);
  * the remaining numba `llvmlite` time is partly GPU kernel launches (counts as GPU) -- isolate the njit
    vs cuda-launch split before claiming the residual.
The BIG win (orth-FE CPU eliminated from the hot path via subsampling) is DONE; literal 99% is this tail.

## 2026-06-21 (cont) -- matrix-native resident MI: foundation shipped + the remaining port

GROUND TRUTH wall (clean, novel seed, separate processes -- the earlier 238s/100s were measurement
artifacts: A/B of two configs in ONE process poisoned the 2nd timing; cuda-experiment = intentional 2x;
56/52/46 were contention/cache noise): orth-OFF 33.5s vs orth-ON 34.8s -> enabling orth-FE costs only
+1.3s (+4%), SAME selection. The default flip is justified, NOT a 6x regression.

Remaining CPU tail (real tottime, pstats.print_callers) ~2.4s of ~34s (~7%): np.median 0.6s (the robust
heavy-tail axis in the orth-basis PREPROCESS -- _hermite_robust:172/193, x1020), argsort 0.35s + reduce
1.4s (the plug-in-MI binning/hist of HOST orth-FE candidate matrices). The pair-search candidate MI is
already on the resident GPU gate; this tail is the ORTH-FE candidate MI (host).

PIECE 1 (shipped db5e80e9): _plugin_mi_classif_batch_cuda_resident -- H2D-free plug-in MI on resident
cupy (X_gpu,y_gpu). Validated bit-for-bit vs host + unit test. The non-regressing MI entry (the naive
host->GPU route is 2x slower; only resident candidates avoid it).

PIECE 2/3 (remaining -- the large port): build the orth-FE basis candidate matrix ON the GPU so it feeds
Piece 1 with no H2D:
  * _gpu_evaluate_basis_column (cupy): preprocess (zscore/minmax/shift -- simple arithmetic; the ROBUST
    heavy-tail path uses np.median/MAD -> either a cupy median or FALL BACK to host for heavy-tail cols)
    + Clenshaw eval (REUSE the cupy _cheb/_leg/_herme/_lag_clenshaw_gpu shipped in R1, _gpu_resident_fe).
  * batch builder -> resident (n,K) cupy candidate matrix; wire score_features_by_mi_uplift to score
    eng_mi via _plugin_mi_classif_batch_cuda_resident (raw_mi too). Gate on CUDA + n>=crossover; host
    fallback for robust/replay/aux paths. ULP-validate against the orth-basis RECOVERY pins (layer2x) +
    canonical single_compound, NOT just no-crash (basis MI ranking is selection-bearing).
This removes argsort+reduce for orth-FE without the 2x H2D. Diminishing returns (~7% of 34s) -- principle
(99% GPU attribution), per the user's directive.

## 2026-06-21 (cont) -- matrix-native MI Piece 1+2 SHIPPED; Piece 3 wiring plan

Tail-cut + matrix-native progress (all committed, green):
  * np.median tail -28% (1086->786) via fit-once hoists in generate_univariate_basis_features +
    basis_route_by_signal (z depends on (x,basis) not degree) -- BYTE-IDENTICAL (0d068e87, 647b09f4).
  * Piece 1 (db5e80e9): _plugin_mi_classif_batch_cuda_resident -- H2D-free plug-in MI on resident cupy
    (bit-identical to host-input variant, unit-tested).
  * Piece 2 (dbc6e1b0): _gpu_evaluate_basis_column -- orth-FE basis candidates built ON device (cupy
    robust-axis preprocess + Clenshaw); parity to host <1e-6 across 4 bases x 4 distributions (16 passed).
    FIXED a latent bug: R1 _lag_clenshaw_gpu Laguerre recurrence was wrong (L_2(0)=-0.5 vs 1), never
    pinned (canonical prewarp = chebyshev); rewrote to the forward recurrence matching _lagval_njit.

PIECE 3 (remaining wiring -- SELECTION-BEARING, needs full validation before default-on):
  hybrid_orth_mi_fe / score_features_by_mi_uplift currently: generate_univariate_basis_features (host
  numpy candidate matrix) -> _mi_classif_batch (njit -> the argsort/reduce tail). Matrix-native:
    1. build the candidate matrix on device: for each (col, chosen_basis, degree) call
       _gpu_evaluate_basis_column(cp, x_dev, basis, degree, robust_axis=_robust_axis_enabled()) into a
       resident (n_sub, K) cupy matrix (operands already resident from R0/R1; one upload of the subsample
       columns at most). Build raw_X columns on device too (identity, no basis).
    2. score eng_mi + raw_mi via _plugin_mi_classif_batch_cuda_resident (Piece 1) -- NO per-call H2D.
    3. assemble the scores DataFrame + two-gate selection UNCHANGED; D2H only the winning columns (or
       rebuild via recipe) for X_aug.
  RISK: the GPU plug-in MI uses equi-frequency percentile-edge binning vs the njit RANK binning -- an
  approved not-bit-identical trade for the FE pair-search (Spearman 1.0, argmax match) but the orth-FE
  BASIS choice (which basis/degree wins) rides on this ranking, so a tie-cluster reorder could flip a
  basis. GATE behind MLFRAME_FE_GPU_RESIDENT_BASIS_MI (default off), and VALIDATE before flipping:
  test_layer2x orth-basis RECOVERY pins + canonical single_compound + the biz-value hybrid_orth suite
  (the 42-min group) must be selection-equivalent. Only then default-on.
  Expected: removes the orth-FE argsort/reduce (~1.7s) + the residual robust median from the tail ->
  closes most of the gap to 99% GPU-attribution. ~7% of a 34s fit (principle, per the directive).

## 2026-06-21 (cont) -- matrix-native MI Piece 3 SHIPPED (gated); validation verdict

Piece 3 (50907f16): hybrid_orth_mi_fe gated matrix-native path -- candidates built on device
(_gpu_evaluate_basis_column) + MI scored resident (_plugin_mi_classif_batch_cuda_resident), no H2D.
Gate MLFRAME_FE_GPU_RESIDENT_BASIS_MI, DEFAULT OFF.

VALIDATION with the gate FORCED ON (full biz-value hybrid_orth suite): 384 passed, 2 failed. The 2
failures are NOT selection bugs -- both are the SAME perf-budget test (test_layer31
TestPerfBudgets::test_hybrid_p200_under_1s). i.e. the GPU path is SELECTION-EQUIVALENT (every
orth-basis recovery / uplift / form pin passes with it on) but perf-LOSES at HIGH feature count
(p=200 -> ~400 candidate columns): the per-column Python build loop (one _gpu_evaluate_basis_column +
cupy launch per col x basis x degree) has launch overhead that dominates at large K. At the canonical
(5 features -> ~10 candidates, n=30k subsample) it wins; at p200 it loses.

DECISION: gate stays DEFAULT OFF (so the perf budgets pass by default; default behaviour unchanged).
The matrix-native chain is COMPLETE + selection-validated + available opt-in. To flip default-on
without the p200 regression, EITHER:
  (a) BATCH the device build -- evaluate all (col,basis,degree) candidates in vectorised cupy calls
      (one preprocess pass + one Clenshaw pass over the stacked operand matrix) instead of the per-
      column Python loop, killing the launch overhead at high K; OR
  (b) K-AWARE gate via kernel_tuning_cache (GPU only below a measured candidate-count / above an n
      crossover) -- mirror _fe_gpu_discretize_enabled; do NOT hardcode the threshold.
Either makes default-on safe; (a) is the real win (also helps the canonical). Until then the path is
opt-in (MLFRAME_FE_GPU_RESIDENT_BASIS_MI=1), selection-equivalent, with host fallback.

## 2026-06-21 (cont) -- matrix-native MI DEFAULT-ON + dual-profiler verdict

SHIPPED + DEFAULT-ON (3887079b): the resident orth-FE basis-MI path (build candidates on device via the
BATCHED _gpu_evaluate_basis_matrix + score via _plugin_mi_classif_batch_cuda_resident, no per-call H2D).
Validated selection-EQUIVALENT (385 hybrid_orth biz-value + canonical single_compound + layer21/22
recovery + 16 basis-parity) AND FASTER (clean canonical 100k 34.8s -> 30.7s, ~12%) AND clears the p200
high-feature perf budget (the batched build killed the per-column launch overhead). Opt out
MLFRAME_FE_GPU_RESIDENT_BASIS_MI=0; host fallback on any GPU failure.

DUAL-PROFILER VERDICT (canonical 100k, default-on):
* cProfile: the orth-FE basis-MI moved to GPU; the residual CPU sort/partition/argsort tail PERSISTS
  (~20% reduced only) -- it is NOT the orth-FE basis-MI but the OTHER paths (host basis ROUTING eval,
  escalation prewarp, pair-search binning). Full 99%-GPU would need those too (much larger surface).
* nvprof --print-gpu-summary: top kernels = radix_select_f32 38.6% (candidate quantile-edge select,
  272 calls), resident-gate hist 20.8%, cupy_copy__float32 19.7% (largely the radix transpose), 
  fe_materialise 7.2%, bin_codes 4.4%.
* nvprof --metrics (no admin needed) on radix_select_f32: achieved_occupancy 0.98, gld_efficiency
  99.64%, sm_efficiency 98.45% -- ALREADY AT PEAK. No occupancy/coalescing headroom; it is genuine
  quantile-binning work at near-100% efficiency. The cupy_copy f32 is the (K,n) transpose that BUYS that
  99.6% coalescing (removing it 8x's the radix-select -- net win to keep). So the GPU path is well-tuned;
  there is no low-hanging kernel optimization -- the dominant cost is real binning work running optimally.

NET: matrix-native orth-FE MI is on the device, default-on, ~12% faster, selection-equivalent; the GPU
kernels profile near-optimal. Remaining CPU tail is the non-orth-FE paths (routing/escalation/pair-search)
-- a separate, larger residency effort, not a kernel-tuning win.

## 2026-06-22 -- CPU-orchestration / GPU-idle reduction (waves 3-5, all bit-identical)

Attacked the GPU-IDLE CPU cost the dual-profiler verdict left on the table: GPU busy ~40% / idle ~60% of
wall, the idle dominated by CPU-side launch orchestration -- cupy._core.core.array (array creation/H2D,
~5.8s) + numba-cuda safe_cuda_api_call (launch/sync, ~3.6s). Levers are redundant-work removal, NOT
kernel tuning (kernels already profile near-peak, see prior entry). All bit-identical; validated against
the 11 FE selection-equivalence pins + 16 GPU bit-identity parity + resident-MI maxdiff-0.

Wave 3 (d303710f) -- FE pair-search CPU critical path (check_prospective_fe_pairs):
  * cache np.isfinite(_corr_y_cont) once (it never mutates) instead of an O(n=subsample) rescan on every
    _safe_abs_corr call (~5-10/accepted pair: clean-form demotion, noise-wrap veto, leader tie-break).
  * hoist sorted(set(numeric_vars) - set(raw_vars_pair)) out of the per-tied-leader _ev_configs loop
    (raw_vars_pair-invariant); the _rng_extval.choice draw stays per-config so RNG state -- and every
    later pair's tie-break -- is bit-identical.

Wave 4 (8433bce2) -- fit-invariant device-vector caches + resident-MI sync batching:
  * _radix_select_interior_edges: the (nbins-1,) gather-indices bi/ai + interp weight w depend only on
    (n, nbins) (derived from np.linspace, not candidate data) -> cache keyed on (n,nbins); was a list-comp
    + 3 tiny H2D every chunk/pair. (n,nbins) take <=2-3 values per fit.
  * _fused_generate_block: the int32 (ua_idx,ub_idx,bop) trio is a pure function of the combo block (a
    slice of the module constant _COMBOS) -> cache on the block tuple; drops 3 list-comps + 3 H2D/chunk/pair.
  * _plugin_mi_classif_batch_cuda_resident: fuse the two blocking .item() syncs (cp.min/cp.max) into one
    cp.stack + single D2H (2 host stalls -> 1); n_classes reproduced exactly (max_orig - y_min + 1).
  * reframed 3 STALE dispatch tests (test_plugin_mi_classif_dispatch): the host-input
    plugin_mi_classif_(batch_)dispatch now defaults to njit via the ground-truth override (end-to-end fit
    njit 3x faster under contention; GPU win lives on the resident path) and no longer consults the KTC
    lookup; _fallback_mi_backend is njit-unconditional, a persisted per-host region the only cuda route.

Wave 5 (43d5befe) -- extend the same caches to the remaining default-on sites:
  * _grand_fusion_block_counts reuses the shared _COMBO_IDX_CACHE for its identical index trio.
  * _quantile_levels_dev caches cp.linspace(0,100,nbins+1,dtype=work) keyed on (nbins,work) at both
    percentile sites (discretize fallback + unconditional grand-fusion edges pass); read-only -> shared.

All caches are MODULE-LEVEL (not on the MRMR instance) so the pickle contract is untouched, mirroring the
other resident-kernel singletons in _gpu_resident_fe. Net effect: the per-chunk-per-pair tiny-H2D + Python
list-comp churn that fed the cupy._core.core.array bucket is now done once-per-fit-invariant-key (<=2-3
H2D total for the radix/linspace vectors; one per distinct block for the combo trio) instead of once per
chunk per pair. Targets the documented top CPU bucket without touching the near-peak kernels.

## 2026-06-22 (cont) -- wave 6: last per-column launch loop fused; orchestration levers EXHAUSTED

Wave 6 (f9682869):
  * _plugin_mi_classif_batch_cuda_resident: the per-column cp.searchsorted binning loop (k-1 extra launches
    + k int64 temps/chunk) -> the already-validated fused _searchsorted_codes kernel (1 launch). This was
    the ONE resident path that never got the fused-kernel treatment the codes path got. Bit-identical
    (code = #(interior edges <= value) == searchsorted side='right', f64-promoted; own per-column fallback).
    Selection-bearing (default-on orth-FE basis-MI scoring) -> validated vs resident-MI maxdiff-0, the full
    cuda-vs-njit equivalence matrix, 11 FE selection-equiv pins, 16 basis parity (60 passed).
  * _radix_select_interior_edges: gate the 2 .astype(f64) edge-gather casts on dtype (no-op copies on the
    f64 path: 2 alloc+cast launches/chunk removed; f32 path unchanged).

EXHAUSTION VERDICT (two independent read-only audit agents, 2026-06-22): the SAFE bit-identical
launch/sync/alloc orchestration levers are now done. Confirmed already-implemented (NOT missed wins):
fused bin-codes kernel, fused-gen one-launch/chunk, radix-select one-launch/chunk, resident-codes +
deferred/pinned D2H, the [min,max] y-stack batching the two .item() syncs, the fit-invariant interp/
qlevel/combo-idx device caches (waves 4-5), and the now-fused resident-MI searchsorted (wave 6). No
remaining per-loop scalar sync to hoist, no safe cp.fuse opportunity (the MI math is cp.where + reductions
+ broadcasting), no unguarded redundant .astype. nvprof already shows the kernels at peak (radix 98% occ,
99.6% gld). The dominant cost is genuine binning/MI compute running optimally + the inherent CPU
orchestration between launches.

REMAINING (NOT a safe bit-identical win -- do NOT re-grind orchestration): the only larger lever is the
matrix-native residency of the OTHER paths (host basis ROUTING eval, escalation prewarp, pair-search
binning) so their argsort/reduce/median tail moves to the device. That is selection-bearing, a much larger
surface, and must be ULP-validated against the recovery pins + canonical single_compound before default-on
-- a dedicated effort, not an orchestration micro-wave. The threshold-conversion queue
(MRMR_HARDCODED_THRESHOLDS_BENCH.md) is benchmark-gated and needs a QUIET machine (this one is not).

## 2026-06-22 (cont) -- host-path residency scoping: verdict + the ONE specced increment (quiet-machine-gated)

Scoped the three remaining host CPU-tail paths (read-only agent) for GPU residency. Verdict:

* (c) pair-search binning -- ALREADY device-resident (gpu_materialise_discretize_codes_host ->
  _gpu_resident_discretize_codes, cp.percentile + _searchsorted_codes). Further residency is a DOCUMENTED
  WASH (float-D2H deferral A/B 160.0 vs 162.6s = 0.98x, the 3rd such wash); the canonical FE fit is
  COMPUTE-bound, not transfer-bound. Remaining host _quantile_bin in _mi_greedy_cmi_fe is bench-rejected
  for GPU (0.60x AND tie-splits CMI selection). DO NOT PORT.

* (b) escalation prewarp (_fe_auto_escalation) -- at canonical n the path admits NOTHING (same 8 eligible
  pairs, 0 proposed; structurally a no-op when every prescreen pair admitted a column). Its cost is
  lstsq/ALS solves with NO resident twin (would need new cupy ALS, not kernel reuse), and it is the
  highest selection-risk path (noise-admission flips; author already proved CPU-side it is selection-
  fragile). DEFER -- not a reuse-existing-kernels increment.

* (a) host basis ROUTING (basis_route_by_signal) -- the actual np.median/corrcoef host tail. It CAN reuse
  the existing batched kernels (_gpu_evaluate_basis_matrix / _gpu_detect_heavy_tail_batched + a batched
  centered-dot |corr| vs the resident y) and the operand matrix M is already uploaded by the adjacent
  basis-MI path. BUT it is strongly SELECTION-BEARING (picks argmax|corr| over 4 bases x 2 degrees -> the
  chosen basis is baked into the EngineeredRecipe) with HIGH ULP risk: _gpu_evaluate_basis_matrix is
  parity <1e-6 (NOT bit-identical), and a ~1e-7 perturbation through corrcoef flips the argmax on a
  near-tie basis -> different recipe -> different feature. Plus the heavy-tail boolean is a hard branch
  that can flip the whole preprocess on a borderline column.

SPEC for (a) (the queued increment -- do on a QUIET machine with full validation budget):
  - Gated, OPT-IN env MLFRAME_FE_GPU_ROUTING (default OFF), inside _gpu_build_and_score_univariate
    (_orthogonal_univariate_fe/__init__.py:571-643): move the M = cp.asarray(column_stack(used_x)) upload
    BEFORE the routing loop (after the cheap host skip rules pick candidate cols), then replace the
    per-column host basis_route_by_signal (:614) with a batched device router: for each candidate basis x
    degree, eval on resident M (reuse _gpu_basis_preprocess_batched + _gpu_detect_heavy_tail_batched +
    Clenshaw), batched centered-Pearson |corr| vs resident y_continuous, argmax per column. Keep the host
    basis_route_by_signal as the PER-COLUMN exception fallback (never per-fit abort).
  - Route block-size/thresholds through pyutilz.system.kernel_tuning_cache (never hardcode GPU params).
  - Add test_gpu_routing_parity (uniform/gaussian/heavytail/skewed x 4 bases) asserting IDENTICAL chosen
    basis per column vs the host router. The parity (selection-equivalence) is checkable on ANY machine;
    only the WALL justification for a default-ON flip needs a quiet box.
  - Flip to default-ON (opt-OUT, mirroring fe_gpu_resident_basis_mi_enabled) ONLY after an A/B proves
    selection-EQUIVALENCE on Layer21/22 + canonical single_compound + the full hybrid_orth biz suite +
    bench_basis_routing AND a measurable wall win (NOT assumed -- the fit is compute-bound + consumer-GPU
    f64 is 1/32-rate, so a wash/regression is the likely outcome; measure before flipping).

NET STATUS: the SAFE bit-identical orchestration program is complete (waves 3-6, all pushed). The only
remaining lever (a) is selection-bearing, quiet-machine-gated, and a likely wall WASH on this compute-
bound fit -- so it is specced + queued rather than force-shipped. (b)/(c) are defer/do-not-port. No safe
shippable optimization remains on this (non-quiet) machine without risking selection for an unmeasured win.

## 2026-06-22 (cont) -- GPU basis routing implemented + validated: SELECTION-EQUIVALENT but wall WASH (opt-in)

Implemented the queued increment (a): MLFRAME_FE_GPU_ROUTING (default OFF) routes each orth-FE source
column's basis on the device -- _gpu_route_bases_batched in _gpu_resident_fe.py evals all 4 candidate bases
x 2 degrees on the resident operand matrix (reusing _gpu_evaluate_basis_matrix) + a batched |Pearson corr|
vs the resident continuous y (_gpu_batched_abs_corr), then runs the EXACT host argmax (corr VALUES from
GPU, tie/argmax logic byte-identical to host). Wired into _gpu_build_and_score_univariate behind the gate
with a per-column host fallback. Ran on the now-QUIET GTX 1050 Ti.

GATE 1 -- selection-equivalence: PASS.
  * test_gpu_routing_parity (new): GPU router matches host basis_route_by_signal on every clear-margin
    column across 3 seeds x 6 distributions; the ONLY divergences were sub-1e-16 chebyshev/legendre
    numerical ties (gap ~1e-17), correctly classified as allowed (the <1e-3 tie band).
  * full selection-bearing suite with the flag ON: test_mrmr_feature_engineering + layer21/22 orth-recovery
    + canonical single_compound + hybrid_orth biz = 112 passed. No canonical-fixture tie flipped a feature.
  * default-OFF path (refactored candidate-collection loop) regression: 14 passed.

GATE 2 -- wall win: FAIL (WASH). Isolated A/B of the routing step (30k x 24 cols, incl. H2D, 7 reps,
warmed): host median 201ms vs GPU 206ms = 0.98x, 24/24 columns matching. As predicted: routing is a small
compute-bound slice and consumer-GPU f64 is 1/32-rate, so moving it to the device buys nothing on wall.

DECISION: keep DEFAULT OFF (opt-in). The path is implemented, parity-tested, and proven selection-
equivalent on every canonical fixture -- available for the GPU-residency principle and ready if a
datacenter-f64 / large-n host flips the economics (re-bench before defaulting on there). NOT flipped to
default-on because GATE 2 (measurable wall win) is unmet -- host routing stays the fastest default. This
closes the host-path residency program: (a) routing done+validated+opt-in, (b) escalation deferred (no-op
at canonical n), (c) pair-binning already resident. No further FE residency lever remains that wins wall.

## 2026-06-22 (cont) -- CORRECTION: GPU routing benchmark was unfair (charged H2D) -> default-ON

The prior entry's "wall WASH 0.98x" A/B was METHODOLOGICALLY WRONG: it included a cp.asarray H2D of the
operand matrix in the GPU timing, but in residency mode M is ALREADY on the device (uploaded once for the
basis-MI build) -- routing must reuse it, not re-upload. Re-measured with M PRE-RESIDENT (the real
scenario): host 229ms vs GPU 214ms = 1.07x (30k x 24 cols, 24/24 cols matching). Not a wash.

Also fixed the WIRING: _gpu_build_and_score_univariate uploaded the operand matrix TWICE (once for routing
cand cols, once for the used cols' basis-MI build). Now it uploads ONCE and the basis-MI build reuses the
resident matrix by device slice (M = _Mr[:, used_idx]) -- removing a redundant (n, n_cand) H2D per fit.

Given selection-equivalence (re-validated after the slicing change: 14 passed) + the corrected 1.07x +
the removed redundant upload + the GPU-residency principle, MLFRAME_FE_GPU_ROUTING is flipped to DEFAULT
ON (opt-out via =0), mirroring fe_gpu_resident_basis_mi_enabled. Per-column host fallback retained.

## 2026-06-22 (cont) -- F2 100k CPU-vs-GPU wall + dual-profiler + [1,5] fragmentation root-cause

GOLDEN EXAMPLE: y = a**2/b + f/5 + log(c)*sin(d), n=100k, canonical single-compound config
(full_npermutations=10, baseline_npermutations=20, fe_max_steps=2, fe_min_pair_mi_prevalence=1.05).
Measured WARM (KTC sweeps pre-completed -- a cold first-fit pays a one-time ~6min async grid sweep that
otherwise poisons the wall; pre-warm via the ensure_*_tuning hooks). n_iters for the sweeps cut 5->2.

WALL (warm, sweep-free):
  data uniform[0.1,1.1]: GPU 30.1s vs CPU 204.1s = 6.8x  -> ONE clean compound add(div(sqr(a),b),mul(log(c),sin(d)))
  data uniform[1,5]    : GPU 21.6s vs CPU  84.3s = 3.9x  -> 3 features (1 full + 2 frag) -- FRAGMENTS
  CPU==GPU selection on both (validates the equivalence work). default 3/2 == canon 10/20 wall + same compound.

DUAL-PROFILER (the two modes have OPPOSITE bottlenecks):
  * CPU (cProfile): batch_mi_with_noise_gate (info_theory/_batch_kernels.py:203) = 190s tottime ~= 93% of
    the fit -- permutation-noise-gate COMPUTE bound (10/20 perms x n=100k x candidates). Everything else <4s.
  * GPU (nsys): ORCHESTRATION/TRANSFER bound, NOT compute. cub DeviceReduce::Max 52% (71,812 calls) + Sum
    31% (48,078) + RadixSort 15% (23,942) -- all from cupy cp.max/cp.sum/cp.median/percentile in Python
    loops; 381,216 cuLaunchKernel; 100,757 tiny D2H (scalar .item() syncs); 2.35 GB H2D (945 ops, one 58MB)
    -- the H2D is the "100% residency" violation. ncu per-kernel metrics would just confirm the cub kernels
    are NVIDIA-tuned/efficient; the cost is call COUNT + transfers, so the residency lever is batching the
    reductions + keeping operands resident (eliminate the 2.35GB H2D), NOT kernel-level tuning.
  (nvprof is non-functional on CUDA 12.8; used Nsight Systems nsys + Nsight Compute ncu instead.)

[1,5] FRAGMENTATION ROOT CAUSE (quantified, NOT a bug): signal-scale imbalance. On uniform[1,5]
var(a**2/b)=13.8 dominates var(log c*sin d)=0.61, so y~=a**2/b and MI(c/d;y)=0.131 is only 10.5% of
MI(a/b;y)=1.245 -- the c/d half sits near the relevance/prevalence gate floor and never survives to be
fused, so the compound degrades to add(sin(d),a**2/b) (drops the log(c) factor) + 2 fragments. On
[0.1,1.1] the halves are balanced (MI ratio 0.625) -> both recovered -> one clean compound. FIX DIRECTION:
residual-aware FE (after capturing the dominant a**2/b term, search the RESIDUAL y - a**2/b where the c/d
half is no longer dominated) -- the escalation path already residualises but admits ~nothing at canonical
n; making step-2 of the main pair-search residualise (or the gate residual-relative) is the principled fix.
Selection-bearing -> must validate against the single_compound + biz-value suite. (Testing existing knobs
-- fe_max_steps, lower prevalence -- first to see if it is a default tweak vs new code.)

## 2026-06-22 (cont) -- profiling-driven optimizations + distribution-robustness goal

LANDED (validated + committed):
* CPU noise-gate (batch_mi_with_noise_gate, the 93%-of-CPU-fit hotspot): restructured to prange-over-columns
  with a serial perm-inner loop, precomputing all Fisher-Yates shuffles once + a contiguous col buffer +
  one reused histogram (new _perm_failcount_col). Removes npermutations-1 fork/join barriers + strided
  gathers + per-(perm,col) allocs. Bit-identical (integer nfailed reassociates; same reduce order; per-k
  thread-private). 106 passed. (900dd660)
* GPU#1: hoist the fit-CONSTANT y min/max out of the per-chunk resident MI (it was recomputed per chunk per
  pair -- the nsys #1 source of the 71,812 cp.max + bulk of the 100,757 tiny D2H). Optional y_min/n_classes
  params, computed once per pair/build. Bit-identical. 37 passed. (ca85bf3f)
* GPU#5: dedup the basis-INDEPENDENT heavy-tail detection across routing's 4 candidate bases (heavy_host
  computed once, passed in). Bit-identical. 43 passed. (1dc6af37)
* n_iters 5->2 for the per-host KTC sweeps (pyutilz time_backend default + the mlframe sweep call sites) --
  one-time sweeps run ~2.5x faster. (785c212 / fa767dbc)

GOAL ENCODED (test_f2_single_compound_across_distributions, 53a12960): F2 must recover ONE clean compound
under EVERY input distribution. uniform = hard regression guard (passes); scaled_1_5 / heavy_tailed / mixed
/ with_outliers = xfail(strict=True) -- the distribution-robustness gap. Root cause: signal-scale imbalance
(a**2/b dominates Var(y); the weak log(c)*sin(d) half falls below the prevalence gate -> fragments).

REMAINING (large, need a fresh-context validation budget -- each is selection-bearing/core):
1. RESIDUAL-AWARE FE (the distribution-robustness fix). Foundation done (SufficientSummaryVerdict.residual
   surfaced). Wiring: in _fit_impl_core.py:6731, when _ss_verdict.reached is False AND blocking_raw>=0 AND
   residual is not None, set self._fe_residual_target_continuous_ = verdict.residual for the next step;
   in _step_core.py:1108-1118 use it (discretised) as classes_y / _prewarp_y_cont / _usab_y_cont for that
   step and recompute cached_MIs against it (the 1.05 prevalence constant is unchanged -- the residual just
   makes the weak half clear it). DOCUMENTED RISK (_mrmr_class.py:1728-1731): a prior admission-relaxation
   admitted (c,d) but construction still failed -- may also need a separable warp-product proposer. GATE:
   the new goal test (flip xfail->pass) + the single_compound pin must NOT regress + no-noise-admission.
2. GPU#2: keep the source operand matrix + y device-resident across the WHOLE pair sweep (slice columns per
   pair) instead of per-pair cp.asarray -- eliminates the nsys 2.35 GB H2D (the residency violation). Touches
   check_prospective_fe_pairs / pair_candidate_mi_dispatch. Bit-identical (same bytes, uploaded once).
3. Deferred audit tail (MRMR_AUDIT_2026_06_22.md): source-name __ split (9-site naming-convention),
   y.to_numpy 53x hoist, lstsq->normal-eq in _orth_extra_basis_fe deflation, _env_truthy DRY, ctor-defaults
   single-source, evaluation.py carve (1144 LOC).
