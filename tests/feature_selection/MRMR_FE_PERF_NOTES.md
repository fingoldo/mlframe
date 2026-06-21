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
