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
