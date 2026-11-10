# Closing the gap to 50 s at 100000 x 100: options and what was measured (2026-11-09)

State at the head: default concurrent fit 61.5 s, sequential stage scale 68.4 s (200000 rows: 111.3 s); target 50 s. The profile notes in `audits/2026-10-22/profiles/stage_sample_100k_gpu_2026_11_05.md` hold the numbers behind every line below.

## What is established

- GPU kernel time is about 18 s of a 67 s fit (27 %); no kernel is above 3.4 s; every GPU leaf tried this session was either small or already near the card's limits. Kernel tuning cannot close 11.5 s of wall.
- In the concurrent FE sections the three stage threads are executing Python and numpy (52 thread-s over a 16.6 s window), not waiting on the device; they share one GIL. `nogil=True` on every numba kernel changed nothing (61.7 / 64.1 s against 62.4 s), `n_jobs` changed nothing (61.6 / 62.0 s).
- The whole-fit host leaves are flat: after this session 108.6 host thread-seconds, the largest leaf 2.65 s.
- Row growth (100000 -> 200000 rows) sits in the pair FE (`score_one_pair` 3.5 -> 12.1 s), `screen_predictors` (6.0 -> 11.4 s), `fdr_gain_floor` (3.8 -> 8.3 s); the subsample-decided cascades are flat.

## Options, with cost and risk

1. **Less Python per stage (vectorise the per-group / per-column loops).** Candidates and their measured Python share: `_group_distance_fe._pair_columns` 2.0-2.6 s, grouped-quantile `_one_num_col` + `_member_pct_ranks` 2.1 s, `fit_oof_warp1d` 0.8-0.95 s, `_dense_codes` 0.9 s, `_unified_fe_gate` list comprehension 0.6-0.8 s. Expected wall gain per item: under 1 s each (the stages overlap). Risk: summation order changes the last bits of float32 columns; the family tests compare exact values.
2. **Share work between the scorer variants** (`orth_mi_*`, 12 variants x 1-2 s): they score the same basis columns with different scorers. A shared resident basis matrix would remove the repeated basis generation (judged "mechanism debt" earlier, not built). Expected: a few seconds of `hybrid_variants` (17 s), unmeasured.
3. **Smaller pool for the single-column families** (`fe_shape_pool="relevant"`, built, opt-in): -21 % at 30000 x 300 on the stage-scale data, 2-5 % on planted-signal frames at p = 300, nothing measurable at p = 100.
4. **Python without a GIL** (free-threaded 3.13 builds): not available in this environment (Python 3.11).
5. **Processes**: rejected by the owner.
6. **Do less by default** (fewer families or steps in the default configuration): changes what the fit can find; a product decision, not an optimisation, and not taken here.

## Recommendation

Options 1 and 2 are the only ones that cut the 11.5 s without changing what is found; together they are many small edits with a combined expectation of a few seconds, which does not reach 50 s on its own. Reaching the target needs either option 6 (a product decision) or a change of the execution model (4). Until then the remaining work is steady, measured trimming of the host leaves listed in the profile note.

## Option 2 measured (same day): the scorer variants spend little on the basis they share

Family ledger of the 100000 x 100 run: the 12 `orth_mi_*` variants take 10.85 s together; `univariate_basis` (the basis generation they all call) takes 1.66 s over exactly 12 calls (one per variant). A shared resident basis would remove at most the 11 repeat calls, about 1.5 s of 10.85 s, and the stages overlap, so under that on the wall. The other ~9 s is the scoring each variant does on its own (CMIM, JMIM, KSG, copula, dCor, HSIC, meta, ...), which differs by design. Not built: the saving does not justify a shared-basis object across 12 modules.

## Addendum (2026-11-09, later): the concurrent windows are 26 % of the wall

Measured with `stage_classify.py` on the default fit (69.2 s): the main thread is inside `run_stages_concurrently` for 17.9 s; the remaining 51 s are serial main-thread stages (ledger wall seconds: `fe_step` 11.8, `screen_predictors` 7.8, `fdr_gain_floor` 4.8, `assign_support_tail` 4.2, `discretize_inputs` + `categorize_dataset` 4.3, `honest_accept_block` 2.8, `retain_pure_forms` 2.5, `cross_stage_dedup` 1.8, `operand_pool` 1.7, `accuracy_gate` 1.6, and smaller ones). The options above address the windows; the estimate "a few seconds from options 1 and 2" holds for the windows only.

Which serial stages could run beside something else, from what each reads: `discretize_inputs` / `categorize_dataset` (4.3 s) read the raw frame and the target only, so they could overlap the first cascade window if their outputs (the code matrix) were not needed by the cascades' own gates, which they are for the family gates that bin; `fdr_gain_floor`, `assign_support_tail`, `honest_accept_block`, `retain_pure_forms` and `cross_stage_dedup` read the engineered pool that the FE cascades produce, so they follow them and cannot be hoisted. The stage that is in the right place and long is `screen_predictors`: its `confirm_one_predictor` (2.5 s) already has a prefetch (`_ConfirmPrefetch`), its honest null (4.2 s) runs on the device through `honest_null_columns_device`, whose acceptance kernel is now 1.6 times faster (`profiles/_honest_null_accept_bits.md`). A reordering that would pay is therefore small (discretisation against the first window, at most about 4 s, and only if the gates that bin can take the cascade-side codes), and needs the same old-versus-new selection comparison as the early_a overlap.

## Hardware headroom, measured (2026-11-09, evening): what extra parallelism can and cannot give

Utilisation of one default fit at 100000 x 100 (`util_timeline.py`, 1 s ticks, pynvml): mean CPU 26-27 % of 8 logical CPUs, mean GPU 24-28 %. The cascade phase (0-46 s) runs at 12-59 % CPU and 5-64 % GPU; the serial tail (46-77 s, 40 % of the wall) runs at 12-13 % CPU (about one core) and 1-5 % GPU, with short GPU bursts of 50-80 % where the honest null runs.

What the idle time is not: (1) the GIL: a probe thread waking every millisecond during the fit is on time within 4 ms for 97 % of wake-ups (median lateness 0.85 ms, p99 9.4 ms); (2) numba's entry lock: the threading layer is TBB, `parallel_kernel_entry` is a no-op; (3) the edge-pool thread share: `MLFRAME_EDGE_JOBS=8` (every pool the full 8 threads) 83.9 s against 82.4 s baseline; (4) the single legacy CUDA stream shared by every stage thread: one blocking CUDA stream per stage and per stage's pool workers, in a patched run, gave 76.2 s against 72.1 s, same selection.

What the four-cascade run does: with early_a beside the other three (`MLFRAME_FE_STAGE_CONCURRENCY_EARLY_A=1`) the planner refuses at 100000 rows ("scaled peak VRAM 2.6 GB exceeds the 2.4 GB available" on the 4 GB card); with the row cap lifted (`MLFRAME_FE_STAGE_VRAM_MAX_ROWS=50000`) it runs and the stage walls are [46.9, 20.7, 27.2, 26.3] s against early_a alone 24 s and the other three 13-18 s together: early_a is slowed 1.9 times, the others 1.2-1.5 times, the cascade phase takes 47 s against 43 s in sequence, and the fit 82.6 s.

The decisive measurement: two identical default fits started together in two processes took 124.6 s and 124.8 s each, against 82.2 s for one alone (same machine state): the pair finishes two fits in 125 s, 62 s per fit, a throughput gain of 1.3 times. So the idle capacity is real but it is about 30 %, not 70 %: the 4 GB card (6 SMs, WDDM launch latency: 104472 kernel launches at 47 us and 23086 `cudaMemcpyAsync` with 16.7 s of summed wait in one fit), the 4 physical cores behind 8 logical ones and the numba parallel bursts that already take all 8 threads fill up when two fits overlap. Reading: perfect overlap of independent work inside a fit would bring about 69 s to about 53 s, which makes 50 s a limit to approach and 20-30 s unreachable by occupancy alone; the rest has to come from doing less work per fit (the host thread-second leaders: `_linear_cv_loss.cv_loss` 4.0, Fourier detection 6.1 with its launch chains, `fleuret_confidence_prange` 2.9, grouped-quantile binning 5.2, the accuracy-gate held-out probe 4.0, `mdlp_bin_edges` 1.6, `_dense_codes` 2.3).

## Rebalancing the three concurrent cascades (tried and reverted, 2026-11-09)

True stage walls (`FE stage concurrency: stage walls`, 100000 x 100, one process): early_b 13.8 s, mid_a 17.7 s, mid_b 18.3 s, window 18.3 s. Moving the oof-warp, row-statistics and wavelet families from mid_b into early_b's slot gave [18.35, 16.74, 15.4] (early_b became the longest: window 18.35 s); moving only oof-warp and row-statistics, with the row-argmax family from mid_a into mid_b's slot, gave [17.88, 17.37, 17.0] (window 17.9 s), and the fit stayed at 72.6 to 76.8 s. The walls do not add: moving 4.5 s of a family between stages moves about 3 s of wall, and the stages that receive it slow down as much as the donor speeds up, because they share the card and the cores. A rebalancing is worth at most 0.4 s here; reverted. The gate-lap numbers in the family ledger (`gate:*`) charge the stage body between two gates to the family whose gate opened last, so they overstate a family's own cost (`row_argmax` lap 2.8 s); the `fe_timed` names are the family's own.

## Why three stage threads do not triple anything: GPU API calls do not scale with threads (2026-11-09, `cupy_launch_scaling.py`)

Sequential stage scale of the fit is 68 s and the concurrent one 61-75 s: three stage threads on 8 logical CPUs buy almost nothing, and the true family walls double under concurrency (quadruplet 2.5 s sequential in cProfile, 4.8 s concurrent). Microbenchmark of the launch path alone (a 20000-element float32 vector, one kernel per call, host-side cost only, one process, GTX 1050 Ti):

| call | 1 thread | 3 threads (total) |
|---|---:|---:|
| three chained cupy operations (`cp.sqrt(y * a + b)`) | 19279 launches/s, 51.9 us per operation | 16169 (x0.84) |
| one `cp.ElementwiseKernel` | 56901/s, 17.6 us | 43610 (x0.77) |
| one `cp.RawKernel` | 38134/s, 26.2 us | 35069 (x0.92) |

Each thread on its own non-blocking stream: 1, 2, 3, 4 threads give 21.6k, 20.6k, 17.8k, 17.5k launches/s. The launch path of cupy (argument handling, kernel lookup, allocator, the driver call) holds the GIL or a driver-wide lock, so the whole process issues at most about 20k cupy operations per second whatever the thread count, and more threads make it slightly worse. A fit issues 104472 `cuLaunchKernel`, 30590 `cudaLaunchKernel`, 23086 `cudaMemcpyAsync`, 11706 `cudaMemsetAsync`, 25240 `cudaSetDevice` calls (nvprof API summary): about 200000 API calls, i.e. 6 to 10 s of launch-path time that no amount of stage concurrency can overlap, plus the wait that follows each blocking read.

Consequence for the plan: occupancy of the device and the cores rises only by issuing fewer, larger launches from the same Python (fused kernels, one launch per batch of columns, one blocking read per phase, CUDA graphs for the fixed-shape chains that repeat per column or per permutation), not by adding threads. The Fourier detector batch (a chain of about 100 launches per column replaced by one per phase), the MI-greedy candidate scoring and the xi / copula blocks are instances; the cost of a family on the device is its number of operations times 20-50 us, which is the number to cut in the rest.

## The "small" call-count leaders, measured one by one (2026-11-09)

Host cost of the entries that led the cupy call counter, measured directly (one process, synchronised):
- sort guard (`_device_sort_safe`: `_room`, `_current_pool`, the pool-room hold): a guarded `device_argsort` of a 40 x 5000 float32 block 1903 us against 1837 us for the bare `cp.argsort`, i.e. 66 us per sort; nothing to gain.
- `binned_mi_from_values_gpu` (n = 30000, K = 8): 547 us per call, most of it the kernel and the one blocking read; the thread-block choice (`pyutilz.performance.gpu_hw_launch.block_size_for`, 64 occupancy evaluations per call, 43 percent of the call under cProfile) is now cached per kernel resources, launch shape and device (pyutilz commit `34ca56f`), 536 us after.
- range guard of the code arrays: the verified range is kept per buffer (`_fe_batched_mi_guards`), so the repeated checks of the fit's target codes launch nothing.
- greedy-CMI candidate stack (`_batched_cmi_argmax`): one `cp.stack` of resident columns per round; kept.

## Launch counts after the fusions, and what CUDA graphs would give (2026-11-09, evening)

nvprof API summary of the default fit at 100000 x 100, before and after the day's launch-count work (edges round-up, cross-basis legs and products, Newton logistic, radix quantile plans, Fourier batch, range-guard cache): device launches 131435 -> 101002, `cuLaunchKernel` calls 104472 -> 76738, `cudaMemcpyAsync` 23086 -> 19374 (11.4 s summed wait). Remaining launch leaders by count: `cupy_copy__float64_float64` 5207, `cupy_take` 3937, `cupy_copy__int64_int64` 3540, `cupy_copy__float32_float32` 2356, `cupy_copy__float32_float64` 2308: dtype and layout copies and gathers (the cupy call counter puts the python callers at `binned_mi_from_values_gpu`, `_radix_select_interior_edges`, `gate_grid_mi_resident`, `_cmi_assemble`, `_build_best_existing_op_candidates_gpu_cm`, `resident_qbin_codes_block`).

CUDA graphs work on this machine (cupy 13.6, GTX 1050 Ti, WDDM): a chain of 90 elementwise cupy operations on a 20000-element vector, 5.22 ms eager and 0.27 ms replayed from a captured graph (19 times). A graph needs static buffers, no host read and no allocation inside the captured region, and a shape that repeats; the chains of the fit that satisfy this are the fixed-shape inner steps of the batched Newton (one iteration per replay, the convergence read between replays), the per-phase kernels of the Fourier batch detector and the per-round candidate scoring of the greedy CMI. None is built yet: each needs its buffers made static (the producer writes into the graph's input slots) and a bit-for-bit comparison with the eager path.

### CUDA graphs for the batched Newton step: ruled out (2026-11-10)

One Newton iteration of the usability batch (40 designs x 24000 rows x 5 columns, float32: two einsums, a batched matmul, a batched solve and the elementwise chain) is 5.50 ms eager and it is arithmetic, not launch overhead; and its capture fails in cupy 13.6 ("calling cuBLAS API during stream capture is currently unsupported": the einsum, matmul and solve go through cuBLAS). The graph lever therefore applies only to chains made of elementwise and raw kernels without cuBLAS / cuSOLVER and without host reads: the per-phase kernels of the Fourier batch detector and the candidate-building chains of the resident MI (`_build_best_existing_op_candidates_gpu_cm`, `gate_grid_mi_resident`) are the ones left to examine.

## hybrid_variants: group 2 was a serial tail behind a long group 1 (2026-11-10)

Lane walls of the `hybrid_variants` window (`main_timeline.py`, 100000 x 100): group 3 (copula, dCor, HSIC, JMIM, TC) done at 11.5 s, group 4 (cmim, auto-scorer, ensemble, meta) at 18.9 s, group 1 (triplet 6.1, quadruplet 3.1, adaptive arity 1.7, adaptive degree 1.2, conditional routing 1.2 s) at 20.2 s, then group 2 (bootstrap, cluster basis, diff basis, KSG, three-gate) on the main thread until 24.1 s: a window of 19.2 s with group 3's lane idle for 8.7 s and group 2 waiting for group 1 only because the three-gate variant reads the hybrid columns appended before it. The last three families of group 1 take their sources from the numeric columns that are not hybrid_orth outputs, so they do not need the triplet / quadruplet columns; they now run in group 3's slot after group 3 (`_hybrid_orth_family_variants_group1_tail`, the head keeps triplet and quadruplet) and the fold keeps the sequential order head, tail, group 2, group 3, group 4. Result: window 19.2 -> 15.3 s, `cascade_early_a` 25.5 -> 20.5 s, the same seven selected columns, the five concurrent-versus-sequential equivalence tests pass. The head of group 1 (triplet 6.1 + quadruplet 3.1 s) is now the longest lane with group 4 (cmim, auto-scorer, ensemble, meta: 13.2 s) close behind.

### Moving MI-greedy and greedy-CMI into group 3's slot: tried and reverted (2026-11-10)

After the group 1 split the `hybrid_variants` window is 15.3 s with group 3's lane free after 11.5 s (then busy with the group 1 tail until about 15.6 s). The MI-greedy (0.9 s) and greedy-CMI (1.5 s alone) families read only the step input, so they were run in that slot after the tail (state and recipes folded back, frames merged at the places they used to run; 49 tests of the variant groups and greedy families passed). Measured: they started at 16.7 s, greedy-CMI took 5.1 s under the concurrent load (3.4 times its solo wall), the window grew to 21.1 s and `cascade_early_a` to 24.4 s (from 15.3 and 20.5 s). Reverted. A family that is short alone is not free to move into a lane that is idle on paper: the lanes share the card and the launch path, so the window is set by the total work that overlaps as much as by the longest chain.

## Cascade-window lane breakdown and OOF-warp fold cache (2026-11-10)

Clean 100k x 100 default fit, window 24.4-43.1 s. `early_b` 14.4 s (binned_numeric_agg 7.4, grouped_quantile 2.3); `mid_a` 17.8 s (composite_group_agg 1.6, pairwise_modular 2.0, row_argmax 3.1, conditional_gate 3.1); `mid_b` 18.7 s (group_distance gate 3.9 with the family itself 0.9, oof_warp gate 3.1, row_stat 1.4, small families under 1 s each). `mid_b` is the longest lane but is a sum of 1-4 s pieces with no dominant one; moving whole families between lanes was measured neutral or worse earlier (see above).

`fit_oof_warp1d` at n=100000 costs 15.1 ms per column: fold permutation 2.5, quantile edges 3.2, bin codes 2.6, rest in the njit core. The fold ids depend only on (n, n_folds, seed), so they are now drawn once per key (`_fold_ids`, 4-entry cache, identical values): about 0.25 s per fit.

`binned_numeric_agg` (7.4 s) has no single hot spot (see `profiles/fe_binned_numeric_agg_cpu_profile.md`); the open lever is the duplicate candidate binning between the redundancy gate and `_keep_linearly_usable`, as a bounded per-pair cache.

## Fit-wide basis-leg cache and agent proposals (2026-11-10)

Pair-cross, triplet, quadruplet and adaptive-arity families each rebuilt the same `(column, basis, degree)` legs in their own `build_leg_product_matrix_gpu` call. Legs are now kept fit-wide (`shared_leg_get/put` in `_fe_resident_operands.py`, key = operand content signature + routed basis + degree + robust-axis switch, 192 MB cap, cleared with the resident operands). A single clean default fit 100k x 100 measured 64.8 s (was 73-77 s over the same session; same seven selected columns), together with the `_BytesLRU` and candidate-chunk changes; one run, so treat the split between the three changes as unmeasured.

Open proposals from the three read-only reviews (estimates, not measurements): shared raw-MI baseline of the cross-basis calls (2-4 s); per-column sort bundle shared by copula/xi/rank families (3-5 s, uncertain); MDLP edge warm-up on a thread during early_a (1.5-2.5 s of `discretize_inputs`); dedup scoring overlapped with the accuracy gate (1.2-1.5 s); content-keyed honest-null cache started early (2-3 s, medium risk); grouped-quantile / group-distance per-group Python loops to numba (3-4.5 s combined).

### MDLP edge warm-up not attempted (2026-11-10)

`discretize_inputs` bins through `categorize_dataset` with the fit's nbins strategy (default `optimal_joint`, with plan-cache and null-binner objects in its kwargs, NaN strategy and median fill applied first). A warm-up thread at fit entry would have to reproduce that exact call and key on every one of those inputs; a mismatch silently changes bin edges. The estimated 1.5-2.5 s does not justify that risk without a dedicated equality test of the whole discretisation, so it stays a proposal. The raw-MI cache (`_raw_baseline_mi`) measured 66.5 s against 64.8 s before it in single runs, i.e. no visible gain inside the 4 s noise; kept because it removes repeated scoring with identical results. Latest single default fit: 67.7 s.

### Shared per-column sort bundle: measured, not worth building (2026-11-10)

Counting `device_argsort` over a default 100k x 100 fit (`D:/Temp/sort_count.py`): 399 calls, 77.7 million sorted elements in total (about 780 column-sorts of 100k rows; at roughly 1 ms per 100k-row device sort that is about 1 s of device time spread over the whole fit). Callers: rank-bin plug-in MI 195, KSG-CD 132, joint renumbering 63, group stats 4, null kernels 5. A fit-wide sort cache would save a fraction of that second, so the proposal (estimated 3-5 s by code reading) is dropped.

### Dedup MI prefetch beside the accuracy gate: no gain, reverted (2026-11-10)

An `EngineeredMiPrefetch` thread scored the appended engineered columns while `_gate_engineered_accuracy` ran, and the dedup took the scores from it (exact equality with the direct call was tested). `stage:cross_stage_dedup` stayed at 1.69 s (1.8 s before) and `stage:accuracy_gate` at 1.5 s; the fit wall of 64.5 s is inside the 65-72 s noise. The dedup's time is therefore in the rank-correlation scan (`scan_engineered_duplicates`), not in the MI scores; any further work there targets the scan.

### Dedup rank scan measured standalone (2026-11-10)

`D:/Temp/dedup_bench.py`: `scan_engineered_duplicates` over 160 synthetic float32 columns of 100k rows, warm, 0.89-0.94 s: `average_ranks` batch kernel 0.37 s, one-vs-many correlations 0.21 s (159 calls), row moments 0.06 s, the rest Python. The in-fit 1.7 s of `stage:cross_stage_dedup` is this plus the MI scores under CPU/GPU contention. A device rank kernel could save at most the 0.37 s of ranking minus the transfers; not built.

### Cupy call volume of a whole fit (2026-11-10)

`D:/Temp/cupy_op_count.py 100000 100`: 107398 cupy c_calls in an 80.6 s profiled fit (about 1.3k/s; the profiler hook inflates the wall). Leaders: `binned_mi_from_values_gpu` 8038, `_candidate_matrix` 5136, `_batched_cmi_argmax` 4649, `_assert_codes_in_range` 3744 (mostly attribute reads on cached ranges), `_newton` 3240, `rank_bin_codes_batch_gpu_resident` 3137. The per-process launch ceiling of about 20k ops/s is therefore not what limits the fit: the card idles because of dependency chains and host stages between small kernels, not because launches are saturated. Merging kernels across families would cut the call count by at most a fraction of an already low rate; the lever is overlap and removing host stages, as in the group-1 split.

### Utilisation timeline after the day's changes (2026-11-10)

`D:/Temp/util_timeline.py`, one default fit 100k x 100: WALL 61.6 s, mean CPU 30 %, mean GPU 23 % (1 s samples; CPU 12-13 % at t = 1, 19 and 43 s, GPU 0 % at t = 1, 7, 16, 22, 37, 40). The card is still idle most of the time: the serial main thread and the cascade chain keep both CPU and GPU under half used, which matches the call-rate measurement (launches are not saturated). Any further gain needs more independent work running at once (a different decomposition of the cascades), not faster individual stages.

### Main-thread timeline of one default fit, 67.7 s (2026-11-10, `D:/Temp/main_timeline.py`)

`cascade_early_a` 0.1-18.8 s (univariate/extra 2.8; hybrid_variants 3.2-17.2 of which the hinge-gated block 10.7; bootstrap 0.9, three_gate 1.6, ksg 0.7; greedy_cmi 1.5), cascade window 18.8-37.4 s (18.6), then the serial tail 37.4-66.1 s (28.7): accuracy_gate 1.5, cross_stage_dedup 1.8, discretize_inputs 2.7 (categorize 1.9, honest_spec 0.6), screen_predictors 5.6 (fdr_gain_floor 3.7), fe_step 11.5 (operand_pool 1.8, pair_mis 0.9, score_prospective_pairs 2.1, check_prospective_pairs 5.1 with 11 sequential `score_one_pair` calls of 0.5-1.1 s each, materialise 1.6), second screen 1.3, assign_support_tail 3.0 (support_fe_max_steps_pass 2.1, retain_pure_forms 1.4), compaction 0.6. The pair loop threads one mutable cross-pair chunk state through `_score_one_pair` (`_pairs_core_parts.py`), so its pairs are not independent; `fe_step` (40 % of the tail) is where a dependency-aware redesign would have to start. Everything else in the tail is 1.5-3.7 s per stage.

### What a parallel `check_prospective_pairs` would have to change (2026-11-10)

The pair loop (`_feature_engineering_pairs/_pairs_core.py`, around the `_score_one_pair` call) shares, across pairs: the chunk-pipeline state (`st._chunk_state`, `_chunk_buffer`, `_pair_to_chunk`, a double buffer with one prefetch thread), one RNG stream for the external validation (`st._rng_extval`), the running `_sweep_best_mi` that `_check_prospective_fe_pairs_part9` folds after every pair, and the rejection records. Pairs can therefore run concurrently only after (1) the external-validation draws become a pure function of (seed, pair), the way the xi tie-break was moved to a counter hash, (2) the best-so-far MI that later pairs read is replaced by a fold done in pair order after the parallel scoring, and (3) each worker gets its own chunk buffer. Selection must be pinned by a test that runs both orders on the same prospective pairs. Expected saving: up to about 3 s of the 5.1 s stage if three pairs run at once; not started.

### Pair loop: kernels are already parallel at n_jobs=1; cost is spread (2026-11-10)

At `n_jobs <= 1` `_mrmr_fe_step` passes `serial_main_thread=True`, so the materialise / searchsorted kernels already use their `parallel=True` twins (`_fe_use_parallel_kernels`; the `mrmr_fe_kernel_parallelism` cache entry is stale on this machine and the fallback picks "parallel" from 256 columns, which every pair chunk exceeds). The 11 `score_one_pair` calls of the 100k fit take 0.10-0.55 s each except two of 1.1 s (the pairs that accept and run emit / external validation), plus 0.11-0.15 s of `pair_family_gate` per accepted pair: 5.1 s for the loop with no dominant item. Making the pairs concurrent stays the only structural lever there (see the previous section).

### Kernel-tuning cache on this machine (2026-11-10)

`mrmr_fe_kernel_parallelism` was stale (code_version changed) and three other kernels were cold (`batch_aucs`, `discretize_2d_array`, `knn_hnsw_crossover`), so fits ran on built-in fallbacks and printed the cold-cache notice. After `tune_spec(..., force=True)` for the first and `mlframe.system.kernel_tuning_cache` `ensure` for the rest (about 15 minutes; the `mlframe-tune-kernels` script is not on PATH here), two default fits gave WALL 61.2 and 63.4 s, `fe_step` 9.8-9.9 s (11.5-12.5 s before) and `check_prospective_pairs` 4.2-4.3 s (5.1-5.8 s before), the cold-cache notice gone, same seven selected columns. The cache is per host and outside the repository, so this is a machine setup step, not a code change: timings taken on a host with a cold or stale cache run about 2 s slower on the pair loop.

### Edge computation has no repeated work (2026-11-10, `D:/Temp/edge_count.py`)

One default 100k x 100 fit calls `compute_col_edges` 184 times on 183 distinct (column, labels, method) contents, 8.6 thread-seconds in total, 0.0 s spent on a repeat. An in-memory content-keyed edge memo therefore gains nothing; the only way to take this off the serial tail is to compute the raw columns' edges earlier than `discretize_inputs`, which needs the exact call parameters of that stage (see the earlier section on why this was not attempted).

### Hybrid-variant lanes are balanced (2026-11-10, same timeline)

Lane ends in the 100k fit: 884-thread lane (copula, hsic, jmim, tc, adaptive_arity, adaptive_degree_basis, conditional_basis_routing) 3.2-13.3 s, 952-thread lane (triplet, quadruplet) 3.2-10.8 s, 420-thread lane (cmim, auto_scorer, ensemble, meta, cmim again) 3.3-13.6 s; then the serial block bootstrap 0.9 + three_gate 1.6 + ksg 0.7 (13.9-17.2 s) and greedy_cmi / mi_greedy 1.5 (17.3-18.8 s). The short lane has 2.8 s of slack but the two long lanes differ by 0.3 s, so moving one family (meta, 2.2 s) into the short lane would shorten the critical lane by 0.3 s at most. Earlier attempts to move the serial block into a free slot (mi_greedy / greedy_cmi) measured neutral or worse. The lane layout is at its floor; what remains is the 3.2 s serial block, which reads the same step input and would need the same device-slot and private-estimator handling as the lanes.

### Bootstrap and KSG moved beside group 1 (2026-11-10)

The serial block after the lanes (bootstrap 0.9 + three_gate 1.6 + ksg 0.7 s) held two stages that read only the step input: bootstrap and KSG take the raw numeric columns as sources like every other variant; only three_gate conditions on the hybrid columns appended before it. They now run in the group-1 head's device slot (the shortest lane, 2.8 s of slack) on private estimator copies, and are folded in the sequential order: group 2 `phase="pre"` (diff_basis, cluster_basis), the bootstrap result, group 2 `phase="post"` (three_gate, with the bootstrap columns in its support set as before), the KSG result; a name an earlier owner holds stays with it. Checks: `test_concurrent_groups_give_the_sequential_fit` passes; a 30000 x 40 fit with the concurrency switch on equals the one with `MLFRAME_FE_HYBRID_VARIANT_CONCURRENCY=0` in selected columns and the hybrid roster (`D:/Temp/equiv_hv.py`); the 100k x 100 default fit selects the same seven columns. `cascade_early_a` 20.8 s -> 17.2 s, `hybrid_variants` 15.5 s -> 12.6 s (WALL 63.1 s in that run, inside the usual noise).

### Timeline after the bootstrap / KSG move (2026-11-10, WALL 62.8 s)

`cascade_early_a` 0.1-17.1 s: part2 2.8 s, hybrid_variants 3.1-15.6 s (lane ends: 884-thread lane 13.3 s, 420-thread lane 13.7 s, group-1 head with bootstrap and KSG 12.7 s; then three_gate 14.0-15.6 s on the main thread), greedy_cmi 15.7-17.1 s. The cascade window is 18.2-35.5 s (mid_b 17.3 s, mid_a 13.8 s, early_b 7.4 s with binned_numeric_agg 5.4 s). The three lanes now end within 1 s of each other, so the remaining early_a critical path is the longest lane (about 10.5 s of sequential families of 1-3 s each) plus the 1.6 s three_gate and 1.4 s greedy_cmi that need the lanes' columns.

### orth_mi_auto_scorer profile (2026-11-10, `D:/Temp/prof_auto.py`, sequential fit under cProfile)

`hybrid_orth_mi_auto_scorer_fe` 1.91 s alone (3.2 s inside the contended lane): `_bootstrap_lcb_tables` 1.82 s, of which `_batched_bootstrap_values` 1.48 s: 1.07 s is waiting on the resample futures (worker-thread work cProfile does not see) and 0.39 s is the main thread's own batched MI call; `one_resample` 0.39 s. The family is already batched over resamples and pooled, with no host loop left on the calling thread; the wait is the workers' compute, which needs a per-thread profile (not available here) before anything can be changed. No code change.

### Auto-scorer bootstrap pool and the extval hash memo (2026-11-10)

`_batched_bootstrap_values` (210 columns x 5 resamples, `D:/Temp/scorer_time2.py`): 4.2 thread-seconds of scorer work inside 2.2 s of wall; KSG 1.48 s is the longest and runs one thread at a time (numba parallel kernel guard), copula 0.75, xi 0.73, hsic 0.39, tail 0.34, plug-in 0.26, dCor 0.22. Two serial stretches were removable: resample 0 ran alone on the calling thread before the pool started (needed only to compile the njit kernels once per process), and the guarded KSG came last in each resample. Now every resample goes to the pool once the kernels are warm and the guarded KSG runs first in each resample: batch wall 2.20 s to 1.87 s (including the cold first call). Selection unchanged (46 auto-scorer tests pass).

The extval step of the pair loop called `resident_operand` 2544 times per fit over 100+ external-factor columns; the 64-entry id memo of content hashes thrashed, so each call re-hashed a 100k column (`_content_hash_memoized` 0.84 s -> 0.17 s, `_candidate_matrix` 0.94 s -> 0.19 s, `gpu_materialise_extval_codes_host` 1.32 s -> 0.63 s in a sequential cProfile fit). The memo now follows the live columns (weakref callback removes the entry with its array).

### KSG batch: already on the device (2026-11-10, `D:/Temp/ksg_bench.py`)

`_mi_cd_columns_auto` takes the device kernel at every size here (`use_device` True): 210 columns, host / device seconds: n = 10000 0.052 / 0.043, 20000 0.100 / 0.083, 50000 0.353 / 0.208, 100000 1.016 / 0.438, max difference 1e-11. The `parallel_kernel_entry` guard around the call is only needed for the host numba kernel; dropping it for the device path would let the KSG calls of different resamples queue on the GPU while the host-parallel plug-in kernel runs, but a device fault falls back to the host kernel inside the same call, which would then enter a numba parallel region unguarded. Not changed: the saving is a fraction of the 0.1-0.15 s per resample.

### Gradient-interaction seeder: Gram-matrix energy (2026-11-10)

`build_fe_operand_pool` (1.8 s of `fe_step`) is `propose_gradient_interaction_pairs`: 13 calls of `_rff_analytic_mixed_partial_energy` (the fit plus the 12 permutation nulls) took 1.0 s, n = 2000 rows, D = 400 features, up to 4950 pairs. The energy of a pair is `coef' G coef` with `G = Mcos' Mcos / n`, so `_rff_analytic_mixed_partial_energy_gram` forms G once (D x D) and scores a chunk of pairs with D x D x k instead of n x D x k plus the n x k square-and-sum: p = 100: 71 ms -> 32 ms per call (`D:/Temp/grad_bench.py`), maximum relative difference 3e-6, same top pairs; 107 seeder tests pass. The batched and per-pair kernels stay (keep-all-kernels rule); the Gram kernel is the routed default. Next inside the same function: the additive-residual fit 0.31 s and the 13 RFF ridge fits 0.19 s (sklearn, host).
