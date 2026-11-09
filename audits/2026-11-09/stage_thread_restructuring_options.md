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
