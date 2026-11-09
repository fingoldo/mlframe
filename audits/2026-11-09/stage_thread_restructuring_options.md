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
