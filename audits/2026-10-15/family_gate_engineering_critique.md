# Critique: cost and hooks of the in-fit sequential family gate

Source: read-only critique agent, 2026-10-15. Measurements are marked "measured" and were taken on this box (8 cores, GTX 1050 Ti 4 GB, 16 GB RAM); estimates are marked "estimate".

## Findings

### F01 - Noise fit selected a feature in the profiling run
- **Disposition**: COMPLETED (traced: the raw screen floor at `screen_fdr_null_quantile = 0.975`, see F14)
- Profiling script `D:/Temp/mypyrepro/critic4/prof_noise.py` (n=2000, 30 Gaussian features, binary target, default MRMR, seed 0) selected `f24`. The rule "select nothing on noise" was not met in that run. The cause was not traced.
- Must be reproduced and traced before the strict default can be claimed for every fit.

### F02 - Family gate is a small share of fit time on that run
- **Disposition**: OPEN
- Measured: 25.2 s total fit; `family_null_max_njit` 0.47-0.48 s over 4 calls (about 2%). Top items: llvmlite ffi 3.77 s, lock acquire 1.95 s, `hermite_fe/__init__.py:75` 1.04 s, io.open 0.99 s, `_usability_njit_pool.py:447` 0.65 s.
- The 2.2 s family share reported earlier was on a different fit (larger K); cost depends on the data.

### F03 - Per-permutation kernel cost
- **Disposition**: OPEN
- Measured with K=1734 quantile columns, 10 bins: 2.27 ms per permutation at n=2000 (22.7 s per 1e4 permutations); 18.7 ms at n=20000 (187 s per 1e4). Shuffled-label null exceedance fraction 0.40 (n=2000) and 0.44 (n=20000).

### F04 - Resolution of the quantile threshold at 999 permutations
- **Disposition**: COMPLETED (quantile rule replaced by exact p-value; see the statistics file, F01)
- Quantile index with `method="higher"` at alpha <= 1e-3 selects the maximum of the null with B=999. Replaced by `exact_pvalue_pass` in `_family_maxt_gate.py`.

### F05 - Sequential stopping cost estimate
- **Disposition**: OPEN
- Estimate: Besag-Clifford style stopping after h=10 exceedances takes about h/q, roughly 25 permutations for noise families (q around 0.4 measured), about 57 ms at n=2000 versus 2.3 s for 999. Plans: 0.7 s and 4.5 s per noise/near-threshold family at n=2000; 5.6 s and 37 s at n=20000 (estimate).
- Implementation pending (statistics file F03).

### F06 - Reaching alpha = 1e-4 needs 1e4 permutations or an extreme-value approximation
- **Disposition**: OPEN
- Exact certification at 1e-4 costs 23 s per family at n=2000 and 187 s at n=20000 (estimate from F03), only for passing families. Alternative: GPD/EVT fit to the null maxima, labelled as an approximation.

### F07 - Narrow the family to configurations above the table floor
- **Disposition**: OPEN
- The table cascade currently checks only the best configuration (`_pairs_emit.py` cascade around the family call). Filtering K to configurations whose observed MI is above the floor before the family test would shrink the kernel cost (estimate: proportional to K).

### F08 - Shared permutations across pairs of the same n
- **Disposition**: REJECTED (not worth it)
- Savings only on permutation generation, which is a small fraction of the 2.27 ms per permutation (estimate). The kernel dominates.

### F09 - Code reuse
- **Disposition**: COMPLETED (already in place)
- Discretised codes are reused from the pair search (`_pairs_score_steps.py`, codes stored per configuration; `_pairs_chunks.py` passes the chunk codes).

### F10 - cupy path for the family kernel
- **Disposition**: OPEN
- Not measured. Estimate 3-10x, but the GTX 1050 Ti has 4 GB and contention with other jobs; not a priority until the CPU design is fixed.

### F11 - Bit-identity across thread counts
- **Disposition**: COMPLETED (verified by construction and a check)
- Permutations are drawn serially from one rng; each parallel task writes its own element; the maximum does not depend on order. Checked: block kernel vs per-permutation loop gives maximum absolute difference 0.0.

### F12 - Strict default hook
- **Disposition**: OPEN
- The alpha must become a constructor parameter in `filters/mrmr/_mrmr_class.py` (`__init__` near line 364), stored on the instance and passed to the pair gate in `_pairs_core.py` (currently `0.01 / len(prospective_pairs)` near line 903). The same parameter must be generated into `training/fs_params/mrmr.py` (generator `fs_params/_spec.py`, `_generate.py`), with the sync test `tests/training/test_fs_params_in_sync.py`.
- The hook covers only the pair path. The raw screen (`_screen_predictors_prescreen.py`) and the fallback need the same parameter; that is where F01 was observed.

### F13 - Claims in the brief that did not match the repository
- **Disposition**: COMPLETED (documented)
- The brief named `_pairs_chunks.py` as part of the pair gate path; the gate itself is in `_pairs_emit.py`. Chunk codes do pass through `_pairs_chunks.py` (verified). Line references in the brief were checked and corrected where they differed.

## Recommendation (from the critique)
1. Reproduce F01 and trace which gate admitted `f24`.
2. Sequential stopping (F05) plus exact p-value (F04, done).
3. Narrow the family to configurations above the floor (F07).
4. Expose alpha as a parameter with the strict default (F12).
5. Keep the EVT approximation (F06) as an option, labelled approximate.

### F14 - Raw screen floor at the 0.975 level admits noise (root cause of F01)
- **Disposition**: OPEN
- Measured on the reproduction data (n=2000, 30 features, seed 0): the first raw confirmation step accepts `f24` with gain 0.00539; the maxT screen floor (`compute_fdr_gain_floor`, quantile 0.975, 999 permutations, pool of 30 features) is 0.003788, so the gate passes (`_screen_predictors_prescreen.py`, `_screen_predictors.py:285` default `screen_fdr_null_quantile = 0.975`).
- The exact rule with the same level gives the same threshold: 0.975 with B=999 selects the 25th largest null maximum, which corresponds to alpha = 0.025 family-wise. So the screen is calibrated at 2.5%, not at a strict level. This is the level decision, not a rule bug.
- Consequence: a pure-noise fit selects something with probability about 2.5% from the screen alone, which matches the three noise failures in the suite.
- Fix direction: the strict level (alpha around 1e-4) has to be applied to the screen floor too, which requires either a sequential order-statistic threshold (F05) or an analytic tail. Until then the screen is documented as a 2.5% family-wise gate.

### F15 - Strict screen level trades off against the dense weak-signal test
- **Disposition**: COMPLETED (owner decision, option B: screen default stays at quantile 0.975 with 999 permutations; the strict level remains a constructor parameter)
- Measured on the pure-noise suite and `test_biz_val_filters_screen_heavytail_floor.py` (n=2000 noise; weak dense test seeds 1, 7, 42):
  - screen quantile 0.975, B=999: noise 15/18 (3 fail), weak test passes on all 3 seeds (earlier run, measured);
  - screen quantile 0.9999, B=9999 (exact alpha 1e-4): noise 16/18 (binary/uniform and binary/student_t2 fail), weak test FAILS on seed 1;
  - screen quantile 0.999, B=9999 (alpha 1e-3): noise 16/18 (same two binary cases fail), weak test FAILS on seed 1.
- Screen floor cost at 10^4 permutations for a 30-feature pool, n=2000: about 0.7 s per call (measured).
- Interpretation: the weak dense signal has per-feature p-values around 1e-3, so a family-wise floor at 1e-3 or 1e-4 over the same pool cannot admit it. Power against weak signals is lost by the level, not by the implementation.
- Two defensible directions: (A) strict default (1e-4) with the weak dense test reframed to an explicit lower-power level; (B) a compromise level (e.g. 0.025 as before) and accept a documented noise-selection rate of about 2.5% per fit. The choice is a product decision, not a code fix.

### F16 - False-selection rate is far above the designed level for a binary target
- **Disposition**: OPEN
- Measured (20 seeds per case, n=2000, 8 features, current defaults: screen 0.975 with 999 permutations, pairs Bonferroni, fallback exact test at 1%): normal/binary 5/20, normal/continuous 1/20, uniform/binary 6/20 (uniform/continuous was still running at the time of writing).
- Binary target: about 25-30% of fits select a feature on pure noise, against the designed 2.5% for the screen. Continuous target: about 5%. So a leak exists outside the screen level for binary targets.
- Consequence: rewriting the noise test to a rate contract now would hide a real defect. The contract change (F16 decision) waits until the rate is explained.
- Next: trace which gate admits the features on the binary-target false cases (`D:/Temp/mypyrepro/prof/binary_fp_trace.py`).

### F17 - Screen gain and its null floor are on different statistics (binary target)
- **Disposition**: OPEN (root cause of the binary-target false selections, F16)
- Measured on seed 0 of the pure-noise binary case (normal features, n=2000, 8 columns): the library's gain for the selected column `c6` is 0.005668, the screen floor is 0.002864, so the gate passes (`_screen_predictors_gate.py`, `compute_selection_gate`, trace `D:/Temp/mypyrepro/prof/binary_floor_trace.py`).
- Independent check on the same data (plug-in MI, 10 quantile bins, 999 label permutations, `_family_maxt_gate.family_maxt_threshold`): `c6` MI = 0.00365, null 97.5% of the family maximum = 0.00624, and 58.6% of null maxima are at or above the observed value. The column is pure noise.
- So the gain fed to the gate is about 0.0020 higher than the plug-in MI of the same column, roughly the size of the Miller-Madow term for this table, and the floor is calibrated on a different quantity. The gate therefore compares unlike statistics.
- Fix direction: the gate must compare the same statistic as the null (plug-in MI on the screen's codes, with the same correction on both sides). Verify on this seed first, then the 20-seed rate again.
- Consequence for F16: the rate contract change must wait until F17 is fixed; otherwise the test would accept a 25% false-selection rate.

### F17 (correction) - the gain differs from the plug-in MI of the same binned column
- **Disposition**: OPEN
- The earlier independent check used 10 bins for `c6`; the screen uses 6 bins for that column (adaptive nbins, measured with `binary_floor_trace.py`). Redone with 6 bins: plug-in MI 0.00282, null 97.5% = 0.00436, 32.6% of null maxima at or above the observed value. The column is noise.
- The gain the gate compares is 0.00567 for the same column, about twice its plug-in MI on the same bins; the floor (0.00286) is on the plug-in scale. So the gate compares gain against a null on a different statistic. The inflation is in the gain path, not in the null.
- Next: trace how `best_gain` is computed for order-1 candidates (mi_direct, null-mean subtraction, debiasing branch in `filters/evaluation.py` around the relevance screen) on this column, and make the gate compare the same quantity the null is built on.

### F18 - Supervised MDLP binning is fitted on y, and the permutation null never refits it (root cause of F16/F17)
- **Disposition**: COMPLETED (root cause addressed: the honest null refits the supervised binning on every permuted target for 'mdlp' and 'mdlp_validated'; pure-noise suite 90/90 on all strategies, 2026-10-15)
- Measured: on the same pure-noise column (`c6`, n=2000), MDLP edges fitted on the real target give a plug-in MI of 0.00282 with the real target; MDLP refitted on 20 permuted targets gives MI with the real target of mean 0.00002, 97.5% 0.00025, max 0.00047. The real-target fit is about 6 times the largest refit value: in-sample selection bias.
- The screen's null (permuting y with fixed codes, `family_null_max` / `compute_fdr_gain_floor`) keeps the y-fitted bins, so it cannot see this bias. The gain and the floor are therefore both computed on codes that were chosen using the same y.
- Also measured: the screen gain is the raw plug-in MI when the 2-permutation significance p-value is below alpha (`filters/evaluation.py` around the relevance debiasing, p=0.030 measured), and the floor is Miller-Madow corrected. Two statistics, and a 2-permutation p-value has a resolution of 1/3.
- Fix options: (a) quantile bins for the significance gates, MDLP only for representation; (b) cross-fit MDLP (fit edges on one half, measure on the other); (c) refit MDLP inside every permutation (exact, expensive). Each changes the default accuracy/validity trade-off.

### F19 - Honest MDLP null and its cost (follow-up of F18)
- **Disposition**: COMPLETED (owner decision A accepted; cost measured: honest floor about 1 s per mdlp fit after the numba chain, cache per fit)
- Correction of the earlier check in F18: the MDLP null must refit the bins on each permuted target and evaluate on that same permuted target. Measured for `c6` (n=2000, 99 refits): observed in-sample MI 0.00282; null mean 0.00006, 97.5% 0.00000, max 0.00306; fraction of null values at or above the observed value 0.010.
- Cost of the honest refit null: one MDLP fit per column is about 8.7 ms (n=2000, measured). For a pool of 8 columns and 999 permutations: about 70 s per fit (estimate from the measured per-call time). With 199 permutations and early stopping: about 14 s per fit.
- Quantile bins for the significance decisions (`nbins_strategy=None`) measured 0/20 false selections on normal/binary (see the rate measurement), against 5/20 with MDLP.
- Options: (A) honest refit null for supervised strategies (correct, slow, about 14 s per fit); (B) significance decisions on quantile codes of the same columns, with MDLP used only for the representation of selected features (fast, needs a parallel factors matrix in the screen).

### F20 - Honest MDLP null implemented (decision A: accepted cost)
- **Disposition**: COMPLETED (module `filters/_honest_mdlp_null.py` and `filters/_mdlp_oos_honest_njit.py` integrated into the screen floor and the fallback)
- Implemented: permutation null with cut points refitted for every permuted target via the production routine (`mdlp_bin_edges(fast_mode=False)`), maximum over candidate columns; permutations parallel in threads; result identical for 1 and 4 workers (measured).
- Measured on the 8-column noise pool (n=2000, binary target): observed max 0.00282; 99-permutation null 97.5% 0.00670, mean 0.00134; 24% of null maxima at or above the observed value (pure noise, not significant). Time: 99 permutations in 2.2 s with 4 threads (3.5 s with 1 thread) on a loaded machine; 999 permutations estimated about 25 s per fit.
- Open: the screen's observed statistic must be produced by the same call (its MDLP parameters are those of `_adaptive_nbins`, which gave 6 bins for `c6` where the test call above gave 2). Integration must use exactly that call for both the observed value and the null.

### F21 - Honest supervised null integrated (screen floor and fallback)
- **Disposition**: COMPLETED (integration verified: pure-noise suite 90/90 with the honest null on all five strategies; the held-out-validated cut logic matches the production recursion on 120 identical splits; the held-out null matches the Python reference in distribution, not bit for bit, since the holdout split uses numba's generator)
- Implemented: the production categorisation (`categorize_dataset` with the fit's `nbins_strategy`) is re-run on each row permutation of the raw target; the screen floor (`honest_fdr_floor`) and the fallback null (`fallback_maxt_floor(honest_codes_fn=...)`) use those codes. The closure lives on the estimator for the fit and is stripped from pickles.
- Seed 0 (normal, binary, default mdlp), measured: honest floor 0.005913 against gain 0.005668 (gate rejects); the fallback rescue then also rejects, `no_significant_features_ = True`. Before integration the same case selected `c6`.
- Cost per fit (measured on the loaded machine): honest floor took 66 s for one fit with 999 permutations and 8 columns, which is the cost accepted in this round.
- Quantile-bin reference, same protocol (20 seeds, `nbins_strategy=None`): normal/binary 0/20, uniform/binary 0/20, uniform/continuous 0/20, normal/continuous 0/20; signal recall 6/6.
