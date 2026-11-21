# Review: categorical_fold_strategy -- alternatives, literature, benchmark and tests

Angle: comparison with established methods, benchmark design, pytest plan. Read-only review (no code touched). Literature points are cited from memory, not freshly searched; verify exact details before quoting them in docs.

## 0. Key observation about the proposal

MRMR is a FILTER whose output is a selection plus encoded columns that feed an MI estimator. This differs from the usual "encode a categorical for a model" setting:

- The fold is an input to a discrete MI estimate. Any target-aware fold computed on the same rows inflates MI. The more groups, the larger the plug-in MI bias (about (G-1)(K-1)/(2n) nats). A fold that is smoother than the raw levels reduces bias, but a target-aware fold that merges by target stat adds selection bias on top (the grouping is chosen to maximise the association).
- Hence the proposal's OOF idea (section 4, step 2) is correct in spirit, but "final mapping built on all fit rows" in the same section re-introduces in-sample leakage into the MI that MRMR then computes. MI must be computed on a fold learned from data that the MI estimate does not reuse, or against a permutation null that applies the same fold procedure (see 3.3).
- The 1-D ordering by target stat is natural only for binary and regression targets. For K>2 it is lossy (see 2.7).

## 1. Comparison table

| approach | what it is | fit for MRMR fold | verdict |
|---|---|---|---|
| frequency top-(cap-1)+other | current | cheap, target-free, no leakage | keep as default |
| rare-label grouping by min support | merge levels with count < m | target-free, very cheap, fixes weak point 3 | ADD (step 1); fold into frequency, not a separate value if `min_support=0` equals legacy |
| smoothed target encoding (m-estimate, Micci-Barreca 2001) | level -> shrunk mean | produces a CONTINUOUS column, then needs discretization; leakage unless OOF | do not add as a separate fold value; use it only as the ordering statistic inside target_stat |
| James-Stein / empirical Bayes shrinkage (Efron-Morris; Stein 1956) | shrinks level means by variance ratio | principled k = sigma2_within/tau2_between, the proposal already has it | ADD as the shrinkage inside target_stat, not a strategy |
| CatBoost ordered TS (Prokhorenkova et al. 2018) | stat for a row from rows earlier in a random permutation | removes the target leakage by construction, O(n) | optionally use for the OOF stat; gives a row-level encoding, not a level mapping, so cannot be stored as a transform recipe directly (recipe = full-data stat). Worth only as the OOF engine |
| leave-one-out TE | stat excluding row itself | known to leak: the row's own target is a deterministic function of the encoded value within a level (Kaggle "LOO leak"), allowing a perfect split for any tree/MI | DROP, never use with MI |
| WoE binning with monotone merging (optbinning, Navas-Palencia 2020; scorecard practice) | order levels by event rate, merge adjacent via solver (MIP/CP or greedy) with min bin size, monotone constraint, IV/chi2 criterion | closest to the proposal's target_stat; mature, binary and continuous targets (multiclass in optbinning via a separate class), min-support built in | ADD as the model for target_stat: ordered adjacent merging with a significance stop rule. Avoid a hard dependency: implement the greedy variant (numba) and optionally use optbinning if installed |
| ChiMerge (Kerber 1992) / CHAID (Kass 1980) | iteratively merge the pair of adjacent (or any, in CHAID) levels with the smallest chi2 until all pairs are significant | statistic-driven stop rule needs no fixed cap, handles K>2 via a KxG contingency table (no 1-D ordering needed for CHAID) | ADD as the multiclass-capable merge criterion (CHAID-style: merge any pair with smallest chi2/G-test p>alpha). Alpha should be Bonferroni-corrected by number of levels, not a fixed 0.05 |
| MDL / BIC-based merging (Fayyad-Irani style, Rissanen; information-bottleneck agglomerative merging, Slonim-Tishby) | merge levels while the description-length gain is positive; the cost of a merge is exactly delta_MI scaled by n minus the parameter penalty | The proposal's `delta_MI` merge cost (section 5) IS the agglomerative information-bottleneck step. Adding an MDL penalty (G-1)(K-1)/2*log n gives a principled stop rule and answers "auto without fixed thresholds" | ADD: this is the best answer to open question 3 and 4. Use it as the stop rule for target_stat instead of "merge to cap" |
| hierarchical clustering of levels by target profile (Hellinger/JS) | agglomerate levels by distribution of y | equals information bottleneck when the linkage is JS-weighted; fine for multiclass | covered by the IB/MDL entry; do not add separately |
| hierarchical clustering by hierarchy in the data (ZIP prefix, taxonomy, product code) | merge by known tree (prefix truncation) | needs side information; MRMR cannot know it | DROP from core; allow an optional user-supplied `level_hierarchy` hook later (low priority). The benchmark should still have hierarchical generators to show where a target-aware fold recovers coarse groups |
| hashing trick (Weinberger 2009) | level -> hash bucket | random collisions, ignores signal, destroys interpretability and fe provenance | DROP. It is "frequency-blind rare merge" with no benefit over `other` |
| embeddings / entity embeddings (Guo-Berkhahn 2016) | learned dense vectors | needs training a model, outside a filter's scope, not discrete | DROP |
| regularized/Bayesian multilevel (GLMM encoder, category_encoders) | random-effect level estimate | equivalent to James-Stein; slower | covered by the shrinkage entry |
| sklearn TargetEncoder (internal cross-fitting, sklearn>=1.3) | K-fold fit_transform + full-data transform | confirms the "OOF for fit, full for transform" convention | cite as the convention |
| level-wise statistical filter without folding (each level vs rest, binary indicator) | one-hot per level and test | answers weak point 1 differently: keep rare informative levels as indicator columns | CONSIDER as a separate FE path (`level_indicators`), orthogonal to folding; flag as an alternative to target_stat for the "rare informative level" case |

### 1.1 Recommended strategy values

- `"frequency"` (default, bit-identical codes).
- `"min_support"` (target-free). Could be just `categorical_fold_min_support` on top of `"frequency"`, with the value `"min_support"` as sugar. With `min_support=None` the legacy codes must be unchanged.
- `"target_stat"`: rename to `"target_merge"` or `"mdl"`. Mechanics: ordered (binary/regression) or CHAID-style (multiclass) merging of statistically indistinguishable levels with an MDL/Bonferroni stop. NOT "merge to cap".
- `"auto"`: not a separate algorithm. Define as: run both frequency and target_merge on an inner split, pick by held-out MI after subtracting the permutation null of the merged column. Disclose its cost (see 3.3).

Do not add: hashing, embeddings, LOO, hierarchy-from-prefix.

## 2. Specific concerns found while comparing

1. "Merge to cap" is the wrong stop. Cap=127 is an int8 storage limit, not a statistical optimum. A significance/MDL stop will usually produce far fewer groups than the cap, which also helps MI bias and the downstream cost (the MI table is G x K x |X_other|). Keep cap as a hard upper bound only.
2. Upper bound "n/50 groups" (section 6) is a fixed constant. Replace by the MDL penalty.
3. min support `max(5, n/(cap*20))`: both constants are arbitrary, and for n=10M gives support 3900 even though 127 levels with 3900 rows are all well estimated. The step-1 threshold must come from the MI estimator's noise (a level of count c contributes standard error about sqrt(K/c) to its class profile) rather than a division of n by the cap. At n=10M with levels > cap, folding by frequency is a storage issue and a very light threshold is safe.
4. The rare-informative-level problem (weak point 1) is a per-level hypothesis test. A level with 30 rows and 30 positives where the base rate is 1% is detectable (binomial tail ~ 1e-50) but a level with 30 rows and 3 positives at base rate 1% is not. Any rule must treat level significance, not frequency, as the keep criterion. Natural step-1.5 (target-aware, still cheap): keep level l out of "other" if its OOF-corrected deviation from the base rate exceeds the Bonferroni-corrected binomial/Poisson/t bound. This gives "keep rare informative levels" without ordering/DP at all, and is the minimal answer to the headline motivation.
5. NaN policy: "NaN own level if share >= threshold" is a fixed threshold. Better: NaN is always its own level (missingness is often informative, and the cost is one level); fold it only if the level fails the same significance test.
6. Leakage in the "final mapping from all fit rows". See section 0. Concrete resolution: compute the fold from rows of a fit-time split A, estimate MI on the complementary rows B (cross-fitted, averaged over folds), or accept in-sample fold but compare MI against the permutation null that replays the whole procedure (expensive, G permutations x fold cost).
7. Multiclass ordering by first principal component of class-proportion vectors is not guaranteed to keep distinguishable levels apart. Use CHAID-style any-pair merging or IB agglomeration (cost O(L^2 K) with L levels, fine for L<=500 and a priority-queue variant for larger L).
8. Regression: the proposal uses the continuous `_fe_prewarp_y_continuous_` for ANOVA while MI uses `classes_y`. Merge decisions should be consistent with the objective MI uses (discretized y); the continuous statistic is only the ordering. Mixed criteria can merge levels MI could still separate, and the reverse.
9. Counts / zero-inflated / survival / multitarget rows of the table: unlikely to be worth implementing as separate stat kinds. For MRMR, y is always discretized to `classes_y`; the multiclass machinery covers them. Recommend dropping those rows from scope to avoid untestable code paths (project policy: no padded parametric pins). Keep count targets in the benchmark only to show that the discretized multiclass path is adequate.
10. Unseen levels at transform: mapping them to "the group with the statistic closest to global" is arbitrary and can inject a spurious strong-signal bucket. Safer: a dedicated "other/unseen" bucket whose code equals the fit-time "other". If the fit had no "other", add one reserved code.
11. int8 and cap 127: the fold must reserve codes for other, NaN, unseen; document that G<=cap-2 or so.
12. Pickle: store the mapping as plain numpy arrays / dict of python scalars in the recipe; no cached closures (policy: runtime caches break pickle).

## 3. Benchmark design

### 3.1 Synthetic generators

All generators take (n, L, K_target_kind, seed). Each frame also contains M noise categoricals (L levels, independent of y) and numeric noise, so precision is measurable. Ground truth: a set of informative categorical columns.

| id | generator | what it tests |
|---|---|---|
| G1 rare informative | one column, L levels Zipf(s=1.1); designated rare level(s) of support s in {10,30,100,300} shifts P(y=1) from base to base+delta | headline weak point 1; frequency fold should fail, target-aware pass above a detectability bound |
| G2 many noisy levels | all categoricals independent of y, L in {50,127,500,5000} | false positive rate, MI inflation, permutation-null calibration |
| G3 dense weak signal | all L levels have small effects ~N(0,tau^2), tau chosen so the total MI is about 0.01-0.05 nat | MDL/shrinkage must not wipe it; n-dependence |
| G4 hierarchical | L=500 leaves under 20 parents; y depends on the parent only | a good fold recovers about 20 groups; frequency fold keeps noisy top leaves |
| G5 opposite effects in the tail | tail levels split in two halves with +delta and -delta | weak point 2: single "other" cancels the signal; a 2-group fold recovers it |
| G6 frequent-noise-vs-rare-signal | 10 frequent levels with no effect + 200 rare ones with effect | frequency fold keeps the wrong ones |
| G7 interaction | y depends on (cat A level group) x (numeric) | folding must not destroy MRMR's redundancy/conditional logic; checks interaction with grouped FE |
| G8 unseen levels | transform set contains 5-20% levels absent from fit | recipe consistency |
| G9 missingness | NaN rate 0, 5, 30%, informative and non-informative | NaN policy |
| G10 correlated categoricals | two columns, one a refinement of the other | redundancy term with folds of different granularity |

Targets: binary (base rate 50%, 5%, 0.5%), multiclass K in {3,10}, regression (Gaussian, heavy-tailed t3, heteroscedastic), count (Poisson with level-specific rate, zero-inflated). Grid for sizes: n in {2k, 10k, 100k, 1M}, L in {20, 127, 500, 5000, n/10}. Use large n for rare-event base rates (policy: rare imbalance needs large n). Everything float32 where relevant. 5 seeds, report mean and 95% CI.

### 3.2 Metrics

Selection quality:
- recall and precision of informative categoricals in the selected set; also level-level recall: share of truly informative levels that are NOT in "other" after folding (G1, G5, G6).
- false-positive rate on noise categoricals (G2) vs the n-scaled null.
- rank of the informative column in the MRMR ordering.
Fold quality:
- number of groups G, share of rows in "other", plug-in MI before and after, MI on a held-out half (generalization), null-MI under y permutation (calibration: the p-value of the real MI vs. permuted MI should be uniform under G2).
- adjusted Rand index of the fold vs the true grouping (G4, G5).
Downstream:
- holdout score (AUC / logloss / R2 / Poisson deviance) of a fixed LightGBM or CatBoost and a logistic/ridge model on the selected columns, vs (a) all columns, (b) oracle informative columns.
- stability across seeds (Jaccard of selected sets).
Cost:
- wall time of the fold step alone and of the full MRMR fit; at 100k x 100 (20 categoricals x 500 levels) and 1M rows; peak RSS and its delta; CPU%/GPU% per the project's RSS_TICK_S logger; record GPU/CPU temperature and clocks (project rule), warm cache, one runner at a time.
- overhead = (fit with strategy) / (fit with frequency).

Baselines inside the benchmark: frequency (current), min_support, target_merge (OOF), optbinning OptimalBinning (as an external reference, optional dependency), CatBoost's native cat handling (as a model-level upper reference), sklearn TargetEncoder + quantile bins, and "oracle" folding by the true groups.

Decision rule: derive acceptance bounds from the null calibration at each n (no fixed thresholds). A strategy is better when its recall gain is outside the seed CI AND its false-positive rate is within the permutation-null CI.

### 3.3 Cost control on "auto"

Permutation control multiplies the fold cost by P (number of permutations). Budget it: cap per column at the cost of one MI evaluation times P, with P chosen from the required p-value resolution (Bonferroni over the number of categoricals), and run it only for columns where L > cap or min_support triggers a fold at all (the common case, L <= cap, never pays). Report the measured overhead at 20 x 500 levels in the benchmark.

## 4. Pytest test plan

Follow project rules: Fast mode, `-n 0`, one pytest process at a time, targeted files only, `PYTHONUNBUFFERED=1 -s --durations=200 --no-cov`, output to file.

### 4.1 Unit (tests/.../discretization/test_categorical_fold_*.py)
1. Parameter validation: unknown strategy raises; min_support <= 0 raises; defaults documented.
2. `frequency` with default params returns codes IDENTICAL to the pre-change function on a fixed frame (golden comparison computed in-test by calling a frozen copy of the old function, not stored arrays).
3. L <= cap leaves the column untouched for every strategy (no "other" code is created).
4. min_support: counts below the threshold go to other; counts exactly at the threshold stay; monotone in n.
5. Deterministic: same seed, same mapping; independence of row order and of thread count.
6. Fit/transform consistency: transform on the fit data reproduces fit codes; unseen levels map to the reserved code; NaN to its code; polars and pandas (Categorical, object, string, Enum) give equal codes.
7. int8 boundaries: L = 126, 127, 128, 129, 1000; code range never exceeds cap; reserved codes do not collide.
8. Recipe pickle round-trip (no runtime-cache attributes) and clone/get_params compatibility with sklearn.
9. Merge algorithm correctness on tiny hand-built tables: two identical level profiles merge, two opposite ones stay apart; CHAID multiclass 3x3 case; IB merge cost equals the MI difference computed directly (property test, tolerance for float32).
10. Shrinkage: with n_l -> inf stat -> raw mean; with n_l = 0 -> global.
11. OOF: the mapping for fold k does not depend on rows of fold k (mutate the target of fold k rows and check the mapping built for that fold is unchanged).
12. Permutation null: on pure noise the fold yields G near 1 or MI below the null band, over many seeds (statistical test, seeds fixed, tolerance from the binomial CI of the number of failures, not a hard constant).
13. Logging: throttled log line contains column, levels before/after, share in other (caplog); slower path emits a warning (policy: log when a gate slows the run).
14. Target kinds: binary, multiclass, regression via `classes_y`; ensure the continuous y is not required (None allowed).

### 4.2 biz_value
1. G1: with `target_stat`/`target_merge`, the column with the rare informative level is selected with recall >= the frequency baseline plus a margin derived from seed-to-seed variance; with `frequency` it is missed at the same n (this pins the motivation; assert the paired difference, not absolute numbers).
2. G2: noise columns are not selected more often than the permutation-null rate.
3. G4/G5: recovered grouping ARI above the frequency baseline.
4. Multiclass and regression and Poisson: the same recall comparison (parametrized over target kind with n large enough for the base rate).
5. Redundancy: two categoricals where one is a refinement; MRMR keeps the more informative one under each strategy.
6. Interaction with grouped FE (grouped_quantile, composite_group_agg): selected set with `frequency` unchanged after enabling the parameter at its default.
7. Downstream: holdout AUC with selected features is not lower than frequency beyond the CI (non-regression of model quality).

### 4.3 Regression on default
- Full-fit golden test on a few small frames: with defaults, `selected_features_`, `support_`, and transformed output are identical before and after the change (compare to a frozen reference computed from the old function path, not a stored baseline).
- Meta test: parameter appears in the docstring and in `_mrmr_class.py` `get_params`; strategy value list in docs equals the validated set. (Meta-tests that are generic belong in py-ci-shared.)

### 4.4 cProfile / perf
- `cProfile` of the fold on 100k x 100 with 20 cats x 500 levels and on 1M rows; assert the fold's cumulative share of the fit stays below the share established by measuring `frequency`, with the threshold derived as a ratio from the measured baseline run in the same process (no fixed seconds).
- Check the per-column cost is O(n + L log L) for min_support, O(n + L*K + merges) for target_merge; the merge phase must be njit (policy: always try njit); a test that tracks that no per-row Python loops occur (call count of the njit kernel equals the number of columns).
- Memory: tracemalloc/RSS delta of the fold is bounded by O(n) int8 + O(L*K), asserted as a ratio to the input size.
- GPU: residency contract tests unchanged (codes remain int8 on host before upload; fold is host-side, once per fit, so no extra syncs).

## 5. Prioritized recommendations

1. Ship step 1 (min_support + NaN own level) first, default-neutral. Replace the arbitrary constants by an estimator-noise-derived rule.
2. Add a per-level significance keep rule (Bonferroni binomial/t/chi2 against the base profile), because it answers the headline case directly at O(n) cost and needs no ordering, DP or K-dependent statistic.
3. For target_stat, implement CHAID/IB agglomerative merging with an MDL stop (this unifies binary, multiclass and discretized-regression, and removes the "cap" as the objective). Optbinning is the reference to benchmark against, not a dependency.
4. Resolve the in-sample leakage question before any code: choose cross-fitted MI or permutation-replay and document the cost.
5. Drop from scope: LOO, hashing, embeddings, hierarchical-by-prefix, separate count/survival/multitarget stat kinds.
6. `auto` = held-out-MI selection between frequency and target_merge, enabled only for columns that actually need folding.

## References (from memory, verify before citing)
Micci-Barreca 2001 (target encoding, m-estimate); Efron-Morris 1973 / James-Stein 1961; Prokhorenkova et al. 2018 (CatBoost ordered TS); Kerber 1992 (ChiMerge); Kass 1980 (CHAID); Navas-Palencia 2020 (optbinning); Slonim-Tishby 1999 (agglomerative information bottleneck); Fayyad-Irani 1993 (MDL discretization); Weinberger et al. 2009 (feature hashing); Guo-Berkhahn 2016 (entity embeddings); scikit-learn TargetEncoder (cross-fitting, >=1.3); category_encoders docs.
