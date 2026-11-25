# Research tt_b4: feature-transform recommender at scale, with correlated features, end to end (2026-11-24)

Code (all new, in `src/mlframe/feature_selection/_benchmarks/nbins_strategy/`): `tt_b4_reference.py` (the recommender), `tt_b4_panel.py` (fixtures), `tt_b4_models.py` (ridge, spline GAM, HGB, brute-force oracle),
`tt_b4_run_fwer.py`, `tt_b4_run_fwer_stepwise.py`, `tt_b4_timing.py`, `tt_b4_run_detect.py` (+ `tt_b4_analyze_detect.py`, `tt_b4_summary_detect.py`), `tt_b4_run_e2e.py` (+ `tt_b4_analyze_e2e.py`),
`tt_b4_blind_clip.py`, `tt_b4_offcenter.py`, `tt_b4_mrmr_run.py`, `tt_b4_a2_check.py`. CPU only, OMP/BLAS 1 thread, numba 2 threads. The machine was shared with four other agents, so wall times are inflated; process CPU
seconds are given where they matter. Every number below comes from a run I made; "not run" marks what I did not do. Raw logs were written to the session scratchpad (not kept in the repo).

## 1. The reference function

`recommend_feature_transforms(X, y, *, selected=None, candidates=None, n_null=50, seed=0, ...)` in `tt_b4_reference.py`.
* Candidates (default 9): id, log, sqrt, 1/x (valid only when the discovery rows have min > 0 / >= 0 / > 0), (x-med)^2, |x-med|, tanh((x-med)/IQR), cbrt, rank-gauss (average ranks, so ties and discrete columns are safe). Constants (median, IQR, rank grid, validity) are fitted on the discovery half and replayed on any other rows (`Recipes`). `CAND_MRMR` is MRMR's own fixed-zero unary table (log1p|x|, sqrt|x|, x^2, x^3, 1/x, tanh x, expm1, |x|).
* Per block of columns: all candidates are built as one (n, b, K) stack, centred, partialled on the selected design Q (orthonormal, intercept included; Freedman-Lane form, so only what the design lacks counts), and correlated with the residual and with B permuted residuals in ONE float32 GEMM (`F^T R`, R is n x (1+B)). The SAME permutations are used for every column, so correlated columns share the null structure. Per permutation and column it stores T_j = max_{f != id}(s_f - s_id) and Rel_j = max_f s_f.
* Rules from the same permutations: `column` (per-column null), `maxT` (null of max over columns), `pooled` (null pooled over columns, Bonferroni by p), `gated` (relevance max-T gate, then a longer per-column gain null on the relevant columns only, Bonferroni by their number). Acceptance additionally requires that the gain replicates on the held-out half (own permutation null for that single pair, alpha / number of discoveries, sign > 0).
* `recommend_stepwise`: forward selection with transforms. Round: relevance gate (max-T over remaining columns given the current design) -> strongest relevant column -> transform decision for it (per-column gain null + replication) -> f*(x) (or identity) joins the design; after the loop a backfit pass re-decides each committed transform given all the others. This is what gives the credit to the right column when features are correlated.
* float32 vs float64 GEMM (20000 x 100): max |dT_obs| 1.3e-7, max |dT_null| 1.4e-9, same accepted set; the null thresholds are about 1e-3.
* Reproduces wave 1: on the 9 `tt_a2` fixtures x 7 seeds (n=20000, y = g(x) + 0.5 z + noise) the vectorised function (9 candidates incl. rank-gauss) is correct 63/63 marginally and 63/63 conditional on z, 0/21 false positives on the controls.

### Time (n=100000, one pass, n_null=50, rule gated, replication on; p=1000 stored as float32; blocks of columns, RSS after the run)
| p | mode | wall s | process CPU s | screening wall s | RSS MB |
|---|---|---|---|---|---|
| 100 | discover on half A + replicate on B | 20.7 | 7.7 | 8.9 | 85 |
| 100 | all rows for discovery | 25.3 | 11.5 | 17.8 | 86 |
| 1000 | discover on half A + replicate on B | 106.9 | 40.0 | 78.4 | 431 |
| 1000 | all rows for discovery | 118.7 | 78.4 | 96.0 | 431 |
Wall is inflated by the other jobs; CPU seconds are the cost. Micro-profile of one 33-column block at n=50000: rank-gauss 0.47 s (np.unique per column), cbrt 0.11, log/tanh 0.05, GEMM (297 x 50000 x 51 float32) 0.06. So rank-gauss is about half of the screening and the GEMM almost nothing; reusing MRMR's rank/quantile codes would remove it (not run). Stepwise at n=20000: 2-5 s (p=20), 10 s (p=100), 55 s (p=500), also on the shared machine.

## 2. Family-wise error with mostly-noise columns (100 datasets unless noted, mixed marginals incl. t3 and discrete, blocks of 5 correlated 0.5)
"Any" = share of datasets with at least one accepted transform. N0: y is noise. N1: y = 4 linear informative columns (every true link linear). N1s: N1 with the 4 informative columns given as `selected`.
| scenario | rule | any, no replication | any, + replication | accepted columns / dataset (no repl. -> repl.) |
|---|---|---|---|---|
| N0 n=5000 p=100 | column | 0.970 | 0.040 | 3.54 -> 0.04 |
| | maxT | 0.010 | 0.010 | 0.01 -> 0.01 |
| | pooled | 0.010 | 0.010 | 0.01 -> 0.01 |
| | gated | 0.020 | 0.010 | 0.02 -> 0.01 |
| N0 n=20000 p=100 (40 ds) | column / maxT / pooled / gated | 0.975 / 0.025 / 0.025 / 0.000 | 0.050 / 0 / 0 / 0 | column 4.35 -> 0.05 |
| N0 n=5000 p=500 (30 ds) | column / maxT / pooled / gated | 1.000 / 0.033 / 0.033 / 0.000 | 0.067 / 0 / 0 / 0 | column 20.87 -> 0.07 |
| N0 n=5000 p=100 studentised null (60 ds) | column / maxT / pooled / gated | 0.967 / 0.367 / 0.367 / 0.250 | 0 / 0.017 / 0.017 / 0.017 | |
| N1 n=5000 p=100 | column / maxT / pooled / gated | 1.00 / 0.96 / 0.95 / 0.99 | 1.00 / 0.95 / 0.94 / 0.99 | see below |
| N1s n=5000 p=100 | column / maxT / pooled / gated | 0.940 / 0.010 / 0.010 / 0.000 | 0.080 / 0 / 0 / 0 | |
| N1s n=20000 p=100 (40 ds) | column / maxT / pooled / gated | 0.975 / 0 / 0 / 0 | 0.075 / 0 / 0 / 0 | |

Findings.
1. A per-column null is NOT enough: 97% of p=100 datasets (100% at p=500, 20.9 false columns per dataset) get a false transform. Replication on the held-out half alone brings it to 4-7%, still above nominal. A max over columns (max-T, or the pooled null with Bonferroni; they behave identically) gives 1-3% before replication, 0-1% after; the gated rule gives 0-2% and is the only one that keeps power when many columns are relevant, because its gain p-value is computed on the relevant columns only with a long null (a Bonferroni of 50 permutations cannot reach alpha / m).
2. Studentising the gains by the per-column null sd (50 permutations) makes max-T worse (36.7% before replication): the sd estimate is noisy and small-sd columns dominate. Use the raw R^2 gain.
3. N1 (marginal, no conditioning) is not a false-positive problem of the null: of the accepted columns 455 (column) / 292 (maxT) / 364 (gated) over 100 datasets are noise columns correlated with an informative column (their marginal relation is genuinely nonlinear because the copula marginals differ), 3 / 2 / 3 are the informative linear columns themselves, 2 / 0 / 0 are other noise. Conditioning on the true linear columns (N1s) removes them (0 accepted for maxT and gated).
4. Stepwise driver on the same panels (n=5000, p=100, 100 datasets each): N0: 5% of datasets commit one column (the nominal gate level), 0 non-identity transforms. N1: 0/100 datasets with a non-identity transform; all 400 informative linear columns committed; 3 extra committed columns in total in 100 datasets (2 datasets with a pure-noise column).

## 3. Detection on correlated panels (judge = test R2 of ridge on all raw columns with ONE column replaced by one candidate, fresh 20000 test rows; gain units are test R2)
Panel (`tt_b4_panel.make_panel`): 3-6 nonlinear informative columns with distinct links from {log, square, recip, sqrt, abs, sigmoid}, 1-2 linear informative columns (one lognormal in odd seeds), partner noise columns correlated at rho 0.3 / 0.6 / 0.9 with each informative column (marginals lognormal / Gaussian / exponential, so a log or tanh of the partner is a tempting wrong credit), noise columns with Gaussian / lognormal / t3 / Poisson / ordinal / exponential / uniform marginals in correlated blocks of 5, odd seeds with a t3 informative column (sigmoid link) and every third seed with a Poisson informative column (sqrt link). Signal share of var(y) 0.7 unless noted.
A "strong" column has an oracle gain >= 0.01 R2. hit = pick realises >= 0.8 of the oracle gain, partial 0.3-0.8, miss otherwise.
M1 = one pass, selected=None, rule gated. M2 = one pass with the true informative columns as `selected` (an idealised MRMR support). M3 = stepwise.
| setting | datasets | strong columns | M3 hit / partial / miss | M1 gated | M2 gated | wrong credit (partner + noise columns given a transform) per dataset M1 / M2 / M3 | useless adds per dataset M1 / M2 / M3 | linear columns given a transform (M3) |
|---|---|---|---|---|---|---|---|---|
| p=20 n=20000 | 8 | 32 | 32/0/0 | 32/0/0 | 32/0/0 | 3.75 / 2.88 / 0.00 | 2.12 / 0.88 / 0.00 | 0/12 |
| p=20 n=2000 | 8 | 33 | 32/1/0 | 30/1/2 | 32/1/0 | 1.62 / 1.25 / 0.00 | 0.50 / 0.12 / 0.00 | 0/12 |
| p=100 n=5000 | 6 | 24 | 24/0/0 | 24/0/0 | 24/0/0 | 5.50 / 3.50 / 0.00 | 2.00 / 0.17 / 0.00 | 0/9 |
| p=100 n=20000 | 6 | 24 | 24/0/0 | 24/0/0 | 24/0/0 | 7.17 / 4.17 / 0.17 | 3.50 / 0.33 / 0.00 | 0/9 |
| p=100 n=20000, signal share 0.3 | 6 | 20 | 20/0/0 | 20/0/0 | 20/0/0 | 5.67 / 3.33 / 0.00 | 2.17 / 0.00 / 0.00 | 0/9 |
| p=500 n=20000 | 4 | 16 | 16/0/0 | 16/0/0 | 16/0/0 | 7.00 / 5.25 / 0.00 | 3.50 / 1.50 / 0.00 | 0/6 |
| total (default candidates) | 38 | 149 | 148 / 1 / 0 | | | | | 0/57 |
Per link, stepwise, all six settings: abs 25/25 hit (pick abs), square 24/24 (square), recip 37/37 (recip), sigmoid 34 hit + 1 partial (picks cbrt 35, tanh 1; the oracle also prefers cbrt over the tanh candidate), log 20/20 (picks log 15, rank-gauss 5, cbrt 1; mean oracle gain 0.028), sqrt 8 hit + 19 weak (oracle gain < 0.01 on exponential / Poisson columns; picks sqrt 26 of 27); 3 more weak columns among log / recip / sigmoid. Heavy-tailed informative column (t3 sigmoid) 19/19 hits.
* Credit assignment: a one-pass marginal analysis (M1) gives a transform to 1.6-7.2 partner columns per dataset, about 70% of the partners (e.g. a tanh or rank-gauss of an exponential-marginal partner correlated 0.9 with the sigmoid column), and 2-3.5 of those do nothing on held-out rows. Handing M1 the true linear columns (M2) halves it but does not remove it (a raw column leaves the nonlinear part of its own effect in the residual and a correlated partner picks it up). The stepwise driver, which puts the transformed column f*(x_j) into the design before looking at its partner, gives 0 to 0.17 wrong credits per dataset at every p. One earlier smoke run at p=20 (not in the table, default `make_panel` config) committed a weak `square` on a 0.9-correlated partner at step 5 (gain 0.001): the leak is not zero when the design transform of the informative column is imperfect (cbrt for a tanh link).
* MRMR's own unary table as the candidate set (`CAND_MRMR`, p=20 n=20000, 8 datasets): 32/32 strong columns hit relative to that table (abs0 5, sq0 5, recip0 8, tanh0 for the sigmoid 7, log_abs / tanh0 for log, sqrt_abs 6), stepwise wrong credits 0.50 per dataset, useless 0.38. The criterion works on whatever columns MRMR generates.
* Off-centre columns (`tt_b4_offcenter.py`, Gaussian square / abs / sigmoid columns shifted by +3 after y is built, 8 datasets, 13 columns): median-centred candidates hit 13/13 with a mean realised gain of 0.081 R2; MRMR's fixed-zero table (x^2, |x|, tanh x) reaches a mean best-candidate gain of only 0.027 (a third), because x^2 of a shifted column is not (x-3)^2. The criterion is not the limit here, the candidate table is.

## 4. End to end (test R2, 20000 fresh test rows; signal ceiling about 0.70 because the signal share is 0.7)
Feature sets: none = raw; blind = raw + every valid candidate of every column (unclipped); onepass / onepass_sel / stepwise = raw + recommended transforms added; oracle = for each column independently the candidate with the best ridge R2 on a 25% validation split of the train rows, replaced, refit on all train rows (brute force); stepwise_rep = stepwise columns replaced instead of added (ridge only).
Models: ridge (standardised, lambda 1), additive cubic-spline GAM (6 quantile knots), HistGradientBoosting (200 iterations, 15 leaves). Mean over seeds.
| setting | feature set | cols | ridge R2 | ridge RMSE | GAM R2 | HGB R2 |
|---|---|---|---|---|---|---|
| p=20 n=2000 (6 seeds) | none | 20 | 0.4618 | 2.1166 | 0.6750 | 0.6187 |
| | blind | 153 | 0.6741 | 1.6415 | 0.5871 | 0.6234 |
| | onepass | 26 | 0.6872 | 1.6077 | 0.6728 | 0.6258 |
| | onepass_sel | 26 | 0.6917 | 1.5964 | 0.6727 | 0.6255 |
| | stepwise | 24 | 0.6922 | 1.5952 | 0.6736 | 0.6263 |
| | oracle | 20 | 0.6897 | 1.6020 | 0.6744 | 0.6242 |
| p=20 n=20000 (4 seeds) | none | 20 | 0.4893 | 2.0788 | 0.6955 | 0.6799 |
| | blind | 151 | 0.6929 | 1.6016 | 0.6921 | 0.6800 |
| | onepass | 28 | 0.6958 | 1.5948 | 0.6958 | 0.6806 |
| | stepwise | 24 | 0.6957 | 1.5950 | 0.6959 | 0.6804 |
| | oracle | 20 | 0.6950 | 1.5968 | 0.6960 | 0.6806 |
| p=100 n=5000 (4 seeds) | none | 100 | 0.4866 | 2.0914 | 0.6486 | 0.6513 |
| | blind | 741 | 0.1904 | 2.4715 | not run | 0.6537 |
| | onepass | 110 | 0.6848 | 1.6315 | 0.6471 | 0.6544 |
| | onepass_sel | 108 | 0.6854 | 1.6297 | 0.6476 | 0.6540 |
| | stepwise | 104 | 0.6857 | 1.6290 | 0.6484 | 0.6544 |
| | oracle | 100 | 0.6706 | 1.6721 | 0.6459 | 0.6566 |
| p=100 n=20000 (3 seeds) | none | 100 | 0.5025 | 1.9451 | 0.6885 | 0.6806 |
| | blind | 745 | 0.6142 | 1.6677 | not run | 0.6834 |
| | onepass | 110 | 0.6954 | 1.5120 | 0.6888 | 0.6833 |
| | stepwise | 104 | 0.6953 | 1.5123 | 0.6890 | 0.6834 |
| | oracle | 100 | 0.6938 | 1.5158 | 0.6841 | 0.6839 |
Replace vs add for ridge: stepwise_rep 0.6913 / 0.6951 / 0.6852 / 0.6950 against add 0.6922 / 0.6957 / 0.6857 / 0.6953 (same four settings): no difference worth a switch.
Blind-add with every added column winsorised to the train 0.5 / 99.5 percentiles (`tt_b4_blind_clip.py`; unclipped 1/x of a positive column with tiny values explodes on test rows, ridge R2 -1.17 in one p=100 seed): ridge p=20 n=2000 0.6748 (GAM 0.6193 vs 0.6736 for stepwise); p=100 n=5000 0.6510 vs 0.6857; p=100 n=20000 0.6885 vs 0.6953. So the cost of adding everything is, for ridge, 0.007-0.035 R2 when the columns are clipped and up to 0.5 when they are not; for the GAM 0.056 at n=2000 (0.088 unclipped); trees are not hurt (+0.000 to +0.005, noise).
Reading. (a) Ridge gains 0.20 R2 (RMSE 2.09 -> 1.63 at p=100 n=5000) from the recommended transforms, indistinguishable from the best additive model; the recommended set equals the brute-force oracle at p=20 and beats it at p=100 n=5000 (0.6857 vs 0.6706: the validation-picked oracle chooses among 800 noisy comparisons and overfits). (b) The spline GAM needs nothing: it already reaches 0.6955 without transforms (it learns the shape). HGB moves by +0.001 to +0.008 (largest at n=2000, p=20; only the non-monotone square / abs can help a tree). The value of the recommender is for linear heads and for avoiding the blind-add cost on GAMs. (c) All three recommended variants end up within 0.001-0.004 R2 of each other; the difference between M1 and M3 is which columns are credited, not accuracy.

### (d) MRMR's own fe_max_steps=1 output (20000 x 20, `MRMR(fe_max_steps=1, verbose=0, random_state=seed)`, default `make_panel` config, 3 seeds, test R2 of ridge on `transform()`)
| seed | informative links | ridge on the 20 raw columns | ridge on MRMR output (n columns) | MRMR fit s | ridge on raw + stepwise recommended transforms | recommender s |
|---|---|---|---|---|---|---|
| 0 | sqrt, recip, sigmoid, abs, id | 0.4858 (RMSE 1.9002) | 0.6691 (11) RMSE 1.5242 | 264 | 0.6971 (RMSE 1.4583) | 4.7 |
| 1 | abs, log, recip, square, id | 0.3465 (2.1633) | 0.6699 (7) 1.5375 | 189 | 0.7027 (1.4590) | 3.8 |
| 2 | sqrt, sigmoid, recip, abs, id | 0.4876 (1.9216) | 0.6803 (14) 1.5178 | 252 | 0.7006 (1.4689) | 4.4 |
MRMR output columns (in_output, origin): seed 0: raw c0..c4, `c3__pade2` and `c1__LL2` (hybrid_orth, mrmr_gain 0.57 / 0.49), `c1__relu_gt1.78`, `c3__relu_gt0.054` (hinge), two `rff__` columns. Seed 1: raw c4, c3 only; `(c0__ratio_log__c2)` (mi_greedy), `lof__...`, `binagg_mean(c6|qbin(c1))`, `c3__relu_lt0.039`, `c2__LL2`. Seed 2: raw c1, c2, c7, c4; `(c3__ratio_log__c2)`, `sir__...__dir1`, three `rff__` and two `opat__` columns, `(c2__sub__c1)`, `c0__L3`, and the only pure unary table column of the three fits, `log_abs(c3)` (c3 has the abs link; mrmr_gain 0.056). MRMR drops raw informative columns (seed 1 keeps raw c3, c4 only) and replaces them with Legendre / Pade / hinge / random-Fourier / SIR bases and pair forms; none of the output columns is a plain recip / sqrt / cbrt / log of an informative column (the single unary-table column in the three outputs is `log_abs(c3)`), whereas the recommender picks exactly those. The `mrmr_gain` of the hinge and binagg columns and of the seed-0 rff columns is NaN (wave 1 saw the same for `log_abs`). Ridge on MRMR's selected output is 0.026-0.036 R2 below ridge on raw + recommended transforms, at 40-70x the fit time (the recommender saw 20 columns of 20000 rows; MRMR does much more than unary transforms, so this is not a like-for-like speed claim).

## 5. Where the criterion misses, adds a useless transform, and what it costs
* Misses (strong columns, 149 in six settings, stepwise): 0 misses, 1 partial (a sigmoid column at n=2000 where cbrt realises under 0.8 of the oracle gain). One-pass M1 misses 2 weak sqrt-on-Poisson columns at n=2000 (33 strong columns, M1 30/1/2). Detection of a +0.01 to +0.03 R2 gain works down to n=2000 and at signal share 0.3.
* Gains below 0.01 R2 (22 weak columns, 19 of them sqrt on exponential or Poisson columns): stepwise gives a transform to all 22, realised gain +0.006 to +0.008 R2 on average per setting, so they are not harmful but are not worth a column either.
* Gap to a GAM: the sigmoid link is closed by cbrt, not tanh (the tanh candidate has a fixed scale, IQR): wave 1 measured a 0.022 R2 single-feature gap to a GAM; in these panels ridge with recommended transforms is 0.0002 below the GAM at n=20000 p=20 (0.6957 vs 0.6959), so the gap is negligible once the other columns dominate.
* Useless adds: one-pass credits 2-3.5 partner columns per dataset that gain nothing on test rows; stepwise 0 (one dataset out of 38 gave 0.17 wrong credits per dataset at p=100). Linear informative columns (including lognormal and t3 marginals): 0 transforms in 57 columns.
* Candidate-table limits, not criterion limits: off-centre even links (above), anything periodic or piecewise (not in the table), links that need a fitted parameter (Box-Cox lambda, not run), transforms of the target (other agents).
* Not run: classification targets, heavy-tailed or heteroscedastic noise in y, LightGBM / CatBoost, GAM with blind-add at p=100 (memory), the recommender applied on top of MRMR's selected raw columns, e2e at n >= 100000, stepwise at n=100000, non-Gaussian copulas, Box-Cox with a fitted lambda, MRMR at p=100.
* Caveat inherited from wave 1: for one column the Pearson criterion and the ridge judge are both functions of the same correlation, so agreement with the judge is partly by construction. What these runs add is not circular: the null calibration with 100-1000 columns, the replication gate, credit assignment under correlation (the judge replaces one column inside a 20-100 column model on fresh rows), and the end-to-end R2 of three model families.

## 6. Verdict
The criterion is sound at scale. With max-T / gated nulls plus held-out replication it keeps the family-wise error at 0-2% with 100-500 mostly-noise columns (heavy-tailed and discrete included), finds 148 of 149 strong transforms (and the best candidate of MRMR's own table when that table is used), gives a transform to none of 57 linear informative columns, and its recommended set is as good as brute-force search on held-out data for ridge (+0.20 R2) while the GAM is unaffected and HGB gains at most 0.008. Two designs matter: (1) a max over columns, not a per-column null, and (2) conditioning on the transformed columns already chosen (stepwise), without which 1.6-7 partner columns per dataset get a wrong transform. Cost at n=100000: about 8-12 s CPU at p=100 and 40-80 s CPU at p=1000 (rank-gauss is half of it).

## 7. Integration recipe for MRMR
MRMR already builds the candidates (`_mi_greedy_fe.UNARY_TRANSFORMS` via `_stage_mi_greedy` in `_mrmr_fit_impl/_fe_stage_cascade_early_a.py`; `create_unary_transformations(preset)` in `feature_engineering.py`; recipes `build_mi_greedy_transform_recipe`) and builds a linear list in `_usability_lists.py` (CV-MAE greedy), but gives unary columns no linear credit (`mrmr_gain` NaN). Proposed, in a new module (no edit of `_mrmr_class.py`):
1. `linear_usable_transforms(candidate_columns, y, selected_design, ...)`: take the candidate columns MRMR already materialised (grouped by source column, identity first) instead of rebuilding them. Generalise `screen()` of `tt_b4_reference.py` to a (n, M) candidate matrix with a group id; the statistic, the shared permutations, the max-T / gated rules and the replication are unchanged. Run it on the shared fit subsample (`_fe_shared_subsample_idx`, <= 50000 rows) and reuse MRMR's rank codes for the rank-gauss candidate.
2. Call it after the FE stage with `selected` = the raw support, then run the stepwise loop (relevance gate -> transform decision -> design grows -> backfit) over the raw support and the unary candidates. Keep every unary candidate the loop commits as a non-identity transform and drop the unary candidates it rejects from the linear list.
3. Record in `fe_provenance_` (via `_pop_prov`) new columns `linear_gain = 0.5*log((1 - s_id)/(1 - s_f))`, `linear_p`, `linear_replicated`; fill the NaN `mrmr_gain` role for unary columns with it; expose `linear_usable_transform_ = {column: recipe}` and feed those columns to `support_linear_` / `transform_usability(which='linear')` as the shortlist, so `usability_greedy` does not need CV-MAE evaluations for single-column unary forms.
4. Candidate table: add the median-centred forms `(x-med)^2`, `|x-med|`, `tanh((x-med)/IQR)` with median / IQR frozen in the recipe, `log(x)` for positive columns (not only `log1p|x|`), `cbrt`, `rank-gauss`; the fixed-zero `x^2`, `|x|`, `tanh x` capture a third of the gain on off-centre columns.
5. Defaults: alpha 0.05, 50 permutations, relevance gate by max-T over columns, gain null on the relevant columns with at least 4 m / alpha permutations (m relevant columns), replication on a held-out half with alpha / m; no studentising. Use for linear / additive heads only; leave `support_` untouched (trees gain nothing).
6. Tests: the `tt_b4_panel.make_panel` panels (one-pass must not give partner columns a transform when the stepwise loop is used), a global-null panel with p=100 (any accepted transform below 5% of datasets, expected 0-2%), and a lognormal / t3 linear control (no transform).
