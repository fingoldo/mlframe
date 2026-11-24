# Plan: recognising useful transforms of the target and of the features in MRMR without retraining the downstream model

Rewritten 2026-11-24 from the first plan of 2026-11-23, the ten research reports of two agent waves (`research_tt_a1..a5.md`, `research_tt_b1..b5.md` in this folder), the unary-transform probe of the real MRMR (`research_mrmr_unary_probe.md`) and the gate probe of 2026-11-24 (section 3.2). The text of the first plan and its measurements is kept in the appendix at the end. Owner's premise, kept as the guiding idea: the only light in the tunnel for recognising the benefit of a transform without a model refit is information-theoretic quantities (MI, CMI, likelihood gain); this plan says which of them work, which do not and why.

## 0. The goals (owner, 2026-11-24)

MRMR must close five defects, each pinned by new tests, on the four probe fixtures of `research_mrmr_unary_probe.md` (`sin_wide`: y = sin(x1) + x2 + noise with x1 in [-6, 6]; `sin_narrow`: x1 in [-1, 1]; `log_lognormal`: y = ln(x1) + x2 with a lognormal x1; `square`: y = x1^2 + x2) and on their classification and target-side twins:

1. **The exact transform is found**, not only an approximation by Fourier, Chebyshev or hinge bases.
2. **The unary table is effective**: the transform it already offers (`log_abs`, `sqrt_abs`, `square`, `cube`, `reciprocal_safe`, `tanh`, `expm1_clip`, `abs`, `sin`, `cos`) is credited when it is the right one.
3. **A monotone transform is credited** (`log` of a lognormal column): rank-based MI cannot do it, a linear-usability criterion can.
4. **The output is not noisy**: 2 signal features give 2 to 3 columns, not 8 to 12 (today: raw x1, a `sir` direction, `round`, four `relu` hinges, `fourier3`, `T3`).
5. **A target built by a specific transform** (power, ln, sin, exp ...) of the inputs, continuous or binarised afterwards, is recognised: for a continuous target the inverse link is recovered (target side, `target_transform_`); for a thresholded target the transform of the features that built the score is found (feature side, classification criterion).

## 1. What the evidence says

All numbers are from this folder; every experiment is synthetic unless stated, CPU, fresh test rows.

### 1.1 Which quantity can recognise a useful transform (regression target)

* **Rank-binned MI is the same for every monotone f** (exactly), so it is a ceiling of "dependence", not a selector. **Equal-width MI** (the proposer of the first plan) is biased: under a safe judge (clipped inverse, Duan smearing; 11 fixtures x 7 seeds x 4 models) it picks a transform worse than identity by more than 2 percent in 49-57 percent of cases; on `linear` it picks `signed_sqrt` 7/7 (ridge RMSE 1.82x), on `lognormal` `yeo_johnson` 7/7 (2.41x), on `hetero_lognormal` 12.5x. The overflow of the unsafe inverse had been hiding how bad the choice is (A4). In classification the same proposer accepts 126/126 gain cases but names the right transform in 70 and accepts 43/78 controls under the permutation null (B1).
* **What selects the transform is the Gaussian information of the transformed target WITH the Jacobian**: `LL_f = -0.5 log MSE_f + mean log|f'(y)|` (A1, A3 independently). Without the Jacobian the number is not comparable across f (ridge errors up to 1.5e9). With a sign-flip residual-bootstrap null for `max_f (LL_f - LL_id)` and a held-out original-scale gate (A3 on 56 cases): correct transform 55/56, false positives 1/28, ridge error vs identity mean 0.796, worst 1.000.
* **The Jacobian LL does not survive the distribution zoo on its own** (B2: 14 target families x 2 noise kinds x 7 seeds, n = 5000): the full recipe ("W1": LL + null + gate) found 57 of 93 helpful runs; it is blind to count data (the null draws clipped at 0 create atoms: fixed by not clipping, 0/5 -> 3-5/5), to a (0,1) target clipped at the ends (LL cannot see `logit`), and on Weibull / multiplicative noise it prefers `cbrt` to the better `log`. A **cross-fitted original-scale paired sign-flip max-T test** ("CV") found 87 of 93, 0 harmful picks, regret 1.003. CV is a closed-form linear-head surrogate of the judge, not an information-theoretic quantity; it is weaker than LL at small n on strongly skewed targets (lognormal / multiplicative 2/5 at n = 300 and 1000 where LL 5/5), so the union "accept if CV or the LL pipeline accepts" is best at n <= 1000 (18/24 and 21/25 hits). [B3: reference implementation `recommend_target_transform` and the fresh-fixture validation: see section 1.6.]
* **The variance side** (A5): the scale of the residuals as a function of the features (rank-MI between |residual| and the features after a cheap spline fit) finds `log` for multiplicative noise, `sqrt` for Poisson, `cbrt` for cube and rejects everything on homoscedastic targets; the RMSE gain is small (0-3 percent) outside the cube. Conditional MI in copula form is exactly invariant to monotone f and is useful to credit a feature only for NEW information, not to rank transforms.
* **Only linear and additive heads gain in RMSE** (A4, B1, B2): trees 0-6 percent under RMSE, up to 30 percent under MAE on lognormal-type targets; HistGradientBoosting gets at most 0.0014 nats from any fitted transform of a classification feature (B1). A recipe is offered to linear / additive heads; real data (diabetes, California housing, A5) shows no gain or a small loss for trees.

### 1.2 Transforms of a FEATURE, regression target

* MRMR already PROPOSES the right transforms (`log_abs(x)` on a lognormal, `x__He2` / `x__sqa_T2` on a square) and does not credit them: the gain of `log_abs(x)` is NaN and the extra columns change with the seed (A2). The missing link is **linear usability**.
* **Criterion** (A2): the Pearson information `-0.5 log(1 - rho^2)` of f(x) with the residual of y after the already selected linear columns, null by permutation of the residual, threshold the 95th percentile of `max_f (s_f - s_id)`. 63/63 correct decisions on 9 fixtures, 0 false positives on 21 controls. For one feature this is the R^2 of a linear model on f(x), so agreement with a ridge judge is partly by construction; what was validated is the null calibration and the candidate order.
* **Over the feature zoo** (B2: 9 feature laws x 6 links, n = 300..5000, 400 runs): 288 hits, 90 true negatives, 22 benign picks (RMSE ratio <= 1), regret 1.0000; type-I rate on true nulls 3/600 at n = 300 and 0/1200 at n >= 1000. **Two defects**, each with a tested fix: a correlated covariate makes the criterion pick a harmful transform in 38 of 40 runs (fix: partial correlation / Frisch-Waugh against the selected design: 0/40, zoo unchanged); with many features the per-feature null has a family-wise error of 0.88-1.00 at p = 50..200 (fix: one max over features x candidates: 0.02-0.10, power unchanged).
* **At scale and with correlated features** (B4: 100 datasets for family-wise error, p up to 500, n up to 100000): with max-T or gated nulls plus replication on a held-out half the family-wise error is 0-2 percent; the stepwise driver (relevance gate -> pick the strongest column -> transform decision -> add to the design -> repeat -> backfit) found 148 of 149 strong transforms, gave a transform to none of 57 linear informative columns, and wrongly credited 0-0.17 partner columns per dataset where one marginal pass credited 1.6-7.2. The recommended set is as good as brute force for ridge (test R^2 0.696 vs 0.695 oracle at p = 20, n = 20000; 0.686 vs 0.671 at p = 100, n = 5000), a GAM and HistGradientBoosting need nothing (at most +0.008). Cost: n = 100000, p = 100: 7.7 CPU s; p = 1000: 40 CPU s. MRMR's fixed-zero unary forms (`x^2`, `|x|`, `tanh x`) capture a third of the gain on off-centre columns (0.027 against 0.081 R^2): median-centred forms are needed.

### 1.3 Transforms of a FEATURE, classification target (B1)

* The regression criterion does not transplant. The right one is the **deviance gain of a one-feature (multinomial) logit on f(x), with the score of the other selected features as an offset**, null = max over candidates of the gain under a parametric bootstrap from the identity logit: the judge-best transform in 126/126 gain cases (K = 2, 3, 5; 16 fixture-by-K combinations), 0 harmful accepts in 244 cases, 3/78 false positives on controls (6/78 with the cheaper permutation null). Per-row deviance gain / 2n equals the held-out log-loss gain (0.1228 vs 0.1230): it is the "linear-usability credit" that is NaN today.
* Gains: 0.02-0.05 nats on monotone fixtures, 0.14-0.68 on variance-shift and `abs` fixtures; AUC moves little on monotone ones (0.001-0.02). The best transform depends on the head (LR wants `square` where LDA wants `abs`; Gaussian NB wants nothing on a variance shift): the acceptance has to know which head is downstream; the default is a logistic-type head.
* The class label has no scale; a monotone transform of a score that defines classes is inert under rank or quantile cuts (labels identical in 7/7 seeds); value-based cuts change the problem, not something to recognise. Cost: 28 ms per candidate at K = 2 in numpy (10 s per feature with B = 40); needs a screen or a compiled kernel.

### 1.4 What MRMR does today on the four probe fixtures (`research_mrmr_unary_probe.md`)

On `y = sin(x1) + x2` MRMR's output is 9 columns: raw x1, `sir__x1_x2_...`, `x1__round_0p01`, four `relu` hinges, `x1__fourier3`, `x1__T3`; raw x2 is not in the output at all, its signal rides inside the `sir` direction. The unary candidate `sin(x1)` and the pair `add(sin(x1), x2)` were both generated and both lost. The causes, from the rejection ledger and the gate probe (3.2):

1. **A deterministic function of a selected column has zero conditional MI**: once x1 is in the support, `sin(x1)` is redundant by construction, however much better a linear model could use it. Credit for a transform therefore cannot come from MI or CMI; it needs a linear-usability criterion (1.2, 1.3). This is the structural reason for goals 1-3.
2. **The pair gate rejected `add(sin(x1), x2)`** on a plug-in MI ratio of 0.77 against a bar of 0.90 although the candidate loses 0.6 percent of the out-of-fold R^2 of a 2-D smoother (3.2). The basis legs (`fourier3`, `T3`) were admitted first, and their redundancy then evicted the exact candidate.
3. **Nothing prunes**: the residual families (`sir`, `round`, `relu`, `fourier`) add columns whose held-out contribution is nil once the main structure is in.

### 1.5 Where nothing helps (limits to state in the documentation)

Trees under RMSE (A4, B1, B2); real-data tabular benchmarks (A5: diabetes, California housing: no gain, `cbrt` costs 1 percent for HGB); targets with a point mass of 3 percent or more or 20 or fewer distinct values (use the cross-fitted test); a bounded target with boundary mass; at most 5 ordinal levels; n < 300 or n / (p + 1) < 5 (a precaution, not measured); a downstream loss that is not known (RMSE and MAE need different gates).

### 1.6 The target-side reference on fresh fixtures, and the cost (B3, B5)

* **B3** (`recommend_target_transform`, `tt_b3_reference.py`; 840 fresh cases: 20 fixture kinds with correlated, mixed-type and nonlinear-mean features, p in {5, 40}, n in {2000, 20000}): the family is right in 96.2 percent (404 of 420) and the null is calibrated (the LL test fires on 14 of 210 exact nulls = 6.7 percent, the final non-identity pick 1 of 210), but the wave-1 numbers were optimistic: a pick is worse than identity by more than 2 percent in 8 of 840 cases (worst 1.30; wave 1 reported 1.000) and the mean regret to the oracle is 1.014. Two defects were found on fresh data and fixed in the reference: the clipped inverse of a bounded-image Yeo-Johnson (test error up to 1.2e10 times identity: no margin toward the asymptote for Yeo-Johnson, Box-Cox, reciprocal) and an "atom at 0" rule that fired on cube-like targets (test exact ties). The largest systematic weakness is the Yeo-Johnson lambda: the density-optimal lambda is not the error-optimal one (an opt-in polish lifts good picks from 88.0 to 93.5 percent). HistGradientBoosting: neutral to slightly harmful (worse by more than 2 percent in 9.5 percent of RMSE cases), so the recipe is for linear and additive heads only.
* **The honest comparison with the code we have** (B3, control experiment `tt_b3_oldext.py`): the wave-1 headline "the proposer is biased, replace it" is true for the PROPOSER and misleading for the old PIPELINE. The shipped pipeline (equal-width-MI shortlist of 3, then the ridge held-out paired-bootstrap gate) has 0 harm and 0 false positives; on its own 4-candidate pool it ties the reference. What the reference adds is recall and range: 91.0 against 71.9 percent good picks on transform fixtures, 0.4 against 11-19 percent of gains missed, a pool of 10 candidates instead of 4 (with the old pipeline on the larger pool the shortlist of 3 misses and p = 40 shows a worst ratio of 8.6), at 3 to 8 times the time. The decision: KEEP the held-out gate as the safety layer and replace the proposer, not the whole pipeline.
* **B5, selection invariance** (MRMR, n = 3000, 735 fits): the MI core is exactly invariant to a monotone map of the TARGET under all three schemes (26/26, gain differences 0.0 to six digits, `uniform` included: MRMR does not bin the target equal-width), and to a monotone map of a FEATURE under `quantile` and `mdlp` (52/52; the only deviations are float-saturation ties of `tanh`), not under `uniform` (32/52, gain differences up to 0.52 nats). So the equal-width-MI "proposer" of the first plan measured a property of that script's private binning of f(T), not of MRMR, and **selecting features against f(T) through MI is pointless: it returns the same features**. The only place f(T) changes the final set is the value-based held-out ridge rescue (`fe_residual_rescue_*`; with it off the target map leaves the selection identical in 26/26). With `fe_max_steps = 1` even a rank-invariant MI gives a different pool when a column is mapped, because the candidate generators read values.
* **B5, cost** (CPU seconds, one thread, K = 9 candidates, B = 40 draws): target criterion 0.09 at n = 2e4, p = 20; 0.42 at n = 1e5; 4.2 at n = 1e6 (a capped design of the 64 best-correlated columns keeps p = 1000 at 0.27 / 1.2 CPU s for n = 2e4 / 1e5; the full design at p = 1000 costs as much as a ridge on all columns, 5.7 s at n = 1e5, and must not be used). Feature criterion with the exact null: 0.11 at n = 2e4, p = 20; 2.2 at n = 1e5, p = 200; 10 at n = 1e5, p = 1000 (so at p = 1000 and n >= 1e5 it runs on a row subsample, as MRMR's own FE search does); the permutation null costs only 1.2-1.5 times the observed pass because it is one GEMM per block. Dominant cost: the scalar transforms (`cbrt` 66 ns, `tanh` 27 ns, `log` 20 ns per element). Against one MRMR fit of 20000 x 20 with `fe_max_steps = 1` (255 s wall, 214 CPU s) both criteria are 0.04 + 0.05 percent. Peak memory 0.25 GB above the input.

## 2. Design

### 2.1 Feature side: a linear-usability stage after the feature engineering (goals 1, 2, 3)

New module `filters/_unary_linear_credit.py` (production port of `tt_b4_reference.py`, which is the validated reference, with the B2 fixes and the B1 classification criterion):

1. **Candidates**: the unary columns MRMR already materialises, grouped by source column with the identity first; add the median-centred forms `(x - med)^2`, `|x - med|`, `tanh((x - med) / IQR)` (median and IQR frozen in the recipe), `log(x)` for positive columns, `cbrt`, `rank-gauss`; `sin` and `cos` of the raw column (the unary table applies them only to bounded columns; a fitted frequency is section 2.2).
2. **Statistic**: regression: `s_f = rho^2(f(x), r)` with `r` the residual of y after OLS on the selected linear design and f(x) partialled on the same design (Freedman-Lane; the B2 correlated-covariate fix); classification: the logit deviance gain with the offset (B1). One GEMM per column block; float32 storage, float64 accumulation where the gain is 1e-3.
3. **Null**: 50 shared permutations of the residual (regression) or 40 bootstrap redraws (classification); the threshold is the 95th percentile of the MAX over columns and candidates of `s_f - s_id` (family-wise, B2 / B4), then **replication on a held-out half** (own null).
4. **Stepwise driver** (B4): relevance gate -> strongest column -> transform decision -> add to the design -> repeat -> backfit pass.
5. **Output**: a credited transform REPLACES its raw column in the linear output unless the raw column still has a significant partial effect after it; recipes are replayable in `transform()`; `fe_provenance_` gets `linear_gain = 0.5 log((1 - s_id) / (1 - s_f*))`, `linear_p`, `linear_replicated`, which fills the NaN `mrmr_gain` of unary columns; `linear_usable_transform_ = {column: recipe}`.
6. **Scope**: linear / additive heads; trees get nothing to credit (1.5); documented.
7. **Where it sits in `support_`**: the credited column enters `support_` (the owner's goal is the default output, not a second list); `support_linear_` and `usability_aware_lists` keep their meaning and use the credited shortlist instead of re-running the CV-MAE greedy.

### 2.2 Exact closed forms by snapping a basis (goal 1, beyond the fixed table)

The fixed table cannot contain `sin(2 x)` or `exp(-x^2)`. The basis families already find the structure approximately. Fit the single closed form that explains the basis fit: `sin(w x + phi)` (w from the dominant Fourier frequency, refined by one-dimensional least squares), `x^p` (p on a grid), `exp(k x)`, `tanh((x - c) / s)`, `|x - c|`. The family is part of the candidate set of the max-T null (so the extra parameters are paid for in the threshold); accepted only if its held-out gain is at least 95 percent of the basis' and the column count falls. The snapped column carries a recipe with its fitted parameters.

### 2.3 The pair gate (goals 1 and 4; section 3.2 has the comparison)

`best_mi / pair_mi > 0.90` compares a 1-D binned feature with a 2-D binned joint. Replace the MI ratio as the acceptance of an engineered pair column by the out-of-fold tests of `_pairs_sufficiency.py` (is the winner an honest 1-D summary of the pair; is there an interaction at all) after the existing operand-uplift floor; final thresholds from the F2 results in 3.2.

### 2.4 Parsimony (goal 4)

After the linear stage, backward elimination of engineered columns that are neither MI-essential nor linear-essential: a column goes when (a) its held-out linear gain over the rest is not significant under the permutation null and (b) its conditional MI given the rest is not significant under the permutation null. Basis legs that an exact credited column or snapped form already explains are the main targets (`fourier3`, `T3`, `relu` hinges, `round`, `sir`).

### 2.5 Target side (goal 5)

Keep the shipped held-out ridge gate of `_target_transform_search.py` (0 harm, 0 false positives on 840 fresh cases) and replace its PROPOSER: the equal-width-MI shortlist of 3 over 4 candidates becomes the Jacobian-LL ranking of the full pool of `recommend_target_transform` (B3's reference): candidates by the range of y (identity, `cbrt`, `signed_sqrt`, `signed_square`, `asinh`, `log` and `log1p` for positive y, reciprocal, `logit` for (0, 1), Yeo-Johnson with a fitted lambda), held-out Jacobian LL, sign-flip null (draws NOT clipped, B2), ties resolved to the simplest family, then the existing gate on the original scale with the clipped inverse (no margin toward the asymptote for Yeo-Johnson, Box-Cox, reciprocal; B3) and Duan smearing. B2's cross-fitted max-T test is the second opinion, and the union "accept if the cross-fitted test or the LL pipeline accepts" applies at n <= 1000. Abstain under the conditions of 1.5. `target_transform_` keeps its fields and gains the reason for an abstention. The lambda polish of B3 is an option, not a default. A thresholded target has no scale; its transform is recovered on the feature side with the classification criterion of 2.1 (second half of goal 5).

## 3. Measurements made for this plan (2026-11-24)

### 3.1 Wave 2 summary

See 1.6 for B3 and B5, 1.1 to 1.3 for A1-A5, B1, B2, B4.

### 3.2 The prevalence gate: Miller-Madow debias, permutation debias and the sufficiency test, compared

Owner's remark: the debias flag `fe_mm_debias_prevalence` was rejected in June because it over-corrects (cross-mix 3/10 -> 9/10 seeds on the weak F2); either the debias is implemented wrongly or the gate has to tell an honest 1-D summary of the pair from an artefact; compare both. Harness `gate_probe2.py` (patches the ratio hook inside a real fit and records per scored pair the raw, the Miller-Madow and the permutation-debiased ratio and out-of-fold regression statistics) and `f2_downstream.py` (the downstream error of `transform()` on fresh rows).

* **Implementation defect found in the debias path** (fixed, `cb86e55e2`): on the path where the chunk buffer is not kept, `_resolve_col(best_config[2])` raised, the `except` logged a warning and `k_eng` fell back to the nominal `quantization_nbins`. The column is now read through a getter that rebuilds it from its recipe.
* **How much the debias moves the ratio**: `y = sin(x1) + x2`, n = 6000: raw 0.77, Miller-Madow 0.80, permutation-debiased 0.80, bar 0.90; at n = 1500: 0.71 / 0.85 / 0.88. Weak F2, n = 20000, genuine pair (c, d): 0.71 / 0.72 / 0.72, additive cross pairs 0.60-0.64. At the sizes of the benchmarks the finite-sample bias is small, so the debias moves a ratio by 0.01-0.03; the gap to 1 is the resolution of a 10-bin 1-D partition against a 100-cell joint, which no bias correction touches. It matters only at about n <= 2000.
* **What separates a genuine pair from an artefact is an interaction test on fitted values, not a ratio of plug-in MIs.** Out-of-fold ridge on hat bases of the ranks, scored by MAE (R^2 is secondary and gives the same split): genuine pairs t 16-47 and MAE-skill gain 0.03-0.12, additive cross pairs and noise t -5..0 and gain about 0; the exact winner of a pair loses 0-1.5 percent of the tensor smooth's skill, a degraded one 86 percent. `_pairs_sufficiency.py` is wired as an extra accept path (default on, `MLFRAME_FE_PAIR_INTERACTION_GATE=0` rolls it back).
* **Downstream, weak F2 (n = 20000, 8 common seeds, MAE of `transform()` on 20000 fresh rows, ridge / HistGradientBoosting; raw columns 0.0546 for the boosting):**

| variant | output columns | ridge MAE | boosting MAE |
|---|---|---|---|
| old gate | 7.9 | 0.1151 | 0.0535 |
| new pair path | 6.1 | 0.0974 | 0.0525 |
| `fe_mm_debias_prevalence=True` alone | 6.9 | 0.0828 | 0.0539 |
| both (5 seeds) | 6.2 | 0.0922 | 0.0525 |

Paired over the 8 seeds: new path against old, ridge -0.0177 +- 0.0020 and boosting -0.0010 +- 0.0002, better in 8 of 8 seeds on both; debias alone against old, ridge -0.0322 +- 0.0048 (7 of 8), boosting +0.0004 +- 0.0002; debias alone against the new path, ridge -0.0146 +- 0.0056 (7 of 8), boosting +0.0014 +- 0.0004 (the new path better in 8 of 8). The June rejection counted "cross-mix" columns, which is not a loss: the debias gives zero of them here and the best linear error. So the debias is NOT useless: it is the stronger of the two for a linear head and slightly worse for trees; the pair path is the one better for both. The two do not add (both: ridge 0.0922). Decision: the pair path is on by default; the default of `fe_mm_debias_prevalence` stays False until the owner decides (it trades 0.4-0.7 percent of tree MAE for 28 percent of ridge MAE).

### 3.3 The unary credit, measured (2026-11-24; `eb7671837`)

Ridge MAE of `transform()` on held-out rows, 6000 rows, two signal features and six noise columns (the noise floor is 0.239): `sin(x1) + x2` 0.607 -> 0.2435, output `['x2', 'sin(x1)']`; `log(x1) + x2` with a lognormal x1 0.525 -> 0.2436, `['x2', 'log(x1)']`; `x1^2 + x2` 0.959 -> 0.2438, `['x2', 'sq0(x1)']`. Four signal features `sin(x1) + log(x2) + x3^2 + 0.5 x4` with four noise columns, with and without four correlated copies of the signal columns (corr 0.7-0.9): `['x4', 'add(sin(x1), log(x2))', 'sq0(x3)']`, MAE 1.219 on the raw signal columns -> 0.245 in both frames: the copies are dropped. The generating link is named exactly over 12 links x normal / Laplace / t3 noise; linear links with uniform / normal / lognormal / t3 inputs, 20 pure-noise frames and a correlated copy credit nothing.

**A nearly linear sine on [-1, 1]** (the case where `sin(x1)` differs from `x1` by a cubic term of 0.4 percent of the variance): credited when the rows resolve the cubic term, not otherwise. With an added-variable test (the residual after the column itself) the detection is 12 of 12 seeds from 6000 rows and about 25-50 percent at 2000-3600 rows (normal noise); the paired t of the curvature is about 3.6 per 1800 rows, so the evidence at 3600 rows is thin. The omnibus statistic (the second and third powers of the robustly scaled column, partialled on the design that holds the column, one permutation null shared with the candidate max, each test at half the level) and an out-of-fold REPLACEMENT filter on the candidates (`cbrt` and `rankgauss` explain part of the leftover curvature but lose the linear signal: held-out MAE worse by 8 to 10 percent) lift it: `sin` is credited in 6 of 12 (normal noise) and 11 of 12 (t3 noise) seeds at 3600 rows, 11 / 10 of 12 at 6000, 12 / 10 at 10000, 12 / 12 at 20000; 2000 rows stay at 4 / 3 of 12. Controls: 0 false credits in 280 datasets (linear target, uniform / lognormal / t3 feature, normal and t3 noise, and pure noise 2000 x 8). The remaining misses at 3600 rows are discovery-half power (half the rows are kept for replication). A paired absolute-error (Vuong-type) statistic was tried instead and is both less powerful (0 of 8 at 3600 rows) and names the wrong form (`sqrt` for `log`); it was dropped.

## 4. Tests that pin the goals

For every fixture the asserted quantity is the structure of the output and the held-out error of a linear head on `transform()`:

1. `sin_wide`: `support_` contains a column equal to `sin(x1)` (recipe name and values) or `add(sin(x1), x2)`, not only `fourier3` / `T3` bases; at most 3 columns for 2 signal features; ridge R^2 on the output within 0.01 of the oracle `[sin(x1), x2]`.
2. `sin_narrow`: `sin(x1)` is nearly linear in x1: the output is `[x1, x2]` (no column added, no false credit).
3. `log_lognormal`: `log(x1)` (or `log_abs`) credited and replaces raw x1; ridge R^2 within 0.01 of the oracle.
4. `square`: `x1^2` (centred or not) credited; output at most 3 columns.
5. Controls: linear target with lognormal / t3 / uniform features: no credited transform (global null with p = 100: any credit below 5 percent of datasets); a correlated covariate does not steal the credit.
6. Targets: `y = exp(x)`, `x^3`, `ln(x)`, `sin(x)` of a linear index, continuous (the inverse link is recovered, the held-out original-scale error does not rise) and thresholded (`1[f(index) > c]`: the feature-side classification criterion credits the transform).
7. Unit tests of every function (statistic, null, replication, stepwise, recipe replay, pickle, legacy pickles), the sufficiency functions, the metadata keys; cProfile of the stage at 20000 x 50 and 100000 x 100 before any GPU twin or KTC (the sequencing rule).

## 5. Order of work

1. Binning split: committed (`6e4a61235`).
2. Pair gate: committed (`cb86e55e2`) from the weak-F2 downstream comparison (3.2).
3. Unary linear credit and the parsimony of the output, regression and binary target: committed (`eb7671837`); `fe_provenance_` columns (`linear_gain`, `linear_p`) and a multiclass criterion are open.
4. Classification criterion for a binary target: committed with 3; multiclass open.
5. Snapped closed forms with a fitted frequency or exponent (section 2.2): open.
6. Target side: port the reference, retire the equal-width proposer, wire the abstention reasons.
7. Profile at scale; document the limits of 1.5 in the MRMR docstring.

## 6. Open decisions for the owner

* Whether the credited transform replaces its raw column in the output by default. Implemented that way: the raw column is dropped only when removing it costs under 0.3 percent of the variance out of fold on a linear and a rank-smooth head (`sin(x1)` + `x2`: `x1` goes; `y = 5 x1 + 0.2 sin(x1) + x2`: `x1` stays), so the output lists `sin(x1)` where it listed `x1`.
* The default of `fe_mm_debias_prevalence` (3.2): decided, True (2026-11-24: a measured-useful mechanism ships on; the tree-head cost of 0.4 to 0.7 percent is sized against the ridge gain of 28 percent and no automatic gate can see the downstream head).
* Which downstream head the acceptance assumes when the suite does not say (recommended: a logistic / ridge head, with trees getting nothing).
* The abstention for an unknown loss (RMSE or MAE): recommended RMSE gate, MAE as an option.

---

# Appendix: the first plan (2026-11-23), kept as history

## Plan: searching for a transform of the TARGET inside MRMR, not only of the features

Owner request 2026-11-23: MRMR finds unary and binary transforms of the features that raise the MI with the target; the same should be possible for the target (log, cbrt, ...), when a transformed target is predicted much better than the raw one.

### What is established (measured 2026-11-23, synthetic, 20000 rows, 8 features, 3 seeds, regression only)

1. MI is invariant to a monotone transform of the target under rank-based binning (equal-frequency bins, and MDL splits, which depend on order only): `MI(x; log T) = MI(x; T)` exactly. Under such binning a monotone target transform is invisible; only a non-monotone one (`|T - median|`, a residual `T - beta*u`) changes the MI. The feature side obeys the same rule.
2. Under EQUAL-WIDTH bins the invariance is gone, and the MI sees the transform that linearises the target:

| target | MI (equal-width) picks, 3 of 3 seeds | ridge RMSE on the original scale, raw -> best | GBM |
|---|---|---|---|
| lognormal (log linearises) | log (0.54-0.56 vs 0.02-0.06) | 2.28 / 2.22 / 2.03 -> 1.40 / 1.30 / 1.26 | within noise; log helps 6 percent on 2 seeds, hurts on 1 |
| cube (cbrt linearises) | cbrt (0.59-0.61 vs 0.24-0.37) | 5.01 / 4.65 / 4.73 -> 2.50 / 2.27 / 2.35 | within noise |
| linear (no transform helps) | signed_sqrt (0.606 vs 0.592): a FALSE positive of margin 0.014 | identity is best: 0.30 vs 0.70 | equal |

3. A first run of this experiment shifted the target (`y - min + 1`) and wrongly reported "no transform helps" for the cube case: the shift broke the linearisation. The experiment script `src/mlframe/feature_selection/_benchmarks/nbins_strategy/target_transform_mi.py` has no shift; it is the source of the biz_val test.
4. Not tested: classification (a class label has no scale to transform), tree models other than `HistGradientBoostingRegressor`, more than 3 seeds, heteroscedastic noise, transforms with a fitted parameter (Box-Cox lambda, Yeo-Johnson), the bias of an inverse transform of a mean prediction.

### Design

The MI under equal-width bins is a good PROPOSER and a bad JUDGE (the linear fixture shows a false positive). So:

1. Proposal. A private equal-width binning of the features and of the candidate target `f(T)`, independent of `nbins_strategy` (the default strategy stays whatever it is; it is not changed to uniform for this). For each `f` in a candidate set (identity, log for `T > 0`, cbrt, signed sqrt, square, reciprocal, and a Box-Cox / Yeo-Johnson with a fitted lambda later) the score is the sum (or the top-k sum) of `MI(x_j; f(T))`. Keep `f` whose score exceeds the identity score by a margin that comes from a permutation null of the score difference, not from a fixed number.
2. Acceptance. On held-out rows (the same even / odd split the target mix uses): fit a cheap model on `f(T)`, invert, and compare the error on the ORIGINAL scale with the raw-target model. Accept only if the error falls by a margin from a paired bootstrap of the held-out residuals. The gate decides; the MI only orders the candidates. A model-free alternative to keep in mind: rank correlation of the held-out prediction on the transformed scale.
3. Output. `MRMR.target_transform_` (name, parameters, MI score, held-out error before and after, accepted yes / no) and a note in `fe_provenance_`. MRMR does NOT change the target it was given: the recipe is returned for the training layer (the composite discovery in `training/composite` already applies and inverts target transforms and scores them on the original scale). The selection of features is made against the accepted `f(T)` only when the owner enables that (`target_transform_apply`), default off.
4. Non-monotone candidates. `|T - c|`, `T - beta*u` already exist in `fe_target_mix`; fold them in the same candidate set rather than keeping two mechanisms.
5. Cost. One equal-width binning per candidate and one cheap model per accepted-by-MI candidate; the candidate set is small. Measure with cProfile at 20000 x 50 and 100000 x 100, GPU twin and KTC only after the CPU path is profiled and optimised (sequencing rule).

### Classification analogue (research item)

A class label has no scale. The analogue is the DEFINITION of the classes: where to cut a continuous or ordinal quantity into classes (thresholds), how to merge or split classes, and which binary reading of a multiclass target a model predicts best. Questions: does the choice of threshold change what MRMR selects; can the MI as a function of the threshold find a cut at which a feature set is most informative; is a held-out log loss at that cut an honest judge. Start with an experiment (continuous latent target cut at several quantiles, features with threshold-specific signal), then decide whether it belongs in MRMR or in the training layer.

### Tests

- biz_val, per transform: lognormal -> log accepted, cube -> cbrt accepted, linear -> identity (no false positive), each over at least 5 seeds, ridge and a tree model, with the original-scale error as the asserted quantity (fails if the accepted transform does not lower it).
- unit: the proposer picks the transform with the highest equal-width MI; the gate rejects the signed_sqrt false positive of the linear fixture; a target with `T <= 0` never gets log.
- contract: `target_transform_` is pickled, defaults on legacy pickles, and the feature selection is unchanged when `target_transform_apply` is off.

### Order

1. Extend the experiment to 5+ seeds, LightGBM and CatBoost, heteroscedastic noise (measure before building).
2. Proposer and gate as functions with unit tests; no MRMR wiring.
3. Wire into MRMR as metadata only (`target_transform_`), biz_val tests.
4. Profile, optimise; GPU twin and KTC if the profile justifies.
5. Optional `target_transform_apply`; classification analogue as its own item once step 1 results are in.

### Decision on MRMR-COMPOSITE-BIN-ONCE (2026-11-23)

The child fit of fe_target_mix_candidates=True re-bins its row subset (55 column-edge computations in the main fit, 125 with the child). Not fixed now: the child runs only when candidates are on (default off) and PLAN-05 has not yet shown value. The red test is narrowed to target-independent strategies (quantile_spacing, freedman_diaconis, sturges, uniform) and stays red, without xfail. If PLAN-05 shows value, the fix is to pass the parent column codes sliced by the child rows for target-independent strategies; supervised strategies re-bin, since the child target T = y - beta*u differs.

### PLAN-05 measurement: value of fe_target_mix_candidates=True (2026-11-23, n=6000, p=10, y = 3*x0 + share*x1*x2 + noise, 3 seeds each)

| fixture | candidates OFF R2 | ON R2 | wall OFF / ON | weak term x1*x2 found OFF / ON |
|---|---|---|---|---|
| share 4 %, noise 0.3 | 0.9857 / 0.9858 / 0.9871 | 0.9904 / 0.9858 / 0.9871 | 33.6 / 16.0 / 11.1 s vs 39.7 / 31.4 / 29.0 s | yes / yes in all seeds |
| share 0.8 %, noise 1.0 | 0.8925 / 0.8900 / 0.8951 | identical | 10-27 s vs 10-11 s | no / no in all seeds |

Conclusion: on both fixtures the candidate search adds nothing (one +0.005 R2 in 6 fits, at up to 2.6x wall on the easy one; on the hard one the weak term is below what either mode finds). No value is proven, so no biz_val test asserting value is written; a test would have to assert the absence of a gain. Open: an intermediate share (about 1.5-2 %) and a fixture where the weak term needs the residual after the dominant one (T = y - beta*u) rather than being visible directly. The composite bin-once fix stays deferred.

Third fixture (share 1.8 %, noise 0.6, 3 seeds): R2 0.9594 / 0.9600 / 0.9593 identical with candidates OFF and ON; the weak term is found in both modes; wall 22.9 / 6.4 / 11.5 s OFF vs 26.4 / 18.1 / 23.6 s ON. Across three fixtures and nine seeds the candidate search gave one +0.005 R2 and never changed what is found, at 1.1x-2.8x wall. PLAN-05 verdict: value NOT proven; no value-asserting biz_val test; fe_target_mix_candidates stays default off. Reopen only with a fixture where the weak term needs the residual T = y - beta*u.

### Step 1 result: the extended experiment (2026-11-23, 5 target kinds x 7 seeds, 20000 rows, 8 features, ridge / HistGB / LightGBM / CatBoost, `_benchmarks/nbins_strategy/target_transform_mi_ext.py`)

The proposer (the transform with the largest sum of equal-width-bin MI over the features) against the judge (the RMSE on the ORIGINAL scale of a model fitted on f(y) and inverted):

| target | the MI picks (every seed) | ridge: best / RMSE(picked) over RMSE(identity) | trees: best / ratio |
|---|---|---|---|
| cube | cbrt 7 of 7 | cbrt 7 of 7 / 0.482 (the error is halved) | scattered / 0.98-1.00 |
| lognormal | yeo_johnson 7 of 7 | log 7 of 7 / the picked inverse overflows in all 7 (non-finite) | scattered / 1.05-1.40 (catboost) |
| hetero_lognormal | yeo_johnson 7 of 7 | log 7 of 7 / non-finite in all 7 | scattered / 0.92 (lgbm), 5.2 (catboost, overflow) |
| linear | signed_sqrt 7 of 7 (FALSE positive) | identity or yeo_johnson (lambda about 1) / 2.28 (the error more than doubles) | identity best in 4-6 of 7 / 1.04-1.06 |
| hetero_linear | signed_sqrt 7 of 7 (FALSE positive) | identity or yeo_johnson / 1.41 | identity best / 1.03-1.04 |

Across the 35 cases the MI-picked transform is the best one for ridge in 7, HistGB in 2, LightGBM in 3 and CatBoost in 3. Findings: (1) the gain exists only for a linear (or linear-like) downstream model and only when the target needs a power transform: cbrt for a cubic target halves the ridge RMSE, log for a lognormal target is the ridge optimum; (2) trees do not gain: the best candidate scatters over seeds and the picked transform is within a few percent of identity, so a transform recipe should be offered to linear heads only; (3) the proposer is biased: on a plain linear target it picks signed_sqrt every time (14 of 14 with the heteroscedastic variant) and that costs a ridge model 1.4-2.3 times its error, and on a lognormal target it prefers Yeo-Johnson to log and the inverse of Yeo-Johnson overflows on a heavy right tail (non-finite RMSE in 14 of 14 ridge cases and 3 CatBoost ones); (4) so the held-out gate of the design is mandatory, not an option, and the candidate set should drop an unbounded inverse (or clip the prediction before inverting) and keep log for positive targets. Decision: step 2 (proposer and gate as functions with unit tests) is justified, scoped to linear heads, with the acceptance test on the original scale; the tree heads are out of scope.

### Classification analogue, first measurement (PLAN-04, 2026-11-23, `_benchmarks/nbins_strategy/class_cut_search.py`, 20000 rows, 8 columns, 5 seeds, cuts at the 10, 25, 50, 75 and 90 percent quantiles of a latent score)

Question of the plan: can the MI as a function of the threshold find the cut at which the features are most informative? Answer: NO in this form. (1) The symmetric uncertainty summed over the columns (the normalisation that makes cuts comparable) peaks at the balanced cut q = 0.5 in every seed of the `uniform` fixture (one linear signal) and of the `tail` fixture (a column that matters only above the 90th percentile): its curve is shaped by the class balance, not by where the signal is. (2) The held-out AUC peaks at the extreme cuts (q = 0.1 or 0.9: the classes are far apart), also by construction; the argmax of the SU curve equals the argmax of the AUC curve in 0 of 5 seeds for both logistic regression and gradient boosting on both fixtures (and in 3 of 5 and 2 of 5 on pure noise, by chance). So no single metric compares two cut definitions: they answer different questions, and which cut is "best" is a property of the downstream task (a cost-weighted utility), not of the data. (3) What the cut sweep DOES show is per-column and local: on the `tail` fixture the SU of the tail column x2 relative to x0 goes 0.126 (q 0.1), 0.057, 0.042 (q 0.5), 0.081, 0.233 (q 0.9), a five-fold rise at the cut where it matters, while the mean SU of the noise columns stays at 0.0002 at every cut. Verdict for the plan: drop the "pick the best cut" search; keep a per-cut SU profile per column as a diagnostic of threshold-specific features (a column whose profile is far from flat is a candidate for a tree head or for a feature built on that tail), to be tried as a metadata item (`column_stats_` per cut) only if the owner asks for it; the choice of the cut stays with the task.

## 6. What the detected forms are good for (owner's question 2026-11-24: "just pass them to the Composite Target? nothing else?")

The credit (and the interaction credit) produce, per input, a closed form `u_i`, and per pair a product term; the target-side search produces an outer link `U` with `y = U(F(u_i(x_i)))`. Together they describe a small parametric model, `MRMR.unary_forms_` / `unary_forms_tied_`, and the recipes. Uses, by cost and value:

**Inside MRMR**
1. Prune the FE search: the unary x binary grid of the pair / triple families is searched blind today; seeding it with the detected `u_i` (and skipping forms the data rejected) shrinks the candidate count (speed) and the false discoveries.
2. Redundancy beyond monotone maps: if `B = u(A)` exactly (the credit names `u`), `B` is redundant given `A` in a way rank-MI cannot see for a non-monotone `u`; the same test finds duplicated and leaked columns.
3. Relevance of a column through its form: the MI of `u(x)` with `y` for the non-monotone forms (`sin`, `abs`, square) is what the downstream head sees; use it in the relevance ranking.
4. The outer link `U` (target side) and the inner forms give a composite target: the downstream model is trained on `U^-1(y)` against `F(u(x))`.
5. A prior library: the forms found per column kind (a price column is a log, a count column is a sqrt) stored like the KTC cache, to order the candidates of the next fit.

**Outside MRMR (the estimators)**
6. Offset / base margin: the parametric fit `U(F(u(x)))` is a baseline for gradient boosting (`init_score`, `base_margin`) and a skip connection of a network; the booster then models only the residual, and the extrapolation outside the training range is the closed form's, not the trees'.
7. Tree constraints: a monotone form (`log`, `sqrt`, `cbrt`, `recip`) is a `monotone_constraints` entry; the interaction structure (which inputs appear together in a product, which only add) is an `interaction_constraints` partition for LightGBM, XGBoost and CatBoost: faster training, less overfitting.
8. Loss function: the residual law of the fit (Laplace and t3 tails give MAE or Huber, a multiplicative error gives a log link or a Gamma / Tweedie loss, a count target a Poisson loss) and the outer link give the GLM family and link; expose it as `suggested_loss_` and `suggested_link_`.
9. Model family: a sum of single-input forms is a GAM (EBM, spline GLM); a pure product is a bilinear model or a factorization machine; a single index with a link is a GLM; the structure detected is the choice of an interpretable additive model over a black box.
10. Network design: the detected forms initialize the edge functions of a KAN, the input activations of a network, or the first layer of a symbolic regression run (PySR, gplearn) as building blocks, which shrinks its search.
11. Weights and heteroscedasticity: the residual variance as a function of the fitted value gives the weights of a weighted least squares or the choice of quantile regression.
12. Data validation and drift: the domain of a form is a rule (`log` needs a positive column); a violation at inference time is flagged, and the closed form tells how far a prediction can be trusted outside the training range.
13. Imputation and synthetic data: a deterministic relation between two columns (`x2 = log(x1)`) imputes one from the other and constrains a generator.
14. Explanations: the closed form of each effect (`0.7 * log(x2)`) is the partial dependence in words; the interaction pairs are the pairs a report should show.

Order of value for the next steps: 6, 7, 8 (cheap, measurable by the MAE of a booster with and without), then 1 (speed), then 4 with the target-side port.
