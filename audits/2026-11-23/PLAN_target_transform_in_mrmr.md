# Plan: searching for a transform of the TARGET inside MRMR, not only of the features

Owner request 2026-11-23: MRMR finds unary and binary transforms of the features that raise the MI with the target; the same should be possible for the target (log, cbrt, ...), when a transformed target is predicted much better than the raw one.

## What is established (measured 2026-11-23, synthetic, 20000 rows, 8 features, 3 seeds, regression only)

1. MI is invariant to a monotone transform of the target under rank-based binning (equal-frequency bins, and MDL splits, which depend on order only): `MI(x; log T) = MI(x; T)` exactly. Under such binning a monotone target transform is invisible; only a non-monotone one (`|T - median|`, a residual `T - beta*u`) changes the MI. The feature side obeys the same rule.
2. Under EQUAL-WIDTH bins the invariance is gone, and the MI sees the transform that linearises the target:

| target | MI (equal-width) picks, 3 of 3 seeds | ridge RMSE on the original scale, raw -> best | GBM |
|---|---|---|---|
| lognormal (log linearises) | log (0.54-0.56 vs 0.02-0.06) | 2.28 / 2.22 / 2.03 -> 1.40 / 1.30 / 1.26 | within noise; log helps 6 percent on 2 seeds, hurts on 1 |
| cube (cbrt linearises) | cbrt (0.59-0.61 vs 0.24-0.37) | 5.01 / 4.65 / 4.73 -> 2.50 / 2.27 / 2.35 | within noise |
| linear (no transform helps) | signed_sqrt (0.606 vs 0.592): a FALSE positive of margin 0.014 | identity is best: 0.30 vs 0.70 | equal |

3. A first run of this experiment shifted the target (`y - min + 1`) and wrongly reported "no transform helps" for the cube case: the shift broke the linearisation. The experiment script `src/mlframe/feature_selection/_benchmarks/nbins_strategy/target_transform_mi.py` has no shift; it is the source of the biz_val test.
4. Not tested: classification (a class label has no scale to transform), tree models other than `HistGradientBoostingRegressor`, more than 3 seeds, heteroscedastic noise, transforms with a fitted parameter (Box-Cox lambda, Yeo-Johnson), the bias of an inverse transform of a mean prediction.

## Design

The MI under equal-width bins is a good PROPOSER and a bad JUDGE (the linear fixture shows a false positive). So:

1. Proposal. A private equal-width binning of the features and of the candidate target `f(T)`, independent of `nbins_strategy` (the default strategy stays whatever it is; it is not changed to uniform for this). For each `f` in a candidate set (identity, log for `T > 0`, cbrt, signed sqrt, square, reciprocal, and a Box-Cox / Yeo-Johnson with a fitted lambda later) the score is the sum (or the top-k sum) of `MI(x_j; f(T))`. Keep `f` whose score exceeds the identity score by a margin that comes from a permutation null of the score difference, not from a fixed number.
2. Acceptance. On held-out rows (the same even / odd split the target mix uses): fit a cheap model on `f(T)`, invert, and compare the error on the ORIGINAL scale with the raw-target model. Accept only if the error falls by a margin from a paired bootstrap of the held-out residuals. The gate decides; the MI only orders the candidates. A model-free alternative to keep in mind: rank correlation of the held-out prediction on the transformed scale.
3. Output. `MRMR.target_transform_` (name, parameters, MI score, held-out error before and after, accepted yes / no) and a note in `fe_provenance_`. MRMR does NOT change the target it was given: the recipe is returned for the training layer (the composite discovery in `training/composite` already applies and inverts target transforms and scores them on the original scale). The selection of features is made against the accepted `f(T)` only when the owner enables that (`target_transform_apply`), default off.
4. Non-monotone candidates. `|T - c|`, `T - beta*u` already exist in `fe_target_mix`; fold them in the same candidate set rather than keeping two mechanisms.
5. Cost. One equal-width binning per candidate and one cheap model per accepted-by-MI candidate; the candidate set is small. Measure with cProfile at 20000 x 50 and 100000 x 100, GPU twin and KTC only after the CPU path is profiled and optimised (sequencing rule).

## Classification analogue (research item)

A class label has no scale. The analogue is the DEFINITION of the classes: where to cut a continuous or ordinal quantity into classes (thresholds), how to merge or split classes, and which binary reading of a multiclass target a model predicts best. Questions: does the choice of threshold change what MRMR selects; can the MI as a function of the threshold find a cut at which a feature set is most informative; is a held-out log loss at that cut an honest judge. Start with an experiment (continuous latent target cut at several quantiles, features with threshold-specific signal), then decide whether it belongs in MRMR or in the training layer.

## Tests

- biz_val, per transform: lognormal -> log accepted, cube -> cbrt accepted, linear -> identity (no false positive), each over at least 5 seeds, ridge and a tree model, with the original-scale error as the asserted quantity (fails if the accepted transform does not lower it).
- unit: the proposer picks the transform with the highest equal-width MI; the gate rejects the signed_sqrt false positive of the linear fixture; a target with `T <= 0` never gets log.
- contract: `target_transform_` is pickled, defaults on legacy pickles, and the feature selection is unchanged when `target_transform_apply` is off.

## Order

1. Extend the experiment to 5+ seeds, LightGBM and CatBoost, heteroscedastic noise (measure before building).
2. Proposer and gate as functions with unit tests; no MRMR wiring.
3. Wire into MRMR as metadata only (`target_transform_`), biz_val tests.
4. Profile, optimise; GPU twin and KTC if the profile justifies.
5. Optional `target_transform_apply`; classification analogue as its own item once step 1 results are in.

## Decision on MRMR-COMPOSITE-BIN-ONCE (2026-11-23)

The child fit of fe_target_mix_candidates=True re-bins its row subset (55 column-edge computations in the main fit, 125 with the child). Not fixed now: the child runs only when candidates are on (default off) and PLAN-05 has not yet shown value. The red test is narrowed to target-independent strategies (quantile_spacing, freedman_diaconis, sturges, uniform) and stays red, without xfail. If PLAN-05 shows value, the fix is to pass the parent column codes sliced by the child rows for target-independent strategies; supervised strategies re-bin, since the child target T = y - beta*u differs.

## PLAN-05 measurement: value of fe_target_mix_candidates=True (2026-11-23, n=6000, p=10, y = 3*x0 + share*x1*x2 + noise, 3 seeds each)

| fixture | candidates OFF R2 | ON R2 | wall OFF / ON | weak term x1*x2 found OFF / ON |
|---|---|---|---|---|
| share 4 %, noise 0.3 | 0.9857 / 0.9858 / 0.9871 | 0.9904 / 0.9858 / 0.9871 | 33.6 / 16.0 / 11.1 s vs 39.7 / 31.4 / 29.0 s | yes / yes in all seeds |
| share 0.8 %, noise 1.0 | 0.8925 / 0.8900 / 0.8951 | identical | 10-27 s vs 10-11 s | no / no in all seeds |

Conclusion: on both fixtures the candidate search adds nothing (one +0.005 R2 in 6 fits, at up to 2.6x wall on the easy one; on the hard one the weak term is below what either mode finds). No value is proven, so no biz_val test asserting value is written; a test would have to assert the absence of a gain. Open: an intermediate share (about 1.5-2 %) and a fixture where the weak term needs the residual after the dominant one (T = y - beta*u) rather than being visible directly. The composite bin-once fix stays deferred.

Third fixture (share 1.8 %, noise 0.6, 3 seeds): R2 0.9594 / 0.9600 / 0.9593 identical with candidates OFF and ON; the weak term is found in both modes; wall 22.9 / 6.4 / 11.5 s OFF vs 26.4 / 18.1 / 23.6 s ON. Across three fixtures and nine seeds the candidate search gave one +0.005 R2 and never changed what is found, at 1.1x-2.8x wall. PLAN-05 verdict: value NOT proven; no value-asserting biz_val test; fe_target_mix_candidates stays default off. Reopen only with a fixture where the weak term needs the residual T = y - beta*u.

## Step 1 result: the extended experiment (2026-11-23, 5 target kinds x 7 seeds, 20000 rows, 8 features, ridge / HistGB / LightGBM / CatBoost, `_benchmarks/nbins_strategy/target_transform_mi_ext.py`)

The proposer (the transform with the largest sum of equal-width-bin MI over the features) against the judge (the RMSE on the ORIGINAL scale of a model fitted on f(y) and inverted):

| target | the MI picks (every seed) | ridge: best / RMSE(picked) over RMSE(identity) | trees: best / ratio |
|---|---|---|---|
| cube | cbrt 7 of 7 | cbrt 7 of 7 / 0.482 (the error is halved) | scattered / 0.98-1.00 |
| lognormal | yeo_johnson 7 of 7 | log 7 of 7 / the picked inverse overflows in all 7 (non-finite) | scattered / 1.05-1.40 (catboost) |
| hetero_lognormal | yeo_johnson 7 of 7 | log 7 of 7 / non-finite in all 7 | scattered / 0.92 (lgbm), 5.2 (catboost, overflow) |
| linear | signed_sqrt 7 of 7 (FALSE positive) | identity or yeo_johnson (lambda about 1) / 2.28 (the error more than doubles) | identity best in 4-6 of 7 / 1.04-1.06 |
| hetero_linear | signed_sqrt 7 of 7 (FALSE positive) | identity or yeo_johnson / 1.41 | identity best / 1.03-1.04 |

Across the 35 cases the MI-picked transform is the best one for ridge in 7, HistGB in 2, LightGBM in 3 and CatBoost in 3. Findings: (1) the gain exists only for a linear (or linear-like) downstream model and only when the target needs a power transform: cbrt for a cubic target halves the ridge RMSE, log for a lognormal target is the ridge optimum; (2) trees do not gain: the best candidate scatters over seeds and the picked transform is within a few percent of identity, so a transform recipe should be offered to linear heads only; (3) the proposer is biased: on a plain linear target it picks signed_sqrt every time (14 of 14 with the heteroscedastic variant) and that costs a ridge model 1.4-2.3 times its error, and on a lognormal target it prefers Yeo-Johnson to log and the inverse of Yeo-Johnson overflows on a heavy right tail (non-finite RMSE in 14 of 14 ridge cases and 3 CatBoost ones); (4) so the held-out gate of the design is mandatory, not an option, and the candidate set should drop an unbounded inverse (or clip the prediction before inverting) and keep log for positive targets. Decision: step 2 (proposer and gate as functions with unit tests) is justified, scoped to linear heads, with the acceptance test on the original scale; the tree heads are out of scope.

## Classification analogue, first measurement (PLAN-04, 2026-11-23, `_benchmarks/nbins_strategy/class_cut_search.py`, 20000 rows, 8 columns, 5 seeds, cuts at the 10, 25, 50, 75 and 90 percent quantiles of a latent score)

Question of the plan: can the MI as a function of the threshold find the cut at which the features are most informative? Answer: NO in this form. (1) The symmetric uncertainty summed over the columns (the normalisation that makes cuts comparable) peaks at the balanced cut q = 0.5 in every seed of the `uniform` fixture (one linear signal) and of the `tail` fixture (a column that matters only above the 90th percentile): its curve is shaped by the class balance, not by where the signal is. (2) The held-out AUC peaks at the extreme cuts (q = 0.1 or 0.9: the classes are far apart), also by construction; the argmax of the SU curve equals the argmax of the AUC curve in 0 of 5 seeds for both logistic regression and gradient boosting on both fixtures (and in 3 of 5 and 2 of 5 on pure noise, by chance). So no single metric compares two cut definitions: they answer different questions, and which cut is "best" is a property of the downstream task (a cost-weighted utility), not of the data. (3) What the cut sweep DOES show is per-column and local: on the `tail` fixture the SU of the tail column x2 relative to x0 goes 0.126 (q 0.1), 0.057, 0.042 (q 0.5), 0.081, 0.233 (q 0.9), a five-fold rise at the cut where it matters, while the mean SU of the noise columns stays at 0.0002 at every cut. Verdict for the plan: drop the "pick the best cut" search; keep a per-cut SU profile per column as a diagnostic of threshold-specific features (a column whose profile is far from flat is a candidate for a tree head or for a feature built on that tail), to be tried as a metadata item (`column_stats_` per cut) only if the owner asks for it; the choice of the cut stays with the task.
