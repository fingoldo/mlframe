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
