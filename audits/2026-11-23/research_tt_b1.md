# Research tt_b1: model-free recognition of a useful unary transform of a FEATURE for classification (2026-11-24)

Code (all new, in `src/mlframe/feature_selection/_benchmarks/nbins_strategy/`):
- `tt_b1_cls_feature_transform.py`: fixtures, candidates, judges, criteria, nulls (main experiment).
- `tt_b1_report.py`: tables from the stored JSON. `tt_b1_nb_criterion.py`: head-matched Jacobian criterion and the criterion x judge cross table.
- `tt_b1_target_side.py`: monotone transform of a continuous score that defines classes. `tt_b1_cheap_lrt.py`: k-step Newton approximation of the main criterion.
Raw outputs: `audits/2026-11-23/tt_b1_*.txt`. All CPU, OMP 1 thread; 20000 rows per case (10000 rows for criteria and judge fit, 10000 held out), unless said otherwise.

## Setup
Class-conditional generation: class c, one feature x whose class-conditional law is a known transform of a Gaussian shift, plus two Gaussian "other" features z (class-shifted, z-only AUC about 0.72, independent of x given the class).
Fixtures (K = 2 binary, 3, 5 classes; 8 seeds each, K=5 7 seeds): `log` (lognormal class shift; `log_weak` weaker shift, `log_imb` prior 0.15), `recip` (x = 1/u, u Gaussian shift), `sqrt` (x = u^2, u Gaussian; weak curvature) and `sqrt_g` (sqrt(x) gamma),
`square` (classes differ in VARIANCE, equal mean), `square_off` (same, centre at 3), `abs` (|x| shifts), `beta_logit` (Beta(2,4) vs Beta(4,2)), `count_logn` (Poisson-lognormal counts), `heavy_signal` (t3 noise, shifted).
Controls (identity is right or nothing to find): `ctl_linear` (Gaussian shift), `ctl_gamma` (gamma scale shift: logit exactly linear in x though x is skewed), `ctl_pois` (Poisson, logit exactly linear), `ctl_noise` (lognormal, no class effect), `ctl_noise_heavy` (t2, no class effect).
Candidates (domain-checked): id, log, recip, sqrt, log1p, logit, (x-med)^2, |x-med|, tanh((x-med)/IQR), cbrt, asinh((x-med)/IQR); `square_off` adds the uncentred `x*x` (what the MRMR unary table has).
Judges, all held out on `[z, f(x)]`: multinomial logistic regression (LR, primary; log loss in nats per row and macro AUC), LDA, Gaussian NB, a spline-GAM ceiling (10 knots on x), HistGradientBoosting on `[z, x]` and on the LR-best and the non-monotone candidates.
Tiers per case by the LR judge (id minus best log loss): gain > 0.010 nats = a transform exists; none < 0.003; ambiguous in between. Correct = (gain tier: accepted AND the pick costs at most 0.005 nats more than the judge best) or (none tier: not accepted) or (ambiguous: not harmful). Harmful = accepted pick costing more than 0.005 nats over identity. FP on controls = accepted on a `ctl_*` fixture.

Criteria (computed on the 10000 train rows, no predictive model beyond a few Newton steps):
- `lrt`: deviance gain 2(ll1 - ll0) of a one-feature multinomial logit (intercept + slope per non-reference class) on f(x), with the score of a logit fitted on the other features as OFFSET (criterion 1 of the brief).
- `bd`: prior-weighted Bhattacharyya distance of the class-conditional Gaussian fits of f(x) (criterion 2). `eta`: ANOVA / point-biserial information -0.5 log(1-eta^2) of f(x) with the class (criterion 3).
- `resid`: information of f(x) against the class residual y - p(z) (the wave-1 feature recipe transplanted: R^2 of f(x) on the K-1 residual columns).
- `mi_ew`: plug-in MI of f(x) with the class, 16 EQUAL-WIDTH bins (the old proposer); `mi_rank`: same with 16 equal-frequency bins (monotone invariant). (criterion 4)
- Decision for every criterion: best non-identity candidate accepted only if its gain over identity exceeds max(95th percentile of a null of max_f(s_f - s_id), 0) (40 draws, no fixed threshold). Two nulls: `perm` (rows of f(x) permuted against the class, offset kept: the brief's permutation null) and `bs` (parametric bootstrap, classes redrawn from the identity logit on [z, x]: the analogue of the wave-1 residual bootstrap).

## Result 1: what the right transform buys (judge, mean over seeds)
| fixture (K) | LR logloss id | LR best (count of seeds) | gain nats | AUC gain | GAM-ceiling gain | HGB gain from any fitted transform |
|---|---|---|---|---|---|---|
| log (2) | 0.4605 | log 8/8 | 0.0354 | +0.0116 | 0.0349 | 0.0000 |
| log_weak (2) | 0.5463 | log 8/8 | 0.0251 | +0.0143 | 0.0246 | 0.0000 |
| log_imb (2, prior 0.15) | 0.2966 | log 8/8 | 0.0235 | +0.0194 | 0.0228 | 0.0000 |
| recip (2) | 0.4727 | recip 8/8 | 0.0382 | +0.0090 | 0.0378 | 0.0002 |
| square (2) | 0.6164 | square 8/8 | 0.1614 | +0.1375 | 0.1601 | 0.0001 |
| abs (2) | | abs 8/8 | 0.3215 | +0.2280 | 0.3309 | 0.0002 |
| heavy_signal (2) | 0.5609 | tanh 8/8 | 0.0262 | +0.0133 | 0.0320 | 0.0000 |
| sqrt weak / sqrt_g (2) | | sqrt 8/8 | 0.0037 / 0.0051 | +0.0008 / +0.0030 | 0.0031 / 0.0048 | 0.0000 |
| beta_logit (2 / 3) | | logit 8/8 | 0.0014 / 0.0035 | +0.0004 / +0.0010 | 0.0010 / 0.0024 | 0.0000 |
| count_logn (2 / 3) | | sqrt 7/8 / log1p 5/8 | 0.0076 / 0.0136 | +0.0030 / +0.0034 | 0.0084 / 0.0143 | 0.0000 |
| log (3) / log (5) | | log 8/8, 7/7 | 0.0418 / 0.0453 | +0.0116 / +0.0093 | 0.0408 / 0.0433 | 0.0000 |
| recip (3), abs (3) | | recip 8/8, abs 8/8 | 0.0359, 0.6807 | +0.0084, +0.2653 | 0.0351, 0.6861 | 0.0003, 0.0014 |
| square (3) / square (5) | | square 8/8, 7/7 | 0.1762 / 0.1440 | +0.1010 / +0.0629 | 0.1752 / 0.1412 | 0.0011, 0.0001 |
| all controls (K=2,3,5) | | none | 0.0000 to 0.0002 | 0.0000 to 0.0001 | negative | 0.0000 to 0.0010 |
- Linear heads gain; trees do not: the best gain HistGradientBoosting gets from any fitted transform, including the non-monotone square and abs, is at most 0.0014 nats (verified on every fixture, K=2,3,5; HGB fitted on 3 seeds per fixture). HGB on raw x trails LR-with-the-right-transform by 0.005 to 0.010 nats at K=2 (by 0.009 even on the controls, where LR is exact, so that deficit is HGB variance and not a missing transform; on `abs` HGB is 0.001 better), and by 0.016 to 0.043 at K=3 and K=5 (100 iterations, no tuning).
- Replacing x by f(x) and appending f(x) to the raw x give the same loss within 0.0001 nats on 15 of 17 binary fixtures (append better by 0.0004 on count_logn, 0.0038 on heavy_signal). The best single transform reaches the spline-GAM ceiling within 0.0013 nats except heavy_signal (0.026 vs 0.032) and abs (0.3215 vs 0.3309).
- A monotone transform of one feature mostly changes calibration and the way the logit combines with the other features, not the single-feature ranking: AUC gains are 0.001 to 0.02 where the log-loss gains are 0.004 to 0.04 (monotone fixtures). Non-monotone fixtures (square, abs) are different in kind: log-loss gain 0.14 to 0.68, AUC +0.06 to +0.27.

## Result 2: which criterion recognises the judge-best transform (LR judge, 244 cases: 126 gain, 28 ambiguous, 90 none, of which 78 are control cases)
| criterion | null | hits on 126 gain cases | FP on 78 control cases | harmful accepts (of 244) | cost per candidate (ms, n=10000) |
|---|---|---|---|---|---|
| **lrt** (deviance gain, offset) | perm | **126/126** | 6/78 (7.7%) | 0 | 28 (K=2), 51 to 58 (K=3), 90 (K=5) |
| **lrt** | bs | **126/126** | **3/78 (3.8%)** | 0 | same |
| eta (ANOVA/point-biserial info) | perm / bs | 75/126 / 75/126 | 51/78 / 5/78 | 11 / 0 | 0.3 |
| resid (info vs class residual) | perm / bs | 76/126 / 76/126 | 52/78 / 6/78 | 11 / 0 | 3 to 11 |
| bd (Bhattacharyya) | perm / bs | 49/126 / 51/126 | 47/78 / 2/78 | 145 / 22 | 0.3 |
| mi_ew (equal-width MI, old proposer) | perm / bs | 70/126 / 70/126 | 43/78 / 7/78 | 2 / 1 | 4 |
| mi_rank (equal-frequency MI) | perm / bs | 31/126 / 32/126 | 5/78 / 5/78 | 0 / 3 | 4 |
By K, lrt: K=2 64/64 hits, FP 3/40 (perm) 2/40 (bs); K=3 48/48, FP 2/24, 1/24; K=5 14/14, FP 1/14, 0/14. Small sample (2000 train rows, 8 seeds, fixtures log_weak, square, three controls): lrt 16/16 hits, FP 1/24 under both nulls; eta/resid 10/16 hits.
Dedicated false-positive run (K=2, 30 seeds x ctl_noise, ctl_linear, ctl_gamma = 90 control cases, 20000 rows, 40 draws; the first 8 seeds overlap the table above, so the two are not pooled; no HGB): lrt FP 3/90 (perm; 95% interval 0.7% to 9.4%) and 2/90 (bs; 0.3% to 7.8%), the accepted picks asinh / tanh / square, none harmful; eta and resid 62/90 (perm) and 4/90 (bs); bd 60/90 and 6/90; mi_ew 58/90 and 11/90; mi_rank 3/90 and 5/90. The 78-case table above gives 6/78 (2.9% to 16.0%) and 3/78 (0.8% to 10.8%) for lrt, so the calibration is near the nominal 5% with both nulls. Log: `tt_b1_fpr30_summary.txt`.
Which transform each criterion proposes (accepted picks, perm null):
- lrt picks exactly the judge best (log, recip, square, abs, tanh) on every gain fixture, K=2,3,5; on count_logn (K=3) it picks sqrt / log1p, within 0.005 nats of the best. On the 40 non-control cases of the ambiguous / none tiers (real but practically negligible effects: sqrt, sqrt_g, beta_logit, count_logn at K=2; beta_logit at K=3) it accepted all 40 and picked the judge-best name in 33, none harmful. It is a significance test, not an effect-size test; the per-row gain (LRT / 2n) is the effect size and is what to report.
- eta and resid prefer saturating transforms: tanh / asinh for log and recip fixtures (hits 4/8 and 0/8 on log (K=2) and recip (K=2)), `abs` instead of `square` on the variance-shift fixtures (0/8 hits on square, square_off, K=2,3,5). Reason: they maximise a Gaussian mean separation, not the logit linearity; |x| separates class means more cleanly than x^2 under a Gaussian fit but costs 0.0079 (K=2) and 0.0096 (K=3) nats against the square under LR.
- bd (Bhattacharyya) is unusable: it has no Jacobian, rewards any transform that inflates the variance contrast, and was harmful in 145 of 244 cases (perm null).
- mi_ew (the old biased proposer) proposes log1p / tanh / asinh on nearly every fixture, accepted 126/126 gain cases but the pick is within tolerance only in 70; with the perm null it is accepted on 43/78 controls (the null is built from destroyed signal; the bs null brings it to 7/78). It cannot pick `square` (0/8 hits on square, K=2,3,5) because equal-width bins of a symmetric variance shift show it only weakly.
- mi_rank (equal-frequency bins) is constant over monotone candidates: 0 hits on every monotone gain fixture (log, log_weak, log_imb, recip, heavy_signal, count_logn at K=3) and on abs. Its 31 hits come only from the non-monotone `square` / `square_off` (8/8, 8/8, 8/8, 7/7 at K=2, 2, 3, 5) via binning resolution, and it accepts 5/78 controls from the same binning artefact.
- The permutation null is the wrong null for criteria that are marginal or that carry a signal in the identity column (eta, resid, bd, mi_ew: FP 43 to 52 of 78 controls) because under permuted classes there is no signal at all, so the observed small differences among candidates exceed the null. The parametric bootstrap from the identity logit repairs them (5 to 7 of 78) and costs one extra offset fit per draw. For `lrt` both nulls work (6/78 vs 3/78); the bootstrap is the safer one.
- Uncalibrated argmax: on the 90 none-tier cases every criterion returns a non-identity pick in 90 of 90 (max over 10 candidates of noise); harmful (over 0.005 nats worse than identity) for bd 51, mi_rank 26, eta 5, resid 5, mi_ew 3, lrt 0. On the 78 controls the same: non-identity 78/78, harmful bd 39, mi_rank 17, eta 3, resid 3, mi_ew 3, lrt 0. A null is mandatory for every criterion; with the null lrt has 0 harmful accepts in 244 cases.
- In-sample deviance gain / (2 n) equals the held-out log-loss gain of the picked candidate on the gain cases (means 0.1228 vs 0.1230 nats per row, correlation 0.999): the criterion is in the units of the judge. Caveat: for one feature `lrt` is the in-sample version of the LR judge, so agreement on LR is close to tautological, as the wave-1 report already said for the feature criterion; what is validated is the null calibration, the candidate ordering, the 0 harmful accepts and the behaviour on the other heads below.

## Result 3: the best transform depends on the head (LR vs LDA vs Gaussian NB), mean held-out log loss
| fixture (K=2) | head | id | best candidate (loss) | note |
|---|---|---|---|---|
| square | LR | 0.6164 | square 0.4550 | abs 0.4629 |
| square | LDA | 0.6164 | abs 0.4735 | square 0.5189 (LDA cannot see variance, abs gives a mean shift) |
| square | Gaussian NB | 0.4551 | identity | NB models the class variance itself; every transform is worse (square 0.6302) |
| count_logn | LR | 0.5188 | sqrt 0.5113 | gain 0.0075 |
| count_logn | LDA / NB | 0.5329 / 0.6075 | sqrt 0.5121 (log1p 0.5122) / log1p 0.5107 | gain 0.021 / 0.097 |
| sqrt_g | LR / NB | 0.5362 / 0.6669 | sqrt 0.5311 / log1p 0.5357 | NB gain 0.131 vs LR 0.005 |
Criterion x judge (all fixtures, bootstrap-null columns, hits / FP on none-tier / harmful):
- LR judge (126 gain cases): lrt 126/126, FP 15/90, harmful 0; eta_bs 75/126, FP 6/90.
- LDA judge (142 gain cases): lrt 106/142, FP 12/85; eta_bs 110/142, FP 5/85, harmful 2; resid_bs 111/142, FP 6/85, harmful 2 (here the simple closed-form information with the bootstrap null is the best match; lrt misses the square fixtures because LDA prefers abs).
- Gaussian NB judge (159 gain cases): lrt 100/159, FP 38/64, harmful 32; eta_bs 71/159, FP 34/64, harmful 33; ccll_l 106/159, FP 2/64, harmful 2.
`ccll_l` is the class-conditional Gaussian log-likelihood of f(x) with the log-Jacobian (the classification analogue of the wave-1 Jacobian LL, pooled variance). It matches the NB head (above) and recognises log / recip with every head, but its permutation null is dominated by the marginal non-Gaussianity of a skewed feature (null threshold 1.05 nats per row on the lognormal fixture against a gain of 1.19), so it has 17 to 21 FPs on the LR and LDA none-tiers (90 and 85 cases), misses square and heavy_signal, and the per-class-variance form `ccll_q` reaches only 16 to 17 hits. Treat it as partially validated (NB head only); a better null (bootstrap from the identity class-conditional Gaussian) was not built.
So: the criterion must match the downstream head. `lrt` is the right one for logistic-type heads, a Gaussian information with the bootstrap null for LDA, and a Jacobian class-conditional likelihood for NB; for tree heads nothing should be offered.

## Result 4: costs and cheap approximations
- Measured on a CPU shared with four other jobs, numpy, n=10000, 10 to 11 candidates: lrt 28 ms per candidate (K=2), 51 to 58 (K=3), 90 (K=5); eta+bd 0.3 ms, resid 3 to 11 ms, MI 4 ms. A full decision with B=40 draws costs 40 x candidates x lrt, about 10 s for K=2 on this machine. Not run on an idle machine; the MRMR file `filters/_fe_accuracy_gate_newton.py` states that a numba Newton fit is about 20x cheaper than scikit-learn, which I did not measure.
- Fewer Newton steps (72 cases, 616 candidates): one step 11.5 ms, same argmax as the full criterion in 46/72 (within 2% of the top value in 54/72), Spearman 0.915; two steps 18 ms, 60/72; three steps 26.7 ms, 66/72, Spearman 0.991; full 33 ms. In numpy a truncated Newton buys at most 20% at three steps, not an order of magnitude; the saving has to come from a compiled kernel or from fewer candidates.
- A batched Newton over all candidates at once (one set of numpy calls for 10 candidates) was written and checked on two datasets (equal to the loop to 5e-10), but it was slower in numpy (0.70 s vs 0.20 s on `square` K=2, 2.1 s vs 0.87 s on `log` K=3, one loaded CPU) and was dropped (not in the code).

## Result 5: off-centre variance shift; MRMR's unary table
`square_off` (x = 3 + s_c e): LR log loss id 0.6164, centred square 0.4550, |x-med| 0.4629, the UNCENTRED `x*x` that the MRMR unary table (`sqr`) produces 0.6048 (gain 0.012 of the 0.161 available). `lrt` picks the centred square 8/8. MRMR's `sqr` / `abs` need centring (median) before a variance-shift feature can be credited; whether the median is the right centre when the class variances are centred elsewhere was not tested (a small grid of centres scored by `lrt` is the obvious extension).

## Result 6: target-side analogue for classification (tt_b1_target_side.py, 7 seeds, 20000 rows, K=3, x1 lognormal, score s = 0.9 log x1 + 0.5 z1 + 0.35 z2 + noise)
There is no scale to transform in a class label: relabelling classes changes nothing for MI or for multinomial LR, and the analogue of a target transform is the CHOICE OF CUT POINTS of a continuous score, which is a class definition (the repository's `class_cut_search.py` is that direction; out of scope here). Monotone transforms g (identity, exp, cube, cbrt, sigmoid, -exp(-s)) of the score:
- quantile terciles of g(s) (Q) and fixed cut values mapped through g (T): labels identical to the identity labelling in 7/7 seeds for all six g (array_equal), held-out LR log loss 0.5940, AUC 0.8911 for every g, `lrt` accepts `log` for x1 in 7/7 for every g. So a monotone transform of the score does not matter whenever the cut is rank-based or maps the same event, as expected.
- equal-width bins of g(s) (E) and mean +- 1 sd cuts of g(s) (Z): the labels change with g (agreement with the identity labelling 0.09 to 0.85, smallest class share 0.000 to 0.244; for exp and -exp(-s) a class has under 30 train rows in all 7 seeds, so the case is undefined). The feature-side pick stayed `log` in every defined cell (E: 7/7 for id, cbrt, sigmoid, 1/1 cube; Z: 7/7 for id, cube, cbrt, sigmoid), but log loss and AUC move (E: 0.2933 to 0.5943), which only reflects a different classification problem.
Conclusion: transform of the score is a non-issue for the rank/quantile cut; for value-based cuts the scale of the score is part of the class definition, not something to be recognised.

## Verdict
1. For classification the feature-side recipe works, with a different criterion from the one that works for regression features: the deviance gain of a one-feature logit with the other features' score as offset, calibrated by a null of max_f(s_f - s_id). It recognised the judge-best transform in 126 of 126 gain cases (16 fixture-by-K combinations, 7 to 8 seeds each), accepted nothing harmful in 244 cases, and had a false-positive rate of 3/78 (bootstrap null) to 6/78 (permutation null) on controls.
2. The wave-1 feature criterion (Pearson information of f(x) against the residual, here `resid`, and ANOVA `eta`) transplants only with the bootstrap null (FP 5 to 6 of 78) and then recognises the judge-best transform in just 76/126 cases (it prefers tanh/asinh/abs); it is the right choice for LDA-type heads (110 to 111/142 hits, FP 5 to 6/85) and costs 0.3 to 11 ms per candidate.
3. Equal-width MI (the old proposer) is not salvageable as a selector: 70/126 hits, 43/78 FP with the permutation null, argmax non-identity on every none case; rank-binned MI cannot pick monotone transforms at all (0 hits there).
4. Trees: nothing to gain (at most 0.0014 nats), as predicted. LR with the right transform sits at the spline-GAM ceiling (within 0.0013 nats except heavy_signal and abs); HGB on raw x trails it by 0.005 to 0.010 nats at K=2, which is also its deficit on the controls.
5. The best transform is head specific (LDA wants abs where LR wants square, Gaussian NB wants nothing on a variance shift and a log1p on counts). The acceptance step has to know which head is downstream.
6. Class-label side: nothing to recognise; monotone transforms of a score are inert under rank cuts.

## Algorithm for MRMR's classification path
Applies when the downstream model is a logistic-type or linear head (default); for tree heads skip; for an LDA head swap `lrt` for `eta`/`resid` with the same null; for NB swap for the Jacobian likelihood (partially validated).
1. For a numeric column x_j (selected, or top-k by rank MI) form the candidate set from the unary table of `_unary_op_table.py` with domain checks, adding median-centred square and abs, tanh and asinh of (x-med)/IQR, logit for (0,1), log1p for x >= 0; drop non-finite or constant columns; standardise each.
2. Base: multinomial (binary: `fit_logistic_ridge_newton`) logit on [1, selected columns without x_j] on a row subsample (the existing `_fe_subsample` budget, 10000 rows was enough; 2000 train rows still gave 16/16); keep logits L0 (n x K) and ll0.
3. For every candidate c: Newton for intercept + slope per non-reference class with L0 as offset; LRT_c = 2 (ll1_c - ll0). Keep LRT_id.
4. Null, B = 40 draws: redraw classes from the identity logit fitted on [others, x_j], refit the base, recompute all LRT_c; T_b = max over c != id of (LRT_c - LRT_id); tau = max(95th percentile of T_b, 0). The permutation of f(x) rows is the cheaper alternative (no base refit, 6/78 vs 3/78 control FPs).
5. Accept the argmax c* iff LRT_c* - LRT_id > tau. Record `(op, centre, delta_ll_per_row = (LRT_c* - LRT_id)/(2 n), tau)` as the recipe, and use delta_ll_per_row as the CMI-style credit (it matched the held-out log-loss gain, 0.1228 vs 0.1230); this is the missing "linear usability" credit that today leaves the gain NaN for engineered columns.
6. Keep raw x and append f(x) (append equals replace within 0.0001 nats on 15 of 17 fixtures; better on heavy tails).
7. Cost: 28 ms per candidate at K=2 in numpy on a loaded machine, so about 10 s per feature for 10 candidates and B=40: run only on columns that survive the cheap screen (for example `resid`/`eta` with the bootstrap null at 0.3 to 11 ms per candidate, which accepted all 126 gain cases) or lower B; a numba kernel and B=20 are untested.

## Not run, limits
Synthetic only; no real data. The effect sizes, the two informative `z` and the candidate list are my choices. `x` is independent of `z` given the class (correlated features were not tested); one feature per decision; K <= 5; n = 20000 (and one small-sample run at 2000 train rows); class prior imbalance only 0.15; no missing values. Not run: linear SVM, LightGBM, CatBoost, the existing `_fe_accuracy_gate` probe (AUC for binary, hit rate for multiclass) as a baseline on these fixtures, a centre search for the square, a better null for the Jacobian criterion, the numba cost, more than K=5, class-cut search. The `sqrt` fixture (x = u^2, u Gaussian) turned out too weak (gain 0.004 nats); `sqrt_g` was added after the first runs (gain 0.005, still small), so no fixture with a large sqrt gain exists. HGB was fitted on 3 seeds per fixture only. The "judge best" for count_logn and beta_logit is within noise of several candidates (tolerance 0.005 nats).
