# Research tt_a2: model-free recognition of a useful unary transform of a FEATURE (2026-11-23)

Code: `src/mlframe/feature_selection/_benchmarks/nbins_strategy/tt_a2_feature_transform.py` (experiment), `tt_a2_mrmr_check.py` (MRMR probe). Raw logs: `audits/2026-11-23/tt_a2_run.txt`, `tt_a2_mrmr.txt`.
Setup: 20000 rows (10000 train for criteria and judge fit, 10000 held out), 7 seeds, CPU. y = g(x) + 0.5 z + 0.7 noise, z a second linear feature.
Candidates: id, log, sqrt, 1/x (only if x>0), (x-med)^2, |x-med|, tanh((x-med)/IQR), cbrt. Judge: held-out R2 of ridge on [z, f(x)]; GAM ceiling = ridge on [z, cubic splines of x].

## Judge (mean held-out R2 over 7 seeds)
| fixture | id | best transform (7/7 seeds) | R2 best | GAM ceiling |
|---|---|---|---|---|
| log (lognormal x) | 0.476 | log | 0.713 | 0.711 |
| recip | 0.428 | recip | 0.655 | 0.654 |
| sqrt | 0.561 | sqrt | 0.595 | 0.594 |
| square | 0.090 | square | 0.816 | 0.812 |
| abs | 0.222 | abs | 0.548 | 0.548 |
| sigmoid (tanh 2x) | 0.673 | cbrt (not tanh) | 0.853 | 0.875 |
| ctl_linear | 0.713 | id | 0.713 | 0.713 |
| ctl_lognormal_linear | 0.731 | id | 0.731 | 0.718 |
| ctl_noise (lognormal x, no effect) | 0.331 | arbitrary (R2 equal) | 0.331 | 0.330 |

## Criteria: correct decisions out of 7 seeds
Correct = (a transform with judge gain > 0.02 R2 exists: accepted and the pick is within 0.02 R2 of the judge best) or (no gain exists: nothing accepted). Acceptance = best-candidate gain over identity > 95th percentile of the permutation null of max_f(crit_f - crit_id) (25 permutations; the conditional form permutes the residual of y given z).
| criterion | log | recip | sqrt | square | abs | sigmoid | ctl_linear | ctl_lognormal_linear | ctl_noise |
|---|---|---|---|---|---|---|---|---|---|
| Pearson info of f(x) vs y, marginal | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 |
| Pearson info, conditional on z (y minus OLS on z) | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 |
| Pearson vs rank-gauss y, marginal | 7 | 7 | 7 | 0 | 7 | 7 | 7 | 0 | 7 |
| equal-width MI (16 bins), marginal | 4 | 7 | 6 | 0 | 5 | 7 | 0 | 0 | 5 |
| equal-width MI, conditional | 5 | 7 | 7 | 0 | 7 | 7 | 0 | 0 | 5 |
| distance correlation (600-row subsample), marginal | 6 | 6 | 0 | 7 | 4 | 7 | 7 | 7 | 6 |
| distance correlation, conditional | 7 | 7 | 0 | 7 | 6 | 7 | 7 | 7 | 5 |

Cost per candidate on 10000 rows (Pearson + rank-gauss + equal-width MI + dcor on 600 rows, both modes together): 54-97 ms; Pearson alone is well below 1 ms (not timed separately: not run). The permutation null multiplies by the number of permutations.

## Findings
1. Gaussian-copula / Pearson information of f(x) vs RAW y (-0.5 log(1-rho^2)), calibrated by the permutation null of max_f(crit_f - crit_id), decided correctly 63/63 times including all three controls (false-positive rate 0/21 on controls, marginal and conditional). It is not a trained model: one correlation per candidate. Caveat: for one feature it equals the R2 of a one-variable linear fit, so it is the judge in closed form, not an independent witness. Its agreement with the ridge judge is therefore near-tautological; what is validated is the null calibration and the candidate ordering.
2. The conditional form (correlate f(x) with y minus OLS on the already-selected linear part) also scored 63/63; with an orthogonal z the marginal and conditional forms coincide, so this fixture does not separate them. Not run: z correlated with x.
3. Rank-gauss y breaks the criterion where the raw-scale linear fit is what matters: it prefers log(x) for y linear in a lognormal x (7/7 wrong on ctl_lognormal_linear) and misses the square (0/7: the rank-gauss of a chi-square-like y is not linear in x^2 either). Use raw y (or a target transform chosen by the target-side method), not rank-gauss y.
4. Equal-width MI is unreliable as a judge: false positive on ctl_linear 7/7 and ctl_lognormal_linear 7/7, 0/7 on square. It is a proposer at best.
5. Distance correlation is monotone-sensitive but not linearity-sensitive: 0/7 on sqrt, 4-6/7 on log/abs and about 5/7 on noise; also O(n^2). Not recommended.
6. Rank-binned MI cannot see any of this (invariant by construction; not run as its own column because it is constant over monotone candidates). The gap I_rank(x;y) minus linear info of f(x) was not turned into a separate decision rule; its argmax coincides with finding 1.
7. Sigmoid fixture: the judge preferred cbrt over tanh; criteria that accepted a transform matched the judge within 0.02 R2 (Pearson 7/7). The GAM ceiling (0.875) still exceeds the best single unary transform (0.853), so a gap remains that no single candidate closes.

## What MRMR does today (fe_max_steps=1, 20000 rows, 2 seeds each; logs in tt_a2_mrmr.txt)
It already GENERATES the right candidates: `log_abs(x)` (mi_greedy) on the log fixture in both seeds, `x__He2` / `x__sqa_T2` (Hermite / squared basis) on the square fixture, plus `x__relu_*` hinge columns. What is missing is the ranking and acceptance: mrmr_gain is NaN for `log_abs(x)` and the selected extras differ per seed (`(x__add__z)`, `sir__x_z_n0_n1_n2_n3__dir1`, `x__modcos_365` for the square fixture). The MI-on-bins score cannot say that log(x) is the transform that makes the relation linear, so the transform is carried or dropped without a linear-usability test. Not tested: recip, sqrt fixtures through MRMR, more than 2 seeds.

## Algorithm to add to MRMR (feature-side linear-usability gate)
For each numeric selected feature x_j and each candidate f in the already generated unary set (domain-valid only):
1. Residualise: r = y - OLS(y ~ [1, currently selected linear columns without x_j]) on a fit half (classification: use the centred 0/1 indicator or the logit-scale residual; not run).
2. Score s_f = rho(f(x_j), r)^2 on the fit half (Pearson; optionally capped by a winsorised f(x_j) at 0.5 / 99.5 percentiles, not tested).
3. Null: B = 50-100 permutations of r, T_b = max_f s_f(perm) - s_id(perm); threshold = 95th percentile of T_b (Bonferroni-free, it already maximises over f).
4. Accept f* = argmax s_f if s_f* - s_id > threshold; confirm on the held-out half with the same sign and gain > the null threshold (cheap replication; replaces a trained-model check). Output f* as a `linear_usable_transform_` recipe and rank it by 0.5 log((1-s_id)/(1-s_f*)) as the CMI-style credit.
5. Cost: one correlation per candidate and permutation; sub-millisecond per candidate for Pearson, so 100 permutations x 8 candidates is well under a second per feature at 10000 rows (extrapolated, not timed).
Only valid for models that consume the feature linearly or additively; tree models are insensitive (not tested here).

## Not run
Classification targets, correlated features z, heteroscedastic noise, fitted-parameter transforms (Box-Cox), the binned MI of the transformed feature with rank bins as a standalone column, timing of Pearson alone, and any GBM judge.
