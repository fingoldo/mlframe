# Agent 4: safe judge, wide fixtures, oracle (2026-11-23)

Files (all in src/mlframe/feature_selection/_benchmarks/nbins_strategy/): tt_a4_oracle.py (API), tt_a4_run.py (driver), tt_a4_analyze.py, tt_a4_results.json (raw). Full printed analysis: D:\Temp\wt2\tt_a4_an.txt.
Run: 11 fixtures x 7 seeds x 4 models (ridge, hgb, lgbm 4.6, ridge on cubic splines) x up to 10 candidates, n=12000, 8 features, CPU, 1 thread. Rows: 60 % fit, 15 % calibration (smearing residuals), 25 % test, original scale.

## Judge
Prediction on the transformed scale is clipped to the training range of f(y) (+5 % margin; margin dropped when the inverse is not finite, needed for Yeo-Johnson with lambda<0 whose image is bounded).
Conditional mean: `mean_k inv(clip(zhat + r_k))` over 64 calibration-residual quantiles (Duan smearing generalised to any monotone f). Conditional median: `inv(clip(zhat))`.
Which metric: RMSE (smeared) if the user needs the expected value or totals; MAE of the median prediction if a typical value is needed. A transformed model estimates the median, so MAE favours transforms far more than RMSE does (lognormal ridge: log 0.49 vs 0.57 of identity).
Smearing effect on log, RMSE_median/RMSE_smear: 1.001 to 1.037 (largest ridge counts 1.037, ridge_spline lognormal 1.036). Real but small; it does not change any ranking I looked at.

## Oracle (rmse_smear ratio to identity, mean over 7 seeds; best candidate)
| fixture | ridge | hgb | lgbm | ridge_spline |
|---|---|---|---|---|
| lognormal | log .57 | box_cox .96 | box_cox .95 | log .70 |
| hetero_lognormal | box_cox .86 | box_cox .81 | log .79 | box_cox .79 |
| cube | cbrt .45 | cbrt .95 | cbrt .94 | cbrt .58 |
| linear / hetero_linear | identity | identity | identity (YJ .999) | identity |
| exp_linear | yeo_johnson .59 | asinh .99 | asinh .99 | yeo_johnson .74 |
| zero_inflated | identity | identity | identity | identity |
| counts | log1p .95 | asinh 1.00 | identity | log1p .97 |
| neg_skew | asinh .985 | asinh .99 | asinh .98 | asinh .95 |
| bounded01 | logit .94 | logit 1.00 | logit 1.00 | logit .95 |
| heavy_tail | identity | asinh 1.00 | asinh .99 | identity |
Under MAE (median) transforms win more: zero_inflated log1p .92 (ridge), neg_skew yeo_johnson .69 (ridge), hetero_lognormal log .45 (ridge) and .71 (hgb), bounded01 logit .87 (ridge).
Trees gain 0 to 6 % under RMSE (cube, lognormal) and up to 30 % under MAE on lognormal-type targets; linear and spline heads gain 25 to 55 % on cube, lognormal, exp_linear. The earlier "only linear heads, only power targets" holds for RMSE but is too narrow: it also holds for exp-linear and logit-bounded targets, and trees do gain under MAE.

## Existing proposer (equal-width MI sum over identity, cbrt, signed_sqrt, log, yeo_johnson), safe judge
Its picks: cube -> cbrt 7/7; lognormal, hetero_lognormal, exp_linear, zero_inflated, neg_skew -> yeo_johnson 7/7; linear, hetero_linear -> signed_sqrt 7/7; heavy_tail -> cbrt 6/7; counts, bounded01 mixed.
Pick is best of its own set (RMSE): ridge 28/77, hgb 14/77, ridge_spline 27/77, lgbm 15/77. Within 2 % of best: 38, 35, 35, 32 of 77.
Mean RMSE ratio pick/identity (per fixture, ridge): cube .45 (correct), exp_linear .59 (correct), counts .96, zero_inflated .96, bounded01 1.00, neg_skew 1.14, hetero_linear 1.31, linear 1.82, heavy_tail 2.53, lognormal 2.41, hetero_lognormal 12.5. Trees: linear 1.03 to 1.04, heavy_tail 1.18 to 1.20, zero_inflated hgb 1.18 / lgbm 1.31, lognormal 1.5, hetero_lognormal 5.8 to 6.0.
Share of 77 cases worse than identity by >2 %: ridge .49, hgb .52, ridge_spline .55, lgbm .57.

## Distortion by the unsafe inverse
Old judge: pick non-finite in 14/77 (ridge), 12/77 (ridge_spline), 1/77 (lgbm), 0/77 (hgb); its best candidate differs from the safe judge's in 16 to 22 of 77 cases per model. Retransformation bias was minor (see above).
Important correction: the picked Yeo-Johnson failure is NOT only an overflow artefact. YJ fits lambda about -0.8 on lognormal and zero-inflated (checked seed 0), a reciprocal-like transform with a bounded image; even clipped to the training range its inverse is very steep near the bound (lognormal seed 0, ridge: YJ 7.2 vs identity 2.2 vs log1p 1.83). So the overflow was hiding a real defect: the proposer prefers a transform that is bad. Finite-ness only changed inf to 2.4x to 12x worse than identity.

## Verdict
The idea is only partly refuted. MI correctly flags cbrt on cube and (via YJ) exp-linear for linear heads. It fails on: false positives on linear targets (signed_sqrt, 1.8x worse for ridge), preferring YJ over log on positive skewed targets (2.4x to 12x worse), heavy tails, and trees. Its score is also not scale-comparable across candidates (equal-width bins reward compression of the tail). Another proposer must be scored against the oracle, with identity as a mandatory competitor and a margin threshold. Limits: single noise level per fixture, one n (12000), equal-width 10 bins only, no real data, CatBoost not run, Duan smearing uses 64 residual quantiles.

## Oracle API
`import tt_a4_oracle as O; O.oracle(fixture, model, seed=0, n=12000, metric='rmse_smear')` returns [(name, {rmse_smear, rmse_clip, mae_clip, mae_smear, rmse_unsafe})] best first; `O.judge(X, y, model)` for your own data; `O.FIXTURES`, `O.MODELS`, metric 'mae_clip' for the median ground truth. (`oracle_table` is slow and untested; use tt_a4_run.py / tt_a4_results.json.)
