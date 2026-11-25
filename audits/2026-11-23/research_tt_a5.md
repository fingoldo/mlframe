# tt_a5: residual-scale (variance-stabilising) information criterion for target transforms

Scripts: `src/mlframe/feature_selection/_benchmarks/nbins_strategy/tt_a5_resid_scale.py` (main; raw output `audits/2026-11-23/tt_a5_run1.txt`), `tt_a5_cond_mi.py` (output `tt_a5_condmi.txt`). CPU only, 4 seeds, 4000 rows x 6 features (synthetic), 60/40 split. Total run 2m40s.

## Criterion
For candidate f: z = f(y) standardised; 2-fold cross-fitted closed-form additive cubic-spline ridge on the features; residual r, fitted mean m. H(f) = mean over [x_1..x_p, m] of the rank-binned (10 equal-frequency bins) MI(. ; |r|). Choose argmin H. Accept over identity only if H(id) exceeds the 95th percentile of a 20-permutation null (|r| shuffled) AND H(id) - H(best) > 1.645*sqrt(2)*sd(null). Skew of r tested as a secondary signal.

## Results (H = mean per candidate; pick counts over 4 seeds; RMSE ratio = picked / identity, original scale, inverse clipped to train z range)
| data | H identity | H best | pick | best by RMSE (ridge / hgb) | ratio ridge / hgb |
|---|---|---|---|---|---|
| mult. lognormal | 0.047 | log 0.018, YJ 0.017 | YJ 3, log 1 | log 2, sqrt 2 / log1p 2 | 0.995 / 0.989 |
| Poisson counts | 0.044 | sqrt 0.029 | sqrt 3, YJ 1 | log1p 3 / YJ 2, id 2 | 0.970 / 1.008 |
| gamma mult. noise | 0.040 | log 0.018, YJ 0.018 | YJ 2, log 2 | sqrt 3 / log1p 2, sqrt 2 | 1.010 / 1.004 |
| linear homosc. | 0.025 | none lower | identity 4/4 | scatter | 1.000 / 1.000 |
| linear positive homosc. | 0.021 | none lower | identity 4/4 | scatter (YJ 3) | 1.000 / 1.000 |
| cube + homosc. noise | 0.101 | cbrt 0.026 | cbrt 4/4 | cbrt 4/4 both | 0.576 / 0.953 |
| hetero linear (mean linear, scale depends on x) | 0.034 | none lower | identity 4/4 | id 3 / scatter | 1.000 / 1.000 |
| diabetes (n=442) | 0.0235 | cbrt 0.017, sqrt 0.017 | id 2, cbrt 1, sqrt 1 | scatter | 0.999 / 0.999 |
| california (cached, 20640 rows, 2 seeds) | 0.032 | cbrt 0.021 | cbrt 2/2 | sqrt (ridge), identity (hgb) | 1.000 / 1.010 |
Null mean H is 0.017 (n=2400 train rows), so identity at 0.02-0.025 means "no heteroscedasticity".

## Findings
1. The criterion cleanly recognises variance-stabilising transforms: log/YJ for multiplicative noise (H falls 0.047 -> 0.018, to the null level), sqrt for Poisson (0.044 -> 0.029, the best of the set), cbrt for cube. It rejects all transforms on the homoscedastic targets (4/4 identity, no false positive in 8 target-seed cells of linear data) and on the hetero-linear target where x-dependent scale is genuine noise structure no monotone f can remove (H is flat across f, so nothing is accepted: correct).
2. The criterion is a good PROPOSER, not a judge of the original-scale RMSE. On lognormal / gamma / Poisson, the picks give only 0-3 percent RMSE gain on the original scale (and +1 percent for gamma ridge, +0.8 percent for Poisson hgb). Reason: the judge metric is RMSE of the inverted mean prediction (retransformation bias, exp of a mean of log) and the heavy-tail dominates RMSE. Variance stabilisation helps the transformed-scale loss and fit, not necessarily original-scale RMSE; a fair judge for these would be MAE / pinball / log-scale error or a smearing-corrected inverse. Not run.
3. Where it clearly pays: nonlinear-mean + scale structure (cube): ridge RMSE ratio 0.576, hgb 0.953. In this case the linearisation effect is also caught because the additive spline cannot fit the interaction-induced mean error, which shows up as scale dependence on m.
4. Separation of effects: the spline residual fit removes the linearisation effect for additive means only; a non-additive mean (cube of a sum) leaks into the scale statistic and is credited to the transform, which is desirable here. A dependence on the feature caused by true noise scale is the part no monotone transform changes (hetero_linear), so H being flat is the informative outcome.
5. Candidate ties: log vs Yeo-Johnson (and log1p vs sqrt) are within noise of each other (H 0.0178 vs 0.0174). Prefer the cheaper / more interpretable one within one null-sd of the minimum.
6. Skew of the residual as a secondary signal is NOT reliable: argmin |skew| matched the criterion pick on cube only; on diabetes/linear it picks YJ or noise. Not recommended as a gate; use as a tiebreaker at most.
7. Real data: california picks cbrt on H but RMSE shows no gain (hgb +1 percent); diabetes picks are inconsistent across seeds (n=442, threshold 0.0074 is wide, 2 of 4 seeds accept sqrt/cbrt, RMSE ratio 0.999). So on real, mildly skewed targets the criterion proposes a change that does not pay on original-scale RMSE: the final acceptance gate must be a held-out check (as in PLAN step 2). No other regression CSVs were found under D:\Temp or tests/data (only profiler traces and ground-truth test code), nothing was downloaded.
8. Conditional MI (tt_a5_cond_mi.py, 20000 rows): the Gaussian-copula (rank-normal) form of I(f(y); x1 | x0) is EXACTLY invariant to monotone f (lognormal: 0.5840 identity vs 0.5840 log; cube 0.9237 vs 0.9237), so it cannot rank monotone transforms; it only credits new information per feature (copula 0.584 versus unconditional 0.153: the conditioning uncovers x1). The raw-Gaussian form is NOT invariant (lognormal: identity 0.106 vs log 0.585; cube 0.184 vs 0.925), reproduces the linearisation effect (a), and equals the copula value at the right f. So: use the copula form for feature selection (transform-free), the raw form (on f(y)) as the linearity proposer; neither sees the variance effect (b). The residual-scale criterion is the one that sees (b).

## Cost
~25-37 ms per candidate at 2400 train rows, 6 features (cross-fit splines + 7 rank-MIs); 119 ms at 12400 rows x 8 features; the 20-permutation null is paid once for identity (about 20 x 7 cheap MIs). A 6-candidate screen is ~0.2 s at 4000 rows. Rank codes of the features are shared across candidates (computed here per call; cacheable).

## Recipe for MRMR
1. Candidate set by domain: identity, cbrt, Yeo-Johnson (fitted lambda), plus sqrt/log1p for y>=0, log for y>0.
2. Per f: z=f(y) standardised; cross-fitted (2 or 5 fold) additive spline ridge on the (selected or top-k MRMR) features; r, m.
3. H(f) = mean MI_rank(x_j,m ; |r|), k = min(10, n_train/60) equal-frequency bins.
4. Null: shuffle |r| codes B=20-50 times for the identity; sd0. Threshold tau = 1.645*sqrt(2)*sd0 (paired difference of two comparably noisy scores). Propose f* = argmin H iff H(id) > q95(null) and H(id) - H(f*) > tau; among candidates within sd0 of the min, choose the simplest.
5. Accept f* only after the held-out check on the original-scale metric (the criterion alone gave no gain on diabetes/california and ~1 percent on mult. noise targets) and report H before/after in `target_transform_`.
6. Not run: classification, larger p, non-monotone transforms, smearing-corrected inverse, other loss metrics, more than 4 seeds.
