# research_tt_b2: distribution zoo and scale of the problem (wave 2, direction B2)

Code (all new, in `src/mlframe/feature_selection/_benchmarks/nbins_strategy/`): `tt_b2_core.py` (generators, candidate sets, judge), `tt_b2_crit.py` (wave-1 criteria, vectorised re-implementation), `tt_b2_fix.py` (fixes), `tt_b2_run_target.py`, `tt_b2_run_feat.py`, `tt_b2_run_p.py`, `tt_b2_calib.py`, `tt_b2_feat_stress.py`, `tt_b2_analyze.py`, `tt_b2_report_tables.py`. Raw results: `audits/2026-11-23/tt_b2_*.jsonl` (+ `.log`, `tt_b2_*table*.txt|md`). CPU only, 1 BLAS thread, no existing file edited, nothing committed.

## 0. Verdict in six lines

1. Wave-1 FEATURE criterion (Pearson information of f(x) with the residual of y, permutation null of the max over candidates) holds up across the whole feature zoo: 400 runs, 288 hits, 90 true negatives, 22 benign non-identity picks (pick RMSE ratio <= 1.0 every time), mean regret 1.0000, identical for RMSE and MAE judges, down to n = 300. Its type-I rate on true nulls is 3/600 at n = 300 and 0/1200 at n = 1000 and 5000. It has two real defects: (i) a correlated covariate makes it pick a harmful transform in 38 of 40 runs (fixed by partial correlation, 0/40), (ii) with many features the per-feature null gives a family-wise error rate of 0.88-1.00 at p = 50-200 (fixed by one max over features x candidates: 0.02-0.10).
2. Wave-1 TARGET recipe (Jacobian LL + sign-flip null + held-out original-scale gate, "W1") does NOT generalise over the zoo: 57 of 93 helpful runs found (CV below: 87 of 93), 15 more picked a wrong transform, 21 missed. The gate is what keeps it safe (raw LL: 13 of 196 runs worse than identity by >2%, worst 3.2x; W1: none), but W1 is blind to count data, boundary-clipped bounded targets, and picks the Gaussian-density-best transform instead of the RMSE-best one under skewed noise.
3. Each W1 failure has a measured cause and a tested fix (section 2). The most general fix is a cross-fitted comparison on the original scale with a paired sign-flip max-T test ("CV"): 87/93 hits, 0 harmful picks, regret 1.003, at n = 5000. It is a closed-form linear-head surrogate of the judge, not an information-theoretic quantity.
4. CV is weaker than LL at small n on strongly skewed targets (lognormal/mult: CV 2/5 at n = 300 and 1000, LL pipeline 5/5). The union rule HX (accept if CV or the LL pipeline accepts; take the lowest cross-fitted error of the two proposals) is the best at n <= 1000 (18/24 and 21/25 hits, regret 1.012 and 1.010).
5. The null stages are calibrated at every n tested (pooled true-null type-I: LL null 4.3 / 2.7 / 3.0 percent at n = 300 / 1000 / 5000; W1 4.3 / 1.0 / 0.7; CV 0.7 / 0 / 0). The raw LL argmax is not usable as an acceptance rule (34-44 percent false picks on true nulls, 100/100 on heteroscedastic normal).
6. Not measured: n < 300, n/(p+1) < 5, trees, classification, correlated target features, real data.

## 1. Setup (what exactly was run)

**Target zoo** (`tt_b2_core.gen_target`): 14 families x {homoscedastic additive noise, multiplicative noise} = 28 cells: normal, lognormal, gamma, Weibull, Pareto (alpha 2.5), neg-skew (4 - exp), bimodal mixture, bounded (0,1) (clipped normal / Beta), bounded [0,10] (censored normal / 10*Beta), Poisson, negative binomial, zero-inflated (35 % exact zeros), ordinal (5 levels), 1 %-contaminated. y depends on a standardised index of 3 informative standard-normal features, p - 3 features are noise. n = 5000, p = 8, 7 seeds per cell = 196 runs for the zoo; B = 99 sign-flip draws.

**Feature zoo** (`gen_feature_fixture`): 9 feature laws (normal, lognormal, t3, uniform(0.25,4), Beta(3,2), 5-level integer, zero-inflated, bimodal, mixed-sign skewed) x 6 links (none, log, sqrt, square, 1/x, abs; only domain-valid pairs = 40 cells), y = amp * standardised g(x) + 0.5 z + 0.7 e.

**Wave-1 criteria, re-implemented vectorised** (no import of the a-files; same formulas): target "W1" = held-out half-B Jacobian LL, sign-flip residual bootstrap from the identity OLS fit (95th percentile of max over candidates, here the equivalent (1 + count)/(B + 1) <= 0.05 rule), Yeo-Johnson with |lambda - 1| < 0.1 collapsed, then "held-out original-scale ridge RMSE not worse than identity" (clipped inverse + 64-quantile smearing). Candidate set exactly as in the synthesis: identity, cbrt, signed_sqrt, signed_square, asinh (scaled by sd), Yeo-Johnson (lambda in [-1,3], stride subsample 4000), log and -1/y for min(y) > 0, logit for 0 < y < 1. Feature = a2 set and a2 permutation test (B = 100).

**Judge** (`judge_target`): same protocol as `tt_a4_oracle.judge` (ridge(1) on f(y), clipped to the training range of f(y) +5 %, 64-quantile smearing, RMSE of the smeared mean; MAE of the clipped median prediction), with two deviations: (a) scored on 50000 FRESH rows from the same generator instead of a hold-out of the same sample, so the label "helps" is not noisy at n = 300; (b) zero clipping margin for Yeo-Johnson with lambda outside [0, 2] (see failure mode T6). Validated against `tt_a4_oracle.judge` on four a4 fixtures: ratios agree within 0.05 except Yeo-Johnson, whose lambda estimator differs (lognormal log 0.537 vs 0.583, cube cbrt 0.442 vs 0.475, linear signed_sqrt 1.867 vs 1.840, bounded01 logit 0.933 vs 0.937).

**Outcome definitions** (per run, ratio = metric(candidate)/metric(identity) on the fresh rows): a transform *helps* if the best non-identity candidate has ratio <= 0.98. If it helps: HIT = pick within 0.02 of the best ratio, WRONG = other non-identity pick, MISS = identity. If nothing helps: TN = identity, FP = any non-identity pick, HARM = FP with ratio > 1.02. Regret = ratio(pick) / min(ratio(best), 1).

## 2. Axis (a): target distributions

### 2.1 Stage by stage over the whole zoo (196 runs, 93 helpful by RMSE, 122 helpful by MAE)

| stage | hit/helps | non-identity picks when nothing helps | of which worse than identity by >0.5% | worse by >2% | mean pick/identity RMSE | worst | regret |
|---|---|---|---|---|---|---|---|
| LL argmax (Jacobian, YJ near 1 collapsed) | 69/93 | 89/103 | 40 | 12 | 0.9706 | 3.211 | 1.0328 |
| LL + sign-flip null | 59/93 | 53/103 | 26 | 12 | 0.9736 | 3.211 | 1.0362 |
| LL + null + held-out gate (wave-1 recipe, W1) | 57/93 | 17/103 | 0 | 0 | 0.9499 | 1.000 | 1.0123 |
| same with paired sign-flip gate (G2p) | 54/93 | 10/103 | 0 | 0 | 0.9512 | 1.000 | 1.0137 |
| Huber-robust LL + null + paired gate (RG) | 60/93 | 22/103 | 0 | 0 | 0.9501 | 1.000 | 1.0125 |
| extended set (+log1p, sqrt) + paired gate (EG) | 56/93 | 12/103 | 0 | 0 | 0.9504 | 1.000 | 1.0136 |
| W1 + support abstention (S) | 48/93 | 8/103 | 0 | 0 | 0.9516 | 1.000 | 1.0140 |
| cross-fitted original-scale max-T (CV) | 87/93 | 31/103 | 0 | 0 | 0.9415 | 1.000 | 1.0032 |
| CV, else W1 (H) | 87/93 | 32/103 | 0 | 0 | 0.9414 | 1.000 | 1.0031 |

Oracle (best candidate per run) mean ratio 0.9387. The gate stage carries the safety: without it 13 runs are worse than identity by more than 2 percent (contam/mult: Yeo-Johnson 7 times, worst 3.211; zero_infl/mult: cbrt 5 times; pareto/mult: Yeo-Johnson once). The "FP" of CV (31/103) are all non-identity picks with ratio <= 1.0 in cells where the oracle gain is below 2 percent (bounded01/mult 5/7, negbin/mult 7/7, ordinal/mult 6/7, contam/mult 7/7, pareto/mult 5/5, bounded0c/mult 1/7), not harm.

### 2.2 Per cell (n = 5000, p = 8, 7 seeds; RMSE unless stated; hit = HIT/helpful runs, FP = non-identity picks/runs where nothing helps)

| target / noise | helps (RMSE) | oracle RMSE ratio | W1 hit, FP, regret | CV hit, FP, regret | helps (MAE) | W1 hit (MAE) | CV hit (MAE) |
|---|---|---|---|---|---|---|---|
| bimodal / homo | 0/7 | 1.000 | 0/0, 0/7, 1.000 | 0/0, 0/7, 1.000 | 0/7 | 0/0 | 0/0 |
| bimodal / mult | 0/7 | 1.000 | 0/0, 0/7, 1.000 | 0/0, 0/7, 1.000 | 0/7 | 0/0 | 0/0 |
| bounded01 / homo | 7/7 | 0.950 | 0/7, 0/0, 1.052 | 7/7, 0/0, 1.000 | 5/7 | 0/5 | 5/5 |
| bounded01 / mult | 0/7 | 0.991 | 0/0, 3/7, 1.005 | 0/0, 5/7, 1.003 | 7/7 | 3/7 | 5/7 |
| bounded0c / homo | 0/7 | 0.999 | 0/0, 0/7, 1.001 | 0/0, 0/7, 1.001 | 0/7 | 0/0 | 0/0 |
| bounded0c / mult | 0/7 | 0.999 | 0/0, 0/7, 1.001 | 0/0, 1/7, 1.001 | 0/7 | 0/0 | 0/0 |
| contam / homo | 0/7 | 1.000 | 0/0, 0/7, 1.000 | 0/0, 0/7, 1.000 | 0/7 | 0/0 | 0/0 |
| contam / mult | 0/7 | 0.990 | 0/0, 0/7, 1.010 | 0/0, 7/7, 1.002 | 7/7 | 0/7 | 6/7 |
| gamma / homo | 7/7 | 0.789 | 7/7, 0/0, 1.003 | 7/7, 0/0, 1.005 | 7/7 | 0/7 | 2/7 |
| gamma / mult | 7/7 | 0.927 | 6/7, 0/0, 1.004 | 6/7, 0/0, 1.006 | 7/7 | 7/7 | 7/7 |
| lognormal / homo | 7/7 | 0.755 | 7/7, 0/0, 1.005 | 7/7, 0/0, 1.005 | 7/7 | 0/7 | 2/7 |
| lognormal / mult | 7/7 | 0.828 | 7/7, 0/0, 1.000 | 7/7, 0/0, 1.000 | 7/7 | 7/7 | 7/7 |
| neg_skew / homo | 7/7 | 0.741 | 7/7, 0/0, 1.005 | 7/7, 0/0, 1.007 | 7/7 | 3/7 | 3/7 |
| neg_skew / mult | 7/7 | 0.907 | 1/7, 0/0, 1.054 | 5/7, 0/0, 1.023 | 7/7 | 5/7 | 6/7 |
| negbin / homo | 7/7 | 0.972 | 6/7, 0/0, 1.006 | 7/7, 0/0, 1.002 | 2/7 | 0/2 | 2/2 |
| negbin / mult | 0/7 | 0.987 | 0/0, 7/7, 1.000 | 0/0, 7/7, 1.001 | 7/7 | 2/7 | 3/7 |
| normal / homo | 0/7 | 1.000 | 0/0, 0/7, 1.000 | 0/0, 0/7, 1.000 | 0/7 | 0/0 | 0/0 |
| normal / mult | 0/7 | 1.000 | 0/0, 0/7, 1.000 | 0/0, 0/7, 1.000 | 0/7 | 0/0 | 0/0 |
| ordinal / homo | 0/7 | 1.000 | 0/0, 0/7, 1.000 | 0/0, 0/7, 1.000 | 0/7 | 0/0 | 0/0 |
| ordinal / mult | 0/7 | 0.997 | 0/0, 2/7, 1.002 | 0/0, 6/7, 1.000 | 0/7 | 0/0 | 0/0 |
| pareto / homo | 7/7 | 0.880 | 7/7, 0/0, 1.001 | 7/7, 0/0, 1.001 | 7/7 | 7/7 | 7/7 |
| pareto / mult | 2/7 | 0.980 | 0/2, 5/5, 1.012 | 2/2, 5/5, 1.000 | 7/7 | 5/7 | 7/7 |
| poisson / homo | 7/7 | 0.902 | 0/7, 0/0, 1.109 | 7/7, 0/0, 1.000 | 7/7 | 0/7 | 7/7 |
| poisson / mult | 7/7 | 0.969 | 3/7, 0/0, 1.019 | 7/7, 0/0, 1.002 | 7/7 | 2/7 | 5/7 |
| weibull / homo | 7/7 | 0.799 | 6/7, 0/0, 1.007 | 5/7, 0/0, 1.013 | 7/7 | 0/7 | 4/7 |
| weibull / mult | 7/7 | 0.924 | 0/7, 0/0, 1.043 | 6/7, 0/0, 1.011 | 7/7 | 7/7 | 7/7 |
| zero_infl / homo | 0/7 | 1.000 | 0/0, 0/7, 1.000 | 0/0, 0/7, 1.000 | 3/7 | 0/3 | 0/3 |
| zero_infl / mult | 0/7 | 0.994 | 0/0, 0/7, 1.006 | 0/0, 0/7, 1.006 | 7/7 | 0/7 | 0/7 |
| **all 28 cells** | 93/196 | 0.939 | 57/93, 17/103, 1.012 | 87/93, 31/103, 1.003 | 122/196 | 48/122 | 85/122 |

Cells with no helpful transform and no pick (normal, bimodal, ordinal/homo, contam/homo, bounded0c, zero_infl): W1 and CV both abstain. Cells where W1 FAILS (finds fewer than half of the helpful runs): bounded01/homo 0/7, neg_skew/mult 1/7, poisson/homo 0/7, poisson/mult 3/7, weibull/mult 0/7, pareto/mult 0/2; and the MAE column: 0/7 on gamma/homo, lognormal/homo, weibull/homo, poisson/homo, contam/mult.

### 2.3 Failure modes of W1, mechanism measured, fix tested

**T1. Point masses (counts, zero-inflated, censored): two opposite artefacts of the Jacobian floor.**
(a) On the observed data the floor makes log|f'(0)| huge (cbrt: +8 nats per exact zero). Raw LL and LL + null pick cbrt in 14/14 zero_infl runs (gain 2.55-3.04 nats against a null threshold of 1.25-2.49); the held-out gate then rejects (RMSE of cbrt 1.007-1.022x identity), so W1 is safe there. Same for bounded0c/homo (censored at 0 and 10): LL picks signed_sqrt 7/7, gate removes it.
(b) In the null the bootstrap draws y* = fit +- r are clipped at 0 when min(y) = 0, which creates atoms at 0 in the draws and inflates the threshold: poisson/homo n = 5000, 4 seeds, gain 0.309 / 0.265 / 0.251 / 0.280 against tau 0.322 / 0.311 / 0.301 / 0.287 (p = 0.14-0.90), unclipped tau 0.005 / 0.005 / 0.001 / 0.007 (p = 0.02 = the minimum for B = 49). poisson/mult: tau 0.65-0.80 clipped vs 0.17-0.20 unclipped; negbin/homo 0.84-0.92 vs 0.30-0.40. Consequence: W1 hits poisson/homo 0/5 at every n in {300, 1000, 5000, 20000} while the oracle gain is 8-10 percent.
Fix N (no clipping of the null draws when min(y) = 0): poisson/homo hits 3/5, 4/5, 4/5, 5/5 at n = 300 / 1000 / 5000 / 20000; pooled over the 8 n-axis configs 15/24, 17/25, 14/25, 14/25 vs W1 12/24, 13/25, 10/25, 9/25. Fix S (abstain when one value has >= 3 % of the rows or there are <= 20 distinct values) is rejected: it removes the zero-inflated false picks but also all count gains (48/93 hits vs 57/93; poisson, negbin 0/7 each). CV needs neither: poisson/homo 7/7 in the zoo (n = 5000) and 5/5 at each of n = 300, 1000, 5000, 20000.

**T2. Bounded target with boundary atoms (clip to [0.001, 0.999]).** bounded01/homo: oracle 0.950 (logit 0.948), LL gain 0.000-0.002 nats, picks identity/Yeo-Johnson(lambda ~ 1): W1 0/7 at n = 5000 and 0/5 at n = 1000, 5000, 20000 (regret 1.055-1.059). Logit wrecks the Gaussian fit (atoms at +-6.9) while it improves the bounded prediction. CV: 7/7 and 5/5 at n >= 1000.

**T3. The Gaussian-density-best transform is not the RMSE-best transform.** weibull/mult: LL argmax is cbrt in 7/7 runs at n = 5000, 5/5 at n = 20000, 3/3 at n = 100000; judge-best is log (ratio 0.913-0.918 at n = 100000 vs cbrt 0.959-0.960; 0.925 vs 0.962 at n = 5000). Regret 1.043-1.045, persists with n. neg_skew/mult: W1 1/7, regret 1.054. gamma/mult 6/7. CV: weibull/mult 6/7 at n = 5000, 5/5 at n = 5000 and 20000 (n-axis), picks log 3/3 at n = 100000; neg_skew/mult 5/7.

**T4. Heavy tails / outliers.** The Gaussian LL rewards tail fit. Raw LL picks Yeo-Johnson in contam/mult 7/7 (lambda -0.89 to -0.93; ratio up to 3.211) and in pareto/mult 7/7 (lambda at the -1 bound in all 7; worse than identity by > 2 % in 1); the gate rejects contam/mult 7/7, W1 abstains. Robust LL (Huber loss, MAD scale, fix R) + null + paired gate recovers pareto/mult 2/2 and gives 60/93 vs 57/93 overall, FP 22 vs 17 (harmless): marginal.

**T5. Metric.** The RMSE-gated W1 is blind to median-type gains: contam/mult MAE oracle ratio 0.681 (log is the MAE-best candidate in 7/7), W1 0/7, CV 6/7. Over the zoo: MAE-helpful runs 122/196, W1 48/122, CV 85/122. An MAE-flavoured CV (CVm: absolute error of the clipped median prediction) was run on the n-axis (MAE judge): 26/30, 33/33, 35/35, 34/34 hits at n = 300 / 1000 / 5000 / 20000 vs W1 11/30, 16/33, 15/35, 15/34. The user must choose RMSE vs MAE to match the downstream loss; the criterion cannot infer it.

**T6. Unsafe inverse of Yeo-Johnson with lambda outside [0, 2].** neg_skew picks lambda 2.4-2.7 (bounded-image on the negative side). With the a4 margin (+5 % beyond the training range) the picked Yeo-Johnson (14/14 neg_skew runs, lambda 2.38-2.73) has RMSE 3.9e2 to 4.2e21 x identity (first pass); with zero margin the same pick is 0.73-0.99 x identity (neg_skew/homo oracle 0.741, W1 7/7 hits). The gate in the first run saw 1e18 and rejected a correct decision. Implementations must clip a bounded-image inverse at the training range (zero margin).

**T7. Small n, strongly skewed target: CV is under-powered.** lognormal/mult hits at n = 300 / 1000: LL pipeline (N_W1) 5/5, 5/5; CV 2/5, 2/5 (regret 1.050, 1.068); at n >= 5000 both 5/5. HX (CV or LL pipeline accepts; lowest cross-fitted error among the two proposals): 5/5 and 3/5, pooled n = 300: 18/24 hits, 5/16 FP (0 harmful), regret 1.012 (CV 15/24, 1.016; N_W1 15/24, 1.024); n = 1000: 21/25, regret 1.010 (CV 20/25, 1.014).

## 3. Axis (b): feature distributions

Zoo, n = 5000, amp = 1, 10 seeds, 40 cells (400 runs): HIT 288, TN 90 (every link-none cell, every feature law), FP 22, HARM 0, mean regret 1.0000, worst pick ratio 1.0; the MAE judge gives the same (287 / 90 / 23). The 22 FP are sqrt links on Beta(3,2), the 5-level integer and uniform(0.25,4) where the best candidate gains only 1.4-2.1 percent and the picked sqrt is that candidate. Zero-inflated with sqrt link: cbrt is picked 10/10 (judge best). The t3, bimodal, mixed-sign skewed and discrete laws behave like the normal one. Caveat from wave 1 still applies: for one feature the criterion equals the held-out R2 of a one-variable linear fit, so hits are close to tautological; what is tested here is the null and the ordering.

**Stress fixtures the zoo does not cover** (20 seeds each, judge = ridge on [z, f(x)], spline ceiling in the comments):

| fixture | n | wave-1 outcome | partial-correlation fix |
|---|---|---|---|
| lognormal x, link log, z correlated with log x | 1000 / 5000 | HIT 18 + WRONG 2 / HIT 20 | HIT 20 / HIT 20 |
| lognormal x, LINEAR link, z correlated with log x | 1000 / 5000 | **HARM 18 + FP 2 / HARM 20**: picks abs(x - med) 40/40, RMSE +3.2 % / +3.7 % on average (worst +4.2 % / +4.4 %) | TN 20 / TN 20 |
| normal x, link (x - 1.5)^2 (off-centre square) | 1000 / 5000 | TN 20 / TN 20 (no candidate helps; spline ceiling 0.869 / 0.858 of identity) | same |
| normal x, link sin(2x) | 1000 / 5000 | HIT 20 / 20 (cbrt, 0.891; ceiling 0.62) | same |
| normal x, step 1[x > 0.5] | 1000 / 5000 | HIT 19 + FP 1 / HIT 20 (tanh; ceiling 0.79) | same |
| 0.5 % of x multiplied by 30, linear link | 1000 / 5000 | TN 19 + MISS 1 / TN 20 | same |

**F1 (new failure, fixed).** The conditional criterion removes the linear part of the other columns from y but correlates the RAW f(x) with that residual. With x correlated with the covariate, the candidate that is least collinear with the covariate scores higher for that reason alone. Fix `feat_decide_partial`: residualise y and every candidate on [1, Z] (Frisch-Waugh) before the correlation, same permutation null. 0/40 harmful picks; the 400-run zoo is unchanged (288 / 90 / 22, regret 1.0000). The multi-feature version needs (I - H_{-j}) F = (I - H) F + x~_j x~_j' F / (x~_j' x~_j); not implemented or run. The unary candidate set is the remaining limit (off-centre square, sin, step: gaps of 0.13-0.40 of RMSE to the spline ceiling that no candidate closes), not the criterion.

## 4. Axis (c): sample size n

**Features** (14 dist/link pairs incl. 4 link-none controls, 8 seeds, amp 1.0 and 0.3; n = 300, 1000, 5000, 20000, 100000; 1120 runs; link-none picks 0/32 in every (n, amp) cell except 1/32 at n = 300, amp 0.3): helpful runs (>= 2 percent) are HIT in 79/80, 80/80, 80/80, 80/80, 80/80 at amp = 1 and 55/55, 53/53, 54/54, 56/56, 56/56 at amp = 0.3; regret 1.0008 at n = 300 (amp 1) and 1.0000-1.0012 elsewhere; worst pick ratio 1.023 (one run at n = 300, amp 0.3).
Type-I on true nulls (link none, 6 laws incl. t3, zero-inflated, 5-level integer, 100 seeds per law, nominal 5 %): 3/600 at n = 300 (Wilson 95 % 0.002-0.015), 0/600 at n = 1000, 0/600 at n = 5000. The permutation null of the max difference is conservative, as expected when the identity effect is real.
Weak effects (5 dist/link pairs, 40 seeds, amp 0.05 / 0.1 / 0.2): accept rate 0.14 / 0.28 / 0.64 at n = 300, 0.28 / 0.59 / 0.83 at n = 1000, 0.67 / 0.81 / 0.96 at n = 5000; oracle ratio 0.997-0.961 at n = 300 (gain <= 4 %); harmful picks (ratio > 1.02) 3/195, 3/178, 0/110 non-helpful runs at n = 300 for the three amplitudes, 0 at n >= 1000. Nothing in the data calls for abstention at n >= 300 for features; the weakest effects are simply missed, at a cost below 4 percent RMSE.

**Targets** (8 configs x 5 seeds per n, p = 8; hit/helps, FP/nothing-helps, regret):

| n | runs | helps | W1 | W1 with unclipped null (N_W1) | CV | oracle ratio |
|---|---|---|---|---|---|---|
| 300 | 40 | 24 | 12/24, 1/16, 1.037 | 15/24, 4/16, 1.024 | 15/24, 3/16, 1.016 | 0.942 |
| 1000 | 40 | 25 | 13/25, 4/15, 1.027 | 17/25, 4/15, 1.015 | 20/25, 5/15, 1.014 | 0.934 |
| 5000 | 40 | 25 | 10/25, 0/15, 1.029 | 14/25, 0/15, 1.018 | 25/25, 5/15, 1.002 | 0.933 |
| 20000 | 40 | 25 | 9/25, 0/15, 1.030 | 14/25, 0/15, 1.019 | 25/25, 7/15, 1.001 | 0.935 |

Configs: lognormal/mult, weibull/mult, poisson/homo, bounded01/homo, normal/mult, zero_infl/mult, pareto/homo, contam/mult. The ungated stages degrade with n: raw LL non-identity picks in the 15-16 nothing-helps runs are 14/16, 15/15, 15/15, 15/15 at n = 300 / 1000 / 5000 / 20000 with 1, 3, 8, 6 of them worse than identity by > 2 percent; the LL null alone keeps 6/16, 10/15, 11/15, 10/15 of them (the same 1, 3, 8, 6 harmful). The gate removes them all at n >= 5000 (W1 FP 0/15) while its hit rate falls (12 -> 9 of 24-25) because T1-T3 do not shrink with n. n = 100000 (4 configs x 3 seeds, B = 49): W1 3/9 hits, CV 9/9, CVm 6/9 (MAE judge CVm 9/9, W1 6/9), no harmful pick; the criterion cost is 5-18 s (machine loaded, includes CV), so a subsample for the null is needed beyond ~1e5.
Type-I on true nulls (normal/homo, positive linear y = 10 + 2u + g, normal/mult; 100 seeds per cell, B = 49): pooled over the three nulls at n = 300 / 1000 / 5000: raw LL 131/300, 114/300, 104/300 (0.437, 0.380, 0.347; normal/mult 100/100 at all n); LL null 13/300, 8/300, 9/300 (0.043, 0.027, 0.030; worst cell 8/100 on normal/mult at n = 1000); W1 13, 3, 2 of 300; CV 2, 0, 0 of 300.
Answer to "below which n is nothing detectable": for features no n >= 300 tested; for targets the null is still calibrated at n = 300 and about half of the helpful runs are found (HX 18/24); the data do not show a threshold, they show lost power on skewed targets for CV and on bounded/count targets for W1. n < 300 was not run.

## 5. Axis (d): number of features p

**Feature criterion, p = 8 / 50 / 200 features, 3 informative (x0 lognormal with log link, x1 normal with square link, x2 linear control), n = 1000 and 5000, 10 runs per cell at p = 8 and 50 runs per cell at p = 50 and 200 (10 + 40 further seeds); Bonferroni was only run in the 10-seed pass:**

| n | p | per-feature null (wave-1): noise false-accept rate, FWER | global max over features x candidates (maxT): noise FA, FWER [Wilson 95 %] | studentised maxT: FWER | HIT x0, x1 (maxT) | false picks on the linear x2 (maxT) |
|---|---|---|---|---|---|---|
| 1000 | 8 | 0.040, 1/10 | 0.0000, 0/10 | 0/10 | 10/10, 10/10 | 0/10 |
| 1000 | 50 | 0.064, 49/50 | 0.0009, 2/50 = 0.04 [0.011, 0.135] | 4/50 | 49/50, 50/50 | 0/50 |
| 1000 | 200 | 0.059, 50/50 | 0.0003, 3/50 = 0.06 [0.021, 0.162] | 14/50 = 0.28 | 50/50, 50/50 | 0/50 |
| 5000 | 8 | 0.060, 3/10 | 0.0000, 0/10 | 0/10 | 10/10, 10/10 | 0/10 |
| 5000 | 50 | 0.060, 44/50 | 0.0026, 5/50 = 0.10 [0.043, 0.214] | 9/50 | 50/50, 50/50 | 0/50 |
| 5000 | 200 | 0.060, 50/50 | 0.0001, 1/50 = 0.02 [0.004, 0.105] | 11/50 = 0.22 | 50/50, 50/50 | 0/50 |

The per-feature null is calibrated per feature (5-7 percent) but the family-wise error is 0.88-1.00 at p = 50 and 200: a max over candidates is NOT enough. One max over features x candidates (shared permutations, `feat_many` / `feat_accept`) gives FWER 0.02-0.10 with the power on the two informative features unchanged (49-50/50). Studentising each feature's statistic by its own null sd inflates the FWER at p = 50-200 (0.08-0.28) and is not recommended; Bonferroni on 100-permutation p-values cannot reject (smallest p 1/101 > 0.05/p): 0 hits. Cost: 0.1 s (p = 8) to 4.3 s (p = 200, n = 5000) for B = 100. Not run: p = 200 at n = 20000, correlated noise features, selection of the tested set by MRMR relevance.

**Target criterion** (the LL test is one test per target, no multiplicity over features; the only p effect is the design size). 3 informative + noise features; hit/helps [FP/nothing helps]:

| cell | n | p = 8 | p = 50 | p = 200 |
|---|---|---|---|---|
| lognormal/mult | 1000 | W1 5/5, CV 2/5 | W1 5/5, CV 5/5 | W1 5/5, CV 4/5 (oracle 0.863) |
| lognormal/mult | 5000 | W1 12/12, CV 12/12 | 5/5, 5/5 | 5/5, 5/5 |
| normal/mult (null) | 5000 | LL null FP 1/7, W1 0/7, CV 0/7 | 0/3, 0/3, 0/3 | LL null 1/3, W1 0/3, CV 0/3 |
| poisson/homo | 5000 | W1 0/7, CV 7/7 | 0/3, 3/3 | 0/3, 3/3 |
| weibull/mult | 5000 | W1 0/7, CV 6/7 | 1/3, 3/3 | 3/3, 3/3 |

No sign of miscalibration or power loss for W1 up to n/(p+1) = 5 (n = 1000, p = 200) on the strong lognormal effect; the T1 and T3 failures are the same at every p. Samples are small (3-5 seeds per cell); n/(p+1) < 5 was not run.

## 6. Preconditions under which the criterion should abstain (return identity)

Each rule is tied to the run that motivates it; rules marked "precaution" are not measured.

1. Raw LL (no null, no gate) is never an acceptance rule: 34-44 percent false picks on true nulls, 13/196 zoo runs worse than identity by > 2 percent (worst 3.2x).
2. A transform is accepted only if a held-out comparison on the ORIGINAL scale with the clipped, smeared inverse confirms it (W1 gate or CV). Harmful picks (> 2 % worse than identity): W1 0 in the 196-run zoo, 1 in the 160 n-axis runs (zero_infl/mult, n = 1000), 0 in 12 runs at n = 1e5; CV 0, 0, 0.
3. LL-based acceptance only for atom-free targets: when a single value carries >= 3 % of the rows (zero-inflated, censored, clipped bounds) or the target has <= 20 distinct values (counts), use CV (or the unclipped null N plus the gate), not the LL null alone: LL picks cbrt in 14/14 zero_infl runs and the clipped null kills 0/5 poisson hits. Do not abstain altogether (rule S loses 9 of 57 hits).
4. Bounded targets with mass at the boundary: LL cannot see logit (0/7); CV only.
5. Heavy tails / outliers (Pareto alpha 2.5, 1 % x30 contamination): never LL-only; Yeo-Johnson lambda at a bound (-1 or 3) is a warning sign; bounded-image Yeo-Johnson (lambda < 0 or > 2) must be inverted with zero clipping margin.
6. Ordinal targets with <= 5 levels: oracle ratio 0.997-1.000, no transform to find; abstain (CV returns tiny-gain picks in 6/7 ordinal/mult runs, ratio 1.000).
7. The downstream metric must be chosen: RMSE-type gates miss MAE gains (zoo, MAE-helpful runs: W1 48/122, CV 85/122; n-axis MAE-helpful runs: W1 11/30, 16/33, 15/35, 15/34 vs CVm 26/30, 33/33, 35/35, 34/34 at n = 300 / 1000 / 5000 / 20000); abstain from changing the target when the downstream loss is unknown.
8. Linear / additive heads only (trees: wave-1, not re-run). Precaution.
9. n < 300 or n/(p+1) < 5: not measured; abstain as a precaution. At n >= 300 the null stages were calibrated (type-I <= 4.3 percent pooled, CV <= 0.7 percent) and no n gave harmful picks after the gate.
10. Features: test with the global max over features x candidates when more than a handful of features are tested; with a correlated covariate use the partial-correlation form (rule F1); no n-based abstention for n >= 300.

## 7. Recommended pipeline (what the numbers support)

Target: candidates as in the synthesis (+ safe inverse with zero margin for bounded-image Yeo-Johnson); acceptance = HX: accept if the cross-fitted original-scale max-T test (CV, B = 499 sign flips, metric matched to the downstream loss) accepts OR the LL pipeline with the unclipped null for min(y) = 0 and the held-out gate accepts; the transform is the proposal with the lowest cross-fitted error. At n >= 5000 this equals CV (87/93 hits, regret 1.003, 0 harmful picks); at n <= 1000 it is the best of the rules tested (hits 18/24, 21/25; regret 1.012, 1.010). LL stays useful as a cheap proposer and as the power source at small n; it must not decide alone.
Feature: wave-1 A2 criterion, partial correlation against the already-selected columns, null as the max over features x candidates.

## 8. Not run / limits

n < 300; n/(p+1) < 5; p = 200 at n = 20000; trees, LightGBM, CatBoost, classification, real data; the unary candidate set beyond wave 1 (gaps to the spline ceiling 0.13-0.40 of RMSE on off-centre square, sin, step); correlated NOISE features; one noise level per target cell; B = 49 or 99 sign flips (smallest attainable p 0.02 / 0.01); 3-7 seeds per target cell (the per-cell hit rates have wide intervals, the pooled counts are the reliable ones); multi-feature partial correlation (formula only); the per-feature judge in the p-axis uses the true links of the other informative columns as context; run times were measured on a loaded machine. My judge scores a fresh sample, wave 1 scored a hold-out of the fitted sample, so absolute ratios differ slightly from wave 1.

## 9. Reproduce

Result files: `tt_b2_zoo2_s*.jsonl` is the zoo reported above; `tt_b2_zoo_s*.jsonl` is the first pass (a4 clip margin for every Yeo-Johnson, no fixes), kept only because section 2.3 T6 quotes it; `tt_b2_tn_s*.jsonl` n-axis; `tt_b2_tn2.jsonl` small-n rerun with HX; `tt_b2_tbig.jsonl` n = 1e5; `tt_b2_tp.jsonl` + `tt_b2_tp2.jsonl` target p-axis (the first was stopped at 32 of 120 runs for time and replaced by a reduced one); `tt_b2_feat1`, `tt_b2_featn_s*`, `tt_b2_featnull`, `tt_b2_featweak`, `tt_b2_featstress`, `tt_b2_multi`, `tt_b2_multi2` feature side; `tt_b2_calib_s*` target true-null calibration.

`PYTHONPATH=D:\Temp\wt2\src`, `OMP_NUM_THREADS=1`, `NUMBA_NUM_THREADS=2`, `CUDA_VISIBLE_DEVICES=` (set inside `tt_b2_core.py`). From the `nbins_strategy` directory:
`python tt_b2_run_target.py --out zoo.jsonl --n 5000 --p 8 --seeds 7 --B 99 --shard 0/2 --fixes R,E,V` (zoo; ~9-19 min on 2 processes), `python tt_b2_run_target.py --cfgs lognormal:mult,... --n 300,1000,5000,20000 --seeds 5 --fixes N,V` (n-axis), `python tt_b2_run_feat.py --out f.jsonl --n 5000 --seeds 10` (feature zoo, 90 s), `python tt_b2_run_p.py --p 8,50,200 --n 1000,5000` (p-axis), `python tt_b2_calib.py` (true-null calibration), `python tt_b2_feat_stress.py` (stress), `python tt_b2_analyze.py target|feat|multipool|md FILE` and `python tt_b2_report_tables.py zoo|stages|naxis` (tables).
