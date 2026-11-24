# Research A1: information-theoretic criterion for a target transform (2026-11-23)

Code: `src/mlframe/feature_selection/_benchmarks/nbins_strategy/tt_a1_gap_criterion.py`. One run, 8 fixtures x 7 seeds = 56 cases, 20000 rows, 8 features, 70/30 split, CPU, OMP 1. Full log of the run: scratchpad `a1_run.txt` (numbers below are copied from it).

Fixtures (right answer): lognormal, hetero_lognormal (log); cube (cbrt); reciprocal y = 1/(4+0.5x0+0.3x1+noise) (-1/y); and the no-transform fixtures linear, hetero_linear, linear_pos (y = 10 + linear, tests a false log), heavy (linear + 0.5*t3 noise). Candidates: identity, cbrt, signed_sqrt, signed_square, log, reciprocal (positive y only), Yeo-Johnson (lambda fitted on 4000 train rows). Judge: ridge (and HGB) RMSE on the original scale, clipped inverse, generalised smearing mean_j inv(zhat + r_j) over 100 train residuals (Duan for log).

## Criteria (computed on train rows only, half A fits, half B scores, no model retraining)
- R2: held-out linear R2 of f(y) on X, i.e. I_lin = -0.5 log(1-R2). Scale free under affine f, but NOT Jacobian-corrected.
- LL: profile log-likelihood per row on the original scale, `-0.5 log MSE_B(f) + mean_B log|f'(y)|` (Box-Cox with covariates; the Jacobian is what makes scales comparable).
- GAP: -(R2_additive spline - R2_linear) of f(y): the linearity gap.
- "+null": gate; accept the argmax f only if score_f - score_identity exceeds the 95% quantile (40 draws) of max_f(score_f - score_identity) under a sign-flip residual bootstrap from the identity linear fit (keeps heteroscedasticity; no fixed threshold).

## Results (56 cases; "right" = equals the known answer; FP = non-identity pick on the 28 no-transform cases)
| criterion | right | = ridge-judge-best | ridge RMSE ratio pick/identity, mean / worst | FP | HGB ratio mean / worst |
|---|---|---|---|---|---|
| R2 | 43 | 42 | 3.2e7 / 1.5e9 (overflow) | 11 | 9.4e6 / 4.1e8 |
| R2 + null | 50 | 43 | 3.2e7 / 1.5e9 | 4 | 9.4e6 / 4.1e8 |
| LL (Jacobian) | 49 | 49 | 0.796 / 1.014 | 7 | 0.990 / 1.033 |
| **LL + null** | **55** | 49 | **0.796 / 1.000** | **1** | 0.990 / 1.033 |
| GAP | 27 | 29 | 2.80 / 49 | 18 | 1.78 / 15.5 |
| GAP + null | 40 | 32 | 1.83 / 19.3 | 0 | 1.47 / 13.7 |

Per fixture, LL+null: lognormal log 7/7 (ridge ratio 0.587), hetero_lognormal log 7/7 (0.937), cube cbrt 7/7 (0.479), reciprocal 7/7 (0.369), linear identity 6/7, hetero_linear 7/7, linear_pos 7/7, heavy 7/7. The single FP is Yeo-Johnson with lambda near 1 on `linear` (ridge ratio 1.000, i.e. harmless: it is the identity). The old equal-width-MI proposer: signed_sqrt on linear 7/7 with ridge ratio 1.4-2.3 (PLAN file); here 0 harmful picks.
Raw R2 fails where the Jacobian matters: on hetero_lognormal it picks Yeo-Johnson in 2/7 (the heavy tail inflates R2 of a transform with an exploding inverse; ridge ratio 2.6e8). Its null-gate does not cure it because the gain is real on the transformed scale.
GAP (additivity/linearity gap) is the worst: it cannot separate two transforms that both linearise approximately and prefers `signed_square` (it makes the target look nearly additive) and `reciprocal` on lognormal (ridge ratio 6.1). The gap normalised by I_rank was not run as a separate criterion: I_rank is the same number for every f, so argmin(I_rank - I_lin) equals argmax(R2), i.e. the "R2" row; normalisation changes nothing in the ranking, only in the interpretation.

Trees: the criterion helps only linear heads. HGB judge-best is scattered over {signed_sqrt, log, cbrt, identity} even on the lognormal / cube / reciprocal fixtures; HGB ratio of the LL+null pick is 0.990 mean, 1.033 worst. So nothing beyond the earlier conclusion (linear heads only) was found for trees. Gain claims: ridge RMSE x0.37-0.94 on the four transform fixtures; zero harm on the four no-transform fixtures.

Speed (measured): all three criteria for one candidate cost 23.8 ms on 14000 train rows x 8 features (two pseudo-inverse fits shared across candidates, one matrix product per criterion). The null (40 draws x 3 criteria, about 7 candidates each) costs 3.5 s per dataset, i.e. about 0.4 s per criterion; for LL alone about 1.2 s. A ridge + inverse judge per candidate on the same data was not timed separately (not run); an HGB fit of 100 iterations is of the order of seconds, so the criterion is roughly 1-2 orders cheaper than model fitting, an estimate, not a measurement.

## Verdict
Works: the Jacobian-corrected Gaussian-linear information (profile likelihood, Box-Cox style) with a sign-flip bootstrap null. It finds log, cbrt and reciprocal with 7/7, has 0 harmful false positives on linear, hetero_linear, linear_pos, heavy (one harmless lambda~1 Yeo-Johnson), and picks the ridge-judge-best transform in 49/56 (the 7 misses are the linear/heavy cases where identity and Yeo-Johnson lambda~1 are within noise).
Does not work: (1) raw R2 (unnormalised scale, heavy-tail inflation); (2) the linearity gap alone; (3) anything for tree heads. The rank-based total I_rank is a constant in f, so it supplies no selection power by itself; it is only useful as a ceiling (a candidate cannot extract more than I_rank).
Caveats: Gaussian likelihood is not robust (heavy tails: handled here only because the null threshold absorbs it; with t3 noise the pick was identity 7/7 with the gate, 4/7 without); positive-domain candidates in the null use y* clipped at min(y), a mild approximation; Yeo-Johnson lambda near 1 should be collapsed to identity (|lambda-1| < about 0.1) in the implementation; features are Gaussian here, skewed features are untested (not run); only ridge+HGB judged (no LightGBM/CatBoost run).

## Recommended algorithm for MRMR (metadata only, linear-head recipe)
1. Candidate set C: identity, cbrt, signed_sqrt, signed_square; log and -1/y only if min(y) > 0; Yeo-Johnson with lambda from scipy `yeojohnson` on a 4000 row subsample (lambda clamped to [-1, 3]); each with a safe inverse (clip the argument to the train range +10 %, reciprocal upper bound below 0).
2. Split train rows into halves A, B. Design D = [1, X] (use the selected features or all numeric). Z = f(y) for all f. Fit pinv(D_A) once, predict B for all columns of Z at once.
3. For each f: mse_f = mean_B (z - zhat)^2, LL_f = -0.5 log(mse_f) + mean_B log|f'(y)| (derivatives: cbrt -log3 - 2/3 log|y|; sqrt -log2 - 0.5 log|y|; square log2 + log|y|; log -log y; -1/y -2 log y; Yeo-Johnson (l-1) log1p(y) for y>=0, (1-l) log1p(-y) for y<0; floor |y| at 1e-6).
4. Null: fit the identity linear model on all train rows, fit and residuals r. Repeat B = 40 times (more if cheap): y* = fit + s*r with random signs s, clip y* at min(y) if y > 0, rebuild candidates on y*, compute max_{f != id}(LL_f - LL_id). Threshold tau = 95% quantile of these maxima (use 99 % or Bonferroni when many candidates).
5. Choose f* = argmax LL. If LL_f* - LL_id <= tau, or f* is Yeo-Johnson with |lambda - 1| < 0.1, return identity. Otherwise return recipe (name, parameters, delta LL, tau). No threshold is fixed: tau is the calibration of "identity is adequate".
6. Apply only to linear heads; the held-out original-scale ridge gate from the PLAN stays as the second, cheap safety check (the criterion orders and filters, the gate confirms). Cost: about 25 ms per candidate plus about 1.2 s null for LL alone at 14000 rows x 8.
