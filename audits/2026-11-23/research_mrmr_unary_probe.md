# What MRMR does with a known unary relation (2026-11-24)

Script: `src/mlframe/feature_selection/_benchmarks/nbins_strategy/unary_probe_mrmr.py`. One MRMR fit per case (n = 6000, 60 percent for the fit, `fe_max_steps=1`, preset `medium`, default families), ridge held-out R2 on the other 40 percent. Ledger of the first case: `mrmr_sin_probe_rejection_ledger.csv`. One seed, one run per case; the wall times (8-367 s) were taken while five other jobs ran, so they say nothing about speed.

| case | ridge R2 on raw x1, x2 | MRMR output | ridge R2 on the output | ceiling |
|---|---|---|---|---|
| y = sin(x1) + x2, x1 in [-6, 6] | 0.673 | x1, x2, x1__fourier3, x1__T3 | 0.941 | 0.943 |
| same, x1 in [-1, 1] | 0.932 | x2, x1 | 0.932 | 0.932 |
| y = log(x1) + x2, lognormal x1 | 0.732 | x2, x2+x1, sir direction, 4 relu hinge columns of x1, x1__LL2 | 0.935 | 0.957 |
| y = x1^2 + x2 | 0.324 | x1, log_abs(x1), 4 rff columns, lof, cond_resid, x1__lga_LL3, 3 hinge columns | 0.943 | 0.971 |

## Four shortcomings found (the goals of the plan)
1. The useful transform is found only approximately: sin(x1) by a Fourier and a Chebyshev basis leg, log(x1) by hinges and a polynomial basis, never by the exact unary function.
2. The unary table (17 functions in `medium`, 41 in `maximal`) does not decide anything in these fits (only `log_abs(x1)` appeared, on the square).
3. A monotone unary transform (log of a lognormal feature) is invisible to the rank-binned MI, so nothing credits the exact function.
4. The output is noisy: 8 and 12 columns for two signal features (rff, lof, sir, cond_resid, hinge).

## Why `sin(x1)` and `add(sin(x1), x2)` lost (ledger and provenance of case 1)
- **`sin(x1)` was a candidate**: produced by the `mi_greedy` family (single-feature unary transforms) and killed by the gate `marginal_uplift_floor`: observed MI 0.187 against a floor of 0.407 (median + k MAD of the raw columns' MI, lifted by the strong x2 at 0.536).
- **All one-feature candidates of x1 have the same MI with y** in `column_stats_`: x1 0.2158, x1__fourier3 0.2169, x1__tnh_L3 0.2174, sin(x1) 0.2190, oofwarp(x1) 0.2191. y depends on x1 only through sin(x1), and a binned MI cannot tell x1 from any function of x1 that keeps what y needs. So the choice between them is made by the gates of their families, not by merit: `hybrid_orth` (Fourier, Chebyshev) left no ledger row; `mi_greedy` was cut by its floor.
- **The pair (x1, x2) was evaluated and rejected**: ledger row `(0, 1)`, operator `add`, gate `engineered_mi_prevalence`, observed 0.7488 against 0.9, reason `best_mi_over_pair_mi_below_floor`. Hand check with the same bins (n = 3600): the plug-in MI of the raw pair (31 x 31 cells) is 1.624 nats and the MEAN of the same statistic under a permuted target is 1.079; the engineered one-feature MI is 1.274 with a null mean of 0.038. Null-corrected: pair 0.544, engineered 1.237 (ratio 2.27); Miller-Madow: 0.455 and 1.237 (ratio 2.7). The raw ratio 0.785 (0.749 in the fit) is a bias artefact: the sufficient one-dimensional summary of the pair is rejected for carrying 75 percent of an inflated denominator.
- **Known and rejected before**: `MRMR.fe_mm_debias_prevalence` (default False) debiases both sides with Miller-Madow; it was bench-rejected as a default (2026-06-09) because on the weak F2 target it over-corrects (ratio about 2) and admits cross-mix artefacts (3/10 to 9/10 seeds). Switching the flag on is therefore not the fix; the gate must separate a genuine one-dimensional summary of the pair (information equal to the pair's, as here) from an artefact whose information is only comparable.
