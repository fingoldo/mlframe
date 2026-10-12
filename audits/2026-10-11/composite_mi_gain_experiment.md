# Does `mi_gain` of `training/composite` predict the out-of-sample gain of a composite target? (2026-10-11)

Question from the owner: the documented reason for `eps_mi_gain = -10` (a correct composite with a very negative `mi_gain`) "cannot be", it does not match what is observed on real data. Experiment: `src/mlframe/feature_selection/_benchmarks/target_lens/exp_composite_mi_gain.py` (`python -m mlframe.feature_selection._benchmarks.target_lens.exp_composite_mi_gain 10 20000`), 6 processes with known truth, 10 seeds, n = 20 000 (half train, half held-out), ridge and HistGradientBoosting, RMSE on the y scale. `mi_gain` is computed with the production code (`discovery/screening._mi_to_target`, binned estimator, 16 bins, mean aggregation, base excluded).

## What `mi_gain` measures

`mi_gain = mean_j MI(x_j, T) - mean_j MI(x_j, y)` over the columns `x_j` of X without the base, **marginal** MI per column. It is not a prediction of the downstream error: it asks whether the other columns relate more strongly to `T = y - base` than to `y`. It fails when a column of X **carries the base** (a lag, a source column of a derived base): that column has large marginal MI with `y` because `y` contains the base, and almost none with `T`, so the gain is negative although the composite is exactly right.

## Results (median over 10 seeds)

| process | `mi_gain` (production) | base-tracking columns removed | composite vs best plain model, HGB (ridge) |
|---|---|---|---|
| AR(1), base = lag1, X = other lags + noise | -0.217 | -0.0002 | +0.1% (+0.5%): no gain, correct |
| `y = base + x1*x2 + sin(2 x3)`, base independent of X | +0.106 | +0.107 | **-14.9%** (0.0%) |
| `y = base * exp(0.4 x1 + 0.3 x2)` (ratio composite) | +0.184 | +0.185 | **-54.1%** (-73.8%) |
| control: base is pure noise | -0.225 | -0.225 | **+1488%** (+318%): harmful, correctly negative |
| `y = base + x2*x3`, base = 3 x1 + noise (correlated with a column) | **-0.102** | **+0.128** | **-13.6%** (0.0%) |
| `y = 8 sin(3 x1) + x2*x3`, base = 8 sin(3 x1) (a transform of a column) | **-0.164** | **+0.123** | **-11.8%** (0.0%, equals the plain model that has the base as a column) |

## Findings

1. **The owner's expectation is right where the gain is computed correctly, and the documented counter-example is real but has a specific cause.** In 4 of 6 processes `mi_gain` has the sign of the real gain. In the two where the base is tied to a column of X (correlated; derived by a transform) it has the wrong sign: -0.10 and -0.16 while the composite lowers the HGB error by 12-14%. The production gate cannot tell these from a harmful composite (the noise-base control has -0.22).
2. **Removing from X every column that carries the base repairs the sign in all six processes.** "Carries the base" is decided with a calibrated null (binned MI of the column with the base above the maximum over 20 permutations of the base), no fixed bar. After it the gain is +0.13 and +0.12 for the two failing processes, and the harmful control stays at -0.22. The AR(1) case becomes about 0, which is right: there the raw model already has the lag as a column and the composite adds nothing.
3. **The documented pure-lag incident is a case of the same defect**: X without the base still contains the other lags, which carry the base; MI(y, X) is then large and MI(T, X) near zero. The fix is the exclusion above, not a gate at -10. Re-enabling the gate in this corrected form is a candidate change for `training/composite` (a positive `eps_mi_gain`, a calibrated one); it needs its own test on the incident data.
4. **For ridge a composite never beats the plain model that has the base as a column** (free coefficient is a superset of offset 1; `comp_ols` equals it). The composite earns its keep for gradient boosting (11-15% on the additive and derived-base processes) and for the multiplicative composites (a log-ratio target is not a monotone transform of y: -54% HGB, -74% ridge). This matters for the idea of mixing target and features inside MRMR: for a **linear** downstream model the additive mix adds nothing over adding the base as a feature, which MRMR's feature side already does; the lens pays for trees and for ratios.
5. **What the owner observed and what the code documents are both right**: MI against `T` does rank the correct composite first once the columns that carry the base are excluded; the production definition does not exclude them.

## Open items

* Prototype the corrected gate inside `training/composite/discovery/_eval.py` (exclusion by permutation null, cached per base) and replay the AR-1 incident that moved the default to -10.
* Check that the cost is acceptable: one extra binned MI per (column, base) pair plus 20 permuted copies; the permutation can be shared across columns.
