# research_tt_b3: one reference implementation of the target-transform recommender and an independent test on fresh data (2026-11-24)

Agent B3, wave 2. Nothing here was committed. All numbers are from runs made for this report (raw per-case results in `audits/2026-11-23/b3_results/*.jsonl`); anything not run is marked "not run".

## 0. Result in six lines

1. `recommend_target_transform(X, y, *, candidates=None, n_null=40, alpha=0.05, seed=0)` exists, numpy only, deterministic for a seed (checked: identical output twice), vectorised over candidates, with a documented dataclass result: `src/mlframe/feature_selection/_benchmarks/nbins_strategy/tt_b3_reference.py`.
2. Fresh fixtures (20 kinds, correlated / mixed / nonlinear-mean / hetero / composed targets, p in {5, 40}, n in {2000, 20000}, 10 to 12 seeds, 840 cases) show that wave 1 **mostly survives but is optimistic**: family choice 96.2 percent right (404 of 420), null calibrated (LL test fires 14 of 210 = 6.7 percent on exact nulls, final non-identity pick 1 of 210), but a pick can be worse than identity (8 of 840 by more than 2 percent, worst 1.30; wave 1 reported worst 1.000) and mean regret to the oracle is 1.014 on transform fixtures, not the 1.000 to 1.008 of wave 1.
3. Fresh data found **two defects of the wave-1 recipe / my first version**, both fixed in the reference: the clipped inverse of a bounded-image Yeo-Johnson (test error up to 1.2e10 times identity), and an "atom at 0" rule that fired on cube-like targets.
4. The wave-1 headline "the old proposer is biased, replace it" is **true for the proposer alone and misleading for the old pipeline**: the old code (MI shortlist + ridge gate) has 0 harm and 0 false positives; on its own 4-candidate pool it ties the reference. The reference wins on recall and on pools larger than 3 (91.0 vs 71.9 percent good picks on transform fixtures, 0.4 vs 11 to 19 percent missed gains) and costs 3 to 8 times more time.
5. The largest systematic weakness is the choice of the Yeo-Johnson lambda: the density-optimal lambda is not the error-optimal one. An opt-in polish (`tt_b3_ext.polish_lambda`) lifts good picks from 88.0 to 93.5 percent and removes the worst case (1.22 to 1.02) on the p=40 blocks.
6. Trees: the recipe is neutral to slightly harmful for HistGradientBoosting (worse than identity by more than 2 percent in 9.5 percent of RMSE cases); apply it to linear and additive heads only.

## 1. Files

All in `src/mlframe/feature_selection/_benchmarks/nbins_strategy/` unless marked.

| file | role |
|---|---|
| `tt_b3_reference.py` | the reference: `recommend_target_transform`, `analyse_target_transforms`, `decide`, `build_transform`, `TargetTransformChoice`, `TransformAnalysis`, `FittedTransform`. Module docstring = algorithm + porting notes for MRMR |
| `tt_b3_ext.py` | proposed, opt-in extension `polish_lambda` (not part of wave 1) |
| `tt_b3_fixtures.py` | 20 fresh fixtures |
| `tt_b3_judge.py` | the tt_a4 oracle protocol for arbitrary candidate sets, and the three old proposers (current mlframe code loaded from its file) |
| `tt_b3_run.py`, `tt_b3_oldext.py` | drivers (reference + judge; old pipeline on the reference's candidate pool) |
| `tt_b3_analyze.py`, `tt_b3_tables.py`, `tt_b3_misc.py`, `tt_b3_calib.py`, `tt_b3_failures.py`, `tt_b3_timing.py` | tables, null calibration, failure lists, timing |
| `audits/2026-11-23/b3_results/` | raw jsonl per block (`e1a`..`e4b`), `oldext_*.jsonl`, `tables_final.md`, `timing.txt`, `e1a_v0_buggy.jsonl` (first version, kept as evidence of the two defects) |

## 2. The reference (what it implements)

Wave-1 recipe, in order (details and formulas are in the module docstring):

1. Candidates restricted by the range of y: identity, cbrt, signed_sqrt, signed_square, asinh(y/std), Yeo-Johnson always; log if y > 0, log1p if y >= 0, reciprocal `-1/y` if single-signed, logit if y in (0, 1); Box-Cox only on request. cbrt and signed_sqrt are dropped when more than 0.2 percent of y are exact zeros.
2. Random halves A and B. Closed-form ridge (ridge 1e-4 relative for `head="linear"`, 1e-2 for `head="spline"` = additive natural cubic splines on every column with at least 10 distinct values; binary columns stay linear). One solve, one matrix product for the whole candidate matrix.
3. Held-out Jacobian log-likelihood per candidate on B, `LL_f = -0.5 log MSE_B(f) + mean_B log|f'(y)|`, gain over identity. Yeo-Johnson lambda: 17-point grid on A refined by a parabola; `|lambda - 1| < 0.1` is the identity and is not a candidate.
4. Sign-flip residual bootstrap null (40 draws, `y* = clip(fit + s r, min y, max y)`, every draw reruns the lambda fits) for the maximum gain; threshold tau = the ceil((1 - alpha)(n_null + 1))-th smallest draw (finite-sample valid p-value).
5. Every candidate above tau is significant; ordered by gain; a candidate within one paired SE of the best is a tie and the simplest family wins.
6. Second gate: held-out original-scale error of the head (clipped inverse, generalised Duan smearing for RMSE, median prediction for MAE) not worse than identity; first candidate of the order that passes wins, otherwise abstain with a reason string.

Return value `TargetTransformChoice`: `name, params, ll_gain, null_threshold, p_value, err_identity, err_chosen, metric, abstained, reason, best_candidate, gains, transform (forward and clipped inverse), seconds`.

Two corrections that fresh data forced (both are in the final file; the pre-fix run is `e1a_v0_buggy.jsonl`):

* **Bounded-image inverse.** The wave-1 text clips the prediction to the training range of f(y) plus 5 to 10 percent. For Yeo-Johnson with lambda < 0 (or > 2) the margin runs into the asymptote of the inverse; finite but astronomically large values pass the "is finite" retry of tt_a4. In the first run 5 of 156 cases had a test MAE of 1.3e9 to 1.2e10 times identity (all Yeo-Johnson at lambda = -1 or 3), while the held-out gate had passed. Fix: no margin toward the asymptote (Yeo-Johnson, Box-Cox, reciprocal). After the fix the worst MAE ratio is 1.04.
* **Atom rule.** My first rule "more than 0.2 percent of |y| below 1e-9 sd is an atom" fired on `cube` (y = u^3 has a density spike at 0) and removed cbrt in 3 of 12 seeds (asinh picked, regret up to 1.9). Fix: test exact ties (`y == 0`). cube went from 75 to 100 percent good.

## 3. Validation design

* **Fresh fixtures** (`tt_b3_fixtures.py`, no wave-1 generator, coefficient set or noise level): features are correlated (blocks of 4 columns with within-block correlation 0.7, across 0.2); `mixed` mode adds binary columns (every third) and right-skewed columns; coefficients are drawn per (fixture, seed) (random active columns, signs, magnitude U(0.4, 1)); the linear predictor is standardised. Kinds: transform: lognormal, log_comp (log y = 0.7 x_a + 0.35 clip(x_b, -2, 2)^2 + noise, nonlinear mean), gamma_log (Gamma noise), cube, sqrt_pos, recip, logit_bnd, hetero_log, hetero_cube (transform AND heteroscedastic noise), mixed_log (binary + skewed features); ambiguous (the right transform is not offered or atoms): exp_add (y = exp(0.7 eta) + noise, can go negative), zero_infl (25 percent exact zeros), neg_skew (y = 6 - exp(.)); controls: lin, lin_hetero, lin_mixed, lin_t3 (t3 noise), lin_pos (all with linear mean and symmetric noise: "core controls", where the null of the test holds exactly), lin_quad (nonlinear mean), lin_skewnoise (gamma noise).
* **Blocks:** n=2000 p=5 (seeds 0 to 11, 240 cases, search rows 1500), n=20000 p=5 (10 seeds, 200), n=2000 p=40 (10 seeds, 200), n=20000 p=40 (10 seeds, 200; ridge only). 840 cases in total.
* **Protocol:** the proposers see only the fit + calibration rows (75 percent); the judge is the tt_a4 protocol (permutation seed 10000 + seed, 60/15/25 split, clipped inverse, 64-quantile smearing) copied for arbitrary candidates (identity numbers verified equal to `tt_a4_oracle.judge`). The oracle best is the lowest score over every candidate the judge scored: tt_a4's set (identity, cbrt, signed_sqrt, asinh, log, box_cox, log1p, sqrt, logit, sklearn Yeo-Johnson) plus signed_square, reciprocal, the regression-lambda Yeo-Johnson and the reference picks as fitted. A "restricted" oracle (the reference's own set only) is available in the analyzer and gives higher numbers (90.4 vs 83.8 percent good at n=2000 p=5, mean regret 1.014 vs 1.031); the tables use the strict (full) one.
* **Metrics:** ratio = score(pick)/score(identity); regret = score(pick)/score(best); gain case = best/identity <= 0.98; good = regret <= 1.02 on gain cases, ratio <= 1.01 on no-gain cases; harm = ratio > 1.02; miss = identity on a gain case; abstain = identity returned; FP = non-identity pick on a control fixture.
* **Models and metrics:** ridge (head linear) and ridge_spline (head spline) on all blocks except ridge_spline at n=20000 p=40; HistGradientBoosting on the first three blocks with a RESTRICTED candidate set (identity, every pick, the best ridge candidates) because 12 candidates x 840 cases was too slow; RMSE of the smeared mean prediction and MAE of the median prediction.

## 4. Results

### 4.1 Headline (ridge, smeared RMSE, strict oracle), per block

Good = pick within 2 percent of the oracle best (gain cases) or no loss (no-gain cases). `old_search` = current mlframe code (equal-width-MI shortlist of 3 over identity, cbrt, signed_sqrt, log + ridge held-out bootstrap gate); `old_ext_search` = the same code on the reference's candidate pool (control experiment, `tt_b3_oldext.py`).

| block | slice | method | good | mean regret | worst ratio | miss on gain | abstain |
|---|---|---|---|---|---|---|---|
| n=2000 p=5 | transform (120) | reference | 87% | 1.024 | 1.30 | 0% | 0% |
| | | old_ext_search | 76% | 1.026 | 1.02 | 8% | 9% |
| | | old_search | 71% | 1.032 | 1.00 | 9% | 10% |
| | core controls (60) | reference / old_search | 100% / 100% | 1.001 / 1.001 | 1.00 / 1.00 | | 100% / 100% |
| n=20000 p=5 | transform (100) | reference | 94% | 1.008 | 0.96 | 0% | 0% |
| | | old_ext_search | 77% | 1.017 | 1.00 | 9% | 9% |
| | | old_search | 74% | 1.024 | 1.00 | 10% | 10% |
| | core controls (50) | reference / old_search | 100% / 100% | 1.000 / 1.000 | 1.00 / 1.00 | | 98% / 100% |
| n=2000 p=40 | transform (100) | reference | 88% | 1.014 | 1.00 | 0% | 0% |
| | | old_ext_search | 68% | 1.108 | 8.64 | 9% | 9% |
| | | old_search | 67% | 1.031 | 1.00 | 10% | 10% |
| | core controls (50) | reference / old_search | 100% / 100% | 1.000 / 1.000 | 1.00 / 1.00 | | 100% / 100% |
| n=20000 p=40 | transform (100) | reference | 96% | 1.008 | 0.96 | 0% | 0% |
| | | old_ext_search | 80% | 1.035 | 2.59 | 10% | 10% |
| | | old_search | 76% | 1.024 | 1.00 | 10% | 10% |
| | core controls (50) | reference / old_search | 100% / 100% | 1.000 / 1.000 | 1.00 / 1.00 | | 100% / 100% |

Pooled over all 840 cases (ridge, RMSE):

| method | good | mean regret | median regret | worst ratio | harm > 2% | abstain | miss on gain | FP on core controls | good on transform kinds |
|---|---|---|---|---|---|---|---|---|---|
| reference | 87.5% | 1.025 | 1.000 | 1.30 | 1.0% (8 cases) | 32.7% | 0.4% | 0.5% (1/210) | 91.0% |
| old_ext_search | 79.8% | 1.033 | 1.000 | 8.64 | 0.5% | 43.8% | 11.2% | 1.0% | 75.2% |
| old_search (current code) | 74.0% | 1.045 | 1.000 | 1.00 | 0.0% | 49.0% | 18.6% | 0.0% | 71.9% |
| old_mi (proposer top-1 only) | 49.8% | 1.125 | 1.015 | 2.89 | 19.8% | 18.6% | 15.8% | 71.9% | 64.0% |
| old_ext_mi (proposer top-1, extended pool) | 36.2% | 3.83 | 1.044 | 169.6 | 37.4% | 4.3% | 2.9% | 91.9% | 33.6% |

Label accuracy (pick equals the generating transform) on the 10 transform fixtures (420 cases): reference 96.2 percent (404/420; 98.4 percent without log_comp), old_search 79.3 percent (it cannot express reciprocal and logit, so 80 percent is its ceiling), old_ext_search 73.6, old_mi 69.5.

Abstention of the reference by fixture group: transform 0.0 percent, ambiguous 15.9, controls 86.7 (the 13 percent non-abstentions on controls are lin_quad and lin_skewnoise, section 4.4). For comparison old_search abstains 10.0 / 66.7 / 97.3 and misses a gain in 18.6 percent of gain cases.

### 4.2 Per fixture (ridge RMSE; good% / mean regret; columns are n=2000 p=5, n=20000 p=5, n=2000 p=40, n=20000 p=40)

| fixture (oracle best/identity per block) | method | n2000 p5 | n20000 p5 | n2000 p40 | n20000 p40 |
|---|---|---|---|---|---|
| lognormal (.73 .70 .69 .70) | reference | 83/1.011 | 100/1.001 | 90/1.008 | 100/1.000 |
| | old_ext_search | 75/1.027 | 100/1.001 | 90/1.008 | 100/1.000 |
| | old_search | 83/1.011 | 100/1.001 | 90/1.008 | 100/1.000 |
| log_comp (.83 .83 .88 .88) | reference | 42/1.136 | 40/1.079 | 60/1.028 | 90/1.030 |
| | old_ext_search | 67/1.042 | 40/1.071 | 50/1.833 | 90/1.028 |
| | old_search | 50/1.097 | 40/1.071 | 60/1.028 | 90/1.028 |
| gamma_log (.90 .88 .92 .91) | reference | 92/1.008 | 100/1.001 | 90/1.009 | 100/1.002 |
| | old_ext_search | 75/1.020 | 80/1.014 | 30/1.070 | 70/1.019 |
| | old_search | 92/1.008 | 100/1.001 | 90/1.009 | 100/1.002 |
| cube (.49 .52 .48 .52) | all three | 100/1.000 | 100/1.000 | 100/1.000 | 100/1.000 |
| sqrt_pos (.95 in all) | reference | 100/1.001 | 100/1.000 | 100/1.002 | 100/1.000 |
| | old_ext_search | 100/1.005 | 100/1.000 | 100/1.001 | 100/1.000 |
| | old_search | 100/1.001 | 100/1.000 | 100/1.000 | 100/1.000 |
| recip (.76 .74 .74 .74) | reference | 100/1.001 | 100/1.000 | 100/1.000 | 100/1.001 |
| | old_ext_search | 92/1.021 | 100/1.000 | 80/1.020 | 100/1.001 |
| | old_search (cannot express) | 8/1.095 | 0/1.120 | 0/1.117 | 0/1.115 |
| logit_bnd (.96 .96 .95 .96) | reference | 100/1.000 | 100/1.000 | 100/1.000 | 100/1.000 |
| | old_ext_search | 25/1.037 | 10/1.040 | 0/1.053 | 0/1.045 |
| | old_search (cannot express) | 8/1.044 | 0/1.044 | 0/1.053 | 0/1.045 |
| hetero_log (.90 .92 .86 .90) | reference | 67/1.033 | 100/1.001 | 60/1.042 | 100/1.001 |
| | old_ext_search | 67/1.033 | 100/1.001 | 50/1.047 | 90/1.186 |
| | old_search | 83/1.015 | 100/1.001 | 50/1.047 | 100/1.001 |
| hetero_cube (.75 .79 .72 .74) | all three | 100/1.000 | 100/1.000 | 100/1.000 | 100/1.000 |
| mixed_log (.59 .66 .54 .50) | reference | 83/1.052 | 100/1.002 | 80/1.052 | 70/1.048 |
| | old_ext_search | 58/1.076 | 40/1.045 | 80/1.052 | 50/1.067 |
| | old_search | 83/1.052 | 100/1.002 | 80/1.052 | 70/1.048 |
| exp_add, ambiguous (.45 .42 .44 .42) | reference | 0/1.301 | 0/1.271 | 0/1.308 | 0/1.283 |
| | old_ext_search | 100/1.000 | 100/1.000 | 30/1.047 | 30/1.058 |
| | old_search | 0/1.329 | 0/1.352 | 0/1.344 | 0/1.380 |
| zero_infl, ambiguous (.99 in all) | reference | 100/1.003 | 100/1.004 | 100/1.004 | 90/1.012 |
| | old_ext_search | 92/1.007 | 100/1.004 | 80/1.009 | 60/1.013 |
| | old_search | 92/1.007 | 100/1.004 | 80/1.009 | 60/1.013 |
| neg_skew, ambiguous (.78 .81 .80 .82) | reference | 58/1.054 | 80/1.041 | 90/1.004 | 20/1.117 |
| | old_ext_search | 0/1.171 | 0/1.158 | 0/1.153 | 0/1.172 |
| | old_search | 0/1.286 | 0/1.238 | 0/1.259 | 0/1.229 |
| lin, lin_hetero, lin_mixed, lin_t3, lin_pos (controls) | all three | 100/1.000 to 1.002 | 100/1.000 | 100/1.000 to 1.001 | 100/1.000 |
| lin_quad (.96 .96 .99 .99) | reference | 58/1.016 | 100/1.006 | 90/1.009 | 100/1.002 |
| | old_ext_search | 92/1.006 | 100/1.004 | 90/1.008 | 90/1.006 |
| | old_search | 58/1.024 | 90/1.008 | 80/1.012 | 90/1.006 |
| lin_skewnoise (1.00 in all) | reference | 92/1.003 | 100/1.000 | 90/1.003 | 100/1.000 |
| | old_ext_search = old_search | 100/1.000 | 100/1.000 | 100/1.002 | 100/1.000 |

Two things stand out in the table. On exp_add the old pipeline with the marginal-likelihood Yeo-Johnson (`old_ext_search`) is the oracle best at p=5 (100 percent) while the reference's profile-lambda Yeo-Johnson never is (section 6, F3). On log_comp, hetero_log and gamma_log at p=40 the extended old pipeline picks a Yeo-Johnson that is occasionally very bad (regret 1.83, 1.19, 1.07).

### 4.3 Other models and metrics (pooled over the blocks where they were run; good / mean regret / worst ratio)

| model, metric | reference | old_ext_search | old_search | old_mi |
|---|---|---|---|---|
| ridge, MAE of the median | 89.3% / 1.010 / 1.04 | 79.3% / 1.017 / 1.00 | 72.6% / 1.031 / 1.01 | 49.2% / 1.109 / 1.97 |
| ridge_spline (spline head), RMSE | 82.8% / 1.019 / 1.82 | 78.9% / 1.032 / 6.28 | 73.3% / 1.026 / 1.69 | 49.5% / 1.087 / 2.25 |
| ridge_spline, MAE | 89.5% / 1.007 / 1.06 | 82.8% / 1.015 / 1.27 | 74.7% / 1.022 / 1.27 | 53.0% / 1.086 / 1.86 |
| HistGB, RMSE (restricted oracle set) | 74.8% / 1.026 / 2.01, harm 9.5% | 77.1% / 1.015 / 1.22 | 76.4% / 1.015 / 1.14 | 58.9% / 1.030 / 1.31 |
| HistGB, MAE (restricted) | 81.2% / 1.009 / 1.10, harm 5.3% | 77.9% / 1.011 / 1.04 | 76.1% / 1.012 / 1.08 | 57.7% / 1.030 / 1.26 |

Under MAE the reference is better than under RMSE (94.5 percent good on transform kinds) because the median prediction has no retransformation bias. For HistGB the old pipeline is slightly better than the reference (it abstains more); consistent with wave 1 (trees do not profit).

### 4.4 Controls and null calibration

* Exact nulls (core controls: linear mean, symmetric noise incl. t3, hetero, mixed binary features, positive range; 210 cases, linear head): the LL test fires in 14 (6.7 percent; per block 3.3, 8.0, 8.0, 8.0 percent; nominal 5), the second gate vetoes 13 of them, the final answer is non-identity in 1 of 210 (0.5 percent, harmless). The finite-sample p-value cannot go below 1/41 = 0.024 with 40 draws; the transform fixtures sit at that floor (median p 0.02).
* lin_quad (nonlinear mean, symmetric noise): the LL test fires in 100 percent with the linear head and 80 to 100 percent with the spline head. With the linear head the transform helps (oracle 0.96 to 0.99; reference ratio 0.982, max 1.021). With the spline head the identity is the oracle (1.000) and the picks (59 percent non-identity) cost 0.6 percent on average, max 5.4 percent: the Gaussian LL rewards a better tail shape that the additive spline does not need.
* lin_skewnoise (gamma noise, linear mean): the sign-flip null is wrong by construction (skewed noise), the LL test fires in 100 percent; the second gate vetoes most (non-identity 29 percent with the linear head, 53 percent with the spline head, mean ratio 1.000 to 1.001, max 1.019). This is the case the second gate exists for.
* Old pipeline on the same controls: old_search 0.0 percent FP on core controls, 2.7 percent on all controls; old_mi (proposer alone) picks a non-identity transform on 71.9 percent of core controls and loses more than 2 percent on 60.5 percent of them.

### 4.5 Wall time per call (median seconds, the whole machine was at 100 percent CPU from other agents and my own 2 workers, so these are upper bounds)

| block | reference linear | reference spline | old search (MI top-3 + gate) |
|---|---|---|---|
| n=2000 p=5 (1500 rows) | 0.21 | 0.23 | 0.13 |
| n=2000 p=40 | 0.33 | 0.47 | 0.26 |
| n=20000 p=5 (15000 rows) | 2.86 | 3.11 | 0.50 |
| n=20000 p=40 | 3.81 | not run | 1.34 |

`tt_b3_timing.py` (3 repeats, same loaded machine): 15000 x 5, linear head, 40 draws: 4.38 s, of which the analysis without the null is 0.41 s (about 0.1 s per null draw); 15000 x 40: 3.77 s; spline head 4.77 s (p=5) and 7.53 s (p=40); 2000 rows: 0.62 to 0.66 s. Deterministic for a seed: True. The reference is 3 to 8 times slower than the old pipeline; the null dominates, so n_null (>= 19 for alpha 0.05), the row count of the null or the lambda grid are the levers (not tested).

## 5. Head-to-head with the old proposer

Three old variants were scored on the same rows: the current mlframe `propose_target_transforms` top-1 (`old_mi`), the same plus Yeo-Johnson as tt_a4 scored it (`old_mi_yj`), and the full `search_target_transform` (`old_search`); the control experiment `old_ext_*` gives the old code the reference's candidate pool.

* **Proposer alone is as bad as wave 1 said**: 71.9 percent non-identity picks on exact-null controls, 60.5 percent of them harmful (ridge, RMSE); with the extended pool 91.9 percent / 71.9 percent and a worst ratio of 169.6 (Yeo-Johnson on a heavy right tail).
* **The old pipeline is not the old proposer**: the ridge held-out bootstrap gate removes all of that. old_search had no harmful pick in 840 cases and 0 FP on core controls. Reason it works on its pool: `default_transforms()` has only 3 non-identity candidates and `top_k=3`, so the MI "shortlist" gates all of them; the gate does the selection. On fixtures its pool can express (transform kinds minus recip and logit_bnd) it ties the reference at n=20000 (92 vs 92 percent good) and at n=2000 p=40 (84 vs 85), and is a little ahead at n=2000 p=5 (86 vs 83).
* **Where the reference wins**: (a) candidates the old pool cannot express: recip 100 vs 0 to 8 percent good, logit_bnd 100 vs 0 to 8 (0 to 25 even when the old code is handed the logit candidate), neg_skew up to 90 vs 0; (b) recall when the pool grows: with the same extended pool the MI shortlist of 3 misses 9 to 11 percent of gains and old_ext_search reaches 75.2 percent good on transform kinds, the LL + null criterion 91.0 (n=20000: 94 vs 77, 96 vs 80); (c) the old gate needs the whole upper bootstrap limit below zero, which abstains in 10 percent of transform cases even at n=20000 while the reference never abstains on a transform fixture; (d) the old pipeline with the extended pool can be harmful at p=40 (worst ratios 8.64 on log_comp and 2.59 on hetero_log, both Yeo-Johnson with the marginal-likelihood lambda, 4 harmful picks in 840 cases).
* **Where the old pipeline is better**: 3 to 8 times faster; zero harm; on exp_add at p=5 the marginal-lambda Yeo-Johnson of old_ext_search is the oracle best (100 percent good) while the reference's profile lambda is 0 percent (section 6, F5); on HistGB it is slightly better because it abstains more.

## 6. Failure cases and diagnoses (reference, ridge RMSE)

Of 840 cases the reference returned a pick more than 2 percent worse than identity in 8 (list below) and left more than 5 percent on the table in 81 of 526 gain cases (exp_add 42, log_comp 14, neg_skew 11, hetero_log 8, mixed_log 4, lognormal 1, lin_quad 1; with the restricted oracle 51).

| id | case | evidence | diagnosis |
|---|---|---|---|
| F1 | Yeo-Johnson lambda at the grid edge (53 picks: neg_skew 41, log_comp 10, hetero_log 2); 13 percent worse than identity, mean ratio 0.866 | log_comp n2000 s5 ratio 1.298, s3 1.166; neg_skew n20000 p40 s2 1.22, n20000 p5 s5 1.188, n2000 s8 1.142; hetero_log s8 1.031 | The profile likelihood keeps rising up to lambda = -1 (right-skewed heavy target) or 3 (the reflected neg_skew, whose right transform -log(c - y) is not offered). The transform becomes reciprocal-like with a bounded image and a very steep inverse; the held-out gate has only 750 rows and passes it. A lambda on the edge means the family is too small: it should be reported as a warning in `target_transform_` (not done) |
| F2 | log_comp (nonlinear mean + linear head): 40 to 60 percent good at p=5 and n=2000 p=40, 90 percent at n=20000 p=40 | pick log (or Yeo-Johnson at the edge); e.g. n2000 s2 log 0.874 vs box_cox 0.635, s6 0.908 vs 0.758 | A linear head cannot represent the x_b^2 term, so the transform that makes f(y) most Gaussian (log) is not the one that minimises the original-scale error; a milder power is better. Old pipelines fail identically (40 to 50 percent), so this is a property of "linear head on nonlinear mean", not of the criterion |
| F3 | exp_add: 0 percent good in all four blocks, regret 1.27 to 1.31 | pick Yeo-Johnson with profile lambda about -0.3 (ratio 0.51 to 0.60); the error-optimal lambda is about -0.5 to -0.6 (ratio 0.40 to 0.52) | The density-optimal lambda is not the RMSE-optimal one on a skewed additive target. The tt_a4 Yeo-Johnson with the marginal-normality lambda does better (old_ext_search 100 percent good at p=5) |
| F4 | hetero_log at n=2000: 60 to 67 percent good, one harmful pick (n2000 s0, log, ratio 1.068) | all four of log, YJ, cbrt, log1p have LL gain above tau; test RMSE order differs | heavy-tailed RMSE on 750 held-out rows in the gate and 500 test rows in the judge is noisy; disappears at n=20000 (100 percent good) |
| F5 | mixed_log: 70 to 83 percent good, regret 1.05 | pick log (the generating transform); e.g. n2000 s7 log 0.602 vs box_cox 0.387, s8 log 0.778 vs Yeo-Johnson 0.735 | likely the skewed columns exp(0.7 z) make log y non-linear in the raw features, the same mechanism as F2 (feature transforms and target transform interact: wave-1 agent 2's topic); not isolated by a separate run |
| F6 | lin_quad with spline head: 59 percent non-identity, mean ratio 1.006, max 1.054 | LL fires although identity is the oracle | LL rewards marginal tail shape (section 4.4); small, bounded cost |
| F7 | 2 misses on gain cases (lin_quad n2000 p40 s2, neg_skew n20000 p40 s9) | the gate vetoed every significant candidate (lin_quad: asinh 1.082 and Yeo-Johnson 1.077 vs identity 1.073 held-out RMSE; neg_skew: Yeo-Johnson 1.305 vs 0.77) | gate noise on 750 held-out rows (neg_skew: the edge-lambda inverse really is bad on that split) |
| F8 | first version only: cube asinh picks (3 of 12 seeds), MAE ratios up to 1.2e10 | section 2 | fixed |

## 7. Variants and the opt-in polish

Decision-rule variants on the same analyses (ridge, RMSE, strict oracle): the wave-1 rule (LL order, tie rule, fall-through) 87.5 percent good over 840 cases; strict argmax without fall-through 87.1 (misses 2.1 percent instead of 0.4); no tie rule 87.6 (worst ratio 1.51); order by held-out error instead of LL 85.5 (worst ratio 1.64); 2-fold cross-fit (n=2000 p=5, 240 cases) 85.4 vs 84 (same). None of the variants changes the picture; the second gate is the only protection that mattered.

`polish_lambda` (`tt_b3_ext.py`): when Yeo-Johnson wins, re-choose lambda on a 33-point grid by the held-out original-scale error of the same head. It was designed after F3 appeared on the p=5 blocks and confirmed on the two p=40 blocks (new random streams, 400 cases; 2 more exp_add smoke seeds 100, 101 at p=5: ratio 0.60 to 0.52 and 0.57 to 0.44):

| metric | reference | with polish |
|---|---|---|
| ridge RMSE good | 88.0% | 93.5% |
| mean regret / worst ratio | 1.0242 / 1.22 | 1.0099 / 1.02 |
| ridge MAE good | 92.0% | 97.5% |
| ridge_spline RMSE (200 cases) | 81.5% | 83.0% |
| exp_add good | 0% | 85% |
| neg_skew good | 55% | 70% |

The grid error is computed on the rows that served for the gate (mildly optimistic); I recommend it with that caveat.

## 8. Verdict: do the wave-1 numbers survive fresh data?

* **Survives**: the choice of the family (96.2 percent right on 10 fixtures; wave 1: 55 of 56 = 98 percent); null calibration (6.7 percent vs 5 percent nominal; final FP 0.5 percent, harmless; wave 1: 0 of 14 to 1 of 28); no overflow after the inverse fix; zero misses on transform fixtures; MAE improvements; independence from the shortlist; abstention on controls (86.7 percent; 100 percent on core controls at three of four blocks).
* **Does not survive as stated**: (a) "never worse than identity" (wave 1: worst ratio 1.000; here 1.30 at n=2000, 1.22 at n=20000, 8 of 840 cases above 1.02, all Yeo-Johnson at the lambda edge or heavy-tailed noise); (b) regret to the best candidate 1.000 to 1.008 (here 1.014 on transform kinds, 1.117 on ambiguous ones, driven by the lambda, F3, and by linear heads on nonlinear means, F2/F5); (c) the case against the old code: the pipeline's gate makes it safe, the reference is better on recall and pool size and not on its own pool; (d) accuracy at n=2000 (1500 search rows) is clearly lower than at 20000 (87 vs 94 percent good on transform kinds).
* **New**: two spec-level defects in the recipe text (inverse clip toward the asymptote; exact-zero atom rule) and the dependence of the best lambda on the error metric, with a tested repair.
* The wave-1 numbers were measured on one fixture family with independent standard-normal features and a hand-picked set of target kinds; on fresh data the ranking of the methods did not change (reference > old pipeline with equal pool > old pipeline > proposer alone), the margins did.

## 9. Not run, limits

Not run: classification; LightGBM and CatBoost; real data; n above 20000; p above 40; noise levels other than the fixed one per fixture; more than 12 seeds; ridge_spline at n=20000 p=40; HistGB at p=40 n=20000 and a full (unrestricted) HistGB oracle; 2-fold cross-fit outside n=2000 p=5; polish at p=5 beyond the two smoke seeds; box_cox as a candidate; the null on a row subsample (speed lever); float32. The oracle is one random 25 percent test block per case (500 rows at n=2000), so the 2 percent threshold of "good" is itself noisy at n=2000. Timings were taken on a fully loaded machine. The fixtures are mine, written after reading wave 1; "fresh" means unseen by the wave-1 agents, not independent of my expectations.

## 10. Reproduce

```
set CUDA_VISIBLE_DEVICES= ; OMP_NUM_THREADS=1 ; NUMBA_NUM_THREADS=2 ; PYTHONPATH=D:\Temp\wt2\src
cd src\mlframe\feature_selection\_benchmarks\nbins_strategy
python tt_b3_run.py --n 2000 --p 5 --seeds 12 --models ridge,ridge_spline,hgb --folds2 1 --fixtures <transform kinds> --out b3_results\e1a.jsonl
python tt_b3_run.py --n 20000 --p 40 --seeds 10 --models ridge --heads linear --folds2 0 --polish 1 --fixtures <kinds> --out b3_results\e4a.jsonl
python tt_b3_oldext.py b3_results            # old pipeline on the reference pool
python tt_b3_tables.py b3_results ; python tt_b3_misc.py b3_results ; python tt_b3_failures.py b3_results\e*.jsonl
python tt_b3_timing.py
```

Usage of the reference:

```python
import tt_b3_reference as R
ch = R.recommend_target_transform(X, y, seed=0, head="linear", metric="rmse")   # head="spline" for an additive mean
if not ch.abstained:
    z = ch.transform.forward(y)                       # fit the model on z
    yhat = ch.transform.inverse(zhat, residuals=resid)  # clipped inverse + smearing
else:
    print(ch.reason)
```
