# Mixing the target with features inside MRMR: brainstorm, synthetic suite and first measurements (2026-10-11)

Follow-up to `target_transform_mrmr_composite.md` after the owner's two objections: (1) y is not always quantile-binned; (2) the idea was wider than a unary transform of y: **mix the target with a base built from the features MRMR already has, and vary the base through MRMR's own feature transforms**, so that the target mix, the feature transform and the feature search are chosen together.

Scripts: `src/mlframe/feature_selection/_benchmarks/target_lens/exp_target_mix.py` (this document), `exp_composite_mi_gain.py` (see `composite_mi_gain_experiment.md`). Raw outputs of the runs: `D:/Temp/agent_brain/target_mix_run{1,2,3}.txt` (not in the repository; the tables below are copied from them).

## 1. The two objections, answered

**(1) Is y always quantile-binned?** No. `quantization_method="quantile"` is the default but the method is a constructor parameter (an equal-width method is not rank-invariant); small-cardinality integer y and class labels are used as codes (`_y_encoding.py:27-110`); and several stages do not bin y at all but use the continuous copy `_fe_prewarp_y_continuous_` on the fence-clipped scale (offset product, row statistics, residual rescue, usability pool). A monotone transform of y is therefore a no-op only for the plain MI path; for those stages it is not. The conclusion of the first document stands for the rank-MI path and is wrong as a blanket statement.

**(2) Does mixing y with a feature column change the verdict?** Yes, and for a precise reason. `T = y - u(x_i)` is a function of `(y, x_i)`, so the joint information is unchanged: `I(T; X | x_i) = I(y; X | x_i)`. What changes is what the **low-order estimators of the selector can see**:

* the plug-in MI screens columns and pairs one or two at a time in rank bins; a dominant term `10 sin(3 x1)` fills the bins of y and a weak term of another pair is invisible (the rank-space masking measured in the residual rescue work: MI 0.07 against 1.85);
* subtracting the dominant term turns a high-order dependence `y ~ f(x1) + g(x2, x3)` into a pair-level dependence `T ~ g(x2, x3)` that the existing pair search can score;
* for gradient boosting the offset removes the large smooth part that would otherwise consume the tree resolution (the composite gain of 11-15% measured in `composite_mi_gain_experiment.md`);
* for a **linear** model a free coefficient on the base is a superset of the offset, so what the mix buys a ridge is not the offset but the features the second search finds.

This is backfitting in MI space: alternate between "find the term given the others" for the target and for the features. It differs from `training/composite` in two ways: the base is not an external column but any column or engineered feature the selector built, and the question asked is "which feature does the mixed target reveal", not "is the composite worth training".

## 2. Variants considered

| id | variant | note |
|---|---|---|
| V1 | residual of a model on the selected features as the target (boosting) | measured as arm B below: adds almost nothing, the downstream model already absorbs it |
| V2 | **feature-base mix**: `T = y - beta u`, `u` = a column or engineered feature of the current selection (or a unary transform of one), beta by OLS on even rows | arm C; the proposal of the owner |
| V3 | ratio mix `log y - log u` or `y / u` for positive y and u | the composite module owns it; only worth it for positive pairs |
| V4 | two-term base `u(x_i) + w(x_j)` | greedy matching pursuit: repeat V2 with the residual as the new target (forward stagewise) |
| V5 | scale mix `T = y * g(x)` (variance stabilising) | helps estimation of the mean under heteroscedasticity; no gain in point metrics by itself |
| V6 | group-normalised target `y - mean_g(y)` | needs groups; belongs to the composite module |
| V7 | quantile-regression residual (median based) | robust to the outliers that carry the variance; matches MAE as the metric |
| V8 | sign / tail indicator targets | information about T(y), not about the mean of y; no point-metric reward |
| V9 | stability ensemble: union of selections over perturbed targets or row subsamples | **the placebo arm D below shows that part of every gain is this effect** |
| V10 | linear-usability scoring of the extra columns | the owner's decision metric; the residual rescue already scores in this space |

## 3. Synthetic suite and what each process tests

Eight processes (`PROCESSES` in the script), `n_train = 6000`, `n_test = 20 000`, five standard normal columns, 6 to 8 seeds, downstream ridge and HistGB on the union of the engineered outputs, held-out RMSE, plus the held-out correlation of the best added column with the planted weak term ("recovery").

| process | truth | tests | H1 (mixing is valuable) predicts | H0 predicts |
|---|---|---|---|---|
| masked_interaction | `10 sin(3 x1) + 0.7 x2 x3` | weak product under a dominant sine | arm C recovers x2 x3, arm A does not | no difference between C and the placebo |
| masked_ratio | `10 sin(3 x1) + 0.8 x2^2 / (1.5 + x3^2)` | same with a ratio the vocabulary builds less easily | C recovers the ratio, A and D do not | no difference |
| derived_base | `8 tanh(2 x1) + 0.8 x2 x3` | the base is a transform of a column, not a column | C improves the ridge, the placebo does not | no difference |
| two_dominant_weak_pair | `6 sin(3 x1) + 6 x2^2 + 0.6 x3 x4` | two bases needed | C improves | no difference |
| regime | `3 x1` if `x4 > 0` else `3 x2 x3` | a switch the target mix cannot express | C no better than the placebo | no difference |
| monotone_control | `exp(0.8 x1 + 0.5 x2)` | a monotone transform of an additive signal | rank-MI path unchanged; mixing may still help ridge | no change |
| heteroscedastic | `x1 x2 + (0.3 + abs(x3)) noise` | a variance-only signal | extra columns, no point-metric gain | extra columns, no gain |
| noise_control | `3 x1 + 0.5 noise` | false positives | nothing added | nothing added |

Arms: **A** `MRMR(X, y)`; **B** residual-of-HGB target; **C** feature-base mix (the 3 best-correlated columns of A, raw or engineered, each used as the base of one `MRMR(X, y - beta u)`); **D** placebo for C (the same bases randomly permuted, so the target keeps y's signal plus noise of the same scale); **E** more search on the identity target (`fe_max_steps = 3`). The union of A and the extra outputs is what the downstream models get.

## 4. Results

Paired change of the held-out RMSE against A (median over seeds; in brackets the paired t in standard errors). Negative is better.

| process (seeds) | ridge C | ridge D (placebo) | HGB C | HGB D | recovery A / C / D | extra columns C / D |
|---|---|---|---|---|---|---|
| masked_interaction (8) | -41.1% (-2.4) | -58.9% (-3.6) | -11.4% (-2.6) | -8.3% (-2.3) | 0.94 / 1.00 / 0.94 | 10 / 3 |
| masked_ratio (8) | -61.7% (-2.7) | -46.0% (-1.3) | -18.7% (-2.7) | -16.9% (-2.7) | **0.02 / 0.85 / 0.05** | 12 / 4 |
| derived_base (8) | **-62.0% (-5.5)** | -5.7% (-0.4) | -0.9% (-1.2) | 0.0% | 1.00 / 1.00 / 1.00 | 6 / 2 |
| two_dominant_weak_pair (8) | -25.4% (-1.8) | -13.9% (-1.2) | -7.4% (-1.4) | -5.1% (-1.0) | 0.94 / 1.00 / 0.94 | 10 / 5 |
| regime (8) | -16.3% (-6.0) | -16.5% (-3.8) | -3.3% (-0.4) | -3.8% (-0.5) | n/a | 6 / 6 |
| monotone_control (8) | **-53.1% (-13.8)** | -22.6% (-3.4) | -6.3% (-3.0) | -2.0% (-1.1) | n/a | 8 / 2 |
| heteroscedastic (6) | 0.0% | 0.0% | **+2.2% (+3.2)** | +0.6% | n/a | 1 / 1 |
| noise_control (6) | 0.0% | 0.0% | **+1.7% (+26.7)** | 0.0% | n/a | 1 / 0 |

Other arms (6 seeds, first runs): **B** (residual of a model) changes the held-out RMSE by -0.0% to -2% except the monotone control (ridge -12.9%) and +2.2% on the heteroscedastic HGB; **E** (more search) adds 0 columns and changes nothing in any process. Time per fit at n = 6000: A 2.5-5 s, B +5 s, C +17-27 s (three full fits), E +3 s.

## 5. What the measurements say

1. **Informative mixing has a clean positive signal in three processes and the placebo does not reproduce it**: `derived_base` (ridge -62% against -5.7%), `masked_ratio` (the weak ratio is recovered with correlation 0.85 by C, 0.02 by A, 0.05 by D) and `monotone_control` (ridge -53% against -23%). These are exactly the situations of the owner's argument: the base is a transform of a column the selector already has, and a dominant term hides a weak one.
2. **A large part of every gain is generic second-pass diversity**: the placebo D, whose bases carry no information, takes -59% (ridge) on `masked_interaction`, -14% on `two_dominant`, -17% on `regime`. A union of selections over perturbed targets is a cheap stability ensemble and the first search leaves the ridge short of usable features. This is a separate finding (V9) with its own host: it should be tested against the usability pool and a row-subsample ensemble, and it must not be credited to the mix.
3. **The mix does not help where the signal is not in the mean**: `heteroscedastic` gets extra columns and a *worse* HGB error (+2.2%, 3.2 se), `regime` is a tie with the placebo. The earlier acceptance criterion (a MAE win on the heteroscedastic and regime targets) was wrong for a point-prediction metric and is replaced below.
4. **The cost of unfiltered extra columns is real**: `noise_control` gets one extra column and a 1.7% worse HGB error; the placebo adds none. The acceptance must be calibrated against the placebo (below), not "every column the second pass selects".
5. **B and E are not worth building**: the residual of a model (B) and more search (E) change nothing material.
6. **Cost**: three full second-pass fits is 3-4x the time of A. A production host cannot afford that; it must reuse what is cached (the candidate code vectors are the same for every target; only the y codes change).

## 6. Design that follows (for the plan)

* **Host**: the pair / operator scan and the offset product, not whole extra `MRMR` fits. Per base `u`: `T_u = y - beta u` (beta by OLS on the even rows), its quantile codes once (one pass), the candidate code vectors reused from the identity scan, K small accumulations per candidate.
* **Bases**: the m best columns of the current selection (raw or engineered) and unary transforms of them from the selector's own vocabulary (V2), then greedy repetition on the residual (V4) with a hard cap.
* **Null and acceptance** (no fixed bars): the placebo of this document is the null: permute the base, rebuild `T`, rerun the whole grid, take the maximum over bases and candidates (the whole-grid null of the offset product work); a candidate enters only if its odd-row paired gain over the identity target beats the null at a family-wise level and its held-out linear OOS gain is positive (the owner's metric). Parameters (beta, edges) from even rows only.
* **Outputs**: ordinary engineered features with recipes that replay with no y reference (the mix is a search device, the feature is built from X alone), so nothing downstream changes and K = 1 is byte-identical.
* **Default**: ON (owner's decision), because the corrective operators of this project are on by default; the placebo-calibrated acceptance is what makes that safe, and a control test on pure noise at three sizes must show the nominal false-positive rate before it ships.
* **Separate items**: the ensemble effect (V9), the corrected `mi_gain` gate of `training/composite` (see `composite_mi_gain_experiment.md`), and feeding a residual target into the unary/binary operator search and the offset product (finding T5 of the first document).

## 7. Open questions

* Does the placebo-calibrated acceptance keep the +1.7% harm of `noise_control` and the +2.2% of `heteroscedastic` out? (First step of the implementation; the script already contains the placebo arm.)
* How much of arm C survives when the bases are limited to a cap of 3 and the hosts are the cached scans (no extra full fits)? Needs the cached-scan prototype.
* Is the diversity effect of D reproducible with a row-subsample ensemble at one third of the cost? If yes, ship that first: it is independent of any target mix.
* The planted weak terms are product and ratio of two columns; a test on a weak term that needs a three-column feature would show whether the mix helps beyond the pair level.
