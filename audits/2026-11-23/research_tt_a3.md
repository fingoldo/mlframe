# research_tt_a3: Jacobian-corrected likelihood as a target-transform criterion

Code: `src/mlframe/feature_selection/_benchmarks/nbins_strategy/tt_a3_jacobian_ll.py`. Raw log: `audits/2026-11-23/tt_a3_full.txt` (56 cases = 8 fixtures x 7 seeds, 20000 rows, 8 features, 70/30 split, CPU, 1 OMP thread). Smoke log: `tt_a3_smoke.txt`.

## Method
- Candidates (domain-checked on y): identity, cbrt, asinh; for y>0 also log, sqrt, log1p, reciprocal, Box-Cox; Yeo-Johnson always. Lambdas are fitted on the training rows by the regression profile likelihood (not marginal normality).
- Criterion: held-out mean of log N(f(y); x b, s2) + log|f'(y)|, b by OLS on train. Version "hetero": log-variance regressed on x (2 WLS passes, constant-corrected).
- Judge: ridge on f(y), prediction clipped to the train range of f(y), Duan smearing (100 residuals), RMSE on the original scale. HistGB (100 iter) for the hetero-LL pick only.
- Gate: sign-flip permutation test on the paired per-row held-out LL difference (best non-identity vs identity), Bonferroni over candidates, alpha 0.01. No fixed margin.
- Baseline: the existing equal-width MI proposer (`mi`).

## Theory: how LL relates to MI
For the same f, LL_f = n/2 log(1/(1-R2_f)) + [marginal Gaussian LL of f(y), with the Jacobian]. The first term is the Gaussian MI of f(y) with x. The Jacobian cancels in that gain (it is in both the model and its null), so it lives entirely in the second term. Checked numerically (seed 0 logs): the sum of the two terms equals the pooled LL, e.g. identity on linear -0.218 vs 1.430 - 1.649.
- Rank-based MI (and the Gaussian-copula MI, -0.5 log(1-R2) of normal scores) is the same for every monotone f. It sees neither term's dependence on f.
- LL differs from MI exactly by (a) how linear f(y) is in x (R2_f) and (b) how Gaussian f(y) is marginally (the Jacobian reward). They disagree whenever f changes linearity or tail shape: cube (cbrt makes R2 jump, GC-MI constant), lognormal (log), and heavy tails (asinh raises Gaussianity: GC-MI 0.34 nats for all f, yet LL moves from -2.82 to -1.69).
- Disagreement that is spurious: LL rewards ANY gain in density fit, including a gain in marginal Gaussianity of the tail that does not reduce the mean-squared error (heavy fixture).

## Results (ridge judge; ratio = RMSE(picked)/RMSE(identity), mean over 7 seeds; "best k/7" = pick equals the judge-best name; regret = RMSE(picked)/RMSE(best))
| fixture | oracle ratio | LL homo: best, ratio, regret | LL hetero: best, ratio | LL + sign-flip gate: best, ratio, #non-identity | EW-MI: best, ratio |
|---|---|---|---|---|---|
| cube | 0.482 | 7/7, 0.482, 1.000 | 7/7, 0.482 | 7/7, 0.482, 7 | 7/7, 0.482 |
| lognormal | 0.626 | 2/7, 0.631, 1.008 | 2/7, 0.631 | 2/7, 0.631, 7 | 0/7, 2.313 (overflow) |
| hetero_lognormal | 0.951 | 4/7, 0.953, 1.002 | 3/7, 0.955 | 3/7, 0.955, 7 | 0/7, 6.122 |
| exp_linear | 0.943 | 2/7, 0.945, 1.003 | 3/7, 0.949 | 3/7, 0.949, 7 | 0/7, 1.051 |
| reciprocal | 0.655 | 2/7, 0.658, 1.005 | 0/7, 0.660 | 0/7, 0.660, 7 | 3/7, 0.659 |
| linear (control) | 1.000 | 1/7, 1.003 (5 false picks) | 4/7, 1.003 (4 false) | 6/7, 1.000, 0 false | 0/7, 1.299 (7 false) |
| hetero_linear (control) | 1.000 | 2/7, 1.004 (5 false) | 4/7, 1.001 (3 false) | 7/7, 1.000, 0 false | 0/7, 1.115 (7 false) |
| heavy (no right transform) | 1.000 | 0/7, 1.021 (7 picks) | 0/7, 1.021 | 0/7, 1.021, 7 | 0/7, 1.032 |

"best k/7" understates quality for lognormal / exp_linear / reciprocal: the LL pick is Box-Cox with lambda near 0 or -1 (equivalent to log / reciprocal) and regret is 1.002-1.008. HistGB with the LL-hetero pick: ratio 0.977-1.003 on all fixtures (the transform is irrelevant for trees; not a regression).

## Verdict
1. The Jacobian LL picks a transform whose ridge RMSE is within 0.8 percent of the judge-best on all seven fixtures with a true transform or null (regret 1.000-1.008), with no overflow, including lognormal where the EW-MI proposer gave RMSE 2.3x identity and hetero_lognormal (6.1x). Clipping plus smearing remove the overflow in the judge.
2. False positives on linear controls: the raw LL argmax picks Yeo-Johnson (lambda ~1.01, practically identity) in 3-5 of 7 seeds; harmless (ratio 1.001-1.004) but still a non-identity pick. The sign-flip gate gives 0 of 14 false positives (EW-MI: 14 of 14).
3. Failure modes seen: (a) heavy-tailed target: LL chooses asinh/cbrt (Gaussianity reward), the gate accepts it 7 of 7 and ridge RMSE is worse by 2.1 percent. LL answers "which density fits", RMSE answers "which mean"; they differ under heavy tails. The gate does NOT protect here, because the LL gain is real. (b) heteroscedasticity: the homoscedastic LL is mildly biased (hetero_lognormal, exp_linear: hetero version is not better, 3/7 vs 4/7 and 3/7 vs 2/7; the gain of the variance model shows only in the linear control, where it removes the false picks). (c) Box-Cox needs y>0 (domain check); Yeo-Johnson lambda is estimated cleanly but its inverse is unsafe without clipping; reciprocal is only valid for single-signed y; cbrt Jacobian is singular at 0 (floored at 1e-6). (d) the judge differences among log / Box-Cox / reciprocal near-ties are inside noise (regret <1 percent).
4. Cost (per case, 20000 x 8, 1 thread): LL for all candidates incl. lambda fits 0.25-0.65 s; EW-MI 0.2-0.55 s; the ridge+HistGB judge 2-7 s (HistGB dominates). So the LL proposer costs about the same as the MI one and is closed-form.
5. Not run: heavier noise levels, p>8, n=100000, LightGBM/CatBoost, a confirmation of the gate's type-I rate with more than 7 seeds x 2 controls (0 of 14 is an upper bound near 20 percent at 95 percent confidence).

## Algorithm proposed
1. Stage 1 (proposer, closed form): for each domain-valid f compute the held-out (even/odd split) Jacobian LL, hetero version when a variance-model F-test on log r^2 vs x is significant, else homoscedastic. Fit Box-Cox / YJ lambda by the profile on the train half. Rank by LL.
2. Stage 2 (model-free confirmation): paired sign-flip permutation test on the per-row held-out LL difference between the top candidate and identity, Bonferroni over candidates; accept only if p < alpha/K. No fixed margin; this removed all linear false positives.
3. Stage 3 (guard for the heavy-tail failure): additionally require the held-out RMSE of the linear head on the original scale (clipped inverse + smearing) not to be worse than identity under a paired bootstrap, or require a linear head downstream; the LL alone must not authorise a transform whose purpose is mean prediction. Report the Gaussian-MI part (R2 gain) and the marginal part separately in `target_transform_`: if the gain comes only from the marginal part, flag as "tail-shape only".
4. Combine with MI: use rank MI / GC-MI as the transform-free reference (it bounds the achievable LL gain: for any f, the linear gain n/2 log(1/(1-R2_f)) <= n * I(y;X) only for the Gaussian-copula part); EW-MI is dropped as proposer (biased) and kept at most as a diagnostic. Trees: do not offer a transform (HistGB ratio 0.98-1.00 everywhere).
