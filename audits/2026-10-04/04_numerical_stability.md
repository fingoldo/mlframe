# 04 Numerical stability and precision (mlframe, read-only audit, 2026-10-04)

## Scope and method
Existing tracker: audits/full_audit_2026-09-20 and mrmr audits were consulted by name only (the k=3,4 and centred-variance fixes in
calibration/_independence_check.py, pre_screen.py, _usability_njit_pool.py, shap_proxy_prefilter_univariate.py are CLOSED and not re-reported).
Patterns grepped over src/mlframe: raw power sum variance (`sumsq`, `total_sq / count - mean*mean`, `ss - cnt*mean*mean`) at any k>=2;
additive epsilons (`+ 1e-(6..12)` ~ 150+ hits, mostly FE transformers); nan_to_num (134 hits, NOT individually read); f32 mean/sum accumulation;
softmax without shift (0 naive hits). Verified by reading: 3 power-sum hits (3 true positives), 4 epsilon hits (2 demonstrated).
Probe: python, numpy only, inputs below. Coverage is partial: GPU/cupy kernels, searchsorted/rank tie order, int32 index overflow, errstate leaks were NOT swept.

## Findings
| ID | file:line | sev | evidence | probe | fix | Disposition |
|---|---|---|---|---|---|---|
| N1 | feature_selection/filters/_temporal_agg_fe.py:215 | P1 | `var = (run_sumsq[g] - cnt * mean * mean) / (cnt - 1)`; `run_sumsq[g] += v * v` (running std stat_code 2) | x=1.7e9+N(0,1), n=1000: true std 0.977, got 0.0 (rel err 100%); sd=0.01: true 0.0102, got 0.0 | Welford running mean/M2 per group (order-dependent already, so no parallel concern) | OPEN |
| N2 | feature_selection/filters/_temporal_agg_fe_rolling.py:79 | P1 | `var = (ss - cnt * mean * mean) / (cnt - 1)` with `ss += v * v` | same probe as N1 (epoch-second values, the stated use case): 0.0 vs 0.977 | centred two-pass over the in-window items, or Welford | OPEN |
| N3 | feature_engineering/entity_inter_event.py:101 | P1 | `var = total_sq / count - mean * mean` ; `stds[i] = np.sqrt(var) if var > 0.0 else 0.0` | same data: 0.0 vs 0.977 (negative noise clipped to 0.0 silently) | Welford; buffer `buf[:count]` is already sorted-kept, can do two-pass | OPEN |
| N4 | feature_engineering/anchor.py:93, :562 | P2 | `slope = num / (den + 1e-12)` with den = sum dx^2 | pos spacing 1e-7 (den~1e-14), true slope 2: got 0.18 (91% error); fine for unit-scale positions | scale-relative guard: `slope = num/den if den > tiny*scale else 0` | OPEN |
| N5 | feature_engineering/hurst.py:255,264,272,317,562,622 | P3 | `num / (den + 1e-12)`, `np.log(f_s + 1e-12)` | polyfit slope 1e-7 for t=0..7: den=42 so error 2e-13 relative; only bites for tiny den | replace with explicit den>0 check | OPEN |
| N6 | feature_engineering/ensemble_features.py:145-146 | P3 | `hi = arr.max(...) + 1e-9`, `span = (hi - lo) + 1e-12` | with |values|>~1e7 the 1e-9 pads are absorbed in float64 (span stays 0 for constant rows; division 0/0 avoided only by +1e-12) | span = max(hi-lo, tiny) relative | OPEN |
| N7 | feature_engineering/spectral.py:273,302,421,463; spatial.py:286,313; bayesian.py:220,487,645,658 | P3 | additive `+ 1e-12` on sums/innovation variance | not probed; corrupts values only if the denominator is < ~1e-9 (e.g. normalised spectra of tiny amplitude, Kalman gain with variance ~1e-12) | relative eps or where(den>0) | OPEN |
| N8 | ~25 files in feature_engineering/transformer/*.py | P3 | `y.std() + 1e-9`, `+ 1e-9` on norms/stds | unprobed; harmless unless target scale < 1e-6 | skip unless unit-scale assumption is wrong | OPEN (low value) |

## Verdict
Real: N1-N3 are the SAME bug family as the closed k=3,4 / centred-variance fixes: three independent running-variance implementations in the
temporal-aggregation and inter-event feature paths still use raw power sums; any timestamp-like or large-offset column (epoch seconds, prices
at ~1e5 with cent moves) yields std 0.0, silently. Fix all three together with a shared Welford helper (feature_engineering/_numerical_stable.py
already documents the technique, line 38; check it is reusable inside njit).
Not worth fixing: N8 and most N7 (unit-scale assumption, features are rank/tree consumed). nan_to_num (134 sites) deserves a dedicated pass,
not done here. Unswept: GPU kernels, int32 overflow, tie ordering, errstate, test tolerances.
