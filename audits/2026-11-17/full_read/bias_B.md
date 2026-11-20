# bias_B: selection-bias items

## (a) 48-fut HybridSelector optimistic holdout - RESOLVED
Synthetic: n=1200, 30 cols (f0,f1 additive, f2*f3 product, 26 noise), 4 seeds.
Fit-on-all vs train-only (mean per fit): engineered-column holdout FI sum 0.404 -> 0.299 (-26%); raw-noise FI sum 0.056 -> -0.014; noise columns selected 3.25 -> 1.5.
Fix: `fit_on_train_only=True` default; MRMR + tree signals fit on the train rows of the shared 30% split (split computed once, `_holdout_split`); `False` = legacy. Compute lower (70% rows; ~52 s/fit).
Test: tests/feature_selection/test_hybrid_selector_fit_on_train_only.py (FI drop >=10%; passed, 8 min).

## (b) 05-09 hybrid_cat_pair_fe - RESOLVED (null-scaled floor)
Max noise ii_mm over 5 seeds x 28 pairs: card10 n=1000 0.028, n=3000 0.0071; card4 n=1000 0.0084 (all > 0.001 default). Real parity interaction ii_mm 0.16-0.20.
Noise max / MM bias = 0.5-1.9 across all cells. A rows-per-cell guard cannot reject the card4 n=1000 case (31 rows/cell, noise 0.0084), so the null-scaled floor wins.
Fix: keep iff ii_mm > max(min_interaction_info, null_floor_mult*bias), `null_floor_mult=2.5` default (0 = legacy). Noise pairs admitted 0 vs >=1 legacy; real pair kept 5/5.
Test: tests/feature_selection/fe/test_biz_val_cat_pair_null_floor.py; neighbouring cat-pair tests 13 passed.

## (c) 07-10 conditional gate tau optimism - RESOLVED (material only at small n)
Pure noise, p=12, 10 seeds: n=1000 0/8, n=300 0/10 false admissions (baseline margin rejects first); n=150 5/10.
Nested null (tau search repeated inside each of the 12 permutations, `_nested_tau_null_hi`): n=150 1/10.
Fix: `nested_tau_null=True` default on cheap_conditional_gate_scan; False = legacy null. Only candidates that clear the baseline margin pay the extra cost.
Test: tests/feature_selection/fe/test_conditional_gate_selection_bias.py. Existing gate suites: 44 passed, 2 failed (test_biz_val_conditional_gate_end_to_end_auc_lift, test_row_argmax_detects_on_argmax_regression_target); both fail identically on clean origin/master (pre-existing, not caused here; power not otherwise regressed by these tests).

## 07-5 ratio sign drop - RESOLVED (real defect)
`u / max(|v|,1e-6)` evaluates u/|v|, so the best-existing-op floor missed a signed ratio. y=(a/b>1), n=4000: floor 0.213 (old) -> 0.55 (signed); old floor under-states by 2.6x, letting a gate beat a baseline that omitted the true ratio.
Fix: `_signed_safe_denominator` (|v|<1e-6 -> copysign(1e-6,v)) in both host paths and the 3 CUDA/cupy copies in _resident_candidate_mi.py (GPU copies not executed here, no CUDA).
Test: test_signed_safe_denominator_keeps_the_sign, test_baseline_floor_sees_a_signed_ratio_target (same file as (c)).

## mypy / interrogate
mypy: no issues in the 4 touched source files. interrogate: 100% (105/105) on them and the 3 new test files.
