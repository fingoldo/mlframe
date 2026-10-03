# MRMR prewarp/compound failures, deep-nightly d085d62f7

Hypothesis (prewarp rebind rule in _pairs_setup.py) REFUTED. Real root cause: e56a6ca71 (joint-gate operand floor, `_beats_the_larger_operand`,
uplift 1.10 over the larger operand marginal) rejected genuine additive/interacting pairs whose partner carries joint information (y = a**2/b + g/k + log(c)*sin(d)),
so raws g/k were subsumed/dropped and noise `e` leaked into a compound.

Experiments: rebind rule replaced by `return not has_binding`: tests 2,3,4 still failed. Floor set to 0.0: test 3 selection returned to the pre-e56a6ca71 result.

Fix: `_pairs_score.py::_beats_the_larger_operand` takes `pair_mi`; the floor is waived when the raw pair's joint MI itself exceeds floor*1.10 (partner adds joint information).
Noise-degraded copies (joint ~ strong operand) are still rejected. Regression tests: tests/feature_selection/mrmr/fe/test_fe_joint_gate_operand_floor.py (3 tests).
Black also reformatted a few unrelated spots in _pairs_score.py.

Per test (reproduced on current tree / after fix):
1. test_clean_library_form_preferred_over_monotone_prewarp: not reproduced (passes before and after; the rebind rule actually helps it).
2. f2 [with_outliers]: reproduced; passes after fix.
3. usability_raw_retention: reproduced (selection lacked g,k raws); fixed, selection now identical to pre-e56a6ca71 (verified by script; test itself not re-run, selection script equals test setup).
4. fe_pair_path_selection_bit_identical: reproduced; STILL FAILS (support [7,14,19,23,29] vs pinned [7,13,14,15,19,23,26,29]). Stale before e56a6ca71 too: parent commit gives [7,14,19,23,26,29]. Not caused by prewarp or the gate. Needs its own bisect / re-capture decision; left untouched.
5. endtoend_invariants I4b and I5 [ratio_plus_trig-lognormal-regression-s305-fe2]: not run separately as failures before; both pass after fix (run with -k, 7 passed 1 skipped).

## Test 4 follow-up: test_fe_pair_path_selection_bit_identical

Pin [7,13,14,15,19,23,26,29] was captured 2026-07-25 (cd5c01909). Bisect in a scratch worktree (first-parent range cd5c01909..0ce96af65, every probe run twice, no flakiness):
last good 4dcc18522, FIRST BAD 153c2e9b9 (2026-08-10, "Recalibrate rankgauss task-axis fixture to a Cauchy tail"), where support becomes [7,14,19,23,29].
That commit's src change is in _fe_auto_escalation.py (rescue pairs get a half-reserved budget instead of absolute priority) and info_theory/_group_mi.py (bin cap).
Later states drift further: 0ce96af65 gives [7,14,19,23,26,29], current tree [7,14,19,23,29]. So the pin has been stale since 2026-08-10, long before the prewarp rebind rule
and e56a6ca71; neither is involved (verified: rebind rule off and gate off do not restore the pin).

Evidence on regression vs intended (scratchpad/ev.py, make_classification n=4000 p=30 inf=8 seed=0):
- the 8 pinned indices are exactly the 8 columns with MI 0.148-0.181 (next best 0.015): they ARE the informative features. The new support drops 3 of them (13, 15, 26).
- HistGB 3-fold CV on raw support columns: pinned 0.938, current 0.801, all 30 columns 0.920.
- the dropped raws are consumed as operands of crude engineered columns on a purely linear-informative fixture: sub(log(f_15),log(f_13)), mul(log(f_15),sign(f_14)),
  mul(log(f_13),sign(f_19)), add(log(f_13),sign(f_26)) (appended unscored, support_rank -1, from the later FE rounds) and add(add(f_7,neg(f_26)),esc_poly_laguerre_mul(f_14,f_19)).
Decision: this is a REGRESSION (informative raws lost to degraded compounds), not an intended change, so the pin must NOT be re-captured.
Not fixed here: the mechanism (which later FE round admits the log/sign compounds that subsume informative raws, and why 153c2e9b changes the escalation outcome) needs its own
investigation; the test stays red as the correct sensor. Scratch worktrees removed, no commits.

## Test 4 resolution

Mechanism. Reverting only the pair-budget hunk of 153c2e9b9 (_fe_auto_escalation.py) restores the pinned support exactly; the bin-cap hunk in _group_mi.py is unused here (no groups).
With the half-reserved budget, escalation now processes high-joint-MI non-rescue pairs, among them (f_14, f_19), two informative raws. The proposers fit a Pearson-validated
warp against the target RANK, which for a 4-class target is the arbitrary label order, and admitted esc_poly_laguerre_mul(f_14,f_19). That column is a second, token-disjoint half for
the C2 additive fusion, which fused it with add(f_7,neg(f_26)) and registered f_26 (and via the fused/log compounds f_13, f_15) as subsumed raws (_raw_redundancy_dropped_ = {f_13,f_15,f_26}),
so the support lost 3 of the 8 informative features. A superadditive-joint-MI filter on the extra pairs was tried and rejected: genuinely informative pairs and noise pairs both sit at 1.2-1.5x the marginal sum.

Fix (production). Escalation is skipped for a nominal multiclass target: `_is_nominal_multiclass_target` (integer/bool 1-D y with 3..20 distinct values, in
_mrmr_fit_impl/_fit_impl_stages/_inputs.py) sets `_fe_escalation_nominal_target_` in `_stash_fe_targets`; `_skip_for_nominal_target` in _fe_auto_escalation.py records
info["skipped"] and run_fe_auto_escalation returns []. Binary, continuous and many-valued integer targets are untouched, so the starvation test added by 153c2e9b9 (regression-style target) still passes.
Result: test_fe_pair_path_selection_bit_identical passes with the ORIGINAL pin [7,13,14,15,19,23,26,29]; the pin is not re-captured.
Note: the flag is a bool assigned at the start of every fit and not reset in _finalise.py (that file has other agents' uncommitted edits).

Regression tests: tests/feature_selection/fe/adaptive/test_fe_escalation_nominal_target.py (detection matrix, skip for nominal, still runs for ordinal).

Size gates. The operand-floor helper `_beats_the_larger_operand` moved to the new sibling src/.../_feature_engineering_pairs/_pairs_operand_floor.py (AST unresolved-name check clean, black clean);
_pairs_score.py was restored to HEAD plus a 2-line change (import + `pair_mi` argument), 1293 lines, with the black-induced churn removed. The nominal guard in
run_fe_auto_escalation costs 0 lines (folded into `if not failed_pairs or _skip_for_nominal_target(...)`), keeping it at its 345-line ceiling.

Run (-n 1): test_long_functions_do_not_grow, test_no_mlframe_file_exceeds_1k_loc, test_fe_escalation_nominal_target.py (9), test_fe_joint_gate_operand_floor.py (3),
fe/adaptive/test_fe_auto_escalation.py, test_fe_pair_path_selection_bit_identical: all pass (final re-run of the meta + nominal tests: 11 passed). No commits.
