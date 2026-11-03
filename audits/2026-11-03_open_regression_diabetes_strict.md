# Open: diabetes strict-path selection collapse (pre-existing, not from the 2026-11-03 speed commits)

`tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_rc2_undersampled.py::test_rc2_diabetes_strict_path_also_recovers_after_discrete_fe_gating`
fails: `MRMR(random_state=42, fe_max_steps=0, use_simple_mode=False, fe_confirm_undersample_rows_per_cell=0.0)` on `load_diabetes` selects
`['s5', 'bmi']`; the test needs >= 5 (it was >= 6 until 2026-08-11).

Established:
- Fails on the pushed master `90f565473` (checked in a separate worktree) and with this session's device paths forced off
  (`MLFRAME_BOOTSTRAP_RANK_TIED_DEVICE=0 MLFRAME_MODULAR_CODES_DEVICE=0 MLFRAME_BINAGG_NULL_BATCHED=0`), and on CPU only (`CUDA_VISIBLE_DEVICES=-1`).
- Independent of the Fleuret-confidence compaction (`c907650ba`): same result with the compaction replaced by a pass-through.
- `random_seed=42` (the test's spelling) and `random_state=42` select the same.
- Not the permutation budget alone: `0da19d5a6` (2026-10-13) raised `full_npermutations` 3 -> 99 (max_failed stays 1); with `full_npermutations=3` the fit
  selects only `['s5']`, fewer than at 99.
- Screen trace (min_relevance_gain 0.0023): Z=[] confirms s5 (0.203); Z=[s5] confirms bmi (0.149); Z=[s5, bmi] leaves sex 0.024, s1 0.039 and one more at
  0.044, all rejected by the conditional confirmation (n=442: a 10-bin three-way joint has ~1000 cells).
- The other 4 tests of the file pass (the default, non-strict path recovers diabetes).

Next: compare the confirmation statistic for those candidates against the June 17 commit `b49063618`, when the test was written and passed.

## Update (same day)

- The 2026-08-11 commit `689a02770` (the last recalibration of this test) selects `['s5', 'bmi', 'bp', 's3', 's6']` on the same call: a real regression after it.
  Screen log there: full_npermutations=3, gains s5 0.231 (conf 0.75), bmi 0.098, bp 0.062, s3 0.058, s6 0.053. Today: full_npermutations=99, s5 0.203,
  bmi 0.149, and every other candidate's gain given {s5, bmi} is 0 or rejected - the gain estimates themselves changed, not only the budget.
- Found and fixed on the way (`_residual_rescue_stage._target_mix_candidates`): the residual-rescue target mix fitted a child MRMR forced to fe_max_steps=1,
  gated only on `fe_target_mix_enable`, so engineered pair/triplet searches ran under fe_max_steps=0. Not the cause of the collapse (the main screen had
  already returned 2 features); the selection is unchanged by the fix.
- Next: compare the binned codes / gain of bp given (s5, bmi) between `689a02770` and today.
- Root difference found: the MDLP feature binning the screen receives changed. `factors_nbins` of (age, sex, bmi, bp, s1, s2, s3, s4, s5, s6, target):
  `689a02770` [5, 2, 2, 2, 5, 5, 2, 5, 4, 2, 10]; today [5, 2, 3, 2, 2, 2, 2, 5, 3, 2, 10] (s1 and s2 collapse 5 -> 2 bins, s5 4 -> 3, bmi 2 -> 3).
  With s1/s2 at 2 bins and the conditioning set finer, every remaining candidate's gain given {s5, bmi} falls under the strict confirmation.
- Next: find which MDLP change since 2026-08-11 merges s1 / s2 into 2 bins on diabetes (n=442) and whether that is the intended stricter split validation.
