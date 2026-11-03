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
