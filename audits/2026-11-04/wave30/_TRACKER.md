# Wave 30 -- stale hybrid-FE tests found while checking the recipe-freeze change

Not an agent wave: four tests in `tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_fe_hybrid_orth` failed with and without the recipe-freeze change. Each was traced to the cause before changing it.

| Disposition | ID | Class | Severity | Text |
|---|---|---|---|---|
| **RESOLVED** | W30HT-1 | stale test | Low | `test_each_fold_fits_hybrid_recipes_fresh` required a `c1__*` basis column; the out-of-fold warp family now engineers the same signal as `oofwarp(c1)`; the assertion requires a feature of `c1`, still failing if the slice engineers nothing and still requiring the two slices to differ |
| **RESOLVED** | W30HT-2 | stale test | Low | `TestLogSignalRecovered` required the |x| signal in `mi_greedy_features_`; on seed 1 the pair family builds `max(sqrt(x), sin(noise_c))` first and the greedy constructor then finds `log_abs(x)` redundant (its gate spent 0.0 s, no ledger record for `log_abs(x)`); both tests accept any recipe built from `x`, as the second test's own comment already allowed |
| **RESOLVED** | W30HT-3 | stale test | Low | `test_corrupted_transform_name_raises_actionable` needed a `mi_greedy_transform` recipe; the ratio fixture now yields a `unary_binary` recipe, whose transform-name field is `unary_names`; the corruption raises a `KeyError` naming the bad function, which the test asserts for both kinds |
| **RESOLVED** | W30HT-4 | stale test | Low | `test_corrupted_src_names_raises_or_recovers`: for a `unary_binary` recipe unresolved sources give a documented neutral all-zero column with a loud warning, not an exception; the test now asserts that the output is not the original-looking values for that kind and keeps the raise assertion for `mi_greedy_transform` |
