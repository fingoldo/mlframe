# Wave 25 audit 2026-11-04 -- tracker

Three read-only reports by Opus 5.5 low-effort agents on areas no earlier wave read: `feature_engineering/transformer` and `_numerical_counts`, the composite transforms and discovery kernels, and the engineered-recipe / pair-emit modules of the MRMR filters. The agents could not write files and each covered only part of its scope (the lists of unread files are at the end), so the reports are their returned text, condensed; each finding was re-checked on HEAD before it was fixed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | W25NUM-1 | stats | Low | `compute_nunique_mode_quantiles_numba` returned a NaN mode whenever the last run of the sorted array had length 1, even if an earlier value repeated; the test is now on the best run length |
| **RESOLVED** | W25NUM-2 | stats | Low | `compute_nunique_modes_quantiles_numpy` counted NaN as a distinct value and could pick it as the mode while its quantiles skipped NaN, and crashed on an all-NaN array; the counts, modes and crossings now run on the finite values |
| **RESOLVED** | W25NUM-3 | stats | Low | a sign change spanning a NaN was lost by `compute_ncrossings` (the NaN reset the previous difference); the NaN-path caller now passes the finite values |
| **FUTURE** | W25NUM-4 | precision | Medium | `squared_dists` (`transformer/_squared_dists_shared.py`) uses `\|a\|^2 - 2a.b + \|b\|^2` in float32, which loses all precision for close points far from the origin |
| **FUTURE** | W25CMP-1 | leakage | Medium | the centred rolling-ratio mode reads base rows up to i+(k-1)//2, which leaks the target when base is a lag-1 column; old params without `mode` default to centred |
| **FUTURE** | W25CMP-2 | alignment | Medium | frac_diff drops non-finite `history_y` values instead of keeping their positions, shifting every older lag |
| **FUTURE** | W25CMP-3 | nan | Medium | one NaN in y makes frac_diff T NaN for the next `lags` rows while the domain mask flags only the NaN row |
| **FUTURE** | W25CMP-4 | alignment | Low | the rolling-ratio continuation tail is built from the finite base values, so NaNs among the last k-1 rows shift the stored window |
| **FUTURE** | W25CMP-5 | binning | Low | `screening._prebin` computes `nanquantile` over a column that still holds +-inf, so inf / NaN edges follow |
| **RESOLVED** | W25REC-1 | replay | High | factorize replay of a pandas Categorical with integer categories skipped the code map (its `to_numpy()` is an integer array of category VALUES) and clipped those values to the top bin, a constant feature at transform time; a non-empty map is now always used, with an integer-valued float falling back to the integer spelling |
| **RESOLVED** | W25REC-2 | replay | Medium | `pd.NA` was not recognised as missing at replay of a string-dtype column, so it went to the sentinel bin instead of the NaN code |
| **FUTURE** | W25REC-3 | emit | Medium | multi-emit survivors are recorded as `(cfg, 0)` and returned as a set, so downstream filters test column 0 and can raise `TypeError` |

## Not covered by this wave

- transformer package: all but `_aggregation.py`, `_squared_dists_shared.py` (about 127 files); `_timeseries_emit.py` was not found.
- composite transforms: `categorical.py`, `extended.py`, `interaction_bases.py`, `linear.py`, `naming.py`, `nonlinear.py`, `registry.py`, `shared.py` and the `_causal_anchor`, `_domain_shared`, `_grouped_extra`, `_multi_extra`, registry and transform-base modules; discovery: `_calibration_gate`, `_filter_and_gate`, `_honest_rmse_gate`, `_mrmr_base_rank`, `_screening_tiny*`, `_tiny_rerank*`, `_yscale_holdout_gate`.
- MRMR filters: `engineered_recipes/shared.py` and four more recipe modules, every `_dynamic_cluster_discovery` file but `_dcd_swap.py` (partly), every `_feature_engineering_pairs` file but `_pairs_emit.py` (partly).

## Implemented

| ID | Regression test |
|---|---|
| W25NUM-1..3 | `tests/feature_engineering/test_numerical_counts_nan_and_mode_fixes.py` |
| W25REC-1, 2 | `tests/feature_selection/fe/test_factorize_replay_of_integer_categoricals.py` |
