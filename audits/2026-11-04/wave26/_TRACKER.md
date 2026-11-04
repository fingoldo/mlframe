# Wave 26 audit 2026-11-04 -- tracker

Three read-only reports by Opus 5.5 low-effort agents on the files wave 25 left unread: the remaining composite transform modules, the discovery gates and tiny-rerank modules, and the dynamic cluster discovery / pair-scoring / recipe modules of the MRMR filters. The agents could not write files, so the reports are their returned text, condensed; each finding was re-checked on HEAD before it was fixed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | W26PAIR-1 | null | Medium | `operand_uplift_null_bar` always permuted the second raw operand, so a strong second operand destroyed the signal the bar measures against and the bar fell to 1.0; `operand_uplift_null_bar_for_pair` now permutes the operand with the smaller marginal MI whichever slot it is in |
| **RESOLVED** | W26DCD-1 | polars | Low | `_resolve_member_matrix` read `dtype.kind` and called a pandas-only `to_numpy(dtype=, copy=)`, which raised on a polars frame |
| **RESOLVED** | W26TRF-1 | `tests/training/composite/test_oof_target_encoding_survives_zero_sample_weights.py` |
| W26GATE-5 | `tests/training/composite/discovery/test_group_disjoint_carve_handles_missing_group_ids.py` |
| W26TRF-5 | `tests/training/composite/test_linear_residual_float32_inputs_use_float64_arithmetic.py` |
| W26TRF-6 | `tests/training/composite/test_grouped_linear_residual_shrinkage_ignores_failed_groups.py` |
| W26GATE-6 | `tests/training/composite/discovery/test_tiny_screen_seed_repeats_survive_a_group_split_fallback.py` |
| W26GATE-4 | `tests/training/composite/discovery/test_structural_gate_ignores_tiny_groups_noise.py` |
| W26GATE-3 | `tests/training/composite/discovery/test_multiseed_early_stop_only_under_union_consensus.py` |
| W26GATE-2 | `tests/training/composite/discovery/test_honest_gate_passes_groups_to_grouped_transforms.py` |
| W26GATE-1 | grouped | High | on the val-split path the y-scale gate called every `requires_groups` transform with `groups=None` for the fit rows (the refit skip reused the same variable), so each grouped spec was rejected as 'fit/inverse raised'; the fit rows now get their group labels from the training frame while the fold-local refit stays off (`tests/training/composite/discovery/test_yscale_gate_forwards_grouped_transforms_on_val_split.py`) |
| **RESOLVED** | W26GATE-2 | grouped | High | the honest RMSE gate forwarded and inverted every spec without group labels, so a grouped transform raised and was rejected; `_spec_holdout_prediction` now takes the labels from the discovery's row-aligned group ids (and reports 'needs group ids' when there are none), and `spec_inverse` accepts them (`tests/training/composite/discovery/test_honest_gate_passes_groups_to_grouped_transforms.py`) |
| **RESOLVED** | W26GATE-3 | gate | Medium | the sequential seed early-stop reports the median of the seeds seen when it stops, which is >= the threshold but not the true median; the min across families (`union`) cannot be changed by that, the mean across families (`borda`) can (a spec that passes on the true numbers was rejected); the threshold is now infinite (every seed runs) unless `tiny_consensus == "union"` (`tests/training/composite/discovery/test_multiseed_early_stop_only_under_union_consensus.py`) |
| **RESOLVED** | W26GATE-4 | gate | Medium | the structural-fragility ratio was the unweighted variance of group means over the total variance, ~0.5-1 for ANY base when groups hold one or two rows (30k rows in 20k groups), so a base with no group structure was rejected; it is now the adjusted between-group share (within-group vs total mean squares), ~0 for an i.i.d. base and unchanged for a real per-group level (`tests/training/composite/discovery/test_structural_gate_ignores_tiny_groups_noise.py`; the existing per-well-base gate tests pass) |
| **RESOLVED** | W26GATE-5 | split | Low | `_carve_group_disjoint` kept every NaN group id as a separate `np.unique` entry and tested `nan in set`, which is always False, so rows with a missing group id were never held out; missing ids are now one group of their own (`tests/training/composite/discovery/test_group_disjoint_carve_handles_missing_group_ids.py`) |
| **RESOLVED** | W26GATE-6 | seeds | Low | the tiny screen collapsed seed repeats to one whenever groups were supplied, although with fewer groups than `cv_folds` the screen falls back to a seed-dependent shuffled KFold; the collapse now needs at least `cv_folds` distinct groups (`tests/training/composite/discovery/test_tiny_screen_seed_repeats_survive_a_group_split_fallback.py`) |
| **RESOLVED** | W26TRF-1 | leakage | High | the zero-weight wrapper dropped zero-weight rows before `target_encoding_residual.fit`, so its out-of-fold fingerprint covered fewer rows than the forward later received, the fingerprint check failed and the forward used the in-sample encoding (each row's own y in its encoding); transforms with `oof_train_forward` are no longer wrapped, and the fit (which documents `sample_weight` as unused) keeps every row (`tests/training/composite/test_oof_target_encoding_survives_zero_sample_weights.py`) |
| **DOC** | W26TRF-2 | seeds | Medium | the grouped ewma / rolling-ratio / frac_diff forward seeds from the per-group mean (or a cold window) while the inverse seeds from the train tail under `recurrence_continuation`; this is the package-wide convention of the recurrent transforms (documented on the ungrouped ones in wave 25: forward on the train sequence, inverse continuing it), the gap decays within `k` rows per group, and the continuation flag is opt-in |
| **DOC** | W26TRF-3 | domain | Medium | `centered_ratio` is `y / (base + c)`; the fit already places the train minimum a full scale unit above the pole (its docstring records why), a predict base at or below `-c` is a base outside anything the model saw, and the sign flip there is the transform's mathematics rather than a missed guard; the estimator's soft base shrink and OOD guards are the lever for such rows |
| **DOC** | W26TRF-4 | roundtrip | Low | `reciprocal_residual` floors |y| at `eps_y` = 1e-6 x median|y| in the forward by design (the floor keeps 1/y finite), so a y more than six orders below the median does not round-trip; the floor is documented in the forward's docstring |
| **RESOLVED** | W26TRF-5 | dtype | Low | the linear and causal-anchor residuals did their arithmetic in the inputs' dtype, so float32 inputs near 1e8 lost the residual to float32 spacing; forward and inverse now compute in float64 (`tests/training/composite/test_linear_residual_float32_inputs_use_float64_arithmetic.py`) |
| **RESOLVED** | W26TRF-6 | stats | Low | the grouped linear residual's James-Stein noise estimate accumulated residuals and rows of groups whose OLS failed (they carry the global fit, not a fit of their own) although those groups are excluded from the shrinkage set; the accumulation is now inside the `_fit_ok` branch (`tests/training/composite/test_grouped_linear_residual_shrinkage_ignores_failed_groups.py`; fails 900 vs 840 rows on the old accumulation) |
| **REJECTED** | W26TRF-7 | domain | Low | the predict-time domain mask of `requires_base=False` transforms is built from `len(base)`; the estimator feeds a zeros placeholder of length n (`_estimator.py`, `_from_fitted.py`), so a `None` base never reaches it on the supported path |

## Not covered by this wave

- discovery gates: the helpers `_honest_oof_select`, `_fold_refit`, `_lgb_shared_fold`.
- composite transforms: `_nonlinear_ewma_fracdiff.py`, `simple.py`, `unary.py` (partly read in wave 25), `_registry_extended.py`, `_registry_shared_adapters.py`, the wrapper that calls fit / forward.
- MRMR filters: `_dynamic_cluster_discovery/__init__.py`, `_dcd_metrics.py`, `_dcd_swap_stages.py`; every `_feature_engineering_pairs` file but `_pairs_code_bins.py` and `_pairs_operand_floor.py`; `engineered_recipes/shared.py`, `_orth_basis_recipes.py`.

## Implemented

| ID | Regression test |
|---|---|
| W26GATE-1 | `tests/training/composite/discovery/test_yscale_gate_forwards_grouped_transforms_on_val_split.py` |
| W26PAIR-1, W26DCD-1 | `tests/feature_selection/fe/test_wave26_operand_null_and_polars_member_matrix.py` |
