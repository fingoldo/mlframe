# Wave 26 audit 2026-11-04 -- tracker

Three read-only reports by Opus 5.5 low-effort agents on the files wave 25 left unread: the remaining composite transform modules, the discovery gates and tiny-rerank modules, and the dynamic cluster discovery / pair-scoring / recipe modules of the MRMR filters. The agents could not write files, so the reports are their returned text, condensed; each finding was re-checked on HEAD before it was fixed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | W26PAIR-1 | null | Medium | `operand_uplift_null_bar` always permuted the second raw operand, so a strong second operand destroyed the signal the bar measures against and the bar fell to 1.0; `operand_uplift_null_bar_for_pair` now permutes the operand with the smaller marginal MI whichever slot it is in |
| **RESOLVED** | W26DCD-1 | polars | Low | `_resolve_member_matrix` read `dtype.kind` and called a pandas-only `to_numpy(dtype=, copy=)`, which raised on a polars frame |
| **RESOLVED** | W26TRF-1 | `tests/training/composite/test_oof_target_encoding_survives_zero_sample_weights.py` |
| W26GATE-3 | `tests/training/composite/discovery/test_multiseed_early_stop_only_under_union_consensus.py` |
| W26GATE-2 | `tests/training/composite/discovery/test_honest_gate_passes_groups_to_grouped_transforms.py` |
| W26GATE-1 | grouped | High | on the val-split path the y-scale gate called every `requires_groups` transform with `groups=None` for the fit rows (the refit skip reused the same variable), so each grouped spec was rejected as 'fit/inverse raised'; the fit rows now get their group labels from the training frame while the fold-local refit stays off (`tests/training/composite/discovery/test_yscale_gate_forwards_grouped_transforms_on_val_split.py`) |
| **RESOLVED** | W26GATE-2 | grouped | High | the honest RMSE gate forwarded and inverted every spec without group labels, so a grouped transform raised and was rejected; `_spec_holdout_prediction` now takes the labels from the discovery's row-aligned group ids (and reports 'needs group ids' when there are none), and `spec_inverse` accepts them (`tests/training/composite/discovery/test_honest_gate_passes_groups_to_grouped_transforms.py`) |
| **RESOLVED** | W26GATE-3 | gate | Medium | the sequential seed early-stop reports the median of the seeds seen when it stops, which is >= the threshold but not the true median; the min across families (`union`) cannot be changed by that, the mean across families (`borda`) can (a spec that passes on the true numbers was rejected); the threshold is now infinite (every seed runs) unless `tiny_consensus == "union"` (`tests/training/composite/discovery/test_multiseed_early_stop_only_under_union_consensus.py`) |
| **FUTURE** | W26GATE-4 | gate | Medium | the structural-fragility ratio uses an unweighted variance of group means, so tiny groups push it to ~1 for any base |
| **FUTURE** | W26GATE-5 | split | Low | rows with a NaN group id never reach the holdout |
| **FUTURE** | W26GATE-6 | seeds | Low | tiny-screen seed repeats collapse to one when groups are supplied even when the split falls back to a seed-dependent KFold |
| **RESOLVED** | W26TRF-1 | leakage | High | the zero-weight wrapper dropped zero-weight rows before `target_encoding_residual.fit`, so its out-of-fold fingerprint covered fewer rows than the forward later received, the fingerprint check failed and the forward used the in-sample encoding (each row's own y in its encoding); transforms with `oof_train_forward` are no longer wrapped, and the fit (which documents `sample_weight` as unused) keeps every row (`tests/training/composite/test_oof_target_encoding_survives_zero_sample_weights.py`) |
| **FUTURE** | W26TRF-2 | seeds | Medium | grouped ewma / rolling-ratio / frac_diff forward seeds differ from the inverse seeds under recurrence continuation |
| **FUTURE** | W26TRF-3 | domain | Medium | `centered_ratio` has no pole guard for a predict-time base at or below zero |
| **FUTURE** | W26TRF-4 | roundtrip | Low | `reciprocal_residual` does not round-trip |y| below its eps floor |
| **FUTURE** | W26TRF-5 | dtype | Low | `linear` / `_causal_anchor` forward and inverse do not cast to float64 |
| **FUTURE** | W26TRF-6 | stats | Low | the James-Stein noise estimate counts groups that are not in the shrinkage set |
| **FUTURE** | W26TRF-7 | domain | Low | the predict-time domain mask of `requires_base=False` transforms is the wrong length for a `None` base |

## Not covered by this wave

- discovery gates: the helpers `_honest_oof_select`, `_fold_refit`, `_lgb_shared_fold`.
- composite transforms: `_nonlinear_ewma_fracdiff.py`, `simple.py`, `unary.py` (partly read in wave 25), `_registry_extended.py`, `_registry_shared_adapters.py`, the wrapper that calls fit / forward.
- MRMR filters: `_dynamic_cluster_discovery/__init__.py`, `_dcd_metrics.py`, `_dcd_swap_stages.py`; every `_feature_engineering_pairs` file but `_pairs_code_bins.py` and `_pairs_operand_floor.py`; `engineered_recipes/shared.py`, `_orth_basis_recipes.py`.

## Implemented

| ID | Regression test |
|---|---|
| W26GATE-1 | `tests/training/composite/discovery/test_yscale_gate_forwards_grouped_transforms_on_val_split.py` |
| W26PAIR-1, W26DCD-1 | `tests/feature_selection/fe/test_wave26_operand_null_and_polars_member_matrix.py` |
