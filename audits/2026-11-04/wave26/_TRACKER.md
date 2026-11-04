# Wave 26 audit 2026-11-04 -- tracker

Three read-only reports by Opus 5.5 low-effort agents on the files wave 25 left unread: the remaining composite transform modules, the discovery gates and tiny-rerank modules, and the dynamic cluster discovery / pair-scoring / recipe modules of the MRMR filters. The agents could not write files, so the reports are their returned text, condensed; each finding was re-checked on HEAD before it was fixed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | W26PAIR-1 | null | Medium | `operand_uplift_null_bar` always permuted the second raw operand, so a strong second operand destroyed the signal the bar measures against and the bar fell to 1.0; `operand_uplift_null_bar_for_pair` now permutes the operand with the smaller marginal MI whichever slot it is in |
| **RESOLVED** | W26DCD-1 | polars | Low | `_resolve_member_matrix` read `dtype.kind` and called a pandas-only `to_numpy(dtype=, copy=)`, which raised on a polars frame |
| **FUTURE** | W26GATE-1 | grouped | High | `_yscale_holdout_gate`: on the val-split path the fit side calls the transform with `groups=None`, so every `requires_groups` transform is rejected |
| **FUTURE** | W26GATE-2 | grouped | High | `_honest_rmse_gate`: forward and inverse are called without `groups`, so every grouped transform raises and is rejected |
| **FUTURE** | W26GATE-3 | gate | Medium | the tiny-screen seed early-stop reports an inflated median, which under `borda` aggregation inflates the cross-family mean |
| **FUTURE** | W26GATE-4 | gate | Medium | the structural-fragility ratio uses an unweighted variance of group means, so tiny groups push it to ~1 for any base |
| **FUTURE** | W26GATE-5 | split | Low | rows with a NaN group id never reach the holdout |
| **FUTURE** | W26GATE-6 | seeds | Low | tiny-screen seed repeats collapse to one when groups are supplied even when the split falls back to a seed-dependent KFold |
| **FUTURE** | W26TRF-1 | leakage | High | `categorical` target-encoding: with any zero sample weight the zero-weight wrapper fits on fewer rows than the forward sees, the fingerprint check fails and the full-train (non-out-of-fold) encoding is used |
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
| W26PAIR-1, W26DCD-1 | `tests/feature_selection/fe/test_wave26_operand_null_and_polars_member_matrix.py` |
