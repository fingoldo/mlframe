# MRMR audit wave 2026-11-20 -- master tracker

Four read-only agents (A core correctness, B FE families and recipes, C GPU / resources / concurrency, D API / tests / performance), 22 findings, reports in this folder. The main session implements every fix itself;
each fix carries a sensor that fails on the pre-fix code. Coverage caveats: agent A did not trace the honest-null device kernels or the CMI / FE gates line by line; agents A and C found fewer than the 8-14 requested.

Statuses: FIXED (code changed, sensor named), WONT-FIX (reason given), TODO (open).

| Finding | Severity | Status | Where / sensor |
|---|---|---|---|
| A-F01 id-keyed content-hash memo | High | FIXED | `_mrmr_fingerprints.py` weakref slot; `test_mrmr_xhash_recycled_id.py` |
| B-F01 float32 subtractions on offset columns | High | FIXED | `_ratio_delta_fe.py`, `_grouped_agg_fe.py`, `_grouped_coerce_shared.py`; `test_float64_deviation_for_offset_columns.py`; benchmark selection identical |
| D-F01 identity fast path vs permuted support_ | High | FIXED | `_mrmr_validate_transform.py`; `test_mrmr_transform_identity_fastpath_identity.py` |
| D-F02 fast-path test stub sorted only | Medium | FIXED | same test file (permuted + boolean mask cases) |
| A-F02 substitution without the add-one ceiling | Medium | FIXED | `_confirm_predictor.py`; `test_prefer_engineered_substitution_ceiling.py` |
| A-F03 SU marginal against a nat floor | Medium | FIXED | `_screen_predictors_gate.py`; `test_screen_predictors_gate_boundaries.py` |
| A-F04 missing marginal bypasses the maxT floor silently | Low | FIXED (warning) | same file / test |
| A-F05 nominal vs occupied bins in the abs/rel floors | Low | WON'T FIX | reason in the report |
| B-F02 lagged diff replay without history | Medium | FIXED | `_ratio_delta_fe.py` `lag_history`, recipe `history`; `test_float64_deviation_for_offset_columns.py` |
| B-F03 modular product wraps int64 | Medium | FIXED | `_pairwise_modular_fe.py`; same test file |
| B-F04 deadline truncation silent | Medium | FIXED (warning) | `_fe_deadline.warn_deadline_truncation`; `test_fe_deadline_truncation_warns.py` |
| B-F05 lagged diff NaN -> 0 | Low | WON'T FIX | finite-output contract, reason in the report |
| B-F06 integer cache keyed on 64 rows | Low | FIXED | `_integer_columns._column_checksum`; same test file |
| C-F01 edge pool drops device / allocator | Medium | FIXED | `_edge_pool._capture_device_scope`; `test_gpu_resource_audit_c.py` |
| C-F02 shared-memory limit | Low | FIXED | `_null_max_perm_gpu.py` |
| C-F03 label cache | Low | FIXED | two keys + fit-end releaser |
| C-F04 unreadable device = fits | Low | FIXED | `_fe_stage_concurrency._stages_fit_device` |
| C-F05 cleanup error hides stage error | Low | FIXED | `_fe_stage_concurrency._run` |
| D-F03 vacuous assertion | Low | FIXED | `test_mrmr_transform_identity_fastpath_identity.py` |
| D-F04 fe_optimizer fallback | Low | FIXED | `_step_core_parts.py`; `test_fe_optimizer_reader_default.py` |
| D-F05 env / reload in test | Low | FIXED | `test_cmi_null_subsample.py` |
| D-F06 wrap warning text | Low | FIXED | `_mrmr_validate_transform._recipe_replay_frame` |
| final subsumption order (found by the survival test, not by an agent) | Medium | FIXED | `_final_subsumption_stage.py`: screened compounds are judged last; `test_family_column_is_created_and_survives_selection[pairwise_ratio]` failed before, passes |

Result: 22 findings, 20 FIXED (two of them made visible only: A-F04, B-F04), 2 WON'T FIX (A-F05, B-F05) with the reasons in the reports. A survival test that failed in a local run (`pairwise_ratio`) led to one more fix, the final-subsumption order, recorded above.
