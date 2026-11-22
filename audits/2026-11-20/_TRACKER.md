# MRMR audit wave 2026-11-20 -- master tracker

Four read-only agents (A core correctness, B FE families and recipes, C GPU / resources / concurrency, D API / tests / performance), 22 findings, reports in this folder. The main session implements every fix itself;
each fix carries a sensor that fails on the pre-fix code. Coverage caveats: agent A did not trace the honest-null device kernels or the CMI / FE gates line by line; agents A and C found fewer than the 8-14 requested.

Statuses: FIXED (code changed, sensor named), WONT-FIX (reason given), TODO (open).

| Status | Finding | Severity | Where / sensor |
| --- | --- | --- | --- |
| **FIXED** | A-F01 id-keyed content-hash memo | High | `_mrmr_fingerprints.py` weakref slot; `test_mrmr_xhash_recycled_id.py` |
| **FIXED** | B-F01 float32 subtractions on offset columns | High | `_ratio_delta_fe.py`, `_grouped_agg_fe.py`, `_grouped_coerce_shared.py`; `test_float64_deviation_for_offset_columns.py`; benchmark selection identical |
| **FIXED** | D-F01 identity fast path vs permuted support_ | High | `_mrmr_validate_transform.py`; `test_mrmr_transform_identity_fastpath_identity.py` |
| **FIXED** | D-F02 fast-path test stub sorted only | Medium | same test file (permuted + boolean mask cases) |
| **FIXED** | A-F02 substitution without the add-one ceiling | Medium | `_confirm_predictor.py`; `test_prefer_engineered_substitution_ceiling.py` |
| **FIXED** | A-F03 SU marginal against a nat floor | Medium | `_screen_predictors_gate.py`; `test_screen_predictors_gate_boundaries.py` |
| **FIXED** (warning) | A-F04 missing marginal bypasses the maxT floor silently | Low | same file / test |
| **WON'T FIX** | A-F05 nominal vs occupied bins in the abs/rel floors | Low | reason in the report |
| **FIXED** | B-F02 lagged diff replay without history | Medium | `_ratio_delta_fe.py` `lag_history`, recipe `history`; `test_float64_deviation_for_offset_columns.py` |
| **FIXED** | B-F03 modular product wraps int64 | Medium | `_pairwise_modular_fe.py`; same test file |
| **FIXED** (warning) | B-F04 deadline truncation silent | Medium | `_fe_deadline.warn_deadline_truncation`; `test_fe_deadline_truncation_warns.py` |
| **WON'T FIX** | B-F05 lagged diff NaN -> 0 | Low | finite-output contract, reason in the report |
| **FIXED** | B-F06 integer cache keyed on 64 rows | Low | `_integer_columns._column_checksum`; same test file |
| **FIXED** | C-F01 edge pool drops device / allocator | Medium | `_edge_pool._capture_device_scope`; `test_gpu_resource_audit_c.py` |
| **FIXED** | C-F02 shared-memory limit | Low | `_null_max_perm_gpu.py` |
| **FIXED** | C-F03 label cache | Low | two keys + fit-end releaser |
| **FIXED** | C-F04 unreadable device = fits | Low | `_fe_stage_concurrency._stages_fit_device` |
| **FIXED** | C-F05 cleanup error hides stage error | Low | `_fe_stage_concurrency._run` |
| **FIXED** | D-F03 vacuous assertion | Low | `test_mrmr_transform_identity_fastpath_identity.py` |
| **FIXED** | D-F04 fe_optimizer fallback | Low | `_step_core_parts.py`; `test_fe_optimizer_reader_default.py` |
| **FIXED** | D-F05 env / reload in test | Low | `test_cmi_null_subsample.py` |
| **FIXED** | D-F06 wrap warning text | Low | `_mrmr_validate_transform._recipe_replay_frame` |
| **FIXED** | final subsumption order (found by the survival test, not by an agent) | Medium | `_final_subsumption_stage.py`: screened compounds are judged last; `test_family_column_is_created_and_survives_selection[pairwise_ratio]` failed before, passes |

Result: 22 findings, 20 FIXED (two of them made visible only: A-F04, B-F04), 2 WON'T FIX (A-F05, B-F05) with the reasons in the reports. A survival test that failed in a local run (`pairwise_ratio`) led to one more fix, the final-subsumption order, recorded above.
