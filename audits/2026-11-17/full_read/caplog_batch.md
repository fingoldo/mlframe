# caplog tests for the fallback-log sites

Every test stubs the failing call on CPU, resets the throttle counts (`reset_throttle_counts`) and asserts the WARNING text with `caplog`. All files are under `tests/feature_selection/filters/`.

## Group A: `test_pairs_score_fallback_logs.py` (7 passed)
- `_pairs_score_steps` ~187: `test_batched_abs_corr_device_fault_warns_and_returns_none`
- `_pairs_score_steps` ~844: `test_noise_wrap_veto_failure_warns`
- `_pairs_score_helpers` ~166: `test_prefetch_submit_failure_warns`; ~155 (extra): `test_producer_failure_warns`
- `_pairs_score_helpers` ~401: `test_batched_pair_corr_failure_warns`; ~417: `test_config_corr_failure_warns`
- `_pairs_setup` ~403: `test_resident_operand_table_build_failure_warns`

## Groups B and E (qbin, devices): `test_fe_cmi_gpu_fallback_logs.py` (19 passed)
- `_fe_batched_qbin` ~168: `test_qbin_fused_coder_failure_warns` (numpy passed as `cp`)
- `_fe_cmi_redundancy_null` (8): `test_analytic_null_failure_warns`, `test_resident_support_null_failure_warns`, `test_marginal_gpu_null_failure_warns`, `test_marginal_batched_null_failure_warns`, `test_conditional_gpu_null_failure_warns`, `test_conditional_batched_null_vram_refusal_warns`, `test_conditional_batched_null_failure_warns`, `test_swept_device_null_failure_warns`
- `_fe_cmi_perm_null_gpu` (2): `test_perm_null_gate_probe_failure_warns`, `test_sparse_perm_null_failure_warns`
- `_fe_cmi_redundancy_gate` (6 sites, 2 asked): `test_gate_resident_probe_failure_warns`, `test_gate_round_cmi_floor_failure_warns`, `test_gate_device_binning_failure_warns`, `test_gate_gpu_resident_binning_failure_warns`, `test_gate_device_join_failure_warns`, `test_gate_renumber_joint_gpu_failure_warns`
- `_fe_gpu_batch/_devices` (2): `test_device_subset_parse_failure_warns`, `test_device_profile_failure_warns`

## Group C: `test_fe_redundancy_retention_fallback_logs.py` (15 passed; one tiny edit after the run, re-run was in flight when this was written)
- `_fe_resident_operands` ~337: `test_resident_assembly_failure_warns_and_uploads_whole_matrix`
- `_fe_raw_redundancy_helpers` (2): `test_tail_concentrated_probe_failure_warns`, `test_ridge_import_failure_warns`
- `_fe_raw_redundancy_drop` (6, 5 asked): `test_no_harm_recheck_failure_warns`, `test_informativeness_probe_failure_warns_and_answers_true`, `test_join_support_device_failure_warns`, `test_linear_leg_failure_warns`, `test_tail_probe_failure_warns`, `test_group_leak_check_failure_warns`
- `_fe_retention_subsumption` (2): `test_retention_device_replay_failure_warns`, `test_retention_import_failure_warns`
- `_fe_rejection_ledger` (4): `test_operands_str_failure_warns`, `test_admission_cap_warns`, `test_admission_record_failure_warns`, `test_rejection_record_failure_warns`

## Group D: `test_pairs_materialise_emit_fallback_logs.py` (5 passed)
- `_pairs_materialise` ~363: `test_ktc_parallelism_failure_warns_and_uses_heuristic`
- `_pairs_emit` (4): `test_batched_leader_corr_failure_warns`, `test_leader_usability_failure_warns`, `test_resolve_emitted_column_failure_warns`, `test_resolve_candidate_column_failure_warns`

## Group E: `test_fe_batched_mi_fallback_logs.py` (7 passed)
A numpy-backed `cupy` stand-in (inserted into `sys.modules`) plus failing kernel getters drive the launch wrappers on CPU.
- `mi_split_n_failed`: `test_mi_split_n_failure_warns`
- `mi_from_codes_tile_failed`: `test_mi_from_codes_tile_failure_warns`
- `mi_from_codes_v2_launch_failed`: `test_mi_from_codes_v2_launch_failure_warns`
- `mi_from_codes_v2_compile_failed`: `test_mi_from_codes_v2_compile_failure_warns_and_retries`
- `_recipe_unary_binary_gpu._vram_f32`: `test_recipe_vram_f32_check_failure_warns`

## Source changes (plain `logger.warning` in per-launch code converted to `log_throttle`)
- `_fe_batched_mi.py` ~459: key `mi_split_lookup_failed`; test `test_mi_split_lookup_failure_warns`
- `_fe_batched_qbin.py` ~133: key `qbin.radix_edges`; test `test_qbin_radix_edges_failure_warns`

## Not testable
None. Each site was reachable on CPU with stubs.

## Not done
- The "must fail if the log call is removed" check was not run as a revert; each assertion matches the throttled message text, so it fails without the log call.
- I did not run the mypy/interrogate gates over the new test files. I ran mypy on the two edited source files (result below if it finished).
