# Tracker: the 25 REAL launch-bound loops (triage A and B)

Procedure per row (owner instruction, no stopping for questions): capture the real call shape from a pillar fit, profile it with nvprof at that shape, find the hotspot kernels, change the hotspot, prove equality (bit-identical or selection-identical against the previous path), paired A/B on the pillar that calls it, regression test, commit locally. Status values: TODO, PROFILED, DONE (with the measured gain), REJECTED (with the measurement), NOT-EXERCISED (not called by the four pillars; needs a shape from another workload before a rewrite).

| Status | # | Site | Measured / note |
|---|------|--------|-----------------|
| **TODO** | A2 | `_batch_mi_noise_gate_screen_v1.py::batch_mi_with_noise_gate_cupy_v1` loop at 139 | per permutation H2D, bincount, D2H |
| **DONE** | A4 | `_binned_numeric_agg_resident.py::_binagg_null_calibrated_floor` 435 | target-free candidate preparation (radix edges, casts, transpose) hoisted out of the permutation loop through `candidate_cache`; 1.5x to 2.0x isolated, floors bit-equal |
| **TODO** | A10 | `_cupy_polynom_optimizer.py::run_cupy_kernel_search` 240 | generations are dependent; only uploads can merge |
| A16 | `_fe_additive_fusion_gpu_resident.py::_gpu_relevant_halves` 195 | TODO | |
| A17 | `_fe_additive_fusion_gpu_resident.py::_subsumed_raw_operand_check_gpu` 429 | TODO | |
| **TODO** | A18 | `_fe_additive_fusion_gpu_resident.py::propose_additive_fusions_gpu` 390 | 0.59 s cum in f2 |
| A24 | `_fe_cmi_redundancy_gate.py::_apply_cmi_redundan_partition_collapses_same_key` 740 | TODO | |
| **TODO** | A25 | `_fe_cmi_redundancy_gate.py::_gate_round` 479 | batched path exists, default off |
| A40 | `_hinge_detect_gpu_resident_batch.py::detect_hinge_breakpoints_gpu_batch` 145 | TODO | |
| B1 | `_mi_greedy_cmi_fe_steps.py::_renumber_joint_gpu` 277 | TODO | |
| B4 | `_mrmr_fe_step/_step_score.py::_score_candidates_one_by_one` 59 | TODO | |
| B5 | `_mrmr_fe_step/_step_score_parts2.py::_materialise_and_fina_step2_esc_failed` 461 | TODO | |
| B7 | `_mrmr_fe_step/_step_score_parts2.py::_register_pair_recipes` 154 | TODO | |
| **DONE** | B11 | `_orthogonal_univariate_fe/_extra_basis_resident.py::_build_extra_basis_matrix_gpu` 142 | real pools are 800+ Fourier names per call; one block launch, bit-identical in both dtypes (a first attempt with `--fmad=false` changed double `sin` and was dropped for a rounded multiply on the chirp product); 63 ms to 1.4 ms per 256 names; also fixed the float32 right-boundary spline bug found by the failing parity tests |
| B13 | `_orthogonal_univariate_fe/_gpu_resident_cross_basis.py::build_leg_product_matrix_gpu` 172 | TODO | |
| **DONE** | B18 | `_pairwise_modular_resident.py::residue_grid_mis_resident` 140 | one broadcast launch for all residues, groups as row slices of the column-major block; 5.9 to 3.7 ms per call, noise pillar 46.4 to 45.5 s, bit-identical |
| B19 | `_unified_fe_gate.py::local_mi_gate` 339 | TODO | |
| B20 | `_unified_fe_gate.py::unified_second_pass_gate` 475 | TODO | |
| B25 | `_usability_greedy_clf_gpu_resident.py::_usability_greedy_c_range_min` 462 | TODO | |
| B26 | `_usability_greedy_clf_gpu_resident.py::_shortlist` 446 | TODO | |
| **TODO** | B28 | `_usability_greedy_gpu_resident.py::_usability_greedy_g_range_min` 410 | 1.04 s cum (`_cv_candidates`) |
| B29 | `_usability_greedy_gpu_resident.py::_shortlist` 269 | TODO | |
| **PARTIAL** | B31 | `_usability_pool_resident.py::score_pair_combos_table_resident` 244 | profiled: bound by radix select 50% and Miller-Madow MI 28%; unary stack kernel + operand cache 1.6-3.3% faster; shared-memory sort prototype rejected (slower at n=3000) |
| **DEFERRED** | B35 | `_wavelet_basis_fe_batched.py::_per_leg_matrices` 131 | reached only when the fused leg kernels do not apply |
| **TODO** | B38 | `hermite_fe/_als_fused_gpu.py::als_sweep_fused` 209 | 0.70 s cum, 542 calls, dependent sweeps |

Related fixes made on the way (not from this list, found by profiling the pillar fits): `family_null_bits_v2` (3.6x), noise gate on bit sets (7.2x), content memos for an FE-step retry, adjacent-column transpose, narrow label gathers.
