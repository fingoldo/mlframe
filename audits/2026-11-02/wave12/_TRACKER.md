# Wave 12 audit 2026-11-02 -- tracker

Six read-only reports by Opus 5.5 low-effort agents: statistical formulas, remaining njit kernels, and CPU/GPU twin parity in three families (batched AUC, discretization and rank binning, MI and CMI). The agents could not write files, so the reports are their returned text, condensed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | PAR12-1 | cpu_gpu_parity | High | GPU `compute_batch_aucs` pooled positives over an (N, M) label matrix (`metrics/_gpu_metrics.py`) |
| **RESOLVED** | PAR12M-1 | wrong_formula | High | batched GPU marginal MI subtracted the Miller-Madow term (`_fe_batched_mi.py`) |
| **RESOLVED** | PAR12M-2 | wrong_formula | High | usability marginal MI (njit, cupy, fused CUDA) subtracted the term (`_numba_utils_usability.py`, `_usability_njit_pool.py`, `_fe_batched_mi.py`) |
| **RESOLVED** | PAR12D-1 | cpu_gpu_parity | Medium | CPU quantile edges used numba's percentile arithmetic, off by an ulp on ties (`discretization/_discretization_edges.py`) |
| **RESOLVED** | PAR12D-2 | cpu_gpu_parity | Medium | CPU rank binner put NaN first, numpy and cupy put it last (`_numba_utils.py`) |
| **RESOLVED** | PAR12D-3 | cpu_gpu_parity | Low | all-NaN column got the top code on the GPU, 0 on the CPU |
| **RESOLVED** | PAR12D-4 | cpu_gpu_parity | Low | a short column holding +inf collapsed to one bin on the GPU |
| **RESOLVED** | STAT12-1 | wrong_formula | Medium | RFECV one-SE band used the ddof=0 std over sqrt(k) |
| **RESOLVED** | STAT12-2 | wrong_formula | Medium | RFECV n_features bootstrap CI drew means with the fold std, not the standard error |
| **RESOLVED** | STAT12-3 | wrong_formula | Low | BCa z0 counted only strictly lower resamples |
| **RESOLVED** | STAT12-4 | wrong_formula | Low | group-MI leak verdict used an interpolated null quantile, not an add-one p-value |
| **DOC** | STAT12-5 | doc | Info | two docstrings misstate the statistics (`calibration/quality.py` chi-square dof note, `calibration/policy.py` ECE grid note); no numeric effect, left for a docs pass |
| **RESOLVED** | NJ12-1 | njit_error_model | High | serial usability kernels raised ZeroDivisionError on `1/0` (`_numba_utils_usability.py`) |
| **RESOLVED** | NJ12-2 | wrong_formula | High | Kendall tau-b left joint ties out of both tie counts (`metrics/regression/_regression_corr.py`) |
| **RESOLVED** | NJ12-3 | hang | High | concordance index never returned on a NaN in y_true |
| **RESOLVED** | NJ12-4 | wrong_formula | Medium | Fayyad-Irani MDL stop rule used 2, size-weighted entropies and the nominal class count (`transformer/mdl_binning_pairwise.py`) |
| **RESOLVED** | NJ12-5 | nan_handling | Medium | sequential log-loss kernel let a NaN probability through the clip |
| **RESOLVED** | NJ12-6 | wrong_formula | Low | explained variance on a constant target returned NaN |
| **REJECTED** | NJ12-7 | wrong_formula | Low | Tweedie deviance for power > 2 scores y = 0 as zero; the code comment records that this follows an sklearn special case, and sklearn's exact behaviour for p > 2 was not re-verified here |
| **RESOLVED** | NJ12-8 | njit_consistency | Low | `compute_extra_aggregates` divided by a zero softmax temperature, the fused kernel guards it |

## Counts

20 rows. Dispositions: 17 RESOLVED, 1 DOC, 1 REJECTED, 0 FUTURE (the remaining row, PAR12M-1 with PAR12M-2, is counted among the RESOLVED).

## Implemented

| ID | Commit subject | Regression test |
|---|---|---|
| PAR12-1 | Score a per-column label matrix column by column on the GPU AUC branch | `tests/metrics/test_batch_aucs_gpu_branch_per_column_labels.py` |
| PAR12M-1, PAR12M-2 | Add the Miller-Madow term in the usability marginal MI and the batched GPU marginal MI | `tests/feature_selection/filters/test_usability_marginal_mi_bias_sign.py` |
| STAT12-3, STAT12-4 | Count tied resamples as half in BCa's z0 and use an add-one p-value for the group-MI leak test | `tests/evaluation/test_bca_bias_correction_counts_ties.py`, `tests/feature_selection/test_group_mi_leak_decision_uses_add_one_p_value.py` |
| NJ12-2, NJ12-3 | Count joint ties in Kendall tau-b and stop the C-index from hanging on NaN | `tests/metrics/test_kendall_tau_joint_ties_and_concordance_nan.py` |
| NJ12-5 | Treat a NaN probability as out of range in both log-loss kernels | `tests/metrics/test_log_loss_nan_probability.py` |
| NJ12-1 | Compile the usability pair-combo kernels with the numpy error model | `tests/feature_selection/filters/test_usability_kernels_zero_division.py` |
| NJ12-6 | Score explained variance on a constant target like sklearn | `tests/metrics/regression/test_explained_variance_constant_target.py` |
| NJ12-4 | Use the Fayyad-Irani MDL cost with the classes present in each part | `tests/feature_engineering/transformer/test_mdl_binning_fayyad_irani_stop_rule.py` |
| NJ12-8 | Guard a zero softmax temperature in the extra aggregates like the fused kernel | `tests/feature_engineering/transformer/test_extra_aggregates_zero_temperature.py` |
| STAT12-1, STAT12-2 | Use the sample standard error for the RFECV tolerance band and size interval | `tests/feature_selection/wrappers/test_rfecv_one_se_band_uses_sample_standard_error.py`; `test_rfecv_one_se_band_width.py` re-framed |
| PAR12D-2 | Order NaN last in the CPU rank binner like numpy and cupy | `tests/feature_selection/filters/test_rank_bin_nan_last.py` |
| PAR12D-1 | Compute CPU quantile bin edges with numpy's lerp | `tests/feature_selection/filters/test_binning_edges_match_numpy_percentile.py` |
| PAR12D-3, PAR12D-4 | Match the CPU on an inf-bearing and an all-NaN column in GPU quantile binning | `tests/feature_selection/gpu/test_discretize_cuda_cpu_parity_inf_and_all_nan.py` |

Selection effects: PAR12M-1/2, STAT12-1, NJ12-4 and PAR12D-1/2 change scores or bin codes that feed feature selection, so selections can move. The directly related test files pass (synergy, copula, greedy CMI, usability, discretization, rank binning, RFECV band, MDL binning); no multi-seed selection benchmark was run.

## Not covered by this wave

- CPU/GPU pairs not read line by line: `_hermite_fe_mi.py`, `batch_pair_mi_gpu.py` with `_batch_pair_mi_cuda_kernels.py`, `_gpu_resident_pair_mi*.py`, `_triple_mi_gpu.py`, `batch_mi_noise_gate_gpu.py`, `_streamed_gpu_mi.py` (grep found no Miller-Madow term in them), `_shap_proxy_treeshap_gpu.py` against the CPU TreeSHAP, `_ksg.py`, `transformer/_kernels_cupy.py` against `_kernels_njit.py`, `signal/dtw.py`.
- Njit files still unread: most of `feature_engineering/transformer/*` beyond the kernels the agent listed, and `_regression_metrics.py` outside lines 100-300 and 560-790.
