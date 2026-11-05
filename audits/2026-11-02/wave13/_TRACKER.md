# Wave 13 audit 2026-11-02 -- tracker

Three read-only reports by Opus 5.5 low-effort agents: CPU/GPU twins of the MI kernels, CPU/GPU twins of TreeSHAP, KSG, DTW and the transformer kernels, and a pattern sweep of bias and small-sample corrections. The agents could not write files, so the reports are their returned text, condensed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | FORM13-1 | wrong_formula | High | JS divergence bias floor `(K-1)/(2N)` assumed equal sample sizes (`metrics/_drift.py`) |
| **RESOLVED** | FORM13-2 | wrong_formula | Medium | threshold stability CV used the ddof=0 std over few folds (`calibration/threshold_optimizer.py`) |
| **RESOLVED** | FORM13-3 | wrong_formula | Medium | asymmetric rescale factor CV used ddof=0 (`calibration/asymmetric_rescale.py`) |
| **RESOLVED** | FORM13-4 | wrong_formula | Low | split-drift standard error used ddof=0 variances (`reporting/charts/_error_analysis_splits.py`) |
| **RESOLVED** | FORM13-5 | wrong_formula | Low | conditional-gate permutation null cut-off used ddof=0 (`_conditional_gate_fe.py`) |
| **REJECTED** | FORM13-6 | wrong_formula | Low | RFECV fold-score penalty uses ddof=0 in `final_score`; `cv_std_perf` is deliberately stored as the ddof=0 std and the one-SE band converts it (`std / sqrt(k - 1)`, wave 12), so changing it needs a joint change of both and the penalty weight is a heuristic knob |
| **RESOLVED** | PAR13-1 | cpu_gpu_parity | High | external-validation candidates holding NaN were binned over all rows on the GPU, over finite rows on the CPU (`_gpu_resident_extval.py`) |
| **RESOLVED** | PAR13-2 | cpu_gpu_parity | High | a run of equal values is one group in every rank binner: it takes the bin of its first sorted position, so the CPU rank binner (`_quantile_bin_njit`, `_quantile_bin_sorted_njit`, the numpy and class-balanced twins) and the GPU rank binners (`_gpu_resident_rank_bin.py`) keep equal values together as the GPU edge binner does; the binners still place boundaries by their own rule. Tests: `tests/feature_selection/filters/test_rank_bin_tie_rule.py` (CPU fast path vs sorted reference, no tie group split, GPU parity). |
| **FUTURE** | PAR13-3 | cpu_gpu_parity | Medium | `_mi_classif_batch_numba` bins by rank or by edge depending on whether the streamed GPU path fits in VRAM (`_orth_mi_backends.py`, `_streamed_gpu_mi.py`); needs the streamed path restricted to explicit edge binning or a rank-binned streamed kernel, with a selection benchmark |
| **RESOLVED** | PAR13-4 | cpu_gpu_parity | Low | GPU noise-gate SU reductions did not floor a last-ulp negative MI (`_batch_mi_noise_gate_kernels.py`) |
| **RESOLVED** | PAR13B-1 | cpu_gpu_parity | High | GPU DTW band one cell wider than dtaidistance's (`signal/dtw.py`) |
| **RESOLVED** | PAR13B-2 | cpu_gpu_parity | High | GPU DTW band not widened by `|n - m|`, endpoint unreachable; auto dispatch now keeps unequal lengths on the CPU |
| **RESOLVED** | PAR13B-3 | cpu_gpu_parity | Low | `mixed_ksg_mi_gpu` fallback dropped `intens`, `max_input_n` and `seed` |

## Counts

14 rows. Dispositions: 10 RESOLVED, 2 FUTURE, 1 REJECTED; 5 of the FORM13 rows and the DTW/KSG/noise-gate/extval rows are the RESOLVED ones.

## Implemented

| ID | Commit subject | Regression test |
|---|---|---|
| FORM13-1 | Use the expected same-distribution value as the JS divergence bias floor | `tests/metrics/test_js_divergence_bias_with_unequal_sizes.py` |
| FORM13-2..5 | Use the sample std for fold thresholds, rescale factors, permutation nulls and split drift | `tests/calibration/test_fold_dispersion_uses_sample_std.py` (FORM13-2, FORM13-3); FORM13-4 and FORM13-5 are covered by the existing split-drift and conditional-gate suites only |
| PAR13B-3 | Forward all arguments from the GPU KSG fallback to the CPU estimator | `tests/feature_selection/test_ksg_gpu_fallback_forwards_arguments.py` |
| PAR13-4 | Floor a negative MI at zero in the GPU noise gate's symmetric-uncertainty reductions | `tests/feature_selection/filters/test_noise_gate_su_clamps_negative_mi.py` |
| PAR13-1 | Keep the host binner for external-validation candidates holding NaN | `tests/feature_selection/gpu/test_extval_gpu_declines_non_finite_candidates.py` |
| PAR13B-1, PAR13B-2 | Match dtaidistance's Sakoe-Chiba band in the GPU DTW and keep unequal lengths on the CPU | `tests/signal/test_dtw_gpu_band_matches_cpu.py` |

FORM13-4 and FORM13-5 have no dedicated regression test; their kernels sit inside report and gate code and the change is a one-token ddof switch. The related existing tests pass.

## Not covered by this wave

- Not read at all: the resident CUDA gate path at `batch_mi_noise_gate_gpu.py:448-600`, the grand-fusion kernels and `_gpu_resident_pair_mi_parts.py`.
- The pattern sweep read the code around the likeliest hits, not all of them.
