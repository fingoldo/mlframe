# Wave 3 tracker (2026-11-05)

Two read-only Opus 5.5 (low effort) audits closing the coverage gaps of wave 2: GPU kernel parity (`kernel_parity_rest.md`) and estimator contracts (`api_contract_rest.md`). Each finding was checked against HEAD before a fix.

## Kernel parity (PAR3)

| ID | Sev | Disposition | Commit / note |
|---|---|---|---|
| PAR3-1 | Medium | RESOLVED | `92066453b` float32 default keeps the double kernel for offset-heavy or non-finite inputs; test `test_usability_corr_float32_guard.py` |
| PAR3-2 | Low | RESOLVED | `fff343b13` one dtype for the |corr| table on every backend; test `test_usability_corr_table_dtype_is_backend_independent.py` |
| PAR3-3 | Low | DOC | CPU reassoc/contract fastmath vs GPU without it; matters only for columns at the near-constant cutoff |
| PAR3-4 | Info | REJECTED | RMSE and AUC GPU vs CPU agree (NaN, float64 accumulation, tie ranks) |

Coverage: the audit read 2 of about 60 GPU-kernel files. Not yet read: the MI and CMI kernels, `_gpu_metrics.py` beyond the two above, DTW, the transformer kernels, the honest-null and resident-select kernels. The list is in `kernel_parity_rest.md`. Next round should cover them.

## Estimator contracts (API3)

| ID | Sev | Disposition | Commit / note |
|---|---|---|---|
| API3-1 | Medium | RESOLVED | `251dc4a4c` `ShapProxiedFS` keeps None through get_params and clone; the per-host default resolves at fit |
| API3-2..11, API3-13 | Medium | RESOLVED | `3b1bfe6de` predict/transform before fit raise NotFittedError; test `tests/test_meta/test_estimators_refuse_predict_before_fit.py` |
| API3-12 | Low | DOC | `_Float32CastTransformer`, `_InfToNaNTransformer` accept *args/**kwargs and transform unfitted |
| API3-14 | Low | RESOLVED | `aafac049e` `EarlyStoppingWrapper` with `base_model=None` raises ValueError naming the parameter |
| API3-15 | High | RESOLVED | `73adbdbbd` composite multi-output predict refuses a different input width |
| API3-5 | Medium | RESOLVED | `a403b4b1a` Tabular1DCNN regressor and classifier record the width and refuse a different one at predict |
| API3-7, API3-8 | Low | RESOLVED | `a403b4b1a` `TrunkResidualMLPRegressor`, `MultiTaskAuxiliaryLossRegressor` record the width and refuse a different one at predict |

## Known failures in the shap-proxied test directory (run on 2026-11-06, 749 passed, 6 failed)

| Test | Status |
|---|---|
| `test_mrmr_then_shap_proxied_fs_reuses_artifacts_e2e`: prefilter does not switch to `precomputed_su` | Pre-existing: fails the same way on a clean worktree at HEAD (`8e4aea9c7`). Not caused by API3-1. Open. |
| `test_column_major_speedup_vs_row_major_reference`: ratio 1.23x vs 1.70x | Timing assertion, labelled load-sensitive in the test. Re-run on a quiet machine. |
| `test_biz_value_revalidation_cap_faster_recovery_preserved`: 1.05x vs 1.30x | Timing assertion. Re-run on a quiet machine. |
| `test_treeshap_gpu_matches_numba[True]`, `[False]`: max rel. diff 1.1e-4 | GPU numerical tolerance. Not checked against HEAD yet. Open. |
| `test_shap_proxied_fs_default_cap_uses_module_default` | Caused by API3-1: the test pinned the old contract. Reframed to the effective-width helper. RESOLVED in `251dc4a4c`. |

## Kernel parity, remaining groups (PARG1..PARG4)

Four read-only audits covered the remaining GPU-kernel files (reports `kernel_parity_g1..g4.md`). Files not read in full are listed in each report.

| ID | Sev | Disposition | Commit / note |
|---|---|---|---|
| PARG2-1 | High | RESOLVED | `1788f6c9b` one row budget for the pair-residual check on both backends; test `test_pair_residual_budget_is_backend_independent.py` |
| PARG4-1 | High | RESOLVED | `812a9a61c` GPU ROC and PR AUC map labels to {0,1} as the CPU path does; GPU test against sklearn |
| PARG4-2 | High | OPEN | Treeshap numba twin is float32 under `MLFRAME_CRIT_DTYPE_RELAXED=1`; cupy twin is float64. Explains the 1.1e-4 gap in `test_treeshap_gpu_matches_numba`. Needs one dtype contract. |
| PARG1-1 | High | OPEN | Device-born binned aggregates upload values as float64; host casts to crit dtype. Bin codes and cell moments can differ near edges and at 1e39. |
| PARG1-2 | High | RESOLVED | `bb1c84dd6` the device ratio uses the host guard `u/max(abs(v),1e-6)`; test `test_candidate_ratio_matches_host_guard.py` |
| PARG3-1 | High | RESOLVED | `ec13aee95` rank binning uses float64 only for offset-dominated columns (offset above 1000x spread); float32 stays the default; tests `test_hermite_rank_bin_keeps_distinct_offset_values.py` |
| PARG3-2 | High | RESOLVED | `2029d5326` both fused preprocess paths centre in float64 before narrowing; test `test_basis_preprocess_centres_before_narrowing.py` |
| PARG2-2 | High (if modulus > 127) | REJECTED (not reproduced on HEAD) | Probed 2026-11-06 on this host (kernel v8, 2 classes): modulus 128 matches the CPU residue score to 1e-15; moduli 129 and above return None before the table is used. No out-of-range result found. Not changed: a guard that cannot be shown to fail on HEAD would not be a valid regression fix | Default modular-rank kernels store residues as int8; shared-memory gate allows moduli above 127 |
| PARG4-3 | Medium | OPEN | DTW GPU kernels use finite 1e18 for unreachable cells; costs above 1e18 reach invalid cells. Checked on HEAD 2026-11-06: the report's example (x=[0,0,0], y=[2e9]*3) gives the same distance on GPU and CPU, so it does not discriminate. The CPU twin `dtw_cpu` (dtw.py:115,143,144) uses the same 1e18 sentinel, so the defect is shared, not a GPU-only mismatch. A fix must change both twins and needs a discriminating input before it is claimed. |
| PARG4-4 | Medium | OPEN | GPU subset scan breaks ties at top-N arbitrarily and returns float32 losses |
| PARG4-5 | Medium | OPEN | Row attention switches njit/GPU by batch size; different precision per batch |
| PARG1-3 | Medium | RESOLVED | `d0d07877f` device row max and min propagate NaN from any operand, as numpy; test `test_candidate_row_extrema_propagate_nan.py` |
| PARG1-4 | Medium | RESOLVED | radix entry points canonicalise NaN to the positive quiet NaN, so a sign-bit NaN sorts last; reproduced on HEAD (rank 0 was NaN, sorted 0.0); test `test_radix_select_orders_sign_bit_nan_last.py`; 85 existing radix/quantile GPU tests pass |
| PARG1-5 | Medium | RESOLVED | `d38ed8df8` block count gives a NaN row the edge count, as the per-column binner; test `test_qbin_block_nan_code_matches_single_column.py` |
| PARG1-6 | Medium | RESOLVED | `7d08849ca` four-entropy CMI fallback reduces in float64; test `test_cmi_entropy_fallback_is_double.py` |
| PARG2-3 | Medium | RESOLVED | `1ccb87f6a` float32 polynomial basis rebuilt in float64 only when the float32 build is not finite; reproduced on HEAD (inf at He_25(40)); test `test_cupy_polynom_basis_stays_finite_for_large_operands.py`; 3 existing cupy polynom tests pass | Cupy polynom optimiser builds basis in float32; high degree overflows where numba float64 scores it |
| PARG2-4 | Medium | RESOLVED | `d93d89716` `gpu_pairs_fe_mi` narrows candidates with the offset-aware dtype (`_binning_dtype`); on HEAD a column 1+i*1e-12 gave MI 0, now ln 2; test `test_gpu_pairs_mi_keeps_near_tied_float64_values_apart.py` | `gpu_pairs_fe_mi` casts candidates to float32 before binning |
| PARG3-3 | Medium | RESOLVED | `2d2271bfe` fused bin-code kernels send a NaN row past every edge, as the fallback and CPU; test `test_discretize_fused_nan_code_is_the_last_bin.py` |
| PARG3-4 | Medium | RESOLVED | `99fa73344` device HSIC centres rows in float64 before narrowing; test `test_hsic_device_is_offset_invariant.py` |
| PARG2-6 | Low | RESOLVED | `b4bc25fcb` fused gate clamps MI at zero as the host; reproduced at -6.5e-16 on HEAD; test `test_fused_su_gate_clamps_negative_mi.py` |
| PARG2-7 | Low | RESOLVED | `a6be24ee2` negative target codes go to the generic path; test `test_analytic_gate_refuses_negative_target_codes.py` |
| PARG3-5 | Low | RESOLVED | `3057db77a` on-device segment entropies computed and returned in float64, as documented; test `test_friend_entropy_is_double.py` |
| PARG3-6 | Low | RESOLVED | `330ac70ff` device sign of NaN is NaN, as the twins; test `test_unary_sign_keeps_nan.py` |
| PARG3-7 | Low | RESOLVED | `0ee82cc02` empty family returns zeros before any launch; test `test_family_null_max_empty_family.py` |
| PARG1-7..PARG1-11 | Low | OPEN | PARG1-10 DOC `3214fb00e` (docstring states the tie exception); PARG1-11 DOC (module header states the reduction-order difference). Open: PARG1-7 (plain vs symmetric lerp, needs a discriminating input), PARG1-8 (negative codes as indices in device CMI: NOT confirmed on HEAD. The device CMI (`_mi_greedy_cmi_fe_gpu.py`) takes codes from at least 12 callers in `filters/` (`_fe_batched_mi_cmi`, `_fe_additive_fusion*`, `_fe_cmi_*`, `_fe_raw_redundancy_*`, `_fe_retention_subsumption`, `_binned_numeric_agg_redundancy`, `_fe_auto_escalation`, ...). Next step: grep each producer for a `-1` sentinel that reaches `cmi_joint_entropies_gpu`; no sentinel found in the first producers checked), PARG1-9 (float32 rounding) |
| PARG4-6 | Low | REJECTED (not reproduced on HEAD) | njit `batch_weighted_std` (`_aggregation.py:100-123`, fastmath) returns 0.0 for a NaN value, the same as the cupy kernel the report uses as reference; the fastmath concern stays a note, no defect shown |
| PARG4-7 | Low | DOC | float32 GPU stage-4 by design (outputs float32 on both paths); documented in the `compute_row_attention` docstring |
| PARG4-11 | Low | RESOLVED | `(hash)` `dtw_cupy_full` and `dtw_cupy_banded` delegate unequal lengths to `dtw_cpu`; reproduced on HEAD (GPU 1e9 vs CPU 24.14 for lengths 1000 and 1300); test `test_dtw_cupy_unequal_lengths_match_cpu.py`; 2 existing DTW GPU tests pass |
| PARG4-8..PARG4-10 | Low | OPEN | fastmath NaN guards, float32 casts, dtype differences, frac-diff division, DTW cupy checks (see g4 report) |
| PARG2-5, PARG2-8..11 | DOC / Info | DOC | Honest-null threshold margin, equal-MI counting, prewarp rounding, polynom ties, pair-residual float32 near-ties (see g2 report) |
| PARG3-8, PARG3-9 | DOC | DOC | Offset-product summation order; col_minmax NaN skip (inputs documented finite) |
| PARG4-12 | Info | REJECTED | GPU RMSE run-to-run reproducibility already documented |

Decision (2026-11-06, owner): speed matters more than extra precision, so `crit_dtype_relaxed()` stays on (float32 by default). Fixes are targeted: the places where centring or ranking loses information centre or rank in float64 first and only then narrow (PARG3-1, PARG3-2, PARG3-4).

The common cause of PARG1-1, PARG3-1, PARG3-2, PARG3-4, PARG2-3 and PARG2-4 is `crit_dtype_relaxed()`, which is on by default and makes `crit_float_dtype()` float32. A single decision on that default, with a float64 path for centring and ranking, would close most of the open High items at once. That choice is the owner's: it trades speed against accuracy.
