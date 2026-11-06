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
| PARG4-2 | High | RESOLVED | numba TreeSHAP takes leaf values, covers and phi as float64 (the contract its docstring already stated), matching the cupy twin. On the old code under relaxed crit dtype, phi came back float32 (test failed on the dtype assert). Test `tests/feature_selection/shap_proxied/test_treeshap_numba_phi_is_float64.py`; existing numba TreeSHAP suites (26 tests) pass. The `test_treeshap_gpu_matches_numba` 1.1e-4 gap should shrink; not re-measured here (no CUDA run in this step). |
| PARG1-1 | High | RESOLVED | Host binned-aggregate path binned group and aggregate values at crit dtype (float32 by default) while the device path binned at float64. Float32 collapsed group values near 1e8 onto one edge, so `binned_numeric_agg_with_recipes` emitted zero `binagg_*` columns under relaxed dtype and four under float64 (verified before and after). Fixed in `_binned_numeric_agg_fe.py` (group/agg values) and `_binned_agg_cheap_mi.py` (quantile edges, the group MI pre-selection). Test `tests/feature_selection/fe/test_binned_numeric_agg_host_bins_in_float64.py`; 45 existing binned-aggregate tests pass (GPU variants not run on this host). |
| PARG1-2 | High | RESOLVED | `bb1c84dd6` the device ratio uses the host guard `u/max(abs(v),1e-6)`; test `test_candidate_ratio_matches_host_guard.py` |
| PARG3-1 | High | RESOLVED | `ec13aee95` rank binning uses float64 only for offset-dominated columns (offset above 1000x spread); float32 stays the default; tests `test_hermite_rank_bin_keeps_distinct_offset_values.py` |
| PARG3-2 | High | RESOLVED | `2029d5326` both fused preprocess paths centre in float64 before narrowing; test `test_basis_preprocess_centres_before_narrowing.py` |
| PARG2-2 | High (if modulus > 127) | REJECTED (not reproduced on HEAD) | Probed 2026-11-06 on this host (kernel v8, 2 classes): modulus 128 matches the CPU residue score to 1e-15; moduli 129 and above return None before the table is used. No out-of-range result found. Not changed: a guard that cannot be shown to fail on HEAD would not be a valid regression fix | Default modular-rank kernels store residues as int8; shared-memory gate allows moduli above 127 |
| PARG4-3 | Medium | RESOLVED | The banded DTW out-of-band cells held a finite 1e18 sentinel, which undercut true in-band costs above 1e18. A discriminating input was found: `x=[2e9,0,0]`, `y=[0,0,0]`, window 1, CPU 2e9 and numba.cuda 1e9 on HEAD. The sentinel is now infinity in every twin (CPU diagonal, numba.cuda band, cupy RawKernels, endpoint lookups). Test `tests/signal/test_dtw_cuda_band_cost_above_sentinel_matches_cpu.py` fails on the old code (1e9 vs 2e9) and passes now; the three existing DTW GPU suites pass. |
| PARG4-4 | Medium | RESOLVED | CPU `_insert_top` rejects a NaN loss (`not loss < worst`), which before took a top-N slot; test `test_brute_force_nan_loss_keeps_slots.py` fails on the old code (returned 2 of 3 candidates). GPU subset scan now uses float64 buffers and a double kernel, with boundary ties taken by lowest combination index and the top-k sorted on host by (loss, index); test `tests/feature_selection/gpu/test_shap_brute_force_gpu_tie_breaks_by_index.py` (skips without CUDA; passes on this host). GPU float64 doubles the buffer memory and may cost wall time on float32-default hardware. |
| PARG4-5 | Medium | RESOLVED | The row-attention RawKernel accumulated logits, softmax sum, weighted mean, variance and projected means in float32, while the njit twin uses float64 accumulators. The kernel now accumulates everything in double (shared memory included). Against a float64 reference on a target near 1000, the GPU error fell from 8e-4 (about 13 float32 ulps) to 1.2e-4 (2 ulps, the njit twin's own output precision). Test `tests/feature_engineering/transformer/test_row_attention_stage4_gpu_matches_njit.py` (GPU only; passes on this host). The batch-size dispatch itself stays, as a speed choice, now precision-neutral. |
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
| PARG1-7..PARG1-11 | Low | OPEN | PARG1-10 DOC `3214fb00e` (docstring states the tie exception); PARG1-11 DOC (module header states the reduction-order difference). PARG1-7 REJECTED (not reproduced on HEAD 2026-11-06: the plain and symmetric lerp forms agreed on all 500,000 random inputs, 300,000 of them float32 with adjacent-ulp order statistics, so no discriminating input exists; no code changed), PARG1-8 RESOLVED: confirmed on HEAD 2026-11-06. Device CMI returned 0.2245 against the host 7.7e-5 on codes containing -1 (the missing code from `_native_kbins_quantile` and `pd.factorize(use_na_sentinel=True)`); the shifted non-negative control matched. `_cmi_from_binned_cupy` now relabels host codes to start at 0 (kx and kz move by the same shift). Test `tests/feature_selection/gpu/test_cmi_from_binned_cupy_negative_codes.py` fails on the old code and passes now. The GPU CMI suite passed 52 with 1 skipped and 5 errors in `test_cmi_residency_traffic.py`: MemoryError on this host (5.65 GiB, discretization path, not touched); the retry gave 3 passed and the same 5 errors. Environment limit, not a defect of this change. PARG1-9 DOC: a single float32 operation matches the double-then-round result; the three-operand row sum and the division can differ by one ulp from the host's float32 arithmetic (selection-equivalence class, see the g1 report). Not changed |
| PARG4-6 | Low | REJECTED (not reproduced on HEAD) | njit `batch_weighted_std` (`_aggregation.py:100-123`, fastmath) returns 0.0 for a NaN value, the same as the cupy kernel the report uses as reference; the fastmath concern stays a note, no defect shown |
| PARG4-7 | Low | DOC | float32 GPU stage-4 by design (outputs float32 on both paths); documented in the `compute_row_attention` docstring |
| PARG4-11 | Low | RESOLVED | `ef4554515` `dtw_cupy_full` and `dtw_cupy_banded` delegate unequal lengths to `dtw_cpu`; reproduced on HEAD (GPU 1e9 vs CPU 24.14 for lengths 1000 and 1300); test `test_dtw_cupy_unequal_lengths_match_cpu.py`; 2 existing DTW GPU tests pass |
| PARG4-8 | Low | DOC | RFF cuBLAS-broken fallback computes in float64 while the healthy GPU and njit paths round to float32; the fallback is the more accurate result. Note placed at the `cublas_matmul_ok()` branch. |
| PARG4-9 | Low | DOC | Per-member MAE/std: numpy float32 path returns float32 and the njit twin float64. The equivalence sweep only uses float64 inputs; the dtype difference is documented, not changed. |
| PARG4-10 | Low | RESOLVED | Python frac-diff prefix fallback divided by `weights[0]` per step; now uses the same reciprocal-multiply as the njit kernel. Test `tests/training/composite/transforms/test_frac_diff_prefix_fallback_matches_njit.py` fails on the old code (205/400 elements differ), passes after; 17 frac-diff tests pass. |
| PARG2-5, PARG2-8..11 | DOC / Info | DOC | Honest-null threshold margin, equal-MI counting, prewarp rounding, polynom ties, pair-residual float32 near-ties (see g2 report) |
| PARG3-8, PARG3-9 | DOC | DOC | Offset-product summation order; col_minmax NaN skip (inputs documented finite) |
| PARG4-12 | Info | REJECTED | GPU RMSE run-to-run reproducibility already documented |

Decision (2026-11-06, owner): speed matters more than extra precision, so `crit_dtype_relaxed()` stays on (float32 by default). Fixes are targeted: the places where centring or ranking loses information centre or rank in float64 first and only then narrow (PARG3-1, PARG3-2, PARG3-4).

The common cause of PARG1-1, PARG3-1, PARG3-2, PARG3-4, PARG2-3 and PARG2-4 is `crit_dtype_relaxed()`, which is on by default and makes `crit_float_dtype()` float32. A single decision on that default, with a float64 path for centring and ranking, would close most of the open High items at once. That choice is the owner's: it trades speed against accuracy.
