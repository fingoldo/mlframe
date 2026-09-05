# mlframe wide audit 2026-09-05 -- master tracker

Eight read-only agents, one per DEFECT CLASS rather than per subsystem: a class shows up the same
way in unrelated subsystems, so an agent that has understood one finds the rest. Every finding below
is implemented (RESOLVED) or given an explicit non-silent disposition (FUTURE / DOC / REJECTED).
Status starts at TODO for all.

**Totals: 7 P0, 25 P1, 48 P2, 44 P3 -- 124 findings.**

## Per-cluster summary

| Cluster | P0 | P1 | P2 | P3 | Total | Report |
|---|----|----|----|----|-------|--------|
| GPU and backend-dispatch correctness | 3 | 2 | 2 | 5 | 12 | [xcut_gpu_dispatch.md](xcut_gpu_dispatch.md) |
| Latched global state and caches | 0 | 5 | 4 | 5 | 14 | [xcut_latched_state.md](xcut_latched_state.md) |
| Memory and resource contracts | 2 | 4 | 3 | 0 | 9 | [xcut_memory_contracts.md](xcut_memory_contracts.md) |
| Thresholds that do not travel between machines | 0 | 3 | 17 | 8 | 28 | [xcut_nontravelling_thresholds.md](xcut_nontravelling_thresholds.md) |
| Silently substituted numbers | 2 | 1 | 3 | 0 | 6 | [xcut_silent_numeric_substitution.md](xcut_silent_numeric_substitution.md) |
| Stale frozen references / duplicated logic drift | 0 | 3 | 11 | 16 | 30 | [xcut_stale_reference_drift.md](xcut_stale_reference_drift.md) |
| Symbol and API drift | 0 | 3 | 1 | 4 | 8 | [xcut_symbol_drift.md](xcut_symbol_drift.md) |
| Guards that pass vacuously | 0 | 4 | 7 | 6 | 17 | [xcut_vacuous_guards.md](xcut_vacuous_guards.md) |

## Every finding, worst first

| ID | Sev | Where | One line | Status |
|---|---|---|---|---|
| XGD-01 | P0 | `feature_selection/filters/evaluation.py:501` | gpu-relevance-null-drops-base_seed | RESOLVED (base_seed forwarded; guard tests/feature_selection/test_gpu_branch_seed_parity.py, verified failing pre-fix) |
| XGD-02 | P0 | `feature_selection/filters/permutation.py:731` | mi_direct-gpu-fastpath-drops-base_seed | RESOLVED (base_seed forwarded to the GPU fastpath) |
| XGD-03 | P0 | `feature_selection/filters/discretization/__init__.py:939` | discretize-bin-edges-depend-on-free-vram | PARTIAL: the silence is fixed (a quantile fallback now WARNs that edges became approximate; min/max says it stays bit-identical). The approximation itself is untouched -- it is documented and selection-equivalence-validated (jaccard 1.0 at 50k+, 0.88 at 5k). FUTURE: whether a transient VRAM probe should be allowed to pick between exact and approximate binning at all needs its own selection-equivalence measurement, not a judgement call |
| XMC-01 | P0 | ``src/mlframe/evaluation/_bootstrap_fused_binary_bundle.py:181`` | bootstrap-resample-index-matrix-int64-unguarded | RESOLVED (index matrix generated per chunk; 55 existing bit-identity tests pass, plus tests/evaluation/test_bootstrap_idx_chunked_generation.py pinning bounded chunks AND an unchanged RNG stream) |
| XMC-02 | P0 | ``src/mlframe/feature_selection/filters/_mrmr_fit_impl/_friend_graph_and_redundancy/_group1.py:176`` | ungated-whole-frame-to_pandas-in-mrmr-synergy-screen | RESOLVED (fe_polars_exceeds gate + warning, matching the sibling cascade; 17 synergy/leak tests pass) |
| XNUM-01 | P0 | ``src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py:43`` | raw-sum Pearson in the FE-pair correlation gate | RESOLVED (both twins two-pass centred; regression tests/feature_selection/test_abs_corr_offset_stability.py -- 11 tests against np.corrcoef across four offsets) |
| XNUM-02 | P0 | ``src/mlframe/training/composite/streaming.py:133`` | raw prefix-sum segment SSE in the streaming Chow change-point test | RESOLVED (prefix sums centred; regression tests/training/composite/test_streaming_chow_offset_stability.py -- 10 tests, 6 verified failing pre-fix) |
| LATCH-01 | P1 | `src/mlframe/feature_selection/filters/polynom_pair_fe.py:388` | fe-deadline-republished-in-loky-worker-never-cleared | TODO |
| LATCH-02 | P1 | `src/mlframe/metrics/_gpu_metrics.py:72` | gpu-metrics-availability-latched-on-broad-except | RESOLVED (ImportError still cached, any other exception warns and re-probes; reset_gpu_metrics_probe added; regression tests/metrics/test_gpu_probe_does_not_latch_on_transient_failure.py, verified failing pre-fix) |
| LATCH-03 | P1 | `src/mlframe/metrics/_core_auc_brier.py:126` | metrics-argsort-gpu-availability-latched-on-broad-except | RESOLVED (ImportError still cached, other exceptions warn and re-probe; reset_gpu_argsort_probe added) |
| LATCH-04 | P1 | `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_cluster_su.py:64` | cluster-su-gpu-availability-latched-on-broad-except | RESOLVED (same narrowing as the sibling _shap_proxy_prefilter; reset_cluster_su_gpu_probe added) |
| LATCH-05 | P1 | `src/mlframe/feature_selection/filters/mrmr/_mrmr_class_fit_helpers.py:91` | two-gpu-circuit-breakers-omitted-from-the-fit-entry-rearm | RESOLVED (KSG and order-1 maxT breakers re-armed; a meta-test now asserts the re-arm covers every reset_*_gpu_circuit_breaker the package defines) |
| NT-15 | P1 | `tests/competition/test_biz_val_naive_bayes_log_odds.py:112, tests/competition/test_biz_val_naive_bayes_log_odds.py:87` | naive-bayes-log-odds-honest-negative-0.0005-auc-margin | TODO |
| NT-16 | P1 | `tests/competition/test_biz_val_gmm_classifier.py:138` | gmm-honest-negative-0.005-auc-margin-at-ceiling | TODO |
| NT-17 | P1 | `tests/feature_selection/biz_val/test_biz_val_e2e_operator_model_lift.py:158` | conditional-gate-e2e-auc-0.999-and-delta-above-zero | TODO |
| SRD-01 | P1 | ``src/mlframe/feature_engineering/transformer/hard_row_attention.py:43`, `src/mlframe/feature_engineering/transformer/multi_temp_residual_band.py:38`, `src/mlframe/feature_engineering/transformer/signed_residual_band.py:42` (twin sites that DID get the fix: `bidir_residual_band.py:41`, `class_balanced_hard_row.py:49`, `multi_temp_cbhr.py:37`, `baseline_surprise.py:43`, `fisher_weighted_residual.py:39`)` | residual-band transformer cluster: three siblings never got the OOF leakage fix | RESOLVED (5 modules, not 3: prediction_band_attention and y_quintile_baseline_knn also leaked; hoisted to _baseline_oof) |
| SRD-02 | P1 | ``tests/metrics/test_ranking_batch_kernel_dispatch.py:37` (twin: `src/mlframe/metrics/_ranking_extras.py:276`, constant at `:35`)` | ERR reference pins `y_true.max()` after production froze `max_grade = 4.0` | RESOLVED (reference reads _DEFAULT_ERR_MAX_GRADE; 0..2 grade scale added, verified failing against the old data-derived ceiling) |
| SRD-03 | P1 | ``tests/models/test_cpx16_optimization_membership_identity.py:58-100` (twin: `src/mlframe/models/optimization.py:265` -> `src/mlframe/models/_optimization_search.py`)` | "pre-CPX16" reference loads the live class; the identity test cannot fail | RESOLVED (the historical-revision approach is unworkable: pre-CPX16 code depends on store_params_in_object's old default postfix and no longer constructs; replaced with a direct set-vs-ndarray equivalence over a real run) |
| VG-01 | P1 | `tests/reporting/test_metric_over_time_direction.py:51 (and :59)` | metric-over-time-direction-tests-assert-nothing | RESOLVED (fixture now fills buckets, assertions unconditional; also found rmse/mae were unsupported by the per-bucket dispatcher so the panel could never render) |
| VG-02 | P1 | `tests/training/test_dataset_cache_fingerprint.py:239 (and :256)` | cache-key-id-scan-reads-only-the-call-header-line | RESOLVED (AST walk of the key= call arguments replaces the physical-line scan; verified catching an id() argument on a continuation line) |
| VG-03 | P1 | `tests/test_meta/test_enum_exhaustiveness.py:83-96 (skip at :113)` | enum-exhaustiveness-police-has-a-hardcoded-module-allowlist | RESOLVED (membership by package prefix, foreign classes now assert; corpus 20 fields/77 values -> 21/79, picking up _model_configs_behavior; skip replaced by a failure) |
| VG-04 | P1 | `tests/training/composite/discovery/test_composite_discovery_parallel.py:239` | wilcoxon-serial-vs-parallel-equivalence-is-empty-vs-empty | RESOLVED (root cause was wider: `enabled` defaults to False and neither fixture set it, so `fit` was a no-op and the whole file compared empty against empty) |
| XGD-04 | P1 | `calibration/_ktc_dispatch.py:101 and inference/_ktc_dispatch.py:52` | ktc-tuner-times-gpu-resident-but-production-pays-h2d | RESOLVED |
| XGD-05 | P1 | `metrics/_gpu_metrics.py:432` | batch-rmse-gpu-returns-float64-cpu-returns-float32 | RESOLVED |
| XMC-03 | P1 | ``src/mlframe/feature_engineering/transformer/_key_bank.py:123`` | key-bank-fingerprint-full-tobytes-copy | RESOLVED |
| XMC-04 | P1 | ``src/mlframe/feature_selection/filters/_fe_accuracy_gate.py:56-57`` | fe-accuracy-gate-baseline-key-double-full-copy-per-candidate | RESOLVED |
| XMC-05 | P1 | ``src/mlframe/training/composite/discovery/_collinear_numba.py:83`` | collinear-keep-mask-hash-2gb-tobytes-copy | RESOLVED |
| XMC-07 | P1 | ``src/mlframe/training/cb/_cb_pool.py:542-543` and `src/mlframe/training/_predict_guards.py:102`` | cb-pool-caches-capped-on-entry-count-not-bytes | RESOLVED |
| XNUM-03 | P1 | ``src/mlframe/feature_engineering/spatial.py:515`` | `+ 1e-12` on a power-law denominator in kNN local density | RESOLVED |
| XSD-01 | P1 | `src/mlframe/estimators/custom.py:154` | linear-model-module-does-not-exist | RESOLVED (sklearn.linear_model; regression test tests/estimators/test_transformed_target_default_regressor.py, verified failing pre-fix) |
| XSD-02 | P1 | `src/mlframe/feature_engineering/_benchmarks/bench_group_sort.py:10` | stable-counting-segments-moved-to-grouped-segments | RESOLVED (re-pointed to _grouped_segments) |
| XSD-03 | P1 | `profiling/bench_binned_numeric_agg_fold_gate.py:17` | raw-moments-renamed-to-per-cell-raw-moments-njit | RESOLVED (dead fallback replaced with an honest ImportError) |
| LATCH-06 | P2 | `src/mlframe/training/core/_phase_config_setup.py:208` | suite-wide-overrides-applied-long-before-their-restore-snapshot-is-recorded | TODO |
| LATCH-07 | P2 | `src/mlframe/models/ensembling/member_metrics.py:23` | lru-cache-over-env-vars-and-the-on-disk-kernel-tuning-cache | TODO |
| LATCH-08 | P2 | `src/mlframe/metrics/_gpu_metrics.py:129` | numba-cuda-metrics-probe-latched-on-broad-except | TODO |
| LATCH-09 | P2 | `src/mlframe/feature_engineering/transformer/_utils.py:88` | cupy-probe-latched-process-wide-with-a-test-only-reset | TODO |
| NT-01 | P2 | `tests/feature_selection/test_prewarm.py:287, tests/feature_selection/test_prewarm.py:289, tests/feature_selection/test_prewarm.py:75` | prewarm-cache-budget-dies-under-NUMBA_DISABLE_JIT | TODO |
| NT-02 | P2 | `tests/metrics/test_prewarm_bool_dtype.py:46, tests/metrics/test_prewarm_bool_dtype.py:68` | brier-logloss-prewarm-50ms-proxy | TODO |
| NT-03 | P2 | `tests/feature_selection/gpu/test_batch_pair_mi_shared_fused.py:170` | gpu-shared-fused-kernel-2s-budget | TODO |
| NT-04 | P2 | `tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_linear_preselect/test_cmim_hotpath_perf.py:163` | cmim-hotpath-5s-budget | TODO |
| NT-05 | P2 | `tests/feature_selection/fe/test_conditional_dispersion_fe.py:435, tests/feature_selection/fe/basis/test_wavelet_basis_fe.py:439, tests/reporting/test_charts_confusion_margins.py:263, tests/reporting/test_charts_prediction_stability.py:279` | cprofile-wrapped-wall-budgets | TODO |
| NT-06 | P2 | `tests/feature_selection/test_polynom_loky_pool_prewarm.py:146` | loky-pool-prewarm-15s-vs-26s-baseline | TODO |
| NT-07 | P2 | `tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_contracts_robustness/test_high_dim_embedding.py:286, :310, :340, :433, :476, :496` | mrmr-high-dim-embedding-wall-budgets | TODO |
| NT-08 | P2 | `tests/feature_selection/shap_proxied/test_shap_proxy_preflight.py:337, tests/feature_selection/shap_proxied/test_shap_proxy_preflight.py:416` | shap-preflight-25s-30s-budgets | TODO |
| NT-10 | P2 | `tests/evaluation/test_bootstrap_fused_binary_bundle.py:162, tests/feature_engineering/test_biz_val_cross_sectional_neighbors.py:170, tests/feature_selection/biz_val/test_biz_val_filters_hermite_fe.py:216, tests/feature_selection/contracts/test_evaluation.py:562, tests/feature_selection/info_theory/test_bulk_shuffle_three_mis.py:259, tests/feature_selection/info_theory/test_gil_release_threading_speedup.py:194, tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_interaction_info_prefilter_speedup.py:141, :355, tests/feature_selection/shap_proxied/test_shap_proxy_cluster_su_fused_setup.py:178, tests/feature_selection/shap_proxied/test_shap_proxy_treeshap.py:222, tests/feature_selection/shap_proxied/test_shap_proxy_treeshap_interactions.py:202, tests/metrics/test_bootstrap_auc_presort.py:116, :221, tests/training/baselines/test_per_group_baseline_polars_native.py:130, tests/training/composite/cache/test_prebin_matrix_cache.py:157, tests/training/composite/test_biz_val_group_aggregate_macro.py:156, tests/training/feature_handling/test_biz_val_ordered_target_encoder_batch.py:51, tests/training/feature_selection/test_mi_y_prebin_speedup.py:130, tests/training/neural/test_ranker_getitems_batched.py:129, tests/training/neural/test_ranknet_loss_optimization.py:124, tests/training/neural/test_ranknet_pair_cache.py:170, tests/training/neural/test_torch_dataset_concurrency.py:376, tests/training/neural/test_weighted_loss_dot.py:171` | speedup-floors-without-perf_speedup_floor | TODO |
| NT-11 | P2 | `tests/feature_selection/info_theory/test_gil_release_threading_speedup.py:194` | gil-release-threading-speedup-1.2x | TODO |
| NT-13 | P2 | `tests/feature_selection/gpu/test_gpu_resident_fe.py:102` | gpu-k-chunk-precondition-depends-on-device-vram | TODO |
| NT-14 | P2 | `tests/feature_selection/shap_proxied/test_shap_proxy_search.py:96` | brute-force-n-chunks-depends-on-core-count | TODO |
| NT-18 | P2 | `tests/reporting/test_diagnostics_dispatch.py:283, tests/reporting/test_diagnostics_dispatch.py:286` | adversarial-auc-0.6-0.7-band-on-identical-splits | TODO |
| NT-19 | P2 | `tests/preprocessing/test_auto_transform_select_fold_leakage.py:81, tests/preprocessing/test_auto_transform_select_fold_leakage.py:106` | pure-noise-transform-score-band-0.3-0.7 | TODO |
| NT-23 | P2 | `tests/training/test_phase_summary_accounting.py:68, tests/training/test_phase_summary_accounting.py:53, tests/training/test_phase_summary_accounting.py:61` | sleep-based-registry-clock-assertions | TODO |
| NT-24 | P2 | `tests/reporting/test_kaleido_recovery.py:79, tests/reporting/test_kaleido_recovery.py:138` | kaleido-recovery-90s-and-15s-budgets | TODO |
| NT-25 | P2 | `tests/training/neural/test_lightning_callback_cache.py:37, tests/training/test_caching_pipeline_cache_obs.py:68, tests/training/test_preprocessing_fastpath_bench.py:64, tests/training/test_schema_drift_perf.py:56, tests/training/feature_selection/test_mrmr_identity_cache_and_monres_autoknot.py:144` | cached-call-overhead-microbudgets | TODO |
| SRD-04 | P2 | ``src/mlframe/feature_engineering/_benchmarks/_cpx36_baseline/fisher_weighted_residual_old.py:117` (twin: `src/mlframe/feature_engineering/transformer/fisher_weighted_residual.py:180-182`; asserted by `tests/feature_engineering/test_cpx36_batched_predict_identity.py:64`)` | cpx36 frozen baseline missed the empty-band fallback fix | TODO |
| SRD-05 | P2 | ``tests/feature_selection/filters/test_cat_interactions_pair_enum_vectorized.py:29-42` (twin: `src/mlframe/feature_selection/filters/_cat_interactions_step.py:296-308`)` | pair-enumeration identity test imports nothing from mlframe | TODO |
| SRD-06 | P2 | ``src/mlframe/feature_selection/filters/permutation.py:155` (canonical `_perm_pvalue`) (twins that inline a frozen add-one form: `src/mlframe/feature_selection/filters/estimators.py:170`, `src/mlframe/feature_selection/filters/_cmi_perm_stop.py:155`, `src/mlframe/feature_selection/filters/_conditional_permutation.py:123`, `src/mlframe/feature_selection/structure_discovery.py:161`)` | `MLFRAME_MRMR_ADDONE_PVALUE=0` is honoured at one of four MRMR p-value sites | TODO |
| SRD-07 | P2 | ``tests/feature_selection/wrappers/test_ranker_fs_group_relevance_identity.py:158-248` (twin: `src/mlframe/training/ranking/_ranker_fs.py:413-436`, `group_aware_mrmr_select`)` | group-aware MRMR greedy: three local copies asserted against each other | TODO |
| SRD-08 | P2 | ``tests/training/composite/kernels_identity/test_y_clip_bounds_quantile_bit_identity.py:15-85` (twin: `src/mlframe/training/composite/estimator/__init__.py:40-61`, `_y_train_clip_bounds`)` | y-clip-bounds "bit-identity" test never calls the function | TODO |
| SRD-09 | P2 | ``tests/feature_selection/regression/test_regression_mrmr_audit_2026_07_22.py:1064` (reference at `:1069`, inline copy of the optimised path at `:1107-1120`; twin: `src/mlframe/feature_selection/filters/_gpu_pairs.py:252-265`)` | GPU joint-MI reduction re-typed inside the test body | TODO |
| SRD-10 | P2 | ``tests/feature_selection/fe/categorical/test_cat_confirm_permutation_single_merge_cache.py:170-216` (twin: `src/mlframe/feature_selection/filters/_cat_confirm_permutation.py:683-960`)` | cat-confirm permutation identity test is degenerate (0.0 == 0.0) | TODO |
| SRD-11 | P2 | ``tests/feature_selection/filters/test_eng_dedup_batch_corr_masked_kernel.py:131-212` (twin: `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_core.py:758-868`)` | engineered-dedup: local "new" copy has already drifted from production | TODO |
| SRD-12 | P2 | ``tests/feature_engineering/test_hurst.py:118` (guard at `:121`) (twin: `src/mlframe/feature_engineering/hurst.py:226`)` | DFA reference keeps the pre-fix `n < 20` guard; production moved to `n < 50` | TODO |
| SRD-13 | P2 | ``tests/feature_engineering/test_numerical_stability_bench.py:113, 138, 157, 180` (twin: `src/mlframe/feature_engineering/_numerical_stable.py`, whole module)` | numerical-stability kernels: four print-only tests over a module with no production consumer | TODO |
| SRD-14 | P2 | ``tests/feature_engineering/transformer/test_local_curvature_quadterm_broadcast_identity.py:38, 64` (twin: `src/mlframe/feature_engineering/transformer/local_curvature.py:77-104`)` | local-curvature quad-term identity compares two local copies | TODO |
| VG-05 | P2 | `tests/training/test_single_slot_memo_write_order.py:78` | pd-view-memo-ordering-test-skips-exactly-when-it-should-fail | TODO |
| VG-06 | P2 | `tests/training/composite/discovery/test_discovery_unary_base_free.py:175` | unary-spec-base-column-contract-skipped-when-nothing-survives | TODO |
| VG-07 | P2 | `tests/training/composite/discovery/test_biz_val_training_composite_discovery.py:133` | composite-spec-schema-test-skips-on-empty-specs | TODO |
| VG-08 | P2 | `tests/feature_selection/mrmr/core/test_mrmr_error_messages_ux_audit.py:101` | fe-auto-userwarning-test-asserts-nothing-when-no-warning-fires | TODO |
| VG-09 | P2 | `tests/training/test_multibase_spec.py:152` | multibase-alphas-validation-skipped-on-domain-check | TODO |
| VG-10 | P2 | `tests/feature_selection/stability/test_stability_transform_validation.py:99` | missing-fit-column-raise-skipped-on-empty-support | TODO |
| VG-11 | P2 | `src/mlframe/feature_selection/filters/_ks_stability.py:64-82` | ks-stability-filter-returns-a-clean-report-when-it-inspected-no-columns | TODO |
| XGD-06 | P2 | `feature_selection/filters/_fe_gpu_strict.py:172 (with info_theory/_cmi_cuda.py:823)` | strict-gpu-defeats-disable_gpu-via-memoised-probe | TODO |
| XGD-07 | P2 | `feature_selection/filters/batch_pair_mi_gpu.py:473` | forced-cupy-silently-downgraded-to-njit | TODO |
| XMC-06 | P2 | ``src/mlframe/training/composite/cache.py:586`` | prebin-signature-tobytes-contradicts-its-own-docstring | TODO |
| XMC-08 | P2 | ``src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py:1155`` | fe-pair-sweep-threadpool-leaked-on-any-exception | TODO |
| XMC-09 | P2 | ``src/mlframe/feature_engineering/transformer/fisher_weighted_residual.py:115`` | fisher-gradient-stack-materialised-twice | TODO |
| XNUM-04 | P2 | ``src/mlframe/feature_engineering/spatial.py:576` (also `:577`, `:585` on the weight sums)` | `+ 1e-12` on `dist**power` in inverse-distance weighting | TODO |
| XNUM-05 | P2 | ``src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_explain.py:752`` | raw-moment model-to-model SHAP variance, clipped to zero | TODO |
| XNUM-06 | P2 | ``src/mlframe/feature_engineering/cat_cooccurrence_svd.py:115` (and `:117`)` | `sqrt(expected + 1e-12)` in the correspondence-analysis chi-square residual | TODO |
| XSD-04 | P2 | `see the table below (11 distinct dead module paths, 20 import sites)` | eleven-bench-and-profile-scripts-import-pre-carve-training-modules | TODO |
| LATCH-10 | P3 | `src/mlframe/feature_selection/filters/_kernel_tuning.py:32` | kernel-tuning-init-attempts-never-decay | TODO |
| LATCH-11 | P3 | `src/mlframe/training/_iterative_stratification_njit.py:35` | numba-global-rng-seeded-without-save-restore | TODO |
| LATCH-12 | P3 | `src/mlframe/feature_selection/filters/_fe_gpu_strict.py:75` | env-derived-cuda-verdict-frozen-in-a-device-availability-cache | TODO |
| LATCH-13 | P3 | `src/mlframe/training/neural/_triton_bootstrap.py:103` | triton-bootstrap-disabled-on-any-unexpected-error | TODO |
| LATCH-14 | P3 | `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_subsetrank.py:50` | LEAD: log-suppression flag latches, behaviour does not | TODO |
| NT-09 | P3 | `tests/feature_selection/shap_proxied/test_shap_proxy_preflight.py:69, :73, :302` | shap-preflight-additive-ratio-thresholds | TODO |
| NT-12 | P3 | `tests/feature_selection/biz_val/test_biz_val_filters_robust_basis_axis.py:83` | robust-fourier-axis-50x-spread-ratio | TODO |
| NT-20 | P3 | `tests/feature_selection/biz_val/test_biz_val_stratified_subsample.py:317` | cross-product-lift-bounded-by-0.02-auc | TODO |
| NT-21 | P3 | `tests/feature_engineering/test_biz_val_state_duration.py:203` | duration-only-auc-below-0.55-for-a-constant-feature | TODO |
| NT-22 | P3 | `tests/feature_engineering/test_biz_val_per_group_rank_causal.py:63` | causal-rank-auc-band-0.65-to-0.90 | TODO |
| NT-26 | P3 | `tests/training/fuzz/test_fuzz_isolate_runner.py:75, tests/training/fuzz/test_fuzz_isolate_runner.py:123` | fuzz-isolate-runner-reap-bounds | TODO |
| NT-27 | P3 | `tests/training/test_stress.py:213, tests/training/test_stress.py:248, tests/training/test_stress.py:249, tests/training/test_utils.py:549, tests/training/test_feature_selection.py:741, tests/training/test_memory_usage_polars_fastpath.py:96, tests/training/composite/cache/test_composite_update_ring_buffer.py:208, tests/training/core/test_training_core_a_fixes.py:310, tests/training/neural/test_neural_high_severity_regressions.py:357, tests/training/neural/test_neural_medium_severity_regressions.py:246, tests/feature_selection/wrappers/test_wrappers_invariants.py:450, tests/feature_selection/filters/test_qs_mah_edge_stress.py:200, tests/reporting/test_renderers_vocabulary.py:201, tests/preprocessing/test_preprocessing.py:233, tests/preprocessing/test_preprocessing.py:383, tests/training/test_confidence_analysis_fixes.py:303, tests/training/test_regression_drift_psi_array_cells.py:93, tests/data_valuation/test_biz_val_training_weight_adapter.py:103, tests/feature_selection/mrmr/core/test_mrmr_sis_screen.py:259, tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_regression_union/test_regression_diff_vs_l52.py:217` | stress-suite-5s-budgets-on-unseeded-fixtures | TODO |
| NT-28 | P3 | `tests/training/test_bizvalue_outliers_earlystop.py:374` | percentage-speedup-floor-on-early-stopping | TODO |
| SRD-15 | P3 | ``src/mlframe/feature_engineering/transformer/prediction_band_attention.py:44` (twin: `bidir_residual_band.py:41`)` | `prediction_band_attention` in-sample predictions (same cluster as SRD-01) | TODO |
| SRD-16 | P3 | ``tests/feature_selection/test_biz_val_greedy_backward_elimination.py:94-110` (twin: `src/mlframe/feature_selection/greedy_backward_elimination.py:151-164`)` | backward-elimination reference freezes the pre-fix coupled bar | TODO |
| SRD-17 | P3 | ``tests/feature_selection/wrappers/test_helpers_importance_scratch_identity.py:188-247` (twin: `src/mlframe/feature_selection/wrappers/_helpers_importance.py:284-302`)` | CPI reference's single-feature branch predates the nanmean/try-except fix | TODO |
| SRD-18 | P3 | ``tests/training/fuzz/test_fuzz_text_row_build.py:33-83` (twin: `tests/training/_fuzz_combo/frame_builder.py:239-280`)` | fuzz text-row test duplicates the builder and its vocab | TODO |
| SRD-19 | P3 | ``tests/feature_selection/screening/test_pre_screen_unsupervised.py:126-155` (twin: `src/mlframe/feature_selection/pre_screen.py:137-228`)` | pre-screen reference freezes a pre-sparse-support branch | TODO |
| SRD-20 | P3 | ``tests/estimators/test_cpx39_decorrelator_identity.py:12` (loop at `:20`) (twin: `src/mlframe/estimators/custom.py:37`)` | decorrelator reference returns labels; production canonicalised to positions | TODO |
| SRD-21 | P3 | ``tests/evaluation/test_bootstrap_fused_binary_bundle.py:38` (`jackknife_fns` at `:68`) (twin: `src/mlframe/training/honest_diagnostics.py:176-178`)` | bootstrap-bundle reference's jackknife wiring is stale | TODO |
| SRD-22 | P3 | ``tests/feature_engineering/transformer/test_adasyn_smote_synthesize_vectorized_identity.py:26` (loop at `:33`; twin: `src/mlframe/feature_engineering/transformer/adasyn_smote.py:53`)` | ADASYN reference lacks production's `n_neighbors` cap | TODO |
| SRD-23 | P3 | ``tests/feature_engineering/test_numerical.py:771` (`_reference_via_unique_path`)` | dead frozen reference in the numerical fast-path test | TODO |
| SRD-24 | P3 | ``tests/feature_engineering/transformer/test_mdl_binning_combo_count_vectorized.py:26-56, 76-117` (twin: `src/mlframe/feature_engineering/transformer/mdl_binning_pairwise.py:222-230`)` | MDL-binning combo tests compare two local copies | TODO |
| SRD-25 | P3 | ``tests/feature_selection/filters/test_cluster_aggregate_mi_compact_stack_identity.py:22-41` (twin: `src/mlframe/feature_selection/filters/_cluster_aggregate.py:582-583`)` | cluster-aggregate compact-stack test never calls the step it pins | TODO |
| SRD-26 | P3 | ``tests/feature_selection/fe/test_binned_numeric_agg_fe.py:226-274` (twin: `src/mlframe/feature_selection/filters/_binned_numeric_agg_fe.py:356-372`)` | binned-numeric-agg reference lacks production's two skip guards | TODO |
| SRD-27 | P3 | ``src/mlframe/training/composite/_benchmarks/bench_compare_bootstrap.py:26` (twin: `src/mlframe/training/composite/compare.py:162`)` | `bench_compare_bootstrap` identity check is now always False | TODO |
| SRD-28 | P3 | ``src/mlframe/votenrank/_benchmarks/bench_minimax_winning_votes.py:26` (twin: `src/mlframe/votenrank/leaderboard/_rules.py:167-173`)` | minimax bench freezes both sides and misses production's empty-opponents guard | TODO |
| SRD-29 | P3 | ``src/mlframe/metrics/_benchmarks/bench_drift_fused_merge_iter78.py:20, 29` (twin: `src/mlframe/metrics/_drift.py:345-348`, `:397-399`)` | drift bench's frozen kernels lack production's finite filter | TODO |
| SRD-30 | P3 | ``src/mlframe/feature_selection/filters/_orthogonal_hsic_fe.py:124` (twin: `src/mlframe/feature_selection/filters/_orthogonal_dcor_fe.py:102`)` | `_subsample_indices` inlined into a sibling module with nothing pinning them equal | TODO |
| VG-12 | P3 | `tests/training/test_base_leakage_guard.py:60` | leaky-bases-negative-assert-is-true-by-construction | TODO |
| VG-13 | P3 | `~25 sites, e.g. tests/feature_selection/biz_val/test_biz_val_filters_conditional_gate.py:188,` | fe-family-negative-assertions-cannot-tell-off-from-absent | TODO |
| VG-14 | P3 | `tests/feature_selection/contracts/test_fs_selector_contract.py:286` | get-feature-names-out-contract-only-checks-selectors-that-have-it | TODO |
| VG-15 | P3 | `tests/test_meta/test_swallowed_failures_are_audible.py:122` | narrowed-except-check-unpinned | TODO |
| VG-16 | P3 | `src/mlframe/training/composite/discovery/_fit_temporal.py:36-38` | base-leakage-guard-silently-no-ops-on-a-short-time-ordering | TODO |
| VG-17 | P3 | `src/mlframe/preprocessing/missing_indicator_pairing.py:76` | missing-indicator-pairing-loop-over-a-self-excluding-filter | TODO |
| XGD-08 | P3 | `feature_selection/filters/_batch_pair_mi_cuda_kernels.py:387` | cupy-reduction-order-claimed-bit-identical-to-sequential-loop | TODO |
| XGD-09 | P3 | `metrics/_gpu_metrics.py:190` | rmse-atomic-add-partials-are-run-to-run-nondeterministic | TODO |
| XGD-10 | P3 | `feature_selection/filters/gpu.py:513` | hardcoded-npermutations-32-fanout-changes-early-stop-granularity | TODO |
| XGD-11 | P3 | `feature_selection/filters/engineered_recipes/_recipe_unary_binary_gpu.py:264` | engineered-recipe-gpu-replay-f32-vs-cpu-fallback-f64 | TODO |
| XGD-12 | P3 | `feature_selection/filters/gpu.py:643` | mi_direct_gpu-permutes-a-caller-owned-device-buffer-in-place | TODO |
| XSD-05 | P3 | `src/mlframe/feature_selection/wrappers/rfecv/_fit.py:1` | stale-prose-path-feature-selection-wrappers-rfecv | TODO |
| XSD-06 | P3 | `src/mlframe/signal/dtw.py:486` | stale-prose-path-mlframe-signal-dtw-autotune | TODO |
| XSD-07 | P3 | `src/mlframe/training/_format.py:22` | stale-prose-path-training-core-short-model-tag | TODO |
| XSD-08 | P3 | `tests/training/test_discovery_cache_version_tuple_expanded.py:20` | stale-prose-path-training-utils-compute-config-signature-v1 | TODO |
