"""Vulture whitelist: confirmed false positives in tests/, not to be re-flagged.

Mirrors ``scripts/vulture_whitelist.py``'s convention (a bare name as an expression
statement counts as a "use" in vulture's model) but scoped to tests/-only names:
pytest fixture parameters that a test never references in its own body (the fixture's
setup/teardown side effect IS the point), and monkeypatch/mock replacement-function
parameters kept only to match the real function's calling signature.

This file is never imported by test or production code, only parsed by vulture
alongside the real target: ``vulture tests scripts/vulture_whitelist.py
tests/vulture_whitelist.py``.
"""
# ruff: noqa: B018, F821 -- every bare name below is a vulture "use" marker for an
# intentionally-undefined name (vulture's own --make-whitelist convention); this file is
# never imported or executed, only parsed by vulture, so ruff's real-bugs checks (useless
# expression, undefined name) are false positives here by construction.
# mypy: ignore-errors

# --- pytest fixture params consumed only for their setup/teardown side effect, never
# referenced in the test body itself (2026-08-04 vulture sweep of tests/). ---
session  # tests/conftest.py pytest_sessionfinish hook -- pytest's own hook signature
exitstatus
no_gpu  # tests/feature_selection/fe/gpu/test_exhaustive_device_independent.py -- monkeypatches GPU-absence
warmed  # tests/feature_selection/test_prewarm.py -- module-scoped prewarm fixture, asserted on via its side effects
reset_enabled  # tests/reporting/test_crash_reporting.py -- resets faulthandler state before/after each test
no_cache  # tests/training/composite/kernels_identity/test_ktc_dispatch_fallback.py -- disables KTC caching for the test
monotonic_lru_clock  # tests/training/test_cache_lru_eviction.py -- patches time.monotonic for deterministic eviction order
fake_phase_timer  # tests/training/test_phases_registry.py -- patches the phase-timing clock
patched_pysr  # tests/training/test_fe_pysr_seed.py -- monkeypatches the vendored pysr module
recwarn  # tests/training/test_hf_provider.py -- pytest's built-in warnings-recorder fixture
sample_categorical_classification_data  # tests/training/test_session_fixture_immutability_extended_sensor.py -- session-scoped fixture immutability probe

# --- Monkeypatch/mock/stub function params kept only to match the real function's
# calling signature (same class as scripts/vulture_whitelist.py's "Framework-mandated /
# documented-no-op parameter names" section, scoped to tests/). ---
constants  # test_core_infra_a_fixes.py fake_cv_func(**constants) -- CV callable signature
init_model  # test_evaluation_fixes.py fake CatBoost fit() -- sklearn/CatBoost fit() signature parity
window_index_name  # test_coverage_fill.py -- apply_fcn callback signature parity
create_features_names
max_bins  # test_mdl_binning_split_kernel.py _mdl_bin_edges_ref -- reference impl matches the real kernel's signature
rss_delta_mb  # test_biz_value_param_oracle.py quality_objective callback signature
segment_min_agreement  # test_monotonic_stability_batched_spearman_equivalence.py -- real function's kwarg-only signature
bytes_needed  # discretization vram-guard tests -- monkeypatched fe_gpu_has_vram_cushion(bytes_needed) signature
stack_arr  # test_orthogonal_scorer_bugfixes.py _fake_coefs -- real scorer callback signature
pre_transform  # test_fe_numeric_hygiene_guards.py _fake_eval -- EngineeredRecipe.eval signature parity
preprocess_params
assume_finite  # test_fe_pairs_failed_transform_no_stale_buffer.py _stub_discretize_2d_quantile_batch -- real kernel signature
required_bytes  # test_batch_pair_mi_gpu_vram_guard.py -- monkeypatched _gpu_upload_fits(required_bytes) signature
with_extremes  # test_gpu_resident_fe.py old_path -- real _radix_select_interior_edges signature parity
package  # test_kernel_tuning_cli_force.py / test_gen_default_tuning.py -- discover_fn(package=...) signature parity
N_CHAR_MAX  # test_mrmr_sklearn_joblib_compat_audit.py -- sklearn BaseEstimator.__repr__ signature parity
resolved_via_apply  # test_mrmr_append_engineered_no_repeated_assign.py _make_recipes -- recipe constructor kwarg
n_cands  # test_audit_cache_locks.py _fake_uncached -- real gate-computation signature parity
edge_cap  # test_shap_proxy_cluster.py _mock_gpu -- real GPU-dense kernel signature parity
art  # test_wide_data_scalability.py _run_shap stub -- real _run_shap(..., art) signature parity
input_features  # get_feature_names_out(input_features=None) -- sklearn transformer API signature parity
y_round  # test_biz_val_zero_importance_pruning.py _importance_fn -- real importance-fn callback signature
current_features  # test_rfecv_stability_swap_ci_regressions.py _fake_get_fi -- real FI-getter signature parity
importance_getter
reference_data  # test_helpers_importance_except_branch.py non_numeric_getter -- real getter signature parity
data_  # test_helpers_newaxes_fixes.py _spy -- real permutation-importance callable signature parity
target_
durable  # _save_threads_zero test helpers -- io.save_mlframe_model's real signature (bypasses a Windows zstd quirk)
auto_open  # test_plotly_kaleido_module_split_inv57.py write_html stub -- plotly's real write_html signature
include_plotlyjs
base1_scale  # test_composite_discovery_per_group.py _block -- synthetic-data builder kwarg
base2_scale
censor_frac  # test_composite_survival.py _make_aft -- synthetic AFT-data builder kwarg
revision  # test_training_feature_handling_fixes.py _fake_from_pretrained -- HF from_pretrained() signature parity
trust_remote_code
logical  # test_mrmr_kwargs_defaults_merge.py -- monkeypatched psutil.cpu_count(logical=...) signature
scale_y  # test_input_normalization_strategies.py _base_params -- config-builder kwarg
cuda_available  # test_mlp_runtime_defaults.py -- monkeypatched _probe_available_memory_bytes(cuda_available=...) signature
expected_self_destruct  # test_pipeline_output_preserves_pandas.py -- pytest.mark.parametrize id-pairing, consumed via the param tuple
kept_specs  # test_tiny_rerank_escape.py _make_self_stub -- self-stub constructor kwarg
timestamp_col  # test_temporal_audit_polars_fastpath.py _fake_audit -- real audit-fn signature parity
predict_params  # predict(self, X, **predict_params) -- sklearn predict() signature parity across TTR/quantile test doubles
dataloaders  # test_critical_fixes_critical.py predict stub -- PyTorch Lightning Trainer.predict() signature parity
model_out  # test_feature_handling_high_feature_handling.py _pool -- HF pooling-callback signature parity
encode_categoricals  # test_pysr_column_names_collision_free.py / test_pysr_y_train_wiring.py fake_run_pysr -- real run_pysr signature parity
train_texts  # test_provider_registry.py fit(self, train_texts) -- provider Protocol signature parity
evals_log  # test_trainer.py after_iteration -- LightGBM/CatBoost callback signature parity
ntree_end  # test_cb_iteration_metrics_target_type.py _RankerModel/_ClassifierModel stubs -- CatBoost predict/predict_proba(pool, ntree_end) signature parity
tokenizer  # test_fairness_computation.py _fake_scorer -- naive_*_score(model, tokenizer, sentence) real scorer signature parity
y_i8  # test_split_njit_fallback_logs.py _boom -- real _iterative_stratification_njit(y_i8, r, seed_int) signature parity
seed_int

# --- unsatisfiable-condition findings: intentionally unreachable branches, each with its
# own inline rationale at the call site (kept as `_NEVER_TRUE`-gated dead code, not deleted,
# per the comment at each site -- vulture has no per-line suppression, so the branch guard
# itself was changed from a literal `if False` to a module-level name so vulture's constant-
# folding no longer flags it; this whitelist entry documents that decision, not a silenced name). ---

# --- tests/feature_selection/conftest.py: IS_FAST_MODE is a documented re-export (see the
# comment directly above its import) so subdir tests can keep importing it from this conftest. ---
IS_FAST_MODE

# --- tests/reporting/test_every_writer_honours_format_subfolders.py: ``subfolders_on`` is a pytest fixture,
# requested by name in each test signature. vulture sees the parameter as an unused local. ---
subfolders_on
