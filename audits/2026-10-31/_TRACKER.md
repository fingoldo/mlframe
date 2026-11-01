# Full attack-surface audit 2026-10-31 -- tracker

Eight read-only reports on `src/mlframe`, one per attack class. No code was changed by the audit. Each finding carries a
proposed disposition: **RESOLVED** (already safe, citation given), **FUTURE** (real, deferred, reason given),
**DOC** (intended behaviour, to be documented), **REJECTED** (not a risk, reason given).

## Counts by class

| Class | Report | High | Medium | Low | Info | Total |
|---|---|---|---|---|---|---|
| Unsafe deserialization | [deserialization.md](deserialization.md) | 1 | 7 | 4 | 10 | 22 |
| Dynamic code execution | [dynamic_exec.md](dynamic_exec.md) | 0 | 0 | 10 | 19 | 29 |
| Subprocess / shell | [subprocess.md](subprocess.md) | 0 | 0 | 3 | 9 | 12 |
| Network / servers | [network.md](network.md) | 0 | 0 | 4 | 5 | 9 |
| Temp files / filesystem | [filesystem.md](filesystem.md) | 0 | 1 | 5 | 10 | 16 |
| Secrets / credentials | [secrets.md](secrets.md) | 0 | 1 | 3 | 1 | 5 |
| Dependencies / supply chain | [supply_chain.md](supply_chain.md) | 0 | 3 | 6 | 6 | 15 |
| Untrusted input / resources | [untrusted_input.md](untrusted_input.md) | 0 | 2 | 3 | 4 | 9 (sampled) |

## Priority items (proposed)

- **DESER-10 (High), RESOLVED** in `303e54be4`: the cache loader deferred to `MLFRAME_ALLOW_UNVERIFIED_PICKLE` (default closed) instead
  of forcing `allow_unverified=True`. Pinned by `tests/training/composite/cache/test_cache_unsigned_pickle_rejected.py`, which fails
  on the previous code.
- **DESER-02 (Medium)**, `training/io.py` `load_mlframe_model`: the default model-bundle loader has no sha256 check.
  **Proposed: FUTURE** (needs a manifest format change).
- **DESER-06 (Medium)**, `estimators/pipelines.py:111`: unrestricted `joblib.load` after an integrity-only sidecar check.
  **Proposed: FUTURE.**
- **FS-02 (Medium), RESOLVED**: `estimators/pipelines.py`: the default dump no longer sits at a predictable name in the shared temp
  root; it goes to a fresh `tempfile.mkdtemp` directory (0o700 on POSIX) created once per top-level call. Pinned by
  `tests/estimators/test_pipeline_gridsearch_default_dump_dir_private.py`, which fails on the previous code.
- **DESER-06 (Medium), FUTURE**: `pipelines.py` `replay_cv_results` still calls `joblib.load` after an integrity-only sidecar.
  `safe_pickle.safe_load` cannot be swapped in because it uses `pickle.loads` and the dumps are zlib-compressed joblib. Needs an
  authenticity key or a format change, which is a design decision.
- **SEC-01 (Medium), RESOLVED** in `620c0d2d2`: the give-up log carries the scrubbed message only, and the re-raised error is no longer
  chained to the original. Pinned by `tests/integrations/test_mlflow_scrubbed_giveup_log.py`, which fails on the previous code.
- **SUP-02 (Medium)**, `pyproject.toml:154`: `pyutilz>=1.0.0` is not on PyPI, so pip users are exposed to dependency confusion.
  **Proposed: FUTURE** (pin or document the pre-install).
- **SUP-03 (Medium)**, `.github/workflows/dependabot-auto-merge.yml:36-43`: every dependabot PR auto-merges without review.
  **Proposed: FUTURE** (review gate or narrower allow-list).
- **INP-01 / INP-02 (Medium)**, `calibration/policy.py:461`, `votenrank/shapley_blend.py:183`: unbounded allocations
  driven by caller sizes. **Proposed: FUTURE** (caps and chunking).

## Status

Two items fixed (DESER-10, SEC-01), each with a regression test that fails on the previous code. The rest are proposals. Per the project's audit rule, the fixes are triaged with the
owner before any code changes, because several change public behaviour (DESER-10 most of all). The remaining Low and Info
rows are recorded in their class reports with the same vocabulary.

Scope limits, stated by the reports themselves: the secrets pass did not finish the whole tree and the tests and YAML/JSON were
not fully enumerated; the resource-exhaustion pass was sampled, not exhaustive; the supply-chain pass could not check live
PyPI or branch protection from this environment.

## Findings and proposed dispositions

Each row is a finding and the disposition its report proposed. None is executed yet: a `RESOLVED` proposal still needs its citation checked, and `FUTURE`, `DOC` and `REJECTED` each need the work or the write-up the report names.

| Status | ID | Class | Location |
|---|---|---|---|
| **DOC** | DESER-01 | deserialization | `src/mlframe/training/io.py:601` |
| **RESOLVED** | DESER-02 | deserialization | `src/mlframe/training/io.py:524` (`load_mlframe_model`, callers `_trainer_train_and_evaluate_parts.py:126`, `core/_phase_persist_post.py:49`)` |
| **DOC** | DESER-03 | deserialization | `src/mlframe/training/_unpickle_policy.py` (`resolve_restricted_class`, allowlist prefixes and `_DENIED_NAMES`)` |
| **DOC** | DESER-04 | deserialization | `src/mlframe/inference/predict.py:245` (`safe_joblib_load` in `read_trained_models`)` |
| **DOC** | DESER-05 | deserialization | `src/mlframe/inference/predict.py:123` (`features.dump` joblib fallback)` |
| **RESOLVED** | DESER-06 | deserialization | `src/mlframe/estimators/pipelines.py:111` (`replay_cv_results`, plain `joblib.load`)` |
| **RESOLVED** | DESER-07 | deserialization | `src/mlframe/training/core/_metadata_loader.py:20-21` (legacy joblib metadata, no sidecar)` |
| **DOC** | DESER-08 | deserialization | `src/mlframe/training/feature_handling/cache.py:574-580` (pickle fallback, `allow_pickle=True`)` |
| **RESOLVED** | DESER-09 | deserialization | `src/mlframe/training/feature_handling/cache.py:557` (`np.load` with `allow_pickle=False` default)` |
| **FUTURE** | DESER-10 | deserialization | `src/mlframe/training/composite/cache_store.py:385` (`safe_load(..., allow_unverified=True)`)` |
| **RESOLVED** | DESER-11 | deserialization | `src/mlframe/training/_trainer_train_and_evaluate_parts.py:117-119` (`trusted_root` defaults to the file's own directory)` |
| **RESOLVED** | DESER-12 | deserialization | `src/mlframe/utils/disk_cache.py:368` |
| **RESOLVED** | DESER-13 | deserialization | `src/mlframe/feature_selection/wrappers/rfecv/__init__.py:598` |
| **RESOLVED** | DESER-14 | deserialization | `src/mlframe/feature_engineering/transformer/_key_bank.py:179-182, 182, 204` |
| **REJECTED** | DESER-15 | deserialization | `src/mlframe/training/composite/discovery/_tiny_rerank_process.py:92-93` (`cloudpickle.loads`)` |
| **RESOLVED** | DESER-16 | deserialization | `src/mlframe/training/_unpickle_policy.py:145` (`torch.load(..., weights_only=True)`)` |
| **RESOLVED** | DESER-17 | deserialization | `src/mlframe/training/neural/_recurrent_wrapper_base.py:581` |
| **RESOLVED** | DESER-18 | deserialization | `src/mlframe/training/neural/_flat_torch_module/__init__.py:591` |
| **RESOLVED** | DESER-19 | deserialization | `src/mlframe/feature_selection/filters/_vendored/infonet/infer.py:51` |
| **REJECTED** | DESER-20 | deserialization | `Dev benchmark caches: `feature_selection/_benchmarks/fs_hybrid/round4_*.py` (47, 64, 78, 132, 244), `reporting/_benchmarks/bench_gil_contention.py:50,70`, `bench_async_render_overlap.py:87` |
| **REJECTED** | DESER-21 | deserialization | `Same-process round-trips: `round4_fe_median_gated_recipe_proof.py:47`, `test_hybrid_tree_member.py:136` |
| **RESOLVED** | DESER-22 | deserialization | `src/mlframe/feature_selection/_benchmarks/fs_hybrid/_realdata_cache.py:197` (`np.load(allow_pickle=False)`)` |
| **RESOLVED** | EXE-01 | dynamic_exec | `src/mlframe/calibration/post.py:79-80` |
| **REJECTED** | EXE-02 | dynamic_exec | `src/mlframe/calibration/post.py:191, 233, 235, 239, 281` |
| **RESOLVED** | EXE-03 | dynamic_exec | `src/mlframe/data/datasets/__init__.py:80-81` |
| **DOC** | EXE-04 | dynamic_exec | `src/mlframe/estimators/pipelines.py:18` |
| **RESOLVED** | EXE-05 | dynamic_exec | `src/mlframe/feature_selection/filters/engineered_recipes/_recipe_dispatch.py:26` |
| **RESOLVED** | EXE-06 | dynamic_exec | `src/mlframe/feature_selection/filters/_orthogonal_meta_scorer_fe.py:16-17` |
| **RESOLVED** | EXE-07 | dynamic_exec | `src/mlframe/feature_selection/filters/_permutation_null_resident_ktc.py:113` |
| **RESOLVED** | EXE-08 | dynamic_exec | `src/mlframe/feature_selection/__init__.py:128` |
| **RESOLVED** | EXE-09 | dynamic_exec | `src/mlframe/training/_param_model_spec.py:48` |
| **RESOLVED** | EXE-10 | dynamic_exec | `src/mlframe/training/composite/_estimator_dispatch.py:70-72` |
| **RESOLVED** | EXE-11 | dynamic_exec | `src/mlframe/training/__init__.py:324-325` |
| **RESOLVED** | EXE-12 | dynamic_exec | `src/mlframe/training/_polars_native_support.py:109` |
| **RESOLVED** | EXE-13 | dynamic_exec | `src/mlframe/training/core/_setup_helpers_pipeline_cache.py:155-156` |
| **REJECTED** | EXE-14 | dynamic_exec | `src/mlframe/system/kernel_tuning_cache/_discovery.py:81, 91` |
| **RESOLVED** | EXE-15 | dynamic_exec | `src/mlframe/preprocessing/opt_in_slow.py:58-59` |
| **RESOLVED** | EXE-16 | dynamic_exec | `src/mlframe/_optional_imports.py:12` |
| **RESOLVED** | EXE-17 | dynamic_exec | `src/mlframe/__init__.py:482` |
| **RESOLVED** | EXE-18 | dynamic_exec | `src/mlframe/training/neural/_flat_torch_module/__init__.py:216-217` |
| **FUTURE** | EXE-19 | dynamic_exec | `src/mlframe/data/datasets/_sapp.py:17` |
| **RESOLVED** | EXE-20 | dynamic_exec | `src/mlframe/data/datasets/_sapp.py:21` |
| **DOC** | EXE-21 | dynamic_exec | `src/mlframe/feature_engineering/_benchmarks/bench_anchor_slope_cache.py:36` |
| **DOC** | EXE-22 | dynamic_exec | `src/mlframe/feature_selection/filters/_benchmarks/bench_conditional_permutation_strata_arrs.py:45` |
| **DOC** | EXE-23 | dynamic_exec | `src/mlframe/feature_selection/filters/_benchmarks/bench_ratio_delta_fe.py:49` |
| **DOC** | EXE-24 | dynamic_exec | `src/mlframe/feature_selection/filters/_benchmarks/bench_target_encoding_apply.py:36` |
| **DOC** | EXE-25 | dynamic_exec | `src/mlframe/feature_selection/_benchmarks/bench_pairwise_modular_cost.py:71` |
| **DOC** | EXE-26 | dynamic_exec | `src/mlframe/training/composite/_benchmarks/bench_venn_abers_ivap_envelopes.py:28` |
| **RESOLVED** | EXE-27 | dynamic_exec | `src/mlframe/feature_selection/_benchmarks/profiling/duplicate_d2h.py:27; fit_wall.py:32` |
| **REJECTED** | EXE-28 | dynamic_exec | `src/mlframe/feature_selection/_benchmarks/profile_mrmr_line_profile.py:58-59` |
| **DOC** | EXE-29 | dynamic_exec | `Benchmark and profiling importlib sites: `_benchmarks/nvprof_autotag.py:44`, `fs_hybrid/_arms.py:86`, `fs_hybrid/_matched_k.py:56`, `bench_mrmr.py:85` |
| **DOC** | FS-01 | filesystem | `src/mlframe/integrations/mlflow.py:95` |
| **RESOLVED** | FS-02 | filesystem | `src/mlframe/estimators/pipelines.py:163-166` |
| **FUTURE** | FS-03 | filesystem | `src/mlframe/training/core/_setup_helpers_pipeline_cache.py:136-138` |
| **FUTURE** | FS-04 | filesystem | `src/mlframe/training/crash_diagnostics.py:130` |
| **RESOLVED** | FS-05 | filesystem | `src/mlframe/training/feature_handling/hf_provider.py:113-124` |
| **FUTURE** | FS-06 | filesystem | `src/mlframe/feature_selection/filters/_benchmarks/bench_temporal_agg_fe.py:39`, `src/mlframe/reporting/_benchmarks/_e2e_slice_finder_ab_iter71.py:49-50` |
| **RESOLVED** | FS-07 | filesystem | `src/mlframe/system/kernel_tuning_cache/_ensure.py:96-101, 131-134` |
| **RESOLVED** | FS-08 | filesystem | `src/mlframe/training/io.py:58-68, 155-167` |
| **RESOLVED** | FS-09 | filesystem | `src/mlframe/utils/disk_cache.py:420-430` |
| **RESOLVED** | FS-10 | filesystem | `src/mlframe/training/composite/cache_store.py:175-178, 563` |
| **RESOLVED** | FS-11 | filesystem | `src/mlframe/feature_selection/wrappers/rfecv/__init__.py:545-560` |
| **RESOLVED** | FS-12 | filesystem | `src/mlframe/feature_selection/wrappers/rfecv/_cb_border_cache.py:114-127, 183-190` |
| **RESOLVED** | FS-13 | filesystem | `src/mlframe/feature_selection/filters/_joblib_safe.py:350-365` |
| **RESOLVED** | FS-14 | filesystem | `src/mlframe/training/cb/_cb_gpu_monitor.py:490` |
| **REJECTED** | FS-15 | filesystem | `src/mlframe/reporting/report_html.py:258`, `src/mlframe/reporting/renderers/plotly.py:418-424`, `src/mlframe/reporting/renderers/_kaleido.py:235-243`, `src/mlframe/training/ranking/_ranker_suite_train_helpers.py:484` |
| **RESOLVED** | FS-16 | filesystem | `whole `src/mlframe` tree (no `chmod`, `umask`, `0o666`, `0o777`)` |
| **DOC** | NET-01 | network | `src/mlframe/integrations/mlflow.py:71-96` |
| **RESOLVED** | NET-02 | network | `src/mlframe/feature_selection/filters/_neural_mi.py:284-291` |
| **RESOLVED** | NET-03 | network | `src/mlframe/feature_selection/filters/_neural_mi.py:443-477` |
| **RESOLVED** | NET-04 | network | `src/mlframe/training/feature_handling/hf_provider.py:66-88, 204-256` |
| **DOC** | NET-05 | network | `src/mlframe/training/feature_handling/hf_provider.py:74-88` |
| **FUTURE** | NET-06 | network | `src/mlframe/training/feature_handling/hf_provider.py:236-256`, `src/mlframe/feature_selection/filters/_neural_mi.py:473` |
| **REJECTED** | NET-07 | network | `src/mlframe/feature_selection/_benchmarks/fs_hybrid/_realdata_cache.py:136-146` (and round3/round4/_downstream_shared bench files)` |
| **RESOLVED** | NET-08 | network | `src/mlframe/integrations/mlflow.py:34-41, 150-177` |
| **REJECTED** | NET-09 | network | `src/mlframe/reporting/renderers/_plotly_interactivity.py:24` |
| **RESOLVED** | SEC-01 | secrets | `src/mlframe/integrations/mlflow.py:148` |
| **FUTURE** | SEC-02 | secrets | `.github/workflows/pandas-matrix.yml:163`, `.github/workflows/polars-matrix.yml:167` |
| **RESOLVED** | SEC-03 | secrets | `src/mlframe/training/feature_handling/providers.py:43-54` |
| **RESOLVED** | SEC-04 | secrets | `src/mlframe/training/feature_handling/providers.py:115-123`, `:126-150` (params handling)` |
| **DOC** | SEC-05 | secrets | `src/mlframe/training/feature_handling/providers.py:269-280` |
| **RESOLVED** | CMD-01 | subprocess | `src/mlframe/utils/native_import_probe.py:56` |
| **FUTURE** | CMD-02 | subprocess | `src/mlframe/feature_selection/_benchmarks/fs_hybrid/_staleness.py:89` (+ `:122-125`)` |
| **RESOLVED** | CMD-03 | subprocess | `src/mlframe/training/_gpu_state_probe.py:88` and `:99` |
| **RESOLVED** | CMD-04 | subprocess | `src/mlframe/training/_polars_native_support.py:90` |
| **RESOLVED** | CMD-05 | subprocess | `src/mlframe/system/kernel_tuning_cache/_ensure.py:164` |
| **RESOLVED** | CMD-06 | subprocess | `src/mlframe/feature_selection/_benchmarks/fs_hybrid/_manifest.py:74` |
| **DOC** | CMD-07 | subprocess | `src/mlframe/utils/native_import_probe.py:36, 73` |
| **RESOLVED** | CMD-08 | subprocess | `src/mlframe/training/pipeline/_pipeline_extensions_pysr.py:12, 246` |
| **DOC** | CMD-09 | subprocess | `src/mlframe/feature_selection/_benchmarks/profiling/family_batch.py:45`, `family_scale.py:167` |
| **RESOLVED** | CMD-10 | subprocess | `17 `_benchmarks/` files that call `git show` / `git rev-parse` / `git -C` (see details)` |
| **RESOLVED** | CMD-11 | subprocess | `feature_selection/_benchmarks/bench_iter87_cumulative.py:119`, `bench_iter100_stratified_anchors_contour.py:113`, `bench_iter101_stratified_anchors_2d.py:119` |
| **DOC** | CMD-12 | subprocess | `feature_selection/filters/_benchmarks/bench_sf3_jmim_exponent_selection.py:80` |
| **DOC** | SUP-01 | supply_chain | `pyproject.toml:176` |
| **FUTURE** | SUP-02 | supply_chain | `pyproject.toml:154` |
| **FUTURE** | SUP-03 | supply_chain | `.github/workflows/dependabot-auto-merge.yml:36-43` |
| **FUTURE** | SUP-04 | supply_chain | `.github/workflows/macos-abort-probe.yml:173-175` |
| **FUTURE** | SUP-05 | supply_chain | `.github/workflows/docs.yml:45-46; codecov-full.yml:75; dep-floors.yml:89` |
| **DOC** | SUP-06 | supply_chain | `pyproject.toml:64-65, 88, 79 (open-ended runtime floors)` |
| **DOC** | SUP-07 | supply_chain | `.github/workflows/pandas-matrix.yml:94; polars-matrix.yml:97; sklearn-matrix-ci.yml:140; numba-coverage.yml:141; gpu-matrix.yml:60` |
| **REJECTED** | SUP-08 | supply_chain | `.github/workflows/fs-benchmark-nightly.yml:89,98; gpu-matrix.yml:38` |
| **REJECTED** | SUP-09 | supply_chain | `.github/workflows/gpu-matrix.yml:67-68` |
| **RESOLVED** | SUP-10 | supply_chain | `requirements-dev.txt:65` |
| **RESOLVED** | SUP-11 | supply_chain | `.github/workflows/*.yml (all `uses:` lines)` |
| **RESOLVED** | SUP-12 | supply_chain | `.github/workflows/dependabot-auto-merge.yml:26; gpu-matrix.yml:16-20; ci.yml:16; ci.yml:215,487` |
| **RESOLVED** | SUP-13 | supply_chain | `.github/workflows/ci.yml:548-559; numba-coverage.yml:326-330; codecov-full.yml:56-62` |
| **RESOLVED** | SUP-14 | supply_chain | `.github/workflows/codecov-full.yml:36, 56-62` |
| **RESOLVED** | SUP-15 | supply_chain | `uv.lock:12331, uv.lock (1797 registry refs)` |
| **RESOLVED** | INP-01 | untrusted_input | `src/mlframe/calibration/policy.py:461` |
| **REJECTED** | INP-02 | untrusted_input | `src/mlframe/votenrank/shapley_blend.py:183` |
| **RESOLVED** | INP-03 | untrusted_input | `src/mlframe/data_valuation/_mc_sampling.py:102` |
| **RESOLVED** | INP-04 | untrusted_input | `src/mlframe/feature_engineering/graph_spectral_features.py:56` |
| **RESOLVED** | INP-05 | untrusted_input | `src/mlframe/reporting/output.py:55` |
| **DOC** | INP-06 | untrusted_input | `src/mlframe/preprocessing/_cleaning_helpers.py:188` |
| **DOC** | INP-07 | untrusted_input | `src/mlframe/training/targets_performance.py:284` |
| **REJECTED** | INP-08 | untrusted_input | `src/mlframe/calibration/post.py:30-33` |
| **RESOLVED** | INP-09 | untrusted_input | `src/mlframe/feature_engineering/_grouped_segments.py:92-96` |


## Dispositions after the implementation rounds (2026-11-02)

Rows marked RESOLVED were implemented in the 2026-11-01 rounds (commits are listed in `audits/2026-11-01/_TRACKER.md`) or were already guarded at HEAD:
INP-01 (`calibration/policy.py` size guard `_resample_matrix_max_bytes`, `tests/calibration/test_statistical_rigor_fixes.py`) and INP-04 (`graph_spectral_features` raises above `max_nodes=2000` before allocating the adjacency).
INP-03 is fixed by `data_banzhaf` drawing its masks in byte-bounded blocks (`tests/data_valuation/test_data_banzhaf_mask_blocks.py`).
INP-02 is REJECTED: `shapley_blend` has no exact 2^n enumeration (`_permutation_shapley` and `_msr_banzhaf` sample), and its only allocations are linear in the caller's permutation count.
DESER-03/04/05 are DOC under the denylist trust model stated in `safe_joblib_load`.
Still open from this tracker: DESER-10, EXE-19, FS-03, FS-04, FS-06, NET-06, SEC-02, CMD-02, SUP-02, SUP-03, SUP-04, SUP-05.
