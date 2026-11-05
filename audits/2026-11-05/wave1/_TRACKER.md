# Wave 1 audit 2026-11-05 -- tracker

Ten read-only reports by Opus 5.5 low-effort agents, one per attack class, on the whole `src/mlframe` tree after the trackers of 2026-10-31 to 2026-11-04 were closed. The agents could not write files, so each report is the agent's returned text, condensed, in the class file named below. Each finding is re-checked on HEAD before it is fixed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED, each with its reason. All rows start OPEN.

Totals: 47 findings (High 5, Medium 15, Low 25, Info 2).

| Class | Report | High | Medium | Low | Info | Total |
|---|---|---|---|---|---|---|
| A1 deserialization | [deserialization.md](deserialization.md) | 0 | 1 | 2 | 0 | 3 |
| A2 dynamic exec, subprocess, filesystem | [exec_filesystem.md](exec_filesystem.md) | 0 | 0 | 1 | 0 | 1 |
| A3 network, secrets, supply chain | [network_supply_chain.md](network_supply_chain.md) | 0 | 0 | 3 | 0 | 3 |
| A4 untrusted input, resource exhaustion | [resource_exhaustion.md](resource_exhaustion.md) | 2 | 2 | 1 | 1 | 6 |
| A5 statistics and leakage | [statistics_leakage.md](statistics_leakage.md) | 1 | 2 | 1 | 0 | 4 |
| A6 numerics, degenerate inputs | [numerics_degenerate.md](numerics_degenerate.md) | 0 | 1 | 3 | 0 | 4 |
| A7 kernels, CPU/GPU parity | [kernels_parity.md](kernels_parity.md) | 0 | 1 | 1 | 1 | 3 |
| A8 concurrency, state, caches | [concurrency_state.md](concurrency_state.md) | 1 | 2 | 5 | 0 | 8 |
| A9 error handling, silent failures | [silent_failures.md](silent_failures.md) | 1 | 4 | 3 | 0 | 8 |
| A10 API contract, configuration | [api_contract.md](api_contract.md) | 0 | 2 | 5 | 0 | 7 |

| Status | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | S5-01 | leakage | High | `feature_engineering/transformer/baseline_surprise.py:80` and nine sibling modules: inner baseline KFold ignores outer groups; confirmed on HEAD, and eight more modules importing the shared baseline under an alias had the same gap; the inner split now comes from `_baseline_oof.kfold_splits` in all 22 Mode A callers (`tests/feature_engineering/transformer/test_inner_baseline_folds_are_group_aware.py`); the aux splitter of `_residual_oof.py` and its users `pred_augmented`, `residual_attention`, `cross_feature_reconstruction` takes the labels too, 25 modules in all |
| **OPEN** | X9-01 | silent_failure | High | `feature_selection/hybrid_selector.py:525-527, 915-916`: transform drops engineered columns on a replay failure and returns the first column |
| **OPEN** | C8-01 | state | High | `feature_selection/filters/_mrmr_fit_impl/_finalise.py:659`: process-wide fit cache holds the live instance |
| **OPEN** | R4-01 | resource | High | `reporting/charts/pdp_ice.py:471-490`: categorical PDP axis sweeps every category |
| **OPEN** | R4-02 | resource | High | `metrics/classification/_weighted_kappa.py:61,74-78`: class count taken from the largest label value |
| **OPEN** | D1-01 | deserialization | Medium | `training/feature_handling/cache.py:557`: `np.load(allow_pickle=True)` unpickles before the sidecar check |
| **OPEN** | R4-03 | resource | Medium | `metrics/classification/_threshold_optimization.py:219,226`: full `(n_boot, n)` index matrix |
| **OPEN** | R4-04 | resource | Medium | `votenrank/rank_percentile_stacking.py:28,74,81`: dense `(n_query, n_oof)` matrices |
| **OPEN** | S5-02 | leakage | Medium | `training/composite/count_weighted_blend.py:115`: auto-k CV with shuffled row KFold |
| **OPEN** | S5-03 | leakage | Medium | `training/composite/hpo_ensembling.py:162`: `honest_score` ignores the purged time split |
| **OPEN** | M6-01 | numerics | Medium | `votenrank/rank_percentile_stacking.py:95-106`: out-of-range test values jump to 0 or 1 |
| **OPEN** | K7-01 | parity | Medium | `feature_selection/filters/_gpu_resident_shared.py:469-470,501-502,593-594`: GPU max/min do not propagate NaN |
| **OPEN** | C8-02 | state | Medium | `shap_proxied_fs/_shap_proxy_revalidate/_shap_proxy_loss.py:80-83,122-123`; `_shap_proxy_explain.py:83-84`: summary-hash cache keys |
| **OPEN** | C8-03 | state | Medium | `_numba_parallel_guard.py:42-45,63-81`: guard lock built through `lru_cache`; unguarded probe |
| **OPEN** | X9-02 | silent_failure | Medium | `models/ensembling/score_gate.py:306-310,408-412,285-292`: catastrophic-member drop check fails silently |
| **OPEN** | X9-03 | silent_failure | Medium | `training/_training_loop_objectives.py:136-159`: MultiOutputClassifier wrapping drops early stopping silently |
| **OPEN** | X9-04 | silent_failure | Medium | `inference/predict.py:228,247,250-251`: skipped models are logged at most 5 times per process |
| **OPEN** | X9-05 | silent_failure | Medium | `training/_training_loop_refit.py:167-192,336-352`: failed refit leaves the live model with the fallback loss |
| **OPEN** | P10-01 | api | Medium | `feature_selection/wrappers/rfecv/__init__.py:450`: `set_params` flat field then `clone()` raises |
| **OPEN** | P10-02 | api | Medium | `feature_selection/wrappers/rfecv/__init__.py:450`: `set_params(search_config=...)` is ignored |
| **OPEN** | D1-02 | deserialization | Low | `training/_trainer_train_and_evaluate_parts.py:132-133`: `trusted_root` defaults to the file's own directory |
| **OPEN** | D1-03 | deserialization | Low | `inference/predict.py`, `training/core/_metadata_loader.py`, `training/io.py`: hash-then-reopen race |
| **OPEN** | E2-01 | filesystem | Low | `training/feature_handling/hf_provider.py:118-122`: lock falls back to a predictable shared-temp name |
| **OPEN** | N3-01 | supply_chain | Low | `.pre-commit-config.yaml:126,200,459`: ruff-pre-commit pinned to a mutable tag |
| **OPEN** | N3-02 | supply_chain | Low | `feature_selection/filters/_neural_mi.py:278-291`: InfoNet checkpoint pin trusts the first file seen |
| **OPEN** | N3-03 | supply_chain | Low | `training/feature_handling/hf_provider.py:74-84`: unlisted model loads hub `main` without a revision |
| **OPEN** | R4-05 | resource | Low | `reporting/charts/slice_finder.py:386,392`: all pairs built before the cap |
| **OPEN** | S5-04 | leakage | Low | `training/composite/count_weighted_blend.py:139`: auto-k error ignores sample weights |
| **OPEN** | M6-02 | numerics | Low | `metrics/calibration/_calibration_metrics.py:78-83`: all-zero hits report perfect calibration |
| **OPEN** | M6-03 | numerics | Low | `core/stats.py:67-78`: Tukey multiplier at quantile 0.5 is `inf` |
| **OPEN** | M6-04 | numerics | Low | `core/proportion_stats.py:89-98`: impossible counts get an answer |
| **OPEN** | K7-02 | parity | Low | `_fe_batched_mi.py:115-117`, `_null_max_perm_gpu.py:39-42,91-93`: NaN bin differs between kernel and fallback |
| **OPEN** | C8-04 | state | Low | `_mrmr_fit_impl/_fit_impl_core.py:35` vs `_fit_impl_core_parts.py:26,232-233`: two different fit-cache locks |
| **OPEN** | C8-05 | state | Low | `_fe_resident_operands.py:132-149`: identity-keyed content memo |
| **OPEN** | C8-06 | state | Low | `info_theory/_cmi_cuda.py:95-121`: identity-keyed F-order copy cache |
| **OPEN** | C8-07 | state | Low | `_gpu_resident_rank_bin.py:80-87`: device-array cache without device id |
| **OPEN** | C8-08 | state | Low | `_neural_mi.py:94`, `_confirm_predictor.py:63`, `_orth_extra_basis_fe_helpers.py:389`: env vars read at import |
| **OPEN** | X9-06 | silent_failure | Low | `training/honest_diagnostics.py:592-594,249-251,278-280`: failed blocks become `skipped` at debug |
| **OPEN** | X9-07 | silent_failure | Low | `inference/predict.py:233-240`: failed version check is logged at debug |
| **OPEN** | X9-08 | silent_failure | Low | `training/honest_diagnostics.py:236-238`; `calibration/ensembling.py:161-163,181-183`: slow-path fallbacks logged at debug |
| **OPEN** | P10-03 | api | Low | `estimators/early_stopping.py:105-112`: documented default 5, actual 7, suite uses 20 |
| **OPEN** | P10-04 | api | Low | `training/strategies/base.py:127-136`: `_InfToNaNTransformer` skips polars frames |
| **OPEN** | P10-05 | api | Low | `training/strategies/base.py:184-196`: `_NumericOnlyTransformer.fit` fits `inner` in place |
| **OPEN** | P10-06 | api | Low | `estimators/custom.py:665-676`: `IdentityEstimator.fit` sets no `n_features_in_` |
| **OPEN** | P10-07 | api | Low | `docs/ENVIRONMENT_VARIABLES.md` vs code: five switches undocumented; `MLFRAME_FE_GPU_FUSE_MI` read nowhere |
| **OPEN** | R4-06 | resource | Info | `evaluation/blend_source_selection.py:85-99`: warns, then allocates two `n x n` matrices |
| **OPEN** | K7-03 | parity | Info | `_null_max_perm_gpu.py:160,186`: no shared-memory check; fallback logged at debug |

## Reports whose scope was incomplete

- A3 finished while its own background work was still running; the full-tree secret scan then completed with no hard-coded credentials (see `network_supply_chain.md`).
- Every report lists its "Not covered" areas; the largest unread parts are the roughly 2900 broad-except sites in `feature_selection/filters/*`, the other composite training modules using raw shuffled KFold, and calibration fitters.
