# mrmr_audit_2026-07-25 — master tracker

15-cluster parallel read-only audit (see `_BRIEF.md`) against git `d8091a138`. Every prior-wave
(`mrmr_audit_2026-07-22`) P0/P1 was re-verified: essentially all are **FIXED** in current source (each
cluster doc lists its prior-finding status). This wave surfaced **no P0**; the findings below are the current
actionable backlog. Per the user's directive, **every** finding — including P3/Low — is to be implemented,
each fix covered by a test and each perf change by a bench.

Coverage/quality baselines this wave: **0 zero-coverage modules**, mypy CLEAN (348 files), ruff CLEAN,
security CLEAN. One live gate failure: the 1k-LOC gate is RED (`evaluation.py` 1003, `_shap_proxied_fit.py` 1012).

Status legend: TODO / DONE (fixed+tested/benched) / DOC.

## P1 — real bugs / live gate failures (fix first)

| ID | File:Line | Summary | Fix | Status |
|----|-----------|---------|-----|--------|
| FIT_IMPL-1 | `_fit_impl_core.py:9944-9954` vs `9992-10042` | `mrmr_gains_` length-align runs BEFORE the group-aware final demotion, so a demotion leaves `len(mrmr_gains_) > n_features_ == len(get_feature_names_out())` | factor `_align_mrmr_gains(self)`, call as very last stmt before `return self` | DONE |
| DISCRETIZATION-2 | `_adaptive_nbins.py:690,258`; `_discretization_edges.py:217` | Bayesian-Blocks `bb_subsample_threshold` defaults 0 (disabled) → unbounded O(N²) DP on ordinary columns | default a bounded threshold (e.g. `min(N,5000)`) at the `per_feature_edges` dispatch layer; keep 0=exact for direct callers | DONE |
| GPU_INFRA-1 | `friend_graph_gpu.py:801,787-788` | `dispatch_friend_graph_stats` cuda branch catches only `(ValueError,RuntimeError)`; `force_backend="cuda"` path has no handling → `CudaAPIError` (Exception, not RuntimeError) escapes vs documented CPU fallback | broaden to `except Exception`, wrap force path | DONE |
| GPU_INFRA-2 | `batch_pair_usability_corr_gpu.py:501,528,540` | dispatcher never consults `gpu_globally_disabled()`; default backend now `cuda_warp` → `MLFRAME_DISABLE_GPU=1` silently defeated | add `if gpu_globally_disabled(): use_cuda=False` guard; +GPU_INFRA-3 stale caller comment | DONE |
| ORTH-1 | `_orthogonal_adaptive_arity_fe.py:720-771` | arity-2/3/4 recipe branches omit `preprocess_params_i/j/k/l` → `transform()` refits basis axis instead of replaying frozen params (leak/drift) | freeze + pass `preprocess_params_*` at all multi-leg recipe-build sites | DONE |
| ORTH-2 | test gap | no arity≥2 slice-replay parity test → ORTH-1 shipped silently | add arity 2/3/4 row-slice replay-parity test | DONE |
| FE_FAMILIES_B-1 | `_cat_mm_correction.py:256,278-332` | `_maybe_rerank_with_mm` double-corrects MM bias (MM entropies + telescoped subtraction); disagrees with `_compute_pair_ii_mm` → alters pair selection | pass `use_mm=False` in the MM branch (+factor shared MM-II primitive) | DONE |
| STABILITY_MISC-1 | `group_aware.py:403-406` | polars→pandas bridge `pd.DataFrame(X.to_numpy())` collapses numeric cols to object dtype → factorize destroys ordering → wrong clustering/support | bridge via `X.to_pandas()` | DONE |
| INFO_THEORY-1 / CROSS-1 | `evaluation.py` (1003 LOC) | over the enforced 1k-LOC gate, not exempt → `test_no_file_over_1k_loc` RED | carve a self-contained block into a re-exported sibling | DONE |
| CROSS-1b | `shap_proxied_fs/_shap_proxied_fit.py` (1012 LOC) | second breach of the same RED gate | carve into a sibling | DONE |

## P2 — bugs / correctness gaps / quality (fix + test/bench)

| ID | File:Line | Summary | Status |
|----|-----------|---------|--------|
| CORE_CLASS-1 | `_mrmr_config_dataclasses.py:124` | `HybridOrthScorersConfig.ensemble_scorers` default `()` ≠ flat 5-tuple → `HybridOrthConfig()` empties it | DONE |
| FE_STEP-1 | `_mrmr_fe_step_helpers.py:404` | `compute_pair_maxt_floor` O(k²) full-materialise, no chunking (~300MB @k5000) | DONE |
| FE_STEP-2 | `_mrmr_fe_step/_helpers.py:35` | `_non_numeric_column_indices` `except Exception: return set()` no logging | DONE |
| FE_PAIRS-1 | `_pairs_core.py:134-194` | unlocked `_GPU_GATE_CACHE` check-evict-write races under threading → KeyError/RuntimeError | DONE |
| FE_PAIRS-3 | `_pairs_core.py:134` | `_GPU_GATE_CACHE` keyed only `(n_rows,n_cands)`, ignores strict-mode/auto-n → stale backend choice | DONE |
| FE_PAIRS-4 | `_fe_matrix_io.py:251-253` | numpy round-trip returns float32, skips float64/null restore (gated-off P0 plane) | DONE |
| FE_PAIRS-5 | `_fe_cpu_batch.py:39` | dead `budget is None` disjunct (source always returns int) | DONE |
| INFO_THEORY-2 | `_cmi_cuda.py:498-500` | `clear_cmi_xc_resident_cache()` unlocked clear vs locked mutate | DONE |
| INFO_THEORY-3 | `_mah.py:118-120` | `clear_mah_y_binning_cache()` unlocked clear vs locked mutate | DONE |
| INFO_THEORY-4 | `_class_encoding.py:17` | `merge_vars` `min_occupancy` param accepted but never used | DONE |
| INFO_THEORY-5 | `_batch_kernels.py:85,92` | `MAX_JOINT_CARDINALITY` gate ignores `n_classes_y` factor in the alloc | DONE |
| INFO_THEORY-6 | `_batch_kernels.py:453,514,610,635` | `classes_dtype=int16` no clamp; nbins>32767 wraps dense code | DONE |
| INFO_THEORY-7 | `_class_mi_kernels.py` (+batch) | dead `dtype` param across ~9 functions | DONE |
| INFO_THEORY-8 | `_renyi_alpha.py:152-181` | `renyi_alpha_cmi` returns bits (unconverted) — latent nats/bits mismatch | DONE |
| INFO_THEORY-9 | `_fastmi.py:146,193` | `fastmi` MISE path hardcodes `default_rng(0)`, no seed param | DONE |
| INFO_THEORY-10 | `_pid_decomposition.py`, `_bur_term.py`, `_jmim_scorer.py`, `_relaxmrmr_3d.py` | dense joint alloc, no cardinality-product cap (OOM for direct callers) | DONE |
| DISCRETIZATION-1 | `discretization/__init__.py:486,531,1089,1211` | uniform NaN sentinel `n_bins` overflows at `nbins==dtype.max+1` (int8@128) | DONE |
| DISCRETIZATION-3 | `discretization/__init__.py:1102` | row-chunk VRAM margin hardcodes `+2` bytes, ignores widened output dtype | DONE |
| DISCRETIZATION-4 | `_discretization_dataset.py:495` | `np.append` full copy on mixed numeric+categorical merge | DONE (bench says the naive prealloc is a wash; the out=-slice win is deferred, noted at the call site) |
| DISCRETIZATION-5 | `supervised_binning.py:479-519` | `optimal_bin_edges` zero test coverage | DONE |
| GPU_RESIDENT-1 | `_permutation_null_pair_resident.py:84,194` | order-2 pair-maxT resident swallows device fault, breaker never trips | DONE |
| GPU_RESIDENT-2 | `_gpu_resident_extval.py:50-53,86` | `gpu_materialise_extval_codes_host` int8 no narrowing guard (B-4 class) | DONE |
| GPU_RESIDENT-4 | `_permutation_null_resident_ktc.py:115` | resident-vs-njit KTC sweep tol 5e-2 (docs ~1e-15) — far too loose | DONE |
| GPU_INFRA-6 | `batch_mi_noise_gate_gpu.py` (920 LOC) | in the ~800-900 carve zone (advisory, under 1k gate) | DOC |
| GPU_INFRA-7 | `_fe_gpu_batch/_executor.py:39,93` | `free_blocks=True` default forces per-batch cudaFree/Malloc | DOC (bench committed; default deliberately NOT flipped -- needs the multi-GPU VRAM-safety leg first) |
| GPU_INFRA-8 | `_gpu_hw_launch.py:36`, `_fe_gpu_vram.py:158` | unlocked `_DEV_PROPS`/`_POOL_LIMIT_DONE` singleton set (benign dup) | DONE |
| GPU_INFRA-9 | `_batch_mi_noise_gate_tuning.py`; usability opt-out | test gaps (tuning fns; usability-corr opt-out regression) | DONE |
| ORTH-3 | `_orthogonal_xi_fe.py:57-86` | Chatterjee Xi uses no-ties formula → biased on tied/discrete (classification) y | DONE |
| HERMITE-1 | `_hermite_fe_mi.py:40` | import-order fragility: first-touch of `_hermite_fe_mi` raises ImportError | DONE |
| FE_FAMILIES_A-1 | `_fe_additive_fusion.py:200`, `_fe_additive_fusion_gpu_resident.py:151` | `astype(int64)` before `np.unique` collapses continuous y | DONE |
| FE_FAMILIES_A-2 | `_integer_lattice_fe.py`, `_pairwise_modular_fe.py` | bare `astype(int64)` on y, no continuous-y guard | DONE |
| FE_FAMILIES_B-3 | `_cat_mm_correction.py:201` | no test pins `_maybe_rerank_with_mm` == `_compute_pair_ii_mm` (drives B-1) | DONE |
| FE_FAMILIES_B-4 | `_cat_target_encoding_and_weighted.py:72` | multi-class nominal TE semantics untested | DONE |
| SCREEN_CONFIRM-1 | `_mrmr_sis_screen.py:257-261` | SIS quantile-bins high-card integer NOMINAL classification target | DONE |
| SCREEN_CONFIRM-2 | `_fe_raw_redundancy_drop.py:650-655` | group-aware leak-exempt can't tell 0.0-leak from 0.0-no-group-cleared-min_rows | DONE |
| SCREEN_CONFIRM-5 | `_confirm_predictor.py:64,168-285` | `_EVALUATE_CANDIDATES_POOL_ENABLED=False` dead parallel-scoring path | DONE |
| STABILITY_MISC-2 | `_mrmr_tree_rescue.py:132-137` | `_apply_tree_rescue` truncates top-k BEFORE filtering by `factors_to_use` → under-adds | DONE |

## P3 / Low — hygiene, dead code, docs, comment sweep

| ID | Summary | Status |
|----|---------|--------|
| CROSS-2 | 273 files / 2254 lines of leftover audit-metadata comments (date stamps / Wave N / loop iter N / finding-IDs / `.md finding` / stale `suppressed in <file>:<line>`) → cleanup sweep | PARTIAL (see _VERIFICATION.md: markers + all 185 stale citations gone; ~840 prose date mentions need human reading, not a regex) |
| CROSS-3 | `_confirm_predictor_context.py:97-104` implicit-Optional `list=None`/`dict=None` + `# type: ignore` → `Optional[...]` | DONE |
| CROSS-4 | `--` in prose comments/docstrings (5141 lines; conservative sweep of worst offenders, keep CLI `--foo`/`i--`) | PARTIAL (~80% of the audited tree; the rest is outside the space-delimited rule) |
| CORE_CLASS-2 | `set_params(nested_config=...)` silently discarded → expand config fields onto flats (GridSearchCV-over-config) | DONE |
| CORE_CLASS-3, FIT_IMPL-2, FE_STEP-3/4, FE_PAIRS-2, INFO_THEORY comments, DISCRETIZATION-6/7, GPU_INFRA-4/5, GPU_RESIDENT-3, ORTH-4/5, HERMITE-2/3/4/5, FE_FAMILIES_A-3, FE_FAMILIES_B-2, SCREEN_CONFIRM-3/4, STABILITY_MISC-3 | per-cluster comment-metadata + stale-line-citation cleanup (subsumed by CROSS-2 sweep) + dead-store (HERMITE-4), crashing docstring example (HERMITE-3), stale docstrings (HERMITE-5, DISCRETIZATION-8 dead `quantize_dig`) | DONE |
| FE_STEP-6 | `_FE_FAMILY_WALL` cross-fit attribution under threading (diagnostic-only) | DONE |
| FE_STEP-7 | direct unit tests for 5 FE-step sub-blocks | PARTIAL (helpers now covered; 4 of the 5 named blocks still lack a direct test) |
| FE_FAMILIES_B-5 | LOC carve advisory (`_extra_fe_families.py` 906 etc., under 1k gate) | DOC |

## Meta-test proposals raised by agents (feed the final meta-test phase)
- config-default parity meta-test: every `(config_field → flat_attr)` default must equal `MRMR()._ctor_defaults()[flat_attr]` (CORE_CLASS-1).
- grep meta-test: no `orth_*_cross` recipe build omits `preprocess_params_*` (ORTH-1 recurred twice).
- grep meta-test: no leftover audit-metadata comments (CROSS-2) — the sweep's guard.
- grep meta-test: no stale `"suppressed in <file>:<line>"` self-citations.
- gains/support/get_feature_names_out length-alignment invariant test under group_aware_mi (FIT_IMPL-1).
