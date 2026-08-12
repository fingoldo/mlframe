# feature_engineering/transformer C (row-attention/kernel family, more SMOTE/baseline variants, conformal) -- mlframe audit

## Scope

All 40 assigned files were read in full (no file was too large to review in depth; the largest, `row_attention.py`, is 388 LOC).

- `src/mlframe/feature_engineering/transformer/row_attention.py` (388 LOC)
- `src/mlframe/feature_engineering/transformer/_kernels_cupy.py` (371 LOC)
- `src/mlframe/feature_engineering/transformer/_key_bank.py` (284 LOC)
- `src/mlframe/feature_engineering/transformer/_kernels_njit.py` (243 LOC)
- `src/mlframe/feature_engineering/transformer/residual_stratified_distance.py` (230 LOC)
- `src/mlframe/feature_engineering/transformer/multi_temp_cbhr.py` (222 LOC)
- `src/mlframe/feature_engineering/transformer/active_virtual.py` (216 LOC)
- `src/mlframe/feature_engineering/transformer/band_conditional_anchor.py` (210 LOC)
- `src/mlframe/feature_engineering/transformer/class_distance.py` (200 LOC)
- `src/mlframe/feature_engineering/transformer/_knn_helper.py` (198 LOC)
- `src/mlframe/feature_engineering/transformer/bgmm_quantile_bands.py` (192 LOC)
- `src/mlframe/feature_engineering/transformer/multi_aux_ensemble.py` (190 LOC)
- `src/mlframe/feature_engineering/transformer/nn_oof_target_mean.py` (188 LOC)
- `src/mlframe/feature_engineering/transformer/bidir_residual_band.py` (183 LOC)
- `src/mlframe/feature_engineering/transformer/bgmm_multiscale.py` (181 LOC)
- `src/mlframe/feature_engineering/transformer/baseline_disagreement_smote.py` (178 LOC)
- `src/mlframe/feature_engineering/transformer/density_weighted_smote.py` (177 LOC)
- `src/mlframe/feature_engineering/transformer/_suite_adapter.py` (172 LOC)
- `src/mlframe/feature_engineering/transformer/denoising_autoencoder.py` (171 LOC)
- `src/mlframe/feature_engineering/transformer/local_lift.py` (169 LOC)
- `src/mlframe/feature_engineering/transformer/smote_distance.py` (168 LOC)
- `src/mlframe/feature_engineering/transformer/pure_pos_smote.py` (163 LOC)
- `src/mlframe/feature_engineering/transformer/multi_temp_band_attention.py` (159 LOC)
- `src/mlframe/feature_engineering/transformer/baseline_disagreement.py` (157 LOC)
- `src/mlframe/feature_engineering/transformer/focal_lgb.py` (148 LOC)
- `src/mlframe/feature_engineering/transformer/lda_projection.py` (148 LOC)
- `src/mlframe/feature_engineering/transformer/baseline_surprise.py` (145 LOC)
- `src/mlframe/feature_engineering/transformer/quantile_band_attention.py` (145 LOC)
- `src/mlframe/feature_engineering/transformer/baseline_disagreement_balanced.py` (139 LOC)
- `src/mlframe/feature_engineering/transformer/decision_region_depth.py` (139 LOC)
- `src/mlframe/feature_engineering/transformer/boosting_leaf.py` (135 LOC)
- `src/mlframe/feature_engineering/transformer/stacked_attention.py` (132 LOC)
- `src/mlframe/feature_engineering/transformer/anomaly_score_features.py` (127 LOC)
- `src/mlframe/feature_engineering/transformer/local_intrinsic_dim.py` (127 LOC)
- `src/mlframe/feature_engineering/transformer/sign_residual_baseline.py` (118 LOC)
- `src/mlframe/feature_engineering/transformer/conformal_coverage_failure.py` (106 LOC)
- `src/mlframe/feature_engineering/transformer/conformal_locally_adaptive.py` (105 LOC)
- `src/mlframe/feature_engineering/transformer/distributional_moments.py` (95 LOC)
- `src/mlframe/feature_engineering/transformer/cross_feature_reconstruction.py` (86 LOC)
- `src/mlframe/feature_engineering/transformer/_residual_oof.py` (62 LOC)

Total files reviewed: 40. Total LOC reviewed: 6967 (4363 + 2604 from the two `wc -l` batches).

I also read `tests/feature_engineering/transformer/test_sign_residual_baseline_score_bug.py` (to confirm a previously-fixed bug is not a live finding) and ran a directory listing of `tests/feature_engineering/transformer/` to assess test coverage per file (see Proposals). Files outside this cluster (`_utils.py`, `_oof.py`, `_projection.py`, `_row_attention_ann.py`, `_aggregation.py`, `_intel_patch.py`, `swap_noise.py`) were only consulted for imported-symbol signatures/docstrings, per the task's "read excluded imports for context only" allowance — no findings are reported against them.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P0 | correctness / caching | `_key_bank.py:95-137`, `row_attention.py:215-223` | `KeyBank` disk-cache fingerprint omits the `projection` (and `dtype`) build parameter, so switching `projection=` with the same `cache_dir` silently serves a stale key-bank built under a different projection method. |
| F2 | P1 | correctness / edge-case | `quantile_band_attention.py:100-104` | An empty y-quantile/class band leaves its centroid at the standardized-space origin and `y_mean`/`y_std` at 0.0, so a phantom "band" competes in the softmax and corrupts `agg_y_mean`/`agg_y_std`/`best_band` for queries near the data center. |
| F3 | P1 | correctness / edge-case | `band_conditional_anchor.py:110-125` | Same empty-band-at-origin issue as F2, plus the empty band's dummy anchor slots keep `anchor_parent_band` at its zero-initialized default, mislabeling them as belonging to band 0 and inflating `band_masses`/`argmax_band` for band 0. |
| F4 | P1 | correctness / edge-case | `multi_temp_band_attention.py:104-108` | Same empty-band-at-origin issue as F2, repeated across all 3 temperatures. |
| F5 | P1 | correctness / edge-case | `bidir_residual_band.py:135-140` | Same empty-band-at-origin issue as F2 (band membership by `|residual|` quintile instead of y-quantile). |
| F6 | P1 | correctness / edge-case | `multi_temp_cbhr.py:137-148` | When a fold's positive (or negative/top-quintile) side has zero rows, the anchor-padding fallback sets all of that side's anchor indices to `0`, silently reusing an arbitrary (likely wrong-class) row as every anchor instead of a neutral/sentinel value. |
| F7 | P1 | robustness / API contract | `distributional_moments.py:18,42-46,61-62` | The public `quantiles` parameter is never validated against the hardcoded 7-element indexing (`preds_q[:, 0]`...`preds_q[:, 6]`) used to derive skew/kurtosis/tail features; passing a `quantiles` sequence of a different length crashes (regression path, `IndexError`) or silently zero-pads and misaligns predictions (binary path, since `gammas[:len(quantiles)]` truncates to `gammas`'s own 7 entries). |
| F8 | P1 | robustness / edge-case | `conformal_coverage_failure.py:51-55` | No guard against a tiny fold: `h1, h2 = idx[:n // 2], idx[n // 2:]` can leave `h1` empty, crashing the subsequent `lgb.fit` on 0 rows. The near-identical sibling `conformal_locally_adaptive.py:58-62` already has an explicit "Wave 39" `n < 4` guard for this exact failure mode; it was not applied here. |
| F9 | P1 | robustness / API contract | `_residual_oof.py:32-34` | `compute_oof_yhat_within` forces `n_splits = 2` whenever the subset has fewer than 2 rows, contradicting its own docstring ("`aux_n_splits` is capped to the subset size so tiny subsets still partition cleanly"); `KFold(n_splits=2)` on 0 or 1 rows raises `ValueError`. |
| F10 | P2 | ML best practice / consistency | `bidir_residual_band.py:48-68`, `multi_temp_cbhr.py:43-63`, `baseline_surprise.py:43-59`, `decision_region_depth.py:67-75`, `sign_residual_baseline.py:69-77` | Band/anchor assignment and "surprise" scores are derived from an IN-SAMPLE baseline (fit and scored on the exact same train rows), unlike the near-identical sibling `residual_stratified_distance.py:74-104`, which explicitly computes honest OOF residuals via an inner `KFold(3)` for the same purpose. No cross-fold leakage results (the outer OOF discipline still holds), but the in-sample bias systematically distorts which rows look "easy"/"hard" or how confident the model looks. |
| F11 | P2 | docs / comment accuracy | `local_lift.py:58` | Stale comment "We use trapezoidal integration" directly contradicts the step-sum Average-Precision implementation on the next 8 lines and the correct docstring at line 64. |
| F12 | P2 | architecture / consistency | `boosting_leaf.py:37-51,99-106` | Unlike every other transformer in this cluster, `compute_boosting_leaf_features` has no `splitter` parameter; it always builds its own internal `KFold(shuffle=True, random_state=seed)` for Mode A, so its OOF fold boundaries can't be aligned with an orchestrating pipeline's outer fold structure. |
| F13 | P2 | perf / feature gap | `stacked_attention.py:91,102` | `compute_stacked_row_attention` hardcodes `gpu_stage4=False` for every layer with no caller-facing way to opt into GPU acceleration, silently forcing CPU even when `compute_row_attention`'s own `gpu_stage4="auto"` would pick GPU; not mentioned in the function's docstring. |
| F14 | P2 | code quality | `row_attention.py:315` | `attend()` identifies the cupy stage-4 backend via `stage4_callable.__name__ == "row_attention_stage4_cupy"` string comparison instead of an identity (`is`) check against the imported symbol; fragile if the callable is ever wrapped (e.g. `functools.wraps` under a different name) or renamed. |
| F15 | P2 | perf | `anomaly_score_features.py:58` | `_fit_anomaly_predict` rescoring the full training set (`iso1.score_samples(Xt)` and `iso2.score_samples(Xt)`) purely to compute `global_mean_train` adds a second full-training-set scoring pass per model per fold that duplicates work already implicitly available. |

### Finding details

**F1 (P0).** `_key_bank_fingerprint` (`_key_bank.py:95-104`) hashes `X_train`, `seed`, `n_heads`, `head_dim`, `metric`, `standardize`, `ann_M`, `ann_ef_construction` — but not `projection` (which selects among `"random"`/`"pls"`/`"importance"`/`"shap"`/`"nca"`, each producing structurally different `projections`/`k_proj` arrays) nor `dtype`. `build_key_bank` (`row_attention.py:215-223`) computes the fingerprint from exactly those un-hashed-`projection` inputs before checking `try_load_key_bank`. A caller who builds a cache under `projection="random"` and later calls with the same `cache_dir`/`X_train`/`seed`/etc but `projection="pls"` gets back the `"random"`-built bank with no error, warning, or shape mismatch (shapes match since `head_dim`/`n_heads` are unchanged) — every downstream `attend()` call silently scores queries against the wrong projection space. Fix direction: add `projection` (and ideally `dtype`) to the fingerprint hash inputs, or store them in `metadata.pkl` and verify on load.

**F2-F5 (P1, shared root cause).** In each of `quantile_band_attention.py`, `band_conditional_anchor.py`, `multi_temp_band_attention.py`, and `bidir_residual_band.py`, per-band statistics arrays (`band_centroids`, `band_y_mean`, `band_y_std`, and in `band_conditional_anchor.py` also `all_anchors`/`anchor_parent_band`) are pre-allocated with `np.zeros(...)` and then populated inside a loop that does `if X_band.shape[0] < 1: continue` for an empty band. Because `Xt_s` is `RobustScaler`-standardized (median-centered), the origin is not a neutral "far away" point — it sits near the bulk of the data. An empty band (plausible whenever a fold's positive/rare class or an extreme y-quantile has zero rows in a small or non-stratified fold — a scenario this same codebase's own docstrings repeatedly cite, e.g. the "mammography 1.3% positive" example, combined with several Mode-A callers using plain `KFold` rather than `StratifiedKFold`) creates a phantom anchor at the data center with `y_mean=0`/`y_std=0` that competes for softmax attention weight against real bands, silently pulling `agg_y_mean`/`agg_y_std`/`band_masses`/`argmax_band` toward zero for any query near that center. Fix direction: skip empty bands from the softmax entirely (e.g. mask their score to `-inf` before `_softmax`), or fall back to the global centroid/mean the way other mechanisms in this cluster (e.g. `bgmm_quantile_bands.py`, which returns an explicit `1e6` sentinel distance for an empty band) already do correctly.

**F6 (P1).** `multi_temp_cbhr.py:137-148`: when `pos_top.size == 0` (i.e. zero rows in `pos_mask_idx` for this fold), the code sets `pos_top = np.zeros(n_hard_per_side, dtype=np.int64)`, i.e. index `0` repeated. Index `0` is an arbitrary row of `Xt` — almost certainly from the *other* class/side — so every "positive-side" anchor for that fold silently becomes a copy of a mislabeled row's features/y/residual, rather than something neutral. Fix direction: when a side is entirely empty, either skip that side's weights/aggregates (mask to zero contribution) or raise/log rather than silently substituting a wrong-class row.

**F7 (P1).** `distributional_moments.py` accepts `quantiles: Sequence[float] = _QUANTILES` as a public keyword parameter, but `_process` hardcodes `preds_q[:, 0]`, `[:, 2]`, `[:, 3]`, `[:, 4]`, `[:, 5]`, `[:, 6]` (`distributional_moments.py:61-62`) assuming exactly the 7-entry default ordering `(0.05, 0.15, 0.25, 0.5, 0.75, 0.85, 0.95)`. A caller supplying `quantiles` of a different length crashes on the regression path (`IndexError`) once `len(quantiles) < 7`; on the binary path, `gammas[: len(quantiles)]` (line 46) is silently truncated to `gammas`'s own 7 entries for `len(quantiles) > 7`, leaving the extra `preds_q` columns at their `np.zeros` initial value, then sorted in among the real predictions and misinterpreted as quantile predictions. Fix direction: validate `len(quantiles) == 7` (and ideally the specific values) at the top of the function, or compute q05/q25/q50/q75/q85/q95 by looking up the nearest supplied quantile value rather than by fixed position.

**F8 (P1).** `conformal_coverage_failure.py:51-55` splits the (possibly per-fold) train array `idx[:n//2]`/`idx[n//2:]` with no minimum-size guard. At `n<2`, `h1` is empty and `lgb.fit(Xt_s[h1], y_t[h1]...)` raises on an empty training array. The sibling file `conformal_locally_adaptive.py:58-62` documents and fixes exactly this failure mode ("Wave 39... tiny-train regime (n<4) empties h1... Return a zero-feature block") but the fix was never mirrored into `conformal_coverage_failure.py`, which implements the identical half-split pattern. Fix direction: port the same `n < 4` early-return guard.

**F9 (P1).** `_residual_oof.py:32-34`: `n_splits = min(aux_n_splits, n) if n >= 2 else 2; n_splits = max(2, n_splits)`. When `n` (the subset row count) is 0 or 1, this forces `n_splits=2` regardless, and `KFold(n_splits=2).split(X_sub)` on fewer than 2 rows raises `ValueError`. The docstring explicitly claims `aux_n_splits` "is capped to the subset size so tiny subsets still partition cleanly," which is false for `n<2`. Fix direction: guard `n<2` explicitly (e.g. return an all-zero / all-constant OOF vector, matching the pattern used elsewhere in this cluster for undersized subsets).

## Proposals

| ID | Category | File(s) | Summary |
|----|----------|---------|---------|
| PR1 | test-coverage | ~27 of the 40 files | The large majority of the "iterNN mechanism" files in this cluster (`band_conditional_anchor.py`, `quantile_band_attention.py`, `multi_temp_band_attention.py`, `bidir_residual_band.py`, `multi_temp_cbhr.py`, `bgmm_quantile_bands.py`, `bgmm_multiscale.py`, `active_virtual.py`, `density_weighted_smote.py`, `pure_pos_smote.py`, `smote_distance.py`, `baseline_disagreement*.py` (4 variants), `focal_lgb.py`, `lda_projection.py`, `nn_oof_target_mean.py`, `decision_region_depth.py`, `boosting_leaf.py`, `stacked_attention.py`, `anomaly_score_features.py`, `distributional_moments.py`, `cross_feature_reconstruction.py`, `conformal_locally_adaptive.py`, `multi_aux_ensemble.py`, `baseline_surprise.py`, `_residual_oof.py`) have **no** dedicated unit test or `test_biz_val_*` file under `tests/feature_engineering/transformer/`, contrary to this repo's own stated convention that every ML trick ships a quantitative `biz_value` test. Only `row_attention.py` (extensively tested), `class_distance.py`/`local_lift.py` (`test_biz_val_class_distance_and_local_lift.py`), `denoising_autoencoder.py` (`test_biz_val_denoising_autoencoder.py`), `conformal_coverage_failure.py` (leakage test), `sign_residual_baseline.py` (a targeted regression test), `local_intrinsic_dim.py` (an identity test), and `_knn_helper.py` (`test_knn_helper.py`) have coverage. This also means none of the F2-F9 findings above have a regression test surfacing them today. |
| PR2 | test-coverage | F2-F6 files | Add a fuzz/edge-case test that constructs a Mode-A fold with a fully-empty positive class or extreme y-quantile band (e.g. `n_splits` deliberately too large relative to a rare-positive `y`, or an unstratified `KFold` on a 1%-positive `y`) and asserts the resulting features are finite and not silently dominated by an origin/zero-labeled phantom band — this is the natural regression test for F2-F6. |
| PR3 | refactor | `boosting_leaf.py` | Accept an optional `splitter` parameter (falling back to its current internal `KFold` when omitted) for consistency with every other Mode-A/B transformer in this cluster and to let callers align its OOF folds with an outer pipeline's folds. |
| PR4 | ML best practice | F10 files | Consider switching the in-sample baseline used for band/anchor assignment (`bidir_residual_band.py`, `multi_temp_cbhr.py`, `baseline_surprise.py`, `decision_region_depth.py`, `sign_residual_baseline.py`) to the already-implemented honest-OOF pattern in `residual_stratified_distance.py` (`_compute_oof_residuals`, inner `KFold(3)`), and measure whether the reduced in-sample optimism bias improves the "hard region" signal quality on the same real-dataset biz-val benchmarks the rest of the suite uses. |
| PR5 | perf | `_kernels_njit.py` | `row_attention_stage4_adaptive_njit` (adaptive per-query softmax temperature variant) has no caller anywhere in the 40 reviewed files or their direct dependents — confirm it's still on a documented roadmap / exposed via a public API elsewhere, or remove it as dead code. |
| PR6 | docs | `row_attention.py` module docstring | Document the `gpu_stage4=False` hardcoding in `stacked_attention.py` (or expose it as a passthrough parameter) so a GPU-equipped caller isn't surprised that stacked attention never uses the GPU stage-4 kernel `compute_row_attention` otherwise picks by default. |

## Coverage notes

- All 40 assigned files were read in full; none were too large to review completely.
- Files imported by this cluster but excluded from the audit scope (`_utils.py`, `_oof.py`, `_projection.py`, `_row_attention_ann.py`, `_aggregation.py`, `_intel_patch.py`, `swap_noise.py`) were referenced only for the imported symbol's signature/docstring as permitted by the task instructions; no line-level review or findings were produced against them, so any bugs living purely inside those modules (e.g. inside `require_seed`, `is_gpu_available`, or the `_oof.kfold_attention_loop` OOF-loop implementation itself) are out of scope of this report and not covered here.
- I did not execute any test or benchmark (read-only audit per instructions), so the F1-F9 findings above are static-analysis conclusions verified by re-reading the exact code paths, not confirmed via a failing/passing pytest run.
- Test-coverage claims in PR1 are based on a filename listing of `tests/feature_engineering/transformer/` and a few adjacent test directories (`tests/training/strategies/`, `tests/training/`); a file could in principle be covered indirectly through a broader integration/suite-level test (e.g. `tests/training/test_bizvalue_preproc_transformers.py`) that I did not open line-by-line to confirm per-mechanism assertions — the PR1 list should be read as "no dedicated test file found by name," not as a guarantee of zero indirect coverage.
