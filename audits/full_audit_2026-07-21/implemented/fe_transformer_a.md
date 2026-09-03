# feature_engineering/transformer A (SMOTE/oversampling family, attention family, misc) -- mlframe audit

## Scope

All 40 files were read in full (no partial/skipped reads):

- src/mlframe/feature_engineering/transformer/random_features.py (475 LOC)
- src/mlframe/feature_engineering/transformer/_utils.py (291 LOC)
- src/mlframe/feature_engineering/transformer/__init__.py (264 LOC)
- src/mlframe/feature_engineering/transformer/_oof.py (250 LOC)
- src/mlframe/feature_engineering/transformer/anchor_attention.py (232 LOC)
- src/mlframe/feature_engineering/transformer/_row_attention_ann.py (224 LOC)
- src/mlframe/feature_engineering/transformer/pseudo_smote.py (219 LOC)
- src/mlframe/feature_engineering/transformer/bgm_clustered_smote.py (208 LOC)
- src/mlframe/feature_engineering/transformer/borderline_smote.py (203 LOC)
- src/mlframe/feature_engineering/transformer/inducing_attention.py (195 LOC)
- src/mlframe/feature_engineering/transformer/residual_band_attention.py (193 LOC)
- src/mlframe/feature_engineering/transformer/disagreement_band.py (192 LOC)
- src/mlframe/feature_engineering/transformer/hard_row_attention.py (184 LOC)
- src/mlframe/feature_engineering/transformer/multi_temp_residual_band.py (184 LOC)
- src/mlframe/feature_engineering/transformer/bgmm_dual_class.py (182 LOC)
- src/mlframe/feature_engineering/transformer/adasyn_smote.py (180 LOC)
- src/mlframe/feature_engineering/transformer/signed_residual_band.py (177 LOC)
- src/mlframe/feature_engineering/transformer/baseline_disagreement_v2.py (172 LOC)
- src/mlframe/feature_engineering/transformer/multiscale_smote.py (171 LOC)
- src/mlframe/feature_engineering/transformer/adversarial_flip.py (168 LOC)
- src/mlframe/feature_engineering/transformer/local_linear.py (165 LOC)
- src/mlframe/feature_engineering/transformer/quantile_neighbours.py (164 LOC)
- src/mlframe/feature_engineering/transformer/density_ratio.py (162 LOC)
- src/mlframe/feature_engineering/transformer/gradient_direction_agreement.py (152 LOC)
- src/mlframe/feature_engineering/transformer/mixup_boundary.py (152 LOC)
- src/mlframe/feature_engineering/transformer/counterfactual_substitution.py (147 LOC)
- src/mlframe/feature_engineering/transformer/local_curvature.py (146 LOC)
- src/mlframe/feature_engineering/transformer/apriori_itemsets.py (144 LOC)
- src/mlframe/feature_engineering/transformer/performer_attention.py (142 LOC)
- src/mlframe/feature_engineering/transformer/residual_attention.py (137 LOC)
- src/mlframe/feature_engineering/transformer/geodesic_kgraph.py (133 LOC)
- src/mlframe/feature_engineering/transformer/boosted_attention.py (132 LOC)
- src/mlframe/feature_engineering/transformer/per_class_spectral.py (129 LOC)
- src/mlframe/feature_engineering/transformer/multiscale_rate.py (123 LOC)
- src/mlframe/feature_engineering/transformer/variance_baseline.py (121 LOC)
- src/mlframe/feature_engineering/transformer/multi_threshold_ordinal.py (110 LOC)
- src/mlframe/feature_engineering/transformer/target_kmeans_codebook.py (103 LOC)
- src/mlframe/feature_engineering/transformer/jackknife_endpoint_stability.py (94 LOC)
- src/mlframe/feature_engineering/transformer/multi_temperature.py (83 LOC)
- src/mlframe/feature_engineering/transformer/_intel_patch.py (62 LOC)

Total files reviewed: 40. Total LOC reviewed: 6965 (sum of `wc -l` on the list above).

No file in this cluster is anywhere near the ~900-1000 LOC split threshold (largest is random_features.py at 475 LOC), so the "monolith split" checklist item does not apply here.

To ground the duplication/reproducibility findings below, I also grepped (read-only, not counted in the reviewed-LOC total) sibling files outside this cluster that share helper names with in-scope files: `_row_attention_ann.build_hnsw_index` call sites package-wide, and `_kth_nearest_dists` definitions package-wide. I also listed (but did not open) `tests/feature_engineering/transformer/` to assess test coverage for Proposals.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P1 | reproducibility | src/mlframe/feature_engineering/transformer/_oof.py:161 ; src/mlframe/feature_engineering/transformer/local_linear.py:94 | `build_hnsw_index` is called without `random_state`, so the ANN graph always uses the hardcoded default (42) regardless of the caller's `seed`. |
| F2 | P1 | correctness / edge-case | src/mlframe/feature_engineering/transformer/anchor_attention.py:191,222 | Mode B uses `np.nanargmin` (an explicit prior fix for NaN-poisoned distances bucketing to anchor 0); Mode A's OOF loop still uses plain `np.argmin`, reintroducing the same bug class in the more commonly-exercised path. |
| F3 | P1 | correctness / edge-case | src/mlframe/feature_engineering/transformer/residual_band_attention.py:149 ; disagreement_band.py:143 ; multi_temp_residual_band.py:128 ; signed_residual_band.py:132 | When a quantile band ends up empty (tied residual/disagreement values collapse a band boundary), its centroid/y_mean/y_std silently stay at `0.0` instead of falling back to a global default; a query row that softly attends to that phantom band gets a spurious near-zero contribution instead of an error or a documented neutral value. |
| F4 | P1 | correctness / edge-case | src/mlframe/feature_engineering/transformer/borderline_smote.py:58-62 | `_find_borderline_positives` comments that it drops the first neighbour "if distance is ~0" but the code unconditionally drops column 0 without checking; with duplicate rows in `X_full` (common in tabular data) the true self-match need not sort first, so a real neighbour gets excluded instead of self, and self can leak into the neighbour set that decides "borderline" classification. |
| F5 | P1 | correctness / edge-case | src/mlframe/feature_engineering/transformer/geodesic_kgraph.py:93-94 | When `target_indices` is empty (no positive-class / no bottom-quintile rows survive a fold split -- realistic for the rare-positive datasets this package's docstrings target), the fallback sets every train row's geodesic distance to `0.0` (i.e. "very close to target"), the opposite direction of the `1e6`-sentinel-for-"very far" convention used consistently elsewhere in the same package for the same "empty subset" situation. |
| F6 | P1 | robustness / crash risk | src/mlframe/feature_engineering/transformer/jackknife_endpoint_stability.py:53-59 | For `task="binary"`, the per-jackknife-subsample `m_dn`/`m_up` LGBMClassifier fits are not guarded against an all-one-class subsample; with a 5% row drop applied `n_subsamples=10` times to a rare-positive fold (the exact scenario the surrounding package repeatedly cites, e.g. "mammography, 52 positives"), a subsample can plausibly lose every positive row and the classifier fit raises. Sibling files in this same cluster (e.g. `multi_threshold_ordinal.py`, `disagreement_band.py`) guard the equivalent single-class case explicitly. |
| F7 | P1 | robustness / API hygiene | src/mlframe/feature_engineering/transformer/gradient_direction_agreement.py:104-105,53-67 | `_process` sets `Xq_s = Xq` (no copy) when `standardize=False`; if the caller's `X_query` was already `float32`, `Xq_s` is the identical array object handed to `X_query`. `_gradient` then perturbs `Xq_s` columns in place and restores them, but with no `try/finally` -- an exception raised by `model.predict`/`predict_proba` mid-loop (e.g. from `stack_elems` fallback edge cases, degenerate model state) leaves the caller's own `X_query` array permanently mutated with no indication. The restore-on-success path is unit-tested (`test_graddir_gradient_colcopy_identity.py`); the aliasing-plus-exception case is not. |
| P8 (P2) | P2 | silent-failure | src/mlframe/feature_engineering/transformer/local_curvature.py:116-118 | The per-row `except Exception: out[q] = 0.0` swallows every numerical failure (not just the documented singular-matrix case) with no logging at all, unlike the structurally identical per-row fallback in `local_linear.py` which logs via `logger.info(...)` on the same kind of failure. |
| P9 (P2) | P2 | silent-failure | src/mlframe/feature_engineering/transformer/apriori_itemsets.py:82-85 | `except Exception: frequent = pd.DataFrame()` swallows any `fpgrowth` failure (mlxtend version mismatch, malformed input) with no logging; the caller only ever observes an all-zero-feature fallback and can't tell "no frequent itemsets found" from "fpgrowth crashed". |
| P10 (P2) | P2 | silent-failure | src/mlframe/feature_engineering/transformer/disagreement_band.py:65-70 ; baseline_disagreement_v2.py:69-74 | `except Exception: preds[:, 2] = ...` (LogisticRegression fit failure) is unlogged, unlike the equivalent fallback in `bgm_clustered_smote.py`/`bgmm_dual_class.py` which log via `logger.info(...)`. |
| P11 (P2) | P2 | perf | src/mlframe/feature_engineering/transformer/local_curvature.py:80-118 | Per-query-row Python loop (`for q in range(n_q): ... np.linalg.lstsq(...)`) instead of a batched solve; the sibling `local_linear.py` demonstrates the exact same shape of computation (per-row local regression on kNN neighbourhoods) batched via `einsum` + `np.linalg.solve` across all rows at once. |
| P12 (P2) | P2 | perf | src/mlframe/feature_engineering/transformer/residual_band_attention.py:155-156 ; disagreement_band.py:150-151 ; multi_temp_residual_band.py:134-135 ; signed_residual_band.py:138-139 ; hard_row_attention.py:142-143 | Query-to-band/anchor distance is computed via the `Xq_s[:, None, :] - centroids[None, :, :]` broadcast cube, materialising an `(n_q, n_bands_or_anchors, d)` temporary; the sibling `anchor_attention.py`/`inducing_attention.py` in the same cluster explicitly avoid this exact pattern via a `||a||^2 - 2a.b + ||b||^2` GEMM decomposition (with an in-code comment explaining why), so the band-attention sub-family is the odd one out. |
| P13 (P2) | P2 | docs | src/mlframe/feature_engineering/transformer/_utils.py:250-254,257-263 | `sigma_median_heuristic`'s docstring says "Always use the block-wise pairwise reduction" to bound memory, but `_median_pairwise_chunked` just calls `scipy.spatial.distance.pdist` directly and documents that its own `chunk` parameter is "accepted for back-compat but ignored" -- the two docstrings describe an implementation that no longer exists. |
| P14 (P2) | P2 | architecture / consistency | src/mlframe/feature_engineering/transformer/apriori_itemsets.py:17-20 | The only file in this cluster whose public function signature has no type hints at all (`X_train, y_train, X_query=None, splitter=None, *, seed, task="regression", ...`), breaking the mypy-clean convention every other file in the cluster follows (same pattern in `multi_threshold_ordinal.py` and `target_kmeans_codebook.py`, listed together since they share the same terse un-annotated style). |

Note: no P0 was found in this cluster -- no bug produced silently-wrong output or a crash on the *common* input path; every finding above requires a specific edge condition (tied quantiles, duplicate rows, an already-fixed bug being re-triggered only via one of two code paths, an empty class subset, an exception mid-computation).

## Proposals

| ID | Category | File(s) | Summary |
|----|----------|---------|---------|
| PR1 | test-coverage | tests/feature_engineering/transformer/ (whole cluster) | Of the 40 in-scope files, the vast majority (all SMOTE variants, the whole residual/disagreement-band family, hard_row_attention, inducing_attention, performer_attention, boosted_attention, geodesic_kgraph, per_class_spectral, multiscale_rate, variance_baseline, multi_threshold_ordinal, target_kmeans_codebook, jackknife_endpoint_stability, apriori_itemsets, counterfactual_substitution, adversarial_flip, mixup_boundary, gradient_direction_agreement) have no `test_biz_val_*` test proving the mechanism actually helps a downstream model, only narrow numerics-identity tests (bit-identical-to-reference-loop) for a handful of them. CLAUDE.md's own convention ("Every ML trick gets a quantitative biz_value test") is met for `row_attention`, `random_features`, `class_distance`/`local_lift`, `bgm_rsd_knn`, `neighbor_aggregate_features`, `pe`, `swap_noise`, `denoising_autoencoder` but not for this cluster's ~30 newer mechanisms. |
| PR2 | architecture / dedup | pseudo_smote.py, bgm_clustered_smote.py, borderline_smote.py, adasyn_smote.py, multiscale_smote.py, mixup_boundary.py, bgmm_dual_class.py (7 of this cluster's files; 19 across the whole `transformer/` package) | Near-identical `_kth_nearest_dists` (sentinel-on-empty kNN-distance helper), `_slice`/`_slice_and_binarize` (pos/neg or quantile-tail splitter), and `_make_df` (k-scale column naming) are copy-pasted verbatim or near-verbatim across the SMOTE-distance-feature family. A shared `_smote_common.py` (or extending `_utils.py`) would remove ~40-60 duplicated lines per file and centralise the sentinel-value/empty-subset policy so a future fix (e.g. this report's F3/F5 empty-band-fallback pattern) only needs to land once. |
| PR3 | perf / vectorization | multi_threshold_ordinal.py:84-87 | `rank_pred` is computed via a per-row Python loop (`for q_i in range(Xq_s.shape[0]): cross = np.where(preds[q_i] < 0.5)[0]; ...`) over what is typically a 5-7 column array; this vectorises directly to `np.argmax(preds < 0.5, axis=1)` with a `.any(axis=1)` mask for the "no crossing" fallback, removing the per-row Python/numpy call overhead on large `n_q`. |
| PR4 | ML-practice | jackknife_endpoint_stability.py, disagreement_band.py, baseline_disagreement_v2.py, gradient_direction_agreement.py (the "fit K auxiliary LGB/Ridge/LogisticRegression baselines per fold" family) | Several files fit 3-4 auxiliary models per fold (some ×`n_subsamples`=10 more); none of them reuse a shared "small LGB baseline factory" helper -- each file hand-rolls the same `n_estimators=50, max_depth=3, learning_rate=0.1, verbose=-1, n_jobs=-1` LGB construction. Centralising this (plus the single-class/degenerate-target guard already present in some files but not others, see F6) would remove duplication and close the guard gap in one place. |

## Coverage notes

Every file in scope was opened and read in full; none were too large to review in depth (largest is 475 LOC). The two explicitly-excluded packages (`filters/`, `shap_proxied_fs/`) were not touched. I did not execute any code, run pytest, or run the fuzz/biz_val suites (read-only audit per instructions), so the edge-case findings above (F2-F6) are derived from static code-path analysis of the exact conditions that trigger them, not from an observed failure; I did not have a way to empirically confirm e.g. that mlxtend's `fpgrowth` or LightGBM's `LGBMClassifier` actually raise the way I describe (their exact exception behaviour on single-class `y` was not verified against the installed library version) -- flagging these as P1 robustness risks rather than confirmed P0 crashes reflects that uncertainty. I did not open files outside the declared 40-file scope except for brief, targeted greps (`build_hnsw_index` call sites, `_kth_nearest_dists` definitions) used only to establish whether an in-scope pattern recurs elsewhere in the package, and I did not read or evaluate those out-of-scope files' own correctness.
