# feature_engineering/transformer B (residual/baseline-disagreement family, projections, autoencoders) -- mlframe audit

## Scope

All 39 assigned files were read in full (no partial-review files):

- src/mlframe/feature_engineering/transformer/_projection.py (455 LOC)
- src/mlframe/feature_engineering/transformer/_aggregation.py (327 LOC)
- src/mlframe/feature_engineering/transformer/mdl_binning_pairwise.py (252 LOC)
- src/mlframe/feature_engineering/transformer/multi_baseline_hard_row.py (245 LOC)
- src/mlframe/feature_engineering/transformer/class_balanced_hard_row.py (239 LOC)
- src/mlframe/feature_engineering/transformer/local_classifier.py (225 LOC)
- src/mlframe/feature_engineering/transformer/class_conditional_anchor.py (219 LOC)
- src/mlframe/feature_engineering/transformer/spectral_attention.py (208 LOC)
- src/mlframe/feature_engineering/transformer/rf_proximity.py (205 LOC)
- src/mlframe/feature_engineering/transformer/cluster_smote.py (195 LOC)
- src/mlframe/feature_engineering/transformer/fisher_weighted_residual.py (195 LOC)
- src/mlframe/feature_engineering/transformer/neighbor_aggregate_features.py (193 LOC)
- src/mlframe/feature_engineering/transformer/local_density_gradient.py (185 LOC)
- src/mlframe/feature_engineering/transformer/bgmm_virtual.py (184 LOC)
- src/mlframe/feature_engineering/transformer/ks_shift.py (184 LOC)
- src/mlframe/feature_engineering/transformer/prediction_band_attention.py (181 LOC)
- src/mlframe/feature_engineering/transformer/target_quantile.py (175 LOC)
- src/mlframe/feature_engineering/transformer/diffusion_noise.py (174 LOC)
- src/mlframe/feature_engineering/transformer/bgmm_density_ratio.py (173 LOC)
- src/mlframe/feature_engineering/transformer/y_quintile_baseline_knn.py (170 LOC)
- src/mlframe/feature_engineering/transformer/adaptive_bandwidth.py (165 LOC)
- src/mlframe/feature_engineering/transformer/pairwise_kl_divergence.py (165 LOC)
- src/mlframe/feature_engineering/transformer/cutmix.py (163 LOC)
- src/mlframe/feature_engineering/transformer/nca_projection.py (156 LOC)
- src/mlframe/feature_engineering/transformer/class_mahalanobis.py (152 LOC)
- src/mlframe/feature_engineering/transformer/ib_baseline_codes.py (147 LOC)
- src/mlframe/feature_engineering/transformer/predictive_info_delta.py (147 LOC)
- src/mlframe/feature_engineering/transformer/autoencoder.py (145 LOC)
- src/mlframe/feature_engineering/transformer/tree_path_boolean.py (141 LOC)
- src/mlframe/feature_engineering/transformer/trust_score_oof.py (139 LOC)
- src/mlframe/feature_engineering/transformer/robustness_budget.py (137 LOC)
- src/mlframe/feature_engineering/transformer/pred_augmented.py (132 LOC)
- src/mlframe/feature_engineering/transformer/persistence_diagram.py (130 LOC)
- src/mlframe/feature_engineering/transformer/per_column_rff.py (123 LOC)
- src/mlframe/feature_engineering/transformer/aux_mlp.py (119 LOC)
- src/mlframe/feature_engineering/transformer/quantile_spread_fan.py (115 LOC)
- src/mlframe/feature_engineering/transformer/fca_closed_concepts.py (100 LOC)
- src/mlframe/feature_engineering/transformer/stacked_qnn.py (97 LOC)
- src/mlframe/feature_engineering/transformer/swap_noise.py (94 LOC)

Total files reviewed: 39. Total LOC reviewed: 6912 (4364 in the first 20 files + 2548 in the remaining 19, per `wc -l`).

For context only (not analyzed for findings, per the task's exception clause for imported symbols): peeked at `require_seed()` in `_utils.py` (a sibling file *not* in my 39-file list) to confirm every in-scope file's seed handling is sound -- it rejects `None`, non-int types, and out-of-range values, so the "hardcoded/defaulted seed" bug class does not apply anywhere in this cluster. Also noted but did not analyze: `_knn_helper.knn_search` (imported by `neighbor_aggregate_features.py`), `_kernels_njit.row_attention_stage4_adaptive_njit` and `_row_attention_ann.{build_hnsw_index,query_topk}` (imported by `adaptive_bandwidth.py`), and `row_attention.compute_row_attention` / `_oof.apply_dedupe` / `_residual_oof.compute_oof_yhat_within` (imported by `pred_augmented.py`) -- all are siblings under `transformer/` not in my assigned list.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P1 | correctness/edge-case | y_quintile_baseline_knn.py:123 | Binary-task strata edges shifted by `-1e-9` silently exclude rows with baseline-prob exactly 1.0 from *every* stratum's neighbour bank |
| F2 | P1 | correctness | pairwise_kl_divergence.py:132-135 | Regression Gaussian-JS collapses the per-row mixture variance into one dataset-wide scalar (`mean_sigma`), so the `js` feature silently loses its per-row structure on every call, not just an edge case |
| F3 | P1 | silent-failure/edge-case | cluster_smote.py:159-160 | Degenerate fold (<2 pos or <2 neg rows) returns literal `0.0` for all 8 distance/log-gap features instead of a "no info" sentinel; `0.0` distance reads as "identical row" |
| F4 | P1 | silent-failure/edge-case | bgmm_virtual.py:147-148 | Same zero-fallback pattern as F3 for BGM-virtual distance features |
| F5 | P1 | silent-failure/edge-case | diffusion_noise.py:127-128 | Same zero-fallback pattern as F3 for diffusion-noise distance features |
| F6 | P1 | silent-failure/edge-case | cutmix.py:128-129 | Same zero-fallback pattern as F3 for CutMix distance features |
| F7 | P1 | silent-failure/edge-case | bgmm_density_ratio.py:120-122 | Degenerate fold returns `0.0` log-density/log-ratio for all 9 features, inconsistent with the `-30.0` "low-density" sentinel the same file's `_fit_bgmm_and_score` uses for its own too-few-rows case |
| F8 | P1 | silent-failure/edge-case | class_mahalanobis.py:111-113 | Degenerate fold (<2 pos or <2 neg) returns `0.0` for `m_pos`/`m_neg`/`m_gap` instead of a value indicating "undefined" |
| F9 | P1 | correctness/edge-case | multi_baseline_hard_row.py:160-171 | When a class/quintile side is entirely empty in a fold, anchor indices fall back to `np.zeros(n_hard_per_side, dtype=np.int64)` — repeats row 0 (an arbitrary, possibly opposite-class row) as the "hardest anchors" |
| F10 | P1 | correctness/edge-case | class_balanced_hard_row.py:149-160 | Same index-0 anchor fallback bug as F9 |
| F11 | P1 | silent-failure | tree_path_boolean.py:110-113 | `except Exception: paths = []` around LGB path extraction swallows the error with zero logging; downstream silently ships constant-1.0 boolean columns |
| F12 | P1 | silent-failure | persistence_diagram.py:100-101 | `except Exception: pass` around the per-row gudhi call swallows the error with zero logging; row silently stays all-zero |
| F13 | P1 | silent-failure | fca_closed_concepts.py:66-67 | `except Exception: top_concepts = []` around FCA lattice construction swallows the error with zero logging |
| F14 | P1 | silent-failure | multi_baseline_hard_row.py:73-75 | LogisticRegression third-baseline fit failure falls back to the class-prior constant with **no** `logger` call, unlike the equivalent BGM/NCA fallbacks elsewhere in this cluster which all log |
| F15 | P1 | correctness/API | mdl_binning_pairwise.py:162,184-229 | `task` parameter is accepted but never read inside `_process`; binary-vs-regression binning is instead inferred purely from whether `y` values are exactly `{0,1}`, so a caller passing `task="binary"` for e.g. `{1,2}`-coded labels silently gets 5-class quantile binning |
| F16 | P1 | robustness/edge-case | quantile_spread_fan.py:68-75 | No guard against a single-class training fold before fitting `LGBMClassifier`; for the `gamma>0` branch a single-class fold makes `sample_weight` literally all-zero (`(1-p_mean)**gamma == 0` when `p_mean==1.0`), risking a crash or a degenerate zero-weighted fit |
| F17 | P2 | dead-code/robustness | mdl_binning_pairwise.py:184-191 | The `else` branch of `if y_t.dtype != np.int32` leaves `y_class`/`n_classes` undefined; currently unreachable only because `y_t` is always cast to `float32` upstream (line 177) — a latent `NameError` if that upstream cast ever changes |
| F18 | P2 | silent-failure/edge-case | class_conditional_anchor.py:198-200 | Degenerate fold (<2 rows per class) skips filling that fold's val rows, leaving the pre-allocated `np.zeros(...)` in place; at least logs a `logger.warning`, unlike F3-F8, so lower severity variant of the same pattern |
| F19 | P2 | correctness/edge-case | prediction_band_attention.py:132-134 | Empty prediction-quantile band leaves `band_centroids[b]`/`band_y_mean[b]`/`band_pred_mean[b]` at their zero-initialized default (a "phantom" band at the scaled-data origin) rather than being excluded from the softmax |
| F20 | P2 | correctness/edge-case | fisher_weighted_residual.py:138-147 | Empty Fisher-weighted-residual band leaves `band_y_mean[b]` at `0.0` |
| F21 | P2 | correctness/edge-case | predictive_info_delta.py:85-98 | Empty prediction bin leaves `H_y_per_bin[b]` at `0.0` (reads as "perfectly certain"/`var==1`, not "no data") |
| F22 | P2 | correctness/edge-case | ib_baseline_codes.py:104-110 | Empty baseline-code cell leaves `code_y_mean[c]`/`code_y_std[c]` at `0.0` |
| F23 | P2 | architecture/duplication | multi_baseline_hard_row.py:42-98, class_balanced_hard_row.py:47-86 | `_softmax` and `_topk_within_subset` are byte-for-byte duplicated between the two files (and the surrounding `_process`/`_make_df`/padding skeleton is ~90% identical) |
| F24 | P2 | architecture/duplication | cluster_smote.py, bgmm_virtual.py, diffusion_noise.py, cutmix.py, bgmm_density_ratio.py | `_K_SCALES`, `_slice`, and `_kth_nearest_dists` are duplicated near-verbatim across all 5 "virtual-positive distance" files |
| F25 | P2 | docs/correctness | swap_noise.py:82-89 | Docstring states each swapped value comes from "a genuinely different row"; the implementation draws replacement rows via an unrestricted `rng.permutation(n)[:n_swap]` that can include the cell's own row index, so a self-assignment (no-op corruption) can occur with probability ~1/n |
| F26 | P2 | perf | local_classifier.py:168-184 | Per-query-row Python loop calling `_solve_weighted_logreg`/`_solve_weighted_linreg` (a small dense linear solve each) — not vectorized/njit despite the repo's numerical-kernel-acceleration convention |
| F27 | P2 | perf | cluster_smote.py:55-63, cutmix.py:53-55 | Per-synthetic-row Python loops (`for i in range(n_synthetic)`) generating virtual rows one at a time |
| F28 | P2 | perf | persistence_diagram.py:73-101 | Per-query-row Python loop calling `gudhi.RipsComplex`/`compute_persistence`; embarrassingly parallel over rows but run serially |
| F29 | P2 | docs/robustness | trust_score_oof.py:92-96 | `_nn_dist`'s empty-subset sentinel returns `nearest=0.0` (reads as "touching", not "unknown"); lower severity than F3-F8 because it is explicitly documented in the docstring |

### Finding details

**F1 (y_quintile_baseline_knn.py:123).** `strata_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) - 1e-9` shifts every boundary down uniformly. The unshifted `>=lo & <hi` (all bands) / `>=lo & <=hi` (last band) scheme already handles boundary ties correctly without any epsilon, so the shift is superfluous — and it introduces a real defect: the last stratum's mask becomes `y >= 0.8-1e-9 & y <= 1.0-1e-9`, so any training row whose baseline probability is *exactly* `1.0` (LightGBM can saturate to this on well-separated small folds) fails every stratum's mask and is silently absent from all 5 neighbour banks used by `_knn_pred_stats`. Suggested fix: drop the `-1e-9` shift (or only apply it to interior edges, never to `0.0`/`1.0`).

**F2 (pairwise_kl_divergence.py:132-135).** For regression, `mean_var` is computed per query row (`(sigmas[i]**2 + (p_i-mean_mu)**2)` averaged over the 3 baselines), but then `mean_sigma = float(np.sqrt(mean_var.mean()))` reduces it to a single scalar averaged across the *entire query set*, and that one scalar is reused as `sigma_j` for every row's `_gaussian_kl(..., mean_sigma)` call. The intended per-row mixture spread is discarded; the `js` feature becomes a blend of each row's true divergence with the query-set average, silently on every call (not an edge case). Suggested fix: keep `mean_sigma` as a per-row array and pass it through `_gaussian_kl` elementwise (the function already accepts array `mu_i`/`mu_j`; only the `sigma_j` argument needs to become an array parameter).

**F3-F8 (cluster_smote.py / bgmm_virtual.py / diffusion_noise.py / cutmix.py / bgmm_density_ratio.py / class_mahalanobis.py).** All six files share one root cause: when a training fold's positive-like or negative-like subset has fewer than 2 rows (a realistic outcome for the 1.3%-positive-style imbalanced binary targets this whole file family is designed around, especially with K>5 folds), `_process` returns a literal all-zeros feature block instead of a value consistent with "this fold had no signal." For the five "virtual-positive distance" files (F3-F7) a `0.0` distance is the worst possible choice semantically — it reads as "identical to the positive/negative manifold" — and it is inconsistent with the `1e6` sentinel the *same file's* `_kth_nearest_dists` helper already uses for its own "empty subset" case. For `class_mahalanobis.py` (F8), `0.0` for `m_pos`/`m_neg` reads as "sits exactly at the class centroid." Suggested fix: reuse each file's own internal sentinel convention (e.g. the `1e6` distance sentinel, or `-30.0` log-density sentinel already used one level down) for the fold-level degenerate case too, or emit an explicit `_degenerate` boolean flag column.

**F9-F10 (multi_baseline_hard_row.py:160-171, class_balanced_hard_row.py:149-160).** When a class/quintile side has zero rows in a fold, `pos_top`/`neg_top` falls back to `np.zeros(n_hard_per_side, dtype=np.int64)` — i.e. repeats training row index 0, `n_hard_per_side` times, as that side's "hardest anchors," regardless of what row 0 actually is (it could belong to the opposite class). Every val row in that fold then gets softmax-attention weights against a phantom anchor set built from one arbitrary, likely wrong-class row. Suggested fix: when a side is fully empty, skip emitting that side's weight columns for the fold (leave them at a documented sentinel) rather than synthesizing a degenerate anchor set.

**F11-F13 (tree_path_boolean.py:110-113, persistence_diagram.py:100-101, fca_closed_concepts.py:66-67).** Each wraps its core "extract structure" step in a bare `except Exception:` that swallows the error completely — no `logger.info`/`logger.warning` call at all, unlike almost every other fallback in this cluster (BGM/NCA/LogisticRegression fallbacks elsewhere call `logger.info` with the exception message before falling back). A real bug (library API change, version mismatch, malformed model dump) would silently degrade these transformers to constant/all-zero columns with zero diagnostic trail in production logs. Suggested fix: add a `logger.info("...: extraction failed (%s); falling back", exc)` call matching the pattern used everywhere else in this cluster (e.g. `bgmm_virtual.py`, `nca_projection.py`).

**F14 (multi_baseline_hard_row.py:73-75).** The third-baseline `LogisticRegression` fit-failure fallback (constant class-prior) has no `logger` call, while the structurally identical fallback in `pairwise_kl_divergence.py`/`ib_baseline_codes.py` (same 3-baseline pattern) also lacks logging there too — but `bgmm_virtual.py`/`nca_projection.py`/`bgmm_density_ratio.py` all log. Inconsistent within the same codebase; flagged separately from F11-13 because it is a smaller, narrower fallback (one of 3 baselines, not the whole transform).

**F15 (mdl_binning_pairwise.py:162,184-229).** `compute_mdl_binning_pairwise_features(..., task="regression", ...)` documents and accepts a `task` argument, but `_process` (called for both Mode A and Mode B) never reads it — it re-derives "is this binary" purely from `(y_t==0).sum()+(y_t==1).sum()==y_t.size`. A caller who explicitly passes `task="binary"` for a target encoded as `{-1,1}` or `{1,2}` (a real, if less common, encoding) silently gets 5-class quantile MDL binning instead of the requested 2-class binning, with no warning that the parameter was ignored. Suggested fix: either honour `task` explicitly, or drop the parameter and document the auto-detection contract.

**F16 (quantile_spread_fan.py:68-75).** Unlike most other binary-task files in this cluster (`cluster_smote.py`, `bgmm_virtual.py`, `class_conditional_anchor.py`, etc.), `quantile_spread_fan.py` fits `LGBMClassifier` directly on the fold's `y_t` with no `len(unique(y_t)) < 2` guard. In the `gamma>0` reweighting branch, if the fold happens to be single-class (`p_mean == 1.0` or `0.0`), `sw = np.where(y_t > 0.5, (1.0-p_mean)**gamma, p_mean**gamma)` becomes literally all-zero, which either raises inside LightGBM or trains against an all-zero-weight objective. Suggested fix: add the same `<2 per class` guard used elsewhere in the cluster before entering the per-gamma fit loop.

**F17 (mdl_binning_pairwise.py:184-191).** Latent bug, not currently reachable: `y_t` is always constructed via `np.asarray(y_train, dtype=np.float32)` upstream, so the `else` branch of `if y_t.dtype != np.int32` (which would use `y_class`/`n_classes` — never assigned in that branch) can never execute today. Flagged as P2 because it is a `NameError` waiting for the first future refactor that changes the upstream cast.

**F18 (class_conditional_anchor.py:198-200).** Same root cause as F3-F8 (degenerate-fold zero-fill) but the code at least logs a `logger.warning` before skipping — kept as a separate, lower-severity finding to make the recurring-pattern audit explicit and complete per file.

**F19-F22 (prediction_band_attention.py, fisher_weighted_residual.py, predictive_info_delta.py, ib_baseline_codes.py).** A second, related instance of the same "silent zero on empty bucket" root cause, this time for empty *quantile bands/bins* rather than empty classes. Given `n_bands`/`n_bins` default to 5-10 on datasets with reasonably-sized folds, an empty band/bin is rarer than an empty class-side (F3-F10), hence P2 rather than P1 — but the mechanism is identical: `np.zeros(...)` pre-allocation plus an `if mask.sum() > 0:` guard that only fills the *non*-degenerate case, silently leaving `0.0` in the degenerate one.

**F23-F24 (code duplication).** `multi_baseline_hard_row.py` and `class_balanced_hard_row.py` differ only in how many/which baselines get fit (3 diverse baselines vs. 1 shallow LGB); everything else — `_softmax`, `_topk_within_subset`, the anchor-padding logic, the `_process`/`_make_df` skeleton — is duplicated near-verbatim (~150+ LOC). Similarly, the five "virtual-positive distance" files (`cluster_smote.py`, `bgmm_virtual.py`, `diffusion_noise.py`, `cutmix.py`, `bgmm_density_ratio.py`) all redefine `_K_SCALES`, `_slice`, and `_kth_nearest_dists` identically. A shared internal helper module (e.g. `_hard_row_common.py`, `_virtual_positive_common.py`) would remove the duplication and, per the project's own "grep ALL instances, fix one pass" convention, would let F3-F10's fix land in one place instead of 7.

**F25 (swap_noise.py:82-89).** The docstring is explicit: "drawn from a genuinely different row of the SAME column." The implementation samples the source rows via `perm = rng.permutation(n)[:n_swap]` with no exclusion of the destination row's own index, so `X_out[col_mask, j] = X[perm, j]` can, by chance, replace a cell with its own original value (probability ~`n_swap/n` per swapped cell, vanishing as `n` grows). Negligible practical impact at the intended N but a real deviation from the stated contract. Suggested fix: derangement-style rejection (redraw indices that collide with their own position) or accept and correct the docstring.

**F26-F28 (perf).** Per-row/per-synthetic-row Python loops in `local_classifier.py`, `cluster_smote.py`/`cutmix.py`, and `persistence_diagram.py`. None of these crash or produce wrong results; each is a candidate for the repo's numba/vectorize acceleration ladder given the stated cost estimates already assume small `n_q`/`n_synthetic` (thousands, not tens of thousands) — worth profiling once these transformers are used at larger fold sizes.

**F29 (trust_score_oof.py:92-96).** Same "misleading zero sentinel" shape as F3-F8 in spirit (`nearest=0.0` for an empty correctness-subset), but explicitly documented in the function's own docstring ("returns sentinel fill values ... when the subset is empty"), so it is a documented tradeoff rather than a silent one — kept as P2 for completeness of the recurring-pattern sweep, not because it is undocumented.

## Proposals

| ID | Category | File | Summary |
|----|----------|------|---------|
| PR1 | test-coverage | (cluster-wide) | Confirmed (via grep across `tests/feature_engineering/transformer/`) that essentially every `compute_*` function in this cluster has at least one dedicated identity/kernel test or a reference inside the 9864-line `test_biz_val_real_datasets.py`; `aux_mlp.py` is covered separately in `test_biz_val_supervised_projection_ops.py`. No blanket "missing tests" gap found for this cluster — flagged as a proposal only because I did not verify each individual biz_val assertion's threshold is meaningful (see Coverage notes). |
| PR2 | robustness | cluster-wide | Systematically add `len(np.unique(y_t)) < 2` guards before every `LGBMClassifier`/`LogisticRegression` fit on a CV fold's `y_t` across files that currently lack one (`class_balanced_hard_row.py`, `multi_baseline_hard_row.py`, `prediction_band_attention.py`, `fisher_weighted_residual.py`, `y_quintile_baseline_knn.py`, `robustness_budget.py`, `aux_mlp.py`, `tree_path_boolean.py`, `predictive_info_delta.py`, `quantile_spread_fan.py`, `pairwise_kl_divergence.py`, `trust_score_oof.py`, `ib_baseline_codes.py`) — a single-class fold is a realistic outcome for the rare-positive-class scenarios this whole cluster targets. |
| PR3 | architecture | multi_baseline_hard_row.py, class_balanced_hard_row.py | Extract the shared `_softmax`/`_topk_within_subset`/anchor-padding logic into one internal helper module (see F23). |
| PR4 | architecture | cluster_smote.py, bgmm_virtual.py, diffusion_noise.py, cutmix.py, bgmm_density_ratio.py | Extract the shared `_K_SCALES`/`_slice`/`_kth_nearest_dists` helpers into one internal module (see F24); fixing F3-F7's sentinel inconsistency there would then be a single-site change. |
| PR5 | perf | local_classifier.py | Vectorize or `numba.njit`-parallelize the per-query weighted-logreg/linreg loop (F26); the file already documents an `O(N_q * k * d^2)` cost estimate that would benefit from `prange` at larger `n_q`. |
| PR6 | perf | persistence_diagram.py | Parallelize the per-query gudhi loop (F28) via joblib (`n_jobs=-1`), matching the `n_jobs=-1` convention already used for the `NearestNeighbors` call two lines above it. |
| PR7 | ML-quality | pairwise_kl_divergence.py | Beyond fixing F2, add a `test_biz_val_pairwise_kl_divergence.py` (or extend an existing biz_val test) asserting the `js` feature actually discriminates rows with genuinely different per-row baseline-disagreement spread — the current bug (F2) would not be caught by a test that only checks column presence/shape. |
| PR8 | docs | swap_noise.py | Either fix F25 (exclude self-index) or soften the docstring's "genuinely different row" claim to note the small-probability self-match at finite `n`. |

## Coverage notes

- Did not execute any tests, benchmarks, or the transformer functions themselves — this is a static, read-only review; all findings are derived from reading source, not from runtime reproduction. Per the task's regression-test convention, each finding above would need a runtime repro before a fix lands, which is out of scope here.
- `tests/feature_engineering/transformer/test_biz_val_real_datasets.py` is 9864 LOC and sits outside my 39-file scope (it is a test file, not one of the assigned src files). I confirmed via `grep` that most `compute_*` functions in this cluster are referenced by name somewhere in it or in a dedicated per-mechanism test file, but I did not read its assertions/thresholds in depth, so I cannot say whether any of F1-F29's failure scenarios are already caught by an existing test (my working assumption, given how narrow each edge case is, is that they are not, but I have not proven a negative).
- Did not analyze the sibling helper files these 39 files import from (`_utils.py`, `_knn_helper.py`, `_kernels_njit.py`, `_row_attention_ann.py`, `row_attention.py`, `_oof.py`, `_residual_oof.py`, `quantile_neighbours.py`) beyond the one `require_seed` function read for seed-handling context — they are not in my assigned file list and may belong to another cluster's scope.
- Did not attempt to reproduce F1 (the exact-1.0 boundary exclusion) or F16 (single-class-fold zero sample weight) against a real LightGBM run to confirm the exact failure mode (silent feature loss vs. a raised exception) — both are derived from careful static trace of the boolean-mask arithmetic, not from an executed repro.
