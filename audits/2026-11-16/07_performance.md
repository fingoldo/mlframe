# 07 Performance audit (mlframe src, MRMR excluded)

Method: grep sweep for row-wise python loops (range(len), iterrows, apply lambda), then read the hits. Shallow pass; not exhaustive.

| ID | severity | file:line | finding | evidence | disposition |
|---|---|---|---|---|---|
| P1 | med | feature_engineering/transformer/apriori_itemsets.py:54 | Per-itemset iterrows loop; each iteration does `np.all(bin_matrix_train[:, cols_idx], axis=1)` (fancy-index copy of n_rows x k) plus mean; cost O(n_itemsets * n_rows * k) with a pandas row object per itemset | `for _, row in frequent.iterrows(): ... mask_train = np.all(bin_matrix_train[:, cols_idx], axis=1)` | TODO: replace iterrows with iterating `frequent["itemsets"]` and use a packed-bits or njit kernel over the bool matrix; proposed test `test_apriori_itemset_lifts_matches_reference_after_vectorization` |
| P2 | low | feature_selection/filters/_cat_pair_fe.py:501, _cat_triple_fe.py:405,522 | iterrows over the kept pair/triple score frames | `for _, row in keep.iterrows()`; body dominated by per-column code/str building, frame is capped to top-k | REJECTED: loop count is small (top_k), per-iteration work is column-sized and already cached via code_cache/str_cache, so iterrows overhead is negligible |
| P3 | low | feature_engineering/mps.py:319,327 | Python-looking fill loops over array | function `backfill_zeros` is decorated `@numba.njit(fastmath=True, cache=True)` (mps.py:293) | REJECTED: njit lever already covers the path |
| P4 | low | core/arrays.py:126,133 | range(len) loops | `_validate_counting_inputs` is `@njit` (arrays.py:119) | REJECTED: already njit |
| P5 | low | feature_engineering/fuzzy_entity.py:52 | Pairwise Levenshtein within blocks is O(b^2) per block | blocked by key prefix; rapidfuzz `score_cutoff` used; C-implemented distance | DOC: quadratic in block size; a very common prefix degenerates; consider rapidfuzz.process.cdist with workers for large blocks (FUTURE if profiling shows it) |
| P6 | low | feature_selection/boruta_shap/_shadow_stats.py:180 | Column-wise permute via DataFrame.apply | comment at :139 documents the per-column independent permutation is intentional | REJECTED: per-column loop is O(cols) numpy permutations, not row-level; semantic requirement |
