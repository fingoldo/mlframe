# MRMR: places where the number of pairs / columns / candidates is cut by a constant instead of streamed in memory-bounded batches

Source: read-only audit agent, 2026-10-22. Nothing was executed; the claims below are the agent's reading of the code and have NOT yet been verified one by one (the disposition column is filled in as each is triaged).
The "identical or better selection" statement for the streamed variants is a design argument, not a measurement.

Conventions: H = free host RAM (`psutil.virtual_memory().available`; helpers `_feature_engineering_mem_budget._available_ram_bytes_cached`, `_fe_min_free_ram_bytes`). V = free VRAM (`cp.cuda.runtime.memGetInfo()` plus `_fe_gpu_vram.fe_gpu_has_vram_cushion`).
Paths are under `src/mlframe/feature_selection/`; `F/` = `filters/`; class defaults are in `F/mrmr/_mrmr_class.py`. Row caps stream over row blocks (contingency counts and Pearson sums are additive; bin edges and quantiles are a global pass and are not).
Categories: A = memory/time guard only, streaming strictly better; B = statistical or output-budget decision, must stay (could still be made memory-safe); C = unclear / part of a feature definition / hardware or time budget.

## 1. Pair search and the FE step

1. [A] `fe_max_engineered_operands=8` (class:847, applied `F/_mrmr_fe_step/_step_pairmi.py:202,233`). Keeps only the top 8 engineered operands by marginal MI for composite pairing. Log line: "dropped ... to bound the pair count"; comment: "To bound the O(k^2) pair blow-up".
   - Stream: pair blocks of the full pool, already chunked by `dispatch_batch_pair_mi_chunked`. Size from H, or V on the GPU path.
   - Global statistic: the order-2 maxT floor `compute_pair_maxt_floor` (`F/_mrmr_fe_step_helpers.py:444`) is over the whole pool; it decomposes as the elementwise max over pair blocks with the same permutations, then a quantile. A bigger pool raises the floor by design.
   - Also: `cached_MIs` is a Python dict holding every pair (O(P) memory).
2. [B] `fe_ntop_features=0` (class:833, `F/_mrmr_fe_step/_step_pool.py:148`). User restriction of the pool to `selected_vars[:N]`. Off by default.
3. [C] `fe_max_pair_features=10` (class:861). Only passed through (`F/_feature_engineering_pairs/_pairs_core.py:320`) and mentioned in a docstring; no slice applying it was found.
4. [A] `fe_pair_usability_prescan_max_pairs=256` (class:1801, `F/_mrmr_fe_step/_step_pairs_rank.py:160,203`). Comment: "Cost guard ... Bounds the O(pairs) x O(n) |corr| scan". The scan already walks pairs in descending MI order.
   - Stream: all pairs in blocks through `dispatch_batch_pair_usability_corr`. Signal: H or V. Global statistic: dominant pair is the max |corr|, a running max.
5. [A] `fe_check_pairs_subsample_n=UNIFIED_FE_SUBSAMPLE_N=30000` (class:1040, `F/_feature_engineering_mem_budget.py:292`, `F/_feature_engineering_pairs/_pairs_core.py:332`). Row cap on the pair sweep. Comment: "so the transformed_vars + shared_buffer allocations scale with subsample_n"; `_can_hoist_shared_buffer` now has a recompute fallback, so what remains is mostly runtime. A benchmark claims jaccard 1.0 at 25k rows and above.
   - Stream: row blocks with additive counts. Quantile edges are the non-additive part.
6. [A] `fe_smart_polynom_subsample_n=30000` (class:1029, `F/polynom_pair_fe.py:104,330`). Same row cap for the polynomial pair search, speed rationale.
7. [A] `MLFRAME_USABILITY_CORR_MAX_ROWS=30000` (`F/_fe_usability_signal.py:143,160`, used `_step_pairs_rank.py:90,226`). Comment calls the path "HOT". Pearson sums are exactly additive over row blocks, so an exact full-n streaming version exists.
8. [A] `MLFRAME_PAIR_NULL_MAX_ROWS=250000` (`_step_pairs_rank.py:352,880`). Comment cites "~2.4s at 1M" for the CMI plus permutation null. Stream the permutation null by row block or permutation chunk.
9. [A] `MLFRAME_FE_GATE_MAX_ROWS=250000` (`F/_mrmr_fe_step/_step_score.py:226-229`). Comment: "caps scoring cost only".
10. [A] `fe_engineered_cmi_max_candidates=64` (class:983, `_step_score.py:265`, `F/_fe_cmi_redundancy_gate.py:190,780-796`). The CMI redundancy gate keeps only the top 64 candidates by marginal MI; the dropped tail is recorded as "dropped_cost_cap". Comment: "bounding the cost to O(M^2)".
    - Stream: score all marginals in column blocks, then run the greedy rounds with chunked candidate scans. Signal: H or V. Caveat: runtime cost, not memory; the greedy is sequential and not decomposable, though the per-candidate permutation null is per candidate, not a family statistic.
11. [A] `fe_synergy_screen_max_features=250` (class:1640, `F/_mrmr_fe_step_helpers.py:50,~90-140`, `F/_fe_interaction_prerank.py`). Above 250 raw columns the all-pairs synergy sweep runs only on the top 250 by an |corr(x^2,y)|+|corr(x,y^2)| pre-rank. The comment concedes it cannot see perfectly balanced interactions.
    - Stream: the exhaustive path already exists (`F/_fe_synergy_exhaustive.py`) and bypasses this cap. Global statistic: the same pair maxT floor as item 1.
12. [A] `fe_synergy_max_sweep_cost=5e8` (class:1683, `_mrmr_fe_step_helpers.py:70`). The sweep is skipped when n*p^2 exceeds the budget: a pure runtime guard; can be dropped if the pairs are streamed.
13. [A] `fe_synergy_max_pairs=16` (class:1691, `F/_mrmr_fe_step/_step_core_helpers.py:16-30`). Keeps the top 16 synergy pairs by joint MI. Comment: "blowing up FE wall-time". Streamed or not, the per-pair unary/binary search cost remains.
14. [A] Rung schedule (`F/_fe_rung_schedule.py:196-232`; class:683 `fe_rung_keep_frac=None`, fallbacks 0.34 large pool / 0.5 moderate; `fe_rung_rel_floor=0.40`; `fe_rung_min_pairs=6`). On by default; keeps the top fraction of pairs by pair-MI rank; explicitly a "search budget"; the relative floor protects moderate winners. Global statistic: `keep_n` and the floor `rel*max_pm` come from the full pair-MI vector (cheap scalars, exact).
15. [A] `fe_escalation_max_pairs=8` and `fe_escalation_max_candidates_per_pair=3` (class:743,760, `F/_fe_auto_escalation.py:346-383,433,510-514`). Pair budget on the expensive ALS and Fourier proposers; half reserved for rescue pairs. Runtime only.
16. [A] Escalation row stride (`F/_fe_auto_escalation.py:113,301-320`: `max_rows=20000`, plus subsampling to `fe_check_pairs_subsample_n`). Comment: "the dominant active orth-FE CPU cost".
17. [B] `fe_additive_fusion_max=4` (class:1732, `F/_fe_additive_fusion.py:349,377`; GPU twin `F/_fe_additive_fusion_gpu_resident.py:347,385`). Caps the number of admitted fusions at a greedy early exit. Output budget.
18. [A] `MLFRAME_FE_FUSION_MAX_ROWS=250000` (`F/_fe_additive_fusion_gpu_resident.py:159`). Subsamples rows for scoring only.
19. [A, low] `F/_fe_additive_fusion.py:190` allocates an (n, H) int64 matrix of all halves at once. Chunk over halves.
20. [A] `detect_synergy_combos(max_candidates=24)` (`F/_fe_synergy_screen.py:216,244`). Keeps the 12 highest- and 12 lowest-marginal candidates. Comment: "to bound the O(P^order) combo count". Stream the combos in blocks.
21. [A] `detect_synergy_combos(max_combos=4000)` (`_fe_synergy_screen.py:216,265,293`). A prefix of the lexicographic `combinations` iterator, so the candidates cut depend on column order. Pure budget.
22. [B] `_MAX_CELLS = 1<<20` and `min_rows_per_cell=5` (`_fe_synergy_screen.py:249,255`). Skip combos whose joint cells are too sparse ("would be unreliable anyway").
23. [A] The n-way synergy screen runs only when 2 <= candidates <= 60 (`F/_mrmr_fit_impl/_friend_graph_and_redundancy/_group1.py:240`); above 60 it is silently skipped. It also calls `fe_to_pandas(X)` on the whole frame.

## 2. Hard skips and guards

24. [A] `FE_EAGER_MATERIALIZE_MAX_BYTES = 2 GiB` (`F/_fe_frame_ops.py:106`) skips K-fold target encoding on a large polars frame (`F/_mrmr_fit_impl/_fe_stage_cascade_early_b.py:56`).
25. [A] Same constant skips the count/frequency/cat-num family (`early_b:172`).
26. [A] Same constant skips the missingness family (`early_b:307`).
27. [A] Same constant skips the ratio, log-ratio, grouped-delta and lagged-diff families (`early_b:513`).
    - Stream (24-28): materialise column blocks, not the whole frame. Signal: H. Global statistic: `local_mi_gate`'s null-max (see 114).
28. [A] Same constant skips binned numeric agg (`early_b:713`).
29. [A, low] `_RANK_BUF_MAX_BYTES = 2 GiB` (`F/_mrmr_fit_impl/_eng_dedup_scan.py:20,149`). Comment: keep/drop is unchanged, only batching coverage shrinks. Size from H.
30. [A] `_COLLINEARITY_PASS_MAX_COLS=4000` (`F/_mrmr_degenerate.py:181`, applied ~244). The whole collinearity pass is skipped above it.
31. [A] `_COLLINEARITY_PASS_MAX_BYTES=2 GiB` (`_mrmr_degenerate.py:183`, ~250). Same pass, same effect; can change `collinear_with` drops. Stream (30-31): block-pair Gram or correlation scans, taking the first matching earlier column. Exact.

## 3. Screening, redundancy and retention

32. [C] `sis_screen_threshold=20000` (class:1220). Survivor selection in `F/_mrmr_sis_screen.py:143-194` uses the median+MAD knee and `max(20*k_target, 1000)`; already sized from free RAM by `_ram_cap_survivors`, but the `max(1000, ...)` floor can exceed the RAM budget. Global statistic: median and MAD over all p fused scores, exact from the p-vector.
33. [A] `_SIS_DEDUP_MAX_ROWS=200000` (`_mrmr_sis_screen.py:187`). Comment: a full gather is "16 GB at n=1M". Pearson is additive over row blocks, so stream it.
34. [A] `friend_graph_max_nodes=200` (class:1314, `F/friend_graph.py:342,383`). Above 200 selected features the O(k^2) edge pass is skipped entirely. Off by default, but `friend_graph_prune` forces it. Stream in pair blocks.
35. [A] `cluster_aggregate_max_candidates=200` (class:1372, `F/_cluster_aggregate.py:375`). Class comment: "O(m^2) cost guard". The pool is already relevance-sorted.
36. [C] `cluster_aggregate_max_cluster_size=12` (class:1365, `_cluster_aggregate.py:421`). Limits cluster members and so the PC1 definition.
37. [C] `dcd_max_cluster_size=12` (class:1435, `F/_dynamic_cluster_discovery/__init__.py:172,624`). Same reasoning as 36.
38. [A, low] `dcd_pairwise_cache_max=50000` with `_DCD_PAIRWISE_CACHE_HARD_CAP` (class:1433, `_dynamic_cluster_discovery/__init__.py:170,217,407`). LRU cache of pair SU values; only costs recompute, no selection effect.
39. [B] `fe_raw_retention_max_n=20000` (class:603, `F/_mrmr_fit_impl/_assign_support_tail.py:719`). Behaviour switch by n, justified statistically.
40. [A, low] `fe_rejection_ledger_cap=None` (500000) (class:609, `F/_fe_rejection_ledger.py:152,221`). Bounds diagnostics memory only; spill to disk if needed.
41. [B] The p>=n false-positive cap `max(20, p//3)` (`F/_mrmr_fit_impl/_assign_support.py:297-328`, `F/_mrmr_fit_impl/_helpers.py:481`). Statistical.
42. [A] `max_base_features=14` for pure-form retention (`F/_fe_pure_form_retention.py:89,107`). Half the base is chosen by MI and half by standard deviation.
43. [A] `max_rows=3000` for pure-form retention (`_fe_pure_form_retention.py:90,629`). Row subsample of the retention pool.
44. [B] `max_added=4` (`_fe_pure_form_retention.py:351`). Output budget.
45. [A] The retention pool search uses `max_pairs=10` (`_fe_pure_form_retention.py:223`) with `rank_pairs_by_joint_mi=True`: a hard top-10 cut.
46. [C] `MAX_STABILITY_REPLAY_ROWS=4000` (`F/_mrmr_fit_impl/_helpers.py:263,310`). Row subsample for the stability replay.
47. [A] `MLFRAME_CMI_NULL_MAX_ROWS=100000` (`F/_fe_raw_redundancy_helpers.py:35,259`). Row stride on the null.
48. [A] `MLFRAME_RETENTION_NULL_MAX_ROWS=250000` (`F/_fe_retention_subsumption.py:161`). Same pattern as 47.
49. [A] `max_pairs=60` in `F/_usability_aware_selection.py:162,323,328`. Comment: "hard top-max_pairs cut". Off by default. The batch GPU rank at ~315 stacks all pairs at once; chunk it.
50. [C] `max_per_pair=12` (`_usability_aware_selection.py:160,411,477`). Limits the forms kept per pair, a quality knob.
51. [A] `usability_max_base_features=16` (class:3130, `F/_usability_lists.py:126`). Bounds the base so the pair space stays small.

## 4. Residual-rescue family and row-sampled scans

52. [A] `MAX_POOL=300` in `F/_residual_rescue_fe.py:32,274-276`. The pool is cut to the highest marginal variation of the residual each round. The Bonferroni z uses the uncapped `len(pool_names)`, so the cut does not change multiplicity. Stream: `_score_columns` in column blocks, then the argmax across blocks. The pool dict of full-length columns and `Qp` are held at once. Signal: H.
53. [B] `fe_residual_rescue_max_admit=8` (class:2972, `_residual_rescue_fe.py:231,270`). Cap on admission rounds.
54. [A] `DEFAULT_SCAN_ROWS=50000` (`_residual_rescue_fe.py:33`, stage `F/_mrmr_fit_impl/_residual_rescue_stage.py:309-311`). The stage subsamples `X.iloc[rows]` before any column conversion.
55. [A] `MAX_COLS=40` in `F/_pair_residual_screen.py:38,123-125` (`pair_interaction_screen`). Keeps the top 40 columns by variation in the additive model; docstring: "a few tens of milliseconds". Global statistic: Bonferroni `z = isf(alpha/len(pairs))`, closed form from the pair count; lifting the cap enlarges the family, thresholds rise, results are not bit-identical. All-column codes `Q` and `Qm` (p x n int64) are held at once.
56. [A] `DEFAULT_SCAN_ROWS=50000` in `_pair_residual_screen.py:39,108`.
57. [A] `MAX_PAIRS=3` in `F/_oof_cell2d_fe.py:35,122`. Bounds the extra cross-validated candidates.
58. [A] `_oof_cell2d_fe.py:36` has `DEFAULT_SCAN_ROWS=100000`.
59. [B] `fe_target_mix_max_bases=2` (class:2980, `F/_target_mix_fe.py:52,82`). Number of mixture bases, defines the feature.
60. [A] `fe_target_mix_scan_rows=20000` (class:2981, `_target_mix_fe.py:108`; `F/_target_mix_gate.py:26,33` uses 50000).
61. [A] `fe_offset_product_max_pair_cols=6` (class:2943, `F/_offset_product_fe.py:245,266`), applied through `_top_mi_num_cols`.
62. [A] `fe_offset_product_scan_rows=100000` (class:2945, `_offset_product_fe.py:46,247`).
63. [A] `fe_oof_warp_scan_rows=100000` (class:2953, `F/_oof_warp_fe.py:41,67`).
64. [A] `MAX_POOL=16` in `F/_row_stat_fe.py:37,97`. The pool is the top 16 columns by MI; the search then scores all pairs times 7 statistics with a held-out odd-row test.
65. [A] `fe_row_stat_scan_rows=20000` (class:2961, `_row_stat_fe.py:41`). Comment: "a pair table and a growth round are n * k per candidate".
66. [C] `MAX_SUBSET=8` (`F/_numba_utils_offset.py:15`). Limits the subset growth, part of the feature definition.

## 5. Discrete-structure families

67. [A] `cheap_row_argmax_scan(max_triples=40)` (`F/_conditional_gate_fe.py:430,464`). Scans only the first 40 triples of a name-sorted `combinations`. Comment: C(p,3) "far exceeds the budget ... the budget truncates the scan". No global statistic, only a per-triple operand floor and a 12-permutation null.
68. [A] `fe_row_argmax_max_cols=30` (class:2790, `_conditional_gate_fe.py:851,867`). Whole sweep skipped above 30 columns.
69. [A] `k_gate=8` and `k_operand=10` (`_conditional_gate_fe.py:698-699,551,603`). Top-k by raw MI; comment: makes the sweep "FLAT in p".
70. [A] `fe_conditional_gate_max_cols=200` (class:2808, `_conditional_gate_fe.py:896,916`). Skips the whole sweep.
71. [A, low] `_conditional_gate_fe.py:~535` builds `np.column_stack` over all candidate columns. Chunk by column.
72. [A] `cheap_modular_scan(max_pairs=24)` (`F/_pairwise_modular_fe.py:311,384-390`). Takes the first 24 pairs of `combinations` in input column order (unlike the argmax scan, which sorts names).
73. [A] `max_triples=12` (`_pairwise_modular_fe.py:312,393-398`). Same prefix cut for triples.
74. [A] `fe_pairwise_modular_max_int_cols=30` (class:2762, `_pairwise_modular_fe.py:535,553`). Skips the whole sweep.
75. [A] `fe_pairwise_modular_max_triple_cols=20` (class:2763, `_pairwise_modular_fe.py:536,561`). Drops the triple sweep.
76. [A] `fe_integer_lattice_max_int_cols=30` (class:2780, `F/_integer_lattice_fe.py:388,405`). Skips the sweep.
77. [A] `structure_discovery(max_int_cols=30)` (`structure_discovery.py:284,196,222`). Skips the modular and lattice scans.
78. [B] `structure_discovery(top_k=20)` (`structure_discovery.py:285,384`).

## 6. Univariate basis families

79. [A] `fe_wavelet_max_cols=100` (class:3015, `F/_wavelet_basis_fe_recipes.py:67,123`). Only the first 100 columns in frame order get wavelet scale selection, the rest get none. Docstring cites wall time (~26% of pre-categorize wall at p=420).
80. [B] `fe_wavelet_max_legs=6` (class:3018, `_wavelet_basis_fe_recipes.py:62`).
81. [A] `fe_univariate_fourier_adaptive_max_cols=100` (class:1563, `F/_orthogonal_univariate_fe/_orth_extra_basis_fe_generate.py:72,147`). Only the first 100 columns, in `cols` order, get adaptive and chirp detection.
82. [A] `MLFRAME_FOURIER_DETECT_MAX_N=200000` (`F/_fourier_detect_cap.py:15`, `_orth_extra_basis_fe.py:319-327`, `_fourier_detect_gpu_resident.py:253`). Docstring: exists "so a wide / 1M-row fit does not OOM building per-column coarse-grid planes". Stream per column and per row block. Signal: V on the GPU path, otherwise H.
83. [C] `fe_hinge_max_breakpoints=2` (class:1603, `F/_hinge_basis_fe.py:368`). Model-complexity limit.

## 7. Orthogonal families

84. [A] `top_pair_seed_k=4` (`F/_orthogonal_univariate_fe/_orth_pair_cross_fe.py:391-420,434`). The seed pool is the top 4 univariate-winner sources, so the pair pool is C(4,2)=6. Docstring: "bounded cost". Stream: all pairs in blocks. Global statistic: `_pair_abs_floor` uses the max baseline MI, a running max.
85. [B] `top_pair_count=3` (`_orth_pair_cross_fe.py:433,517`).
86. [A] `top_triplet_seed_k=4` (`F/_orthogonal_triplet_fe.py:608-650,664`). Pool seeded by the strongest raw column plus the rest ranked by CMI given it. Comment: a pure raw-MI cut drops XOR legs.
87. [A] `TRIPLET_ENUM_MAX=200000` (`_orthogonal_triplet_fe.py:488`). Enumeration is exhaustive only when C(p,3) <= 200000.
88. [B] `TRIPLET_ENUM_TOP_M=10` (`_orthogonal_triplet_fe.py:489,558`). Number of triples carried to the gate.
89. [B] `top_triplet_count=2` (`_orthogonal_triplet_fe.py:663`).
90. [A] `top_quadruplet_seed_k` (`F/_orthogonal_quadruplet_fe.py:415,436`). When `cols` is given, the seeds are the first k in caller order. Same XOR caveat.
91. [A] Adaptive-arity `seed_k=4` and `max_arity=3` (`F/_orthogonal_adaptive_arity_fe.py:518-520,574,595`). Same shape as 90, with positional seeds when `cols` is given.
92. [A] `max_pairs=200` in `F/_orthogonal_diff_basis_fe.py:125,191`. Docstring: "Cap ... to bound the downstream MI batch size". Pairs sorted by |corr|.
93. [A] `detect_correlated_pairs` builds a dense p x p correlation matrix at once (`_orthogonal_diff_basis_fe.py:~170`, `cached_dense_finite_corr_matrix` in `F/_orthogonal_univariate_fe/_orth_scoring_memo.py:107`). Block it. Exact.
94. [C] `max_cluster_size=20` (`F/_orthogonal_cluster_basis_fe.py:121,205`).
95. [C] `_OOF_BATCH_BINNING_MAX_TRAIN_ROWS=16000` (`F/_orthogonal_three_gate_mi_fe.py:99,227`). Switches between batched and per-fold binning.
96. [C] `fe_hybrid_orth_dcor_n_sample=500` and `fe_hybrid_orth_hsic_n_sample=500` (class:2260,2276). Subsamples for O(n^2) statistics; off by default.

## 8. Extra-FE families

97. [A] `fe_conditional_residual_max_pair_cols=6` (class:2844, `F/_extra_fe_families.py:585-605,614,642-644`). Docstring: "the cardinality bound that keeps the O(p^2) conditional-residual pair pool tractable". The top-MI column choice (`_top_mi_num_cols`) is already chunked. Stream: all column pairs in blocks with a chunked `enc_df`, gated by `local_mi_gate` (see 114).
98. [A] `fe_conditional_dispersion_max_pair_cols=6` (class:2862, `F/_extra_fe_families_dispersion.py:597,625`).
99. [A] `fe_conditional_quantile_rank_max_pair_cols=6` (class:2884, `F/_conditional_quantile_rank_fe.py:246,274`).
100. [A] `fe_ordinal_pattern_max_cols_for_tuples=5` (class:2893, `F/_ordinal_pattern_fe.py:279,308`). "mirrors the existing triplet/quadruplet arity-cap cost guard". Off by default.
101. [C] `fe_random_fourier_max_cols_for_block=8` (class:2905, `F/_random_fourier_features_fe.py:238,266`). Block width is part of the feature definition.
102. [C] `fe_sir_direction_max_cols_for_block=8` (class:2916, `F/_sliced_inverse_regression_fe.py:247,276`). Same.
103. [C] `fe_lof_max_cols_for_block=8` (class:2927, `F/_lof_fe.py:230,257`). Same.
104. [C] `fe_mahalanobis_density_max_cols_for_block=20` (class:2935, `F/_mahalanobis_density_fe.py:172,199`). Same.
105. [C] `fe_lof_max_ref=2000` (class:2926, `_lof_fe.py:142,156`). The recipe stores a bounded reference set ("RAM discipline"), so the cap is part of the stored model.
106. [A] `_lof_transform` builds a full (n_new x ref) distance matrix (`_lof_fe.py:126-128`) with no row batching. Not a cap but an unbatched allocation, and it also runs at transform time. Batch over rows from H.
107. [A] `_auto_detect_cat_cols(max_cols=8)` for rare-category (`_extra_fe_families.py:287-304`). Takes the first 8 in detector or frame order, with no relevance ranking.
108. [A] `auto_detect_cat_pair_cols(max_cols=8)` (`F/_cat_pair_fe.py:339,366`). Same pattern.
109. [A] `auto_detect_num_cols_plain` and `auto_detect_num_cols_skip_grp` with `max_cols=8` (`F/_grouped_coerce_shared.py:37,54,57,77`). The cut is applied before `_filter_num_cols_by_relevance` (`F/_grouped_agg_fe.py:454,530`), so a strong column outside the first 8 is never scored.
110. [A] `auto_detect_group_cols(max_cols=4)` (`_grouped_coerce_shared.py:101,115,127`).
111. [A] Composite `_auto_detect_group_cols(max_cols=6)` (`F/_composite_group_agg_fe.py:520,540`).
112. [A] Composite `auto_detect_key_sets(max_sets=12)` (`_composite_group_agg_fe.py:548,573`). The `_key_cache` holds keys for every combination.
113. [B] `fe_rankgauss_top_k=10` (class:3035, `_extra_fe_families.py:887,908-911`). Comment: RankGauss is monotone and cannot add MI, so the family is "NOT MI-gated"; the cap is the only selection.
114. [A] `local_mi_gate` casts the whole candidate matrix at once (`F/_unified_fe_gate.py:360`) and scores it in one batch (`:379`). Comment at :362-368 says the host upload "is irreducible".
    - Global statistic: `floor = max(raw median+MAD floor, q95 of the per-permutation max MI over the candidate set)` (`:370-380`). It decomposes exactly: the per-permutation max is the elementwise max of per-chunk maxima, with the same permutations from `local_gate_permutations` and the same target; the quantile is taken afterwards.
    - Stream: candidate-column chunks. Signal: H, or V on the resident path.
115. [A] `raw_mi_noise_floor` does `np.column_stack` of all raw numeric columns (`_unified_fe_gate.py:233`). The median+MAD needs only the full p-vector of per-column MIs (small). Compute it in column blocks.
116. [A] The dispersion resident twin (`F/_extra_fe_families_dispersion_resident.py:268-345`) builds the whole `mat_gpu` on device. It also appears to compute only the raw-baseline floor and not the null-max floor, so its result may differ from the host `local_mi_gate`. Downstream effect not verified.
117. [A] `unified_second_pass_gate` pre-bins every engineered column into a dict (`_unified_fe_gate.py:471`). `fe_unified_second_pass_max_keep=None` is only an output cap. The greedy needs repeated passes, so recompute bins per round or spill.
118. [B] `fe_local_mi_gate_top_k=20` (class:2620) feeds `select_above_floor` (`_unified_fe_gate.py:292-321`). Deliberate top-k after the floor.

## 9. Categorical interaction FE

119. [B] `CatFEConfig.top_k_pairs=32` (`F/cat_fe_state.py:48`). Uses argpartition over all pairs, then keeps the top K. Docstring: "Memory linear in this value".
120. [B] `max_combined_nbins=None`, the Paninski ceiling `max(4, n*0.05/3+1)` (`F/cat_interactions.py:121-135`). Statistical cap.
121. [A] `hard_cap=10_000_000` (`cat_interactions.py:121`). A constant OOM guard that does not use free RAM.
122. [A, low] `enumerate_candidate_pairs` uses `np.triu_indices` (`F/_cat_interactions_step.py:81-105`), creating several int64 arrays of C(p,2) at once. Block it.
123. [A] The full Westfall-Young gate `use_full_wy` (`F/_cat_interactions_step_stages.py:258`; allocation `F/_cat_confirm_fwer.py:79`). Condition `len(pairs)*n*4 < 500 MiB`, otherwise it falls back to Bonferroni on survivors. `pair_classes_buf = np.empty((m, n))` pre-merges every pair; comment: "If memory is a concern, fall back".
    - Stream: pair blocks (merge, then score under all shuffles). Signal: H.
    - Global statistic: `max_ii_per_perm[b]` is the max II over all m pairs per shuffle; decomposes exactly as the elementwise max over pair blocks. The shuffles must be the same: replay `wy_rng` from `base_seed` per block or store the (n_perm, n) shuffle matrix. Default `fwer_correction` is "none", so this path is opt-in.
124. [A] `permutation_subsample=None` (`cat_fe_state.py:152`, `F/_cat_confirm_permutation.py:611-629`). Docstring cites ~25 s versus ~2.5 s.
125. [B] `beam_width=3` and `top_k_pairs=3` for triples (`F/_cat_triple_fe.py:280-281`). Evaluation bounded by `n_rounds*max(...)*(p-2)` instead of C(p,3); a heuristic by design (the parity-trap comment).
126. [A, low] `n_exhaustive = len(list(combinations(cat_cols, 3)))` (`_cat_triple_fe.py:407`) materialises the combinations just to count them. Use `math.comb`.

## 10. Family-wise maxT gates

127. [A] `family_pass_batches` stacks every config batch of one pair into a single (K_total, n) matrix (`F/_family_maxt_gate.py:609-634`, `_stack_codes_transposed` at :575-590; called from `F/_feature_engineering_pairs/_pairs_emit.py:108`). Docstring: an earlier version "exhausted host memory on a wide frame"; the fix only narrowed the dtype.
    - Stream: column batches of the family. Signal: H, or V on `family_pass_batches_device`.
    - Global statistic: the null is the max MI over the whole family per permutation, so it decomposes exactly as the elementwise max over batches with the same permutation block. The analytic triage's Bonferroni p sums per-column tails grouped by degrees of freedom, so it is additive across chunks. The early rejection rule uses the median of the first `FAMILY_PROBE_BLOCK` null maxima and needs those first-block maxima across all chunks before deciding, so it is two-pass.
128. [A] `F/_periodic_fe.py:441-472,496-521` streams candidate generation, but the family gate (`family_maxt_pass_codes`) stacks all int16 codes. Same decomposition as 127.
129. [C] `_PERM_BLOCK_CACHE_BYTES=256 MiB` (`_family_maxt_gate.py:183`). A cache only, no effect on selection.

## 11. Binned numeric agg

130. [A] `fe_binned_numeric_agg_max_pairs=64` (class:2508). `F/_mrmr_fit_impl/_fe_stage_cascade_early_b.py:731` always passes an int. `F/_binned_numeric_agg_fe.py:625,671` now supports `None` ("every pair is scored, in memory-bounded batches"); the batched path exists but MRMR never selects it. Global statistic: `_family_wise_null_z(n_candidates)` (:526) is a Bonferroni z, closed form from the total count.
131. [A] `max_group_cols=16` (`_binned_numeric_agg_fe.py:626,655`). Group columns are ranked by cheap MI, so this is a hard top-16 cut that lets MI-dependent ranking decide.
132. [A] `max_agg_cols=16` (`_binned_numeric_agg_fe.py:626,541`). Ranked by variance, so unsupervised.
133. [A] `_binned_numeric_agg_fe.py:682` has the `[:_pair_cap]` post-cap as a "no-op safety net" under 130. It goes away if 130 is lifted.

## 12. Seeders and tree rescue

134. [A] `_GRAD_DEFAULT_MAX_P=200` (`F/_gradient_interaction_seeder.py:75,127-130`). Above 200 pool columns the seeder is routed off. Comment: "the O(p^2) energy tally + null is the cost wall". Off by default.
135. [C] `_GRAD_DEFAULT_ROW_CAP=2000` (`_gradient_interaction_seeder.py:73,302`, finite-difference `max_rows=2000` at :247). The surrogate is fit on 2000 rows.
136. [B] `fe_gbm_seeder_top_k_pairs=12` and `top_k_triples=8` (class:1919-1920, `F/_surrogate_interaction_seeder.py:289-290,405,408`), and `_GRAD_DEFAULT_TOPK=8` (`_gradient_interaction_seeder.py:~88,464`). Proposal budgets.
137. [B] `tree_rescue_top_k=20` (`F/_mrmr_tree_rescue.py:59,165`).

## 13. GPU memory limits

138. [C] The grand-fused pair-MI path uploads a (nperm+1, n) shuffle matrix (`F/_gpu_resident_pair_mi_parts.py:179-190`) and needs `P1*nbins*K_y*4 <= sharedMemPerBlock` (`F/_gpu_resident_pair_mi.py:531-535`); otherwise it falls back to the non-fused path. With `fe_npermutations=99` it can be tight. A permutation-chunked version is exact: the null max is a max over candidates per permutation, so chunks cover disjoint permutations.
139. [A] The external-validation matrices `cp.empty((n, n_ext*n_ops), float64)` (`F/_gpu_resident_extval.py:76`) and `np.empty((n, _ev_K))` (`F/_feature_engineering_pairs/_pairs_emit_extval.py:31`). Unbatched over external factors; `fe_max_external_validation_factors=0` by default, so the path is off. Batch over factors from V or H.

## 14. Row subsampling for binning and nulls

140. [A] The CUDA row-chunked discretiser takes quantile edges from a `quantile_subsample_rows=UNIFIED_FE_SUBSAMPLE_N=30000` subsample (`F/discretization/_discretization_cuda.py:185-225`). Docstring: exact quantiles are "NOT reducible across row-chunks without a streaming quantile algorithm". Global statistic: per-column bin edges; exact edges need a multi-pass histogram refinement or the radix select that already exists in `F/_gpu_resident_select.py`. Column batching is exact.
141. [A] `MLFRAME_FE_NOISE_FLOOR_MAX_ROWS=30000` (`F/_hermite_fe_optimise_pair.py:202-204`). Row stride used for a noise-floor estimate.

## 15. Hybrid selector wrapper

142. [A] `mrmr_synergy_cap=250` (`hybrid_selector.py:178,207,357`). Forwards to item 11.
143. [A] `hybrid_corr_max_features=2000` (`hybrid_selector.py:179,208-217,729-745`). Comment: the blocked `corr_clusters` path already keeps memory at O(n*p). The cap drops low-importance raw columns (FI 0, not MRMR-relevant, not engineered) from clustering, so it is only partly needed.
144. [B] `tree_cooccur_pairs=12` (`hybrid_selector.py:174,424`).

## 16. Fixed byte budgets (already batched, not sized from free memory)

145. [C] `_FE_BUFFER_ABSOLUTE_MAX_GB=8` (`F/_feature_engineering_mem_budget.py:222`). Effective budget `min(0.3*usable/(3*workers), 8 GiB)`. Whether to lift the 8 GiB ceiling when much more RAM is free is a judgement call.
146. [C] `_GPU_BATCH_BYTES=768 MiB` (`F/_usability_njit_pool.py:255`).
147. [C] `_BUDGET_CAP_BYTES=768 MiB` (`F/_batch_mi_noise_gate_bits.py:32`).
148. [C] `BITSET_BUDGET_BYTES=128 MiB` (`F/_family_maxt_null_v2_njit.py:190`).
149. [C] `HOST_PERM_BUDGET_BYTES=256 MiB` (`F/_gpu_host_permutations.py:12`).
150. [C] `BLOCK_BYTES_MAX=512 MiB` (`F/_orthogonal_univariate_fe/_streamed_gpu_mi.py:30`). Comment: "whatever the VRAM".
151. [C] `batch_pair_mi_gpu`: `cap = min(1.5 GB, 0.5*free)` (`F/batch_pair_mi_gpu.py:338-357`).
152. [C] `_MODULAR_CHUNK_BYTES=256 MiB` (`F/_periodic_fe.py:293`).
153. [C] `_HONEST_NULL_MEMO_BYTES=128 MiB` (`F/_mrmr_fit_impl/_fit_impl_stages/_binning.py:16`).
154. [C] `_CMI_FORDER_MAX_MB=4096` (`F/info_theory/_cmi_cuda.py:84`).

## 17. Wall-clock budgets

155. [C] `max_runtime_mins` (class:530, `F/_mrmr_fit_impl/_finalise.py:524`) and `F/_fe_deadline.py`. They stop optional generators partway.
156. [C] `_DEFAULT_EXHAUSTIVE_BUDGET_SECONDS=300` and `fe_synergy_exhaustive_max_seconds` (`F/_fe_synergy_exhaustive.py:324,327-430`). When the predicted wall time exceeds the budget, the exhaustive sweep falls back to the pre-rank.
157. [C] `reallocate_budgets` (`F/_fe_family_budget.py:156`) reassigns per-family budgets by return on investment.

## 18. Final survivor top-k after a gate (deliberate output budgets), all [B]

158. `fe_grouped_agg_top_k=10`, `F/_grouped_agg_fe.py:496,550`.
159. `fe_grouped_quantile_top_k=8`, `F/_grouped_quantile_fe.py:610,664`.
160. `fe_composite_group_agg_top_k=10`, `F/_composite_group_agg_fe.py:584,642`.
161. `fe_group_distance_top_k=6`, `F/_group_distance_fe.py:400,446`.
162. `fe_temporal_agg_top_k=10`, `F/_temporal_agg_fe.py:736,797`.
163. `fe_numeric_decompose_top_k=5`, `F/_numeric_decompose_fe.py:265,354`.
164. `fe_modular_top_k=6`, `F/_periodic_fe.py:313,409,482,524`.
165. `F/_parity_fe.py:121,145`, 4.
166. `fe_hinge_top_k=5`, `F/_hinge_basis_fe.py:833,906`.
167. `fe_wavelet_top_k=8`, `F/_wavelet_basis_fe_recipes.py:241,358`.
168. `fe_missingness_pattern_top_k=5`, `F/_missingness_fe.py:280,321`. Defines the pattern labels, the rest map to an "other" bucket.
169. `fe_mi_greedy_top_k=5` and `fe_mi_greedy_seed_cols_count=5`, `F/_mi_greedy_fe.py:520-521,562,508`. The seed count is an A-flavoured cap on the pool.
170. `fe_mi_greedy_cmi_top_k=5` and seed 4, `F/_mi_greedy_cmi_fe.py:633-634,757`.
171. `fe_cat_pair_top_k=5`, `_cat_pair_fe.py:436,480`.
172. `fe_cat_triple_top_k=3`, `_cat_triple_fe.py:422,471`.
173. `fe_row_stat_top_k=2`, `_row_stat_fe.py:200,227`.
174. `fe_oof_warp_top_k=5`, `_oof_warp_fe.py:129,159`.
175. `fe_offset_product_top_k=3`, `_offset_product_fe.py:246,325`.
176. `fe_ordinal_pattern_top_k=5`, `_ordinal_pattern_fe.py:282`.
177. `fe_random_fourier_top_k=8`, `_random_fourier_features_fe.py:239`.
178. `fe_sir_direction_top_k=2`, `_sliced_inverse_regression_fe.py:248`.
179. `fe_lof_top_k=1`, `_lof_fe.py:231`.
180. `fe_mahalanobis_density_top_k=1`, `_mahalanobis_density_fe.py:173`.
181. `fe_conditional_gate_top_k=4`, `_conditional_gate_fe.py:895`.
182. `fe_row_argmax_top_k=4`, `_conditional_gate_fe.py:813-823`.
183. `fe_integer_lattice_top_k=4`, `_integer_lattice_fe.py:318,335`.
184. `fe_pairwise_modular_top_k=4`, `_pairwise_modular_fe.py:444,464`.
185. `fe_rare_category_top_k=10`, `_extra_fe_families.py:312,353`.
186. `fe_conditional_residual_top_k=10`, `_extra_fe_families.py:613,663`.
187. `fe_conditional_dispersion_top_k=10`, `_extra_fe_families_dispersion.py:596,680`.
188. `fe_conditional_quantile_rank_top_k=10`, `_conditional_quantile_rank_fe.py:245,291`.
189. `fe_hybrid_orth_top_k=5`, and the orth routing, diff-basis and cluster-basis `top_k` of 3 to 5: `_orthogonal_univariate_fe/__init__.py:588,677`, `_orthogonal_routing_fe.py:342,441`, `_orthogonal_diff_basis_fe.py:354,442`, `_orthogonal_cluster_basis_fe.py:538,620`.
190. `fe_multi_emit_max_per_pair=1` (class:1824). A candidate-buffer cost cap, off at 1.

## Global statistics calibrated on a whole candidate set

| Statistic | Where | Decomposes exactly? |
|---|---|---|
| `local_mi_gate` floor, the q95 of the per-permutation max MI | `_unified_fe_gate.py:370-380` | Yes: elementwise max of chunk maxima with identical permutations. |
| Order-2 maxT pair floor | `F/_permutation_null.py` via `compute_pair_maxt_floor` | Yes: same, per pair block. The family size depends on the pool, so changing the cap changes the floor by design. |
| Family maxT with the analytic chi-square triage | `_family_maxt_gate.py` | Max and Bonferroni sums are additive. The early-reject probe is two-pass. |
| Westfall-Young max II over pairs | `_cat_confirm_fwer.py:107-135` | Yes: elementwise max over pair blocks with replayed shuffles. |
| Bonferroni z from the family size | `_pair_residual_screen.py`, `_residual_rescue_fe.py`, `_target_mix_gate.py`, `_binned_numeric_agg_fe.py:526` | Closed form from the total count, so streaming is exact. Enlarging the family raises thresholds, so results differ from the capped run. |
| Median+MAD over all raw columns' MI | `raw_mi_noise_floor`, `survivor_count` | Exact from the small p-vector. |
| Bin edges (quantiles) | `_discretization_cuda.py` | Not additive over row chunks without a sketch or multi-pass histogram. Column batching is exact. |
| Greedy CMI and the unified second pass | `_fe_cmi_redundancy_gate.py`, `_unified_fe_gate.py:489-506` | Sequential and not decomposable. Per-round candidate scoring can be chunked. |

## Counts by category (agent's tally, 190 findings)

- A (memory or runtime guard, streaming strictly better): 107 upper bound, 103 firm (the 103 A-tagged entries in items 1-157 plus 4 tagged B in the survivor top-k list that are really pool caps: the mi_greedy seed counts and the two `top_k_pairs` seeds).
- B (statistical or output-budget decision, must stay): 55 firm (21 in items 1-157 and 34 in the survivor top-k block).
- C (unclear, feature definition, hardware limit, time budget, fixed-byte batch ceiling): 32.

Best streaming targets named by the agent: 1, 4, 10, 11, 13, 55, 64, 79, 81, 92, 114, 123, 127, 130-132, 140. Items 123, 127 and 114 carry the global-statistic exactness argument.

Already memory-safe (not counted): pair-MI dispatch `dispatch_batch_pair_mi_chunked` and `_gpu_upload_fits` (chunked, sized from free VRAM); SIS block width (chunked, sized from free RAM); `_periodic_fe` generation (streamed); chunk-buffer widths in `_pairs_core_helpers` (sized from free RAM).

## Disposition (filled in as each item is triaged)

| Item | Disposition |
|---|---|
| 114 (`local_mi_gate` candidate matrix and null) | Implemented 2026-10-22: the null is binned once per matrix (`_orth_mi_null_max.py`, CPU rank, STRICT GPU and streamed GPU routes) and the binned-aggregate host path scores candidates through `StreamingLocalGate` in memory-bounded batches (`_binned_numeric_agg_batches.py`). |
| 130 (binned numeric agg pair cap) | Library side done: `max_pairs=None` scores every pair in batches. MRMR still passes 64; switching the MRMR default is open. |
| 19 (fusion half codes matrix) | Implemented 2026-10-23: the (n, H) int64 code matrix is scored in column blocks sized from free memory (`_collect_relevant_halves`); the MI of a half is independent of the others, so the result is unchanged. |
| 106 (`_lof_transform` distance matrix) | Implemented 2026-10-23: `X_new` is scored in row blocks sized from free RAM (`_lof_row_block`); rows are independent. Test: `test_lof_transform_row_blocks.py`. |
| 114 (`local_mi_gate` candidate matrix) | Implemented 2026-10-22/23: the gate itself scores column blocks (`StreamingLocalGate`, `column_blocks`) when the candidate matrix does not fit; identical survivors, order and rejection records (tests in `test_binned_agg_batches.py`). |
| 115 (`raw_mi_noise_floor` column_stack) | Implemented 2026-10-23: the raw MI vector is scored in column blocks; the floor arithmetic is shared (`_floor_from_raw_mi`). |
| 126 (`list(combinations)` only to count) | Implemented 2026-10-23: `math.comb`. |
| 93 (dense p x p correlation matrix in `detect_correlated_pairs`) | Deferred: the matrix is the shared `cached_dense_finite_corr_matrix` memo used by other callers; a block-wise top-k needs the same finite-row semantics for every caller. |
| 130 (MRMR binagg pair cap) | MRMR now passes `fe_binned_numeric_agg_max_pairs=None` through (score every pair, batched); the default stays 64 until the effect on the selection and the wall time of a default fit is measured. |
| 121 (`hard_cap=10_000_000` in `resolve_max_combined_nbins`) | Not a reduction of relevant data: it clamps an explicit misconfiguration (`max_combined_nbins=10**9`); the data-aware default governs. No change. |
| 122 (`np.triu_indices` of C(p,2)) | Low: only reaches gigabytes at tens of thousands of categorical candidates; block it when a frame that wide appears. |
| 127 (`family_pass_batches` stacks the whole family) | Implemented 2026-10-23 (`_family_maxt_stream.py`): a family whose stacked code matrix does not fit 25% of the free host RAM is scored one group of columns at a time. Observed MI and the occupied-bin counts of the analytic triage are per column; the null maximum is the elementwise maximum over the groups' maxima, every group scored against the same permutation draw (the generator is rewound per group and ends where one pass leaves it); the early rejection and early stop read the whole family on every call, so they are two-pass by construction. Nulls are equal element for element and the generator ends in the same state (`test_family_maxt_streamed.py`); a family that fits keeps the single pass and its column deduplication. Not covered: the pair emit step still accumulates the family's input batches before the gate, which the upstream chunk widths already size from free RAM. |
| 30, 31 (collinearity pass skipped above 4000 columns / 2 GiB) | Implemented 2026-10-23: above the Gram budget the pass streams (`_collinear_hits_streaming`: random projections for candidates, exact dot products to confirm; a few vectors of memory), same verdicts as the Gram pass; `test_degenerate_collinearity_streaming.py`. Also removes the cost that made the cap necessary. |
| all others | Not yet triaged. |
