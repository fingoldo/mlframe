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

Standing condition for every item in this table: each new or changed path is profiled (cProfile, per-kernel timing on the wide case) and its hotspots are optimized before the item is marked done. A timing sentence in the row is not enough; the profile and the hotspot fix are part of the item.

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
| 23 (n-way synergy screen runs only for 2 to 60 raw candidates) | Open, reclassified as a runtime budget, not a memory cap: the combinations grow as C(p, order) (C(60,3) = 34,220; C(250,3) = 2,562,000), and each combo pays a joint-cell count, so streaming the combos (item 20) does not remove the cost. The cap needs a time budget, or a pre-rank that keeps the most promising combos, decided with a measured wall time; the silent skip is now a logged warning (`_group1.py`, `_synergy_screen_pass`); the bound itself stays until the time budget is measured. |
| 4 (`fe_pair_usability_prescan_max_pairs=256`) | Implemented 2026-10-23: default `None` scans every pair; the dominant-pair maximum is taken over blocks sized from the free host RAM (`_prescan_block_pairs`), first-on-tie preserved. Tests: `test_prescan_block_pairs.py` and the prescan suites. Profile at 30000 rows, 2000 operands, 20000 pairs (`D:/Temp/prof_prescan.py`, the same blocks as the code): before, the scan fell back to the CPU block by block and took 9.9 s; after, the scan is one upload of the operands (GPU) and takes 0.16 s, with the GPU maximum equal to the CPU reference (max diff 6e-16). Hotspot found first: the host-to-device copies of each block (`safe_cuda_api_call`, 4.2 s of 5.1 s), fixed by uploading the scan operands once when they fit. Also changed: the dispatcher footprint now uses the operand matrix at its own dtype (`nbytes`), since the GPU path uploads it as given; it was sized at 8 bytes. Tried and rejected: a fused single-pass CPU kernel (`batch_pair_usability_corr_fused_njit`, all five forms per row, per-pair buffer): same numbers (max diff 2e-17) but 1.67 s against 1.44 s for the per-form kernel, so it was reverted; the per-form kernel is the one that stays. |
| 3 (`fe_max_pair_features=10`, no slice applies it) | Dead parameter, not a cap in effect: the round-3 benchmark (`_benchmarks/fs_hybrid/AGENT_IDEAS_ROUND3.md`, M3-3) measured 10 and 25 giving identical output. Kept as a public argument; removing it is an API change for a separate decision, and no new code path depends on it. |
| 5 (`fe_check_pairs_subsample_n=UNIFIED_FE_SUBSAMPLE_N=30000`, pair-sweep row cap) | Kept as a measured statistical choice. Measured 2026-10-24 on heavy_tailed at 100000 rows (`D:/Temp/subsample_bench.py`): the selection is the same at 30000 rows and at all rows (one compound), and the all-rows fit takes 44.6 s against 34.4 s, about 30% more. At 10000 rows the cap does not apply and the two fits are identical. The quantile edges of the sweep are computed on the subsample, so streaming to all rows would change the edges, not only the cost; that needs its own accuracy check before the cap goes. |
| 6 (`fe_smart_polynom_subsample_n=30000`, polynomial pair search row cap) | Kept, no measured cost of removing it. Measured 2026-10-24 on heavy_tailed at 100000 rows (`D:/Temp/polysub_bench.py`): the selection is the same in every run, at 30000 rows and at all rows. The wall times do not agree between repeats (all rows 32.6 s against 30000 rows 20.7 s in one pair, the reverse order in the earlier pair), so there is no stable time effect to act on. Two repeats returned in 0.1 s: a cache hit, not a fit, and excluded from the comparison. Removing the cap needs the accuracy check of the sweep quantile edges (item 5) first. |
| 7 (`MLFRAME_USABILITY_CORR_MAX_ROWS=30000`, usability |corr| row subsample) | Kept as a measured speed choice. Accuracy: the full-n run (env 10000000) passes all 9 F2 profiles, so the subsample does not change the selection there. Cost of full n, measured 2026-10-24 on the usability kernel (3000 pairs): 2 ms at 30000 rows against 8 ms at 100000 rows, and the earlier note in `_fe_usability_signal.py` measured 3.46x on the corr pass at 100k, which was about 15% of a 100k fit. The default stays 30000 because the cost is real and the selection gain from full n is not shown; the env var stays available. |
| 8 (`MLFRAME_PAIR_NULL_MAX_ROWS=250000`, pair permutation-null row cap) | Kept, no measured effect to act on. Measured 2026-10-24 on heavy_tailed at 300000 rows (`D:/Temp/pairnull_bench.py`, each setting in a fresh process): the cap at 250000 fits in 114.0 s and all rows in 110.3 s, and the selection is the same in both (the two halves). The comment cites 2.4 s at 1M rows for the null alone, which this fit does not isolate. Repeats in one process return from the fit cache in 0.1 s and are excluded. Changing the cap needs a measurement of the null step itself at 1M rows. |
| 9 (`MLFRAME_FE_GATE_MAX_ROWS=250000`, candidate gate row stride) | Kept, measured 2026-10-24 at 1000000 rows in fresh processes (`D:/Temp/gate_bench_1m.py`): the cap (stride 4, 250000 rows) fits in 229.7 s and all rows in 231.1 s, the selection is the same in both, so the cap saves under 1% on this profile. Kept because the stride rule is the cap of the gate, not a measured loss; removing it is a cost-free change only on this profile, so the stride should also be stated as a row count. Earlier note (superseded): the 300000-row measurement did not exercise the cap, because the stride is 1 below 500000 rows. The stride is `n // cap`, so at 300000 rows and a cap of 250000 it is 1 and the gate reads every row; the cap only takes effect from 500000 rows, and then the rows used are `n // (n // cap)`, not `cap`. The two fits in fresh processes (110.2 s with the cap, 110.6 s without) give the same selection, but that is the same code path. The open check is at 1000000 rows with the cap and with all rows, in fresh processes, and the stride rule itself should be stated as a row count rather than a divisor. |
| 10 (`fe_engineered_cmi_max_candidates=64`, CMI redundancy gate candidate cap) | Kept; open check on a wide pool. The cap binds only when more than 64 engineered survivors reach the gate. It did not bind in the F2 profiles (checked in the verbose logs of the heavy_tailed fit: no `dropped_cost_cap`), so there is no evidence either way on this data. The audit row called it runtime only, but it drops the tail by marginal MI (`dropped_cost_cap`), so it can change the selection on a wide pool; it is not a memory cap. The check is a pool with more than 64 survivors, with the cap and with all candidates, comparing the selection and the time. |
| 11 (`fe_synergy_screen_max_features=250`, synergy sweep pre-rank cap) | Kept, measured earlier. The round-4 benchmarks put 250 at the cost/benefit sweet spot (`_benchmarks/fs_hybrid/AGENT_IDEAS_ROUND4.md`, shipped #2; `round4_mrmr_cap_largen_cost_bench.py` measured its large-n cost), and the n-aware cost gate `fe_synergy_max_sweep_cost` (item 12) bounds the sweep where n*p^2 is large. No new code, so no new profile is claimed. The exhaustive path (`_fe_synergy_exhaustive.py`) remains the complete alternative for balanced interactions. |
| 12 (`fe_synergy_max_sweep_cost=5e8`, n*p^2 cost gate on the synergy sweep) | Kept: a runtime guard by the audit's own note, and it can be dropped only when the pairs are streamed, which is not implemented (the exhaustive sweep in `_fe_synergy_exhaustive.py` still builds all pairs). The round-4 fix made the gate n-aware for exactly this reason (`AGENT_IDEAS_ROUND4.md`, shipped). Measuring the sweep at 1000000 rows times 250 columns is hours of CPU and gives no decision until streaming exists, so no new profile is claimed. |
| 13 (`fe_synergy_max_pairs=16`, synergy pair budget by joint MI) | Kept; open check on a wide pool. It drops the pairs ranked below the top 16 by joint MI "to bound FE search cost", so it can change which synergy pair is searched on a wide frame. The rung-schedule note (`_fe_rung_schedule.py`) already treats it as a deliberate budget, and the F2 profiles have too few pairs for it to bind. The check is a wide frame with more than 16 synergy pairs, with the budget and without, comparing the selection and the time; the per-pair search cost is the open part, as the audit says. |
| 14 (`fe_rung_keep_frac`, `fe_rung_rel_floor=0.40`, `fe_rung_min_pairs=6`, rung schedule of the pair search) | Kept as a tuned search budget; open check on a wide pool. The schedule keeps the top fraction of pairs by pair-MI rank, and the fraction is chosen by the kernel-tuning cache (`get_or_tune`, `_fe_rung_schedule.py`), so it is a selection-relevant budget, not a memory cap. The relative floor protects a moderate winner; the F2 profiles have too few pairs for the schedule to cut, so they do not test it. The check is a wide frame where the schedule cuts, with the schedule and with all pairs, comparing the selection and the time. |
| 15 (`fe_escalation_max_pairs=8`, `fe_escalation_max_candidates_per_pair=3`, escalation budget for the ALS and Fourier proposers) | Kept; open check on a wide pool. The budget decides which pairs get the expensive ALS and Fourier proposers and how many candidates each keeps (`_fe_auto_escalation.py`, half the pairs reserved for rescue), so it is selection-relevant on a wide frame, not only runtime as the audit row said. The F2 profiles do not reach the escalation budget. The check is a wide frame with more pairs than the budget, with the budget and with a larger one, comparing the selection and the time. |
| 16 (escalation row stride `max_rows=20000`, `_fe_auto_escalation.py`) | Kept; open check. The stride subsamples the rows the ALS and Fourier proposers see, which the audit names as the dominant active orth-FE CPU cost. It changes the proposers' estimates, so it is selection-relevant, and no measurement of it is recorded yet. The check is the escalation at 100000 rows with the stride and at all rows, comparing the selection and the time, in fresh processes. |
| 17 (`fe_additive_fusion_max=4`, admitted-fusion output budget, host and device) | Kept: an output budget by the audit's own classification (B). The loop takes admissible pairs greedily in order of the strongest half, and stops after four admitted fusions, so a fifth genuine compound on a wide frame is not admitted. Not a memory or time cap; a larger value changes the output set. Open check: a frame with more than four genuine disjoint fusions, with 4 and with a larger value, comparing the selection. |
| 11b (`fe_synergy_screen_max_features=250`, recall check on a wide frame) | Measured 2026-10-24: a planted product of two columns at p=400, the compound with both planted columns inside the name, 6 seeds (`D:/Temp/wide_strict.py`): found 6 of 6 with the 250 cap and 6 of 6 with the cap lifted. No pair lost on this frame. A constructed case where the cap alone loses the pair was not found: decoy columns that dominate the screen also hide the pair with the cap lifted (`D:/Temp/counterexample*.py`). |
| 2b (`fe_row_argmax_max_cols=30`, the sweep was skipped above 30 eligible columns, and the triple budget `max_triples=40` took the first triples in name order) | Fixed 2026-10-24. Measured: a planted row-argmax over x0..x4 at p=40 was not found with the skip (`D:/Temp/argmax_cap.py`) and is found after the change; at p=200 the full sweep with the name-ordered triple budget found nothing, and with triples ranked by marginal MI it finds the planted structure (`D:/Temp/argmax_cost200.py`). Profile at p=200, steady state: full sweep 0.29 s against 0.28 s for the old 30-column subset, so the full sweep costs no more here. The skip message and the column subset are gone; `max_cols` now only bounds nothing and stays for compatibility. |
| 2c (`fe_univariate_fourier_adaptive_max_cols=100`, by column index) | Not a loss on the planted case: an off-grid periodic column at index 150 of p=160 gets the same engineered columns with the cap at 100 as with no cap (`D:/Temp/fourier_cap.py`). The cap sends columns beyond the index to the fixed grid only. A real difference remains between index 10 and index 150 (the `modsin`, `He2` and `oofwarp` columns appear only at index 10), but the cap does not explain it, so it stays open as a separate question about position-dependent families. |
| 2d (`fe_conditional_gate_max_cols=200`, skip of the whole gate sweep above 200 columns) | Fixed 2026-10-24. A planted gate (`where(c > 0.3, a, b)`) at p=260 was not found with the skip and is found without it (`D:/Temp/gate_cap.py`). The sweep is relevance-pruned (top-k gate and operand columns), so its cost does not grow with p: 2.3 to 3.8 s at p=150 and 3.2 s at p=600 (`D:/Temp/gate_cost.py`). The skip is removed; the budget is logged when exceeded. The stale test that pinned the skip now asserts the sweep runs and finds the gate. F2 and fusion suites: 12 passed. |
| 2e (`fe_wavelet_max_cols=100`, index-order truncation of the held-out Haar scale selection) | Measured 2026-10-24 with a valid target (a Haar-type indicator of the column values, `D:/Temp/wavelet_valid.py`): at index 150 the result is the same with the cap at 100 and lifted, so the cap is not the loss. The detector also does not produce a Haar feature at index 10 for this target, so the wavelet path misses this target at any index; that detector gap is the open question, not the cap. The code stops at column index 100 (`_wavelet_basis_fe_recipes.py`, `break` on `_col_idx >= max_cols`), the same kind of loss as the Fourier cap had; but the planted check built here (`D:/Temp/wavelet_cap.py`) has no valid signal: its target depends on the row index, not on the column values, so nothing is found even at index 10 and the result proves nothing. The check needs a target that is a real Haar leg of the column (a dyadic block mean difference of the column values), at index 10 and 150, with the cap and without. |
| 2f (triple budgets: `max_triples=40` in the argmax and conditional-gate scans, `fe_gbm_seeder_top_k_triples=8`) | The argmax `max_triples=40` is fixed by the MI-ranked triple order (item 2b). GBM seeder (`_surrogate_interaction_seeder.py`): measured 2026-10-24 on synthetic triples (`D:/Temp/gbm_triple*.py`): with 20 strong decoy triples and a weaker planted triple, the planted triple is not found at top_k_triples=8 nor at 100, and not even with top_k_pairs raised from 12 to 300, so the caps are not what hides it; the surrogate does not see this weak triple at all. With a stronger planted triple it is found at both 8 and 100. No loss attributable to the cap was found in this setup; the cap stays. |
| 13b (`fe_synergy_max_pairs=16`, recall check on a wide frame) | Measured 2026-10-24: a planted product of two columns at p=400, the compound with both planted columns in the name, 3 seeds (`D:/Temp/pair_budget.py`): found 3 of 3 with the budget at 16 and 3 of 3 with the budget lifted. No pair lost on this frame. As with the screen cap (item 11b), a constructed case where the budget alone loses the pair was not found. |
| 13c (`fe_synergy_max_pairs=16`, decoy frame) | Measured 2026-10-24 (`D:/Temp/pair_cap_ce.py`): 40 decoy columns correlated with y and 40 noise columns, a planted product of two columns: the compound is found with the budget at 16 and without it. The decoys did not push the pair out of the budget here, so no loss attributable to the budget was found; a frame where the budget does lose the pair is still to be constructed. |
| 2g (wavelet leg admission floor `median + 3.5*1.4826*MAD` of held-out leg MI) | Not a cap: a statistical admission rule (B-class, noise control). Measured: the exact planted leg (j=1, k=0, held-out MI 0.1882) sits just above the floor 0.1877, because several legs of the same column carry the signal (0.181, 0.141, 0.141) and shift the median and the MAD; the direct selector call returns no leg. The floor is calibrated for pure noise, so it is borderline when the signal spreads across legs. Open question for the wavelet owner: a rule that does not let the signal legs set their own floor. (`D:/Temp/leg_mi.py`, `D:/Temp/leg_direct.py`) |
| 13d (`fe_synergy_max_pairs=16`, decoy-strength sweep for a budget-only loss) | Searched 2026-10-24 (`D:/Temp/pair_cap_ce*.py`): decoy strengths 0.2, 0.23, 0.26, 0.3 (60 decoy columns, 30 noise). At each strength the planted product is either found by both budgets or missed by both; no strength gives a loss that only the budget causes. The synergy pair budget is not shown to lose a pair on these frames. |
| 11c (synergy sweep in `auto` mode: exhaustive only with CUDA and predicted time <= 300 s, else the top-250 pre-rank) | Measured 2026-10-24 (`D:/Temp/strict_nogpu.log`): a planted balanced product (zero marginal, p=400) is found as a compound in 6 of 6 seeds with the GPU disabled, the same as with it enabled. The documented loss of balanced pairs under the pre-rank cap was not reproduced on this frame; the pair is found by another route. The exhaustive-or-prerank gate is still the rule to check on a frame where no other route exists. |
| 24 (`FE_EAGER_MATERIALIZE_MAX_BYTES`, K-fold target encoding skipped on a large polars frame) | Fixed 2026-10-24 for K-fold: a large polars frame is now encoded natively (the fit reads one column at a time, the appended columns are added with `hstack`), no pandas copy of the frame and no skip. Measured on a forced guard (`D:/Temp/kfold_skip.py`): the planted category effect `cat__te` is found with the guard forced on, where the skip lost it. The count/frequency, missingness, ratio and binned-agg families (items 25-28) still skip above the same constant and are not changed here. |
| 25 (count/frequency/cat-num family skipped above the 2 GiB constant) | Profiled 2026-10-24 on a normal-size polars frame (`D:/Temp/count_prof.py`, cProfile): the hotspot in the CMI null is `permutation_sources` (`_cmi_strata_njit.py`), 101k calls to `Generator.permutation` (12.3 s of 13.2 s in that function), one call per stratum and permutation draw. It is held to a fixed RNG order by its docstring, so a vectorised or njit replacement changes the random stream and the selection the tests pin; not changed here, needs a decision to re-baseline. The family itself still skips above the constant; the planted frequency feature did not appear in this run, so no recall claim is made. |
| 26b (MDLP honest null, `honest_null_columns_bycol`, the largest compute kernel left in the count-family profile) | Profiled 2026-10-24 (`D:/Temp/count_prof.py`): 2 calls, 3.3 s each, parallel over columns and already njit. The device path is taken only when the spec has no NaN columns (`_honest_mdlp_null.py`); this run used the host kernel. A large cut needs a rewrite of the tree growth per permutation, not a hot-loop fix, so it is not changed here; the profile and the reason are recorded. |
| 2h (synergy `max_triples`, the modular FE triple budget) | Checked 2026-10-24: the synergy screen has no triple budget of its own. The only triple budget left in this area is the modular FE (`_pairwise_modular_fe.py`, `max_triples=12`), which runs on integer columns only and is set to 0 by the caller under a condition. It is a narrow family; no planted modular triple is tested yet. The argmax and conditional-gate triple budgets are fixed (items 2b, 2d). |
| 2i (pairwise modular FE caps: `max_int_cols=30` skips the whole sweep above 30 integer columns; `max_triple_cols=20` drops triples above 20) | Open, inconclusive. A planted `x0 mod 7` target is found at 20 and at 40 integer columns (`D:/Temp/modular_cap.py`), but the found name `k0__mod_7` belongs to the periodic family, not the pairwise modular sweep that carries these caps, so the test does not exercise the capped path. A planted pairwise modular target (for example `(x0 + x1) mod 7`) at p above 30 is the valid check. The whole-sweep skip is the same class as the argmax and gate skips already fixed. |
| 2i fixed (pairwise modular `max_int_cols=30` skipped the whole sweep) | Fixed 2026-10-24: a planted pairwise modular residue `(k0+k1) mod 7` at p=40 was lost with the cap at 30 (`pmod_sum` found at 20, not at 40) and found with the cap at 100 (`D:/Temp/pair_modular_cap.py`). The cost is the triple term, which keeps its own cap `max_triple_cols=20`; the pair sweep measured flat, 1.1 to 1.8 s at p=40 to 100 (`D:/Temp/pmod_cost.py`). Default raised to 200 in the function, the MRMR constructor, the fs_params and the cascade fallback; the stale skip test now asserts the planted pair survives at 35 columns. Modular FE tests 21 passed; F2 and fusion 12 passed. |
| 2j fixed (integer-lattice `max_int_cols=30` skipped the whole gcd/lcm/AND sweep) | Fixed 2026-10-24: a planted shared-factor pair (`gcd(k0,k1)` coprimality target) at p=40 was lost with the cap at 30 (`il_gcd__k0__k1` found at 20, not at 40) and found with the cap at 100 (`D:/Temp/lattice_cap.py`, `lattice_cap2.py`). The sweep is pairs-only, the same cost class as the modular pair term measured flat. Default raised to 200 in the function, the MRMR constructor, the fs_params and the cascade fallback; the stale skip test now asserts the planted pair survives at 35 columns. Lattice tests 18 passed; F2 and fusion 12 passed. |
| 11d (synergy pre-rank cap 250, planted balanced product beyond the cap, GPU disabled) | Measured 2026-10-24 (`D:/Temp/syn_cap_case.py`): 260 strong decoys and a planted product `a*b` at p=400: the compound is NOT found with the cap at 250 and NOT found with the cap lifted. The loss is not the cap: the decoys dominate the selection before the synergy step. A case where only the cap decides was not found. |
| 11e (where the planted product is lost in the decoy case) | Traced 2026-10-24 (`D:/Temp/syn_ledger.log`): the ledger holds only `oofwarp(pa)` (gate `oof_warp_gain`, not significant). No pair record for `pa*pb` exists, so the pair was never built. Its operands are chosen by marginal relevance, and a balanced product has zero marginal relevance, so the decoys push it out before the pair search. The loss is operand selection by marginal relevance, not the 250 cap; the fix would be an operand selection that does not rely on marginal relevance, which is a design change, not a cap change. |
| 64 (row-stat `MAX_POOL=16`, top 16 columns by MI or spread MI) | Checked 2026-10-24 (`D:/Temp/rowstat_pool.py`): a planted group row-statistic (`|x40..x43|` mean above the median, p=60) is found as `rowstat_std` and `rowstat_rng` with the pool at 16 and at 60. The pool is ranked by spread MI, which is the statistic the search looks for, so the group survives the cut here; no loss found. |
| all others | Not yet triaged. |
