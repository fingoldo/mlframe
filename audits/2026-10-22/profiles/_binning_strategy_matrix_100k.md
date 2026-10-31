# Binning-strategy matrix, BASE fit, 100k rows x 100 columns (CPU)

Profiled with cProfile (`fam_prof.py BASE --mode cpu --binning <strategy>`); wall time of one fit, selection identical across the optimised and unoptimised runs of a strategy.

| strategy | before | after | what changed |
|---|---|---|---|
| mdlp | 51 s | 51 s | (default; honest null already vectorised earlier) |
| mdlp_validated | 840 s | 385 s | `honest_validated_null_columns` reads the distinct pairs and the train rows off the per-column sort order instead of sorting per (permutation, column); a float32 column at 100k rows always has ties, so the earlier tie-free-only fast path never applied |
| optimal_joint | > 1500 s (did not finish) | 140 s | `PermutedTargetBinner`: frame prepared once, code column memoised per (column, winning bin count), one parallel kernel scores the candidate bin counts of all columns, a candidate's confirmation bins one column per permutation |
| fd | 121 s | not changed | 44 s in `np.argsort` of the within-stratum shuffle in `_conditional_cpu_null` (per-permutation CMI null); heavily benchmarked before, no exact batching found |
| sturges | 29 s | - | - |
| none_quantile | 39 s | - | - |
| none_uniform | 96 s | 79 s | `np.nan_to_num` of pair candidates replaced by a one-pass scrub; remaining: family bitset null (23 s), permutation block fills (5 s) |

Shared changes: `quantile_codes_block` selects the ~2*(n_bins+1) order statistics it needs instead of sorting each column (1.9x); the honest-null root precheck gathers and counts in one pass for tie-free columns; the triplet enumeration kernel builds the (i, j) part of the cell index once per pair; the family gate's permutation blocks are shared by every target of the same length.

Open: honest null of plain `mdlp` (~18 s of 51 s BASE, memory-bound, 3.5x on 8 threads), family-gate null for `oof_warp` / `modular` families, `_conditional_cpu_null` for `fd`.
