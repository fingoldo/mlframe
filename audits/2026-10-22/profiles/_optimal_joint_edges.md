# optimal_joint edges: the per-column plan in the FE worker pools

Warm all-families 100k x 100 fit with `nbins_strategy="optimal_joint"` (D:/Temp/ss_all.py, all-threads stack samples, 2026-11-05): 98.2 s before
this work, 96.3-99.7 s after (single runs; the walls vary by a few seconds between runs). About 26 s of worker time went into the optimal_joint
edges, plus 9.2-9.6 s of `honest_fdr_floor` on the main thread.

| change | worker samples before | per call | commit |
|---|---|---|---|
| one `np.percentile` per fold for every candidate bin count (`_quantile_edges_many`) instead of one per candidate | 9.0 s | 100k column plan 44.1 -> 26.8 ms | 33f9e68fb |
| near-collapse refinement counts bins from the column's distinct values and counts | 5.4 s | 1.47 -> 0.34 ms | 4f37a5f87 |
| fold masks drawn once per (rows, folds, seed), read-only and shared by every column | 2.8 s | plan 26.8 -> 20.3 ms | e0f764794 |

All three keep the bins and edges equal (randomised checks in `test_optimal_joint_plan_batched_quantiles.py`,
`test_supervised_refine_counts_from_uniques.py`).

Left: `column_facts` (`np.unique` of each distinct column, cached by content) 5.2 s; the per-candidate `searchsorted` of the validation values
3.0 s; `honest_fdr_floor` is GPU-bound (3.6 s of its 9.2 s waits on `oj_cell_mi` in `picks`; the private-counter variant was slower on this card,
see `_honest_oj_fused_gpu.md`).
