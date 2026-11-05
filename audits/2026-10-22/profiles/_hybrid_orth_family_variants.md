# hybrid_orth variant groups: groups 1, 3 and 4 run together

`early_a` ran its hybrid_orth variants in four groups, one after another, at about 1.6 busy cores: in a warm 100k x 100 fit group 1 6.6 s,
group 2 3.6 s, group 3 2.1 s, group 4 6.2 s (of `early_a` 24 s). Every variant takes its sources as the numeric columns that are not hybrid
outputs (it re-reads `hybrid_orth_features_`), so the groups read the same sources in any order; the exception is the three-gate variant in
group 2, whose support set is the hybrid columns appended before it.

Groups 1, 3 and 4 now run together on private copies of the estimator and of the recipe dict (`run_stages_concurrently`); group 2 runs after
group 1 on its frame; everything is folded back in the order 1, 2, 3, 4 with the first owner of a name kept. The sequential path lists a name
two variants emit once as well (it listed `x4__fourier3` twice). `MLFRAME_FE_HYBRID_VARIANT_CONCURRENCY=0` runs the groups one by one.

| fit (one process, order seq, conc, conc, seq) | sequential | concurrent |
|---|---|---|
| warm 30000 x 40 | 31.2 / 31.8 s | 28.4 / 29.0 s |
| warm 100000 x 100 | 85.6 / 84.6 s | 79.5 / 80.4 s |

Same selected features, `hybrid_orth_features_`, recipes and transformed values in both runs (D:/Temp/hyb_eq.py);
`test_hybrid_orth_variant_groups_concurrent.py` checks a small fit and the fold rules.
