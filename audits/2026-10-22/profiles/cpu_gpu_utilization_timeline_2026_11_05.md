# CPU / GPU utilization timeline, all families, 100k x 100 (2026-11-05)

Tool: `scripts/util_timeline.py` (psutil per-core CPU%, nvidia-smi GPU%, 1 s samples, main-thread stage + leaf at 1 s and 0.1 s). Warm second fit, 8 logical cores.

Result (wall 90-95 s): mean CPU 46-55 %, 3.2-4.2 cores above 50 %, mean GPU 24-28 %.

| window | CPU % | cores >50 % | what runs |
|---|---|---|---|
| 0-15 s | 47-63 | 3-6 | `_fe_stage_cascade_early_a_part1` (concurrent hybrid_orth groups, basis generation) |
| 16-26 s | 31-41 | 0.6-2 | hybrid_orth group 2 (three_gate ~2 s, ksg, bootstrap) runs ALONE after group 1 while groups 3/4 are done |
| 27-64 s | 44-65 | 4-6 | `_fit_body` with concurrent stages, best phase |
| 65-95 s | 28-48 | 0.1-2.8 | serial post-screen tail: pair emission step2, prospective pairs, retain_usable_pure_forms (CV log-loss, abs-corr), selection compaction; no single step above 1.3 s, ~25 distinct steps |

Findings:
- The tail (last ~25 s, 0.2-1 busy cores) is a long chain of small single-threaded host steps plus GPU syncs; the largest at 0.1 s sampling are `retain_usable_pure_forms` log-loss (1.3 s) and `screen_predictors._continuous_accept_block` (1.2 s). Raising utilization there needs the independent post-screen passes to run as concurrent stages, not a kernel.
- hybrid_orth group 2 is the only serial stretch inside the FE cascade; three_gate depends on group 1's columns (its support set), so only diff_basis/cluster_basis/bootstrap (small) could move next to group 1.
