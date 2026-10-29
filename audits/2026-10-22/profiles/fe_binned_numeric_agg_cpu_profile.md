# fe_binned_numeric_agg_enable: CPU profile inside the all-flags fit (n=20000)

Measured 2026-10-28 with `D:/Temp/bna_allflags.py` (all FE flags on, the family's own wall 17.65 s in `fe_family_walls_gpu_vs_cpu_n20000` CPU run; the warm fit runs on other data so the fit cache does not hit). cProfile inflates small functions, so use the shares, not the absolute seconds.

Family cumulative (profiled, 3.68 s in `_stage_binned_numeric_agg`):

| part | cum s | share of family | note |
|---|---:|---:|---|
| `_host_candidates` | 2.18 | 59% | builds the host candidate frame and gates it |
| `_redundancy_gate_binagg` | 1.45 | 39% | MI redundancy gate |
| `_keep_linearly_usable` | 1.10 | 30% | linear-usability permutation test per (group, agg) pair |
| `fit_binned_numeric_agg` | 0.78 | 21% | OOF per-cell statistics, 249 calls of `_oof_stat_columns` (0.63 s) |
| `_candidate_codes` (quantile binning of each candidate) | 0.48 | 13% | called 376 times: once in the gate, again in `_keep_linearly_usable` |
| `_derive_cell_stats` / moments | 0.13 | 4% | already njit |

Attempts (none kept):
- Vectorising `linear_usability_keeps` across candidates (one `_mean_gains_by_cell` call per permutation batch instead of one per candidate): output equal on the synthetic pair, wall 0.067 s to 0.063 s (7%). Not kept: the gain is inside the noise of a single pair and the change touches a statistical test.
- Memoising candidate codes between the gate and `_keep_linearly_usable`: saves about half of the 0.48 s, but the codes are int arrays of n rows per candidate (about 97 MB for 608 candidates at n=20000), so the cache would move the memory problem rather than remove it. Not done.

Reading: the family has no single hot spot. Its cost is split between the redundancy gate (candidate binning done twice), the linear-usability test, and the OOF per-cell statistics. The next lever is the duplicate binning (a bounded per-pair cache, not a global one), and the test in `_keep_linearly_usable` already runs per pair. Documented here so the next attempt starts from these shares.
