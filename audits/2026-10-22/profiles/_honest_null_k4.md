# _honest_null_k4 (honest-null root split, one pass, full x ln x table)

Default is v5 (`MLFRAME_HONEST_ROOT_SPLIT=k5`, 287 ms; `k4` exact 535 ms; `k2` original); kernel `root_split_stats_k4` in `src/mlframe/feature_selection/filters/_honest_null_k4.py`; dispatched from `_honest_null_accept._root_split_kernel` (binary target). `MLFRAME_HONEST_ROOT_SPLIT_K4=0` returns to v2.

Measured (GTX 1050 Ti, 60 columns x 200 permutations x 100000 rows, synchronized, one process, min of 3):

| variant | ms | outputs vs v2 |
|---|---|---|
| v2 (two passes, table 32768, `log` above it) | 861 | reference |
| v2 + table over every count | 770 | gain within 2e-15, index and count identical |
| v4 (one pass, full table, double gain) | 535 | gain within 2e-15, index and count identical |
| v4f (one pass, float gain) | 577 | gain error up to 2.4e-6 against gains of 1e-4 (not dispatched) |
| v5 (one pass, float entropy score, exact gain once) | 287 | gain error up to 1.8e-7, 7% of indices differ (not dispatched) |

Finding: the first one-pass attempt (2.3x SLOWER than v2) was not slowed by the double additions. Counts of 32768 and more went through `log((double)v)`, a double-precision call on almost every step at 100000 rows;
a table over all counts (`xlogx_table(n)`, 800 KB) removes it. The table already existed for the host kernel (threshold 32768) and was reused.

Next: v5 at 287 ms is the remaining headroom (selection-equivalence only); the stage also spends time outside the kernel (host sort of the columns, tie flags, label bit packing, the redo of accepted pairs).
Tests: `tests/feature_selection/fe/test_honest_null_k4.py`, `tests/feature_selection/discretization/test_honest_null_device_equivalence.py`.

## GPU profile of v5 (nvprof, GTX 1050 Ti, 60 columns x 200 permutations x 100000 rows, 12000 threads)

achieved occupancy 0.50 (53 registers per thread), multiprocessor activity 95.3%, device memory utilization low (2 of 10), warp execution efficiency 66.6% (the candidate branch diverges inside a warp).
Register cap: `--maxrregcount` 32 / 40 / 48 / 64 gave 247 / 297 / 302 / 299 ms against 267 ms uncapped, inside the run-to-run noise, so the cap is not applied.
Reading: the 12000 threads are one resident wave on 6 SMs, so the kernel is bound by per-thread latency (the dependent label-bit gather), not by DRAM or by occupancy; more work per launch (250 columns x 999 permutations in the real stage) raises SM coverage, not the per-thread chain.
