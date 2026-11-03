# `_bootstrap_rank_mi_gpu.py` (device walk of the bootstrap rank-MI, fully distinct columns)

Device twin of the sorted walk in `_bootstrap_rank_mi_njit._walk_sorted`: `cupy.argsort` orders the columns, one thread per (column, resample) walks the sorted rows, emitting each row `counts[row, resample]` times
into the equi-frequency slices and finishing with the plug-in MI. Columns with equal values of different rows (the host emits those in draw order) and non-finite columns stay on the host walk. The kernel reads only
integers (row ids, multiplicities, class codes, slice ends); the MI sums are double (no `{REAL}` needed).

## Contract points (CUDA.md section 2)

CPU twin kept (the njit walk, which also answers the columns the device declines); column-block streaming (256 columns per block); registered tuner `fe_bootstrap_rank_device` (sweep over 5000 / 30000 / 100000 rows, 6 s,
fallback: the host walk; `MLFRAME_BOOTSTRAP_RANK_DEVICE=0|1` forces a path); tests `test_bootstrap_rank_mi_gpu.py` (odd sizes, 2 to 5 classes, tied / NaN columns, more than 16 classes declined; MI equal to 1e-12).

## Measured (GTX 1050 Ti, warm, 105 columns x 10 resamples of 80000 rows out of 100000)

| path | seconds |
|---|---|
| host njit walk | 0.24 |
| device walk (argsort + walk kernel) | 0.56 |

The tuner picks the host at every size here (the MI agrees exactly, so the choice is speed only). BENCH-ATTEMPT-REJECTED as the default on this GPU: the host walk is already fast once compiled (the 3 to 4 s the stage shows
is many calls, not one slow one), and the device path pays for sorting 105 columns of 100000 floats (about 0.3 s on Pascal) plus a walk with only 1050 threads. Retry condition: a card with a faster sort (a newer GPU),
or many more columns / resamples per call, where the tuner will switch by itself.

## Re-measure (2026-11-01, later): the device walk wins

105 float32 columns, 10 resamples of 80% of the rows, min of 5 in one process (GTX 1050 Ti): 30000 rows host 0.05 s / device 0.04 s; 100000 rows host 0.29 s / device 0.15 s (2x). The earlier verdict (host 0.24 s against device 0.56 s) was taken under contention and does not reproduce.
Dispatch fallback (no kernel-tuning entry on the machine) is now the device walk from 30000 rows (`DEVICE_WALK_FALLBACK_ROWS`); `MLFRAME_BOOTSTRAP_RANK_DEVICE=0|1` still forces a path. A host/device split of the columns (20 / 30 / 40% on the device, the rest on the host at the same time) took 0.25 / 0.22 / 0.21 s,
slower than the device alone (0.15 s), so no split is dispatched.

Full-fit A/B of the new default (100000 x 100, every family, back to back in two processes, same machine state): device walk forced on 169.6 s, host forced 169.3 s. The call itself is 2x faster (about 1 to 2 s over the fit), which sits inside the run-to-run noise
of the whole fit (145 to 170 s for identical code on this machine on the same day), so the fit-level gain is not resolvable by wall time; the stage seconds in the sampler are the instrument for it.

## Pool pressure found in the full fit (2026-11-01) and its fix

With the device walk as the default, the 100000 x 100 every-family fit printed `OutOfMemoryError` tracebacks (a 614 MB then 525 MB thrust temporary of the argsort against the 2.58 GB cupy pool) and the numeric-decompose family lost its columns to the same full pool;
with the walk forced to the host there were none. Causes and fixes: (1) a block of 256 columns holds several hundred MB at 100000 rows, so the block is now bounded by what the pool can still hand out (`_column_block`, 30%, 8 to 256 columns);
(2) the pool was full of cached blocks of other sizes that do not merge into a large contiguous request, so the unused cache is released before and after the walk (`_release_pool_cache`); (3) any device fault in the walk now returns to the host walk.
After the fixes the full fit prints no OutOfMemoryError and no family raises. Lesson for every new kernel: a full-fit run must be grepped for OutOfMemoryError / `FE raised`, not only timed.

## Re-measure (2026-11-03): the host walk wins again; the tuning cache is right
Quiet machine, same shape as the 2026-11-01 re-measure (105 float32 columns, 10 resamples of 80%), host / device alternating in one process, warm runs:
30000 rows host 0.045-0.050 s / device 0.112-0.116 s; 100000 rows host 0.231-0.296 s / device 0.544-0.577 s; MI identical. `mlframe-tune-kernels refresh
fe_bootstrap_rank_device` (quiet) again records `host` for every size. The 11-01 "device wins" numbers do not reproduce; the fallback constant
`DEVICE_WALK_FALLBACK_ROWS` only matters on a machine without a tuning entry. In the 50k all-families fit the host walk scores 522 columns in 0.50 s
(9 calls) and the tied device kernel 782 columns in 1.29 s (10 calls).
