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
