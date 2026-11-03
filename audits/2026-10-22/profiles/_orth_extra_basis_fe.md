
## 2026-11-03: Fourier extra basis, py-spy of the worker threads

Fit stage sample: 14 samples on the main thread waiting for the column pool (low CPU). Standalone (100000 x 100, 30000-row decision subsample, fourier
basis): cold 2.62 s, warm 0.62 s. py-spy over all threads of the cold run: `np.var` from the `np.std(leg) > 1e-12` constancy checks (two per frequency per column) 361
of 1443 worker samples, `_is_int_as_cat_axis` waiting in `acquire` (the numba first-call dispatch lock, cold only) 219, the heavy-tail detector's partitions 260.
The constancy checks now go through `_std_vs`: one min / max pass, and the std only when `range / sqrt(2 n)` cannot decide (a zero range is not shortcut: the
float32 std of a constant column is a 9.5e-7 residue the threshold sees). Same verdicts; cold 2.62 -> 2.47 s, warm 0.62 -> 0.56 s.
