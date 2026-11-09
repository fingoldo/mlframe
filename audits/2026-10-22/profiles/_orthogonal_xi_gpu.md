# `_orthogonal_xi_gpu` (device twin of `_xi_columns_njit` / `_xi_columns_radix_njit`)

Written 2026-11-09 after the measured coverage table in `fe_gpu_coverage.md` showed that the dependence scorers of the bootstrap / ensemble / meta / auto-scorer families (xi, copula MI, dCor, HSIC, tail dependence) had no device code.

Kernels (`xi_inverse_perm`, `xi_gather`, `xi_walk` plus a stable device argsort): the rank sums are integer counts held exactly in doubles, so the device value equals the host value to the last bit (7 tests: float32 and float64, 1500 to 40001 rows, columns with continuous values, heavy ties, four values, a constant and signed zeros; the permutations are the host's, so the tie-break stream is the same). Columns go in blocks of at most a quarter of the free device memory; a device fault returns `None` and the host kernel runs. Dispatch: `use_device` (`MLFRAME_XI_DEVICE=0|1`, else the kernel-tuning cache `fe_xi_columns_device`, fallback host).

Timing (GTX 1050 Ti, 40 columns, resident device data, synchronised, mean of 3, `bench_xi.py` in D:/Temp):

| rows | host (prange radix) | device |
|---:|---:|---:|
| 5000 | 1.0 ms | 4.3 ms |
| 30000 | 10.1 ms | 22.1 ms |
| 100000 | 55.1 ms | 75.9 ms |

Breakdown at 100000 rows: upload of columns and permutations 9.0 ms, device argsort 22.5 ms (a loop of 1-D argsorts 27.0 ms), the three kernels about 34 ms (random-access scatter, gather and a double indirection in the walk; this card has 112 GB/s). The host kernel is the faster path at every measured size, so the device twin is registered with the tuner but the fallback is the host: the sweep decides on a card where the device wins. Not done and the next lever if this is wanted on the device: sort (key, row) pairs together to remove the indirection of the walk, draw the permutations on the device (a different tie-break stream, selection-equivalent only), keep the columns resident between the scorers of one resample.
