# `_orthogonal_xi_gpu` (device twin of `_xi_columns_radix_njit`)

Rewritten 2026-11-09 after the first version (exact host permutations) lost to the host kernel.

## What changed and why

The first version had to be fed the host generator's permutations over the bus and spent most of its time in a scatter (`xi_inverse_perm`, global store efficiency 12.5 %), a gather (`xi_gather`, load efficiency 22 %) and a walk with a double indirection (`xi_walk`, 9.8 ms per 40 columns at 100000 rows, load efficiency 27 %). On the host the same permutations were 66 of the 99 ms of a 30000 x 100 batch. The permutation stream was pinned by `test_xi_batch_stream.py`, `test_xi_batch_kernel.py` and `test_xi_radix_kernel.py`; those tests now pin the new order, and the stream is gone.

New tie-break: the rows of a column are ordered by `(value, tie key, row)`, the tie key being the high word of `splitmix64(column seed + (row + 1) * golden)`; the column seed is `splitmix64(random_state * stride + j)` with `j` the number of the finite column in the batch (`xi_correlation` is column 0). Nothing is drawn or stored, the same arithmetic runs in numba and in CUDA, and `tests/feature_selection/_xi_reference.py` is an independent numpy `lexsort` reference. Chatterjee's construction only needs the tie-break to be random; it is as random as the permutation was. The values of tied columns change (the estimator's expectation does not).

Host effect: `xi_correlation_batch` on 30000 x 100 float32 99 ms -> 38 ms, 30000 x 24 27 ms -> 10 ms (one process, mean of 3).

## Device kernels (three launches)

`xi_sort_keys` (one 64-bit key per element: order-preserving code of the value, `-0.0` as `+0.0`, in the high word, tie key in the low word), a batched stable argsort of the keys (cub radix; its result is the row order, so no inverse permutation), `xi_walk` (class code of the row from a small table, rank tables through the read-only cache, integer sums in `{ACC}` = double, two atomic adds per block). `{REAL}` / `{ACC}` placeholders substituted before compilation; a float64 column goes to the host kernel. Operands: a numpy block is uploaded, a resident cupy block is used as is (`xi_correlation_batch(..., device_block=...)`; the auto-scorer bootstrap uploads each resample once for xi and copula). Blocks of at most a quarter of the free device memory; a fault returns `None` and the host kernel runs. Results equal the host's to the last bit (`test_xi_device_matches_host.py`: uploaded and resident, 1500 to 40001 rows, ties, a constant, signed zeros, the public batch).

## Profile (GTX 1050 Ti, 100000 rows x 40 columns, nvprof metrics in an elevated shell)

| kernel | time per call | achieved occupancy | global load eff. | global store eff. | warp exec. eff. |
|---|---:|---:|---:|---:|---:|
| `xi_sort_keys` | 0.56 ms | 0.85 | 85 % | 100 % | 100 % |
| `xi_walk` | 1.18 ms (was 9.8 ms) | 0.99 | 31 % (the class-code gather) | - | 98 % |
| cub radix sort | 14.5 ms | - | - | - | - |

The sort is now 90 % of the device time; the walk and the key build are launch-light and bandwidth-limited (30 GB/s read of 112 GB/s). Remaining lever: a 32-bit sort of the value codes plus a tie-run fix kernel (half the radix passes) for columns without long ties.

Timing, resident data, synchronised, mean of 5 (`bench_xi3.py`, 40 columns, 10 of them rounded to three levels): 5000 rows host 2.0 / device 2.1 ms; 30000 rows host 23.6 / device 12.0 ms; 100000 rows host 76.4 / device 40.2 ms (47 ms with the upload).

## Dispatch

Kernel-tuning entry `fe_xi_columns_device` (sweep over 5000 / 30000 / 100000 rows, equivalence rtol 1e-12), sweep run on this host with `tune_spec`: host at 5000 and 30000, device at 100000. No environment switch; `MLFRAME_DISABLE_GPU` still turns the device off.
