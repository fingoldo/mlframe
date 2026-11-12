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

## Float64 wide path, measured 2026-11-12 (GTX 1050 Ti, 40 columns, synchronized, best of 5)

| rows | host `_xi_columns_njit` | device, host upload | device, resident |
|---|---|---|---|
| 30000 | 27.1 ms | 40.4 ms | 38.8 ms |
| 100000 | 117.6 ms | 151.0 ms | 142.3 ms |

Both device paths equal the host kernel exactly (`equal=True`). The device loses on this card at both sizes, so `use_device` keeps the host for float64 (the KTC default `DEVICE_FALLBACK_ROWS` is 10**9 and a sweep would also pick the host here).

Where the device time goes, from the code, not yet from a GPU profile: the wide path runs three `device_argsort` calls per chunk (pass one on the (tie, row) keys, the dense-rank sort of the order values, and the composite sort), plus a cumsum, two gathers and one scatter over `(w, n)` u64 arrays. The float32 path runs one argsort. Next step before a KTC decision: an nvprof breakdown of the three sorts against the walk kernel, then a cheaper order: the rank sort can reuse the (tie, row) pass when the order values are already sorted, or the composite can be built from a radix pass over the 64-bit order alone.

### nvprof breakdown of the float64 wide path (2026-11-12, GTX 1050 Ti, 30000 rows x 40 columns, 4 calls: one warm-up and three measured, whole script profiled)

Command: `nvprof --csv` with `PATH` extended by the CUPTI directory of CUDA 12.8 (without it nvprof fails to load `cupti64_2025.1.0.dll`). Time column is milliseconds.

| kernel | total ms | share of GPU time | launches |
|---|---|---|---|
| CUB radix sort (the three `device_argsort` calls: (tie, row) keys, the order values, the composite) | 111.1 | 78% | 300 |
| `xi_keys_wide` (key build) | 1.19 | 0.8% | 4 |
| `xi_walk` (Xi sums) | 1.84 | 1.3% | 4 |
| all other kernels (cupy elementwise, take, scan, scatter, memcpy, memset) | about 28 | 20% | the rest |

GPU total 142.2 ms over 4 calls, about 35 ms per call, which agrees with the synchronized wall time of 40 ms. The key build and the walk are negligible; the three 64-bit radix sorts are the device time, so the path is sort-bound. The host kernel does the same work in about 27 ms per call. Options that were considered: a 32-bit composite (only valid below 65536 rows), dropping the (tie, row) pass (needs stable order from the device, which the design avoids), and CUB begin/end-bit control (not exposed by cupy's argsort). None is a cheap change, so the host stays the float64 choice.

## nvprof breakdown, float32 device path (2026-11-12, GTX 1050 Ti, 30000 rows x 40 columns, 1 warm-up and 3 measured calls)

Command as in the float64 breakdown above (CUPTI directory on `PATH`). Time column in ms.

| kernel | total ms | share | launches |
|---|---|---|---|
| CUB radix sort (the batched argsort of the sort keys, 44 + 4 + 44 launches) | 27.81 + 8.81 + 1.25 = 37.9 | 88% | 92 |
| `xi_walk` (Xi sums) | 1.88 | 4.4% | 4 |
| `xi_sort_keys` (key build) | 0.62 | 1.4% | 4 |
| other (copies, memcpy HtoD) | about 2.6 | 6% | the rest |

GPU total 43.0 ms over four calls, about 10.8 ms per call, against a host kernel of 23.6 ms at this size (the wall timing above). The walk and the key build are small; the batched sort is the cost. Optimization stays open with the same candidate as the copula module: one value sort shared by the scorers that take the same resample block.
