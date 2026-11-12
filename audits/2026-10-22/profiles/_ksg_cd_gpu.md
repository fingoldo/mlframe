# _ksg_cd_gpu.py: device twin of the Ross continuous-versus-discrete k-NN MI (2026-11-09)

Host kernel: `_ksg_cd_fast._mi_cd_columns` (radix order, walk over the class's sorted values, prange over columns). Device twin: batched argsort of the columns, then one thread per point walks the sorted column skipping other classes, runs the host's widen-then-trim count, and each block writes a partial sum of psi(m); the class sums of psi(k) and psi(count) are formed on the host. Dispatch: kernel-tuning-cache entry `fe_ksg_cd_device` (fallback 20000 rows; sweep not run yet). Columns are streamed in blocks within 25 % of the free memory; any device fault returns to the host kernel. Tests: `tests/feature_selection/fe/test_ksg_cd_device_matches_host.py`.

Measured (100000 rows x 60 columns, 3 classes, GTX Pascal, 8 host cores, host kernel already parallel): host 0.216-0.237 s; device 0.192 s with the host-side transpose of the block, 0.102 s with the block uploaded as laid out and transposed on the device (2.1x). Agreement with the host: max abs difference 7e-12 (floating-point order of the psi sum).

nvprof (4 calls, 60 columns): cub per-column sorts 88 ms (1920 launches, 22 ms per call, 41 % of device time), `ksg_psi_m_sums` 52.8 ms (13.2 ms per call), host-to-device copy 30.5 ms, `cupy_take` + indexing 24 ms. Open: the per-vector sort launches (8 per column) dominate; a segmented radix sort over all columns in one launch would remove most of them. The whole `orth_mi_ksg` family is about 1.1 s of a 100000 x 100 fit and the auto-scorer calls this kernel per block, so the expected gain per fit is a few tenths of a second.

## 2026-11-12: the block transpose moved from the host to the device

`ksg_mi_discrete_target_columns` on 100 columns x 100000 rows took 495 ms; the device MI inside it 308 to 328 ms. nvprof showed about 80 ms of GPU work per call (cub radix sort 46%, `ksg_psi_m_sums` 19%, copies 15%, `cupy_take` and indexing 12%), so three quarters of the wall was around the kernels. Timing the stages of `mi_cd_columns_device` with synchronization:

| stage | before (ms) |
|---|---|
| host transpose of the (n, 100) block to (100, n) | 180 |
| upload | 23 |
| `device_argsort` (a loop of per-column sorts) | 111 |
| `take_along_axis` | 10 |
| `d[order]` | 2 |
| `ksg_psi_m_sums` | 21 |
| whole call | 308 |

The host transpose of 80 MB is cache-unfriendly and was 58% of the call. The block is now uploaded as it is (a row-wise host copy, none when the block is the whole array) and transposed on the device with the project's coalesced tiled kernel `_transpose_to_cm`. Whole call 308 ms -> 159 ms (1.9x) with identical results (the transpose only moves data; 5 equivalence tests pass, including float32 input). The sort is now 65% of the call: a per-column loop of cub radix sorts (800 launches per call), the next thing to profile. The host preparation of the batch (float64 copy, scaling, noise: 160 ms at 100000 rows) stays on the host on purpose, because moving the standard deviation to the device would change its summation order and so the last bits of the scaled values.
