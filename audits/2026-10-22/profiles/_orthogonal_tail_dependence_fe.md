# `_orthogonal_tail_dependence_fe` / `_tail_dependence_gpu`: host and device profile

Measured 2026-11-12 on the development host (GTX 1050 Ti, Pascal sm_61; `ncu` does not support this card, `nvprof` kernel timing was used), 100 float32 columns scored against one rank-uniformised binary target, `q = 0.95`. Times are the best of five synchronized calls after a warm-up; the device columns follow CUDA.md section 6 (host CPU, device with the per-call upload, device on data already resident). Every device result equals the host kernel's bit for bit.

## Host path

`_tail_dep_columns_njit`: one parallel numba task per column, radix order for float32 columns, then a walk over the runs of equal values. 15 ms at 30000 rows, 94 ms at 100000, 388 to 439 ms at 300000, 1.9 s at 1000000.

## Device path: versions and what the profile changed

| version | what it is | 30000 rows | 100000 | 300000 | 1000000 |
|---|---|---|---|---|---|
| v0 chain | `argsort` per column, then about twenty array operations with `(b, n)` int64 tables in global memory (kept as `_block_counts_chain`, the float64 path and the test oracle) | 0.33x (48 ms) | 0.72x (129 ms) | 1.5x | 2.0x |
| v1 | float32 key packed with the target's two tail flags and the column number, one flat sort without a payload, one counting kernel (`tail_counts`); the key packed by about eleven cupy element operations | 1.08x | 1.86x | 2.8x | 3.1x |
| v2 | the key packing fused into one kernel (`pack_keys`) | 1.33x | 2.61x | 3.33x | 3.96x |
| v3 | the sort done in place (no copy of the key array) | 1.36x | 2.76x | | 4.24x |

Device data already resident, v3: 1.83x at 30000, 3.70x at 100000, 5.23x at 1000000. Below 30000 rows (upload path): 0.41x at 2000, 0.73x at 5000, 0.94x at 10000; the device path takes over near 12000 rows, which is `DEVICE_FALLBACK_ROWS`; the kernel-tuning sweep (`fe_tail_dep_device`, rows 5000 to 1000000) refines it per host.

The host-versus-device threshold was set after the profile-driven optimisation, not before: the first device version (v0) lost to the host and would have set a wrong threshold.

## nvprof kernel table

v1 at 100 columns x 100000 rows (6 calls): cub radix sort 45% (onesweep 40.3%, histogram 5.1%), `tail_counts` 16.7%, the key packing in about eleven `cupy_*` element operations about 36% (`bitwise_or` 9.4%, `left_shift` 4.6%, `copy` 3.9%, `equal` 2.9%, `xor` 2.6%, `add` 2.5%, `right_shift` 2.1%, `where` 2.0% and others).

v2 (packing fused): sort 62% (onesweep 56.3%, histogram 5.6%), `tail_counts` 16.5%, `pack_keys` 8.4%, a `cupy_copy__uint64` of the key array 6.6%, host-to-device copies 5.5%.

v3 removes that copy. Remaining headroom is the sort, about 60% of the GPU time: cupy's `sort` runs the 64-bit radix over the full key although only 41 bits (7 for the column, 32 for the value, 2 flags) carry information; a begin and end bit are not exposed, so a narrower sort needs the cub call directly. Not done.

## Contract points

CPU kernel: `_tail_dep_columns_njit`. Fused GPU kernels: `pack_keys` and `tail_counts` (integer keys and counts; the only double is the average rank, compared with the tail quantile, which is the `dtype-ok` reviewed line). Streaming: column blocks sized from `device_room` (48 bytes per element for float32, 144 for the float64 chain path); a device fault returns `None` and the host kernel scores the columns. KTC: `fe_tail_dep_device`, sweep rows 5000 to 1000000. Offline NVRTC to ptxas check: both kernels compile on sm_61, 70, 75, 86, 89, 90 and 120 with no spill.

`tests/test_meta/_fe_gpu_contract.json` keeps `_orthogonal_tail_dependence_fe` pending: the checker reads one module file, and the device kernels, the tuner and the streaming live in the twin file `_tail_dependence_gpu.py`.
