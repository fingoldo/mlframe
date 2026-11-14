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

## Shared (k, n) block inside the auto-scorer resample (2026-11-12)

Each bootstrap resample builds an `(n, k)` float32 block and several scorers wanted it as `(k, n)`; each transposed it on the host (14 ms at 30000 rows, 134 ms at 100000 rows, float64 24 and 195 ms). `_device_block` now uploads the `(n, k)` block as it is and transposes it once on the device with the coalesced tiled kernel; xi, copula MI and tail dependence read that one block (`tail_dependence_columns(None, y, device_block=...)`), and the host copy is fetched from the device only when the device path is not taken. The whole family `hybrid_orth_mi_auto_scorer_fe_with_recipes`, 100 columns: 1.98 s -> 1.90 s at 30000 rows and 8.06 s -> 6.96 s at 100000 rows (this includes the device tail dependence and the KSG transpose change), with an identical augmented frame, score table and recipes on a fixed frame (40000 x 30). Scorers that need row-major data (the plug-in batch, the KSG preparation) keep the host block.

## Contract view with fresh measurements (2026-11-14, idle machine)

### cProfile

The host path (`_host`, 60 float32 columns x 1 000 000 rows): 1.271 s under the profiler, all of it inside the njit kernel `_tail_dep_columns_njit` (6 function calls in total): a host call is its kernel.

### Kernels (nvprof, one device call, 60 x 1 000 000)

Kernel launches per call: `pack_keys` 4 launches (12.6 ms, 5.4 percent of the device time of 235 ms), `tail_counts` 4 launches (72.2 ms, 30.7 percent, 18 ms each: one block per column walks the runs of the sorted keys), the key sort as 32 cub radix-sort launches (95.0 ms, 40.5 percent) and 4 sort-bookkeeping launches (12.0 ms), 7 host-to-device copies (40.8 ms, 17.4 percent). Four launches of each kind: the 60 columns go in 4 column blocks.

### Occupancy and throughput

nvprof counters were not collected on 2026-11-14 (`Internal superseded by the Hardware counters section below` for every kernel today). `pack_keys` reads a 4-byte value and an 8-byte flag word and writes an 8-byte key, at most 20 bytes per element (the flag word is shared by all columns and cached), for 60 million elements in 12.6 ms: at most 95 GB/s of the card's 112 GB/s; `tail_counts` reads 480 MB of 64-bit keys in 72.2 ms, 6.7 GB/s, with one block per column (60 blocks for 6 SMs, a column's run walk is sequential inside its block): the run walk is the kernel to improve, not the packing. The kernel source now takes its types from the placeholders (`{REAL}` for the value, `{ACC}` for the rank formed in double precision), compiled for float32 columns; the result stays bit-equal to the host (`np.array_equal` at 5000 and 100000 rows after the change).

### Host-or-device choice

Medians of 7 calls, host njit / device, ms, 60 columns: 5000 rows 2.04 / 2.67; 10000 3.86 / 3.81; 30000 12.8 / 7.8; 100000 63.6 / 22.5; 300000 259.3 / 62.7; 1 000 000 1174.9 / 361.3 (3.3x). The counts are integers, so the result is `array_equal` to the host at every size; the device is ahead from 10000 rows.

### Streaming

`tail_dependence_columns_device` uploads the host array column block by column block (blocks fit `_BUDGET_FRACTION` of the room the allocator has left) or reads a resident cupy array in place; any device fault returns `None` and the caller runs the host kernel.

## Hardware counters (nvprof --metrics, elevated shell, 2026-11-14)
Superseding the theoretical occupancy stated above (the counters were blocked by the driver's restriction until the process ran as administrator). Kernel alone, last sweep size, one profiled call after a warm call. Pascal sm_61: Nsight Compute reports the device as unsupported, nvprof is the tool.

| kernel | achieved occupancy | SM efficiency | DRAM read | DRAM write | global load eff. | global store eff. | shared eff. |
|---|---|---|---|---|---|---|---|
| tail_counts | 11.7 % | 98.0 % | 8.8 GB/s | 0.12 GB/s | 93.4 % | 4.2 % | 0.96 % |
| pack_keys | 89.2 % | 98.9 % | 46.2 GB/s | 29.8 GB/s | 100 % | 99.7 % | 0 |

`tail_counts` runs at 11.7 percent achieved occupancy: that is the lever, not the throughput.

## Optimization from the counters: `tail_counts` split into column segments (2026-11-14)
The counters showed `tail_counts` at 11.7 percent achieved occupancy (one block per column, 15 columns per block on 6 multiprocessors) and it took 32.6 percent of the device call (97.3 ms over 4 launches at 60 columns x 1,000,000 rows). A column is now cut into segments (grid `(columns, segments)`, `_BLOCKS_PER_SM` = 8 blocks per multiprocessor, no segment under 4096 rows); a run belongs to the element that starts it, so a segment boundary cuts nothing, and the four integer counts are added with warp shuffles and 4 integer atomics per warp, so the result does not depend on block order. After: 28.7 ms over the same 4 launches (3.4x on the kernel), achieved occupancy 39.8 percent, global load efficiency 92.0 percent, DRAM read throughput 14.6 GB/s. Counts equal to the bit on the existing device-vs-host tests and on the new segment-cut test (segments of 64 and 1000 rows on tied columns). Measured with nvprof traces of the same call, before and after (one run each, not an interleaved pair; the mechanism is the launch shape).
