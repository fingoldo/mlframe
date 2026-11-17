# Device table TreeSHAP kernel profile (GTX 1050 Ti, sm_61), 2026-11-17

Module `_shap_proxy_treeshap_table_gpu` (the production device TreeSHAP; the path kernel `_shap_proxy_treeshap_gpu` is its fallback). Host twin: the numba table kernels of `_shap_proxy_treeshap_table` (profile record `_shap_proxy_treeshap_numba.md`).

Workload: LightGBM 100 trees depth 6, 100 features. Whole call (host tables + upload + kernel + download), float32, 5 interleaved rounds, GPU 68-84 C, no throttle.

| rows | before (row-major X, stack walk) | feature-major X, stack walk (best/median) | + preorder walk (best/median) | diff to reference |
|---|---|---|---|---|
| 20000 | 91-94 ms | 59.4 / 60.0 ms | 46.4 / 46.6 ms | 0 (bit identical) |
| 100000 | 530 ms | 307.5 / 318.1 ms | 218.8 / 223.7 ms | 0 |

Kernel only (events, 7 interleaved rounds, 20000 rows): row-major X 72.9 ms, feature-major 38.95, + int4 AoS records 33.1, + static preorder records 27.5, + __restrict__ 24.7 ms (1.58x vs feature-major, 2.9x vs row-major).
Rejected (kernel ms, same harness): block 64/256/512 and --maxrregcount 24/40/48/64 no change (+-0.5 percent); loop unroll 4 slower (28.4 vs 27.5); paired (value, feature) table entries 26.1 vs 24.7; shared-memory child masks 25.8 vs 24.7 (spill-free but 4 percent slower); shared-memory DFS stack 15 to 23 percent slower than the stack frame (register-pressure exemption, seven interleaved pairs).
nvprof of the stack kernel (feature-major): occupancy 0.95, IPC 1.76, stall_memory_dependency 85 percent, gld_efficiency 44 percent, local_hit_rate 50 percent, warp_execution_efficiency 99.6 percent, dram 10 GB/s read 17 GB/s write (latency bound, not bandwidth bound); GPU utilization of the whole call is dominated by the host table build at small row counts.
Launches per call: 1 kernel per row block (+ transpose, zeros); one row block at the sizes above (the block is sized from the free device memory, `_row_block`).
Host side (cProfile of the whole call, 20000 rows): the numba table build `build_tables` and the preorder records are the host share, the rest is the kernel, the upload and the download.
Tuner: `shap_proxy_treeshap_backend` (numba table against the device table over rows x trees): the device kernel wins from 1500 rows at 100 trees and from 6000 rows at 20 trees on this host.
