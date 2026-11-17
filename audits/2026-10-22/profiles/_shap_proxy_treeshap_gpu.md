# TreeSHAP GPU kernels profile (GTX 1050 Ti, sm_61), 2026-11-17

Workload: LightGBM 100 trees depth 6, 100 features. Whole call (host tables + upload + kernel + download), float32, 5 interleaved rounds, GPU 68-84 C, no throttle.

| rows | before (row-major X, stack walk, 11-04 figure) | feature-major X, stack walk (best/median) | + preorder walk (best/median) | diff to reference |
|---|---|---|---|---|
| 20000 | 91-94 ms | 59.4 / 60.0 ms | 46.4 / 46.6 ms | 0 (bit identical) |
| 100000 | 530 ms | 307.5 / 318.1 ms | 218.8 / 223.7 ms | 0 |

Kernel only (events, 7 interleaved rounds, 20000 rows): row-major X 72.9 ms, feature-major 38.95, + int4 AoS records 33.1, + static preorder records 27.5, + __restrict__ 24.7 ms (1.58x vs feature-major, 2.9x vs row-major).
Rejected (kernel ms, same harness): block 64/256/512 and --maxrregcount 24/40/48/64 no change (+-0.5 percent); loop unroll 4 slower (28.4 vs 27.5); paired (value, feature) table entries 26.1 vs 24.7; shared-memory child masks 25.8 vs 24.7 (spill-free but 4 percent slower).
nvprof of the stack kernel (feature-major): occupancy 0.95, IPC 1.76, stall_memory_dependency 85 percent, gld_efficiency 44 percent, local_hit_rate 50 percent, warp_execution_efficiency 99.6 percent, dram 10 GB/s read 17 GB/s write (latency bound, not bandwidth bound).
Interaction pair-table kernel (3000 x 40 x 200 trees depth 4): numba 7088 ms, conditioned-scan path kernel 16648 ms, table kernel 55 ms (contended run); 20000 rows 252 ms best (frame stack) vs 280 ms (shared). Occupancy 0.24 at 3000 rows (too few threads).
Launches per call: 1 kernel (+ transpose, zeros). Host side (cProfile, interactions, 3000 rows): build_tables+pair tables 25 ms of ~110 ms, the rest waiting on the kernel/download.
