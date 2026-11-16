# `_fe_additive_fusion` (host proposal of additive fusions of unfused engineered halves)

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores; hardware: GPU 83 C, SM 1683/1911 MHz, no throttle, CPU 2808/2808 MHz, load 33 %. Bench and the device twin: `_fe_additive_fusion_gpu_resident.md`.

## Host profile (cProfile, 100000 rows, six halves, 3 fusions admitted)

Before: 14251 function calls, 0.313 s: `_evaluate_fusion_pair_weighted` 0.174 s over 3 calls, of which `_multiple_r` 0.123 s over 12 calls (an n x k `lstsq` per fit) and `_quantile_bin` 0.115 s over 18; `_subsumed_raw_operand_check` 0.071 s over 3. After: the four separability fits of a pair come from one centred Gram matrix of `[a, b, fused, y]` (`_pair_multiple_r`, `multiple_r_from_gram`), proposal 0.299 -> 0.194 s.

## Launches per call and the GPU profile

The host path launches no kernel; the device twin's launches per proposal (about 330), nvprof table, occupancy and throughput are in `_fe_additive_fusion_gpu_resident.md` (whole-GPU time about 34 ms of 220 ms wall). Choice between the paths: `fe_additive_fusion_backend`.
