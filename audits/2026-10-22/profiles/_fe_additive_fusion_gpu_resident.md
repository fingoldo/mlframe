# `_fe_additive_fusion` / `_fe_additive_fusion_gpu_resident` (+ carved `_fe_additive_fusion_ktc`): additive fusion of two unfused engineered halves

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores. Hardware at the end of the runs: GPU 83 to 84 C, SM 1683/1911 MHz, mem 3504 MHz, throttle none; CPU 2808/2808 MHz, load 32 to 34 %.
Bench: `propose_additive_fusions` / `propose_additive_fusions_gpu` on 100000 rows, six squared-Hermite halves, an 8-class target of their noisy sum (3 fusions admitted on both paths), min of 3 after a warm-up.

## Whole pass, by path

| path | wall per proposal |
|---|---|
| host before (numpy `lstsq` per fit) | 0.299 s |
| host after (one centred Gram matrix per pair, `multiple_r_from_gram`) | 0.194 s |
| device before (double design, normal equations) | 0.288 s |
| device after (one float32 Gram matrix per pair) | 0.221 s |

The host `_multiple_r` (0.123 s of the 0.313 s host cProfile: 12 `lstsq` fits) got the same Gram form, so the host is now ahead of the device. The `fe_additive_fusion_backend` sweep (host / device, identical admitted set, 0 difference) chooses the host at 5000, 20000, 100000 and 400000 rows (GPU 83 C, SM 1683/1911 MHz, no throttle); the fallback before a sweep is the host.

## Launches per call and the GPU profile (nvprof, device path, 4 proposals)

Before: 54 % of the device time was the double `dgemm_largek` (30.5 %) and `gemv2T` (23.6 %) of the OLS design (`design.T @ design`, 48 launches each, 1/32 double rate on this card). After: one float32 `matmul` of the stacked `[a, b, fused, y]` per pair and a 4 x 4 host solve; the top of the table is now the cub merge sort of the quantile binner (16.7 %, 168 launches), the radix sorts (7 % each), `_nanvar_core` 6.8 %, `marginal_mi_entropies` 4.2 % (48 launches of 118 us). Launches per proposal: about 330. Whole-GPU time is about 34 ms of the 220 ms wall (device busy about 15 %): the proposal is host-latency bound (`cp.unique` 0.057 s of 0.244 s cProfile, `_gpu_quantile_bin_codes` 0.085 s over 10 calls, `_subsumed_raw_operand_check_gpu` 0.119 s over 3).
`marginal_mi_entropies` (the batched MI kernel of `_fe_batched_mi`): achieved occupancy 0.125, device-memory read throughput 5.8 to 19.5 GB/s (mean 8.4 GB/s) of the card's 112 GB/s, so it is small-launch latency bound at this size; its own record is `_fe_batched_mi.md`.

## Backend choice and memory

Choice: `fe_additive_fusion_backend` (kernel-tuning entry of `_fe_additive_fusion_ktc`, `MLFRAME_FE_FUSION_DEVICE=0|1` forces a side), reached when the AUTO / STRICT fit state is engaged; STRICT-resident forces the device. Memory: the scoring row block (`MLFRAME_FE_FUSION_MAX_ROWS` subsample) of the candidate halves' columns, bounded by the cap times the half count; the full-length column is built only for an admitted fusion.
