# `_fe_gpu_batch` (device half of the FE batcher: resident edge-binned plug-in MI, VRAM column chunks, multi-GPU packing)

Measured 2026-11-16 on the host and bench of `_fe_cpu_batch.md` (GTX 1050 Ti, sm_61; GPU 80 C, SM 1695/1911 MHz, no throttle; CPU load 18 %). Whole-call figures against the numba path: 100000 x 400 in 0.067 s against 0.771 s, 100000 x 2000 in 0.578 s against 3.983 s. Host profile (cProfile): the call is `multi_gpu_fe_batch_mi` -> `gpu_fe_batch_mi`, whose Python time is the chunk loop and the cupy kernel launches; the device work below is the cost.

## Launches per call and the GPU profile (nvprof, 100000 x 2000, device path)

The device call takes 0.586 s of which the host-to-device copy of the matrix is 46 % of the GPU time (526 ms for 14 copies, 800 MB over the pageable PCIe path), then `radix_select_f32_v5` 12.0 % (8 launches, 17 ms), the NaN scrub (`cupy_where` 9.3 %, `cupy_isfinite` 6.6 %, `cupy_copy` 5.9 %), `transpose_f32` 6.1 %, `mi_hist_split_f32` 5.9 % (8 launches of 8.5 ms): 8 launches of each kernel for the 8 VRAM column chunks, so launches per call are about 60. Occupancy: `mi_hist_split_f32` 0.985 with 47 GB/s device-memory read (42 percent of 112 GB/s), `cupy_where` 0.91 at 43 GB/s, `mi_from_counts_col_f32` 0.45. The copy dominates, so whole-GPU utilization is about 50 % kernel and 50 % transfer; an already-resident candidate matrix (the production caller `collect_and_score` hands over host families) would skip the copy.

## Backend choice and memory

Choice: `fe_batch_backend_choice` of `_fe_batch_dispatch_ktc` (STRICT forces the device). Memory: VRAM column chunks of `_gpu_k_chunk`; the (n, chunk) float block, its transpose and codes are the only device buffers, never the full width.
