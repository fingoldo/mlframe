# `_orth_mi_backends` (batched plug-in MI of the orthogonal FE stage: backend choice and column-block routing)

Measured 2026-11-16, GTX 1050 Ti (sm_61, 112 GB/s). Shape: 100000 rows x 100 float32 columns, 2 classes, 10 bins, one `_mi_classif_batch` call.

This module chooses and routes; the kernels are `hermite_fe` / `_hermite_fe_mi` (the numba plug-in MI `plugin_mi_classif_batch_njit`, parallel over columns, the CPU twin) and the resident CUDA MI of `_hermite_fe_mi` (`_plugin_mi_classif_batch_cuda_resident`; radix-select edges, `mi_hist_split`, `mi_from_counts_col`), both under the contract with their own records and kernel-tuning entries. The path (`resident` / `streamed` / `cpu_blocks`) is `_streamed_gpu_mi.choose_mi_path` from the free VRAM and RAM.

## Host profile (cProfile, 5 calls, CPU path)

Before: 62.8 ms per call, 0.33 s for 5 calls, 1466 function calls: 0.261 s (52 ms per call) was `column_fingerprints`, the content hash that keys the per-column MI memo; the MI itself came from the memo on the repeated input. The hash walked down each column of the row-major matrix (4 of the 64 bytes of every cache line used). Now it hashes 16 neighbouring columns in one pass over the rows, the same fingerprints bit for bit: 63.0 -> 7.9 ms (float32) and 93.1 -> 13.6 ms (float64) for 100000 x 100.

## Launches per call and the GPU profile (nvprof, 6 calls, resident edge-binned path)

Launches per call: 1 `radix_select_interp_f64_v3` (5.2 ms), 1 `transpose_f64` (1.7 ms), 1 `mi_hist_split` (1.5 ms), 1 `cupy_copy__float32_float64` (1.2 ms), 1 `mi_from_counts_col` (0.08 ms) and the host-to-device copies: 5 kernel launches per call, about 10 ms of kernel time against 7 ms of copies (the copy is 42 % of the GPU time: the host-built input is uploaded per call). Throughput: `mi_hist_split` reads 40 MB in 1.5 ms (about 27 GB/s of the card's 112 GB/s); `radix_select_interp_f64_v3` is the dearest kernel at 31 % of the GPU time. Achieved occupancy of these kernels is in `_hermite_fe_mi.md`.

The rank-binning resident path (`rank_binning=True`, STRICT) is much dearer on this card: 220 ms per call, of which `cupy argsort` 100 launches per call (one per column, 1 ms each, 0.50 s for 5 calls) and the blocking `.get()` 0.38 s: a batched sort of the whole matrix is the open item (tracker CT11-93).

## Backend choice and streaming

The choice: `MLFRAME_NUMBA_MI` env or the trial import of the numba dispatcher (`_select_mi_backend`), STRICT-GPU through `_orth_mi_gpu_enabled`, then `choose_mi_path` per call. Streaming: `cpu_blocks` scores the CPU path in column blocks sized to free host RAM, `streamed` is the device twin in column blocks sized to free VRAM; the resident path is chosen only when the batch fits.
