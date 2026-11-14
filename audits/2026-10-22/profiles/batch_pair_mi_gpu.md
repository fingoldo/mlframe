# batch_pair_mi_gpu.py (pair MI dispatcher) with its carved module `_batch_pair_mi_cuda_kernels`: profile record (2026-11-14)

Family: `batch_pair_mi_gpu` (the four backends of the batched pair MI, the per-host dispatcher and the VRAM guards) with `_batch_pair_mi_cuda_kernels` (the numba.cuda kernels: one block per pair, the row-chunked variant, the zero fill). The two feature-major and shared-fused backends have their own records (`_batch_pair_mi_cuda_fm.md`, `_batch_pair_mi_cuda_shared_fused.md`). Idle machine, GTX 1050 Ti, synthetic inputs of the dispatcher's own sweep (`_make_batch_pair_mi_inputs`: 8-bin features, 4 classes).

## cProfile

One `dispatch_batch_pair_mi` call (500000 rows, 64 pairs): 0.146 s under the profiler, of which the tuned backend choice (`_BPMI_SPEC.choose`, the cache lookup) 0.083 s on the first call of the process and the chosen device backend (`batch_pair_mi_cuda_feature_major`) 0.060 s; the Python around the kernels is negligible, a call is its kernel.

## Kernels (nvprof, one call each of the numba.cuda and the cupy backend, 500000 x 64)

Kernel launches per call: the numba.cuda backend 1 launch for all pairs (21.3 ms, 29.3 percent of the profiled GPU time); the cupy backend 64 pairs x about 8 launches, 576 in all (`cupy_add` 128 launches 17.9 ms, `cupy_multiply` 128 launches 15.5 ms, `cupy_bincount_kernel` 64 launches 8.4 ms, `cupy_less` 64 launches 3.1 ms, two reductions of 64 launches each 1.8 and 1.6 ms): the cupy backend is launch-bound, the numba.cuda one is not.

## Occupancy and throughput

nvprof counters were not collected on 2026-11-14 (`Internal profiling error 4211` for every kernel today). The numba.cuda kernel runs one block per pair with 128 threads (`threads_per_block=128`) and a shared joint histogram, so a 64-pair call gives 64 blocks for 6 SMs, 10.7 blocks per SM (at most 8 resident per SM at 128 threads, about 1.3 waves), and a 256-pair call 256 blocks, 43 per SM: the small calls leave a tail wave, the large ones fill the card. Throughput from the timings: the 2 000 000 x 256 call reads the int32 factor matrix columns of its pairs (2 x 2 000 000 x 256 x 4 B = 4.1 GB) in 469 ms, 8.7 GB/s of the card's 112 GB/s: that is consistent with a kernel bound by the shared-memory atomics of the joint histogram, as the other histogram kernels of the code base are; this was not isolated here (counters unavailable).

## Host-or-device choice

`_BPMI_SPEC` (`kernel_tuner`, sweep over n_samples 50000 to 2 000 000 x n_pairs 16, 64, 256) picks per host among `njit_serial`, `njit_parallel`, `cuda` and `cupy`. Medians of 5 calls, ms (njit serial / njit parallel / numba.cuda / cupy): 100000 x 64 pairs 70.1 / 68.1 / 9.1 / 127.1; 500000 x 64 358.0 / 348.8 / 29.0 / 133.7; 500000 x 256 1419 / 1446 / 102.7 / 548.8; 2000000 x 256 5614 / 5627 / 469 / 1093. The numba.cuda backend is 7.4 to 12.3 times the njit kernel from 100000 rows and 64 pairs on, the cupy backend never wins here; the host twin is the njit `batch_pair_mi_prange` of the `info_theory` package (numerical reference, the sweep requires every backend to agree to 1e-9 relative).

## Streaming

A call whose full upload would not fit the free VRAM (`_gpu_upload_fits`) does not fall to the host: `batch_pair_mi_cuda_row_chunked` uploads `factors_data` in VRAM-sized row blocks and accumulates the joint histogram across them (bit-identical result), and the pairs of a large call are split into chunks sized from the free memory (`_fallback_pair_chunk_size`): the column blocks of this family are pair chunks and row blocks.
