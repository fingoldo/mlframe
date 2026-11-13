# `_batch_pair_mi_cuda_shared_fused` (single-launch pair MI, one block per pair, dynamic shared joint table)

Measured 2026-11-13, GTX 1050 Ti (4 GB, 112 GB/s, sm_61), 100000 rows x 100 int32 code columns x 2000 pairs, 21 bins per column (441 joint cells), 20 classes (8820 joint cells, 36 KB of shared memory per block).

## What changed in this pass and why

1. **Block size by the occupancy rule.** The launch used a fixed 128 threads. One block per pair holds the whole joint table in shared memory, so the table leaves room for a block per SM, and the threads of that one block are what hide the latency of the shared atomics. Kernel time in isolation, same inputs: 128 threads 83 ms, 256 57, 512 48, 1024 38 ms. `threads_per_block` now defaults to `None` and `block_size_for(kernel, dyn_smem=..., max_width=1024)` picks the widest block the table allows. End to end through the public function: 124 ms with 128 threads, 70 ms with the rule.
2. **Marginal counts from the joint table.** The second shared atomic per row (the counts of the joint code) is replaced by a row-sum over the finished joint table (`max_joint` threads, `n_classes_y` cells each): 2 to 8 percent of the kernel, values unchanged.
3. **Rejected: byte-wide codes** (the kernel reading `unsigned char` columns and labels): 46.7 against 47.3 ms at 512 threads, no gain; the kernel is bound by the shared atomics, not by the bytes it reads.
4. **Column blocks.** The whole narrowed matrix still goes up once and the device gathers the referenced columns (3x cheaper on this host than gathering on the host first: 69.6 against 198.8 ms). When the VRAM cushion refuses the referenced set, the pairs go in chunks touching a bounded set of columns (`pair_chunks_by_operand_block`) and only each chunk's columns are gathered on the host and uploaded; equal to the single upload to 1e-15 (`test_pair_mi_shared_fused_column_blocks.py`).

## cProfile (main thread, 5 calls after warm-up)

105 ms per call; `cupy.ndarray.get` (the wait for the device) is 0.444 s of the 0.472 s spent in the launcher over 5 calls, the Python of the launcher is about 1 ms. Device-bound.

## nvprof, GPU summary (4 calls)

| | per launch | share |
|---|---:|---:|
| `pair_mi_shared_fused` | 64.4 ms mean (min 33.1, max 75.6; the first launches run at ramping clocks) | 79.4 % |
| host-to-device copies (29: the narrowed matrix, the pair vectors, the labels) | 26.2 ms in total | 8.1 % |
| `cupy_copy__int32_int32` (widening) | 6.3 ms | 7.8 % |
| `cupy_take` (the column gather) | 3.8 ms | 4.7 % |
| device-to-host copies | 17 us | 0 % |

Launches per call: one kernel launch for all pairs, three small device kernels for widening, gather and transpose, one upload of the matrix and five small ones.

## nvprof metrics of the kernel (replayed)

Achieved occupancy 0.987, multiprocessor activity 97.3 %, global load efficiency 99.8 %, device-memory read throughput 36.9 GB/s of 112, shared-memory load throughput 63 GB/s. The device is busy with the atomics of the histogram, not saturated on bandwidth.
