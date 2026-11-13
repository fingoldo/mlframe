# `_batch_pair_mi_cuda_fm.py` — feature-major byte-code pair-MI kernel

Measured 2026-11-10, GTX 1050 Ti (Pascal, 6 SMs, 4 GB), 100000 rows, `D:/Temp/bench_pairmi.py`, `D:/Temp/bench_pairmi2.py`.

| kernel | pairs | time | per pair | effective read |
|---|---:|---:|---:|---:|
| `batch_pair_mi_cuda` (numba.cuda, row-major int32) | 179700 | 18.9 s | 105 us | 2.8 GB/s |
| `batch_pair_mi_cuda_shared_fused` (RawKernel, row-major int32) | 40000 | 6.8 s | 171 us | — |
| `batch_pair_mi_cuda_feature_major` (this kernel), through `dispatch_batch_pair_mi_chunked` | 179700 | 0.60 s | 3.4 us | 89 GB/s |

Why the old kernels were slow: a warp reads `factors_data[i * n_features + a]` for 32 consecutive rows i, 601 bytes apart at 600 columns, i.e. 32 sectors fetched for 32 useful bytes (~3 % of the card's ~112 GB/s). The new kernel transposes the codes once on the device to `(n_features, n_pad)` unsigned bytes, rows padded to a multiple of four so each column is word-aligned; a thread loads four rows of a column as one 32-bit word (coalesced, full sectors), counts into a dynamic-shared int32 histogram and finishes with the shared-fused kernel's warp-shuffle reduction. Counts are exact integers, so the MI equals the row-major kernels' (max difference 1.5e-18 in the benchmark; `tests/feature_selection/gpu/test_batch_pair_mi_cuda_fm.py` checks 1e-12 relative).

Applies when every code, class and cardinality fits a byte; else `ValueError` and the dispatcher keeps the row-major kernels (`_PairMiDispatch.try_feature_major`, tried first inside the existing "cuda" decision, so the kernel-tuning-cache choice between njit and CUDA is unchanged; backend tag stays `cuda`). VRAM: the byte matrix is a quarter of the int32 one; the existing `vram_ok` gate runs first.

Effect on the 100k x 600 fit: the pair-MI dispatcher was 36 s of the 118 s `fe_step` (cProfile).

## 2026-11-13 pass: cProfile, nvprof, launches per call, column blocks, variants tried

Shape: 100000 rows x 100 byte-code columns x 2000 pairs, 21 bins per column, 20 classes (441 joint cells x 20 classes = 36 KB of shared memory per block), GTX 1050 Ti (sm_61, 112 GB/s).

* **End to end** through `batch_pair_mi_cuda_feature_major`: 58.7 ms per call (median of 5). **cProfile**: `cupy.ndarray.get`, the wait for the device, is 0.197 s of the 0.24 s over 5 calls; the Python of the launcher is a few milliseconds. Device-bound.
* **Launches per call**: one `pair_mi_feature_major` launch for all pairs, one `cupy_copy` for the device-side transpose (3.5 ms), one upload of the byte matrix and five small ones. nvprof summary (4 calls): the kernel 85.6 % of GPU time, 32.0 ms mean (26.5 to 36.9); the transpose copy 9.4 %; host-to-device copies 4.6 % (6.9 ms in total).
* **Metrics of the kernel** (nvprof, replayed): global load efficiency 99.8 % (four consecutive rows of a column per 32-bit word), multiprocessor activity 98.4 %, achieved occupancy 0.25 (256-thread blocks, one block per SM because the joint table takes 36 KB of shared memory), device-memory read throughput 9.8 GB/s of 112. The kernel is bound by the shared-memory atomics of the histogram, not by bandwidth: the loads are as coalesced as a byte layout allows and each SM works on one table at a time.
* **Variants tried, none adopted** (same inputs, MI equal to 3e-17): block size 256 / 512 / 1024 threads 31.7 / 40.7 / 31.9 ms (the run-to-run spread on this card is larger than any difference; unlike the row-major shared-fused kernel, whose 128-thread default was a real loss, this kernel's 256 is already at the floor); marginal counts derived from the joint table instead of the second atomic per row: 39.5 / 32.9 / 42.9 ms, no consistent gain. The byte layout and the word loads are what made this kernel 20x faster than the row-major ones (3.4 us per pair at 600 columns, above); what remains is atomic throughput.
* **Column blocks**: the byte matrix goes up whole and is transposed on the device while the VRAM cushion admits the referenced columns twice over; otherwise the pairs go in chunks touching a bounded set of columns (`pair_chunks_by_operand_block`), each chunk's columns gathered on the host and uploaded, the values equal to 1e-15 (`test_pair_mi_fm_column_blocks.py`).
