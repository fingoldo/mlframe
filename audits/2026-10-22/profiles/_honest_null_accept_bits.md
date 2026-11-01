# _honest_null_accept_bits (binary-target honest-null acceptance by a sorted-space bitset)

Kernel `root_split_accept` in `src/mlframe/feature_selection/filters/_honest_null_accept_bits.py`, dispatched first by `_honest_null_accept._continuous_accept_block` for a binary target (`MLFRAME_HONEST_ROOT_SPLIT=accept`, the default; `k5`, `k4`, `k2` select the per-step kernels, which also run when the bitset does not fit or the permutations hold different numbers of label-1 rows).

What the caller needs is only whether each (column, permuted target) pair's best root split is accepted: `p(g) * max(1, ncand) < alpha`, `p = erfc(sqrt(n g))`, i.e. `g > erfcinv(alpha / ncand)^2 / n`. The kernel packs the labels along the sorted column into `n / 32` words (warp ballots over a shared row-space bit table), counts the valid cuts with popcounts, bounds the gain of each word from its prefix
(mutual information `<= log(1 + phi^2) <= phi^2`, computed in single precision with a 1e-3 margin) and scans exactly, with the k4 expression and tie rule, only the words that can hold a cut above the bar. Accept booleans and the (gain, count, index) of the accepted pairs equal the exact full scan (`tests/feature_selection/fe/test_honest_null_accept_bits.py`: planted signal, ties, a constant column, 7% and 95% targets, n not a multiple of 32; `test_honest_null_device_equivalence.py` through the dispatcher).

Variants tried (GTX 1050 Ti, 60 columns x 200 permutations x 100000 rows, one process, the machine loaded by a test run, so read the ratios):
- Scatter of the label-1 rows into the shared bitset with `atomicOr` (rank of each row in the sorted order): kernel about 247 ms; shared memory efficiency 4.9%, warp execution efficiency 61%, occupancy 0.61: bound by the shared atomic throughput (about 4 cycles per atomic per SM). KEPT AS A RECORD, NOT IN THE CODE.
- Ballot assembly (current): each warp reads 32 sorted positions, looks the row's label up in the shared bit table and packs the word with `__ballot_sync`, four independent steps per iteration. Ballot phase 97 ms to 41 ms; whole kernel about 150 to 170 ms.
- Float bound instead of a double `log1p`: no change in the loaded timing (the double cost was not the limiter); kept because it is cheaper on parts with a 1/64 double rate.
- Serial thread-0 scan and final reduction replaced by warp shuffles; block of 512 threads from the device limits (see below): 147 ms against 162 ms (256), 170 ms (1024), 220 ms (128).

Against the per-step kernels in the same process: exact k4 about 520 to 880 ms, float k5 about 390 to 610 ms, this kernel about 250 to 350 ms including the host preparation (valid-cut bits, threshold table, packed targets). The full-stage effect (250 columns x 999 permutations, `honest_null_columns_device` 8.8 s) is NOT yet measured on a quiet machine.

GPU profile (nvprof, current version, width 512): achieved occupancy 0.497 (2 blocks x 512 threads per SM, as the arithmetic predicts: 25 KB of shared memory per block against 96 KB per SM), multiprocessor activity 95.5%, warp execution efficiency 86.9%, shared memory efficiency 24.9%, global load efficiency 65%, device memory utilization low.
Launch width: `_gpu_hw_launch.fixed_shared_block_size` from the kernel's 47 registers and 528 B static + 25 KB dynamic shared memory (CUDA.md section 14); `MLFRAME_GPU_BLOCK_HONEST_ACCEPT` overrides.
Limits: the bitset must fit the shared budget (about 22 KB per table, n up to about 180000 rows); beyond it the per-step kernels run.
