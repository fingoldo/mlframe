# _unary_credit_logit_cuda (bootstrap null of the unary credit for a binary target)

Module: `src/mlframe/feature_selection/filters/_unary_credit_logit_cuda.py`; callers: `_unary_credit_logit._test_column`. Measured 2026-11-24 on the development host (GTX 1050 Ti, sm_61, 6 SMs; i7-7700HQ, 8 logical cores),
GPU 67 C at the start of the profile run, SM clock 1721 of 1911 MHz, memory 3504 MHz, throttle mask 0x0, GPU utilization 2% before the run, nothing else heavy running. Tool: nvprof 12.8 (counters work since the
2026-11-24 permission change), cProfile for the host side.

## What it computes and why a kernel

The credit tests a column by the maximum over its candidate forms of the deviance gain of a logit `offset + a + b * z_f`; its null redraws the classes `D` times (50 per batch, up to 2000) and refits every
candidate on every draw. Per batch that is `K` candidates (13 at most with the zero row) times 50 draws of a two-parameter Newton fit (5 steps) over up to 10000 rows (`MAX_ROWS` 20000, a discovery half).

Versions kept (every one stays in the tree):

| version | where | note |
|---|---|---|
| numpy passes over `(rows, candidates, draws)` | `_unary_credit_logit._deviance` | the reference; six passes per Newton step, each over `n * K * D` cells (26 MB per temporary at 10000 x 13 x 50 in float32); the tests compare the others against it |
| njit twin | `_unary_credit_logit_cuda._loglik_njit` | one thread per (candidate, draw) pair, no `n * K * D` array, double precision sums; equals the numpy reference to 9e-13 |
| fused device kernel | `_unary_credit_logit_cuda.logit_null_loglik` | one block per pair; `{REAL}` (crit dtype) per-row terms, `{ACC}` = double log-likelihood sums |

## Equivalence

n = 3001, K = 12 + the zero row, D = 20, 5 steps (`tests/feature_selection/fe/unary_credit/test_unary_credit_logit_kernels.py` pins shapes from 1 row to 3001, 1 to 20 draws, both value types, near-separation and an offset of up to 30):
njit vs numpy max abs diff 9.1e-13 (LRT scale 124); device double vs numpy 2.3e-11; device float32 terms vs numpy 1.5e-5 (relative 1.5e-5).

## cProfile (host side), `credit_prof.py 100000 100`: 30 columns, 100000 rows, binary target, cProfile on, 14.0 s

| function | calls | cumulative s | note |
|---|---:|---:|---|
| `credit_columns_binary` | 1 | 13.85 | |
| `_test_column` | 62 | 12.59 | 0.2 s per tested column |
| `backend_choice` (kernel-tuning entry) | 62 | 4.24 | 2.36 s is the one-time load of the tuning cache and the numba cache of the njit twin (paid by the first KTC use of the process whoever it is); warm 0.034 ms per call (measured over 100 calls) |
| `_fit_offset_logit` (numpy Newton of the offset logit, 25 iterations of an `n x (m + 1)` design) | 124 | 3.62 | 2.0 s own time; the next host cost, not a device candidate (a 30-column design, `np.linalg.solve` on 31 x 31) |
| `loglik_host` (observed fits, 8 steps, njit twin) | 64 | 3.16 | 2.38 s of it numba cache loads at first use |

Without cProfile the same stress takes 12.2 s warm (109 s before the float32 / gated null / pass skip, 22.8 s with the first device version built from numpy-style cupy passes, 44 s per call of the stage in the
100000 x 100 all-families fit before any of it). In that fit the credit stage is now 1.2 s per call (stage ledger).

## GPU profile (nvprof), one null batch at the credit's shape: 10000 rows, 13 candidates + the zero row, 50 draws, 5 steps, float32 values

Kernels per call (the batch = draw the classes, fused kernel, LRT arithmetic, copy back): 7 launches and 1 device-to-host copy per batch (the generator fill, the comparison with the row probabilities, `logit_null_loglik`, four small
cupy elementwise kernels of the LRT arithmetic of 1 to 3 us each, one 600-byte copy). The fused kernel is 1 launch per batch where the numpy-style cupy version launched every pass of every Newton step (six passes per step, five steps, a final pass and the intercept-only model; not counted).

| kernel | calls | time per call | share of GPU time |
|---|---:|---:|---:|
| `logit_null_loglik` | 10 | 1.887 ms (1.833 to 2.112) | 95.3% |
| `cupy_less` (class draw comparison) | 10 | 52 us | 2.6% |
| generator fill (`curand` XORWOW) | 10 | 33 us | 1.7% |
| four small LRT kernels + the copy back | 50 + 10 | 1 to 3 us each | 0.4% |

Wall of one batch (draw, kernel, copy back, synchronize): 4.71 ms under nvprof, 3.1 ms without it (median of 40; the profiled batch is `_benchmarks/profiling/nvprof_unary_credit_null.py`). The kernel is 1.84 ms of it; the synchronizing copy (`cudaMemcpyAsync`, 1.2 ms on average
under nvprof) is where the host waits for the kernel, and the rest is the call overhead of the cupy launches (`cuLaunchKernel` 33 us each on average, the Python side of each call about 0.1 to 0.5 ms; CUDA.md section 7).
Moving the four small LRT kernels to the host (one copy of the 14 x 50 log-likelihoods, the arithmetic in numpy) measured 3.06 ms against 3.08 ms: no gain, kept on the device.

Launch statistics (`--print-gpu-trace`): grid (14, 50, 1) = 700 blocks, block 256 threads, 32 registers per thread, 924 B static shared memory, 0 B dynamic; no local memory (local load and store transactions 0).

Counters (`nvprof --metrics`, 10 launches):

| metric | value |
|---|---:|
| achieved occupancy | 0.974 |
| SM efficiency (multiprocessor activity) | 98.8% |
| executed IPC | 3.92 (peak 4) |
| issue-slot utilization | 91.0% |
| warp execution efficiency | 99.7% |
| global load efficiency | 94.7% |
| global store efficiency | 20% (one result per block; 0.16% of the transactions) |
| shared-memory efficiency | 4.7% (one 5-value exchange per Newton step, negligible) |
| DRAM read throughput | 2.6 GB/s average (the 14 x 10000 float32 candidate rows, the offset and the 50 x 10000 class bytes are 0.6 MB: they stay in L2) |
| DRAM write throughput | 0.4 GB/s |

Reading: the kernel is issue-bound, not memory-bound (IPC 3.92 of 4, issue slots 91% busy, occupancy 0.97 with 32 registers and 256-thread blocks; the limiter is the instruction stream of the 5 sums and the `tanh`
per row and step, not DRAM or shared memory). 700 blocks over 6 SMs is 117 blocks per SM, so there is no tail effect. A faster sigmoid (`ex2` and `rcp` intrinsics in place of `tanh`) is the one lever left on the
kernel; not tried, because the stage is 1.2 s of the 60 s fit and the kernel 1.9 ms of a 4.7 ms batch.

## Dispatch (kernel-tuning cache)

Entry `unary_credit_logit_null` (`kernel_tuner`, axes `n_samples` x `n_candidates`, `gpu_capable`), swept for real with `python -m mlframe.system.kernel_tuning_cache refresh unary_credit_logit_null` (6 cells, 7 regions
persisted, 5 steps, 50 draws, equivalence tolerance rtol 1e-3 / atol 1e-2 because the device sums float32 terms and the host double ones; the largest difference seen 1.4e-5). Host-in, host-out placement (the credit's
offset and candidate rows start on the host, the null gains are consumed there), so the timed device callable includes the upload, the kernel and the copy back. Times (median of 5, ms):

| rows | candidates (+ zero row) | njit twin | device, upload + kernel + copy back | device kernel alone (resident) |
|---:|---:|---:|---:|---:|
| 2000 | 5 | 21.2 | 0.74 | 0.31 |
| 2000 | 13 | 44.1 | 0.87 | 0.58 |
| 2000 | 25 | 72.8 | 1.27 | 0.94 |
| 10000 | 5 | 95.7 | 1.68 | 0.94 |
| 10000 | 13 | 202.0 | 2.34 | 1.95 |
| 10000 | 25 | 319.9 | 4.23 | 3.89 |

The device wins every cell of the grid by 28 to 86 times (the njit twin runs on all 8 logical cores), so the sweep chooses `cuda` everywhere; the fallback (`1e5` pair-rows, set from this table) applies until a host has its sweep.
The stored entry is checked against the code version of the njit twin: after any edit of `_loglik_njit` the regions are stale and `choose` answers from the fallback (and sweeps in the background) until `python -m mlframe.system.kernel_tuning_cache refresh unary_credit_logit_null` (or `ensure`) is run again; the first benchmark run after the final edit logged `no matching region` for this reason. The
transfer share of the device time is 5 to 45% (the resident kernel alone against the host-in time): residency upstream would not pay for a refactor. `MLFRAME_FE_CREDIT_GPU=0` keeps the null on the host.

## Memory bound and fallback

No `n * K * D` array exists on either backend: the device working set is `(K + 1) * n * itemsize + D * n + K * D * 8` bytes (0.6 MB at the shape above), checked by `fits_device` against a quarter of the free
device memory; a call that does not fit, or a device error, returns to the njit twin (`try_loglik_device` returns `None`, throttled warning). Out-of-range or non-finite inputs do not reach the kernel: the credit skips
columns with non-finite values, and the Newton update guards its determinant.

## Not done

- Having the kernel draw its own classes (a counter-based generator in the block) would remove the generator and comparison launches and the 0.6 MB class array (1.2 ms of the 3.1 ms batch is the draw, the copy and the launches); not done: a batch is 3 ms and a column needs one to a few dozen of them.
- A device twin of `_fit_offset_logit` (3.6 s of the 12.2 s stress, host numpy Newton on a 31-column design): a candidate for an njit version first.
- `ex2` / `rcp` sigmoid in the kernel (see above).
- The multiclass credit has no null yet.
