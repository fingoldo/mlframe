# _fe_accuracy_gate_kernel

## Fused gate-probe kernel: profile and honest win axis (2026-11-11)

### Register pressure (2026-11-11, `tests/test_meta/test_cuda_kernel_register_pressure.py`)
`cupy`'s compiled `local_size_bytes`/`num_regs` (the cubin numbers, equivalent to `ptxas -v` without parsing its text) on float32: the thread-private Hessian (`h`) and the thread-0-only
Cholesky factor (`L`) were stored as full `MAXP x MAXP` arrays but only ever read/written in their lower triangle (the accumulation loop is already `for b <= a`); packing both into a
triangular `MAXTRI = MAXP*(MAXP+1)/2` layout cut the spill from 1512 to 936 bytes/thread (72 -> 70 registers) for free, with `maxdiff 0.0` against the host on the existing 100-combo
correctness test. The remaining 936 bytes come from the per-row `g[MAXP]`/`x[MAXP]`/`h[MAXTRI]` work vectors in the gradient-accumulation loop, which the compiler cannot keep in registers
because `k` (0..`MAXK`) is a runtime kernel argument, not a compile-time unroll bound. Removing it would need a second column pass (recompute `z` without storing `x[]`), doubling global
reads per Newton iteration on a kernel that already had one silent-divergence incident (see below) — not worth it for an already compute-bound kernel per CUDA.md section 3 ("a spill is not
worth a higher occupancy"). Tracked as a reasoned exemption in the test, not a blanket skip.

### Host profile (unchanged from `_fe_accuracy_gate.md`)
Per-candidate host probe ~26 ms (10 folds x standardise + Newton + AUC); the Newton kernel itself is 1.5 ms/fold, 8 Newton iterations typical. At N candidates sharing the same subsample rows
and folds (the gate's pass 1), the host cost is N x 26 ms, parallelised over the edge-jobs pool.

### Device profile (nvprof, no elevated shell needed — CUPTI on PATH)
`nvprof --metrics achieved_occupancy,sm_efficiency,ipc,warp_execution_efficiency` on 96 designs x 10 folds x 5000 rows, k=2 (one base + one engineered column), block=128, grid=(96,10):

| metric | min | max | avg |
|---|---|---|---|
| achieved_occupancy | 0.4145 | 0.4357 | 0.4343 |
| sm_efficiency | 94.5% | 98.1% | 97.7% |
| IPC | 0.424 | 0.434 | 0.429 |
| warp_execution_efficiency | 59.7% | 59.8% | 59.7% |

`nvprof --print-gpu-trace`-style summary: 21 launches (20 timed + 1 warm-up), 99.68% of GPU activity is this one kernel, mean 100.2 us... no — **mean 100.18 ms per launch** (960 blocks of
128 threads), min 3.8 ms (an early under-cache run), max 110.5 ms. One launch per call (`cuLaunchKernel` count: 1), covering every (design, fold) pair — the fusion goal of CUDA.md section 2.

### Reading the numbers
SM efficiency is high (97.7%: the card's 6 SMs are almost always running some block) but IPC is low (0.43) and warp efficiency is 59.7%: classic **warp divergence**, not a memory-bound
kernel. The cause is in the design, not a coding slip: the Newton loop's Cholesky factorisation and the line-search loss reduction are done by **thread 0 alone** (`if (threadIdx.x == 0) { ... }`
blocks), with the other 127 threads of the block idling at the surrounding `__syncthreads()` barriers while thread 0 runs the sequential (p x p) Cholesky solve (p = k+1 = 2 or 3 here — too
small to parallelise across threads without more overhead than it saves). The 128-wide parallelism pays off in the row-walk passes (mean, deviation, gradient, Hessian accumulation, AUC
pairwise count), which read `n` rows in a coalesced stride; it does NOT pay off in the small dense linear-algebra step, which stays single-threaded by design.

### Honest wall comparison
At 96 designs x 10 folds x 5000 rows the fused kernel is 104-107 ms per call (single `nvprof`-quiet run) against about 96 x 26 ms / 8 workers = 312 ms for the per-candidate host probe on an
8-way thread pool (silghtly favourable to the host here: the pool parallelises 96 independent calls across 8 physical threads at ~26 ms each). The kernel wins when host parallelism is NOT
available — i.e. when the gate's probe pool is shared with other concurrent FE lanes, which is the actual situation at 1000 columns (the pool is busy with other families). It is not a universal
win at small candidate counts measured ALONE in a quiet process; the dispatcher threshold (`_MIN_CANDIDATES`, to be moved to a KTC-tuned entry per CUDA.md section 4) must reflect the
concurrent-pool case, not the quiet-process A/B.

### Correctness incident and fix (same day)
The first integration gave silent `NaN` on 4 of 100 (design, fold) pairs where the host's `fit_logistic_ridge_newton` (the njit `_newton_kernel`) converged cleanly. Root cause: the kernel's
line-search acceptance test was `f_new <= f_cur` (strictly non-increasing), while the host's njit kernel accepts `f_new <= f + 1e-11 * abs(f)` — a tolerance against float32-rounding noise near
the optimum. Without it, the kernel's line search spuriously exhausted its 12 halvings a few iterations before the gradient norm would have reached the stop tolerance, exactly on inputs the
host solves in 4-8 iterations. Fixed by copying the host's tolerance verbatim; after the fix, 10 base designs and 10 augmented designs across 10 folds (100 AUC values each) match the host
bit for bit (`maxdiff 0.0`), and every keep/drop verdict agrees. Lesson for the next fused probe kernel in this family: port every numerical tolerance from the exact host kernel it mirrors,
not from the paper formula or the python reference twin — the njit kernel and the `_numpy` reference twin in this module differ in this one tolerance, and only the njit kernel is what the
host probe actually calls.

### Contract status
`_fe_accuracy_gate_batched.py` is `pending` in `tests/test_meta/_fe_gpu_contract.json` (CUDA.md section 2 ratchet): it has the CPU-twin Newton (`fit_logistic_ridge_newton_numpy`), the fused
`{REAL}`/`{ACC}` RawKernel (this module), a measured fallback (any device fault returns `None`, the caller keeps the per-candidate host probe) and this profile note. It is not yet `compliant`:
KTC dispatch (section 4) still uses a hand-set `_MIN_CANDIDATES` threshold rather than a tuned entry, and the VRAM-block sizing (`_device_block_designs`) has not been tested against an
actual OOM fallback on this 4 GB card.
