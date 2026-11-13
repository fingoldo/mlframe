# `_family_maxt_gate_device` (the family-wise permutation test on device-resident codes)

Measured 2026-11-14, GTX 1050 Ti (sm_61), three signal-free batches of `k` int8 columns plus one signal batch of 3, 4 classes, 8 bins, 999 permutations, `alpha` 0.01. Kernel launches, occupancy and throughput of the bit-set null kernels the module drives (`family_null_bits_v2` / `v3`, `family_null_launcher`) are recorded in `_family_maxt_cuda_bits.md` (kernel launches per call: one per block of permutations; achieved occupancy, device-memory throughput and the whole-GPU utilization there); this module is the orchestration around them (packing the codes into bit sets on the card, the observed MI, the analytic triage, the early-stopping permutation loop on small vectors).

## Dispatch is by data residency, and the resident placement wins everywhere measured

Whole call, seconds, three placements of the same family:

| rows | columns per noise batch | device-in, device gate | device-in, copy back, host gate | host-in, host gate |
| --- | --- | --- | --- | --- |
| 20000 | 30 | 0.009 | 0.013 | 0.012 |
| 100000 | 30 | 0.031 | 0.059 | 0.053 |
| 100000 | 100 | 0.032 | 0.136 | 0.109 |
| 400000 | 30 | 0.029 | 0.297 | 0.321 |

The host gate is the same arithmetic with the codes uploaded for the null kernel, so its time is dominated by the copies (at 400000 rows the device gate is 10x faster, and its time barely moves with the row count because only small vectors cross the bus). The gate is therefore not a crossover to tune: `family_pass_batches` takes the device gate whenever the batches are resident device arrays (`LazyHostCodes` over cupy arrays) and declines to the host gate otherwise, and no measured size favours the other placement. The value of keeping the codes resident is exactly this table: 1.4x at 20000 rows, 1.9 to 4.3x at 100000, 10x at 400000, measured against what the host route costs when the codes had to cross the bus.

## Host profile (cProfile, 100000 rows x 30 columns, device-in)

The device gate's Python is the orchestration loop (a few small `.get()` reads per block of permutations); the bulk of the 0.031 s is the null kernel it waits for, which the kernel record covers.

## Streaming

The bit sets of the whole family must fit the card (`_NotApplicableError` when they would not, then the host gate answers); the kernels themselves run in blocks of permutations (`fill_block_width`).
