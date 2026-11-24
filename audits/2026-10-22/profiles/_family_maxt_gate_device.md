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

## Distinct partitions only: `_distinct_columns_on_device` and `_family_columns_gather_cuda` (2026-11-24)

A pair family at 100000 rows is 1944 to 2166 single-column batches of the 30000-row screen subsample (the default pair search of the all-families fit, five families per fit), and 80% of its columns are distinct
partitions (1559 of 1944 in the dumped family; monotone transforms of the same operands repeat their codes, a decreasing one reverses them). The host gate already dropped the repeats (`_family_columns_dedupe.unique_columns`);
the device gate scored all of them. Measured on that dumped family (nvprof 12.8, GTX 1050 Ti, 67 C, SM 1721 of 1911 MHz, no throttle, one warm call in the cupy profiler window):

| call | wall | null kernel (`family_null_bits_v3`, 33 launches) | dedupe overhead |
|---|---:|---:|---:|
| all 1944 columns, no dedupe (`--no-dedupe`) | 1.042 s (1.016 s in an earlier run) | 0.814 s (24.7 ms per launch) | none |
| dedupe through `cp.concatenate` + `ascontiguousarray(block.T)` + `unique_columns` + `cp.take` | 1.016 s | 0.634 s | 62.8 ms generic transposing copy (`cupy_copy__int8_int8`) + 29.5 ms 1944-way concatenate + 8.8 ms take: 101 ms, half the saving |
| dedupe through the two gather kernels below | 0.907 to 0.920 s | 0.637 to 0.641 s (19.3 ms per launch, 22% less than without) | 3.0 ms `transpose_multi` + 5.7 ms `gather_multi` + the host hash (24 ms for 1944 x 30000, `unique_columns`) + 14.3 ms device-to-host copy |

The flags are the same as without the dedupe (1659 of 1944 pass; `tests/feature_selection/gpu/test_family_maxt_device_dedupe.py` pins equality on repeats that are identical and reversed, against the device gate without
the dedupe and against the host gate). The analytic triage still counts every column of the family in its Bonferroni sum (the `inverse` argument): summing over the distinct columns only let batches through that the
exact test rejects (1402 of 1559 distinct passed there against 1659 of 1944).

### Kernels (`_family_columns_gather_cuda.py`; record of the type placeholder: `{DT}`, the integer code type of the batches; no real-valued operand)

`transpose_multi` reads every column of the family in place through a per-column address and stride table (built on the host: no kernel searches for the batch of a column) and writes the family as `(columns, rows)`
through a 32 x 33 shared tile (padded against bank conflicts): coalesced reads of adjacent columns of a row and coalesced writes of adjacent rows. `gather_multi` writes the chosen columns as a row-major
`(rows, chosen)` matrix, the layout of the bit-set packing kernels. One launch each for the whole family, whatever the number of batches (1944 here; the concatenate it replaces was 1944 copies).

First version (pointer table plus a per-thread binary search for the batch of a column) against the final (per-column table), counters from `nvprof --metrics`:

| kernel | version | time | occupancy | IPC | global load efficiency | global store efficiency |
|---|---|---:|---:|---:|---:|---:|
| `transpose_multi` | binary search | 9.56 ms | 0.735 | 1.99 | 55% (15.2 transactions per request) | 64% |
| `transpose_multi` | column table (kept) | 2.97 ms | 0.698 | 3.79 | 86.9% | 66.7% |
| `gather_multi` | binary search | 16.8 ms | 0.861 | 2.11 | 49.2% | 48.3% |
| `gather_multi` | column table (kept) | 5.69 ms | 0.803 | 2.09 | 78.4% | 49.8% |

Launches per call: 2 (`transpose_multi`, `gather_multi`) against 1944 + 3 before; the whole gate call is 33 null-kernel launches, 1 `pack_bins_multi`, 1 `colmax_multi`, the 2 gathers and 5 small cupy kernels. Share of the
call: the null kernel 90.7% of the GPU time (the `family_null_bits_v3` record, `_family_maxt_cuda_bits.md`, covers it: occupancy 0.365, IPC 1.94, global load efficiency 87.9%), the two gathers 1.2%, the copies
(36 device-to-host, 41 host-to-device) 3.4%. Store efficiency of 50 to 67% is the one-byte codes: a warp writes 32 bytes of a 128-byte line; the kernels are 0.9% of the call.

Other counters seen in the same window and not touched here: `colmax_multi` (2.3 ms, 0.3% of the call) has global store efficiency 2.8% and achieved occupancy 0.13; the null kernel's own stores are 35% efficient
(its output is the `(permutations, columns)` block, not a hot path).

Profile script: `_benchmarks/profiling/nvprof_family_gate_dedupe.py`.
