# _cmi_null_strata_gpu (per-stratum count of the conditional permutation null)

Module: `filters/_cmi_null_strata_gpu.py`, kernels `strata_xy_entropy` (direct) and `strata_xy_entropy_rows` (rows), called from `conditional_perm_null_gpu` after the flat within-stratum shuffle
(GPU11-77). They replace the dense `batched_cmi_gpu` of the shuffled columns once the (x, y, z) joint no longer fits shared memory (Kz of about 5000 and more at 100000 rows).

## Why
cProfile of a default 100000 x 100 fit: `_gate_round` 18.6 of 104 s, all of it 420 per-candidate nulls of about 45 ms. After the flat shuffle (sort 15.5 -> 5 ms) and the shared (x, z) joint
(`precomp_xz`) the remaining 15.7 ms of a 25-permutation null was `batched_cmi_gpu`: the (x, y, z) count went through about 2.5 million scattered global atomics into 25 tables of Kx * Kyz cells.
In the z-sorted order every stratum is a contiguous run of a few rows, so its (x, y) counts fit a small histogram in shared memory.

## Kernels (CUDA.md section 3)
- Both: 128 threads (a power of two, tree reduction of the entropy partials), grid `(ceil(S / 128), permutations)`, integer counts, `{ACC}` (float64) sums rendered through `render`, one module per process,
  `kernel(cp, real)` / `kernel_rows(cp, real)` factories in the register-pressure registry (32 registers, `local_size_bytes` 0 each), a table `ent[v] = -(v / n) ln(v / n)` instead of a double `log` per
  cell (15.1 ms against 1.4 ms at 36698 strata: FP64 runs at 1/32 of the FP32 rate on this card), per-block partials summed on the device in a fixed order, no floating-point atomics.
- direct (v1): one thread per stratum reads its own run and keeps a private histogram (odd stride `cells | 1`, so the 32 lanes hit 32 banks).
- rows (v3): the threads of a block walk the rows of the block's 128 strata together (coalesced) and count into one shared table of (stratum x cell) with shared atomics; then one thread per stratum reduces its row.
- Dispatch by shape (`LONG_STRATUM_ROWS = 64`): rows for an average stratum of 64 rows and more, direct below. Guards: at most 64 cells per table, codes in a byte, at least 256 strata, else the dense route runs.
- Tables prepared on the device (CUDA.md section 10, item 9): the entropy table is cached per `n` (it was rebuilt by five launches per call), the run starts come from `bincount` + `cumsum` (the
  `flatnonzero` forced a host sync), the shared (x, z) entropy stays on the device. The whole strata call went from 2.8 to 1.0 ms at 5000 strata.

## Profile (nvprof on this host, elevated session, CUDA 11.8 CUPTI directory on `PATH`; 100000 rows, 25 permutations)
| metric | direct, 5000 strata 10 x 2 | rows, 5000 strata 10 x 2 |
|---|---|---|
| kernel time (gpu trace) | 236 us | 270 us |
| kernel time, 5000 strata 8 x 8 | 668 us | 1.06 to 1.27 ms |
| grid / block | (40, 25) / 128 | (40, 25) / 128 |
| registers / local bytes | 32 / 0 | 32 / 0 |
| shared memory per block | 1.5 KB static + 10.5 KB dynamic | the same |
| achieved occupancy | 0.451 | 0.308 (0.12 to 0.49 over the launches) |
| sm_efficiency / ipc | 89.2% / 1.29 | not collected |
| gld_efficiency / gld transactions per request | 6.7% / 9.3 | 64.4% / 16.5 |
| shared_efficiency | 38.5% | 36.7% |
| warp_execution_efficiency | 78.2% | 98.2% |
| stall: memory dependency / sync / execution dependency | 40.7% / 12.1% / 28.9% | 39.2% / 10.5% / not collected |
| local load transactions | 0 | not collected |

Reading: the direct kernel is uncoalesced on its loads (the lanes' runs are one stratum apart, about 20 bytes) and diverges on the run lengths; the rows kernel fixes both, but its shared atomics contend
(same cell in one warp, bank conflicts on a 64-cell table), so it is not faster in kernel time where strata are short and tables wide. Both are latency-bound (memory dependency 39 to 41%). Per section 3
the warp-aggregated atomic is not tried (the duplicate rate at 20 cells and 32 lanes is about 1.6 per cell, under the break-even of about 4 in the atrex record).

An earlier draft of this record said the counters need an admin shell; that was wrong: the first nvprof attempt had no CUPTI directory on `PATH` and printed nothing.

## Measurements (interleaved best-of-9 including the launcher, ms for the 25-permutation (x, y, z) count; before the table cache and the sync removal)
| strata | table | direct | rows | staged tile (dropped) |
|---|---|---|---|---|
| 300 | 10 x 2 | 0.96 | 0.81 | 0.80 |
| 1000 | 10 x 2 | 0.77 | 0.57 | 0.57 |
| 5000 | 10 x 2 | 0.62 | 0.65 | 0.86 |
| 36688 | 10 x 2 | 1.58 | 1.57 | 2.12 |
| 5000 | 8 x 8 | 1.05 | 1.56 | 1.44 |
| 63215 | 10 x 2 | 2.24 | 2.19 | not run |

Whole null (flat shuffle, strata count, floor and mean), `conditional_perm_null_gpu`, 100000 rows, 25 permutations, ms at 300 / 5000 / 40000 strata: 25.0 / 37.7 / 44.1 before this series, 15.1 / 16.4 / 16.9 now,
the same floor and mean to the printed digits (0.189763 and 0.187732 at 5000 strata). The kernels are 0.24 to 0.7 ms of that; the rest is the shuffle sort (5 ms), the layout and keys, and the fixed H(z), H(yz) terms.

## Verification
`tests/feature_selection/gpu/test_cmi_null_strata.py`: equal to the dense batched CMI to 1e-12 for four shapes x direct / rows / the shape rule x two datasets; the routes that do not apply return `None`.
Mutants: dropping the occupied-cell count and shifting the rows kernel's row start by one each fail the suite (8 of 25 for the shift).

## Not done
Vector (4-byte) loads in the rows kernel and a warp per stratum for 20 to 60 row strata: the kernels are under 5% of the null, so the gain on the wall is below 0.1 ms (section 6: a microbenchmark can lie).
