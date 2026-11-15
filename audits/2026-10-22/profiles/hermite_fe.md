# Profile record: `hermite_fe` (device kernels of the Hermite basis package)

Recorded 2026-11-12 on the GTX 1050 Ti (Pascal, sm_61, 6 SMs, 4 GB), CUDA 12.8, cupy with the NVRTC kernels of `_als_kernels_gpu.py` and `_nonpoly_cells_gpu.py`.
Harness: `n = 1000000` rows, `K = 20` columns for the non-polynomial cells (fourier degrees 1-5, 100 cells; rbf degrees 1-4), `D = 8` for the ALS design
(`weighted_gram` and `design_matvec`). Timings are min of 5 after a warm-up, device synchronized.

## Host side (cProfile)

cProfile of the steady-state calls (3 repetitions, warm-up excluded) is almost all Python dispatch: the cupy launches return before the device finishes, so the
host profile does not show the device time. The first profile of the whole process is dominated by the package imports and is not used.

## GPU side (nvprof)

Kernel launches per call: `nonpoly_cells` 1 per basis call (the degrees of a basis are one block of cells, `nonpoly_block_gpu`), `weighted_gram` 1 per call,
`design_matvec` 1 per call. The fused cell kernel computes every (column, degree) cell of a basis in one launch.

Time per call (min of 5, synchronized), n = 1000000:
- `nonpoly_block_gpu` fourier, 100 cells: 134.6 ms in the first measurement, 139.7 ms after the change below (run-to-run noise on the whole call); the cell kernel itself is about 81 ms per call on average in the nvprof summary (61% of the device time, the rest is the cupy min, max and quantile work around it).
- `nonpoly_block_gpu` rbf, 40 cells: 104 ms.
- `weighted_gram`: 9.5 ms.
- `design_matvec`: 0.9 ms.

Counters of one warm call (nvprof, metrics collected per kernel):
- `nonpoly_cells`: achieved occupancy 0.77, multiprocessor activity (sm_efficiency) 98.8%, DRAM read throughput 0.99 GB/s, FP64 instructions 2.61e9.
- `weighted_gram`: achieved occupancy 0.25, multiprocessor activity 99.0%, DRAM read throughput 8.1 GB/s, FP64 instructions 2.73e8.

Reading: both kernels are bound by double-precision arithmetic, not by memory. The cell kernel evaluates `sin`, `cos`, `exp` in double for each of 1e8 (row, cell)
pairs; on this card the double-precision rate is 1/32 of the single-precision rate. The DRAM traffic is about 1 GB/s, so the input read is not the limit.
`weighted_gram` is limited by its 36 double accumulators (occupancy 0.25).

## Changes measured here

- `nonpoly_cells`: the cell parameters are staged once per block in shared memory (up to 256 cells; above that the block reads global memory), and the cell index is 32-bit.
  Kernel-level A/B on the same inputs: 129.8 ms -> 122.5 ms (6%), output bit-identical (maxdiff 0). Shipped as a small, measured gain.
- The bare `double` in the kernel sources of `_als_kernels_gpu.py` and `_nonpoly_cells_gpu.py` is now the accumulator placeholder `{ACC}` (value casts `{REAL}`); the modules
  substitute `{ACC}` as double, so the generated code is the same.

## Next step (not done)

The cell kernel is double-precision bound. The two options that would cut its arithmetic: a float32 accumulation path for the fourier and rbf cells (the device
values are already float64 by the crit-dtype contract, so this changes the result precision and needs the equivalence tolerance re-decided), and a harmonic
recurrence for the fourier cells (one `sincos` per row and column, higher harmonics by the Chebyshev recurrence) instead of one `sin` or `cos` per cell.

## 2026-11-15: the modules of the package carved under this family

`hermite_fe` is a package; five of its modules are judged with it (the family has one verdict and this record, as `CARVED_FAMILIES` requires; none of them adds a CUDA kernel of its own):

- `_hermite_basis_eval`: the per-degree Horner njit evaluators (single and `prange`), the GEMV-style basis-matrix builders and the `build_basis_matrix` dispatcher: the numba CPU side of the family, and the baseline the fused device kernel (`_basis_fused_gpu`, 8 to 13 times over the cupy loop at 20000 to 1.6M rows, see its tuner entry) is checked against.
- `_hermite_oracle`: the `polyeval` njit-versus-njit_par backend oracle (gated off by default) and `_lookup_polyeval_thresholds`, the per-basis lookup of the kernel-tuning entry `polyeval` (its tests pin the per-basis region choice and the defaults for a basis without a region).
- `_hermite_prewarp`: the closed-form per-operand warps, the rank-1 ALS warm start and the saturating L2 penalty (host, numba).
- `_hermite_prewarp_gpu_resident`: the device twin of the ALS warm start and the per-degree cupy loop `_build_basis_matrix_loop` that `_basis_fused_gpu` replaced. By its own contract it is a residency twin, not a wall win: the ALS system is tiny and strictly sequential (one initial solve plus two dependent normal-equation solves per iteration), so the host solve stays the default and the device twin is off unless the resident flags are set.
- `_hermite_robust`: the outlier-robust axis normalisation and warp-coefficient fitting, each behind its own switch and heavy-tail gated; its device counterpart is the heavy-tail kernel of `_gpu_resident_basis` (row-share grid fill, 2.85 -> 0.54 ms, see `_gpu_resident_fe.md`).

No new counters were taken for these five: they hold host code, a tuner lookup and a residency twin whose kernels are the ones recorded above.
