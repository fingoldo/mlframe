# Profile record: Fourier / chirp block (`_extra_basis_fourier_cuda`, numba CPU twin `_fourier_block_host`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores, elevated process (hardware counters available). Measured 2026-11-14 on an idle machine, medians after a warm call. Workload: 64 names (linear powers 1 and 2, chirp every fourth) over 4 source columns of 100,000 rows.

## What was added
* `_fourier_block_host.fourier_block_host`: a numba `prange` loop over the rows of every name, the same expressions as the device kernel in the element type of the operand (parameters rounded once to that type). It is the CPU reference of the block kernel.
* `_fourier_block_host.sincos_legs` / `sincos_axis_host`: the sine and cosine legs of one frequency in one parallel pass; the host emission of `_orth_extra_basis_fe_generate` (linear and chirp legs) calls `sincos_legs`. A double-precision axis of 20,000 rows or more takes the fused pass; a single-precision axis keeps numpy (its SIMD single-precision sine and cosine are 2.2x faster than the numba loop).
* `fourier_block_cuda` launches in column blocks of names (`_device_blocks`): each block stacks only the source columns it references and writes `out[lo:hi]`, halved on an out-of-memory error (test forcing one name per block gives the single-launch block bit for bit).
* Kernel-tuning entry `fourier_block_backend` decides from how many names the block replaces the per-column device build (`block_preferred`, used by `_extra_basis_resident` instead of the constant 4).

## cProfile / host numbers
Block of 64 names x 100,000 rows: numba twin vs the numpy per-column chain: float32 24.7 vs 36.9 ms (1.5x, max difference 4e-6), float64 34.9 vs 128.2 ms (3.7x, max difference 0.0); device kernel, operands resident, including the parameter upload: float32 2.8 ms, float64 10.2 ms (twin vs device: 6.5e-6 and 1.4e-14). The fused sincos pass for six frequencies over 100,000 rows: float64 4.45 vs 17.19 ms (3.9x, identical to the bit), float32 4.90 vs 2.19 ms (slower: numpy keeps that case).

## nvprof (hardware counters)
`fourier_block` (launches per call: 1 per name block): achieved occupancy 89.2 percent, single-precision FLOP efficiency 10.0, DRAM read throughput 21.2 GB/s, global load efficiency 43.6 percent (the cause is not diagnosed: the value loads of a warp are 4-byte contiguous, the per-name parameter reads are warp-wide broadcasts and the `src` indirection adds a dependent load). DRAM read is 21.2 GB/s, 19 percent of the card's peak, so the kernel is not memory-bound; whether it is bound by `sinf` / `cosf` / `powf` is a hypothesis not tested here. Whole-card throughput of the block is 64 x 100,000 outputs x 4 B = 25.6 MB written in 2.8 ms, 9.1 GB/s of the card's 112 GB/s bandwidth.

## Utilization and the tuned choice
Occupancy is high and DRAM is far from saturated, so a candidate lever is the transcendental cost (one `sinf` or `cosf` per output, plus `powf` for power 2 names that could use a multiply); not tried. The tuned block-versus-per-column choice on this host is in the kernel-tuning cache (`fourier_block_backend`); the earlier measured break-even was 3 to 4 names (0.25 ms per name per column against 0.75 ms fixed).
