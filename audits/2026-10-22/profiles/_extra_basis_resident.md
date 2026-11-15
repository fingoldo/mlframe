# Profile record: device-born extra-basis candidates (`_extra_basis_resident`, host twin `_fourier_block_host` and the generators of `_orth_extra_basis_fe_generate`)

Host: GTX 1050 Ti (sm_61, 4 GiB, 112 GB/s), 8 logical cores, elevated process (hardware counters). Measured 2026-11-15, medians of 3 after a warm call. Workload: 4 float32 source columns, the spline and Fourier columns the generator emits for them (68 candidate columns: 36 cubic-spline legs, 32 sine / cosine columns), 100000 rows, 10 bins, binary target.

## What the module is
`score_features_by_mi_uplift` scores each extra-basis engineered column (spline, Fourier, chirp, wavelet) by `MI(col; y)` against its raw source. Under STRICT residency the host-built `engineered_X` matrix would be uploaded whole and scored by the resident plug-in MI; this module rebuilds the matrix on the device from the resident raw operands and the per-column fit metadata (frequencies, knots, lo / span) and scores it by the same estimator, so nothing crosses PCIe and the uplift ratio stays internally consistent. The Fourier and chirp columns of a pool are one launch (`_extra_basis_fourier_cuda`), the cubic-spline columns are now one launch as well (`bspline_block`), the wavelet legs use the shipped Haar leg kernel. The kernel-tuning entry `extra_basis_gate_backend` chooses between this build and the upload of the host matrix by row count; when the host is chosen the module answers `None` and the caller's scorer uploads and scores the host matrix with the same resident estimator.

## Column blocks (streaming)
`_mi_by_name_blocks` builds a column block of names, scores it and releases it: a block is a quarter of the room under this thread's pool limit over three copies of its columns, so the working set does not grow with the number of candidates. A column's MI depends on that column alone, so the blocks give the single-matrix values exactly (a test with a room of three columns compares equal).

## What was changed on this path (2026-11-15)
The cubic-spline columns were built one at a time by a cupy Cox-de Boor chain (about 40 launches per column; 54 spline columns of a 4-source pool took the device build to 130 ms at every size). `bspline_block` evaluates every spline column at every row in one launch over the (S, n) output with the rows fastest: z, the clip into the knot support, the four degree-0 indicators and the three triangle levels of the recursion in registers, with the host's zero-denominator guard. It is compiled with `--fmad=false`: with fused multiply-adds the values differed from the chain by one or two float32 ulp (28 of 36 columns), without them all 36 columns are equal value for value (a test also covers double, rows on the boundary knots and a NaN row). Device build and scoring of the 68-column pool at 100000 rows: 124 -> 30.7 ms; at 5000 rows 132 -> 8.0 ms.

## cProfile (one device scoring, 100000 rows, 68 candidates)
0.015 s for 6141 function calls: 0.007 s in the one blocking copy of the MI vector (the host waiting for the queued kernels), 0.001 s in the builder itself and 0.001 s in 20 small uploads. The Python side is a handful of launches, not a per-column loop.

## nvprof (hardware counters)
Launches per scoring: 1 `bspline_block`, 1 `fourier_block`, 1 concatenation of the two blocks, 1 `radix_select_f32_v5` for the edges of the block, 1 `transpose_f32`, 1 `mi_hist_split_f32` and 1 `mi_from_counts_col_f32`. GPU time 10.8 ms: `bspline_block` 44 percent (4.7 ms), `radix_select_f32_v5` 20 percent (2.16 ms), `cupy_concatenate_same_size` 12 percent (1.3 ms), `transpose_f32` 6 percent, `fourier_block` 6 percent (0.64 ms), `mi_hist_split_f32` 6 percent (0.63 ms).

| kernel | achieved occupancy | other counters |
|---|---|---|
| `bspline_block` | 0.51 | DRAM read 2.9 GB/s, global load efficiency 38 percent (the knot table is read by every thread of a warp at one address: a broadcast, counted as waste), double-precision FLOP efficiency 8.5 percent (the denominators are formed in double, as the host does), single-precision 3.9 percent |
| `fourier_block` | 0.89 | single-precision FLOP efficiency 13 percent, global load efficiency 44 percent (broadcast of the per-name parameters) |
| `radix_select_f32_v5` | 0.74 | DRAM read 21 GB/s, global load efficiency 99.9 percent |
| `mi_hist_split_f32` | 0.94 | DRAM read 47 GB/s, global load efficiency 66.6 percent |

Whole-GPU utilization: 10.8 ms of GPU time in a 30.7 ms wall (35 percent busy): the rest is the host side (the blocking copy, the stacking of the operands and the per-source uploads). The concatenation of the spline and Fourier blocks (1.3 ms) is a copy that writing both kernels into one output would remove; it is open.

## Wall clock (upload of the host matrix + resident MI against the device build + resident MI, same top candidates)
| rows | host | device |
|---|---|---|
| 5,000 | 3.6 ms | 8.0 ms (host wins) |
| 30,000 | 10.6 ms | 7.2 ms (1.5x) |
| 100,000 | 40.3 ms | 30.7 ms (1.3x) |

The device path is ahead from about 30000 rows on this card; below that the host matrix is already built and uploading it is cheaper than the launches. The estimator on both sides is the resident plug-in MI. (Against the host's own rank-binned MI the two differ on the spline legs at the support edges, whose columns are mostly exact zeros: the edge estimator keeps the ties in one bin where the rank coder splits them; that is the documented difference between the strict-resident and the default estimator, not a property of this build.)
