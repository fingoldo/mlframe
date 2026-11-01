# `_orthogonal_routing_device.py` (conditional basis routing cells on the device)

Device twin of `_enumerate_routing_cells` + the MI / routing-score step of `generate_conditional_basis_routing_features`. It adds no CUDA source: the cells come from the batched Clenshaw kernels of
`_gpu_resident_basis` (`_gpu_evaluate_basis_matrix`), the MI from `_plugin_mi_classif_batch_cuda_resident`, the |corr| from `_gpu_batched_abs_corr`. The host path stays as the CPU twin and runs for an
unported basis, a device failure or a frame under `MLFRAME_ROUTING_DEVICE_MIN_ROWS` (5000).

## Contract points (CUDA.md section 2)

| point | status |
|---|---|
| CPU twin kept | yes: `_enumerate_routing_cells` + `_stack_candidates` + `_mi_classif_batch` |
| fused GPU kernel | reuses the fused Clenshaw / abs-corr / MI kernels (no new kernel source) |
| column-block streaming with a VRAM bound | source columns per block = `_gpu_k_chunk(n)` (the KTC-tuned VRAM fraction) / cells per source |
| device error -> host | the whole call falls back to the host path (the block result is not partially reused) |
| type-generic | values are held in `crit_float_dtype()` (float32 by default); tests run both criterion dtypes |
| KTC dispatch | block width from the KTC fraction; host-vs-device choice from the registered tuner `fe_basis_cells_device` (`_basis_cells_ktc.py`, 9 s sweep over 2000 to 100000 rows, fallback 5000 rows, `MLFRAME_BASIS_CELLS_DEVICE=0\|1` forces a path) |
| GPU profile | kernel-time table below (nvprof, warm path); occupancy / SM / DRAM counters need an elevated shell on this host (`ERR_NVGPUCTRPERM`) and are not taken |

## Measured (GTX 1050 Ti, warm, synchronized by the D2H of the scores, 10 source columns, 4 bases x 2 degrees x all pre-transforms)

| rows | device s | host s (enumerate + stack + MI + score) | ratio |
|---|---|---|---|
| 2000 | 0.026 | 0.321 | 12.5 |
| 5000 | 0.032 | 0.195 | 6.2 |
| 10000 | 0.036 | 0.296 | 8.4 |
| 20000 | 0.061 | 0.469 | 7.7 |
| 50000 | 0.141 | 1.008 | 7.1 |
| 100000 | 0.263 | 1.906 | 7.2 |

20 columns at 100000 rows: 0.50 s against 3.75 s. Agreement with the host: same 640 cells in the same order, MI within 7e-6, routing score within 1e-7, best cell per source equal for 19 of 20 sources (the
one other is a near-tie between two cells; this is the documented relaxed identity of the device basis, parity under 1e-6).

The first device call of a process costs about 1.4 s more (NVRTC / kernel compilation); `stage_sample` warms it with a small fit.

Remaining host work inside the path (cProfile, 20 columns): the pre-transforms on the host 0.13 s, the column_stack 0.10 s, the device-to-host score copies 0.21 s (waiting for the kernels).

## GPU kernel picture (nvprof `--profile-from-start off`, warm call, 20 columns x 100000 rows, 4 bases x 2 degrees x all pre-transforms; GTX 1050 Ti)

nsys 2022.4 recorded no kernels for this python (0 records in the trace), so the kernel table comes from `nvprof`; kernel total about 260 ms inside a call of about 500 ms wall.

| kernel | share | calls | avg ms | note |
|---|---|---|---|---|
| `clenshaw_basis` | 16.6% | 8 | 5.4 | the cells themselves, one launch per basis x block |
| `radix_select_f32_v3` | 12.4% | 10 | 3.2 | quantile edges of the MI binning |
| `cupy_mean` + `cupy_var_core` + `cupy_all` + `cupy_isfinite` | 10.7 + 8.0 + 5.3 + 1.8% | 16 / 8 / 8 / 8 | 1.7 / 2.6 / 1.7 / 0.6 | the usable-cell mask (finite and std > 1e-12) and the abs-corr centring, as separate generic cupy launches |
| `abs_corr` | 8.2% | 8 | 2.7 | the fused routing score |
| `cupy_take` | 7.2% | 16 | 1.2 | column selection of the usable cells |
| `mi_hist_split_f32` | 5.4% | 8 | 1.8 | the MI histogram |
| `cupy_copy`, `transpose_f32`, `basis_preprocess_nr` | 4.7 / 4.3 / 4.3% | 24 / 10 / 8 | 0.5 / 1.1 / 1.4 | layout and preprocess |

About 38% of the kernel time is generic cupy elementwise / reduction launches around the fused kernels (validity mask, column take, copies). A fused validity kernel (finite flag and variance in one pass,
one block per column) and a `take`-free MI call on the full block with the unusable columns masked would remove most of it; not done yet (measure before restructuring, CUDA.md section 6).

## After the fused validity kernel (`_cell_validity_gpu.py`)

The usable-cell mask (isfinite, all, std, take) and the per-column mean for `abs_corr` come from one kernel, `cell_moments` (32 columns x 8 row lanes per block, Welford per lane, fixed-order Chan merge, partials per
row chunk merged on the host in chunk order; reciprocal table in shared memory because a double division runs at 1/32 rate on Pascal): 3.08 ms -> 1.58 ms per launch on 100000 x 160. The block goes to the MI
and the score without the column take when every cell is usable.

Kernel time of the warm 20-column call, nvprof: 260 ms -> about 190 ms (-27%): the generic `cupy_mean`, `cupy_var_core`, `cupy_all` and `cupy_isfinite` launches (about 40% of the time) are gone and
`cell_moments` is 7%. Wall of the call is unchanged at 0.50 s: the remaining wall is host work (the pre-transforms and `column_stack`, 0.23 s) and the host-device score copies. What is left on the device:
`clenshaw_basis` 22%, `radix_select_f32_v3` 14%, `abs_corr` 11% (global-load efficiency 22%: one block per column reads a row-major block with stride m; a column-tile layout like `cell_moments` would
coalesce it), `mi_hist_split_f32` 6%, `basis_preprocess_nr` 6%, `cupy_copy` 6%, `transpose_f32` 5%, `cupy_take` 3% (only when some cell is unusable).

## abs_corr version 2 (`_abs_corr_tiled_gpu.py`)

v1 (one block per column, stride m through the row-major block) kept global-load efficiency at 22%. v2 uses a block of 32 columns x 8 row lanes, so a warp reads consecutive columns of one row; centred double
sums per lane, fixed-order merge, row-chunk partials summed on the host in chunk order. For `m >= 32` columns (below that v1 runs, both kept). Measured (nvprof, 20 columns x 4 bases x 2 degrees x pre-transforms,
100000 rows): 2.70 ms -> 1.05 ms per launch (2.6x); global-load efficiency 22% -> 82.5%, SM efficiency 94.9%, achieved occupancy 0.64. Agreement with numpy's Pearson: 2e-5 (float32), 1e-9 (float64).
