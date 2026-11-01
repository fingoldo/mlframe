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
| KTC dispatch | the block width comes from the KTC; the row threshold is a measured fallback with an env override, not yet a registered tuner (pending) |
| GPU profile (occupancy, SM throughput, DRAM) | not taken: no nsys / nvprof pass over this path yet (pending) |

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
