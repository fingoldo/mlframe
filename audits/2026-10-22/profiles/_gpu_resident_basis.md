# Family record: basis-evaluation kernels of the resident FE path (`_gpu_resident_basis`, compile points in `_gpu_resident_basis_factories`)

The module is a carved part of the `_gpu_resident_fe` family (its profile, with cProfile, nvprof counters, launches per call, occupancy and utilization of the whole resident generation + scoring loop, is `_gpu_resident_fe.md`). This record covers what changed for the kernels of the basis file on 2026-11-15.

## What was carved
`_gpu_resident_basis.py` was at the module-size limit. The compile points of its nine kernels (`clenshaw_basis`, `robust_scale_combine`, `heavy_tail_stats_tiled`, `heavy_tail_finish`, `basis_preprocess`, `basis_preprocess_nr`, `col_meanstd_tiled`, `col_minmax_tiled`, `abs_corr`) and the `abs_corr` source with its getter now live in `_gpu_resident_basis_factories.py`; the lazy getters of the basis file call the factories, so the register-pressure registry compiles exactly the code a fit compiles. The basis module re-exports the carved `abs_corr` names (`test_gpu_abs_corr_kernel_large_offset`). The factories module imports the basis module lazily (the registry may import it first).

## Register pressure (cubin counters on the GTX 1050 Ti, sm_61, 2026-11-15)
| kernel | float32 | float64 |
|---|---|---|
| `clenshaw_basis` | 48 regs, 0 B local | 48 regs, 0 B local |
| `robust_scale_combine` | 22 / 0 | 22 / 0 |
| `heavy_tail_stats_tiled` | 38 / 0 | 40 / 0 |
| `heavy_tail_finish` | 40 / 0 | 40 / 0 |
| `basis_preprocess` | 29 / 0 | 29 / 0 |
| `basis_preprocess_nr` | 22 / 0 | 22 / 0 |
| `col_meanstd_tiled` | 32 / 0 | 40 / 0 |
| `col_minmax_tiled` | 32 / 0 | 32 / 0 |
| `abs_corr` | 27 / 0 | 26 / 0 |

No kernel spills in either element type; the registry (`test_cuda_kernel_register_pressure.py`) pins that.

## Not measured here
Wall time, launches per call and achieved occupancy of these kernels inside a fit are in `_gpu_resident_fe.md` and the heavy-tail rows of `_gpu_resident_fe` / `hermite_fe`; this record adds no timing.
