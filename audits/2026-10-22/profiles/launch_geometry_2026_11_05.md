# Launch geometry from the device, 2026-11-05

The meta test `tests/test_meta/test_cuda_launch_geometry_gate.py` (shared gate `cuda_launch_geometry` of py-ci-shared) fails a kernel launch whose block size is a literal. It started with a baseline of 119 launches in 62 places (92 times 256, 15 times 128) and now has no baseline: the count is 0, and a new literal fails the test.

## How the 119 launches were resolved

| Resolution | Launches | How |
|---|---|---|
| Block taken from the device | 102 | `pyutilz.performance.gpu_hw_launch.block_size_for(kernel, dyn_smem=..., work_items=..., pow2=..., max_width=...)`: the warp multiple that keeps the most threads resident, from the kernel's own `num_regs` / `shared_size_bytes` and the card's limits. The grid is recomputed from the derived block. |
| Marked `# launch-ok: <reason>` | 17 | The kernel fixes the geometry: 8 warps for the `[8][NACC]` warp-partial arrays of the ALS kernels (6 launches), the 32x32 transpose tile (3), the (32, 8) column-by-row tiles (3), one warp for the 3x3 solve kernels (2), a single working thread (1), the `THREADS` compile-time macro of the offset-product scan (1), and one launch the gate took for a launch but is an `itertools.product` grid (1, fixed in the gate as well). |

## Kernel classes and the constraint each one passes to `block_size_for`

* Flat one-thread-per-item and grid-stride kernels (element-wise builders, binning, materialise, quantile codes, modular codes, candidate builders): no constraint, `work_items=` the item count.
* Block reductions that halve `blockDim.x` over a fixed shared scratch (`sh_h[256]`, `red[128]`, the MI / entropy / mean-std / min-max / abs-corr family): `pow2=True, max_width=<scratch width>`.
* Reductions with a dynamic scratch of one value per thread (HSIC, dCor, `cmi_from_joint`): the scratch is sized from the derived width after it is chosen; the width is chosen against the largest scratch, so the 48 KB block limit holds at the largest supported `m`.
* Warp-partial reductions with 32 rows (`[6][32]`, `[9][32]`, `warp_sums[32]`): any warp multiple up to 1024.
* Kernels compiled with `__launch_bounds__(128, 16)` (pairwise modular rank): `pow2=True, max_width=128`.

## What was not measured

The derived widths were checked against the CPU twins by the module tests; wall-time A/B per kernel was not run for this change. `_cell_moments_kernel` keeps its grid of 4 blocks per SM (the rejected-attempt note in that file says more blocks only add atomic contention); only its block width now comes from the device.
