# fe_conditional_gate_enable: profile inside the all-flags fit (n=20000, default GPU mode)

Measured 2026-10-28 with `D:/Temp/cg_allflags.py` (all FE flags on, warm fit on other data, default device mode; GTX 1050 Ti).

The gate's scan (`cheap_conditional_gate_scan`) takes 4.56 s of cProfile time in this fit:

| part | cum s | calls | note |
|---|---:|---:|---|
| `_gate_device_born_all_mi` | 3.81 | 24 | device-born gate MI for every pending tau grid |
| `gate_grid_mi_resident` (inside it) | 3.80 | 24 | 0.16 s per call: the resident grid MI kernel on the device |
| `_scan_gate_column` (whole scan) | 4.24 | 1 | the rest is host work and bookkeeping |

Reading: the family is already GPU-resident (the GPU profile in `fe_family_walls_gpu_vs_cpu_n20000.md` shows 4.68 s on GPU, 7.74 s on CPU). Its remaining cost is the device grid-MI kernel, 0.16 s per flush, so the next lever is that kernel (batch size per flush, or fewer flushes), which is a GPU change with its own equivalence check. Not changed here.

Separate heavy part in the same fit, not in this family: `_gate_engineered_accuracy` (`measure_feature_uplift`, 4.21 s, a different gate). Recorded for the next profile round.

## Correction: Nsight Systems on the warm all-flags fit (2026-10-28)

The cProfile reading above (device grid-MI kernel as the remaining cost) was a hypothesis from cProfile wait time. It was checked with `nsys` (2025.1.3, `--capture-range=cudaProfilerApi`, `D:/Temp/gate_nsys.py`, report `D:/Temp/gate_fit.nsys-rep`, summaries `cuda_gpu_kern_sum` and `cuda_api_sum`). The window covers the whole warm fit, not the gate alone, so the shares below are whole-fit shares.

GPU kernel time (share of the kernel total):
- cub `DeviceMergeSortMergeKernel` + `DeviceMergeSortBlockSortKernel` (cupy sorts): about 50 percent together (29.7, 13.3, 3.7, 2.9).
- `root_split_stats_k2` (`_honest_null_device.py`, device honest MDLP null): 15.9 percent, 8 instances, 124 ms average.
- `cupy_copy__float64_float64`, `cupy_scatter_update`, `radix_select_interp_f64_v3`: about 12 percent together.
- `gate_grid_mi_resident` does not appear in the top ten kernels: the grid MI kernel is not the dominant GPU cost.

CUDA API time: `cudaStreamSynchronize` 3.81 s over 825 calls, `cudaMemcpyAsync` 2.47 s over 3188 calls, `cuLaunchKernel` 0.13 s over 10578 launches. The fit is sync- and transfer-bound and dominated by sorting, not by the gate's grid MI.

So the claim that the device grid-MI kernel is the remaining cost of the conditional gate is refuted. The gate-only attribution is not yet measured: it needs an NVTX range around `cheap_conditional_gate_scan` (not done here). Next step for this family: NVTX-label the gate, then re-run the same `nsys` recipe and read the kernels inside that range.

## NVTX-labelled Nsight Systems run (2026-10-28): the gate's own GPU time

`D:/Temp/gate_nsys2.py` wraps `hybrid_conditional_gate_fe_with_recipes` in an NVTX range `cond_gate` (one call per fit) and runs the same warm all-flags fit under `nsys` (`D:/Temp/gate_fit2.nsys-rep`). Read from the sqlite export: kernels that start and end inside the range.

- Range wall: 5.53 s. Kernel time inside it: 4.43 s (80%).
- cub `DeviceMergeSort` (the stable `cp.argsort` merge sort): 3.14 s, 696 launches, 71% of the gate's kernel time. The grid MI kernels `mi_from_codes` (0.11 s) and `mi_hist_split_f32` (0.04 s) are a few percent. The earlier claim that the grid MI kernel was the remaining cost is refuted for this range.
- Callers (cProfile, `D:/Temp/gate_callers.py`): 24 `cupy.argsort` calls from `_gpu_resident_rank_bin.rank_bin_codes_batch_gpu_resident`, one per gate flush, costing 3.25 s cumulative.
- Why: `_conditional_gate_fe._gate_device_born_all_mi` passes `rank_binning=_gate_rank_binning() or not strict`. Outside STRICT the rank binner is used on purpose: the percentile-edge binner lost a planted pair on the F2 two-pair frame at n=1000 (the comment at that line). So the gate's GPU cost is the stable argsort that makes it match the host.

Lever (not done): a stable rank that is exact but cheaper than the cub merge sort. A selection-equivalence check is needed before any change, because the rank must reproduce the host's tie order.

## Rank-binner sort: is there an exact cheaper stable rank? (2026-10-28, `D:/Temp/rank_bench.py`, `D:/Temp/rank_bench2.py`)

Microbenchmark, n=20000, 200 columns, values rounded on every fifth column so ties exist:
- GPU stable `cp.argsort(axis=0)` in float32: 39.6 ms. Its order equals the host stable argsort exactly (checked).
- Host `np.argsort(kind="stable")` plus upload: 407 ms, so the host is about 10 times slower.
- Float64 input: 43.7 ms, about 11 percent slower than float32. The dtype is not the lever.

Reading: the stable device sort is already near the throughput of the merge sort the device exposes, and no exact host or dtype alternative is cheaper. The rank binner's cost per gate flush is its column count times the sort cost, so the real lever is algorithmic: fewer candidate columns sorted per flush (a change to the candidate set), which needs a selection-equivalence study before any change. No code change here.

## Checked: how many columns the rank sort handles, and whether they recur (2026-10-28)

`D:/Temp/rank_shapes.py`, `D:/Temp/rank_unique.py`, `D:/Temp/gate_cols.py` (all-flags warm fit, default device mode):
- Each of the 24 gate flushes has 30 or 31 pending specs (743 in total), but the rank binner sorts a (20000, 527) matrix per flush, not 31 columns. Measured per call: 145 to 174 ms, 3.66 s over the 24 calls.
- Columns handled across all flushes: 12631; distinct by content hash: 11864. So 94 percent are unique and a cross-flush rank cache would save at most about 6 percent of the sort time (about 0.2 s of the 3.66 s).

Verdict on the lever stated above: fewer columns per flush is the only real lever, because almost every column is new in every flush. A rank cache is exact but worth about 6 percent. The earlier estimate, based on 31 specs, undercounted the sort work by about 17 times; this is the corrected figure.

## Fewer columns per flush: tau grid 17 points against 9 points (2026-10-28, `D:/Temp/tau_sel.py`, logs `D:/Temp/tau17.log`, `D:/Temp/tau9.log`)

Experiment, no code change: `_TAU_QUANTILES` patched in `_conditional_gate_fe_helpers` and `_conditional_gate_fe` (the 17-point grid 0.1..0.9 step 0.05 against a 9-point grid 0.1..0.9 step 0.1), all-flags warm fit, two data seeds (0 and 2).

- Selected features: identical in both seeds for the two grids (seed 0: six features; seed 2: eight features, the same set under both grids).
- Wall, seed 0 and seed 2: 70.6 s and 71.3 s with 17 points; 70.2 s and 67.8 s with 9 points. The difference is within the run-to-run noise of one fit, so the time saving is not measurable at this shape even though the rank sort is about half as large.

Reading: on these two frames the 9-point grid keeps the selection and does not measurably cut the wall. Before any change to the grid, the F2 and fusion sets must be run (the rule for this repository), and a wall gain must show on a full fit. Not changed.
