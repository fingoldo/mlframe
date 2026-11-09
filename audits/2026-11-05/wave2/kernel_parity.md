# Wave 2: CPU / GPU kernel parity audit (2026-11-05)

Scope covered: `discretization/_discretization_cuda.py` (quantile and uniform GPU paths, fused searchsorted RawKernel) against the CPU
`discretize_2d_array` njit path; `_row_argmax_gpu.py` against `np.argmax`. 79 files hold CUDA kernels; most were NOT reviewed in this
pass (see "Not covered").

## PAR-W2-1 (High): fused searchsorted RawKernel assumes C-order input, Fortran-order arrays are binned from the wrong cells

- GPU twin: `src/mlframe/feature_selection/filters/discretization/_discretization_cuda.py:106` (`d_arr = cp.asarray(arr)`), `:135-140`
  (n_cols >= 1000 branch), kernel `:370-392` (indexes `arr[gid]` with `row = gid / n_cols`, i.e. C-order), launcher `:463`
  (`d_arr.astype(cp.float64, copy=False)`, which keeps the layout).
- CPU twin: `src/mlframe/feature_selection/filters/discretization/__init__.py:645-646` (`np.ascontiguousarray(arr2d)` before the njit kernel),
  and the per-column `cp.searchsorted` loop at `_discretization_cuda.py:142-143`, which is layout-independent.
- Statement: when `arr` is F-contiguous float64 (or float32, whose `astype` keeps order 'K'), `cp.asarray` keeps F order, and the kernel
  reads element `gid` as `(gid // n_cols, gid % n_cols)` while it actually holds `(gid % n_rows, gid // n_rows)`; the bin codes are those of
  other cells compared against the wrong column's cuts. Nothing on the dispatch path (`try_discretize_2d_cuda`,
  `_discretize_2d_cuda_attempt.py:17`) forces C order.
- Disagreeing input: `arr = np.asfortranarray(np.random.rand(2000, 1000))` (e.g. `df.to_numpy()` of a homogeneous float pandas frame,
  which is commonly F-ordered), `discretize_2d_array_cuda(arr, n_bins=10)` vs `discretize_2d_array(arr, n_bins=10, prefer_gpu=False)`:
  CPU codes are per-column deciles, GPU codes do not match. With n_cols = 999 the per-column loop is used and the two agree, so the bug
  only appears at the 1000-column crossover. The row-chunked path (`:347`) slices rows of an F array, whose layout after `cp.asarray` is not
  guaranteed C either; check it with the same input.
- Not executed (no GPU in this audit); the conclusion follows from cupy's documented `asarray` order preservation and `astype(copy=False)`
  returning the same array. Confirm with the input above on a GPU host.
- Disposition: RESOLVED-needed. Pass `cp.ascontiguousarray(d_arr, dtype=cp.float64)` (and the same for each `d_chunk`) to the kernel, or pass
  strides; add a GPU-marked parity test with an F-ordered 1000-column input.

## Checked and found consistent (no finding)

- NaN rows in the quantile branch: `cp.searchsorted(side="right")` and the RawKernel both return `n_cuts` for NaN (comparison `cuts > NaN`
  is always false), matching the CPU routing noted at `_discretization_cuda.py:104`.
- Uniform branch: NaN values go to `n_bins`, NaN-poisoned min/max use `nanmin/nanmax`, range in float64; matches the CPU kernel per the
  existing fixes.
- `_row_argmax_gpu.py:23-33`: strict `>` gives first-max ties like `np.argmax`; NaN rows are excluded by the caller's finiteness gate
  (`all_finite_cached`), as documented.

## Not covered

The remaining ~75 kernel files (MI/CMI kernels, `_gpu_metrics.py`, DTW, transformer kernels, honest-null kernels, resident-select
kernels) were not reviewed in this pass and need a follow-up round.
