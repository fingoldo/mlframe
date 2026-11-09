# Kernel parity, wave 3 (files not covered in wave 2)

Read-only audit. No code changed. Base: the working tree at C:\Users\Admin\Machine learning\mlframe, 2026-11-06.

## Findings

### PAR3-1 (MEDIUM): the default usability-|corr| GPU kernel computes in float32, the CPU twin in float64
- GPU: `src/mlframe/feature_selection/filters/batch_pair_usability_corr_gpu.py:484` (`ft = float32` unless `MLFRAME_USABILITY_CORR_F32=0`), with casts at :354-356 and :384-386.
- CPU: same file :178-231 (`_abs_pearson_form_reduction`, every value cast to float64) called from `batch_pair_usability_corr_njit_parallel` :239-277.
- Statement: the un-forced default (`cuda_warp`) rounds each operand to float32 before forming x0, x0**2, x0/x1 and the moments. The CPU path keeps float64. The documented gap of 1.7e-6 was measured on one well-scaled fixture and does not bound the gap in general.
- Inputs that make them disagree:
  - (a) Large offset with small spread: `x0 = 1e8 + 0.1*arange(n)`, `y = x0`. In float64 |corr| = 1. In float32 the values snap to a grid of 8, so x0 becomes a step function and |corr| drops well below 1. If the spread stays under one float32 ULP the column becomes constant and the CV floor returns 0.
  - (b) Overflow: `|x0| > ~1.8e19`. Then `x0*x0` is inf in float32 and the row is dropped from the x0**2 form, while float64 keeps it. With |x0| > 3.4e38 the raw value itself becomes inf.
  - (c) Near-ties between forms. `batch_pair_tail_concentration_rankaware` (:775) takes `argmax` over the 5 pair forms. Forms that sit within 1e-6 of each other can give a different winning form, which changes the rank leg (:791-799) and so the verdict.
- Disposition: RESOLVED-needed. Either gate float32 on a predicate the host can check cheaply (for example `max|x| * 2**-24` small relative to the column std, and `max|x| < 1e18`) and fall back to the float64 kernel otherwise, or default to float64. Add a parity test on input (a).

### PAR3-2 (LOW): the three usability backends return different output dtypes
- CPU: `batch_pair_usability_corr_gpu.py:263` returns float64.
- `cuda` backend: :538 returns `crit_float_dtype()`.
- `cuda_warp` backend: :508 returns `crit_float_dtype()`.
- Statement: the output dtype depends on which backend ran. Under a float32 crit dtype, the CPU fallback (taken on VRAM shortage or a CUDA exception, :636/:647) returns float64 while the GPU path returns float32. Because `_CORR_MEMO` (:697) keys on `force_backend` and not on the backend that actually ran, the dtype a repeat call returns depends on which backend ran the first time.
- Input: a float32 crit dtype plus a VRAM-cushion miss on one call. The same request then yields float64 instead of float32.
- Disposition: RESOLVED-needed (cheap fix): cast the CPU result to `crit_float_dtype()`, or cast every backend to float64, so that all three return one dtype.

### PAR3-3 (LOW): the CPU twin runs with reassoc/contract fastmath, the GPU warp kernel does not
- CPU: `batch_pair_usability_corr_gpu.py:178` (`fastmath={"reassoc","contract","arcp","afn","nsz"}`).
- GPU: :345, :375, :430 use plain `cuda.jit` with a lane-strided summation order.
- Statement: the summation order differs, and the CPU side may fuse FMAs. The `<=` comparisons against the CV floor (:223 vs :460) can therefore flip for a column whose variance sits exactly at the floor. This is the ULP-level difference already documented, and it only matters at that boundary.
- Input: a column whose `saa` lands within one ULP of `cnt*cv2*ma*ma`. This is rare.
- Disposition: DOC. The module docstring already accepts summation-order differences. Add one line saying the CV-floor boundary can flip.

### PAR3-4 (INFO): RMSE with NaN, and GPU vs CPU AUC — checked, at parity
- `src/mlframe/metrics/_gpu_metrics.py:246-346` (numba atomic kernel and cupy ReductionKernel) vs :541-543 (numpy). NaN propagates to NaN on both sides. Both accumulate in float64 and return the caller's dtype (:525). AUC NaN scores are masked identically on both backends (:546-554). The tie-averaged ranks at :386-409 match sklearn by construction.
- Disposition: REJECTED (no divergence found). The run-to-run nondeterminism of the atomic RMSE is already documented at :277-282.

## Files read
- src/mlframe/metrics/_gpu_metrics.py (lines 210-629)
- src/mlframe/feature_selection/filters/batch_pair_usability_corr_gpu.py (whole file)

## Files NOT read (contain GPU kernels per the grep and still need a pass)
signal/dtw.py; filters/info_theory/_cmi_cuda.py; discretization/__init__.py and _discretization_cuda.py;
feature_engineering/transformer/_kernels_cupy.py, _kernels_njit.py, _oof.py, _aggregation.py, row_attention.py;
filters/friend_graph_gpu.py, _binned_numeric_agg_resident.py, hermite_fe/__init__.py, gpu.py,
_batch_pair_mi_cuda_shared_fused.py, _batch_pair_mi_cuda_kernels.py, _batch_mi_noise_gate_kernels.py, _cell_moments_kernel.py,
_family_maxt_cuda.py, _family_maxt_cuda_bits.py, _dcor_gpu.py, _wavelet_basis_fe.py, _wavelet_basis_fe_batched.py,
_cupy_polynom_optimizer.py, _fe_batched_qbin.py, _fe_batched_mi_mm_cm.py, _fe_batched_mi_cmi.py, _fe_sufficient_summary_pairs.py,
_gpu_resident_select.py, _gpu_resident_select_kernels.py, _gpu_resident_select_f32_compact.py, _fe_gpu_batch/_devices.py,
_gpu_resident_pair_mi.py, _gpu_resident_operand_build.py, _gpu_resident_materialise.py, _gpu_hw_launch.py,
_gpu_device_permutations.py, _gpu_resident_fe.py, _gpu_resident_extval_fused.py, _gpu_resident_discretize.py,
_gpu_resident_basis.py, _unary_stack_cuda.py, _gpu_kernel_sources.py, _gpu_pairs.py, _pairs_analytic_gpu.py, _hsic_gpu.py,
_hermite_fe_mi.py, _honest_null_k3.py, _honest_null_k4.py, _honest_null_device.py, _internals.py, _resident_candidate_mi.py,
_mi_greedy_cmi_fe_gpu.py, _qbin_block_gpu.py, _prewarm.py, _offset_product_gpu.py, _pair_mi_perm_cuda.py,
_pairwise_modular_rank_gpu.py, _orthogonal_univariate_fe/_gpu_resident_cross_basis.py,
shap_proxied_fs/_shap_proxy_gpu.py, _shap_proxy_treeshap_gpu.py, models/ensembling/per_member_tuning.py,
training/composite/transforms/_nonlinear_ewma_fracdiff.py.
Skipped as already covered: _gpu_resident_shared.py, _fe_batched_mi.py, _null_max_perm_gpu.py, _row_argmax_gpu.py,
discretization/_discretize_2d_cuda_attempt.py. Skipped as non-production: everything under _benchmarks/.
