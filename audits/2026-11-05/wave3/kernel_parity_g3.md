# GPU kernel parity audit, group 3 (wave 3, 2026-11-05)

This was a read-only pass. It compares each GPU kernel with its CPU twin (numpy or numba njit). All paths below are under `src/mlframe/feature_selection/filters/`.

One cause sits behind four of the findings (PARG3-1, -2, -4, -5). `_fe_gpu_batch/_devices.py:48` `crit_dtype_relaxed()` is **ON by default**, so `crit_float_dtype()` (`_devices.py:65`) hands back float32. Several device paths cast their raw operand to float32 *before* centring or ranking it. Their CPU twins do that work in float64. For a column with a large offset, float32 rounding wipes out the signal before any statistic is computed.

---

## PARG3-1: HIGH. Hermite plug-in MI ranks float32-rounded values on the device
- GPU: `_hermite_fe_mi.py:185` (`X_gpu = cp.asarray(X_cols, dtype=crit_float_dtype())`). It is reached through `hermite_fe/__init__.py:182` `_plugin_mi_classif_cuda` and the dispatcher.
- CPU: `hermite_fe/__init__.py:81` `_plugin_mi_classif_batch_njit`, which rank-bins the float64 column (`_numba_utils.py:436` `_quantile_bin_njit`).
- Statement: the device path casts the column to float32 before the quantile rank-binning. Values that are distinct in float64 merge into one tie group, which changes the partition and so the MI.
- Disagreeing input: `x = 1e8 + np.arange(20000) * 1e-4`, `y = (np.arange(20000) >= 10000).astype(int)`, `n_bins = 20`. The CPU gives MI = ln 2, about 0.693, because the bins separate the two halves cleanly. On the GPU the float32 spacing at 1e8 is 8, so every value rounds to 1e8. That makes one tie group, one bin, and MI = 0.
- Disposition: **RESOLVED-needed.** Centre (subtract the column median or min) in float64 before the float32 cast, or keep the operand dtype for the rank sort. The sort is the only place where the cast matters.

## PARG3-2: HIGH. Fused basis preprocess narrows to float32 before centring; its own fallback and the host do not
- GPU: `_gpu_resident_basis.py:632` (non-robust fused) and `:699` (robust fused) run `Mc = M.astype(crit_float_dtype())`, then (x - mean)/std or 2(x - lo)/span - 1 in kernels `basis_preprocess_nr` and `basis_preprocess`.
- Fallback twin: the cupy chain at `_gpu_resident_basis.py:723-750` works in M's own dtype. Host twin: `hermite_fe/_preprocess_sweep.py:30-44` (`_preprocess_zscore`) centres in float64 and casts only the result (`hermite_fe/__init__.py:608`).
- Statement: the default fused path rounds the raw operand to float32 before subtracting the mean. Its output dtype is float32, while the fallback chain returns M's dtype. So the answer and the dtype both depend on whether the RawKernel compiled.
- Disagreeing input: a Hermite column `x = 1e10 + rng.standard_normal(10000)`. In float32 (spacing 1024 at 1e10) x takes about one distinct value, so z is about 0 everywhere, or 0/guarded. The host z-score gives z ~ N(0,1). Every Hermite / Legendre / Chebyshev candidate built from that column is then degenerate on the GPU only. `_gpu_resident_basis.py:868-871` (abs-corr) already documents this exact hazard for 1e10 offsets and avoids it there.
- Disposition: **RESOLVED-needed.** Read M in its native dtype (the kernels already compute in double; template `{REAL}` on M's dtype, not the crit dtype). Narrow only the output z.

## PARG3-3: MEDIUM. Fused bin-codes kernels send NaN to bin 0; their fallback and the CPU binner send it to the rightmost bin
- GPU: `_gpu_resident_discretize.py:68` (`bin_codes_*`: `if (edges[mid] <= v) lo = mid + 1; else hi = mid;`) and `:138` (`bin_codes_f32e_*`, same predicate). For NaN, `edge <= NaN` is always false, so the code is 0.
- Fallback in the same function: `_gpu_resident_discretize.py:219` (`cp.searchsorted(..., side="right")`) gives NaN the code `ne`. CPU: `discretization/_kernels.py:163` `_searchsorted_2d_right_njit` puts NaN at `len(edges)` (its docstring lines 174-176). The sibling device kernel `discretization/_discretization_cuda.py:388` uses `cuts > v`, which is NaN-correct.
- Statement: the same function gives a NaN row code 0 when the fused kernel runs and code `ne` when it falls back. NaN rows merge with the lowest real bin instead of the post-max bin. `_gpu_resident_extval.py:11-13,35` states that this binner routes NaN rightmost, which is false.
- Disagreeing input: one column `[1, 2, ..., 10, NaN]`, `nbins = 4`. The fused kernel returns code 0 for the NaN row; the per-column fallback and the CPU return 3. The extval caller pre-filters non-finite columns (`_gpu_resident_extval.py:131`). The other callers (`_gpu_resident_materialise.py:639/642/808`, `_gpu_resident_pair_mi.py:105`, `_gpu_resident_pair_mi_parts.py:70`) do not appear to guard NaN; I did not trace their upstream scrub.
- Disposition: **RESOLVED-needed.** Write the predicate as `if (edges[mid] > v) hi = mid; else lo = mid + 1;` in both kernels so NaN goes to `ne`. Fix the extval docstring.

## PARG3-4: MEDIUM. Device biased HSIC runs the kernel distances in float32 by default; the host runs in double
- GPU: `_hsic_gpu.py:83` (`dt = crit_float_dtype()`); kernel `_hsic_gpu.py:20-48` takes `d = xa - sx[b]` in `{REAL}` = float and calls `expf`.
- CPU: `_orthogonal_hsic_fe.py:264-290` `_hsic_biased_columns_njit`: float64 differences and `math.exp`. The caller says "in double precision" (`_orthogonal_hsic_fe.py:523`).
- Statement: the raw column values are rounded to float32 before the pairwise difference. For a column on a large offset the differences are quantised, while the bandwidth (computed on the host in float64) is not.
- Disagreeing input: `x = 1e6 + 0.01 * rng.standard_normal(500)` with a label-dependent shift of 0.01. sigma is about 0.01 (computed in float64). On the device every difference is 0 or a multiple of 0.0625 (the float32 spacing at 1e6), so the Gram entries are about 1 or about e^-19. The HSIC score differs from the host by orders of magnitude. The backend sweep (`_hsic_gpu.py:180`, rtol 1e-5) uses standard-normal data, so it never sees this.
- Disposition: **RESOLVED-needed.** Subtract a per-column centre on the host (in float64) before the float32 cast, or make the kernel read double.

## PARG3-5: LOW-MEDIUM. The opt-in on-device friend-graph entropy computes in float32 but claims double / rtol 1e-9
- GPU: `_friend_graph_gpu_entropy_resident.py:74` (`counts_f = counts_dev.astype(crit_float_dtype())`), reached from `friend_graph_gpu.py:249/337/405` when `MLFRAME_FRIEND_GRAPH_GPU_ENTROPY_RESIDENT=1` (`friend_graph_gpu.py:130`; default OFF).
- CPU: `friend_graph_gpu.py:133` `_entropy_from_counts` computes `counts / n` in float64 and calls the njit `entropy()`.
- Statement: the counts, frequencies and logs are float32. Edge MI = H_a + H_b - H_ab then subtracts entropies that carry about 1e-7 relative error, and any count above 2^24 is no longer exact. The module docstring claims a double-precision reduction validated at rtol 1e-9.
- Disagreeing input: two independent 8-bin columns, n = 20000. The true plug-in MI is about 1.7e-3, and the float32 entropy error (about 5e-7 absolute) gives a relative MI error of about 3e-4, far above 1e-9. With n > 16,777,216 the single-bin counts round.
- Disposition: **RESOLVED-needed.** Hard-code float64 in this reduction (it reduces n_segments values; the cost is nil).

## PARG3-6: LOW. Unary-stack kernel `sign` maps NaN to 0; both twins give NaN
- GPU: `_unary_stack_cuda.py:35` (`case 8: r = (T)((v > 0) - (v < 0))`).
- Twins: `_usability_njit_pool.py:137` (`cp.sign`, NaN gives NaN) and `_numba_utils_usability.py:44-52` (`_apply_unary_low`, which explicitly returns NaN for NaN).
- Statement: on a NaN operand the fused stack writes 0.0 where the per-unary path and the njit reference write NaN.
- Disagreeing input: `d_x = cp.array([nan, 1.0])`, codes `[8]`. The kernel gives `[0, 1]`; the reference gives `[nan, 1]`.
- Disposition: **RESOLVED-needed** (a one-line `isnan(v) ? v : ...`). It is unreachable today because the only caller scrubs NaN first (as `_numba_utils_usability.py:45-49` notes), but the module docstring claims bit-identity.

## PARG3-7: LOW. The family maxT device path crashes on an empty family where the CPU returns zeros
- GPU: `_family_maxt_cuda.py:68-70` (`device_supported` returns True when `n_x` is empty, because 0 <= 1024) and `:91` (`out_d.max(axis=1)` on a (B, 0) array raises).
- CPU: `_family_maxt_gate.py:165-186` `family_null_max_njit` returns `zeros(B)` for k = 0.
- Disagreeing input: `cols.shape == (0, n)`, `n_x = []`, B = 10. The CPU gives `[0]*10`; the GPU raises ValueError (zero-size reduction).
- Disposition: **RESOLVED-needed.** `device_supported` should return False for an empty `n_x`, or `family_null_max_device` should return `np.zeros(B)`.

## PARG3-8: DOC. Offset-product device scan: reduction order differs, and the bitonic sort is undefined on NaN
- GPU: `_offset_product_gpu.py:42-53` (tree `block_sum`) and `:89-103` (bitonic sort with `(x > y) == up`). CPU: `_numba_utils_offset.py:207-270` (`ols2_shift`, sequential sums) and `:272-279` (`_edges_of`, `vals.sort()`, NaN last).
- Statement: the shifts differ by summation order (documented, about 1e-12). The edge rule, bin side (`edges <= x`), MI loop order and NaN shift handling otherwise match line for line. One more difference: if a regenerated feature is NaN (for example `(u+s)` overflows to inf and `(v+t) == 0`), numpy sorts it last while the bitonic network leaves the order undefined.
- Disagreeing input: a near-singular task where the shift sums sit on a decision threshold (`det > 1e-12 * tot[4] * tot[7]`). Reduction order can flip `wb` between finite and NaN, and so flip the whole task between scored and NaN.
- Disposition: **DOC.** It is already documented as "selection matches up to reduction order"; add the threshold-flip and NaN-sort caveats to the module docstring.

## PARG3-9: DOC. `col_minmax` RawKernel skips NaN; the cupy `min`/`max` it replaces propagates NaN
- GPU: `_gpu_resident_basis.py` `_COL_MINMAX_SRC` (`if (v < lmin) ...`, so NaN never wins), used at `:637/700`. Twin: `M.min(axis=0)` / `M.max(axis=0)` in the fallback chain (`:741`, `:751`).
- Statement: on a column containing NaN the fused path returns the finite min/max, while the chain returns NaN. Its docstring says "bit-identical".
- Disagreeing input: a column `[nan, 1, 2]`. The fused path gives lo = 1, hi = 2; the chain gives NaN / NaN.
- Disposition: **DOC.** `_gpu_evaluate_basis_matrix` documents its operand as finite. Qualify "bit-identical" with "on finite input".

---

## Checked and found at parity (no finding)
- `_gpu_kernel_sources.py`: the joint histograms (stride `cx*nbins_y+cy`, int atomics) and `compute_mi_from_classes_cuda` are double throughout, with the corrected stride.
- `_gpu_resident_extval_fused.py`: all 9 ops match `_extval_op_gpu`, including NaN max/min, sign(NaN)·|b| = NaN, and the div 1e-9 floor.
- `_gpu_device_permutations.py`: the same uint64 LCG recurrence as the host generator.
- `_fe_batched_mi_mm_cm.py`: the replaced substring `X[i * (long long)K + c]` occurs once in the source (`_fe_batched_mi.py:250`), so the column-major twin really is re-indexed.
- `_pair_mi_perm_cuda.py`: matches `_pair_mi_perm_scalar` (`info_theory/_batch_kernels.py:106`): `*inv_n` in both, the same py > 0 skip, and the same cap zeroing.
- `_fe_batched_qbin.py`: the mono-key integer compare equals the double compare for finite values; the interior/ndistinct==2 logic equals the broadcast fallback and the host `np.unique` + searchsorted rule.
- `discretization/_discretization_cuda.py`: uniform (float64 range, NaN code n_bins, inf to 0 or n_bins-1) matches `discretize_uniform`, since `arrayMinMax` returns float64. The quantile searchsorted kernel sends NaN rightmost, as the CPU does.
- `_dcor_gpu.py`: double throughout, the same moment formula.
- `_wavelet_basis_fe_batched.py`: the Haar leg masks are identical; the fixed-cardinality-3 codes give the same partition.
- `friend_graph_gpu.py` (default path): the integer counts are reduced on the host njit.
- `_gpu_hw_launch.py`, `_prewarm.py`, `_fe_gpu_batch/_devices.py`: launch geometry, warm-up and policy only; no arithmetic twin. `_devices.py` is the source of the default-ON float32 relaxation noted at the top.

## Files read
- `_fe_gpu_batch/_devices.py`
- `_gpu_kernel_sources.py`
- `friend_graph_gpu.py` (lines 112-148, 340-660)
- `_prewarm.py` (grep plus lines 500-560)
- `_unary_stack_cuda.py`
- `_pair_mi_perm_cuda.py`
- `_gpu_resident_extval_fused.py`
- `_gpu_device_permutations.py`
- `_fe_batched_mi_mm_cm.py`
- `_family_maxt_cuda.py`
- `_wavelet_basis_fe.py` (lines 168-301)
- `_wavelet_basis_fe_batched.py` (lines 60-160)
- `_offset_product_gpu.py`
- `_gpu_resident_discretize.py`
- `_gpu_resident_basis.py` (lines 400-760)
- `_fe_batched_qbin.py`
- `hermite_fe/__init__.py` (lines 81-206, 425-512)
- `discretization/__init__.py` (lines 343-440, 671-874)
- `discretization/_discretization_cuda.py`
- `_hsic_gpu.py`
- `_gpu_hw_launch.py` (lines 140-238)
- `_dcor_gpu.py`

CPU twins and helpers read to support these findings:
- `_usability_njit_pool.py` (lines 121-171)
- `_numba_utils_usability.py` (lines 14-73)
- `_family_maxt_gate.py` (lines 134-186)
- `_gpu_resident_extval.py` (lines 1-72, 120-145)
- `_gpu_nan_quantile_codes.py` (lines 60-100)
- `_orthogonal_hsic_fe.py` (lines 213-290, 505-545)
- `_orthogonal_dcor_fe.py` (lines 113-140, 274-287)
- `_numba_utils_offset.py` (lines 153-371)
- `_friend_graph_gpu_entropy_resident.py`
- `_hermite_fe_mi.py` (lines 173-240)
- `_numba_utils.py` (lines 386-458)
- `hermite_fe/_preprocess_sweep.py` (lines 30-96)
- `info_theory/_batch_kernels.py` (lines 106-139)
- `discretization/_kernels.py` (lines 163-176)
- `core/arrays.py` (lines 27-47)
- `_mi_greedy_cmi_fe_binning.py` (lines 106-147)
