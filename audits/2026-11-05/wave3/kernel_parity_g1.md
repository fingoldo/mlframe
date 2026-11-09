# GPU kernel parity audit, group 1 (2026-11-05, wave 3)

This audit was read-only and nothing was executed against a GPU. Every disagreement input below was derived by reading the code. None was reproduced on a device.
Out of scope, already covered: `_gpu_resident_shared.py` max/min NaN, `_fe_batched_mi.py` / `_null_max_perm_gpu.py` NaN bin, `discretization/_discretize_2d_cuda_attempt.py`, `batch_pair_usability_corr_gpu.py`.

Paths are relative to `src/mlframe/feature_selection/filters/`.

## Findings (most severe first)

### PARG1-1: HIGH. The device-born binned-agg works in float64; the host twin works in the crit dtype (float32 by default)
- GPU: `_binned_numeric_agg_resident.py:351-357` (the group values are uploaded as float64 and `cp.searchsorted(edges_g, gvals_g, side="right")` runs in float64), `:369-375` (agg values in float64, `cp.isfinite` in float64).
- CPU: `_binned_numeric_agg_fe.py:441,448` (`gvals` cast to `_crit_np_dtype()`, which is `np.float32` while `MLFRAME_CRIT_DTYPE_RELAXED` keeps its default of on), `:459-460` (`av` cast to float32, then `np.isfinite`).
- Statement: the host bins the float32-rounded group value and takes its moments over float32 agg values. The device bins the raw float64 value and takes its moments over float64 values. Bin codes, the finite mask and the cell moments can all differ, although the device code comments say "bin assignment must match the host".
- Input: group value `0.99999999` (float64) with a stored edge `1.0`. On the host, float32 rounding gives `1.0`, so the code is 1. On the device, `0.99999999 < 1.0`, so the code is 0. For the agg column, the value `1e39` is finite in float64 but `inf` in float32. The host drops that row from every cell moment; the device keeps it.
- Disposition: RESOLVED-needed. Upload `gvals` and `av` in `_crit_np_dtype()` (round first, then widen), or have the host fit in float64. Pin it with a host-vs-device OOF-column equality test under the relaxed default.

### PARG1-2: HIGH. The op-candidate ratio uses a different formula on the device than on the host
- GPU: `_resident_candidate_mi.py:59` (row-major) and `:96` (column-major): `u / (fabs(v) + 1e-6)`.
- CPU: `_conditional_gate_fe.py:240`: `u / np.maximum(np.abs(v), 1e-6)`. The device's own cupy fallback at `_resident_candidate_mi.py:216` uses the host form, so the kernel disagrees with its own fallback as well.
- Statement: the shift is not monotone across rows, so the order of ratio values changes, and with it the quantile codes and the gate's baseline MI.
- Input: row A `(u=1, v=0)` and row B `(u=1.5, v=1e-6)`. The host gives A=1e6 and B=1.5e6 (B > A). The kernel gives A=1e6 and B=7.5e5 (B < A). With `v=1` the host gives exactly `u` and the kernel gives `u*0.999999`.
- Disposition: RESOLVED-needed. Change the kernel to `u / fmax(fabs(v), 1e-6)` in both sources, and add a parity test that compares the kernel against `best_existing_op_mi`'s numpy block.

### PARG1-3: MEDIUM. Row max and row min in `build_op_cands` drop NaN unless it sits in the first operand; numpy propagates NaN
- GPU: `_resident_candidate_mi.py:62-65` and `:99-102`: `acc = (w > acc ? w : acc)`. A NaN `w` loses every comparison, so it is skipped. A NaN `acc` (operand 0) wins every comparison, so it stays.
- CPU: `_conditional_gate_fe.py:243-244`: `stk.max(axis=1)` / `stk.min(axis=1)`, which return NaN whenever any operand is NaN. The cupy fallback at `_resident_candidate_mi.py:218` does the same.
- Statement: the result depends on operand order, and it differs from the host whenever a NaN operand is not the first one.
- Input: operands `[1.0, NaN, 3.0]`. The host max is NaN; the kernel max is 3.0. Operands `[NaN, 1.0, 3.0]` give NaN on both.
- Disposition: RESOLVED-needed. Propagate NaN explicitly (`if (w != w || acc != acc) acc = w + acc`), the way `fe_materialise` does.

### PARG1-4: MEDIUM. The radix select ranks a sign-bit-set NaN below -inf, while sort and percentile put every NaN last
- GPU: `_gpu_resident_select_kernels.py:48,92,158,252,564,631,717` and `_gpu_resident_select_f32_compact.py:53` use the key transform `u = (u & sign) ? ~u : (u | sign)`. A NaN with the sign bit set (`0xFFF8...`, which is what x86 produces for `inf - inf` and `0 * inf`) therefore maps below `-inf`.
- CPU/sort twin: `device_sort` / `cp.percentile` / `np.quantile` sort every NaN to the end whatever its sign. `_qbin_block_gpu.py:38` (the sort fallback) and `_mi_greedy_cmi_fe_binning.py:84` take the per-column sort path.
- Statement: in a column that holds a negative NaN, the radix order statistics are shifted by one rank compared with the sort path, and the minimum edge becomes NaN. `_radix_quantiles` documents a finite precondition; `_radix_select_interior_edges` and `_order_statistics_rows` neither document nor check one.
- Input: `x = [np.inf - np.inf, 0.0, 1.0, 2.0]` with `nbins = 2`. The radix rank-0 statistic is NaN; the sorted rank 0 is 0.0.
- Disposition: RESOLVED-needed. Canonicalise the NaN key in the transform (`if NaN, u = 0xFFFF...`) or add a finite guard in front of the radix entry points.

### PARG1-5: MEDIUM. The block qbin count kernel gives a NaN value code 0; the per-column twin gives it the last code
- GPU: `_qbin_block_gpu.py:114`: `kept <= (double)x` is false for every edge when x is NaN, so the code is 0.
- Twin: `_mi_greedy_cmi_fe_binning.py:103`: `cp.searchsorted(e2, x, side="right")` follows numpy and places NaN after every edge, so the code is `len(e2) = nbins + 1`.
- Statement: the module docstring claims the codes are "bit-identical to the per-column binner for every column shape", but a NaN row goes to opposite ends. `resident_qbin_codes_block` does not check for finite input.
- Input: column `[0.0, 1.0, NaN, 2.0, ...]` with `k >= 2` columns, which routes it through `_count_kernel`: code 0. The same column alone (`len(part) == 1`): code `nbins + 1`.
- Disposition: RESOLVED-needed. Either map NaN to `ne` in the kernel (`if (v != v) c = ne`) or add a finite precondition and guard at `resident_qbin_codes_block`.

### PARG1-6: MEDIUM. The fallback path of the batched CMI reduces entropies in the crit dtype (float32); the fused kernel and the CPU twin use double
- GPU fallback: `info_theory/_cmi_cuda.py:470-475`, which calls `_entropy_from_counts_axis` (`:314-316`) with `_get_xlogx_kernel(crit_float_dtype())`. The per-cell term and the `.sum(axis=1)` both run in float32.
- Primary GPU: `:247-273`, double accumulation. CPU: `_cpu_cmi_loop*` / `conditional_mi`, float64.
- Statement: when the fused kernel raises (shared memory or compile failure), the CMI becomes a difference of four float32 entropies of size about `log(n)`. The cancellation error, about 1e-6 nats, is far above the documented 1e-9 parity gate and can flip the `max(.,0)` clamp and the ranking of near-zero candidates.
- Input: n = 1e6, nbins = 16/16/16, a candidate with true CMI of about 2e-7. The double path gives about 2e-7. The fallback gives anything in roughly [0, 3e-6] depending on rounding.
- Disposition: RESOLVED-needed. Build the fallback xlogx kernel and its sum in float64, and cast only the final CMI to the crit dtype.

### PARG1-7: LOW. `group_mi_block_gpu` uses the plain lerp, while `np.quantile` and the radix interp use the symmetric lerp
- GPU: `_qbin_block_gpu.py:203`: `xlo + (xhi - xlo) * frac`.
- CPU: `_binned_agg_cheap_mi.py:43` (`np.quantile`, numpy's `_lerp` with the `t >= 0.5` branch). The device's own `sync_free_qbin_codes_block:43` and the radix interp (`_gpu_resident_select_kernels.py:580,650,750,907`) use the symmetric form.
- Statement: for `frac >= 0.5` the two forms can differ by 1 ULP. The `_sync_free_qbin_codes` comment records that exactly this moves a tied value's whole mass into the next bin when the edge lands on a data value. Here edges only touch data when `xlo == xhi`, which both forms handle exactly, so a flip needs an edge that rounds onto a neighbouring order statistic.
- Input: consecutive order statistics `a = 1.0` and `b = nextafter(1.0, 2)` with `frac = 0.75`. The plain form can round to `b` while the symmetric form stays below it, so a row equal to `b` changes bin.
- Disposition: RESOLVED-needed. Make it a one-line change to the symmetric form, matching the module's other edge builder.

### PARG1-8: LOW. The device CMI uses the raw codes as histogram indices; the host densifies arbitrary integers
- GPU: `_mi_greedy_cmi_fe_gpu.py:187-207,227-246`. `Kx = dx.max() + 1` and `joint_entropy_gpu` / `cmi_joint_entropies_gpu` index `counts[(x*Ky+y)*Kz+z]` directly. There is no lower-bound check.
- CPU: `_mi_greedy_cmi_fe.py:233-257`: `_entropy_from_classes` / `_renumber_joint` accept any int64 label, negatives included.
- Statement: a `-1` sentinel code (a NaN bin) or a caller `kx` below the true maximum gives a correct CMI on the host. On the device it writes to the wrong cell or out of bounds. Whether `-1` can reach this site was not traced. This is the same class as the covered `_fe_batched_mi` NaN-bin finding, at a different call site.
- Input: `x = [-1, 0, 1, 1]`, `y = [0, 1, 0, 1]`, `z = None`. On the host the MI is finite and correct. On the device, `-1 * Ky + y` indexes before row 0.
- Disposition: RESOLVED-needed. Call `_assert_codes_in_range` (already used by `_cmi_cuda.py:423-427`) before the device joints, or route negative codes to the host.

### PARG1-9: LOW. The single-row-sum and ratio candidates are computed in double and stored as float32; the host computes them in float32
- GPU: `_resident_candidate_mi.py:53-72` and `:90-109` (`double val`, then `out = ({REAL})val`).
- CPU: `_conditional_gate_fe.py:224,237-246` (`cols_arr` in `_crit_np_dtype()` float32, so `u + v`, the ratio and the 3-way `stk.sum` are float32 ops with intermediate rounding).
- Statement: a single float32 `*`, `-` or `+` of float32 inputs matches the double-then-round result. The 3-operand row sum and the division can differ by 1 ULP (double rounding against single rounding), which can move a boundary row.
- Input: operands `1.0, 2**-24, 2**-24`. The host computes `(1 + 2**-24) + 2**-24`, which is `1.0` under ties-to-even. The kernel sums exactly in double and gets `1 + 2**-23`, which stores as `1.0000001f`.
- Disposition: DOC (selection-equivalence class), unless PARG1-2 is fixed by making the kernel mirror the host's float32 arithmetic. Then do this one in the same edit.

### PARG1-10: LOW (DOC). The docstring of `_plugin_mi_classif_batch_cuda` claims equivalence to njit "up to fp64 round-off"
- GPU: `_hermite_fe_mi.py:173-179` (docstring) against `:281-304,346-355` (percentile-edge binning, and the f32 cast under `relax_binning`).
- CPU: `_plugin_mi_classif_batch_njit` / `_plugin_mi_from_binned_njit` (`:60-98`, rank-based quantile codes).
- Statement: the in-code comment at `:302-304` records that edge binning is not bit-identical to rank binning at ties (an approved trade). The public docstring still promises fp64-round-off equivalence.
- Input: any heavily tied column, for example 50% zeros with `nbins = 20`. The rank codes split the tied block across bins; the edge codes keep it in one bin.
- Disposition: DOC. Correct the docstring to say "selection-equivalent; edge vs rank binning differs at ties".

### PARG1-11: LOW (DOC). The shared-fused pair MI reduces with a warp tree; the CPU finalize is sequential
- GPU: `_batch_pair_mi_cuda_shared_fused.py:111-131` (warp shuffle tree, then a serial sum over warps).
- CPU: `_batch_pair_mi_cuda_kernels.py:412-443` (`_mi_from_joint_counts`, sequential `total +=`), plus the numba single-launch kernel at `:204-221`.
- Statement: the results differ in the last ULP, so an exact-equality argmax over pairs can break ties differently. The cupy twin documents this (`:387-393`); this module does not. The kernel also assumes `blockDim.x % 32 == 0` (`nt >> 5`).
- Input: two pairs with identical joint tables in different cell order. Sequential summation gives equal MI; the tree sum can differ by 1 ULP.
- Disposition: DOC. State the ULP and tie caveat and the blockDim precondition in the module docstring.

## Read and found parity-clean
- `_gpu_resident_materialise.py` `fe_materialise` (`:55-100`) against `_feature_engineering_pairs/_pairs_materialise.py:83-140`: same float32 ops, same double-promoted div and ratio_abs, the same NaN propagation for max/min/signed, and the same nan_to_num.
- `_family_maxt_cuda_bits.py` v1 and v2 against `_family_maxt_null_v2_njit.family_null_max_bitsets`: the same per-bin class-then-remainder summation order, accumulated in double. The float32 table branch is latent: every caller read (`_family_maxt_gate.py`, `_family_maxt_gate_device.py`, `_honest_null_device.py`) passes the float64 `xlogx_table`.
- `_hermite_fe_mi.py` fused MI term against `_plugin_mi_from_binned_njit`: the same expression, and the empty-cell guards are equivalent.
- `_gpu_resident_select.py` interp against cupy/numpy linear: the symmetric lerp `ww < 0.5`, matching `frac >= 0.5` at the 0.5 boundary.
- `_gpu_pairs.py`: the merged code `va + vb*nba` gives a different label order than the CPU merge but the same partition, so MI is invariant. Products of int32 marginals are int64 under the installed numpy 2.3.5.
- `_fe_batched_mi_cmi.py` entropy/nnz kernels: double accumulation with a deterministic tree. Miller-Madow terms are consumed identically by `_mi_greedy_cmi_fe_gpu._cmi_from_binned_cupy` and the host `_cmi_from_binned` (checked line by line).

## Files read
- `_qbin_block_gpu.py` (full)
- `_family_maxt_cuda_bits.py` (full)
- `_gpu_resident_select_f32_compact.py` (full)
- `_gpu_resident_select.py` (full)
- `_gpu_resident_select_kernels.py` (grep of the key-transform and interp lines: 31-103, 134-267, 546-750, 905-907)
- `_hermite_fe_mi.py` (60-440)
- `_binned_numeric_agg_resident.py` (75-414, plus grep)
- `info_theory/_cmi_cuda.py` (1-600)
- `_gpu_pairs.py` (full)
- `_batch_pair_mi_cuda_kernels.py` (140-460)
- `_batch_pair_mi_cuda_shared_fused.py` (1-140)
- `_fe_batched_mi_cmi.py` (1-330, plus a grep of the remaining entropy kernels)
- `_resident_candidate_mi.py` (30-395)
- `_mi_greedy_cmi_fe_gpu.py` (full)
- `_gpu_resident_materialise.py` (55-104, plus grep)
- CPU twins: `_binned_agg_cheap_mi.py:25-68`, `_binned_numeric_agg_fe.py:436-475,511-517`, `_conditional_gate_fe.py:174-253`, `_mi_greedy_cmi_fe.py:176-271`, `_mi_greedy_cmi_fe_binning.py:52-103`, `_feature_engineering_pairs/_pairs_materialise.py:83-193`, `_family_maxt_null_v2_njit.py` (grep), `_fe_usability_signal.py:148-158`, `_fe_gpu_batch/_devices.py:65-70`
