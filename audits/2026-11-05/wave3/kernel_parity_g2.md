# GPU/CPU kernel parity audit, group 2 (wave 3, 2026-11-05)

Scope: the 15 files listed at the end. This was a read-only audit: no code was edited and no tests were run. Every disagreement input below comes from reading the code, not from a run.
Paths are relative to `src/mlframe/feature_selection/filters/`. Where the CPU twin lives in a file outside the scope and was not read, the finding says so.

## Findings (most severe first)

### PARG2-1: High. The pair residual verdict depends on which backend is chosen
- GPU: `_fe_sufficient_summary_pairs.py:557-563` (`budget = PAIR_WORK_BUDGET_GPU`, 1e11) feeding `pair_mi_block_v2` at `:228-272`
- CPU: the same lines with `PAIR_WORK_BUDGET` (4e9) feeding `_pair_mi_block_cpu` at `:137-175`
- Statement: the row stride, and so `n_used`, the scored rows, the chi-square `ceiling` (`:578`) and the `n_used < PAIR_MIN_ROWS -> None` exit, are computed from a budget that differs 25 times between the GPU and CPU backends. `pair_residual_backend_choice` takes the backend from the per-host KTC, so the early-stop verdict depends on the host. The comment at `:535` ("both backends return the same result, so a broken cache cannot change the verdict") is false.
- Disagreeing input: p=2000 raw columns (n_pairs = 1 999 000) and n=10 000 rows. CPU: 2e10 > 4e9, so stride 5, n_used 2000, and the score and ceiling come from 2000 rows. GPU: stride 1, n_used 10 000. At p=3000 and n=6000 the CPU branch gives stride 7 and n_used 858 < 1500, so it returns `None` (FE keeps running), while the GPU branch certifies and can stop the FE search.
- Disposition: RESOLVED-needed. Choose the stride from one budget that does not depend on the backend (the backend should change speed only), or make the budget part of the documented contract and remove the "cannot change the verdict" claim. Add a test that forces both backends at a shape over 4e9 pair-rows and asserts the same `(mi, ceiling)`.

### PARG2-2: High (when a modulus above 127 is used). The int8 residue table wraps in modular-rank kernels v7 and v8 (v8 is the default)
- GPU: `_pairwise_modular_rank_gpu.py:333-335` (`cp.mod(cols_d, k).astype(cp.int8)`), consumed at `:208-215`
- CPU: `_pairwise_modular_fe._residue_rank_scores` / `_residue_rank_scores3` (called at `:405`; the bodies were not read in this audit)
- Statement: residues are stored as signed int8, so any modulus k >= 129 maps the residues 128..k-1 to negative values. The kernel then indexes `joint[r * n_cls + ycls]` and `ptab[ra * k + rb]` with a negative `r` or `ra`, which writes shared memory out of bounds or miscounts. The shared-memory gate at `:309-313` admits such k: k=200 with n_cls=2 needs 200*2*4 + 200*4 + 200*200 = 42 400 bytes, under the 48 KB limit. Versions 1 to 6 compute the residue in int or long long and are not affected.
- Disagreeing input: `mods` containing 150 or 200, any `cols` with values at or above 128 mod k. The CPU int64 residue is in [0, k) and the GPU residue is negative.
- Disposition: RESOLVED-needed. Use a `uint8` table and refuse the table versions when `max_k > 255`, or refuse when `max_k > 127`. Add a regression test with k=150 against the CPU kernel. If the callers' moduli are capped at 127 or below, add that cap as an assert at this boundary and set the finding to DOC.

### PARG2-3: Medium. The cupy polynom optimiser's float32 basis and GEMM overflow where the float64 numba twin stays finite
- GPU: `_cupy_polynom_optimizer.py:60-72` (float32 `B`), `:208-209`, `:241-242` (float32 operands and coefficients), and `:155` (`col_finite` rejects any column with an inf)
- CPU: `_numba_polynom_optimizer` (the float64 `_*val_njit` kernels; not read)
- Statement: the polynomial basis recurrence and `Ba @ Ca.T` run in float32, so a value whose magnitude exceeds about 3.4e38 becomes inf. The candidate is then marked non-finite and can never win. The double-precision twin scores the same candidate normally. The docstring (`:52-58`) treats only rounding noise, not overflow.
- Disagreeing input: a hermite basis with degree 25 and a z-scored operand containing one outlier with |z|=40. He_25(40) is about 40^25 = 1.1e40, which is inf in float32 and finite in float64. Lower degrees reach the same point through the GEMM when large coefficients (`coef_range`) are combined with |x| around 1e3 (laguerre or raw operands).
- Disposition: RESOLVED-needed. Before the search, check whether `max|x|^degree * max|coef| * (degree+1)` exceeds the float32 range, and if it does, build `B`, `Ca` and `Cb` in float64 for that pair. Add a test with an outlier operand that the float32 path currently rejects.

### PARG2-4: Medium. `gpu_pairs_fe_mi` casts float64 candidates to float32 before binning; the CPU dispatcher bins the input dtype
- GPU: `_gpu_resident_pair_mi.py:104` (`np.ascontiguousarray(cand, dtype=np.float32)`), then `_gpu_resident_discretize_codes`
- CPU: `_feature_engineering_pairs._pairs_dispatch._dispatch_batch_mi_with_noise_gate`, which bins `cand` through `discretize_2d_quantile_batch` without changing its dtype (the call is shown at `:175`; the body was not read)
- Statement: for a float64 candidate, distinct doubles that round to the same float32 fall into one bin on the GPU and into separate bins on the CPU. The occupied-bin count `bx` and the observed MI then change, and so does the analytic keep/reject decision. The docstring says "GPU quantile binning == CPU" (verified maxdiff 0), but that check only holds for float32 input (the sweep at `:144` passes float32).
- Disagreeing input: a candidate column `1.0 + i * 1e-9` for i in 0..n-1, with y correlated to i. In float32 every value is 1.0, so there is one bin and MI is 0. In float64 the CPU builds 20 equal-frequency bins and gets a positive MI.
- Disposition: RESOLVED-needed. Upload in the input's native dtype (as the comment at `_gpu_resident_fe.py:380-384` says the resident binning does), or cast on the CPU side as well. Add a parity test with a float64 near-tied column.

### PARG2-5: Medium. The default honest-null root-split kernel v5 picks the cut by the float32 score, so accept/reject can differ from the host
- GPU: `_honest_null_k4.py:84-131` (`root_split_stats_k5`: first strict minimum of the float32 `ENT2` score; the exact double gain is computed only at that cut)
- CPU: `_mdlp_best_split_inc_njit` / `_count_candidates_njit` (named in `_honest_null_device.py:7-8`; not read). Exact versions on the device: `_SOURCE_K4` at `:145-194` and `_SOURCE_K2` in `_honest_null_device.py:141-198`.
- Statement: when two candidate cuts have float scores within float rounding of each other, v5 can pick a cut whose exact gain is lower than the host's best. The analytic test (`_accepted`, `_honest_null_device.py:237-250`) then sees a smaller gain and can reject a (column, permutation) pair that the host accepts. Rejected pairs are never redone on the host (`:451-483` redoes only accepted ones), so that pair keeps its fallback-code MI. The module docstring documents this (7% of indices differ, gain error up to 1.8e-7, "can only miss a pair whose gain sits within 1e-7 of the threshold").
- Disagreeing input: n=100 000, two cuts whose summed `ENT2` values differ by less than one float32 ulp (about 0.008 at 6.9e4 nats) while their exact gains differ by about 1e-7, with the host's best gain within 1e-7 of the threshold `p * ncand = alpha`.
- Disposition: DOC (documented trade-off, outside the selection-equivalence bar only at the 1e-7 margin). Open question for the owner: should pairs near the threshold be redone on the host as well, for example any pair with `p * ncand < 10 * alpha`?

### PARG2-6: Low. The fused symmetric-uncertainty (SU) noise-gate kernel does not clamp a negative MI; the host and the two-kernel device path do
- GPU (fused): `_batch_mi_noise_gate_kernels.py:738-749` (`_fused_statistic`: `2.0 * mi / denom`)
- CPU: `_mi_from_counts_cpu` at `:198-200` (sets `mi_xy` to 0 when it is negative before SU). The two-kernel device twin at `:389` also clamps (`max(mi, 0.0)`).
- Statement: when MI rounds to a value just below 0 (constant or near-constant column), the fused SU is a tiny negative number while the host gives 0.0.
- Disagreeing input: a candidate column with a single occupied bin, with `freqs_y` such that `jf/(1*py)` rounds below 1 in one cell. The fused kernel returns about -1e-17; the CPU returns 0.0. `_gate_from_mi_njit` keeps `om <= 0` unchanged, so the gated output also differs in sign.
- Disposition: RESOLVED-needed (one line: `mi = max(mi, 0.0)` in `_fused_statistic` before the SU division). Add a test with a constant column against `_mi_from_counts_cpu(use_su=True)`.

### PARG2-7: Low. The fused analytic-gate kernel does not check for negative target codes
- GPU: `_feature_engineering_pairs/_pairs_analytic_gpu.py:105-121` (the code side is checked with `lo < 0`; `ky_eff` bounds only the maximum y), with shared-memory indexing at `:46`
- CPU: the host `npermutations=0` observed-MI kernel (not read). The generic device path at `:147-159` hands y to `binned_mi_from_codes_gpu`.
- Statement: a target code below 0 indexes `sh[cj*M + cx*Ky + cy]` out of bounds (corrupting a neighbouring column's histogram or faulting), while the fallback paths handle or reject it.
- Disagreeing input: `classes_y` containing a -1 sentinel.
- Disposition: RESOLVED-needed. Return `None` (generic path) when `yc.min() < 0`, matching the guard that already exists for the code side.

### PARG2-8: Low. In `mi_direct_gpu`'s per-permutation loop, the observed MI comes from the host and the permutation MIs from the device
- GPU: `gpu.py:414-431` (`compute_mi_from_classes_cuda` result compared with `mi >= original_mi`)
- CPU: `gpu.py:532-535` (`original_mi` from the host `compute_mi_from_classes`)
- Statement: a permutation that keeps the contingency table unchanged has exactly the same MI mathematically. The device reduction order can make it one ulp below the host value, so the GPU does not count it as a failure where the CPU `mi_direct` (host against host) does. `nfailed`, `confidence` and the zeroing at `max_failed` can then differ.
- Disagreeing input: n=6, x=[0,0,0,1,1,1], y=[0,0,1,1,1,0], and a permutation that only swaps rows 0 and 1 (same x and same y). The device MI equals the observed MI only up to rounding.
- Disposition: DOC (the docstring at `:490-492` already says counts differ "only through floating-point rounding"). A cheap fix exists: compute the observed MI with the same device kernel.

### PARG2-9: Low. The prewarp kernel and its own exception fallback round differently under the float32 critical dtype
- GPU kernel: `_gpu_resident_fe.py:456-467` (works in double, then casts to `{REAL}`=float)
- Fallback: `_gpu_resident_fe.py:566-577` (a cupy chain on float32 `xf` with Python-float scalars, so every step is computed in float32)
- Statement: the result of the prewarp depends on whether the fused kernel compiled. The docstring at `:489` ("returns a device double column") is out of date under `crit_float_dtype()=float32`.
- Disagreeing input: legendre min-max with lo=1e4 and span=1e-2, x=1e4+3e-3. Kernel: `2*(x-lo)/span-1` in double, then rounded. Fallback: `x-lo` in float32 loses about 1e-3 relative precision.
- Disposition: DOC. Fix the docstring, and possibly make the fallback upcast to float64 and round once, to match the kernel.

### PARG2-10: Low. Tie handling in the cupy polynom rank binning differs from the numba twin
- GPU: `_cupy_polynom_optimizer.py:93-105` (value-edge `batched_quantile_bin_gpu`: tied values go to one bin)
- CPU: `_quantile_bin_njit` (rank split; not read)
- Statement: tied candidate values are binned differently, so MI differs whenever the combined column has ties (for example the `div` eps branch, `sign` in logabs, or saturated `atan2`). The module docstring (`:21-23`) still describes `cp.argsort`, which the code no longer uses.
- Disagreeing input: `bf=atan2` with operands that are mostly 0, producing a column with about 50% exact zeros, at `n_bins=10`.
- Disposition: DOC (already documented as selection-equivalence). Update the header docstring at `:12` and `:21`.

### PARG2-11: Low. The fused pair-residual kernel computes per-cell terms in float32 by default; the CPU uses double
- GPU: `_fe_sufficient_summary_pairs.py:253-263` (`{REAL}` = float under the relaxed critical dtype, `gpu_real_type` at `:282-286`)
- CPU: `:158-169`
- Statement: the best pair can change on near-ties, and `best_mi` compared with `ceiling` can flip when it sits within about 1e-6 relative of the ceiling. This is covered by `pair_equiv_tolerance`.
- Disagreeing input: two pairs whose double MIs differ by less than 1e-7 relative.
- Disposition: DOC (tolerance documented, and the decision direction is safe only when the GPU errs high; it can err either way).

## Checked with no parity finding
- `_honest_null_k3.py`: arithmetic and tie rule identical to k2, and it is not dispatched.
- `_honest_null_device.py`: v1/v2 compute the XL table and log term exactly as the host's 32768-entry table (`fmad=false`). NaN rows are coded `nb_reg` in both the fallback and the redo path.
- `_cell_moments_kernel.py`: the double atomic accumulators use the scatter form's per-row terms; only the order of the atomic additions differs.
- `_batch_mi_noise_gate_kernels.py`: the histogram kernels (global, shared, column-major) use integer counts and are bit-identical. The MI-from-counts kernel matches `_mi_from_counts_cpu`'s reduction order. The LCG shuffle builder is bit-identical per row.
- `_gpu_resident_operand_build.py`: `batch_unary` op codes match `_unary_apply` (pow(x,-1), sqrt(abs), pow(x,-1/3), pow(x,-1/2), sign with NaN passing through), and only the float64 group is batched.
- `_orthogonal_univariate_fe/_gpu_resident_cross_basis.py`: the mean-fill of non-finite rows and the operand dtype mirror the host. The ~2.5e-6 drift (backward Clenshaw vs forward recurrence) is documented.
- `_internals.py`: no device kernel. `smart_log` keeps the dtype.

## Files read
- `_batch_mi_noise_gate_kernels.py`
- `_gpu_resident_fe.py`
- `_cupy_polynom_optimizer.py`
- `_orthogonal_univariate_fe/_gpu_resident_cross_basis.py`
- `gpu.py`
- `_gpu_resident_operand_build.py`
- `_honest_null_k3.py`
- `_honest_null_k4.py`
- `_honest_null_device.py`
- `_cell_moments_kernel.py`
- `_gpu_resident_pair_mi.py`
- `_pairwise_modular_rank_gpu.py`
- `_internals.py`
- `_feature_engineering_pairs/_pairs_analytic_gpu.py`
- `_fe_sufficient_summary_pairs.py`
- `_gpu_resident_shared.py` (only the `_unary_apply` lines, grep, to check the op-code parity of `batch_unary`)

Not read (CPU twins named above): `_numba_polynom_optimizer.py`, `_pairwise_modular_fe.py`, `_mdlp_honest_njit.py`, `_pairs_dispatch.py`, `discretization`, `_fe_batched_mi.py`, `_gpu_kernel_sources.py`, `_gpu_batched.py`.
