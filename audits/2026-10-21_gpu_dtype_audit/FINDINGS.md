Written for: mlframe maintainers who touch the GPU kernels or the criterion-dtype policy.

# GPU kernels and the criterion dtype: what the launch-time audit and the GPU test run found

The criterion dtype (`_crit_np_dtype()`, float32 by default) was threaded through the FE and screening code. Two kinds of fault followed. A buffer in the new dtype was bound to a kernel pointer that still said `double*`, which the launch-time audit (`_benchmarks/kernel_arg_dtype_audit`, run as a pytest plugin over `tests/feature_selection/gpu`) reports as `expects float64*, got a float32 array`. And values that are statistics or exact comparisons were narrowed to float32 before the arithmetic that needs more, so a test of the numerical result failed. All of the following are fixed; each has a test that fails without the fix.

## Kernel argument and narrowing defects (real, fixed)

| Where | What was wrong | Effect |
|---|---|---|
| CUDA polynomial kernels (`_hermite_fe_mi`, `hermite_fe._polyeval_cuda`) | x uploaded as float32 into `const double* x` | garbage values (a near-constant vector) when the CUDA backend is chosen (n >= 500k or forced). Now the kernel takes `{REAL}` for x, compiled per dtype, and the upload follows the caller's dtype; computation stays double. Forcing float32 on a float64 column was measured and rejected: 3e-5 relative against the CPU path at degree 5 |
| `_screen_predictors` -> `mi_direct_gpu` | `freqs_y` uploaded as float32 into `compute_mi_from_classes_cuda` (`const double *freqs_y`) | the null-mean of the permutation test read as `inf`; found by the launch audit |
| `_device_quantile` | data and quantile levels cast to the criterion dtype, so the bit-identity self-check against `np.quantile` always failed | the device quantile was never used (25 tests failed). Now dtype-faithful |
| `searchsorted_right` (`_fast_host_ops`) | float64 edges rounded to float32 above 200k rows | a value on an edge moved to the neighbouring bin; the host binner and the device binner disagreed on one row of 250k |
| `_sync_free_qbin_codes` | interpolated edges in a form numpy does not use, compared in float32, and uploaded float64 when the VRAM flag was off while the host cast to float32 | the same disagreement; now numpy's symmetric lerp, comparison in the edge dtype, upload in the criterion dtype |
| `_gpu_batched_abs_corr` | float64 column narrowed to float32 before centring | a signal on a 1e7 or 1e10 offset was erased (|corr| 0.476 and 1e-8 instead of 0.498) |
| `audit_degenerate_columns` | collinearity tolerance 1e-9 on a float32 Gram matrix | exact linear duplicates went undetected on the CPU; the tolerance now follows the dtype of the matrix |
| `heldout_r2_scorer` (raw protection) | float32 design with a 1e-10 rank test | a column collinear with the selection looked well conditioned and gave noise as gain |
| `batch_pair_mi_cupy` | `freqs_y` uploaded in a dtype the CUDA twin does not share | two resident uploads of one vector |
| bit-set noise gate | per-column bin counts uploaded on every call; the (P, K) MI matrix copied back whole | resident-upload regression and 33 MB of D2H per fit; the matrix is now reduced to K gated values on the device |
| `_family_maxt_gate` | exact family gate read host copies of device codes | 17 MB of D2H per fit; the device branch is new (`_family_maxt_gate_device`) |

## Contracts that no longer matched the code (tests reframed, not loosened)

Parity tolerances written for float64 (`1e-9`, `1e-12`) now derive from the criterion dtype's epsilon with a stated multiple. The strict-GPU gate test follows the documented rule that an explicit `MLFRAME_FE_GPU_STRICT=1` skips the per-call work floors (the floors bind in AUTO mode). The upload test states its invariant as "a second dispatch re-uploads nothing".

## Open

- A static gate (`tests/feature_selection/test_no_unmarked_double_kernels.py`) requires every kernel module with a `const double*` parameter to carry an `Intentional float64` reason. It checks per file, not per parameter: a data column bound to `double*` inside a file that has some other marked site is not caught. The per-parameter caller check (template on `{REAL}` or mark) is still open for the larger kernel modules.
- The launch audit covers only the kernels that the GPU test directory reaches; kernels exercised elsewhere (the SHAP proxy kernels, for one) have not been audited.
- `test_f32_nameset_matches_f64` (three cases) still fails: the f32 and the f64 fit each select a different composite that takes a noise column as one operand (tracker row 2).
