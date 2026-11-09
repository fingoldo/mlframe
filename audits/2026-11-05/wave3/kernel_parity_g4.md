# Kernel parity audit, group 4 (wave 3, 2026-11-05)

Read-only audit. No code was changed and no state-changing git command was run. Each GPU or accelerated kernel was read next to its CPU twin.
Paths are relative to `src/mlframe/`.

## Findings (most severe first)

### PARG4-1: the GPU ROC/PR-AUC kernels skip the label remap the CPU path applies (HIGH)
- GPU: `metrics/_gpu_metrics.py:383` (`n_pos = cp.sum(actual)`), `:411` (`actual == 1`), `:447` (`total_pos = cp.sum(actual)`)
- CPU: `metrics/_core_auc_brier.py:844` (`y_true = _binary_labels_01(y_true)`), `metrics/_auc_labels.py:8-21`
- What differs: the CPU path maps {-1,1}, {1,2} and {0,2} labels to {0,1}, with the larger label as the positive class. The GPU path uses the raw labels, so `n_pos` is wrong and ROC counts only `== 1` as positive.
- Disagreeing input: `y_true` in {-1,+1} (balanced), `y_score` of shape (200_000, 5), auto-dispatch on a GPU host. On GPU `n_pos = sum(y) ≈ 0`, so ROC comes out as garbage or inf and PR is NaN. CPU returns the real AUCs. `y_true` in {1,2} also disagrees.
- Disposition: RESOLVED-needed. Apply `_binary_labels_01` in `compute_batch_aucs` before the GPU branch, and add a parity test with {-1,1} labels.

### PARG4-2: the numba TreeSHAP kernel accumulates in float32 while the cupy kernel uses float64 (HIGH)
- numba: `feature_selection/shap_proxied_fs/_shap_proxy_treeshap.py:167-168` (values and cover cast to `crit_np_dtype()`), `:454` (phi buffer is `crit_np_dtype()`), `:343` (`phi_row[...] +=` into that buffer), `:358-359` (cover ratio is float32/float32)
- cupy: `_shap_proxy_treeshap_gpu.py:244-249` (values, cover and phi forced to float64), `:139-140`, `:125`
- What differs: `crit_np_dtype()` is float32 by default (`MLFRAME_CRIT_DTYPE_RELAXED` defaults to "1", `filters/_fe_usability_signal.py:148-158`). The numba twin therefore:
  - rounds every phi accumulation to float32;
  - computes hot/cold cover ratios in float32;
  - rounds the leaf values and covers to float32 when the ensemble is extracted.

  The GPU twin widens the same float32-rounded values and covers to double, keeps the cover ratios and phi in double, and returns float64 phi. Routing is identical in both: float32 X, float32 thresholds, `isnan` sends a row to the default child, `<` goes left.
- Disagreeing input: the test fixture itself (xgb, 40 trees, depth 6, 220x9). Small phi entries pick up about 1e-7 of float32 rounding per added term, which shows up as about 1e-4 *relative* difference on near-zero entries.
- Disposition: RESOLVED-needed. Pick one contract: either the GPU follows `crit_float_dtype` for phi and the cover ratios, or the numba phi is float64 regardless of the crit relaxation. Additivity (`base + phi.sum == margin`) argues for float64. See the treeshap conclusion below.

### PARG4-3: the DTW GPU kernels use a finite 1e18 sentinel where the true DTW cost is +inf (MEDIUM)
- GPU: `signal/dtw.py:177-178` and `:368-370` (band initialised to 1e18), `:423` (full matrix), `:141-142` and `:256-257` (out-of-band neighbours read as 1e18), `:299` and `:308` (endpoint and backtrace sentinel)
- CPU: `signal/dtw.py:80-88` (`dtaidistance.dtw.distance` / `warping_path`). I did not read dtaidistance's internals; the claim below rests only on the GPU side treating the sentinel as a real, finite cost.
- What differs: once a real accumulated cost exceeds 1e18, an unreachable cell (row 0 / column 0 border, or outside the band) wins the min-of-three. The DTW then routes through a non-existent cell and the distance is underestimated.
- Disagreeing input: `x = [0, 0, 0]`, `y = [2e9, 2e9, 2e9]`, window >= 3, `backend="cupy"`. The true squared cost is 3 * 4e18 = 1.2e19, so the distance is about 3.46e9. On GPU, cell (1,2) takes `cost[0,1] = 1e18` (diag) instead of 4e18 (left), and later cells carry the same error, so the result is smaller. The same thing happens at any |x - y| scale above about 1e9.
- Disposition: RESOLVED-needed. Use `inf` (IEEE comparisons with `<=` work unchanged), or bail to CPU when `max|x - y|^2 * (n + m)` approaches 1e18.

### PARG4-4: the GPU subset-scan returns float32 losses and breaks ties arbitrarily; the CPU keeps the first-enumerated tie and lets a NaN take a slot (MEDIUM)
- GPU: `feature_selection/shap_proxied_fs/_shap_proxy_gpu.py:127-128` and `:148` (`out` is `crit_float_dtype`, float32 by default), `:67`, `:154-157` (non-finite becomes +inf, then `argpartition` and an unstable `device_argsort`)
- CPU: `_shap_proxy_search.py:125` (float64 top buffer), `:147-148`, `:54-69` (`_insert_top`: `loss >= worst` rejects a later equal loss, so the earliest combination wins ties; a NaN loss fails `>=` and is inserted into the last slot, then filtered at `:376`, so that cardinality loses one candidate)
- What differs:
  - Losses that are equal or within float32 rounding are cut at the top-N boundary differently. The GPU cut is non-deterministic.
  - When a cardinality has a NaN loss, the CPU can return one candidate fewer for it.
  - The margin sum order also differs (`base + phi_0 + phi_1 + ...` on GPU, `base + (phi_0 + ... + phi_{r-2}) + phi_{r-1}` on CPU). That is only FP-reorder noise, but it is enough to flip a float32 tie.
- Disagreeing input: phi with two identical columns (a duplicated feature), `top_n` small enough that the boundary falls inside the tied group. The CPU keeps the lower-index combination; the GPU keeps either one.
- Disposition: RESOLVED-needed. Make GPU `out` float64 (the kernel already accumulates in double). Break ties by combination index (a stable sort on `(loss, c)`). On CPU, map a NaN loss to +inf before `_insert_top`.

### PARG4-5: the row-attention stage-4 backend depends on batch size, and the two twins accumulate at different precision (MEDIUM)
- Dispatch: `feature_engineering/transformer/row_attention.py:423-424` (`n_queries_hint < 256` selects njit), `:338` (re-selected per `attend` call from `X_query.shape[0]`), `:176`
- njit: `_kernels_njit.py:219-237`. `dot`, `mean`, `var` and `s` start as the float64 literal `0.0`, so they accumulate in float64 (with `fastmath=True`), while weights are stored as float32.
- cupy: `_kernels_cupy.py:117-163`. Everything is float32 (`float dot`, `expf`, `float mean/var`).
- What differs: in Mode B, the same rows give features that differ by roughly 1e-6 depending on whether the batch had fewer or more than 256 rows. `apply_dedupe` (`_oof.py:251`, `|corr| > 0.99`) can then keep different columns for a near-threshold pair, unless `keep_columns` is passed.
- Disagreeing input: one fitted bank, `attend` on 100 query rows versus the same 100 rows inside a 1000-row batch on a GPU host. `y_mean` differs at about 1e-7 relative, and `y_std` more for sharply peaked weights.
- Disposition: DOC. The tolerance is documented at `_kernels_njit.py:3-7`. The docstring of `compute_row_attention` should add that the backend, and so the last bits, depend on batch size, and point to `gpu_stage4=True/False` for reproducible inference.

### PARG4-6: `fastmath=True` makes the njit softmax and std guards unreliable; the GPU guards are exact (LOW)
- njit: `_kernels_njit.py:18` (`fastmath=True`), `:43`, `:53` (`np.isfinite`), `:161` and `:237` (`var > 0.0` before sqrt)
- cupy: `_kernels_cupy.py:133`, `:141` (`isfinite`), `:163`. NVRTC without `--use_fast_math` keeps IEEE NaN and inf semantics.
- What differs: under LLVM fast-math flags (nnan/ninf), the compiler may assume `isfinite(x)` is true and that `var > 0.0` holds for NaN. The CUDA kernel evaluates them literally. `X` is checked for finiteness (`_utils.py:162-192`) and zero-norm rows are clamped (`_projection.py:401-404`), so logits stay finite. `y_train` is never checked for finiteness.
- Disagreeing input: `y_train` with one NaN reachable as a neighbour. GPU gives `y_mean = NaN` and `y_std = 0.0`, because `NaN > 0` is false. Under fastmath the njit result for `y_std` is undefined (NaN or 0).
- Disposition: RESOLVED-needed (cheap). Validate that `y_train` is finite in `_validate_row_attention_inputs`, since `X` is already held to that rule. That removes the only path that reaches the guards with a NaN.

### PARG4-7: the cupy stage-4 kernel silently downcasts float64 inputs to float32 (LOW)
- cupy: `_kernels_cupy.py:339-341` (`cp.asarray(..., dtype=cp.float32)` for q, k and y)
- njit: `_kernels_njit.py:170-243` reads the caller's dtype (float64 when `dtype=np.float64` is passed to `compute_row_attention`)
- What differs: with `dtype=np.float64`, the njit path does float64 dots on float64 data, while the GPU path rounds the inputs to float32 first. The output buffers are float32 on both paths (`row_attention.py:356-358`, `_oof.py:189-191`), so the loss is bounded but silent.
- Disagreeing input: `compute_row_attention(..., dtype=np.float64)`, where neighbour logits differ by less than float32 eps, on a GPU host versus a CPU host.
- Disposition: DOC. State that stage 4 is float32 on GPU whatever `dtype` is, or log once when float64 input is narrowed.

### PARG4-8: the RFF twins differ in precision, and the cuBLAS-broken fallback switches to float64 (LOW)
- njit: `_kernels_njit.py:88-96`. `acc = b[j]` takes `b`'s dtype; float32 b and W give a float32 accumulator.
- cupy: `_kernels_cupy.py:252-277` (cuBLAS matmul in `W.dtype`, input silently cast to `W.dtype` at `:256`/`:270`)
- Fallback: `_kernels_cupy.py:203-206`, `:244-246` (`_rff_numpy` computes in float64)
- What differs: on a host with a broken cuBLAS the "GPU" result is float64-accurate, so it differs from both the healthy GPU path and the njit path by float32 rounding of `XW + b`. That rounding is amplified by cos/sin when `|XW + b|` is large.
- Disagreeing input: d = 20_000 standardised features and m = 128. `|angle|` reaches about 1e2, so the float32 angle error is about 1e-5 and cos/sin differ at that level.
- Disposition: DOC. The fallback is more accurate, not wrong. Note the precision change next to the `cublas_matmul_ok()` branch.

### PARG4-9: the per-member MAE/std numpy reference and njit twin differ in dtype and summation for float32 input (LOW)
- numpy: `models/ensembling/per_member_tuning.py:82-94` (`diffs.mean`/`np.var` on float32 return float32, with pairwise float32 summation)
- njit: `models/ensembling/base.py:62-90` (float64 accumulators `_s_diff`/`_s_dev_sq`, float64 outputs `out_mae`/`out_std`)
- What differs: for float32 `arr` the two variants return different dtypes (float32 versus float64) and differ at about 1e-7 relative. With float64 input they agree (FP reorder only). NaN propagates in both: `if d < 0` lets NaN through, and so does `np.abs`.
- Disagreeing input: `arr` float32 of shape (3, 10_000_000), values about 1e3 plus noise. numpy float32 pairwise and njit float64 sequential differ in the last bits, and the result dtype differs.
- Disposition: DOC. The sweep uses float64 `standard_normal`, so the equivalence gate (`:131-132`) never sees float32. Either cast to float64 at the dispatcher or add a float32 grid point.

### PARG4-10: the frac-diff prefix pure-Python fallback divides by `weights[0]`; the njit kernel multiplies by its reciprocal (LOW)
- njit: `training/composite/transforms/_nonlinear_ewma_fracdiff.py:140`, `:149` (`* inv_w0`)
- Python fallback: `:173` (`/ weights[0]`)
- What differs: a 1-ulp difference per step, carried forward through the recurrence. The other twins are identical expression for expression: EWMA `:55-84` versus `:326-333` and `:359-368` (`isfinite` carry-forward in all of them), and frac-diff inverse `:87-128` versus `:401-413` and `:442-455`, all using `* inv_w0`.
- Disagreeing input: any `weights[0]` whose reciprocal is not exact, for example d = 0.4 with non-unit `weights[0]`, on a numba-less install versus a numba install.
- Disposition: RESOLVED-needed (one-line change to `* (1.0 / weights[0])`). Only matters with numba absent.

### PARG4-11: a forced DTW GPU backend bypasses the psi and unequal-length guards (LOW)
- Dispatch: `signal/dtw.py:585-591` (the env/kwarg-forced cupy/cuda branch returns before the guards at `:594-598`)
- GPU: `dtw_cupy`/`dtw_cuda` ignore `psi` and use a fixed `|i - j| <= window - 1` band (`:60-63`)
- CPU: `dtw_cpu` honours `psi`. Per the comment at `:597`, dtaidistance widens the band by `|n - m|`; that is the code's own statement, I did not read dtaidistance.
- What differs: `MLFRAME_DTW_BACKEND=cupy` with `psi=20`, or with `len(x) != len(y)`, silently returns a different distance and path than CPU. When `|n - m| > window - 1` the GPU returns `sqrt(1e18) = 1e9` (`:299`) as the distance of an endpoint it cannot reach.
- Disagreeing input: `x` of length 1000, `y` of length 1300, `window=200`, `MLFRAME_DTW_BACKEND=cupy`. The GPU returns 1e9; the CPU returns a finite distance.
- Disposition: RESOLVED-needed. In the forced branch, raise a `ValueError` for `psi > 0` or unequal lengths.

### PARG4-12: the numba.cuda RMSE result is not run-to-run reproducible (LOW)
- GPU: `metrics/_gpu_metrics.py:252-261`, `:331-341` (`cuda.atomic.add` per element)
- CPU: `:543`
- Disposition: REJECTED as a new finding. It is already documented in the `gpu_multiple_rmse_scores` docstring (`:277-282`). The difference is about 1e-15, with no decision impact in the CPU-vs-GPU metric reporting it feeds.

## Checked and found at parity (no finding)
- TreeSHAP routing: both twins use float32 X and thresholds, `isnan` sends a row to `children_default`, `x < thr` goes left, cold is pushed before hot, the per-sample tree order is the same, and phi is per-thread on GPU, so the `atomicAdd` there is deterministic.
- `compute_batch_aucs`: NaN score columns are masked on both backends (`:546-554`). The GPU ROC uses tie-averaged ranks; the GPU PR uses tie-run boundaries, so the order within ties does not matter.
- DTW full-matrix versus banded GPU kernels: same min-of-three tie order (diag <= up, then left), same sentinel, and the same backtrace policy (`:302-332` versus `:450-470`).
- EWMA single versus batched njit kernels, and frac-diff inverse single versus batched njit kernels: identical arithmetic.
- `_rolling_median`: the bottleneck fast path runs only on all-finite input, and the boundary windows are computed directly. I checked the `k > n` clamp by hand (n=3, k=5) and it is correct.

## Treeshap conclusion
The 1.1e-4 maximum relative difference in `test_shap_proxy_treeshap.py::test_treeshap_gpu_matches_numba` comes from a **real kernel difference, not tolerance**.
Under the default `MLFRAME_CRIT_DTYPE_RELAXED=1`, the numba twin does three things in float32: it accumulates phi (`_shap_proxy_treeshap.py:454`, `:343`), it computes the cover ratios as float32/float32 (`:358-359`), and it rounds leaf values and covers at extraction (`:167-168`).
The cupy twin widens the same values and covers to double, accumulates in double, and returns float64 phi (`_shap_proxy_treeshap_gpu.py:244-249`).
The routing, tree traversal and path polynomial are identical, so the whole gap comes from float32 accumulation and the float32 cover ratios. It is largest relative to small-magnitude phi entries.
The test asserts `atol=1e-9` (`tests/feature_selection/shap_proxied/test_shap_proxy_treeshap.py:184`), which only holds with `MLFRAME_CRIT_DTYPE_RELAXED=0`.
Fix the dtype contract (PARG4-2) rather than loosening the test.

## Files read
- `src/mlframe/signal/dtw.py` (full)
- `src/mlframe/metrics/_gpu_metrics.py` (full)
- `src/mlframe/metrics/_core_auc_brier.py` (`fast_aucs`, :824-861)
- `src/mlframe/metrics/_auc_labels.py` (:8-26)
- `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_gpu.py` (full)
- `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_search.py` (:54-150, :282-380)
- `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_objective.py` (:100-149)
- `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_treeshap_gpu.py` (full)
- `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_treeshap.py` (:200-462, plus dtype/routing grep)
- `src/mlframe/feature_selection/filters/_fe_usability_signal.py` (`_crit_np_dtype`, :148-158)
- `src/mlframe/feature_engineering/transformer/row_attention.py` (full)
- `src/mlframe/feature_engineering/transformer/_oof.py` (full)
- `src/mlframe/feature_engineering/transformer/_aggregation.py` (full)
- `src/mlframe/feature_engineering/transformer/_kernels_cupy.py` (full)
- `src/mlframe/feature_engineering/transformer/_kernels_njit.py` (full)
- `src/mlframe/feature_engineering/transformer/_utils.py` (`validate_numeric_input`, :162-192)
- `src/mlframe/feature_engineering/transformer/_projection.py` (L2-normalise guard lines)
- `src/mlframe/models/ensembling/per_member_tuning.py` (:80-139, plus grep)
- `src/mlframe/models/ensembling/base.py` (`_per_member_mae_std_njit`, :62-113)
- `src/mlframe/training/composite/transforms/_nonlinear_ewma_fracdiff.py` (:50-174, :322-561)
- `tests/feature_selection/shap_proxied/test_shap_proxy_treeshap.py` (:170-185)
