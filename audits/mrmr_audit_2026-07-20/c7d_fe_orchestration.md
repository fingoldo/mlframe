# FE orchestration / gating / budget infrastructure

10 findings, 3 proposals.

## Findings

### [P1] cpu_gpu_parity -- src/mlframe/feature_selection/filters/_fe_additive_fusion_gpu_resident.py:199

**The GPU-resident additive-fusion twin decides ALL fusion admissions (relevance floor, fused add/sub MI, OLS-R separability) on a strided subsample above MLFRAME_FE_FUSION_MAX_ROWS (default 250,000 rows), while the CPU sibling _fe_additive_fusion.propose_additive_fusions always decides on the full n -- a genuine change of decision basis, not just float-reduction-order noise, with no test comparing the two backends' actual admission verdicts above the 250k threshold.**

On a >250k-row fit under MLFRAME_FE_GPU_STRICT_RESIDENT, the GPU twin strides rows (`_stride = n_rows // _fus_max`) and scores every relevance-floor / fused-MI / OLS-R decision on that subsample (lines 199-249), while the CPU path (_fe_additive_fusion.py) scores on the complete data. A borderline-separable pair whose relevance/floor margin is thin can pass on one backend and fail on the other purely because the sample differs (e.g. a rare-stratum genuine additive term that the stride skips), producing a different set of admitted fused features / subsumed fragments between backends -- a real selection divergence the docstring calls 'selection-equivalent' but which is asserted, not verified: the only test (tests/feature_selection/mrmr/fe/test_fe_fusion_scoring_subsample.py) checks the stride arithmetic and that admitted output values are full-n, never that CPU and GPU agree on WHICH pairs get fused at n>250k.

### [P2] cpu_gpu_parity -- src/mlframe/feature_selection/filters/feature_engineering.py:666

**greater/less/equal binary ops return int dtype on CPU replay but float dtype on the GPU-resident twin, so a unary_binary recipe whose top-level binary_name is one of these emits a numerically-identical but dtype-divergent engineered column depending on backend.**

create_binary_transformations() defines greater/less/equal as `lambda x,y: np.greater(x,y).astype(int)` (feature_engineering.py:666-668), producing an int-dtype column when CPU-replayed via `_apply_unary_binary`. `_gpu_binary` in engineered_recipes/_recipe_unary_binary_gpu.py:168-173 instead does `cp.greater(a,b).astype(a.dtype)` where a.dtype is float32/float64 (per fe_gpu_f32_enabled()), so the GPU-resident replay of the identical recipe returns a float column. Values are the same 0/1, but the returned ndarray dtype differs by backend for any recipe using these three binary ops as the outer operator. The existing parity test (tests/feature_selection/gpu/test_recipe_unary_binary_gpu_parity.py) casts both sides to float64 before comparing, so it can never catch this dtype divergence.

### [P2] cpu_gpu_parity -- src/mlframe/feature_selection/filters/_fe_pure_form_retention_gpu_resident.py:183

**adds_nonlinear_value_batch_gpu_resident's docstring claims it keeps 'the EXACT per-candidate lstsq (SVD ...)' but the actual code solves the batched OLS via normal equations (Xc^T Xc) plus a trace-scaled ridge through cp.linalg.solve, never calling cp.linalg.lstsq -- a real (self-acknowledged) formula divergence from the CPU sklearn LinearRegression/SVD path on near-collinear candidate designs.**

For a candidate whose 12-column additive basis (from two correlated operand bases) is near-singular, the CPU path's SVD-based LinearRegression effectively rank-truncates and returns the minimum-norm least-squares solution, while the GPU path's normal-equations solve with an added 1e-10*trace ridge (line 220) returns a differently-regularized solution -- the code comment itself says this 'handles the near-collinear designs lstsq would rank-cut', i.e. concedes the two solvers do not agree on such inputs. The residual-based non-separability/relevance gate could flip near its threshold on such a near-singular design even though both paths are internally self-consistent; no test isolates a deliberately near-collinear candidate to confirm the gate verdict is stable across backends in that regime.

### [P2] test_gap -- src/mlframe/feature_selection/filters/_fe_family_timing.py:1

**_fe_family_timing.py (fe_family_timer, fe_timed, record_fe_family_wall, get_fe_family_wall, reset_fe_family_wall, log_fe_family_summary) has zero direct unit tests anywhere in the tests tree.**

No test asserts that fe_family_timer/fe_timed correctly accumulate wall time + invocation count under concurrent joblib-threading calls (the module's own docstring cites this as the reason for the module-level lock), that reset_fe_family_wall actually clears state between fits, or that get_fe_family_wall's snapshot matches what was recorded -- a regression in the lock or the accumulation logic (e.g. losing an increment under threading, or reset not being called between fits and wall times silently accumulating across unrelated fits) would go undetected.

### [P2] test_gap -- src/mlframe/feature_selection/filters/_fe_deadline.py:32

**_fe_deadline.py's public contract (set_fe_deadline / fe_deadline_passed / fe_budget_active / clear_fe_deadline) has no direct unit test; it is only exercised indirectly through full end-to-end MRMR.fit(max_runtime_mins=...) runs, which do not pin the thread-local semantics (e.g. clearing on exception via the finally in MRMR.fit, or the seconds-per-minute conversion) as an isolated behavioral test.**

A future edit to this module (e.g. changing the *60.0 conversion, or a code path that raises before the finally that calls clear_fe_deadline) would only surface as a flaky/slow full end-to-end fit test rather than a fast, targeted failure -- for example if clear_fe_deadline stopped being called on an exception path, a stale deadline could leak into the next fit on the same thread and silently truncate an unrelated unbudgeted fit, with no direct test to catch the leak.

### [P2] test_gap -- src/mlframe/feature_selection/filters/_fe_batch_dispatch.py:52

**choose_fe_batch_backend's dispatch precedence (MLFRAME_FE_VRAM_BACKEND force env > MLFRAME_FE_GPU_STRICT > KTC-tuned crossover > conservative CPU default, with a GPU choice downgraded to CPU when CUDA is unavailable) has no dedicated unit test pinning each branch.**

A refactor that reorders the precedence (e.g. making the KTC lookup win over an explicit MLFRAME_FE_VRAM_BACKEND=cpu force) would only be caught, if at all, by a broader GPU-batch-parity test that happens to run under one specific env configuration -- the explicit-force / STRICT / KTC / CUDA-downgrade branches are each independently untested.

### [P2] test_gap -- src/mlframe/feature_selection/filters/_fe_cpu_batch.py:31

**_cpu_col_chunk's RAM-budget-derived column-chunking logic (the per-column transient-scratch sizing that bounds cpu_fe_batch_mi's working set on very wide candidate matrices) has no direct unit test exercising the chunk-boundary / multi-chunk-vs-single-call code paths.**

A change to the `per_col = max(1, n) * 16` sizing or the `fit = budget // per_col` floor-division could silently produce a chunk width of 0 or 1 on a tight-RAM host (with no test forcing a small budget to exercise the multi-chunk branch in cpu_fe_batch_mi lines 86-91), degrading a wide-candidate FE batch to an unnecessarily slow per-column-sized loop or, in the other direction, failing to bound the working set on a genuinely RAM-constrained host.

### [P2] design -- src/mlframe/feature_selection/filters/_fe_rung_schedule.py:1

**No issues found in this cluster for the off-by-one / bin-edge-construction angle: _fe_rung_schedule.py's keep_frac / rel_floor cut logic, _fe_edge_mi.py's percentile-edge binary search, and the recipe-replay quantile/factorize edge code were all read in full and are correctly fenced (no off-by-one in loop bounds, slicing, or edge construction).**

_(no issues found in this cluster for this angle)_

### [P2] design -- src/mlframe/feature_selection/filters/engineered_recipes/_recipe_factorize.py:1

**No issues found in this cluster for the recipe-replay bit-exactness angle beyond the greater/less/equal dtype note above: unary_binary (incl. nested-engineered parents, prewarp, gate_med, frozen log-shift), factorize (pair + k-way chained), and target/hermite/cluster recipes were read/traced and replay closed-form from recipe.extra with no y-reference, matching their documented fit-time computation (including the documented float-rounding fix in _coerce_to_int_with_nan_handling and the frozen smart_log shift anchor).**

_(no issues found in this cluster for this angle)_

### [P2] gpu_residency -- src/mlframe/feature_selection/filters/_fe_batched_mi.py:1

**No wasteful GPU residency round-trips found in this cluster's GPU-touching files: every cp.asnumpy/D2H site reviewed (_fe_batched_mi.py, _fe_cmi_perm_null_gpu.py, _fe_additive_fusion_gpu_resident.py, _recipe_unary_binary_gpu.py, _fe_pure_form_retention_gpu_resident.py) is a single bounded scalar/output pull at a function boundary, most gated behind an explicit return_device opt-out so a caller doing several device calls back-to-back can stay fully resident.**

_(no issues found in this cluster for this angle)_

## Proposals

### (coverage_gap) Add a CPU-vs-GPU additive-fusion admission-parity test above the 250k scoring-subsample threshold

tests/feature_selection/mrmr/fe/test_fe_fusion_scoring_subsample.py only pins the stride formula and that admitted output values stay full-n. Add a test that runs the SAME >250k-row fixture through both _fe_additive_fusion.propose_additive_fusions (CPU) and _fe_additive_fusion_gpu_resident.propose_additive_fusions_gpu (GPU, forcing MLFRAME_FE_GPU_STRICT_RESIDENT + a MLFRAME_FE_FUSION_MAX_ROWS below n) and asserts the SAME set of admitted fusion names / subsumed fragments -- turning the docstring's 'selection-equivalent under a large strided subsample' claim into a verified property instead of an assumption.

### (coverage_gap) Add direct unit tests for _fe_family_timing.py, _fe_deadline.py, _fe_batch_dispatch.py, _fe_cpu_batch.py

Four small, single-purpose orchestration/budget modules in this cluster have no dedicated unit test file: the family wall-clock timer (concurrency-safe accumulation + reset), the thread-local FE deadline (set/pass/clear semantics, including clearing on an exception path), the CPU/GPU batch backend dispatcher (env-force / STRICT / KTC / CUDA-availability precedence), and the CPU batch RAM-budget column chunker (chunk-boundary behavior under a tight budget). Each is currently reachable only transitively through full end-to-end MRMR fits, so a regression in any of them surfaces (if at all) as a slow, hard-to-diagnose full-suite failure rather than a fast targeted one.

### (edge_case) Pin a near-collinear candidate design for the GPU pure-form-retention batched OLS gate

_fe_pure_form_retention_gpu_resident.py solves the batched 12-column OLS via normal equations + a small ridge rather than the CPU sklearn SVD path, and its own comment concedes the two solvers diverge on near-collinear designs. Add a regression fixture with two near-duplicate operand bases (e.g. two additive-basis columns that are near-linearly-dependent) and assert the retention verdict (kept/dropped) matches between the CPU _adds_nonlinear_value loop and the GPU-resident batch on that specific near-singular input, not just on well-conditioned canonical fixtures.
