# Test Coverage Map

11 findings, 16 proposals.

## Findings

### [P1] test_gap -- src/mlframe/feature_selection/filters/_screen_predictors_gate.py:76

**compute_selection_gate() -- the Fleuret-style acceptance gate (Miller-Madow bias correction, relative-to-first diminishing-returns floor, maxT permutation-null FDR floor) -- has zero direct unit test anywhere in tests/, only indirect exercise via full MRMR.fit() e2e tests.**

The function's own comments document a subtle historical bug (seed=101: using the FIRST selected feature's corrected gain instead of the running MAX collapsed the relative floor and let a cardinality-biased column survive). A future edit that reintroduces 'first' instead of 'max', or flips the MM-bias sign, or floors the maxT check on the conditional gain instead of the cached marginal MI (the exact bug class the inline comment warns against), would only be caught if it happens to move a full end-to-end MRMR.fit() selection on one of the ~600 existing e2e tests -- a narrow, indirect detection surface for a numerically subtle acceptance-gate bug.

### [P1] test_gap -- src/mlframe/feature_selection/filters/_param_accuracy_warnings.py:98

**warn_accuracy_suboptimal_params() and the ACCURACY_SUBOPTIMAL registry (the user-facing 'you set a value that degrades accuracy' UserWarning system) have zero test coverage anywhere in tests/.**

A future edit to a predicate (e.g. the `quantization_nbins < 5` lambda) or to the fire-once guard (`_accuracy_caveats_warned_`) could silently break the warning (never fires, fires every call instead of once, or the wrong caveat text is emitted) with nothing to catch it -- a user setting `min_features_fallback=0` or `fe_accuracy_gate=False` unintentionally would get no notice and no test would flag the regression.

### [P1] test_gap -- src/mlframe/feature_selection/filters/_fe_deadline.py:32

**The thread-local max_runtime_mins deadline gate (set_fe_deadline/fe_deadline_passed/clear_fe_deadline/fe_budget_active) that bounds the optional FE enrichment generators has zero test coverage, including the thread-local isolation property the module's own docstring calls out as load-bearing.**

The docstring explicitly warns that if an enrichment generator ever moves into a joblib worker thread, the thread-local deadline will silently fail to propagate (`threading.local` does not cross threads) -- exactly the kind of stale/scoped-state bug this task is meant to catch, and there is no regression test pinning current single-thread behaviour or asserting the deadline does NOT leak across two different threads or across two successive fits on the same thread (the 'never leak into the next fit' contract stated in `clear_fe_deadline`'s docstring).

### [P1] test_gap -- src/mlframe/feature_selection/filters/_fe_raw_redundancy_anchors.py:38

**build_raw_redundancy_anchors() -- a 347-line dual host/GPU-resident-code-path builder feeding drop_redundant_raw_operands -- has zero direct test; its GPU-resident vs host binning parity (_dev_from_cont/_dev_from_codes fallback) and its four early_return short-circuit branches are only reachable transitively through full-fit raw-redundancy e2e tests.**

A GPU-resident quantile-bin (_quantile_bin_gpu_resident) that silently diverges from the host _quantile_bin at a tie or NaN-cleaned boundary would change which raw operands get dropped as redundant, but nothing directly compares the `_gate_resident=True` and `_gate_resident=False` code paths' outputs on the same input -- only whichever backend happens to run on the CI/dev machine is exercised by the e2e suite.

### [P1] test_gap -- src/mlframe/feature_selection/filters/_gpu_resident_radix_ktc.py:76

**radix_select_threads() / radix_select_f32_variant() -- the kernel-tuning-cache dispatch for the radix-select block-size and f32-window-match variant -- have zero test coverage, including their CPU-only, no-GPU-required fallback branches.**

The module's own comment (line ~139) documents a real, previously-shipped-silently bug: the sweep probe wrote its variant override onto the wrong module binding (a re-export alias) and was a complete no-op the whole time it was live, discovered 2026-07-18. No regression test pins that fix or the fallback-to-default behaviour when `_RADIX_THREADS_SPEC is None` / `.choose()` raises / returns a malformed string -- all of which are plain-Python paths requiring no cupy/CUDA hardware to test.

### [P2] test_gap -- src/mlframe/feature_selection/filters/mrmr/_mrmr_class_shared.py:15

**_mrmr_y_columns() (the multi-output y column iterator shared by MRMR's fit machinery) has zero test coverage under any of the three branches (pandas.DataFrame, polars.DataFrame, raw ndarray) by name; existing multi-output tests exercise MRMR.fit end-to-end but never call this helper directly.**

The polars.DataFrame branch is duck-typed via `type(y).__module__.startswith('polars')` rather than isinstance, and the raw-ndarray branch synthesizes label names as f'y{k}' -- a regression in either (e.g. polars renaming its module path, or an off-by-one in the ndarray label index) would only surface if an e2e multi-output test happens to use that exact y type, which the existing multi-output suite may not cover for polars y specifically.

### [P2] test_gap -- src/mlframe/feature_selection/filters

**Batch of confirmed zero-coverage-anywhere-in-tests/ modules, mostly GPU-hardware-gated kernel-tuning-cache infrastructure: _gpu_hw_launch.py, _gpu_policy.py, _gpu_resident_histgate_ktc.py, _batch_mi_noise_gate_tuning.py, _resident_bincount.py, _resident_raw_mi.py, _cupy_polynom_optimizer.py, _fe_pure_form_retention_gpu_resident.py, _usability_gpu.py, _feature_engineering_pairs/_pairs_common.py, _fe_linear_explainability.py.**

Most of these need real CUDA hardware to exercise their GPU kernel path meaningfully, so a full omission is understandable at P2 rather than P1, but every one of them also has a plain-Python fallback/dispatch surface (no-cupy default, KTC-lookup-miss default) that is cheap to unit test on CPU and currently is not, unlike sibling KTC modules elsewhere in the same package that DO have a 'no sweep on default' regression test (test_ktc_dispatch_no_sweep_on_default.py).

### [P2] bug -- src/mlframe/feature_selection/filters

**Widespread `except Exception: <bare fallback, e.g. pass/return -inf/return True>` pattern (178 occurrences across ~120 files) with no logger call and no re-raise, e.g. _mrmr_fit_impl/_fit_impl_core.py:6619,6910,7102,7217,7477,7761,8470 and _gpu_resident_basis.py:986,1019,1047,1064,1118,1146,1170,1187.**

Each sampled instance is a deliberate, well-documented degrade-to-a-safe-sentinel design (e.g. a QR/lstsq numerical-gate returning -inf on failure, or a GPU-kernel-tuning lookup falling back to 'cpu' on any exception) rather than a bug in itself, but because none of them log at WARNING/DEBUG with the exception, a genuine new bug introduced inside one of these try-blocks (e.g. a shape mismatch in the lstsq design matrix) is indistinguishable in production logs from the intended, harmless fallback -- it would silently degrade selection quality with zero diagnostic trail rather than surfacing as an error.

### [P2] design -- src/mlframe/feature_selection/filters

**No bare `except:` clauses (as opposed to `except Exception:`) found anywhere in the cluster.**

No issues found in this cluster for this angle -- every broad-catch site uses `except Exception` (or a narrower type), never a bare `except:` that would also swallow KeyboardInterrupt/SystemExit.

### [P2] cpu_gpu_parity -- src/mlframe/feature_selection/filters/_gpu_resident_radix_ktc.py:105

**The 'linear' / 'bsearch' / 'v3' f32 radix-select window-match variants and the swept threads/block values are asserted bit-identical in the module's docstrings and comments ('order-statistic invariance', 'sum-reduction invariance') but this claim is enforced only by the internal sweep's own equiv_rtol=1e-6/equiv_atol=1e-6 check inside sweep_backend_grid, never by an independent pytest asserting exact/near-exact equality on a fixed seed -- there is no committed regression test pinning the parity claim outside the tuning-sweep machinery itself.**

If sweep_backend_grid's equivalence check were ever loosened, disabled under a budget flag, or skipped on a GPU-absent CI box, a genuine numeric divergence between the 'v3' fast path and the reference 'linear' path (e.g. an off-by-one in the parallel per-rank scan) could ship as the new default variant with nothing outside the sweep itself to catch it.

### [P2] gpu_residency -- src/mlframe/feature_selection/filters/_fe_raw_redundancy_anchors.py:178

**cp.asnumpy() round-trips inside _raw_codes()/the eng_bin loop (lines ~178, ~211, ~304) pull GPU-resident quantile-bin codes back to host immediately after computing them.**

No issues found in this cluster for this specific angle -- each cp.asnumpy() call is load-bearing, not wasteful: the resident `_dev` array is retained separately in `_raw_dev_cache`/`eng_bin_dev` and returned to the caller for device-side scoring, while the host copy (`_out`/`eb`) is the documented fallback the caller uses when a downstream device site is unavailable. The dual host+device storage is intentional, not an avoidable extra round trip.

## Proposals

### (coverage_gap) test_screen_predictors_gate_boundaries.py

Directly unit-test compute_selection_gate() in _screen_predictors_gate.py with hand-built cached_MIs/predictors/factors_nbins fixtures: (a) MM-bias-corrected gain uses the RUNNING MAX over already-selected predictors, not the first-selected one, for the relative-to-first floor; (b) the maxT FDR floor (fdr_gain_floor) compares the candidate's cached MARGINAL MI, not the conditional Fleuret gain; (c) the MM correction is skipped for interactions_order>=2 (joint candidates); (d) build_dcd_state() returns None (never raises) when dcd_config.get('enable') is False, is None, or DCD init throws.

### (coverage_gap) test_param_accuracy_warnings.py

Unit-test warn_accuracy_suboptimal_params() in _param_accuracy_warnings.py against a stub estimator object: default-config estimator emits nothing; setting one ACCURACY_SUBOPTIMAL-listed attr to its bad value emits exactly one UserWarning naming that attr, its cost, and its restore text; a second call on the same estimator (after _accuracy_caveats_warned_ is set) emits nothing; an estimator missing one of the registry's attr names is skipped without raising; multiple simultaneous bad values are all listed in one consolidated warning.

### (coverage_gap) test_fe_deadline_thread_and_leak_isolation.py

Unit-test set_fe_deadline/fe_deadline_passed/clear_fe_deadline/fe_budget_active in _fe_deadline.py: fe_deadline_passed() is False just before the deadline and True just after (boundary at >=); fe_budget_active() reflects whether ANY deadline is set regardless of elapsed time; a deadline set on one Python thread reads as unset (fe_deadline_passed()==False, fe_budget_active()==False) on a second thread; clear_fe_deadline() resets state so a second fit on the same thread does not inherit the prior fit's deadline.

### (coverage_gap) test_raw_redundancy_anchors_direct.py

Directly call build_raw_redundancy_anchors() in _fe_raw_redundancy_anchors.py (not via a full MRMR.fit) on a small synthetic fixture: assert the four early_return short-circuit conditions (no engineered survivors, no raw survivors, no replayable anchor, no eng_consumers) each return (sel, []); assert gate_resident=True and gate_resident=False produce identical raw_marginal()/eng_bin outputs on the same input (CPU/GPU-resident parity), skipping the resident assertions gracefully when cupy is unavailable.

### (coverage_gap) test_gpu_resident_radix_ktc_cpu_fallback.py

Unit-test radix_select_threads()/radix_select_f32_variant() in _gpu_resident_radix_ktc.py with _RADIX_THREADS_SPEC/_RADIX_F32_VARIANT_SPEC monkeypatched to None (returns the documented 512/'v3' defaults), and with a stub spec.choose() raising or returning a malformed 'th_' / non-listed string (falls back to default without raising). Also pins the 2026-07-18 override-wiring fix: verify _radix_edges_with_threads()/_radix_edges_with_f32_variant() write the override onto _gpu_resident_select_kernels (the owning module), not the _gpu_resident_select re-export alias.

### (coverage_gap) test_mrmr_y_columns_multioutput_types.py

Directly unit-test _mrmr_y_columns() in mrmr/_mrmr_class_shared.py against all three input types with 2+ output columns: pandas.DataFrame (labels = real column names), polars.DataFrame (labels = real column names via the duck-typed module-name branch), and a raw 2D ndarray (labels = 'y0','y1',... in column order) -- assert both the label and the array values for every yielded column.

### (coverage_gap) test_mrmr_config_mixin_direct.py

Directly unit-test _MRMRConfigMixin methods in mrmr/_mrmr_class_config.py on an isolated MRMR instance: _coerce_target_dtype at the exact int16 boundary (vmin=-32768/vmax=32767 downcasts and logs info; vmax=32768 keeps int64 and emits the UserWarning-level logger.warning with both ranges in the message); _effective_random_seed precedence (random_state wins when both random_state and the deprecated random_seed are set, falls back to random_seed when random_state is None, None when neither set); _effective_n_jobs resolves n_jobs=-1 to psutil.cpu_count(logical=False) and passes through any other int unchanged; clear_fit_cache() drains _FIT_CACHE and returns the pre-clear count.

### (coverage_gap) test_batch_mi_noise_gate_tuning_fallback.py

Unit-test _batch_mi_noise_gate_tuning.py's KTC dispatch functions with no-cupy / no-tuned-cache-entry stubs, mirroring the existing test_ktc_dispatch_no_sweep_on_default.py pattern used for sibling KTC modules, to close the parity gap between this module and its already-tested siblings.

### (coverage_gap) test_resident_bincount_identity.py

Numerically verify the GPU-resident bincount kernel(s) in _resident_bincount.py against plain np.bincount on random int arrays across a few sizes/dtypes (skip gracefully when cupy/CUDA is unavailable) -- currently this small, hot, foundational kernel has literally zero test references anywhere in tests/.

### (coverage_gap) test_resident_raw_mi_cpu_gpu_parity.py

Compare _resident_raw_mi.py's GPU-resident raw-MI computation against the CPU MI path on identical discretized codes for a handful of shapes/cardinalities, asserting near-bit-identical MI values -- currently zero test coverage for a function explicitly named as a CPU/GPU parity-sensitive kernel.

### (coverage_gap) test_fe_linear_explainability.py

Unit-test the linear-explainability helper(s) in _fe_linear_explainability.py on a small synthetic dataset where the target is an exact known linear combination of a subset of engineered features, asserting the returned coefficients/contributions correctly identify the true signal columns and near-zero out the noise columns.

### (coverage_gap) test_fe_pure_form_retention_gpu_resident_parity.py

Compare the GPU-resident retention-decision functions in _fe_pure_form_retention_gpu_resident.py against their host counterparts in _fe_pure_form_retention.py on identical synthetic engineered-feature pools, asserting the retained/dropped column sets match exactly -- currently zero direct test of the GPU twin despite the host module having several dedicated tests.

### (coverage_gap) test_cupy_polynom_optimizer_parity.py

Compare _cupy_polynom_optimizer.py's cupy-backed polynomial-fit optimizer against _numba_polynom_optimizer.py (its CPU/numba sibling, which has test_numba_polynom_optimizer.py) on the same synthetic fixture, asserting selection-equivalent (or bit-identical within tolerance) results -- currently only the CPU twin is tested.

### (coverage_gap) test_usability_gpu_fallback.py

Unit-test _usability_gpu.py's usability-scoring dispatch functions with cupy unavailable / CUDA absent, confirming a clean CPU fallback (no exception, correct sentinel/None) -- currently zero coverage anywhere, unlike the CPU-only _usability_lists.py / _usability_njit_pool.py siblings which do have tests.

### (coverage_gap) test_gpu_hw_launch_occupancy_no_device.py

Unit-test occupancy_block_candidates()/device_props() in _gpu_hw_launch.py with CUDA/cupy unavailable, asserting device_props() returns a falsy/None sentinel and every KTC caller that depends on it (_gpu_resident_radix_ktc._radix_threads_variants and its histgate/noise-gate siblings) correctly falls back to its raw seed candidate list rather than raising.

### (coverage_gap) test_except_exception_fallback_logging.py

A targeted regression/lint-style test asserting that the highest-risk broad except-Exception fallback sites in _mrmr_fit_impl/_fit_impl_core.py's raw-operand-readd gates (lines ~6619, ~7477, ~8470 -- the permutation-significance 'estimator error -> permissive re-add' gates) log at DEBUG/WARNING with the exception repr when they fire, so a genuine new bug inside those try-blocks is distinguishable in logs from the intended graceful-degrade path; currently none of them log.
