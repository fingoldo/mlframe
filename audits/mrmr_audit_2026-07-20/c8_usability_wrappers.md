# Usability-aware selection / estimator bridge / misc utilities

14 findings, 7 proposals.

## Findings

### [P1] design -- src/mlframe/feature_selection/filters/friend_graph.py:610

**The 2026-07-20 Inf-FS centrality addition is computed, tested, and written into to_meta()'s JSON summary, but never reaches the rendered plot (the module's primary human-facing artifact) -- friend_graph_to_figurespec's node_hovertext, node_color and node_size never reference centrality/low_centrality.**

A user calls build_friend_graph(...) then plot_friend_graph(...) to visually diagnose universal-soldier/aggregator features (the module's stated purpose). The rendered graph's hover text lists entropy/relevance/weighted_degree/shared_frac/neighbors_unique_target but never centrality, and node color/size never encode low_centrality membership -- so the new Inf-FS re-rank signal is invisible in the artifact a reviewer actually looks at; only someone who separately inspects graph.to_meta()['low_centrality'] (a different, non-visual code path) ever sees it. This is short of fully 'computed-and-discarded' (to_meta does carry it) but the primary consumption surface never surfaces it.

### [P1] test_gap -- src/mlframe/feature_selection/filters/_usability_greedy_gpu_resident.py:67

**The regression GPU-resident usability-greedy twin (usability_greedy_gpu_resident) has zero test coverage anywhere in the suite -- only its classification sibling (_usability_greedy_clf_gpu_resident) has a CPU/GPU parity test.**

grep across tests/ for `usability_greedy_gpu_resident` returns only a code-audit baseline JSON entry, never an actual test import. This 386-line module hand-rolls bordered normal equations, per-fold coalesced D2H, and a majority-of-folds gate the module's own docstring admits has 'NO explicit near-tie detector'. A future edit to the bordered-solve math or the fold partition could silently diverge the selected feature set from the CPU path under MLFRAME_FE_GPU_STRICT_RESIDENT=1 with nothing in CI to catch it (unlike test_usability_greedy_clf_resident_parity.py for the classification sibling).

### [P1] test_gap -- src/mlframe/feature_selection/filters/_usability_gpu.py:51

**The entire gated GPU usability-scoring module (gpu_abscorr, gpu_abscorr_batch, gpu_additive_basis_residual, fe_gpu_usability_enabled) has zero test coverage: no test sets MLFRAME_FE_GPU_USABILITY or imports these functions directly.**

The module's own docstring states 'SELECTION-EQUIVALENCE IS THE BAR (non-negotiable)' for these cupy twins, yet nothing in the test suite verifies that bar (no parity test against _abscorr/_usability_aware_selection's CPU path). A regression here would only manifest as an unexplained selection change on a host where a user has explicitly opted into this env var with cupy installed, with zero CI signal beforehand.

### [P1] bug -- src/mlframe/feature_selection/filters/_joblib_safe.py:218

**fit_constant_memmap's cache key (_fit_constant_key) hashes only a bounded SAMPLE of the buffer (first/last 64KB + a coarse stride), so two different-content arrays of the same shape/dtype that happen to agree at the sampled points can collide and silently return the WRONG cached memmap.**

Two fit-constant matrices of identical (shape, dtype) recur in one process (e.g. two different targets' X in a multi-target MRMR fit, or two CV folds' data blocks) and differ only in the unsampled interior between stride points (for a ~400MB float64 matrix, stride is ~800 elements, so most bytes are never hashed). fit_constant_memmap then hands the SECOND caller the FIRST array's read-only memmap with no exception, no log, and no test catching it -- FE workers silently compute features from the wrong dataset. The existing test (test_fit_constant_memmap.py) only checks 'byte-identical content' vs 'globally different content' (a + 1.0), never the actual collision boundary this sampling scheme is vulnerable to.

### [P1] bug -- src/mlframe/feature_selection/filters/_boruta.py:137

**boruta_select's correction="bh" path (Benjamini-Hochberg) is re-run every round on accumulating hit_counts at a FIXED per-round alpha, unlike correction="bonferroni" which explicitly divides alpha by rounds_run -- contradicting the docstring's blanket claim that 'the correction keeps the per-round repeated testing from inflating the false-positive rate'. Zero test coverage exists for correction="bh" anywhere in the suite (only used in a benchmark script).**

With resolve_tentative=True, correction="bh", a truly-irrelevant (null) feature is re-tested every round against the SAME alpha=0.05 FDR threshold using ever-more-powerful cumulative binomial evidence -- an optional-stopping problem the bonferroni branch explicitly guards against (via rounds_run in the divisor) but the bh branch does not. Over many rounds this compounds the chance of a false confirm/reject for a feature with no real signal, which is exactly the false-positive inflation resolve_tentative's correction is documented to prevent.

### [P1] bug -- src/mlframe/feature_selection/filters/composition.py:225

**validate_pair_fe_cv's per-fold try/except around optimise_hermite_pair silently converts ANY exception (not just expected non-convergence) to res=None with zero logging, corrupting the honest OOS-uplift statistic the function exists to report.**

If optimise_hermite_pair raises for a genuine bug (not numerical instability) on one CV fold, that fold's oos_mi/uplift_vs_trivial silently become 0.0 with no exception surfaced anywhere -- not even under a verbose flag (unlike the sibling swallow in compose_pair_fe at line 127, which at least logs when verbose=True). This drags down oos_mean and inflates optimism_ratio silently, undermining the exact leakage/overfitting detection this validator exists to provide, and no test currently forces a fold-level exception to check this path.

### [P2] bug -- src/mlframe/feature_selection/filters/composition.py:127

**compose_pair_fe's per-pair exception handler only logs (logger.debug) when verbose=True; at the default verbose=False every real failure in optimise_hermite_pair for a candidate pair is completely silent.**

A genuine bug in the FE optimiser (not just 'no useful signal') silently drops that pair from consideration on a default-config call, with zero diagnostic trace -- a caller sees fewer engineered columns than expected and has no log line to explain why.

### [P2] test_gap -- src/mlframe/feature_selection/filters/_param_accuracy_warnings.py:102

**_accuracy_caveats_warned_ is a one-shot latch set True on the FIRST call regardless of whether any caveat triggered, so a re-fit after set_params() changes a param from good to bad never re-warns; and the entire module (warn_accuracy_suboptimal_params, ACCURACY_SUBOPTIMAL) has zero test coverage anywhere in the suite.**

est = MRMR().fit(X, y) (default, clean, latch set True); est.set_params(fe_pairwise_modular_enable=False).fit(X, y) on the SAME instance (a common manual/notebook re-fit pattern) silently degrades selection accuracy with no warning, because the guard is keyed on 'has this instance ever fired', not on the current parameter values. No test asserts the warning fires on a bad value, stays silent on defaults, or survives a set_params+refit cycle.

### [P2] design -- src/mlframe/feature_selection/filters/estimators.py:13

**The module docstring documents an MRMR(estimator="ksg" | "miller_madow" | "nsb" | "plugin") API and a miller_madow_mi function that do not exist -- MRMR has no `estimator` kwarg at all, and this module defines no miller_madow_mi (Miller-Madow lives entirely in info_theory/, wired via the unrelated mi_correction kwarg).**

A developer reads this docstring and calls MRMR(estimator="ksg") expecting KSG-based relevance scoring; it fails (no such constructor kwarg) or is silently swallowed by an unrelated **kwargs path. The functions in this module (ksg_mi_with_target, ksg_mi_pair, ksg_mi_with_significance, nsb_mi) are in fact never called from MRMR.fit's production path at all -- confirmed dead from MRMR's perspective, reachable only by calling them directly or from _benchmarks/ scripts.

### [P2] design -- src/mlframe/feature_selection/filters/_usability_pool_resident.py:32

**This module's own docstring (dated iter17, 2026-06-23) still claims the resident pair-combo MI table is 'NOT WIRED INTO build_usability_candidate_pool' and that selection-inequivalence is 'the blocking reason', but the caller (_usability_aware_selection.py, comment dated 2026-06-27) confirms it WAS wired in under the _seleq flag once the ULP-tie issue was fixed via an _mi_key grid-snap.**

A future maintainer reading only this file's header would conclude the kernel is dead code and could delete it, or re-wire it unconditionally without realizing the actual (already-shipped) integration depends on the caller-side _mi_key grid-snap for selection-equivalence -- silently reintroducing the exact ULP-tie selection-flip bug (~6 forms differed on a 125-form pool) that originally blocked this path.

### [P2] design -- src/mlframe/feature_selection/filters/bases.py:113

**The four EXTRA_BASES fit() functions (_fourier_fit, _rbf_fit, _sigmoid_fit, _pade_fit) guard only the degenerate-constant case; _rbf_fit/_sigmoid_fit call np.quantile/np.std directly on the raw column with no NaN/Inf guard of their own (unlike _fourier_fit/_pade_fit which at least handle the near-constant case).**

If a caller outside this cluster (hermite_fe's dispatcher, not audited here) ever routes an unscrubbed column with NaN/Inf into the 'rbf' or 'sigmoid' EXTRA_BASES entry (the polynomial-only prewarp path in hermite_fe/_hermite_prewarp.py explicitly excludes non-polynomial bases, so today's known call sites are safe), np.quantile/np.std would silently propagate NaN into centres/bandwidth/thresholds/slope with no error -- this module has no test of its own pinning that guarantee, relying entirely on a caller contract outside this cluster.

### [P2] gpu_residency -- src/mlframe/feature_selection/filters/_usability_greedy_gpu_resident.py

**GPU residency practices across the reviewed resident-GPU modules in this cluster (_usability_greedy_gpu_resident.py, _usability_greedy_clf_gpu_resident.py, batch_pair_usability_corr_gpu.py, _usability_pool_resident.py) are sound: one bulk H2D at entry, integer-index row gathers instead of boolean-mask re-syncs, coalesced whole-round/whole-table D2H instead of per-candidate .get(), and resident_operand content-hash caching to avoid re-uploading recurring operand columns. No wasteful repeated host<->device round-trips were found beyond the specific issues already reported separately (the pool_resident stale docstring above).**

_(no issues found in this cluster for this angle)_

### [P2] cpu_gpu_parity -- src/mlframe/feature_selection/filters/_usability_njit_pool.py:575

**Every CPU/GPU twin pair actually reviewed in this cluster (score_pair_combos's serial/parallel/cupy kernels, batch_pair_usability_corr's njit/cuda/cuda_warp kernels, the greedy resident twins) is dispatched via kernel_tuning_cache with an explicitly measured and documented numeric-equivalence tolerance (bit-identical or ~1e-15/1e-9 FP-reorder, verified by name in the docstrings and backed by parity tests) rather than a silently different formula or threshold.**

_(no issues found in this cluster for this angle)_

### [P2] design -- src/mlframe/feature_selection/filters/estimators.py

**No implicit-Optional / mismatched-return-type patterns were found across the reviewed cluster files (bases.py, composition.py, estimators.py, group_aware.py, friend_graph.py, the _usability_* modules) -- Optional[...] / "T | None" annotations are used consistently where a default of None appears.**

_(no issues found in this cluster for this angle)_

## Proposals

### (coverage_gap) Add a CPU/GPU parity test for usability_greedy_gpu_resident (regression path)

Mirror tests/feature_selection/gpu/test_usability_greedy_clf_resident_parity.py for the regression twin in _usability_greedy_gpu_resident.py: under MLFRAME_FE_GPU_STRICT + MLFRAME_FE_GPU_STRICT_RESIDENT, assert the resident greedy returns the same selected UsableCandidate indices/order as the CPU usability_greedy on a synthetic regression fixture, and assert it returns None (clean CPU fallback) for a degenerate/singular pool.

### (coverage_gap) Add parity tests for _usability_gpu.py's gated cupy primitives

Add tests that set MLFRAME_FE_GPU_USABILITY=1 (skip if no cupy) and assert gpu_abscorr/gpu_abscorr_batch/gpu_additive_basis_residual match their CPU references (_abscorr, the CPU shortlist loop, _fe_pure_form_retention's residual) to float64 round-off, plus a test that fe_gpu_usability_enabled() is False by default and False when the global GPU off-switch is set.

### (coverage_gap) Add a false-confirm-rate biz_value test for boruta_select's correction="bh" path

Mirror test_biz_val_boruta_select_resolve_tentative_cuts_false_confirms but with correction="bh": run resolve_tentative=True on all-noise features across many rounds and assert the false-confirm rate stays controlled, to establish whether the missing rounds-correction actually inflates false positives in practice before deciding whether to fix the correction formula.

### (coverage_gap) Add tests for warn_accuracy_suboptimal_params

Test: (1) a bad param value triggers exactly one UserWarning listing the caveat; (2) an all-default estimator is silent; (3) a second .fit() call never re-warns (documents current behavior); (4) after this is fixed, a set_params() that changes a param from good to bad before a second .fit() DOES re-warn.

### (edge_case) Quantify the _fit_constant_key collision window before deciding a fix

Add a test that constructs two arrays of identical shape/dtype/first-64KB/last-64KB/strided-sample bytes but different interior content, and asserts fit_constant_memmap does NOT alias them (it currently would). If it fails, either hash the full buffer (measure the cost against the ~200-400MB fit-constant matrices this cache targets) or shrink the stride / add a cheap secondary discriminator (e.g. full-buffer running xor/sum) to close the gap.

### (residency_step) Surface Inf-FS centrality in the rendered friend graph

In friend_graph_to_figurespec, add centrality to node_hovertext and encode low_centrality membership visually (e.g. reduced opacity or a distinct marker border) so the signal build_friend_graph already computes is visible in the artifact a reviewer actually looks at, not only in to_meta().

### (other) Fix or retire the stale estimators.py module docstring

Either wire ksg_mi_with_target/nsb_mi into MRMR via a real `estimator=` dispatch (matching the documented API) or rewrite the docstring to describe the module's actual standalone-utility contract and point to the real mi_correction='miller_madow' mechanism in info_theory/, so a reader is not misled into calling a nonexistent constructor kwarg.
