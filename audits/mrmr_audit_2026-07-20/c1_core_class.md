# Core MRMR class & fit orchestration

9 findings, 4 proposals.

## Findings

### [P1] bug -- src/mlframe/feature_selection/filters/mrmr/_mrmr_class.py:3556

**MI-correction thread-locals (SU/JMIM/BUR/Miller-Madow/Chao-Shen) are activated before the try/finally that restores them, so a ValueError raised in that window leaks corrupted state into subsequent fits on the same thread.**

MRMR(mi_normalization='su', redundancy_aggregator='typo').fit(X, y) sets the SU thread-local True at line 3556, then raises ValueError at the redundancy_aggregator allow-list check (line 3564) -- before the protective try/finally starting at line 3714 ever runs. The SU-normalization thread-local stays True forever (same thread). The next, completely unrelated call MRMR().fit(X2, y2) (default mi_normalization='none') silently scores relevance/redundancy MI as Symmetric Uncertainty instead of raw MI, changing its selection with no error or warning. A second, independent trigger: MRMR(mi_normalization='su', group_aware_mi=True).fit(X, y, groups=wrong_length_groups) raises the groups-length ValueError at line 3618, by which point su/jmim/bur/miller-madow/chao-shen have ALL already been set (lines 3556-3606), leaking all five. This is a realistic single-threaded/notebook sequence: fix a ctor typo and refit.

### [P1] bug -- src/mlframe/feature_selection/filters/_mrmr_stability_report.py:124

**selection_stability_report()'s bootstrap RNG seed falls back only to the deprecated `random_seed` alias, never to the canonical `random_state`, so it silently uses seed=0 for any estimator seeded the recommended way.**

m = MRMR(random_state=42, ...).fit(X, y); m.selection_stability_report(). During fit(), _fit_body lazily writes the resolved seed onto self.random_seed for the fit's duration only, then restores self.random_seed to its pre-fit value (None, since the user never touched the deprecated alias) in the finally block (mrmr/_mrmr_class.py:3894-3896). By the time the post-fit accessor runs, `getattr(self, "random_seed", 0) or 0` evaluates to 0, not 42 -- the user's chosen determinism seed is silently discarded and the bootstrap always reseeds at 0, contradicting the docstring's own claim ('falls back to the estimator's random_seed'). The one existing test (tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_selection_stability_report.py) only exercises the deprecated random_seed= kwarg, masking this.

### [P1] test_gap -- tests/feature_selection/mrmr/core/test_mrmr_sklearn_contract_and_threadlocals.py

**No test exercises the MI thread-local restore contract on a fit() that raises partway through the toggle-activation section (before the protective try/finally starts) -- only the successful-fit and toggles-start-off paths are covered.**

test_d3_fit_restores_mi_thread_locals_to_pre_fit_snapshot only calls m.fit(X, y) successfully; it never constructs a fit that raises after `set_su_normalization`/`set_jmim_aggregator`/etc. have already run (e.g. mi_normalization='su' + an invalid redundancy_aggregator, or mi_normalization='su' + group_aware_mi=True + mismatched groups length), so the P1 thread-local leak above has no regression coverage.

### [P2] bug -- src/mlframe/feature_selection/filters/_mrmr_tree_rescue.py:125

**MRMRTreeRescued's post-fit LightGBM rescue resolves the RNG seed with priority random_seed-then-random_state, the reverse of the canonical `_effective_random_seed()` order used by the main fit.**

MRMRTreeRescued(random_state=42, random_seed=99).fit(X, y): the ctor warns and the main MRMR fit uses random_state=42 as authoritative (via _effective_random_seed()); but _apply_tree_rescue's `seed = getattr(self,"random_seed",None) or getattr(self,"random_state",None) or 0` picks random_seed=99 instead, so the rescue's LightGBM model is seeded differently from the fit it is rescuing -- a silent divergence from the estimator's own stated seed-precedence contract. Currently masked in the common case (only one of the two aliases set) by Python's `or` short-circuit, so it only manifests when both are explicitly set to conflicting values.

### [P2] bug -- src/mlframe/feature_selection/filters/_mrmr_partial_fit.py:270

**Dead/always-true conditional `and is_first is False` in partial_fit's sample_weight-length-mismatch branch: `is_first` is always False at that point since the branch is only reached after the early-return `is_first` block.**

No functional bug (the guard is vacuously true), but it is misleading dead code that could hide a real intent if the function is ever refactored -- a future edit that moves this check earlier (before the is_first early-return) would silently reactivate an unintended no-op branch.

### [P2] test_gap -- tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_selection_stability_report.py

**selection_stability_report()'s reproducibility is only tested via the deprecated random_seed= ctor kwarg (line 39, line 157), never via the canonical random_state= kwarg that the class's own docstrings recommend.**

A test using MRMR(random_state=N).selection_stability_report() with no explicit random_state= argument to the report call would have caught the P1 finding above; none exists.

### [P2] test_gap -- src/mlframe/feature_selection/filters/_mrmr_fit_helpers_mixin.py

**MRMR.export_artifacts() has no direct unit test in the mrmr test tree -- only indirect exercise via the shap_proxied pipeline tests, and neither of its two documented raise paths (retain_artifacts=False; retain_artifacts=True but self._artifacts_ empty, e.g. after an identity-shortcut or FIT_CACHE-hit fit) is covered directly.**

A regression that silently returns an empty/partial artifact dict instead of raising (e.g. a future fit path that bypasses artifact capture) would not be caught by any test currently grep-visible under tests/feature_selection/mrmr/.

### [P2] design -- src/mlframe/feature_selection/filters/_mrmr_degenerate.py:160

**GPU residency check for this cluster: clean. No issues found -- the single GPU dispatch in the cluster (_gram_matrix's cupy Gram-matrix path) does exactly one host->device upload, one on-device GEMM, and one cp.asnumpy() download, correctly gated behind fe_gpu_strict_enabled's work-floor so small frames stay on CPU.**

_(no issues found in this cluster for this angle)_

### [P2] cpu_gpu_parity -- src/mlframe/feature_selection/filters/mrmr/_mrmr_class.py

**No CPU/GPU kernel-twin pairs live in this cluster's core-class/orchestration files (they belong to the kernel-level modules -- info_theory, permutation, gpu.py -- owned by a different audit cluster), so there is nothing to check for divergence here.**

_(no issues found in this cluster for this angle)_

## Proposals

### (coverage_gap) Regression test: fit() raising mid-toggle-activation must still restore MI thread-locals

Add a test that sets mi_normalization='su' plus a second, independently-invalid config (e.g. redundancy_aggregator='typo', or group_aware_mi=True with a wrong-length groups array) so fit() raises AFTER set_su_normalization()/set_jmim_aggregator()/etc. have run but BEFORE the try/finally at mrmr/_mrmr_class.py:3714 starts. Assert use_su_normalization()/use_jmim_aggregator()/get_bur_lambda()/use_mi_miller_madow()/use_mi_chao_shen() are all back to their pre-fit values after the pytest.raises(ValueError) block. This both documents and (once fixed) locks in the fix for the P1 thread-local-leak finding -- the fix itself should move the toggle-activation block inside the existing try, or wrap it in its own try/finally.

### (coverage_gap) Regression test: selection_stability_report() reproducibility via canonical random_state

Fit two MRMR instances with MRMR(random_state=7, ...) (no random_seed=), call selection_stability_report(as_text=False) on each, and assert the feature_selection_frequency dicts are identical -- then repeat with random_state=8 and assert they differ (proving the seed is actually threaded through). This test would fail today (both would silently use seed=0) and should pass once _mrmr_stability_report.py:124 is changed to `self._effective_random_seed()`.

### (coverage_gap) Direct unit tests for export_artifacts()'s two raise paths

Add tests/feature_selection/mrmr/core/test_mrmr_export_artifacts.py covering: (1) MRMR(retain_artifacts=False).fit(...).export_artifacts() raises ValueError naming retain_artifacts=True; (2) a fitted MRMR whose fit hit the cross-target identity-shortcut or a _FIT_CACHE replay (so self._artifacts_ was never populated even with retain_artifacts=True) raises the documented ValueError pointing at _FIT_CACHE.clear().

### (other) Unify the random_seed/random_state fallback pattern

Three different call sites in this cluster resolve the effective seed three different ways: _effective_random_seed() (canonical: random_state then random_seed), _mrmr_stability_report.py:124 (random_seed only, no random_state fallback -- the bug), and _mrmr_tree_rescue.py:125 (random_seed then random_state -- reversed priority). All non-fit-scoped callers (anything running after fit() returns, or in a sibling class's own method) should call self._effective_random_seed() directly instead of reading self.random_seed, since self.random_seed only carries the resolved value for the duration of fit()'s lazy reconciliation window.
