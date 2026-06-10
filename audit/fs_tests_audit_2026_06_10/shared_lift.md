# shared_lift — tests written for ONE selector that should be LIFTED to a shared cross-selector level

Audit date: 2026-06-10. Scope: tests/feature_selection/** + tests/training FS tests + repo-root contract machinery, vs the selector families in src/mlframe/feature_selection/ (MRMR, RFECV, BorutaShap, ShapProxiedFS, GroupAwareMRMR, StabilityMRMR, StabilityFESelector, MRMRTreeRescued, HybridSelector, hetero_vote, pre_screen, optbinning).

Sampling statement (so the verifier can check): read FULLY: tests/feature_selection/test_selectors_shared.py (599 LOC), conftest.py (412 LOC), test_fs_selector_contract.py (325 LOC), src/mlframe/feature_selection/registry.py, test_get_feature_names_out_sklearn.py, test_set_output_pandas_works.py, test_sklearn_is_fitted_and_get_support.py, test_transform_n_features_in_validation.py, test_transform_ndarray_dtype_promotion.py, tests/test_sklearn_compliance.py. Read by-header (grep `def test_` + targeted bodies): test_shap_proxied_fs_contract.py, test_concurrency_determinism.py, test_integration_contract.py, test_boruta_shap_rng_isolation.py, test_rfecv_hashseed_determinism.py, test_selector_registry.py, test_group_aware.py, test_groupaware_wraps_sklearn_selector.py, test_cluster_reduce_kwargs_reachable.py, test_hybrid_selector.py, test_hybrid_selector_production.py, test_stability_input_validation.py, test_mrmr_sample_weight_unit.py, test_wrappers_invariants.py, tests/test_rng_determinism.py, hybrid_selector.py:73-132. Concept cells filled via repo-wide greps (pickle, polars, sample_weight, PYTHONHASHSEED, NotFittedError, set_output, get_support, nan/inf test names, duplicate/collinear test names, column-order/permutation, set_index/RangeIndex). The 97 layer files were NOT read individually; they were grepped for concept names only (they are MRMR-only by construction). Already-dispositioned items in audit_disposition.md (orth-basis recipe parity, GPU SU gate, shap binning) are not re-reported.

Verified landscape facts the findings build on:
- tests/feature_selection/test_selectors_shared.py:57-60 parametrizes ONLY `RFECV` and `MRMR` (18 concept groups A-R).
- tests/feature_selection/test_fs_selector_contract.py:71-75 parametrizes ONLY `mrmr`, `rfecv`, `shap_proxied` (a SECOND, independent shared suite, 2026-05-28).
- src/mlframe/feature_selection/registry.py:146-152 registers 4 selectors: MRMR, RFECV, BorutaShap, ShapProxiedFS; `_instantiate_rfecv` (lines 90-105) and `_instantiate_boruta_shap` (lines 121-132) wrap the bare selector in `GroupAwareMRMR(...)` BY DEFAULT (`cluster_reduce=True`).
- tests/test_sklearn_compliance.py covers mlframe.estimators only (no FS selector); tests/test_rng_determinism.py:48 has a source-scan `test_no_np_random_or_bare_random` over wrapper modules + MBH determinism — reusable machinery for finding 16.

---

### [shared_lift-01] BorutaShap is absent from BOTH shared contract suites
Severity: P1
Kind: share
Files: tests/feature_selection/test_selectors_shared.py:57-60; tests/feature_selection/test_fs_selector_contract.py:71-75; src/mlframe/feature_selection/registry.py:148; tests/feature_selection/test_boruta_shap_transform_unfit_raises_notfitted.py (the only API-contract test BorutaShap has)
Evidence: `_SELECTOR_FACTORIES = [pytest.param(_make_rfecv, id="RFECV"), pytest.param(_make_mrmr, id="MRMR")]` and `SELECTOR_FACTORIES = [("mrmr", ...), ("rfecv", ...), ("shap_proxied", ...)]`. BorutaShap is a registry-registered, suite-wired selector (`register(_SimpleSpec(name="BorutaShap", ...))`) yet gets zero of the ~30 shared invariants (fit-returns-self, n_features_in_, support length, transform shape, constant-column exclusion, refit determinism, clone, pipeline integration, empty-y, y-length mismatch, multiclass, dtype variety, imbalance...). Its only per-selector contract test is the standalone NotFittedError file.
Proposal: add a `_boruta_shap_factory(task)` to test_fs_selector_contract.py mirroring `_shap_proxied_factory` (small config: `BorutaShap(n_trials=10, classification=True, ...)` with a cheap sklearn RF base, `pytest.importorskip("shap")`), append `("boruta_shap", _boruta_shap_factory)` to SELECTOR_FACTORIES, and add an equivalent `pytest.param(_make_boruta, id="BorutaShap", marks=pytest.mark.slow)` to test_selectors_shared.py `_SELECTOR_FACTORIES`. Where BorutaShap legitimately differs (e.g. no `get_feature_names_out`), the existing skip/adapter branches (`_as_bool_mask`, `has_names or has_support`) already absorb it — no new adapters needed. Mark slow tests with the existing `slow` marker so MLFRAME_FAST=1 keeps CI fast (conftest.py:129-139 already implements the skip hook).
Effort: M

### [shared_lift-02] Two parallel shared suites with divergent selector lists, duplicated concepts, and no registry tripwire
Severity: P1
Kind: share | quality
Files: tests/feature_selection/test_selectors_shared.py:37-60; tests/feature_selection/test_fs_selector_contract.py:31-75; tests/feature_selection/test_selector_registry.py:14-76; src/mlframe/feature_selection/registry.py:64-66
Evidence: both files implement fit-returns-self, NotFitted-before-transform, n_features_in_, support/transform-shape consistency, constant-column exclusion, pipeline integration, refit determinism — against DIFFERENT selector subsets and with DIFFERENT factory configs (`MRMR(verbose=False, random_seed=0)` vs `MRMR(min_relevance_gain=0.0, cv=3, run_additional_rfecv_minutes=False, full_npermutations=3, random_seed=0, min_features_fallback=1)`). A new selector must currently be added in two places, and nothing fails if it is added in neither: test_selector_registry.py only checks registration mechanics (`test_builtin_registrations_present`), never that every registered selector is contract-covered.
Proposal: (a) carve a single factory module `tests/feature_selection/_selector_factories.py` exporting `SELECTOR_SPECS: dict[str, SelectorSpec]` where `SelectorSpec(make: Callable[[str], Any], tasks: tuple, deterministic: bool|float, pickle_safe: bool, accepts_ndarray: bool, supports_sample_weight: bool, has_gfno: bool)`; both suites import it (decorator-time parametrize over `SELECTOR_SPECS.items()`). (b) add the tripwire test `def test_every_registered_selector_has_contract_factory(): from mlframe.feature_selection import registry; assert set(registry.available()) <= set(SELECTOR_SPECS)` so a 5th registry entry without contract coverage fails CI. (c) longer-term merge the two files (keep test_selectors_shared.py as the survivor; port the contract-only concepts: clone, set_params roundtrip, NaN-in-y rejection, single-class-y) — per the module-size rule the merged file will exceed 1k LOC, so split as a `test_selectors_shared/` style pair (shared file + factories module) rather than one monolith.
Effort: L

### [shared_lift-03] The production-DEFAULT GroupAwareMRMR(selector) composition is never run through any contract battery
Severity: P1
Kind: integration
Files: src/mlframe/feature_selection/registry.py:90-105 (RFECV default wrap), 121-132 (BorutaShap default wrap); tests/feature_selection/test_cluster_reduce_kwargs_reachable.py:44-55; tests/feature_selection/test_group_aware.py:20-265; tests/feature_selection/test_groupaware_real_rfecv_dropin.py
Evidence: `_instantiate_rfecv`: "cluster-medoid pre-reduction is DEFAULT-ON for the suite's RFECV ... return GroupAwareMRMR(base, ...)" — i.e. what production trains is `GroupAwareMRMR(RFECV)` / `GroupAwareMRMR(BorutaShap)`, but test_selectors_shared.py/_fs_selector_contract.py build BARE `RFECV(...)`/selectors. test_cluster_reduce_kwargs_reachable.py only asserts `isinstance(wrapped, GroupAwareMRMR)`; test_group_aware.py tests medoid mechanics on its own. Nobody asserts that the WRAPPED object satisfies fit-returns-self / n_features_in_ / support alignment / get_feature_names_out / pickle / NotFitted / pipeline-integration — exactly the surface the training suite consumes (e.g. `_inner_support_indices` reading BorutaShap's `accepted`).
Proposal: add two entries to the shared factory registry of finding 02: `"GroupAware(RFECV)": lambda task: registry.get("RFECV").instantiate(estimator=LogisticRegression(max_iter=200, random_state=0), cv=3, max_refits=2, random_state=0)` and `"GroupAware(BorutaShap)"` analogously — instantiating VIA `mlframe.feature_selection.registry` so the test exercises the same default-ON code path production uses. Expect and document asymmetries with explicit xfail markers (e.g. if GroupAwareMRMR lacks `get_feature_names_out`, that itself is a prod gap worth surfacing, not skipping). Dataset: the existing `small_clf_problem` works; to make the medoid path non-trivial add 4 near-duplicate columns (`f0+eps*noise`) so clustering actually reduces (>5% min_reduction).
Effort: M

### [shared_lift-04] Pickle round-trip: untested for BorutaShap / ShapProxiedFS / GroupAwareMRMR / StabilityMRMR, and the one shared pickle test self-disarms via pytest.skip
Severity: P1
Kind: share | quality
Files: tests/feature_selection/test_selectors_shared.py:272-276; tests/feature_selection/test_hybrid_selector_production.py:43-58 (HybridSelector has one — proof the concept generalizes); grep "pickle" over test_boruta*/test_shap_prox*: only tests/feature_selection/test_shap_prox_disk_cache_perm_fit.py (disk-cache, not estimator pickling)
Evidence: shared test: `try: blob = pickle.dumps(selector); restored = pickle.loads(blob) except Exception as exc: pytest.skip(f"selector not pickle-safe: {exc}")` — if MRMR or RFECV REGRESSES to unpicklable (the documented recurring bug class: runtime caches / compiled fns / CUDA handles captured on self), the sensor silently skips instead of failing. BorutaShap and ShapProxiedFS (both shipped to the training suite, whose model-persistence path pickles pre_pipelines) have NO pickle round-trip test at all.
Proposal: (a) replace the skip with a capability flag from the factory registry: `if spec.pickle_safe: blob = pickle.dumps(selector)` with NO try/except (hard fail on regression); selectors genuinely not pickle-safe get `pickle_safe=False` plus a separate test asserting they raise a CLEAR error (so the limitation is pinned, not invisible). (b) parametrize the round-trip over all factories incl. GroupAware-wrapped and StabilityMRMR; assert `restored.transform(X)` equals pre-pickle output (np.testing.assert_array_almost_equal already there). (c) keep n=200/20-col data; runtime is dominated by fit which the test already pays.
Effort: M

### [shared_lift-05] get_feature_names_out(input_features) sklearn protocol (drift raise, wrong-length raise, ndarray-fit user names) tested ONLY for MRMR
Severity: P2
Kind: share
Files: tests/feature_selection/test_get_feature_names_out_sklearn.py:36-99 (all 6 tests construct `MRMR(verbose=0)`); tests/feature_selection/test_fs_selector_contract.py:258-265 (only checks len(names)==n_cols, never the input_features argument)
Evidence: "Pre-fix: the `input_features` argument was accepted in the signature but silently ignored on every code path" — that exact bug class (signature accepted, semantics ignored) is undetectable for RFECV / BorutaShap / ShapProxiedFS / GroupAwareMRMR today: no test passes `input_features` to any of them.
Proposal: lift the 4 protocol assertions into the shared suite parametrized over factories that have `get_feature_names_out` (capability flag `has_gfno`): (1) `gfno(None) == gfno(list(X.columns))`; (2) `gfno(["xx","yy",...])` raises ValueError; (3) wrong length raises; (4) after `fit(X.values, y)`, `gfno(["a","b",...])` returns user names not `feature_N`/`x0`. Selectors lacking the method get a single explicit `pytest.xfail("no get_feature_names_out — sklearn-parity gap")` rather than silent skip, so the asymmetry stays visible.
Effort: S

### [shared_lift-06] set_output(transform="pandas") contract tested ONLY for MRMR
Severity: P2
Kind: share
Files: tests/feature_selection/test_set_output_pandas_works.py:33-104 (4 tests, all MRMR)
Evidence: the file documents the exact failure mode — "bottom-of-module late rebind overwrote the slot that `_SetOutputMixin.__init_subclass__` had wrapped ... set_output silently became a no-op". RFECV, BorutaShap, ShapProxiedFS, GroupAwareMRMR all subclass BaseEstimator/TransformerMixin (boruta_shap.py:64, _rfecv.py:79, shap_proxied_fs/__init__.py:43, group_aware.py:193) so the same mixin contract applies and the same regression (any future `Cls.transform = fn` rebind) would be silent for them.
Proposal: parametrize the 4 assertions (pandas-out for ndarray-in under set_output; DataFrame-in DataFrame-out default; ndarray-in ndarray-out default; output columns == get_feature_names_out) over the shared factory registry. Use each factory's standard small dataset; for selectors whose transform legitimately returns DataFrame always, assert-and-document via a capability flag instead of skipping.
Effort: S

### [shared_lift-07] get_support() mask/indices/unfitted contract + __sklearn_is_fitted__ half-fit rejection tested ONLY for MRMR
Severity: P2
Kind: share
Files: tests/feature_selection/test_sklearn_is_fitted_and_get_support.py:50-104 (MRMR only); grep get_support over test_{wrappers,rfecv,boruta,group,stability,hybrid}*.py -> only test_hybrid_selector_production.py mentions it
Evidence: the MRMR file pins: check_is_fitted rejects an instance with `feature_names_in_` but no `support_` (the half-fit-after-mid-screen-crash state); `get_support()` bool mask of len n_features_in_; `get_support(indices=True)` == support_; unfitted raises NotFittedError. test_fs_selector_contract.py:248-256 only asserts the METHOD EXISTS (`has_names or has_support`). RFECV/BorutaShap/ShapProxiedFS get_support semantics (mask dtype, length, indices parity, unfitted behavior) are untested — and these selectors have exactly the same long fit bodies where a mid-fit crash can leave half-fit state.
Proposal: lift 4 parametrized tests: (a) `np.where(sel.get_support())[0] == sel.get_support(indices=True)`; (b) mask dtype bool + shape (n_features_in_,); (c) unfitted `get_support()` raises NotFittedError; (d) half-fit probe: construct unfitted instance, set only `feature_names_in_`/`n_features_in_` attrs, assert `check_is_fitted` raises (behavioral, no source inspection). Per-selector xfail where get_support doesn't exist (and surface that as a prod parity gap in the xfail reason).
Effort: S

### [shared_lift-08] transform-time n_features_in_ validation (wrong ndarray col-count raises; extra DataFrame cols realigned; missing col raises) tested ONLY for MRMR
Severity: P2
Kind: share
Files: tests/feature_selection/test_transform_n_features_in_validation.py:36-103 (all MRMR); tests/feature_selection/test_selectors_shared.py:309-324 (Group H covers only the dropped-SELECTED-column case, for 2 selectors)
Evidence: the documented production scenario — "an ETL step prepended an ID column before predict-time; transform happily indexed the ID column as if it were feature 0" — applies verbatim to every selector with positional support indexing. Today fit-on-4-cols + transform-on-5-col ndarray is asserted to raise only for MRMR.
Proposal: parametrize over the shared factory registry: (1) `sel.transform(np.random.standard_normal((n, k±2)))` raises ValueError (match="features"); (2) DataFrame with all fit-time cols + 1 extra: transform succeeds, output cols == n_features_ (name realignment); (3) DataFrame missing a selected col raises; (4) correct-shape ndarray works. Reuse `@pytest.mark.parametrize("wrong_cols", [k-1, k+1, k+3])`. Selectors that hard-require DataFrame input get case (1) replaced by an explicit TypeError assertion (capability flag `accepts_ndarray`).
Effort: S

### [shared_lift-09] Regression task is dead weight in the 3-selector contract suite: regression_df fixture defined, never used
Severity: P2
Kind: gap-mask
Files: tests/feature_selection/test_fs_selector_contract.py:90-94 (fixture), 31-68 (factories accept task="regression" with Ridge / RandomForestRegressor branches), grep "regression_df" -> definition only
Evidence: `def regression_df(): X, y = make_regression(...)` has zero consumers; every test passes `factory("binary")`. So the carefully-built `task` plumbing (`_rfecv_factory`: `Ridge()` branch; `_shap_proxied_factory`: `RandomForestRegressor` branch) is never executed — the suite LOOKS regression-aware but tests nothing. test_selectors_shared.py Group N covers continuous-y for RFECV+MRMR only, with a skip-on-ValueError escape.
Proposal: add `@pytest.mark.parametrize("task", ["binary", "regression"])` to TestUniversalContract (fixture selection via `request.getfixturevalue(f"{'binary' if task=='binary' else 'regression'}_df")`), or minimally a dedicated `TestRegressionTask` class that fits each factory("regression") on regression_df and asserts n_features_in_/support/transform-shape invariants. Expected behavior: MRMR regression-mode and RFECV+Ridge must pass; ShapProxiedFS with classification=False must pass (it already has a regression biz_val at test_biz_val_shap_proxied_fs.py:266 — so a contract failure would be a real bug, not an expected limitation).
Effort: S

### [shared_lift-10] Contradictory determinism contracts: exact-equality in one shared suite vs Jaccard>=0.6 in the other, for the SAME seeded selectors
Severity: P2
Kind: quality
Files: tests/feature_selection/test_selectors_shared.py:213-227 (`n1 == n2` exact); tests/feature_selection/test_fs_selector_contract.py:192-205 (`assert jacc >= 0.6`)
Evidence: contract suite comment says "Same-RNG MRMR / RFECV should match exactly" yet the assertion floor is 0.6 Jaccard FOR ALL THREE selectors — a seeded-MRMR determinism regression that drops re-fit agreement to 70% passes this test while violating its own comment (it would only be caught if the OTHER suite's test runs, i.e. coverage depends on file-level redundancy, not design). 0.6 was presumably chosen for ShapProxiedFS's bootstrapped CV, then applied to everyone.
Proposal: move the threshold into the factory spec (`deterministic=True` -> exact set equality; `deterministic=0.9` -> Jaccard floor). For ShapProxiedFS, measure the actual same-seed refit Jaccard once (it should be 1.0 with `random_state=0, n_models=1` — bootstraps are seeded) and pin 5-15% below measured per the biz_value threshold convention; if measured 1.0, it is deterministic and gets exact equality too. Delete the duplicate weaker test after unification (finding 02).
Effort: S

### [shared_lift-11] sample_weight semantics not lifted: unit contracts only for MRMR+RFECV; doubling==row-duplication metamorphic only for RFECV; nothing for BorutaShap/ShapProxiedFS
Severity: P2
Kind: share | param
Files: tests/feature_selection/test_mrmr_sample_weight_unit.py:29-76; tests/feature_selection/test_rfecv_sample_weight_unit.py; tests/feature_selection/test_wrappers_sklearn_coverage_gaps.py:369 (`test_sample_weight_doubling_equivalent_to_row_duplication`); tests/feature_selection/test_sample_weights_fs.py:26-70 (suite plumbing with _DummySelector, selector-agnostic but mock-level)
Evidence: the 4 MRMR unit contracts (None==omitted; uniform==unweighted; shape/negative validation; nonuniform smoke) and RFECV's metamorphic test are textbook selector-agnostic. The training suite forwards sample_weight to ANY selector carrying the `_mlframe_use_sample_weights_in_fs_` marker (test_sample_weights_fs.py docstring), so a BorutaShap/ShapProxiedFS that ignores or crashes on sample_weight is reachable from prod config yet untested.
Proposal: shared parametrized class with capability flag `supports_sample_weight`: (1) `fit(X, y, sample_weight=None)` selection == `fit(X, y)`; (2) `sample_weight=np.ones(n)` == unweighted; (3) wrong-length / negative weights raise ValueError; (4) metamorphic: `fit(X, y, w=2x on subset S)` == `fit(X+rows-of-S-duplicated, y+dup)` selection (exact for deterministic selectors). Selectors whose fit signature lacks sample_weight: assert a TypeError is raised (clear rejection) and mark the capability False — that makes the suite-level marker stamping for them a visible question rather than a silent no-op.
Effort: M

### [shared_lift-12] Zero tests anywhere for non-default / misaligned pandas index (X.index != RangeIndex, y Series with its own index)
Severity: P2
Kind: gap-mask
Files: grep `set_index|RangeIndex\(|non.?default.?index|index.{0,15}misalign` over tests/feature_selection -> No files found; tests/feature_selection/test_selectors_shared.py:386-391 (y-as-Series test uses default aligned index only)
Evidence: every selector test constructs fresh `pd.DataFrame(...)` with default RangeIndex. The classic production shape — X and y sliced by a CV splitter or `df.sample(frac=1)` so both carry a shuffled non-contiguous index — is untested for ALL selectors. Any internal `pd.Series(y_pred, index=...)` arithmetic, `X[col] - y` alignment, or reset_index assumption would silently mis-align and SELECT DIFFERENT FEATURES with no error.
Discriminating recipe: `X, y = small_clf_problem; perm = rng.permutation(len(X)); Xs = X.iloc[perm]; ys = pd.Series(y[perm], index=Xs.index)`. Then: (a) `sel.fit(Xs, ys)` must select the SAME feature-name set as `sel.fit(X, y)` (row order is irrelevant to MI/CV-based selection up to seed handling — use n=500, class_sep=2.0 so selection is stable); (b) the killer case: `ys2 = ys.sort_index()` (values now misaligned with Xs rows IF the selector aligns by index, aligned IF positional) — assert the selector either treats y positionally (documented) or raises; selection silently differing from BOTH (a) and the aligned baseline exposes an alignment bug.
Proposal: add `TestSharedIndexAlignment` to the shared suite parametrized over all factories, with the two cases above; pin expected behavior per selector (all current selectors are positional: convert y via np.asarray — so (a) equal-selection and (b) equal-to-shuffled, i.e. positional semantics, is the expected contract).
Effort: M

### [shared_lift-13] Column-order permutation invariance untested for every selector
Severity: P2
Kind: share | gap-mask
Files: tests/feature_selection/test_shap_proxied_fs_contract.py:57 (`test_transform_preserves_input_column_order` — output ORDER only, ShapProxiedFS only); tests/feature_selection/test_rfecv_hashseed_determinism.py:115 (`test_post_fix_preserves_canonical_column_order` — RFECV internal canonicalisation); grep `permut.*column|column.*order` -> no fit-on-reversed-columns test anywhere
Evidence: no test asserts that `fit(X[reversed_cols], y)` selects the same feature NAMES as `fit(X, y)`. Selectors with positional tie-breaking (argmax over scores, first-wins dedup of duplicates, cluster medoid pick) can flip selections under column reordering — a real reproducibility hazard since upstream polars/pandas pipelines do not guarantee column order.
Proposal: shared test: `cols_rev = list(X.columns)[::-1]; s1 = make().fit(X, y); s2 = make().fit(X[cols_rev], y); assert set(names(s1)) == set(names(s2))`. Use a dataset WITH a near-tie to make it discriminating: 8 informative + 2 exact-duplicate columns (`dup_a = f0`) so positional dedup order matters; the assertion then is set-equality on the union {f0, dup_a}-quotient (treat the duplicate pair as one equivalence class: `{n if n != "dup_a" else "f0" for n in names}`); a selector keeping f0 in one order and dup_a in the other is acceptable, but the rest of the selection must match exactly. Document any legitimate per-selector deviation with xfail + reason.
Effort: M

### [shared_lift-14] NaN / Inf-in-X policy is pinned only for MRMR; BorutaShap has a logging test only; ShapProxiedFS has nothing at selector level
Severity: P2
Kind: share
Files: tests/feature_selection/test_edge_cases_robustness.py:137 (`test_mrmr_nan_in_X_native_tolerance`), :161 (`test_mrmr_inf_in_X_raises`); tests/feature_selection/test_boruta_shap_logger_warn_not_print.py:20 (only checks check_missing_values WARNS); grep np.nan over test_shap_prox*.py -> only treeshap-internal kernels (preflight/treeshap files), no ShapProxiedFS.fit-with-NaN test; tests/feature_selection/test_mrmr_nan_strategy.py + test_propagate_nan_strategy.py (MRMR-only knobs)
Evidence: each selector has SOME NaN policy (MRMR: native tolerance; BorutaShap: warn + tree models tolerate; RFECV: model-dependent; ShapProxiedFS: unknown/untested) but only MRMR's is pinned by tests. An accidental policy flip (e.g. a new validate_data call rejecting NaN, or a silent imputation) would pass everything for the other selectors.
Proposal: shared parametrized policy test with a per-selector expectation table in the factory spec (`nan_policy: "tolerates" | "raises" | "warns"`, `inf_policy: ...`): fit on X with 5% NaN in 2 informative cols, assert per policy (tolerates -> fit succeeds AND still recovers >=1 informative; raises -> pytest.raises(ValueError)); same for `np.inf`. Building the table requires one measurement run per selector — pin what is MEASURED today, so any future change is a deliberate table edit (per the enable-bugfix-by-default rule, if a selector turns out to CRASH opaquely on NaN, that is a prod finding to fix, not to encode).
Effort: M

### [shared_lift-15] Duplicate column NAMES rejection tested only for MRMR
Severity: LOW
Kind: share
Files: tests/feature_selection/test_mrmr_edges_coverage.py:102 (`test_validate_inputs_duplicate_column_names_raises`)
Evidence: `pd.concat([X, X["f0"]], axis=1)` style frames (duplicate names) break name-based support extraction in every selector, but only MRMR pins a clean ValueError. RFECV/BorutaShap/ShapProxiedFS behavior on duplicate names is undefined-by-test (silent positional pick is the dangerous default).
Proposal: shared 5-liner parametrized over factories: build X with two columns literally named "f0", assert `pytest.raises(ValueError)` on fit. If a selector currently accepts it silently, that is a small prod fix (add a `X.columns.duplicated().any()` guard at fit entry) shipped with the test per the regression-test rule.
Effort: S

### [shared_lift-16] Global RNG hygiene: ctor-level tests exist for BorutaShap + ShapProxiedFS only; no shared FIT-level np.random-state test for any selector
Severity: LOW
Kind: share
Files: tests/feature_selection/test_boruta_shap_rng_isolation.py:10-36 (ctor only); tests/feature_selection/test_shap_proxied_fs_contract.py:87 (`test_construction_does_not_mutate_global_numpy_rng`, ctor only); tests/feature_selection/test_screen_rng_hygiene.py (screen module); tests/test_rng_determinism.py:48 (source-scan over wrappers modules only)
Evidence: the audit history shows this bug class recurs at FIT time (FS-P0-3 `_target_prefix` global np.random; FS-P1-5 screen np.random.seed — both RESOLVED per audit_disposition.md, but their regression sensors are per-module). No test asserts "selector.fit() leaves np.random global state untouched" uniformly.
Proposal: shared test: `state_before = np.random.get_state(); sel.fit(X, y); state_after = np.random.get_state(); assert all components equal (np.testing.assert_array_equal on the MT19937 keys)`. Cheap (no extra fit), parametrized over all factories. Also assert a draw parity: `np.random.seed(123); a = np.random.rand(4)` before vs after a fresh-seeded fit in a subprocess-free way. Note: selectors that legitimately consume global RNG when `random_state=None` still must NOT do so when an explicit seed is passed — the factories all pass seeds, so the strict assertion is correct.
Effort: S

### [shared_lift-17] Parallel/n_jobs parity tested only for MRMR (n_workers 1 vs 4) and RFECV (parallel vs sequential)
Severity: LOW
Kind: share | param
Files: tests/feature_selection/test_concurrency_determinism.py:55 (`test_mrmr_n_workers_1_vs_4_identical_with_seed`); tests/feature_selection/test_wrappers_invariants.py:228 (`test_parallel_matches_sequential_single_thread_estimator`); tests/feature_selection/test_shap_proxy_inner_n_jobs_cap.py (caps, not parity)
Evidence: BorutaShap and ShapProxiedFS both expose parallelism knobs (ShapProxiedFS `n_jobs` is set in the contract factory at test_fs_selector_contract.py:67), yet no test asserts n_jobs=1 vs n_jobs=2 selection identity for them. Thread-count-dependent selection is a silent reproducibility bug class already observed in this repo (loky vs threading test exists for MRMR only, test_concurrency_determinism.py:287).
Proposal: lift to shared with capability flag `parallel_param: str|None` ("n_workers" / "n_jobs" / None): `s1 = make(**{flag:1}).fit(X,y); s2 = make(**{flag:2}).fit(X,y); assert set(names(s1)) == set(names(s2))`. Mark `slow` (two fits) and gate via fast_subset to one selector under MLFRAME_FAST=1 (conftest.py:101-105 helper already exists).
Effort: S

### [shared_lift-18] PYTHONHASHSEED (subprocess) determinism tested only for RFECV
Severity: LOW
Kind: share
Files: tests/feature_selection/test_rfecv_hashseed_determinism.py:48-115
Evidence: the file proves selection used to depend on Python hash randomization (set/dict iteration order) for RFECV and pins the fix with two subprocess runs under different PYTHONHASHSEED. MRMR, BorutaShap, ShapProxiedFS, GroupAwareMRMR all build dicts/sets over feature names internally; none has an equivalent sensor, so the same bug class is invisible for them.
Proposal: generalize the existing subprocess harness (it already serializes a fit + selection print): parametrize the runner over selector names, running `{sys.executable} -c "<fit script>"` with `env={"PYTHONHASHSEED": s}` for s in {0, 42}; assert identical printed selections. Mark `slow` + `no_xdist` (subprocess + numba warmup); under MLFRAME_FAST run only MRMR as representative via fast_subset. ASCII-only prints in the child script (cp1251 rule).
Effort: M

### [shared_lift-19] Polars-input coverage is one-off per selector with wildly different rigor; no shared polars==pandas selection-parity contract
Severity: LOW
Kind: share | param
Files: tests/feature_selection/test_boruta_shap_polars_input.py (works-at-all); tests/feature_selection/test_shap_proxied_fs_contract.py:68 (`test_polars_input_supported`); tests/feature_selection/test_regression_w2b_rfecv_arrow_bridge.py (RFECV arrow bridge regression); tests/feature_selection/test_integration_contract.py:285 (`test_mrmr_polars_input_equals_pandas_input` — the ONLY parity-grade test)
Evidence: only MRMR asserts polars selection EQUALS pandas selection; the others assert "did not crash". A polars path that silently reorders/coerces columns (the silent-coercion bug class from memory) would pass the boruta/shap/rfecv tests.
Proposal: shared test (importorskip polars): `Xpl = pl.from_pandas(X); assert set(names(make().fit(Xpl, y))) == set(names(make().fit(X, y)))` parametrized over all factories with capability flag `accepts_polars`; selectors without polars support assert a clean TypeError (and the flag documents the gap). This subsumes the shape-only checks; keep the arrow-bridge regression file (it pins a different internal path).
Effort: S

### [shared_lift-20] clone()/get_params/set_params round-trip absent for BorutaShap, GroupAwareMRMR, StabilityMRMR
Severity: LOW
Kind: share
Files: tests/feature_selection/test_fs_selector_contract.py:207-216 + 239-246 (3 selectors only); tests/feature_selection/test_integration_contract.py:136 (MRMR clone); tests/feature_selection/test_biz_value_mrmr_layer54.py:448 (MRMR clone provenance)
Evidence: GroupAwareMRMR is a wrapping meta-estimator (`GroupAwareMRMR(base, corr_threshold=...)`) — exactly the estimator shape where sklearn `clone()` classically breaks (params not round-tripping through `get_params(deep=False)`, base estimator shared not cloned). It is the production default wrapper (finding 03) and has zero clone tests; the training suite clones custom_pre_pipelines via sklearn.base.clone (audit_disposition.md FS-P1-6), so a clone break is production-reachable.
Proposal: extend the contract clone test over the full factory registry (incl. wrapped variants). For GroupAwareMRMR specifically add: `c = clone(wrapped); assert c.base is not wrapped.base` (deep-clone of inner estimator) and `c.get_params()["corr_threshold"] == wrapped.get_params()["corr_threshold"]`. If GroupAwareMRMR lacks proper get_params for its ctor args, that is a prod fix to ship with the test.
Effort: S

### [shared_lift-21] StabilityMRMR, StabilityFESelector, MRMRTreeRescued, hetero_vote have zero shared-contract coverage
Severity: LOW
Kind: share
Files: src/mlframe/feature_selection/filters/stability.py:25 (StabilityMRMR(BaseEstimator, TransformerMixin)); src/mlframe/feature_selection/filters/_stability_fe.py:278 (StabilityFESelector(BaseEstimator, TransformerMixin)); src/mlframe/feature_selection/filters/_mrmr_tree_rescue.py:33 (MRMRTreeRescued(MRMR)); src/mlframe/feature_selection/hetero_vote.py:83; tests/feature_selection/test_stability_input_validation.py:47-132 (ctor param validation only); tests/feature_selection/test_stability_transform_validation.py; tests/feature_selection/test_mrmr_tree_rescue.py; tests/feature_selection/test_hetero_vote.py
Evidence: three of these are sklearn-API transformers (BaseEstimator+TransformerMixin) yet none appears in either shared suite; their test files cover algorithm specifics and param validation, not the API battery (NotFitted, n_features_in_, support/transform consistency, pickle, clone). hetero_vote is a function, not an estimator — only determinism/NaN concepts apply there.
Proposal: add StabilityMRMR + MRMRTreeRescued to the factory registry (they wrap/subclass MRMR; cheap configs: `StabilityMRMR(n_bootstraps=3, sample_fraction=0.7, random_state=0)`); StabilityFESelector likewise if its fit cost permits (mark slow). For `heterogeneous_relevance_vote` add two function-level tests in its own file (same random_state -> identical votes; NaN-X policy pinned) — not part of the estimator battery.
Effort: M

### [shared_lift-22] Shared-suite escape hatches convert contract failures into skips (ndarray input, pipeline integration, gfno-unfitted)
Severity: LOW
Kind: quality
Files: tests/feature_selection/test_selectors_shared.py:181-187 (`except (TypeError, ValueError, AttributeError): pytest.skip("selector does not accept np.ndarray input")`), 294-303 (`except Exception as exc: pytest.skip("selector not sklearn-Pipeline-compatible")`), 547-554 (unfitted gfno: skip on AssertionError), 272-276 (pickle skip — finding 04)
Evidence: for the TWO selectors actually registered (RFECV, MRMR) these capabilities are KNOWN-supported today (Group B numpy test passes; Group G pipeline test passes). The try/skip wrapper means a regression in either capability flips the test to SKIP, not FAIL — the suite literally cannot catch the regressions it was written for. This is the "documented skip hides real bugs" pattern.
Proposal: replace dynamic skips with declarative capability flags in the factory spec (finding 02): selectors flagged `accepts_ndarray=True` run the assertion with NO try/except; flagged False get `pytest.xfail(reason="requires DataFrame — name-based support extraction")` at parametrize time via marks. Same for pipeline-compat and pickle. Net effect: today's green stays green, tomorrow's regression goes red.
Effort: S

### [shared_lift-23] dtype-promotion "e2e" test imports the prod function but never calls it — tests a local re-implementation instead
Severity: LOW
Kind: quality
Files: tests/feature_selection/test_transform_ndarray_dtype_promotion.py:104-139 (imports `_append_engineered` at line 113-115, then hand-rolls `np.hstack([... astype(common_dtype) ...])` at 134-138); :32-45 (`_stack_pre_fix`/`_stack_post_fix` local replicas)
Evidence: `from mlframe.feature_selection.filters._mrmr_validate_transform import (_append_engineered,)` — the imported symbol is unused; every assertion runs against the test's own copy of the fix. If prod `_append_engineered` regresses to the pre-fix coercion, this file stays green. (The P0 it guards: int base + float engineered -> engineered column silently zeroed.)
Proposal: make the e2e test call the REAL path: fit a small MRMR configured to emit one engineered recipe (the existing `multiplicative_synergy_data` conftest fixture reliably triggers FE), then `out_df = sel.transform(X)` vs `out_nd = sel.transform(X.to_numpy().astype(np.int64-compatible cast via a binned copy))` — or minimally call `_append_engineered(fake_self, base_int, recipes, X)` directly with a stub recipe and assert the engineered column is non-constant float. Keep the replica tests as documentation of the bug pattern but add the real-path call as the load-bearing assertion. ALSO: once lifted, parametrize the broader concept "transform output dtype preserves engineered/float columns" over selectors with mixed-dtype output (capability-gated; only MRMR has engineered tails today).
Effort: S

### [shared_lift-24] Degenerate-y rejection asymmetry between the two shared suites: constant-y asserted for MRMR in one, MRMR deliberately excluded in the other
Severity: LOW
Kind: share | quality
Files: tests/feature_selection/test_selectors_shared.py:195-206 (Group C: constant y -> ValueError, parametrized over RFECV+MRMR); tests/feature_selection/test_fs_selector_contract.py:315-325 (TestClassificationRobustness single-class y: parametrizes ONLY rfecv + shap_proxied, silently omitting mrmr with no comment)
Evidence: the omission is undocumented — a reader cannot tell whether MRMR is excluded because it legitimately accepts constant y (contradicting the other suite, which asserts it raises) or by oversight. BorutaShap is in neither.
Proposal: unify under one parametrized test with the per-selector expectation in the factory spec (`constant_y: "raises"` for all four — test_selectors_shared already proves MRMR raises). Add BorutaShap. One dataset recipe: `y = np.zeros(n, dtype=int)`; assert `pytest.raises(ValueError)`. Delete the duplicate after merge (finding 02).
Effort: S

---

## Summary

| Severity | Count |
|---|---|
| P0 | 0 |
| P1 | 4 |
| P2 | 10 |
| LOW | 10 |
| **Total** | **24** |

By kind: share 18, quality 5 (overlapping), gap-mask 3, param 3, integration 1.

Verdict: the repo HAS the right idea — two shared contract suites exist — but they grew independently, cover only 2-3 of the 4 registered selectors (BorutaShap: zero shared coverage; the production-default GroupAwareMRMR composition: zero), and the deepest sklearn-protocol contracts (get_feature_names_out input_features, set_output, get_support, transform-shape validation, pickle, sample_weight semantics) live as MRMR-only "Wave 9.1" regression files that were never generalized. The single highest-leverage change is the unified capability-flagged factory registry (finding 02) with a registry tripwire: it converts findings 01, 03-08, 11, 14-24 into mostly-mechanical parametrize extensions and prevents the next selector from repeating the drift. Two pure coverage holes stand out beyond lifting: misaligned-pandas-index (12) and column-order permutation invariance (13) are untested for every selector in the codebase.
