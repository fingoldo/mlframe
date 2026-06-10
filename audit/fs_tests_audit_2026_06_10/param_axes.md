# param_axes — missed parametrization opportunities in FS/FE tests (2026-06-10)

Sampling statement (for verifier): mapped all 468 files in tests/feature_selection/ via grep (test-count vs parametrize-count per file; helper-name dedup across the 97 layer files; axis greps for regression/multiclass/float32/polars/sample_weight/groups per file). Deep-read in full: tests/feature_selection/conftest.py, test_selectors_shared.py, test_biz_value_mrmr_layer35.py. Structurally scanned (test-name + key-line level): layers 12, 15, 16, 20, 40, 49(skim), 53(skim), 65, 66, 67, 70, 71, 72, 73, 74, 83, 101, 104; test_fs_selector_contract.py, test_cat_interactions_coverage.py, test_shap_proxy_revalidate.py, test_bases.py, test_hetero_vote.py, test_hybrid_selector.py, test_gpu_kernel_equivalence.py, test_plugin_mi_classif_dispatch.py, test_biz_val_mi_estimators.py, test_wrappers.py, tests/training/test_feature_selection.py. Source cross-checks: filters/mrmr.py, _mrmr_fit_impl.py, _mi_dispatch.py, _cluster_aggregate.py, _target_encoding_fe.py, registry.py, hetero_vote.py, hybrid_selector.py, boruta_shap.py, shap_proxied_fs/__init__.py, training/core/_setup_helpers_pre_pipelines.py. Already-dispositioned items (orth-basis recipe parity, GPU SU gate, shap binning) were checked against audit_disposition.md and are NOT re-reported.

Positive baseline (so the verifier knows what was checked and found healthy): GPU dispatchers are well parametrized (test_plugin_mi_classif_dispatch.py parametrizes n x k x n_classes; test_gpu_kernel_equivalence.py parametrizes basis x backend incl. forced-env override and cupy-missing fallback); layer12 parametrizes seeds; test_wrappers.py parametrizes cv/aggregation/cost axes and covers CatBoost/XGB/LGBM/sklearn estimators; ShapProxiedFS regression mode (classification=False) is covered in 10 test files; cluster-aggregate methods incl. signed_max_abs/median_z/signed_l2_sum are covered across layers 43-47.

---

### [param_axes-01] Modern FE mechanisms (62 fe_*_enable flags) are tested ONLY under binary classification — regression and multiclass layers pin FE off
Severity: P1
Kind: param
Files:
- tests/feature_selection/test_biz_value_mrmr_layer15.py:182-198 (regression layer; "Only ``fe_max_steps=0`` and ``interactions_max_order=1`` are pinned", `fe_max_steps=0`)
- tests/feature_selection/test_biz_value_mrmr_layer16.py:67,190-198 (multiclass layer; same pin)
- all FE layers sampled (21, 26, 32, 33, 34, 38, 56, 60, 77, 87, 88, 89, 90, 92, 93, 94, 95, 104): grep for `regression|make_regression|multiclass|n_classes=[3-9]` = 0 hits in every one
- src/mlframe/feature_selection/filters/_mrmr_fit_impl.py:3024-3025 ("TE works for both binary classification and regression as-is (mean of {0,1} = P(y=1); mean of continuous = mean)"), :3698 ("CMI gate needs a class-typed target; bin continuous y"), :757 ("they want the RAW continuous y")
- src/mlframe/feature_selection/filters/_target_encoding_fe.py:156-157 ("For binary classification this is treated as {0, 1}; ... For regression any")
Evidence: of 120 test files that set FE flags, only 4 also touch `make_regression` (test_biz_val_filters_mrmr.py, test_mrmr_basic.py, test_mrmr_edge_cases.py, test_mrmr_feature_engineering.py) — and those exercise ONLY the legacy `fe_max_steps` unary/binary synergy FE (test_mrmr_feature_engineering.py greps 0 hits for fe_hybrid/fe_kfold). The 60-odd modern recipe families (fe_hybrid_orth_* x 23 scorers/arities, fe_kfold_te, fe_grouped_*, fe_cat_*, fe_rankgauss, fe_rare_category, fe_temporal_agg, ...) have ZERO regression-target and ZERO multiclass coverage, while prod source explicitly claims/implements continuous-y branches (file:lines above). Those branches (continuous-y binning for the CMI gate, mean-of-continuous TE, gains on a qcut'd y) are real user paths and currently unwitnessed.
Proposal: new file tests/feature_selection/test_fe_mechanisms_task_axis.py. (a) Add `_kitchen_sink_regression(seed, n)` to _biz_val_synth.py: reuse layer35's signal architecture but emit `y = logit + noise` (continuous) instead of Bernoulli; plus `_kitchen_sink_multiclass` via 3-way quantile cut of logit. (b) Parametrize:
```python
MECHS = [
    pytest.param(dict(fe_hybrid_orth_enable=True, fe_hybrid_orth_pair_enable=False, fe_hybrid_orth_basis="hermite"), id="orth-univ"),
    pytest.param(dict(fe_kfold_te_enable=True, fe_kfold_te_cols=("cat_region",)), id="kfold-te"),
    pytest.param(dict(fe_mi_greedy_enable=True, fe_mi_greedy_top_k=8), id="mi-greedy"),
    pytest.param(dict(fe_cat_num_interaction_enable=True, fe_cat_num_interaction_cat_cols=("cat_region",), fe_cat_num_interaction_num_cols=("price",)), id="cat-num-resid"),
    pytest.param(dict(fe_rankgauss_enable=True), id="rankgauss"),
]
@pytest.mark.parametrize("task", ["regression", "multiclass"])
@pytest.mark.parametrize("mech_kwargs", MECHS)
def test_fe_mech_fits_transforms_and_lifts(task, mech_kwargs): ...
```
Assertions: fit completes, transform replays on holdout, engineered columns appear, and downstream Ridge R2 (regression) / multinomial LogReg accuracy (multiclass) lift >= +0.02 over no-FE for mechanisms whose signal is in the fixture (calibrate floors once, set 5-15% below measured per CLAUDE.md). Fast mode: `fast_subset(MECHS, 2)` and n=1500. Expected runtime ~10 cells x 3-8s full, 2 cells fast.
Effort: M

### [param_axes-02] Selector-contract suites cover 2-3 selectors while the registry ships 4 (+HybridSelector); production-shape GroupAwareMRMR-wrapped variants never pass the contract
Severity: P1
Kind: param|integration
Files:
- tests/feature_selection/test_selectors_shared.py:57-60 (`_SELECTOR_FACTORIES = [RFECV, MRMR]`; docstring line 14: "To register a new selector, append to ``_SELECTOR_FACTORIES``")
- tests/feature_selection/test_fs_selector_contract.py:71 (SELECTOR_FACTORIES = MRMR, RFECV, ShapProxiedFS)
- src/mlframe/feature_selection/registry.py:146-151 (registers MRMR, RFECV, BorutaShap, ShapProxiedFS), :75-105 (`_instantiate_rfecv`/`_instantiate_boruta_shap` wrap in GroupAwareMRMR with `cluster_reduce=True` DEFAULT)
- src/mlframe/training/core/_setup_helpers_pre_pipelines.py:112,138 (production suite instantiates via `registry.get`)
- src/mlframe/feature_selection/boruta_shap.py:64 (`class BorutaShap(BaseEstimator, TransformerMixin)` — fully contract-eligible)
- src/mlframe/feature_selection/hybrid_selector.py:73,357,523 (fit/transform/get_feature_names_out — adapter-protocol compatible)
Evidence: BorutaShap appears in NEITHER contract suite despite being a registered, sklearn-API selector with its own polars/multiclass one-off test files. HybridSelector appears in neither. Worse: the production path is `registry.get("RFECV").instantiate()` which returns a GroupAwareMRMR(RFECV) wrapper (cluster_reduce default-ON since 2026-06-03), but both contract suites construct the BARE classes — so the 17-group shared contract (pickle, clone, refit, column-drift, multiclass, regression, dtype, imbalance) has never run against the wrapper users actually get from the suite.
Proposal: in test_selectors_shared.py, derive factories from the registry instead of hand-rolling:
```python
_FAST_KWARGS = {"MRMR": dict(verbose=False, random_seed=0),
                "RFECV": dict(estimator=LogisticRegression(max_iter=200, random_state=0), cv=3, max_refits=2, random_state=0, leakage_corr_threshold=None),
                "BorutaShap": dict(n_trials=10, random_state=0),
                "ShapProxiedFS": dict(n_splits=2, random_state=0)}
_SELECTOR_FACTORIES = [pytest.param(lambda n=name: fs_registry.get(n).instantiate(**_FAST_KWARGS[n]), id=name + "-prod")
                       for name in fs_registry.available()]
```
plus keep two bare-class ids (RFECV-bare via cluster_reduce=False) so wrapper-vs-bare regressions are distinguishable, and append a HybridSelector factory (id="Hybrid", skip pickle group if unsupported via the existing skip hooks). Mark BorutaShap/ShapProxiedFS params with `pytest.mark.slow` so MLFRAME_FAST=1 keeps the matrix at 2 selectors. Expected runtime: +2-4 min full suite, +0 fast.
Effort: M

### [param_axes-03] Input-container axis in the shared contract stops at pandas/numpy — polars input untested at the contract level
Severity: P1
Kind: param
Files:
- tests/feature_selection/test_selectors_shared.py:168-189 (`TestSharedInputTypes`: only `test_pandas_dataframe_input` + `test_numpy_array_input`)
- src/mlframe/feature_selection/filters/mrmr.py:2823-2826 ("iloc preserves dtypes ...; works on pandas + polars (via take) + numpy", `isinstance(X, _pl.DataFrame)`)
- tests/feature_selection/test_boruta_shap_polars_input.py (polars handled per-selector ad hoc)
- CLAUDE.md "Open work items": "Polars support for MRMR — both the selector core and feature engineering — landed 2026-04-22"
Evidence: polars is an officially supported MRMR input (and the project rule forbids silent polars->pandas down-conversion), yet the selector-agnostic contract suite never feeds a pl.DataFrame; polars coverage lives in scattered one-selector files (boruta polars, tests/training/test_mrmr_polars_fe.py). A regression in e.g. RFECV's or ShapProxiedFS's polars path would pass both contract suites.
Proposal: replace the two methods with one parametrized test plus container-specific assertions:
```python
@pytest.mark.parametrize("container", ["numpy", "pandas", "polars"])
def test_container_input_fit_transform(self, selector_factory, small_clf_problem, container):
    X_df, y = small_clf_problem
    X = {"numpy": X_df.values, "pandas": X_df, "polars": pl.from_pandas(X_df)}[container]
    ...
    assert out.shape[0] == X_df.shape[0]
```
and a parity assertion: selected feature-name SET identical across pandas vs polars input for the same data/seed (`sorted(get_feature_names_out())` equality; numpy compared by index count only). `pytest.importorskip("polars")` for the polars id. Runtime impact negligible (3 ids x existing fixture).
Effort: S

### [param_axes-04] `score_pair_mi` — the documented "single entry point to all MI estimators" (9 estimator ids) — has zero direct test coverage
Severity: P1
Kind: param|gap-mask
Files:
- src/mlframe/feature_selection/filters/_mi_dispatch.py:61-108 (`def score_pair_mi(... estimator: str = "plug_in" ...)`; docstring "Single entry point to all MI estimators."; routes plug_in / mixed_ksg / ksg_lnc / mine / infonet / mist / fastmi / median / genie; raises ValueError on unknown)
- verified: `grep -rln "score_pair_mi" tests/` (excluding worktrees) returns NO matches
- tests/feature_selection/test_biz_val_mi_estimators.py:197 (tests "alternative estimators via standalone modules" — i.e. bypasses the dispatcher)
Evidence: every estimator is biz_value-tested through its standalone module import, but the routing function itself — the public seam including `estimator_kwargs` forwarding, the `nbins_strategy` plumbing, the float64 coercion at :85-87, and the unknown-id ValueError at :108 — is never called by any test. A typo in one routing branch (wrong fn in `fn_map`, dropped kwargs) would pass the whole suite.
Proposal: new tests/feature_selection/test_mi_dispatch_contract.py:
```python
ESTIMATORS = [pytest.param("plug_in", id="plug_in"), pytest.param("mixed_ksg", id="mixed_ksg"),
              pytest.param("ksg_lnc", id="ksg_lnc"), pytest.param("fastmi", id="fastmi"),
              pytest.param("median", id="median"), pytest.param("genie", id="genie"),
              pytest.param("mine", id="mine", marks=pytest.mark.slow),
              pytest.param("infonet", id="infonet", marks=pytest.mark.slow),
              pytest.param("mist", id="mist", marks=pytest.mark.slow)]
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_score_pair_mi_signal_exceeds_noise(estimator):
    rng = np.random.default_rng(0); x = rng.normal(size=800); y_sig = (x + 0.3*rng.normal(size=800) > 0).astype(np.int64); y_noise = rng.integers(0, 2, 800)
    mi_s = score_pair_mi(x, y_sig, estimator=estimator); mi_n = score_pair_mi(x, y_noise, estimator=estimator)
    assert np.isfinite(mi_s) and mi_s >= 0.0 and mi_n >= 0.0 and mi_s > mi_n + 0.05
```
plus `test_unknown_estimator_raises` (pytest.raises(ValueError, match="unknown estimator")) and one `estimator_kwargs` forwarding check (`mixed_ksg` with k=3 vs k=15 produce different finite values). Neural ids guarded by `pytest.importorskip("torch")` + slow marker. Runtime: ~2-5s for the 6 fast ids.
Effort: S

### [param_axes-05] Entire 97-layer FE biz suite (including the comprehensive state-of-the-union layers) pins the production-DEFAULT redundancy mechanisms OFF — `dcd_enable=False` appears 62 times, defaults never validated end-to-end
Severity: P1
Kind: param|bizvalue
Files:
- src/mlframe/feature_selection/filters/mrmr.py:937 (`dcd_enable: bool = True` — default-ON since 2026-05-30 per CLAUDE.md), :891 (`cluster_aggregate_enable: bool = True`)
- tests/feature_selection/test_biz_value_mrmr_layer35.py:106-121 (`_make_mrmr` pins `dcd_enable=False, cluster_aggregate_enable=False, build_friend_graph=False`)
- tests/feature_selection/test_biz_value_mrmr_layer101.py:205-206 and test_biz_value_mrmr_layer70.py:159 (the COMPREHENSIVE regression layers pin the same `dcd_enable=False, cluster_aggregate_enable=False`)
- grep: `dcd_enable=False` = 62 occurrences across layer files; `_make_mrmr(**overrides)` duplicated in 48 layer files
Evidence: per-mechanism isolation pins are a defensible test-design choice, but NOT ONE of the headline all-on / kitchen-sink / state-of-the-union contracts runs with the constructor defaults users get from `MRMR()`. CLAUDE.md itself: "The default config is the ONE config that matters for blast radius." The interaction class (DCD pruning or cluster-aggregate swallowing an engineered column that an FE contract depends on — exactly the L91-gate-vs-L35-AUC interaction already documented at layer35:332-335) is structurally invisible to the suite.
Proposal: parametrize the two headline contracts (layer35 `test_logreg_auc_clears_absolute_bar` + `test_all_enabled_lifts_at_least_010_over_no_fe`; layer101's headline equivalent) over a defaults axis:
```python
@pytest.mark.parametrize("default_mode", [pytest.param("isolated", id="isolated"), pytest.param("production", id="prod-defaults", marks=pytest.mark.slow)])
```
where "production" drops the dcd/cluster_aggregate/build_friend_graph pins from `_make_mrmr` (i.e. `MRMR(verbose=0, random_seed=0, fe_ntop_features=25, **_all_fe_kwargs())`). Measure the prod-defaults AUC once and pin its own floor (expected within a few hundredths of the isolated number; if it is materially lower, that is a real prod-default bug to fix, not a reason to skip the axis). Runtime: +2 fits of layer35 scale (~30s budget each) under slow marker only.
Effort: S (test change) — M if the prod-defaults run surfaces a real interaction bug

### [param_axes-06] sample_weight and groups axes absent from FE-mechanism tests: 1/97 layer files touches sample_weight, 0/97 touch groups
Severity: P2
Kind: param|gap-mask
Files:
- src/mlframe/feature_selection/filters/mrmr.py:2897 (`def fit(self, X, y, groups=None, sample_weight=None, ...)`)
- grep over tests/feature_selection/test_biz_value_mrmr_layer*.py: `sample_weight` only in test_biz_value_mrmr_layer53.py (partial_fit layer); `groups=` 0 files
- tests/feature_selection/test_mrmr_sample_weight_unit.py:29-76 (none/uniform/shape-validation/nonuniform-runs — core selection only, no FE flags)
- tests/feature_selection/test_cat_fe_weighted_and_bootstrap.py (cat FE standalone helpers only)
- src/mlframe/feature_selection/filters/_target_encoding_fe.py: grep `sample_weight` = 0 hits (TE ignores weights — silently, if MRMR-level weights are set)
Evidence: a user fitting `MRMR(fe_kfold_te_enable=True).fit(X, y, sample_weight=w)` exercises a path no test witnesses: either the weights flow into per-cell mean(y) (they currently cannot — TE has no sample_weight parameter) or they are silently dropped for the engineered columns while the MI screen IS weighted. Either way the behavior is unpinned.
Proposal: tests/feature_selection/test_fe_mechanisms_weight_axis.py with the canonical weight-vs-duplication equivalence, parametrized over 3 mechanisms:
```python
@pytest.mark.parametrize("mech_kwargs", [orth-univ, kfold-te, mi-greedy ids as in finding 01])
def test_int_weights_match_row_duplication(mech_kwargs):
    # fit A: X with rows duplicated per weight w in {1,2}; fit B: X with sample_weight=w
    # assert selected feature-name set equal AND engineered recipe values on a probe frame allclose (rtol=1e-6)
```
If a mechanism legitimately ignores weights, the test should FAIL first (per repo rules surface it as a prod gap and add weight support or an explicit warn), not be written tolerant. Plus one `groups=` smoke per the same parametrize (fit completes, groups forwarded — assert via the grouped-CV attribute or recipe metadata). Runtime ~6 cells x 2 fits, n=1500.
Effort: M

### [param_axes-07] hetero_vote: `classification=False` branch (regressor panel + R2-based skill) has zero coverage — all 4 tests pin classification=True
Severity: P2
Kind: param
Files:
- tests/feature_selection/test_hetero_vote.py:26,47,50,63 (every `heterogeneous_relevance_vote(...)` call passes `classification=True`; grep `classification=False` = 0 hits; this is the only test file referencing the function)
- src/mlframe/feature_selection/hetero_vote.py:43-49 (`_default_panel(classification)` — regressor panel branch), :62-68 (`R2 - 0` skill branch), :84 (public default `classification: bool = True`)
Evidence: the regression panel construction, the R2-skill computation, and the skill-floor logic under regression are an entirely dead axis in tests; any breakage (e.g. a classifier-only scorer sneaking into `_cv_skill`) ships silently.
Proposal: parametrize the existing three behavioral tests:
```python
@pytest.mark.parametrize("classification", [True, False], ids=["clf", "reg"])
def test_hetero_vote_keeps_signal_drops_noise(classification):
    X, y_clf, sig = make_signal_plus_noise(...); y = y_clf if classification else X[:, :3].sum(axis=1) + 0.3*noise
    accepted, info = heterogeneous_relevance_vote(X, y, classification=classification, n_shadow_trials=4, vote_threshold=0.5, random_state=0)
    assert set(sig) <= set(accepted) and len(accepted) <= len(sig) + 2
```
(`test_skill_weighting_*` pair likewise). Runtime: roughly doubles a small file (~x2 of a few seconds).
Effort: S

### [param_axes-08] Lifecycle-tail copy-paste across layer files: clone/pickle/ctor-default/default-off quartet duplicated in 46 files — should be ONE registry-parametrized suite over the 62 fe_*_enable flags
Severity: P2
Kind: share|param
Files:
- grep: `clone_preserves` in 46 of 97 layer files; `def test_default_ctor_values` in 12 (layers 62, 63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 76); identical shapes `test_default_off_no_<scorer>_columns` / `test_pickle_roundtrip_preserves_<scorer>_recipes` per scorer layer (e.g. test_biz_value_mrmr_layer65.py:366-437, layer66.py:348-385, layer71.py:426-466)
- tests/feature_selection/test_biz_value_mrmr_layer104.py:434-554 (same triple-family pattern within one file: default_off / enabled_adds_columns / pickle_round_trip / clone for rare_category, conditional_residual, rankgauss)
- src/mlframe/feature_selection/filters/mrmr.py: 62 distinct `fe_*_enable` flags (grep -oE "fe_[a-z0-9_]+_enable" | sort -u)
Evidence: ~150-180 near-identical test functions differ only in (flag-kwargs, recipe-name-prefix, params-tuple). Beyond the dedup, the real cost is COVERAGE DRIFT: the quartet exists only for mechanisms whose layer author remembered it; any of the 62 flags without a layer-local copy (and every FUTURE mechanism) silently lacks clone/pickle/default-off guarantees.
Proposal: new tests/feature_selection/test_fe_mechanism_lifecycle.py with a single mechanism table (mechanism id -> enable kwargs -> recipe-column predicate), e.g.:
```python
FE_MECHS = [pytest.param(dict(fe_hybrid_orth_ksg_enable=True), "ksg", id="ksg"),
            pytest.param(dict(fe_hybrid_orth_copula_enable=True), "copula", id="copula"), ...]
@pytest.mark.parametrize("flags,prefix", FE_MECHS)
def test_default_off_emits_no_columns(flags, prefix): ...   # fit MRMR() defaults, assert no prefix-matching engineered col
def test_clone_preserves_params(flags, prefix): ...
def test_pickle_roundtrip_recipe_replay_allclose(flags, prefix): ...
```
Start the table with the 12 ctor-quartet scorers + the layer104 trio + kfold_te/count/frequency (those have the cheapest fixtures); keep the old per-layer copies until the parametrized suite is green, then delete them layer by layer (the layer files keep their mechanism-specific biz_value heads untouched). Fast mode: `fast_subset(FE_MECHS, 3)`. Add a roster-completeness sensor: `assert set(table_flags) >= set(grep'd fe_*_enable flags) - KNOWN_EXEMPT` implemented behaviorally via `MRMR().get_params()` key scan (no inspect.getsource).
Effort: L

### [param_axes-09] OOF-leak contract is ad hoc per layer; no single parametrized permuted-y leak gate over the target-aware FE families
Severity: P2
Kind: param|quality
Files:
- tests/feature_selection/test_biz_value_mrmr_layer104.py:382-417 (`test_rare_category_no_y_leak`, `test_conditional_residual_no_y_leak`, `test_rankgauss_no_y_leak` — hand-rolled per family)
- 42 of 97 layer files mention "leak" (each with its own fixture + threshold style); target-aware families: fe_kfold_te (L33), fe_cat_num_interaction (L34), fe_grouped_agg/quantile (L87/L88), fe_temporal_agg (L92), fe_composite_group_agg (L93), fe_conditional_residual + fe_rare_category + fe_rankgauss (L104), target_encoding in cat FE
Evidence: each target-aware mechanism re-invents the leak assertion with a different recipe (some compare in-fold vs OOF MI, some pin AUC on permuted y, some only check the smoothing math). There is no uniform "fit on permuted y -> downstream holdout AUC of engineered cols ~ 0.5" gate, so the NEXT target-aware family can ship with a weaker hand-rolled check (or none).
Proposal: tests/feature_selection/test_fe_target_aware_leak_contract.py:
```python
TARGET_AWARE = [kfold-te, cat-num-resid, grouped-agg, grouped-quantile, temporal-agg, composite-group-agg, conditional-residual, rare-category, rankgauss]  # flag-dicts as in finding 01
@pytest.mark.parametrize("flags", TARGET_AWARE)
def test_permuted_y_engineered_auc_is_chance(flags):
    X, y = _kitchen_sink(); y_perm = pd.Series(np.random.default_rng(0).permutation(y.values))
    m = _make_mrmr(**flags).fit(X_tr, y_perm_tr)
    eng_cols = [engineered-only columns of m.transform(X_ho)]
    if eng_cols: assert holdout LogReg AUC(eng_cols -> y_perm_ho) <= 0.56
```
0.56 floor absorbs n=900-holdout noise; a true in-fold leak drives this to 0.7+ (the failure mode the per-layer tests each guard). Fast mode subset 3. Runtime ~9 fits x 3-6s.
Effort: M

### [param_axes-10] n<p (wide-data) axis: the only p>n fixture is consumed solely by test_wrappers.py — MRMR core and FE never fitted with p>n
Severity: P2
Kind: param|gap-mask
Files:
- tests/feature_selection/conftest.py:219-238 (`high_dimensional_data`: n=50, p=103)
- grep `high_dimensional_data` consumers: conftest.py + test_wrappers.py ONLY
- tests/feature_selection/test_biz_value_mrmr_layer20.py:411 (embedding layer is p=500 at n=1500 — still n>p)
Evidence: every MRMR layer uses n in 1200-3000 with p<=20ish; the redundancy matrix, DCD clustering, quantile binning (10-quantile bins on 50 rows = 5 rows/bin), and FE candidate explosion all behave differently at p>n, and none of it is witnessed for the filters family. Wide frames (embeddings, omics) are a real user shape.
Proposal: add to test_mrmr_edge_cases.py (or a new test_mrmr_wide_data.py):
```python
@pytest.mark.parametrize("n,p", [(50, 103), (200, 500)], ids=["n50p103", "n200p500"])
@pytest.mark.parametrize("fe", [pytest.param({}, id="no-fe"), pytest.param(dict(fe_hybrid_orth_enable=True, fe_hybrid_orth_pair_enable=False), id="orth-univ")])
def test_mrmr_wide_data_recovers_signal(n, p, fe):
    # 3 informative cols (x0+x1>0 target), rest noise; MRMR(verbose=0, random_seed=0, **fe).fit
    # assert fit completes, n_features_ <= 25, recall of informative >= 1, transform replays on fresh rows
```
Mark (200,500) slow. Expected: either passes (then the axis is pinned) or surfaces a real wide-data crash to fix.
Effort: S

### [param_axes-11] test_mrmr_different_quantization_methods: loop-instead-of-parametrize PLUS except-swallow — the test cannot fail
Severity: P2
Kind: param|gap-mask
Files:
- tests/training/test_feature_selection.py:137-159 (`for method in ["quantile", "uniform"]: ... except Exception as e: warnings.warn(f"Quantization method {method} failed: {e}")`)
Evidence: verbatim body shows every failure — including a hard crash of `quantization_method="uniform"` — is converted to a warning; the only assertion is `hasattr(selector, "n_features_in_")` which is skipped on exception. This is a zero-power test for both methods.
Proposal:
```python
@pytest.mark.parametrize("method", ["quantile", "uniform"])
def test_mrmr_quantization_method(self, sample_regression_data, method):
    ...
    selector = MRMR(verbose=0, max_runtime_mins=0.5, quantization_method=method, quantization_nbins=5, use_simple_mode=True, n_workers=1)
    selector.fit(X, y_subset)            # no try/except — a crash IS the failure
    assert selector.n_features_in_ == X.shape[1]
    assert selector.n_features_ >= 1
```
Same file also holds the task x estimator copy-paste quartet (`test_rfecv_basic_regression`/`_classification`/`_with_catboost_regressor`/`_with_catboost_classifier`, lines 171-249) — fold into `@pytest.mark.parametrize("task,est_factory", ...)` with importorskip on catboost. Runtime unchanged.
Effort: S

### [param_axes-12] layer83 10x7 showdown asserts via for-loops over the matrix — one failing cell hides the rest; parametrize the assert layer over (dataset, mechanism) ids
Severity: P2
Kind: param
Files:
- tests/feature_selection/test_biz_value_mrmr_layer83.py:97-111 (MECHANISMS tuple, LINEAR_DATASETS), :310-344 (matrix built in fixture; `test_full_matrix_populated_no_nans` loops `for ds ... for mech`), :443-465 (`for ds in DATASET_LOADERS` inside asserts)
Evidence: the expensive 70-cell matrix is correctly cached in a fixture, but per-cell contracts are asserted inside Python loops, so (a) the first broken cell aborts the assertion walk and masks sibling failures, (b) there are no per-cell test ids for selective rerun (`-k "breast_cancer-jmim"` impossible).
Proposal: keep the session/module-scoped `matrix` fixture exactly as is (zero extra compute) and convert the per-cell checks to:
```python
@pytest.mark.parametrize("ds", sorted(DATASET_LOADERS), ids=str)
@pytest.mark.parametrize("mech", MECHANISMS, ids=str)
def test_cell_populated_and_finite(matrix, ds, mech):
    cell = matrix[ds]["mech"][mech]
    assert np.isfinite(cell["score"]) ...
```
(70 fast parametrized reads). Leaderboard-level contracts (`best_within_tolerance`, `at_least_5_within_002`) stay loop-style since they are per-dataset aggregates — parametrize those over `ds` only. Runtime impact ~0 (reads of a cached dict).
Effort: S

### [param_axes-13] X-dtype axis (float32 / int) untested for FE recipe fit+replay — only selection-level dtype tests exist
Severity: P2
Kind: param
Files:
- grep `float32` over the 97 layer files: hits only in test_biz_value_mrmr_layer15.py:469 (`test_float32_y_matches_float64_support` — y dtype, FE off) and layer16 (same shape)
- tests/feature_selection/test_selectors_shared.py:560-582 (`TestSharedDtypeVariety`: int32/float32 X but bare default selectors, assertion only `n_features_ >= 1`)
- tests/feature_selection/test_fe_narrow_code_dtype.py (narrow code dtype — internal bin codes, not input X dtype)
Evidence: engineered-recipe REPLAY numerics under float32 inputs (Hermite recurrence, rankgauss erfinv, ratio/log families) are never pinned: fit on float32 frame -> pickle -> transform must stay finite and allclose to the float64 fit within a documented tolerance. Float32 frames are the common memory-saving prod shape (CLAUDE.md 100+ GB frames).
Proposal: extend finding-01's file with a dtype axis on 3 numerically-sensitive mechanisms:
```python
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("mech_kwargs", [orth-univ, rankgauss, mi-greedy])
def test_fe_replay_finite_and_close_across_dtype(dtype, mech_kwargs):
    # fit on X.astype(dtype); assert transform(X_holdout) all finite;
    # assert selected name-set equals the float64 run's; engineered values allclose rtol=1e-4
```
If name-sets legitimately diverge on f32 (selection-altering ~1e-3 MI divergence), that is exactly the class CLAUDE.md flags as NOT acceptable — surface it.
Effort: S

### [param_axes-14] layer35 TestPerMechanismIndividualLift + layer104 family blocks: within-file copy-paste methods differing only in kwargs
Severity: LOW
Kind: param|share
Files:
- tests/feature_selection/test_biz_value_mrmr_layer35.py:403-453 (4 methods `test_orth_univariate_lifts_on_quadratic` / `test_fourier_lifts_on_periodic` / `test_spline_lifts_on_threshold` / `test_cat_num_residual_lifts_on_price_within_region` — identical bodies modulo `mech_kwargs` and the id)
- tests/feature_selection/test_biz_value_mrmr_layer104.py:434-554 (3 families x {default_off, enabled_adds, pickle} as separate methods)
Evidence: bodies are verbatim `self._run_one_mech(fixture, **kwargs)` + the same `>= 0.02` assert; only the kwargs/message differ.
Proposal (layer35):
```python
@pytest.mark.parametrize("mech_id,mech_kwargs", [
    ("orth-univ", dict(fe_hybrid_orth_enable=True, fe_hybrid_orth_pair_enable=False, fe_hybrid_orth_basis="hermite")),
    ("fourier",   dict(..., fe_hybrid_orth_extra_bases=("fourier",), fe_hybrid_orth_fourier_freqs=(1.0, 2.0))),
    ("spline",    dict(..., fe_hybrid_orth_extra_bases=("spline",), fe_hybrid_orth_spline_knots=7)),
    ("cat-num-resid", dict(fe_cat_num_interaction_enable=True, ...))], ids=lambda p: p if isinstance(p, str) else "")
def test_mechanism_lifts_at_least_002(self, fixture, baseline_auc, mech_id, mech_kwargs): ...
```
Same treatment for layer104 with `@pytest.mark.parametrize("family", ["rare_category", "conditional_residual", "rankgauss"])`. Runtime identical (same fits).
Effort: S

### [param_axes-15] test_cat_interactions_coverage: miller_madow true/false/auto trio and unseen-cell strategy duplicates are unparametrized
Severity: LOW
Kind: param
Files:
- tests/feature_selection/test_cat_interactions_coverage.py:989-1012 (`test_use_miller_madow_true` / `_false` / `_auto` — same body modulo the flag value), :882-932 + :1294-1305 (clip/sentinel/raise unseen-cell strategies as 6 separate methods in two classes)
Evidence: the file already parametrizes elsewhere (`test_fwer_correction_variants` :1087, `test_select_on_resorts` :594), so the trio/strategy blocks are inconsistent stragglers.
Proposal: `@pytest.mark.parametrize("mm", [True, False, "auto"], ids=["on","off","auto"])` collapsing :989-1012 into one test asserting fit-completes + engineered-set non-degenerate per mode; `@pytest.mark.parametrize("strategy,expect", [("clip", ...), ("sentinel", ...), ("raise", pytest.raises(...))])` for the unseen-cell pair of blocks. Runtime unchanged.
Effort: S

### [param_axes-16] test_bases.py: per-family roundtrip/coef_size/empty-coefs contract repeated for 4 hand-listed families instead of iterating the live registry
Severity: LOW
Kind: param
Files:
- tests/feature_selection/test_bases.py:43-70 (Fourier: roundtrip/coef_size/canonical_seeds/empty_coefs), :120-149 (RBF same shapes), :188-214 (Sigmoid), :238-269 (Pade), :293 (`test_all_four_families_registered` — a registry exists and is already imported)
Evidence: the generic contract subset (`roundtrip`, `coef_size`, `empty_coefs_returns_zeros`) is structurally identical across the four classes; a 5th family added to the registry would get zero contract coverage unless the author copies a class.
Proposal: keep the family-specific biz tests (`test_biz_fourier_recovers_periodic_signal` etc.) in their classes; lift the generic trio into
```python
@pytest.mark.parametrize("family", sorted(BASIS_REGISTRY), ids=str)
def test_family_roundtrip(family): fam = BASIS_REGISTRY[family]; ...
def test_family_empty_coefs_zero(family): ...
```
so registration implies contract coverage (mirrors the `test_all_four_families_registered` sensor). Runtime unchanged.
Effort: S

### [param_axes-17] cluster-aggregate method loops in test bodies cover hand-picked subsets of the 9-method roster instead of parametrizing over CLUSTER_AGGREGATE_METHODS
Severity: LOW
Kind: param
Files:
- tests/feature_selection/test_cluster_aggregate.py:57 (`for method in ("mean_inv_var", "pca_pc1", "factor_score"):` inside one test)
- tests/feature_selection/test_biz_value_mrmr_layer50.py:317 (`for method in ("mean_z", "mean_inv_var", "pca_pc1", "pca_pc2", ...` inside a perf/bit-equivalence test)
- src/mlframe/feature_selection/filters/_cluster_aggregate.py:34-43 (full roster: mean_z, mean_inv_var, median, pca_pc1, factor_score, pca_pc2, median_z, signed_max_abs, signed_l2_sum)
Evidence: the loops are hand-synchronized subsets of the source constant; the 9 methods ARE individually covered somewhere (layers 43-47), but these two structural tests silently skip whichever methods the author left out, with no per-method failure ids.
Proposal: `from mlframe.feature_selection.filters._cluster_aggregate import CLUSTER_AGGREGATE_METHODS` (it is a module constant) and `@pytest.mark.parametrize("method", sorted(CLUSTER_AGGREGATE_METHODS), ids=str)` in both sites — new methods then auto-enroll. Use `fast_subset(methods, 2)` in layer50's perf test to keep fast mode flat.
Effort: S

---

## Summary

| Severity | Count |
|---|---|
| P0 | 0 |
| P1 | 5 (01, 02, 03, 04, 05) |
| P2 | 8 (06, 07, 08, 09, 10, 11, 12, 13) |
| LOW | 4 (14, 15, 16, 17) |
| Total | 17 |

Verdict: the suite is exceptionally deep on the MECHANISM axis (97 layers, per-scorer biz contracts, well-parametrized GPU dispatch) but systematically narrow on the cross-cutting axes — essentially everything FE-related is binary-classification + pandas + float64 + n>p + non-default DCD config, and several documented prod branches (continuous-y TE, regression hetero-vote panel, the score_pair_mi router, registry-wrapped production selectors) have zero witnesses. The highest-leverage moves are registry-driven parametrization (findings 02, 08, 16, 17) which converts "author remembered to copy the block" into "registration implies coverage", and the task/container/defaults axes (01, 03, 05) which test what users actually run.
