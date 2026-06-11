# Coverage asymmetry across selector families (KEY=coverage_asymmetry_wrappers) — 2026-06-10

Mission: quantify the test asymmetry between MRMR (97 biz-value layer files) and the other selector families
(RFECV/wrappers, BorutaShap, ShapProxiedFS, HybridSelector, hetero_vote, cluster_aggregate, pre_screen, optbinning,
importance), and produce concrete port plans for the proven MRMR test patterns. Per project rule: extend the lacking
side, never trim the leading one.

## Sampling / method statement (for the verifier)

- File census + per-family `def test_` counts via `ls` globs + `grep -c "def test_"` over `tests/feature_selection/`
  (465 files; `.claude/worktrees/**` excluded everywhere).
- Param-coverage scans: AST-extracted `__init__` / function signatures (RFECV 85 params, ShapProxiedFS 95,
  BorutaShap 25, HybridSelector 20, `heterogeneous_relevance_vote` 12, `run_cluster_aggregate_step` 26), then regex
  `\b<param>\s*=` over (a) ALL test files and (b) the family's own test files only (the in-family scan removes
  false positives from generically-named kwargs).
- Deep-read (full): `hetero_vote.py` (137 LOC), `hybrid_selector.py` (551), `pre_screen.py` (203), `optbinning.py`
  (86), `importance.py` (371), `test_hetero_vote.py`, `test_hybrid_selector.py`, `test_hybrid_selector_production.py`,
  `test_pre_screen.py`, `test_optbinning.py`, `test_importance.py`, RFECV `__init__` signature (`_rfecv.py:166-430`),
  `_rfecv_configs.py` fields. Deep-read (targeted greps): `registry.py`, `test_selectors_shared.py`,
  `test_fs_selector_contract.py`, `test_selector_registry.py`, `test_cluster_aggregate.py`,
  `test_biz_val_cluster_aggregate.py`, `test_biz_val_filters_boruta_shap.py`, `test_concurrency_determinism.py`,
  `golden/capture_baseline.py`, `_biz_val_synth.py`, `LAYER_INDEX.md`, `_cluster_aggregate.py` cluster-discovery body.
- MRMR layer files NOT read individually (47.9k LOC); their contracts were taken from `LAYER_INDEX.md`.
- Checked `audit_disposition.md` (no hits for hetero/hybrid/pre_screen/optbinning/importance/cluster_aggregate) —
  nothing below re-reports a dispositioned item; the three known-blocked items (orth-basis recipe parity, GPU SU gate,
  shap binning) are untouched.
- No pytest executed (read-only mission).

## The asymmetry table (measured 2026-06-10)

| Family | test files | test LOC | `def test_` | source surface (entry points) | tests : source-LOC |
|---|---|---|---|---|---|
| MRMR layer files | 97 | 47,923 | 1,095 | filters/ (~100 modules) | massive |
| MRMR other | 50 | 15,409 | 429 | (same) | massive |
| RFECV / wrappers | 43 | 10,462 | 481 | wrappers/ = 7,052 LOC | good breadth, knob holes (F-03) |
| ShapProxiedFS | 55 | 10,971 | 403 | ~25 `_shap_proxy_*` modules | good, su_seeded knob holes (F-15) |
| BorutaShap | 16 | 1,339 | 48 | boruta_shap.py 35KB + 3 siblings | thin biz-value (F-12) |
| cluster_aggregate | 7 | 992 | 39 | `filters/_cluster_aggregate.py` 516 LOC | decent, 7 knob holes (F-14) |
| pre_screen | 3 | 483 | 34 | pre_screen.py 203 LOC | decent, polars edge holes (F-18) |
| HybridSelector | 3 | 437 | 19 | hybrid_selector.py 551 LOC | WEAK: 13/20 params untested (F-04..07) |
| optbinning | 1 | 203 | 10 | optbinning.py 86 LOC | FS variants never fitted (F-16) |
| importance | 1 | 189 | 10 | importance.py 371 LOC | 2 of 4 public funcs untested (F-17) |
| hetero_vote | 1 | 69 | 3 | hetero_vote.py 137 LOC | WEAKEST: 365x fewer tests than MRMR (F-08..11) |

Weakest five (tests per source LOC + missing pattern classes): **hetero_vote, HybridSelector, optbinning,
importance, BorutaShap-biz-value** (cluster_aggregate and pre_screen are sixth/seventh — knob-level holes only).

MRMR-proven patterns available for porting (from `LAYER_INDEX.md` + shared infra): (a) hard-case synthetic layers
(decoys L6, MNAR L7, outliers L11, imbalance L13, regression-y L15, multiclass L16, leakage L17, degenerate inputs
L18, train/test shift L19); (b) golden selection baselines (`golden/capture_baseline.py`); (c) determinism /
concurrency (`test_concurrency_determinism.py`, n_jobs parity); (d) shared synthetic generators
(`tests/feature_selection/_biz_val_synth.py`: `make_imbalanced`, `make_correlated_redundant`, `make_3way_xor`,
`make_two_latent_groups`, `downstream_auc`, `signal_recovery_count`); (e) observability/report-field pinning
(`test_metadata_feature_selection_report_observability.py` style); (f) shared contract parametrization
(`test_selectors_shared.py` / `test_fs_selector_contract.py`).

---

### [coverage_asymmetry_wrappers-01] Shared selector-contract suites omit BorutaShap and HybridSelector
Severity: P1
Kind: share
Files: tests/feature_selection/test_selectors_shared.py:38-59; tests/feature_selection/test_fs_selector_contract.py:3,32-63; src/mlframe/feature_selection/__init__.py:26 (HybridSelector is a curated public export)
Evidence: `test_selectors_shared.py` parametrizes its entire 599-LOC contract over exactly two factories — `pytest.param(_make_rfecv, id="RFECV"), pytest.param(_make_mrmr, id="MRMR")` (lines 58-59). `test_fs_selector_contract.py` docstring: "Parametrizes 3 selector implementations -- MRMR, RFECV, ShapProxiedFS". BorutaShap (a registered, training-wired selector) and HybridSelector (a top-level package export) run through NEITHER uniform contract (fit/transform idempotence, constant-column handling, refit-state isolation, get_support/get_feature_names_out consistency, pickle).
Proposal: add two factories to `test_fs_selector_contract.py`'s param list: `BorutaShap(model=RandomForestClassifier(n_estimators=25, random_state=0), n_trials=10, train_or_test="train", random_state=0, verbose=False)` and `HybridSelector(use_fe=False, random_state=0)` (use_fe=False keeps it deterministic + fast per its own production test). Absorb representation asymmetries the same way the file already absorbs MRMR-int-index vs RFECV-bool-mask `support_` (its lines 146-148 pattern). Mark both ids with `pytest.mark.timeout` and a small-n fixture (n=400, p=10) so the whole matrix stays under the heavy-test budget; reuse the existing `_data`-style generator.
Effort: M

### [coverage_asymmetry_wrappers-02] Registry asymmetry: HybridSelector unregistered; registry test doesn't assert ShapProxiedFS
Severity: P2
Kind: share
Files: src/mlframe/feature_selection/registry.py:146-152; src/mlframe/feature_selection/__init__.py:26-42; tests/feature_selection/test_selector_registry.py:14-18,46-51
Evidence: registry registers `MRMR`, `RFECV`, `BorutaShap`, `ShapProxiedFS` — `HybridSelector` is exported as the package's headline composition ("hybrid (compute-once-share-many composition...)" in `__all__`) but cannot be reached via `registry.get(...)`, i.e. the documented "adding a fourth selector" registry-dispatch path (`_setup_helpers_pre_pipelines.py:110-113`) cannot ever wire it. `test_builtin_registrations_present` asserts only `"MRMR" in available / "RFECV" / "BorutaShap"` — ShapProxiedFS registration has no presence sensor, and only MRMR has an instantiate-returns-right-type test (`test_mrmr_spec_instantiate_returns_mrmr`).
Proposal: (1) extend `test_builtin_registrations_present` with `assert "ShapProxiedFS" in available`; (2) add `test_rfecv_spec_instantiate`, `test_shap_proxied_spec_instantiate` mirroring the MRMR one (assert `isinstance` against the real classes; for BorutaShap assert the default `cluster_reduce=True` wrap returns `GroupAwareMRMR` — currently only the bare `cluster_reduce=False` branch has an isinstance test in `test_cluster_reduce_kwargs_reachable.py:54`); (3) either register a `HybridSelector` spec (lazy import, `use_fe` passthrough) + presence assert, or pin the deliberate omission with a one-line comment in registry.py — flagging for the fixer to decide; the test should match whichever lands.
Effort: S

### [coverage_asymmetry_wrappers-03] RFECV: 12 of 85 constructor knobs never exercised by any test
Severity: P1
Kind: param
Files: src/mlframe/feature_selection/wrappers/_rfecv.py:184 (best_desired_score), :189 (min_train_size), :216 (estimators_save_path), :221 (report_ndigits), :223 (special_feature_indices), :262 (stability_top_k), :285 (keep_loser_subset_fi), :335 (cpi_max_depth), :337 (cpi_min_samples_leaf), :356 (drop_nan_score_fi), :370 (noimprove_counts_revisit), :382 (prescreen_fdr_level)
Evidence: AST + repo-wide kwarg scan (all tests/, worktrees excluded): these 12 names never appear as `<name>=` in any test file; bare-name grep confirms 0 mentions for 10 of them, and the 2 mentions that exist are not behavioral (`keep_loser_subset_fi` only as a default-value assert at test_wrappers_rfecv_fi_correctness.py:129; `stability_top_k` only inside an unrelated shap test name). The grouped-config tests (`test_wrappers_grouped_configs.py`) exercise override precedence + pydantic validators, not these knobs' behavior. Several are decision-altering: `keep_loser_subset_fi=True` / `drop_nan_score_fi=False` / `noimprove_counts_revisit=True` are the documented legacy A/B sides of Wave-1 correctness fixes — none has the "legacy side reproduces old (biased) behavior" pin the fix-wave convention requires; `special_feature_indices` short-circuits the whole search; `prescreen_fdr_level` directly gates the L7 univariate prescreen.
Proposal: one new file `tests/feature_selection/test_rfecv_unexercised_knobs.py`, small-n Ridge/LogReg fixtures, one behavioral test per knob:
- `best_desired_score`: set it to a trivially-reachable score; assert the outer loop stops early (`len(r.evaluated_scores_...)` / refit count strictly below an unconstrained twin run).
- `min_train_size`: pass a value > fold size; assert the documented behavior (raise or fold-merge) — pin whichever the code does.
- `special_feature_indices`: assert support_ equals exactly that subset and the optimizer was short-circuited (refits == small constant).
- `stability_top_k`: stability_selection=True with top_k=2 vs default (p//4) on `make_correlated_redundant`; assert selected set shrinks monotonically with top_k.
- `keep_loser_subset_fi=True` / `drop_nan_score_fi=False` / `noimprove_counts_revisit=True`: A/B vs default on a fixture with one degenerate fold (inject an estimator that returns NaN score on fold 2 via a wrapper); assert the legacy flag changes voting tables / iteration count in the documented direction.
- `cpi_max_depth` / `cpi_min_samples_leaf`: `importance_getter='conditional_permutation'` on a 6-feature correlated fixture; assert both knobs reach the auxiliary tree (FI vector changes between max_depth=1 and None; behavioral, no getsource).
- `prescreen_fdr_level`: prescreen='univariate_ht' on signal+noise; assert level=1e-6 keeps fewer columns than level=0.5 and the true signal survives both.
- `estimators_save_path` + `keep_estimators=True`: fit into `tmp_path`; assert dump files exist per estimator/fold and `required_features.dump` is written.
- `report_ndigits`: LOW-value cosmetic — cover with a single smoke assert (no crash at ndigits=1), do not over-invest.
Effort: L

### [coverage_asymmetry_wrappers-04] HybridSelector tree-member subsystem has ZERO behavioral tests (13/20 ctor params unexercised in-family)
Severity: P1
Kind: gap-mask
Files: src/mlframe/feature_selection/hybrid_selector.py:79-81 (knobs), :205-249 (_tree_signals), :251-273 (_admit_tree_products), :36-41 (_TREE_OPS replay table), :369-383 (gate-then-prune in fit); tests/feature_selection/test_hybrid_selector.py (no mention of any tree knob)
Evidence: in-family AST scan: `prescreen, corr_thr, use_mrmr, boruta_driver, use_tree_member, tree_top_k, tree_cooccur_pairs, tree_n_estimators, tree_max_depth, tree_prod_gate, tree_rich_ops, mrmr_synergy_cap, name` never appear as kwargs in the 3 hybrid test files. The tree member is **default-ON** (`use_tree_member=True`) and carries measured-win claims in comments ("madelon +0.020 3-seed", "hard_synth +0.030"), yet: the 3 `tree_prod_gate` modes ("synergy"/"relevant_median"/"raw_median", :256-273), the rich-op replay (`_TREE_OPS["absd"/"sign"/"rat"]`), the gate-then-prune invariant ("rejected ones are pruned from the frame AND the replay pairs", :377-383), and `mrmr_synergy_cap` plumbed into `fe_synergy_screen_max_features` (:178-179) have no test anywhere. A regression silently disabling the default-ON member would pass the entire suite.
Proposal: two tiers, mirroring the file's own `_combine`-injection pattern:
(1) PURE-FUNCTION tests for `_admit_tree_products`: build `h = HybridSelector()`, inject `h.fi_ = {...}`, `h._tree_prod_pairs_/_tree_prod_names_/_tree_op_` by hand; `@pytest.mark.parametrize("gate", ["synergy", "relevant_median", "raw_median"])` with FI fixtures designed so each gate admits a DIFFERENT subset (e.g. prod FI 0.05 vs operand FIs 0.02/0.06 → synergy rejects, raw_median admits) — asserts the three rules genuinely differ.
(2) one e2e (n=1500, 4 informative + product target `1.4*z0*z1`, like the existing FE e2e): assert (a) `h._tree_prod_names_` non-empty and every name present in `h.transform(X).columns ∪ rejected-pruned`; (b) with `tree_rich_ops=("mul",)` only `tmul_*` names appear; (c) a y-permuted twin admits ~0 products (gate binds); (d) `use_tree_member=False` produces no `t*_` columns and no "tree" key in `member_selections_` (the dormant-member contract at :436-437); (e) transform on FRESH rows replays op columns bit-equal to applying `_TREE_OPS[op]` manually.
Plus a biz_value test per CLAUDE.md: tree member ON vs OFF on an interaction-only pool, assert downstream AUC delta >= measured-minus-margin.
Effort: M

### [coverage_asymmetry_wrappers-05] HybridSelector degraded-member and vote-fallback paths untested
Severity: P1
Kind: gap-mask
Files: src/mlframe/feature_selection/hybrid_selector.py:201-203 (MRMR stage degrades to `([], None)`), :421-428 (shap/boruta members degrade to `[]` with warning), :485-486 (`if not chosen: chosen = list(cluster_votes.keys())`), :442 (`self.raw_selected_ = ... or cols[:1]`)
Evidence: every member fit is wrapped in `try/except Exception → warnings.warn(... degraded ...)`. No test exercises ANY degradation: grep for "degraded" in tests/ returns nothing. These guards are exactly the silent-error-swallowing class the project tracks (memory: silent correctness bug classes) — today a permanently-broken member (import error, API drift) would degrade every fit to a 2-member vote with only a UserWarning nobody asserts.
Proposal: in `test_hybrid_selector.py` add: (1) `monkeypatch.setattr(HybridSelector, "_run_shap", raising_stub)` → `with pytest.warns(UserWarning, match="shap member degraded")`; assert fit completes, `member_selections_["shap"] == []`, and the boruta+mrmr vote still recovers the informative block; same for `_run_boruta_premerge`; (2) monkeypatch the MRMR import target to raise inside `_run_mrmr` → assert `mrmr_selected_ == []`, `artifacts_ is None`, warning text contains "MRMR stage degraded"; (3) pure `_combine` test where members vote only for features whose `cluster_of_` lookup misses (empty cluster_votes) → assert the `cols[:1]` floor keeps the output non-empty and deterministic; (4) `_combine` test with `vote=3` and disjoint single-member selections → assert the `chosen = list(cluster_votes.keys())` union fallback fires (selection equals union, not empty).
Effort: S

### [coverage_asymmetry_wrappers-06] HybridSelector boruta_driver="permutation", use_mrmr=False, prescreen=False paths untested
Severity: P2
Kind: param
Files: src/mlframe/feature_selection/hybrid_selector.py:343-348 (driver → `train_or_test` switch), :360-362 (use_mrmr=False branch), :397-403 (prescreen branch + `len(relevant) < 2` floor)
Evidence: in-family scan — none of the three appear in any hybrid test. `boruta_driver="permutation"` flips `train_or_test` to "test" and `importance_measure` (documented high-precision option); `use_mrmr=False` must zero out `_mrmr_member/_eng_names` and still produce a working vote; `prescreen=False` must pass ALL columns to members.
Proposal: parametrized e2e on the existing `_linear_dataset` (n=800 to keep wall down): `@pytest.mark.parametrize("kw", [{"boruta_driver": "permutation"}, {"use_mrmr": False}, {"prescreen": False}])`; assertions per case: permutation-driver → fit completes, `member_selections_["boruta"]` non-empty, >=3 of 4 informative recovered; use_mrmr=False → `"mrmr" in member_selections_` maps to the relevant-fallback list (`or list(relevant)` at :418), `n_engineered_ == 0`, no MRMR warning raised; prescreen=False → `h.relevant_ == list of all X_aug columns`. Also pin the `len(relevant) < 2` floor: monkeypatch `_shared_perm_fi` to return all-zeros with use_mrmr=False/use_fe=False → assert `relevant_` falls back to all columns rather than crashing the members.
Effort: S

### [coverage_asymmetry_wrappers-07] HybridSelector FE-mode pickle round-trip and engineered get_support semantics untested
Severity: P2
Kind: gap-mask
Files: src/mlframe/feature_selection/hybrid_selector.py:536-542 (get_support excludes eng_N), :544-551 (__getstate__ keeps `_mrmr_member` "needed to replay feature engineering at transform time"); tests/feature_selection/test_hybrid_selector_production.py:46 (`use_fe=False` only)
Evidence: the only pickle test fits with `use_fe=False`, so the load-bearing claim of `__getstate__` — that the pickled fitted MRMR member can replay engineered columns post-unpickle — is never verified. Likewise `get_support`'s documented exclusion of engineered names ("Engineered (eng_N) selections are not original columns, so they are excluded") is never asserted in a fit that actually engineered something. Memory rule "runtime caches break pickle" makes the FE-carrying pickle the risky one.
Proposal: extend `test_fit_transform_support_and_pickle_roundtrip` (or add a sibling) using the interaction dataset from `test_fe_default_augments_and_transform_replays`: after fit with `use_fe=True`, (a) if any engineered name in `raw_selected_`: assert `mask.sum() == len([c for c in raw_selected_ if c in X.columns])` and every engineered survivor present in `get_feature_names_out()` but absent from the mask-implied columns; (b) `h2 = pickle.loads(pickle.dumps(h))`; assert `h2.transform(X_fresh)` reproduces `h.transform(X_fresh)` column-for-column and value-equal on the eng_N columns (the replay-after-unpickle contract).
Effort: S

### [coverage_asymmetry_wrappers-08] hetero_vote regression path (classification=False) has zero tests
Severity: P1
Kind: gap
Files: src/mlframe/feature_selection/hetero_vote.py:55-58 (regression panel), :72-75 (`scoring="r2"`, `chance=0.0`); tests/feature_selection/test_hetero_vote.py:26,47-50,63-64 (all three tests pass `classification=True`)
Evidence: the regression branch (RandomForestRegressor / Ridge / KNeighborsRegressor panel, KFold + R2 skill) is completely dark — a typo in the regression dict or a classification-only assumption in `_importance` (e.g. `coef_.max(axis=0)` on a 1-D Ridge coef) would never be caught.
Proposal: add `test_hetero_vote_regression_keeps_signal_drops_noise`: `y = z @ [1.5, -1.2, 1.0, 0.9] + 0.3*noise` (continuous), 4 signal + 20 noise, `classification=False, n_shadow_trials=3, random_state=0`; assert all 4 signals accepted, `len(accepted ∩ noise) <= 1`, `info["n_models"] == 3`. Add a second regression case with `weight_by_cv_skill=True` asserting weights are R2-derived (all in [0.05, 1.0]) — covers the `chance=0.0` line.
Effort: S

### [coverage_asymmetry_wrappers-09] hetero_vote: models=, percentile=, per_model_hit_frac=, vote_threshold boundary, ndarray X all unexercised
Severity: P2
Kind: param
Files: src/mlframe/feature_selection/hetero_vote.py:112 (ndarray fallback names `x{i}`), :116 (custom panel), :124 (`np.percentile(imp[P:], percentile)`), :126 (`hits/n >= per_model_hit_frac`), :133-134 (`vote_frac >= vote_threshold` with >=)
Evidence: in-family kwarg scan: `models`, `percentile`, `per_model_hit_frac`, `cv_skill_floor` never passed; X is always a DataFrame; the `>=` boundary of vote_threshold is never pinned (with the default 3-member panel, a feature passed by exactly ceil(3*0.5)=2 members sits AT 0.667 — but vote_threshold=2/3 exactly is the discriminating boundary).
Proposal: (1) `test_hetero_vote_ndarray_input_uses_xN_names`: pass `X.values`; assert accepted ⊆ {f"x{i}"} and `set(info["vote_fraction"]) == {f"x{i}" ...}`. (2) `test_custom_two_model_panel`: `models={"a": RandomForestClassifier(20), "b": LogisticRegression pipeline}`; assert `info["n_models"] == 2` and `model_weights` keys == {"a","b"}. (3) percentile monotonicity: same data/seed, `percentile=50` accepts a superset of `percentile=100` (lower shadow bar → more hits). (4) per_model_hit_frac boundary: `n_shadow_trials=2, per_model_hit_frac=0.5` → a feature hitting in exactly 1/2 trials passes (`>=`); with `per_model_hit_frac=0.6` it fails — construct via a deterministic stub estimator whose `feature_importances_` alternates by trial seed (inject through `models=`, no prod change needed). (5) vote_threshold boundary: with the stub panel produce vote_frac exactly 2/3 and assert accepted at `vote_threshold=2/3`, rejected at `2/3 + 1e-9`.
Effort: M

### [coverage_asymmetry_wrappers-10] hetero_vote weight_by_cv_skill lacks the discriminating "blind member downweighted" test
Severity: P2
Kind: bizvalue
Files: src/mlframe/feature_selection/hetero_vote.py:98-108 (docstring: intent is to downweight a near-chance member; measured no-op only because the BENCH bed had no near-chance member), :127-131; tests/feature_selection/test_hetero_vote.py:57-69 (only asserts "runs and reports weights >= floor")
Evidence: the one skill-weighting test asserts plumbing, not the mechanism. The docstring itself says the option is kept "for datasets that DO contain a near-chance member" — exactly the case with zero coverage.
Proposal: `test_skill_weighting_rescues_feature_vetoed_by_near_chance_member`: panel = {"tree": RandomForestClassifier, "dist": KNN-pipeline, "blind": DummyClassifier(strategy="prior") wrapped to expose uniform `feature_importances_` via a tiny adapter class in the test}. Target: 2-feature nonlinear signal both real members detect; the blind member never hits anything and has CV skill ~0 (gets weight = cv_skill_floor=0.05). Set `vote_threshold=0.7`: equal weighting gives vote_frac=2/3 < 0.7 → signal REJECTED; skill weighting gives (w_tree + w_dist)/(w_tree + w_dist + 0.05) ≈ 0.9 >= 0.7 → ACCEPTED. Assert the flip (rejected under False, accepted under True) — this is the quantitative win assertion the biz_value convention requires, with the floor (`cv_skill_floor`) exercised as a bonus.
Effort: S

### [coverage_asymmetry_wrappers-11] hetero_vote `_importance` permutation-importance fallback and n>1000 subsample untested
Severity: P2
Kind: gap
Files: src/mlframe/feature_selection/hetero_vote.py:36-40 (fallback when estimator has neither `feature_importances_` nor `coef_`; subsample to 1000 rows when n > 1000)
Evidence: the default panel always exposes FI or coef_ (RF native; LogReg/Ridge coef_ through pipeline? NO — the panel members are `make_pipeline(StandardScaler(), LogisticRegression())`, and a sklearn Pipeline exposes NEITHER `feature_importances_` nor `coef_` → the LINEAR and DISTANCE members always go through the permutation fallback at :36-40). So the fallback is in fact the hot path for 2 of 3 default members, yet no test pins its correctness directly, and the `n > 1000` rng-subsample branch is only ever hit incidentally (test data n=1500).
Proposal: (1) direct unit test of `_importance`: a Pipeline(StandardScaler, Ridge) on `y = 3*x0 + noise`, n=300 → assert importance of x0 strictly above every other column (the fallback works); same estimator at n=2500 with `random_state=0` twice → identical vectors (subsample is seeded). (2) A comment-level trap to pin: pass a bare `LogisticRegression()` (exposes coef_) and assert the coef_ branch returns |coef| (`c.max(axis=0)` multi-class collapse covered by a 3-class y). This kills the latent risk that someone "optimizes" the panel to bare estimators and silently changes which branch computes importances.
Effort: S

### [coverage_asymmetry_wrappers-12] BorutaShap biz-value depth = exactly ONE scenario (balanced binary linear); port the MRMR hard-case layers
Severity: P1
Kind: bizvalue
Files: tests/feature_selection/test_biz_val_filters_boruta_shap.py:17 (single `def test_`); contrast: tests/feature_selection/LAYER_INDEX.md layers 6,7,11,13,15,16,18; shared generators tests/feature_selection/_biz_val_synth.py:62,91,131,152
Evidence: the whole biz-value contract for BorutaShap is one test (2 informative + 8 noise, balanced binary, linear logit). The 47 other BorutaShap tests are targeted regression pins (rng isolation, polars input, premerge, shadow fast-path...). Zero quantitative coverage for: regression y (`classification=False` exists as a ctor param), multiclass (a dedicated 3D-shap-axis fix test exists — so the path is LIVE — but no recovery contract), imbalanced y, correlated-redundant decoys (the registry's default medoid premerge makes this the production-shaped case), heavy-tail/outlier resistance.
Proposal: extend `test_biz_val_filters_boruta_shap.py` with 4 parametrized scenarios reusing `_biz_val_synth`: (1) `make_imbalanced(imbalance=0.05, n=5000)` (memory rule: rare 1pct needs n≳5000; 5% at n=2000 is fine too) → assert signal recovery >= 2/3 and noise_kept <= 3; (2) regression: continuous y = 0.8*x0 + 0.4*x1 + noise, `classification=False`, RandomForestRegressor → informative_kept == 2; (3) multiclass 3-class y via thresholding the linear score at terciles → informative kept, no crash on the 3D shap axis; (4) `make_correlated_redundant(n_corr=4)` → at least one member of the redundant cluster accepted AND total accepted <= signal+2 (parsimony under redundancy; also exercises the default medoid premerge when run through `registry.get("BorutaShap").instantiate()`). Floors set per CLAUDE.md (measure once at dev time, assert measured-minus-10-15%). Use n_trials<=20, n_estimators<=60 to keep each under ~10s; module-level FAST env-var knob halving n.
Effort: M

### [coverage_asymmetry_wrappers-13] Determinism + golden-baseline patterns exist only for MRMR (and one RFECV hash test) — hetero_vote/HybridSelector have neither
Severity: P2
Kind: share
Files: tests/feature_selection/test_concurrency_determinism.py (8 tests, 0 mentions of RFECV/Boruta/Shap/Hybrid/hetero); tests/feature_selection/golden/capture_baseline.py:65-68 (MRMR only); tests/feature_selection/test_rfecv_hashseed_determinism.py (RFECV-only seed test)
Evidence: golden selection baselines (`golden/pre_refactor/g_n2k_p50_clf.json` etc.) are captured for MRMR alone; the determinism suite (same-seed repeated fit identical; n_jobs=1 vs 4 identical) covers MRMR only. hetero_vote builds shadows with `default_rng(random_state + tr)` and HybridSelector chains FOUR seeded members — both completely unpinned: a refactor moving an rng draw would silently change production selections with no failing test.
Proposal: (1) `test_hetero_vote_deterministic`: same call twice (seed=0) → identical `accepted` AND bit-equal `vote_fraction` dict; seed=1 → different shadow draws (assert vote_fraction differs on at least one column — guards against an ignored random_state). (2) `test_hybrid_selector_deterministic`: `HybridSelector(use_fe=False, random_state=0)` fit twice on `_linear_dataset` → identical `raw_selected_`, `member_selections_`, `fi_` keys; this doubles as the n_jobs-stability sensor since members use n_jobs=-1 internally. (3) extend `golden/capture_baseline.py` with a `--selector hybrid|boruta` mode capturing `raw_selected_` for the same g_n2k_p50_clf grid, committed under `golden/`, with the same compare-test harness MRMR uses — gives the weak families the refactor-safety net that caught MRMR regressions across ~100 layers.
Effort: M

### [coverage_asymmetry_wrappers-14] cluster_aggregate: 7 cluster-discovery knobs unexercised (min_member_relevance, max_cluster_size, homogeneity_tau, max_candidates, mi_eps, edge_significance, is_polars_input)
Severity: P2
Kind: param
Files: src/mlframe/feature_selection/filters/_cluster_aggregate.py:300-390 (`_discover_clusters` consumes all seven), :387-395 (`run_cluster_aggregate_step` signature defaults); tests: tests/feature_selection/test_cluster_aggregate*.py + test_biz_val_cluster_aggregate.py (in-family kwarg scan: 0 hits for all seven)
Evidence: the family's 39 tests cover weights/sign-alignment/recipes/gating/MRMR-wiring well, but every knob that shapes WHICH clusters get discovered is dark: `max_cluster_size` truncation (:359-364 keeps the top members by relevance), `homogeneity_tau` variance-ratio rejection (:375), `min_member_relevance` pool filter (:326), `max_candidates` pool cap (:330-331), and the `mi_eps`/`edge_significance` edge test (:347). `is_polars_input` parity is likewise untested.
Proposal: extend `test_cluster_aggregate.py` with a fixture of ONE latent factor replicated into 6 members of graded relevance + 4 noise cols (reuse `make_latent_reflections` from `_biz_val_synth`): (a) `max_cluster_size=3` → recipe `src_names` length <= 3 and contains the 2 highest-relevance members (truncation rule); (b) `min_member_relevance` set above the weakest member's MI → that member absent from `src_names`; (c) `homogeneity_tau=0.99` (extreme) → heterogeneous cluster rejected, no aggregate appended; tau=0.0 → appended (monotone gate direction); (d) `max_candidates=2` → at most 1 cluster possible (pool starvation degrades gracefully, no crash); (e) `edge_significance`/`mi_eps` sweep: strict values prune all edges → zero clusters, permissive values recover the latent cluster; (f) polars parity: same frame as pl.DataFrame with `is_polars_input=True` → recipe equal (names, method, weights allclose) to the pandas run.
Effort: M

### [coverage_asymmetry_wrappers-15] ShapProxiedFS: su_seeded_* tuning block (7 knobs) + 5 other ctor params never passed in any test
Severity: P2
Kind: param
Files: src/mlframe/feature_selection/shap_proxied_fs/__init__.py (ctor params: su_seeded_top_k, su_seeded_n_bins, su_seeded_max_screen_cols, su_seeded_snr_z, su_seeded_snr_null_quantile, su_seeded_snr_abs_floor, su_seeded_n_permutations, min_selected_ratio, active_learning_budget, cluster_use_gpu, cluster_su_n_bins, prescreen_top); tests/feature_selection/test_shap_proxy_su_seeded_interactions.py (passes only the master `su_seeded_interactions=` flag)
Evidence: corpus-wide AST scan: 12 of 95 ShapProxiedFS ctor params never appear as kwargs. The su_seeded master switch has a good biz_value pair (recovery + byte-identical no-op), but its 7 sub-knobs — the SNR gate thresholds that decide whether a seeded pair survives — are dark; `min_selected_ratio` and `active_learning_budget` are selection-altering floors/budgets with zero tests. (`cluster_use_gpu` may be implicitly covered by the GPU-suite env dispatch — verify before writing tests; if covered, document and skip.)
Proposal: smallest-sufficient additions in `test_shap_proxy_su_seeded_interactions.py`: (a) `su_seeded_top_k=1` vs default on a 2-interaction-pair fixture → only the stronger pair's operands recovered (top-k binds); (b) `su_seeded_snr_z` set absurdly high (e.g. 50) → report shows screen ran but admitted 0 pairs and selection byte-equals the OFF run (gate monotonicity, reuses the existing byte-identical harness at :150); (c) `su_seeded_n_permutations=0-or-min` smoke + determinism. Separately: `min_selected_ratio` test asserting `len(selected_) >= ratio * n_features_in_` on a noise-heavy frame, and `active_learning_budget` asserting the budget caps refinement model count (read the report_ field the path already publishes).
Effort: M

### [coverage_asymmetry_wrappers-16] optbinning: the FS pipelines are never FITTED — IV selection behavior has zero validation; biz_value assert hides behind a version-skip
Severity: P2
Kind: bizvalue
Files: src/mlframe/feature_selection/optbinning.py:43-59 (fs variants), tests/feature_selection/test_optbinning.py:145-154 (only `bp_nocats_nofs` ever fitted), :183-188 (pytest.skip when signal IV==0), :53-128 (everything else is construction-shape asserts)
Evidence: the module's whole point is IV-based feature SELECTION, yet no test fits `bp_nocats_fs`/`bp_withcats_fs` — `BinningProcess(selection_criteria={"iv": ...})` actually dropping a noise column is never demonstrated. The single quantitative test (signal IV > 2x max noise IV) self-skips when optbinning collapses the signal to one bin ("defensive skip" pattern the project treats as a smell), so on those env combos optbinning has ZERO executed quantitative coverage. `memory=` (sklearn cache) and `n_jobs` propagation untested.
Proposal: (a) `test_fs_pipeline_fit_drops_noise_keeps_signal`: fit `bp_nocats_fs` on the existing `_make_synthetic_binary_df(n=600)` with `iv_kwargs={"min": 0.05, "strategy": "highest"}`; assert `transform(df).shape[1] < df.shape[1]` and the support (via `BP.get_support(names=True)`) contains `signal_step` and excludes at least half the noise cols. (b) De-flake the IV biz_value instead of skipping: pass a step-function signal optbinning cannot collapse — make y literally `(x>0)` PLUS construct the signal column as `np.sign(x)` (2-valued; the binner has exactly one split available and IV>0 is guaranteed arithmetically) — then DELETE the `pytest.skip` branch (per the no-documented-skips rule; the current skip text itself admits the assertion silently vanishes on CI). (c) `test_withcats_fs_fit_with_categorical`: fit `bp_withcats_fs` on `_make_synthetic_with_categorical` (the CatBoostEncoder→BP chain is currently never fitted either); assert transform width and row count. (d) one-line `memory=str(tmp_path)` smoke (pipeline caches without error).
Effort: S

### [coverage_asymmetry_wrappers-17] importance.py: explain_top_feature_importances, show_shap_beeswarm_plot and the filename-sanitizer have zero tests
Severity: P2
Kind: gap
Files: src/mlframe/feature_selection/importance.py:39-41 (_sanitize_for_filename), :48-55 (show_shap_beeswarm_plot), :354-371 (explain_top_feature_importances; writes `join("reports", f"{safe_name}_shap_beeswarm.png")` relative to CWD); tests/feature_selection/test_importance.py:6-7 (docstring NAMES both functions as public surface, then tests neither)
Evidence: grep across tests/: zero references to either function or to `_sanitize_for_filename`. The sanitizer is a path-traversal guard ("strips anything that could turn model_name into a path traversal") — a security-flavored helper with no regression pin; `explain_top_feature_importances` writes into a CWD-relative `reports/` dir (test-environment pollution risk worth pinning deliberately).
Proposal: (a) pure-unit tests for `_sanitize_for_filename`: `("../../etc/passwd", no "/" or ".." prefix)`, Windows-reserved chars `<>:"|?*` stripped, 200-char input truncated to 120, empty/whitespace input → "unnamed", `strip(" .")` kills trailing-dot hidden-file trick — 5 parametrized cases, no deps. (b) `explain_top_feature_importances` smoke: tiny `GradientBoostingClassifier` (TreeExplainer-supported) wrapped in a stub object exposing `.model/.metrics/.columns` per the call contract, `monkeypatch.chdir(tmp_path)`, `save_chart=True` → assert exactly one `reports/*_shap_beeswarm.png` exists with size>1KB and the filename round-trips through the sanitizer (no raw `[`/`@` traversal chars). Agg backend + `pytest.importorskip("shap")` per CI conventions. (c) `show_shap_beeswarm_plot` covered transitively by (b); add `plt.close("all")` teardown.
Effort: S

### [coverage_asymmetry_wrappers-18] pre_screen: polars non-numeric branch, exact-threshold boundary, and the 1-row degenerate frame untested
Severity: P2
Kind: gap
Files: src/mlframe/feature_selection/pre_screen.py:93-110 (polars non-numeric skip + `var_val is None → drop`), :115 (`<= _var_cutoff`), :154 (`null_count > null_cutoff` strictly), :179-184 (pandas var NaN → drop)
Evidence: `test_pre_screen.py` covers the pandas extension-dtype and sparse branches well, but: (a) a polars Utf8/Categorical column never goes through `compute_unsupervised_drops` in any test (the polars dtype-guard at :97-102 is dark); (b) the strict `>` on null fraction is asserted nowhere at the boundary (null_count == 0.99*n exactly must KEEP); (c) a 1-row frame: polars `var()` returns null and pandas `var()` returns NaN → EVERY numeric column is dropped via the "treat as constant" branch — plausibly intended (1 row is degenerate) but currently an accident nobody pinned, and `apply_drops` downstream would empty the frame.
Proposal: (a) `test_polars_string_and_categorical_not_dropped`: pl.DataFrame with Utf8 + Categorical + one constant numeric → drops == [constant numeric] only. (b) boundary: n=100, threshold=0.99, column with exactly 99 nulls → kept (99 > 99 is False); 100 nulls → dropped. Parametrize pandas/polars. (c) `test_single_row_frame_drops_all_numeric_documented`: 1-row pandas + polars frames → assert current behavior (all numeric dropped, strings kept) with a comment marking it the pinned contract — if the fixer instead decides 1-row should return [], the test flips with the fix (either way the behavior stops being accidental).
Effort: S

### [coverage_asymmetry_wrappers-19] Hybrid unit tests import the BENCHMARK module path, not the production one
Severity: LOW
Kind: quality
Files: tests/feature_selection/test_hybrid_selector.py:23,75,97,117,133,159 (`from mlframe.feature_selection._benchmarks.fs_hybrid.hybrid_selector import HybridSelector`); tests/feature_selection/test_hybrid_selector_production.py:25-29 (identity pin `H_pkg is H_prod is H_bench`)
Evidence: 6 import sites in the main unit-test file target `_benchmarks.fs_hybrid....`. Today the identity test guarantees coverage transfers; but if the bench package is ever decoupled (bench dirs are explicitly "committed rejected-idea museums" per project policy and CAN fork), the entire `_combine` unit suite would silently keep testing the bench copy while production drifts.
Proposal: flip the 6 imports to `from mlframe.feature_selection.hybrid_selector import HybridSelector`, keeping `test_public_import_paths_resolve_to_one_class` as the single place referencing the bench path. Pure test refactor, zero behavior change.
Effort: S

### [coverage_asymmetry_wrappers-20] Heavy hybrid/hetero e2e tests have no fast mode
Severity: LOW
Kind: quality
Files: tests/feature_selection/test_hybrid_selector.py:113,127 (`@pytest.mark.timeout(900)`); tests/feature_selection/test_hybrid_selector_production.py:42; tests/feature_selection/test_hetero_vote.py:10 (n=1500 x RF-120 panel x 3-4 shadow trials per test)
Evidence: project convention (memory: "Tests always need Fast mode") requires an env-flag / small-n representative for every heavy test. The three hybrid e2e fits run full MRMR+ShapProxiedFS+BorutaShap stacks (the in-source bench note says ~22s each on the dev box, and 900s timeouts admit much worse on CI); hetero_vote fits 9-12 RandomForest-120 models per test. None reads a fast-mode switch.
Proposal: module-level `_FAST = os.environ.get("MLFRAME_TEST_FAST", "")` in both files; under fast mode use n=400/p=8, `n_shadow_trials=2`, RF `n_estimators=40`, and for hybrid pass `tree_n_estimators=30, fe_max_steps=1`; assertions unchanged (they are structural/plumbing, not threshold-tight). Keeps default behavior identical, gives the 1-representative path the convention requires.
Effort: S

## Summary

| Severity | Count |
|---|---|
| P0 | 0 |
| P1 | 6 (01, 03, 04, 05, 08, 12) |
| P2 | 12 (02, 06, 07, 09, 10, 11, 13, 14, 15, 16, 17, 18) |
| LOW | 2 (19, 20) |
| Total | 20 |

Verdict: the asymmetry is real and steep — MRMR carries ~1,500 tests / 63k LOC while hetero_vote ships on 3 tests,
HybridSelector (a top-level public export with a default-ON tree-FE member) on 19, and optbinning's actual
feature-selection behavior is never fitted at all. RFECV/ShapProxiedFS have good breadth but a measurable knob-level
hole (12 and 12 constructor params respectively never exercised). The highest-leverage ports are: shared-contract +
determinism + golden-baseline enrollment for the weak families (01, 13), the HybridSelector tree-member and
degradation suites (04, 05), and the BorutaShap hard-case biz-value layer pack (12) — all buildable from the
existing `_biz_val_synth` generators without new infrastructure.
